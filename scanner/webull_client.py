"""
webull_client.py
----------------
Single point of contact for all Webull Open API calls.

This wrapper uses the official webull-openapi-python-sdk (pip package) so
that authentication, request signing, and response handling are handled by
Webull's own code — no hand-rolled crypto that can drift from the spec.

Endpoint paths (confirmed from installed SDK source in
webull/data/request/*.py and webull/trade/request/v2/*.py):

    Stock snapshot  : GET  /openapi/market-data/stock/snapshot   (x-version: v2)
    Stock bars      : GET  /openapi/market-data/stock/bars        (x-version: v2)
    Option snapshot : GET  /openapi/market-data/option/snapshot   (x-version: v2)
    Option contracts: GET  /openapi/instrument/option/contracts   (x-version: v2)
    Paper order     : POST /openapi/trade/order/place             (x-version: v2)

Auth (from SDK source webull/core/auth/composer/default_signature_composer.py):
    Algorithm : HMAC-SHA256 (base64-encoded, NOT hex)
    Headers   : x-app-key, x-timestamp (ISO-8601 UTC), x-signature-algorithm,
                x-signature-version, x-signature-nonce, x-signature, x-version
    Sign string: URL-encoded concat of sorted(all_headers + all_query_params)
                 + "&" + SHA-256-hex(body_json) — assembled by the SDK

There is NO /openapi/market-data/option/chain endpoint. The options chain is
built by:
  1. Fetching all option contracts for the ticker + expiry from the
     instruments endpoint (/openapi/instrument/option/contracts).
  2. Fetching bid/ask/OI snapshots for those symbols in batches of 20
     from /openapi/market-data/option/snapshot.

Credentials (GitHub Secrets, never hardcoded):
    WEBULL_APP_KEY            – App Key
    WEBULL_APP_SECRET         – App Secret
    WEBULL_PAPER_ACCOUNT_ID   – DEN92WP9
    DISCORD_WEBHOOK_ALERTS    – #alerts webhook URL
    DISCORD_WEBHOOK_LOG       – #trade-log webhook URL

Install: pip install webull-openapi-python-sdk
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from webull.core.client import ApiClient
from webull.data.data_client import DataClient
from webull.data.common.category import Category
from webull.data.common.timespan import Timespan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TICKERS = {"TSLA", "NVDA", "SPY"}

# category string used for US equities
US_STOCK = Category.USStock.value      # "US_STOCK"
US_OPTION = "US_OPTION"

# Timespan value for 1-minute bars (from webull/data/common/timespan.py)
TIMESPAN_1MIN = Timespan.m1.value      # "m1"

# Max option symbols per snapshot request (API limit = 20)
OPTION_SNAPSHOT_BATCH = 20

# ---------------------------------------------------------------------------
# WebullClient
# ---------------------------------------------------------------------------


class WebullClient:
    """
    Wrapper around the official Webull Python SDK.

    The SDK handles all authentication internally: it builds the signed
    headers (x-app-key, x-timestamp, x-signature-algorithm HMAC-SHA256,
    x-signature-version, x-signature-nonce, x-signature, x-version) and
    retries on transient failures.
    """

    def __init__(self) -> None:
        self.app_key    = _require_env("WEBULL_APP_KEY")
        self.app_secret = _require_env("WEBULL_APP_SECRET")
        self.paper_account_id = _require_env("WEBULL_PAPER_ACCOUNT_ID")

        use_sandbox = os.environ.get("WEBULL_USE_SANDBOX", "").lower() in ("1", "true", "yes")
        endpoint = "api.sandbox.webull.com" if use_sandbox else "api.webull.com"

        self._api_client = ApiClient(self.app_key, self.app_secret, "us")
        self._api_client.add_endpoint("us", endpoint)

        self._data = DataClient(self._api_client)

        logger.info(
            "WebullClient initialised — endpoint=%s sandbox=%s", endpoint, use_sandbox
        )

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_ticker_snapshot(self, ticker: str) -> dict:
        """
        GET /openapi/market-data/stock/snapshot  (x-version: v2)

        Returns a normalised dict:
            {
                "symbol":       str,
                "last":         float,   # latest trade price
                "volume":       int,     # intraday cumulative volume
                "avgVolume20D": float,   # 20-day average daily volume
                "vwap":         float,   # today's VWAP
                "open":  float, "high": float, "low": float,
                "bid":   float, "ask":  float,
            }
        """
        _validate_ticker(ticker)
        resp = self._data.market_data.get_snapshot(
            symbols=ticker,
            category=US_STOCK,
        )
        _raise_if_error(resp, "get_snapshot")
        raw = resp.json()
        return _parse_snapshot(raw, ticker)

    def get_options_chain(self, ticker: str, expiry_date: str) -> list[dict]:
        """
        Build the 0DTE options chain for *ticker* expiring on *expiry_date*.

        Step 1 — fetch option contract metadata from:
            GET /openapi/instrument/option/contracts  (x-version: v2)

        Step 2 — fetch real-time bid/ask/OI snapshots in batches of 20 from:
            GET /openapi/market-data/option/snapshot  (x-version: v2)

        Returns a flat list of contract dicts:
            [
                {
                    "strike":        float,
                    "type":          "CALL" | "PUT",
                    "open_interest": int,
                    "volume":        int,
                    "bid":           float,
                    "ask":           float,
                    "mid":           float,
                    "iv":            float,
                    "delta":         float,
                    "symbol":        str,   # OCC option symbol
                },
                ...
            ]
        """
        _validate_ticker(ticker)

        # ── Step 1: instrument contracts ──────────────────────────────────
        contracts_resp = self._data.instrument.get_option_contracts(
            category=US_OPTION,
            underlying_symbols=ticker,
            start_date=expiry_date,
            end_date=expiry_date,
            status="ACTIVE",
        )
        _raise_if_error(contracts_resp, "get_option_contracts")
        contracts_raw = contracts_resp.json()
        contracts_meta = _parse_option_contracts(contracts_raw)

        if not contracts_meta:
            logger.warning("%s: no option contracts found for expiry %s", ticker, expiry_date)
            return []

        logger.info("%s: %d contracts for expiry %s", ticker, len(contracts_meta), expiry_date)

        # ── Step 2: snapshot quotes in batches of 20 ─────────────────────
        symbols = [c["symbol"] for c in contracts_meta if c.get("symbol")]
        snapshot_map: dict[str, dict] = {}

        for i in range(0, len(symbols), OPTION_SNAPSHOT_BATCH):
            batch = symbols[i : i + OPTION_SNAPSHOT_BATCH]
            snap_resp = self._data.option_market_data.get_option_snapshot(
                symbols=",".join(batch),
                category=US_OPTION,
            )
            if snap_resp.status_code == 200:
                batch_data = _parse_option_snapshots(snap_resp.json())
                snapshot_map.update(batch_data)
            else:
                logger.warning(
                    "Option snapshot batch %d-%d returned %d",
                    i, i + len(batch), snap_resp.status_code,
                )

        # ── Merge metadata + quotes ───────────────────────────────────────
        chain: list[dict] = []
        for meta in contracts_meta:
            sym = meta.get("symbol", "")
            quote = snapshot_map.get(sym, {})
            bid = quote.get("bid", 0.0)
            ask = quote.get("ask", 0.0)
            mid = (bid + ask) / 2 if (bid and ask) else 0.0
            chain.append({
                "strike":        meta.get("strike", 0.0),
                "type":          meta.get("type", "CALL"),
                "open_interest": quote.get("open_interest", 0),
                "volume":        quote.get("volume", 0),
                "bid":           bid,
                "ask":           ask,
                "mid":           mid,
                "iv":            quote.get("iv", 0.0),
                "delta":         meta.get("delta", 0.0),
                "symbol":        sym,
            })

        logger.info("%s: built chain with %d contracts", ticker, len(chain))
        return chain

    def get_minute_candles(self, ticker: str, count: int = 10) -> list[dict]:
        """
        GET /openapi/market-data/stock/bars  (x-version: v2, timespan=m1)

        Returns the most recent *count* one-minute OHLCV candles, sorted
        oldest → newest:
            [
                {
                    "timestamp": int,    # Unix ms
                    "open": float, "high": float, "low": float,
                    "close": float, "volume": int,
                },
                ...
            ]
        """
        _validate_ticker(ticker)
        resp = self._data.market_data.get_history_bar(
            symbol=ticker,
            category=US_STOCK,
            timespan=TIMESPAN_1MIN,
            count=str(count),
        )
        _raise_if_error(resp, "get_history_bar")
        raw = resp.json()
        return _parse_candles(raw)

    def place_paper_order(
        self,
        ticker: str,
        option_symbol: str,
        side: str,
        quantity: int = 1,
        order_type: str = "MKT",
    ) -> dict:
        """
        POST /openapi/trade/order/place  (x-version: v2)

        Places a simulated order in the Webull paper account for logging.
        The bot manages position exits independently of this order record.

        Returns:
            {
                "orderId":      str,
                "status":       str,
                "filled_price": float | None,
            }
        """
        _validate_ticker(ticker)
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL', got '{side}'")

        # Use the SDK's trade client for order placement
        # The paper account is identified by account_id
        from webull.trade.trade_client import TradeClient
        trade_client = TradeClient(self._api_client)

        resp = trade_client.account_v2.place_order(
            account_id=self.paper_account_id,
            action=side,
            order_type=order_type,
            tif="DAY",
            extended_hours_trading=False,
            qty=quantity,
            category="US_OPTION",
            option_symbol=option_symbol,
        )

        if resp.status_code == 200:
            return _parse_order_response(resp.json())
        else:
            logger.warning(
                "Paper order returned %d: %s", resp.status_code, resp.text[:200]
            )
            return {"orderId": "", "status": "FAILED", "filled_price": None}

    def get_option_quote(self, option_symbol: str) -> dict:
        """
        GET /openapi/market-data/option/snapshot  (x-version: v2)

        Polls the bid/ask/mid of a single open option contract every 60 s.

        Returns:
            {
                "symbol":        str,
                "bid":           float,
                "ask":           float,
                "mid":           float,
                "volume":        int,
                "open_interest": int,
            }
        """
        resp = self._data.option_market_data.get_option_snapshot(
            symbols=option_symbol,
            category=US_OPTION,
        )
        _raise_if_error(resp, "get_option_snapshot")
        snap_map = _parse_option_snapshots(resp.json())
        quote = snap_map.get(option_symbol, {})
        bid = quote.get("bid", 0.0)
        ask = quote.get("ask", 0.0)
        return {
            "symbol":        option_symbol,
            "bid":           bid,
            "ask":           ask,
            "mid":           (bid + ask) / 2,
            "volume":        quote.get("volume", 0),
            "open_interest": quote.get("open_interest", 0),
        }


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_snapshot(raw: Any, ticker: str) -> dict:
    """
    Normalise /openapi/market-data/stock/snapshot response.
    SDK returns: { "data": [ { ...fields... } ] }
    """
    data = raw.get("data") or raw
    if isinstance(data, list):
        data = data[0] if data else {}

    def f(*names, default=None):
        for n in names:
            v = data.get(n)
            if v is not None:
                return v
        return default

    last = _to_float(f("close", "latest_price", "last_price"))
    vwap = _to_float(f("vwap", "VWAP")) or last

    return {
        "symbol":       ticker,
        "last":         last,
        "volume":       _to_int(f("volume", "vol")),
        "avgVolume20D": _to_float(f("avg_volume_20d", "avgVolume20D", "avg_vol_20d")),
        "vwap":         vwap,
        "open":         _to_float(f("open")),
        "high":         _to_float(f("high")),
        "low":          _to_float(f("low")),
        "bid":          _to_float(f("bid", "bid_price")),
        "ask":          _to_float(f("ask", "ask_price")),
    }


def _parse_option_contracts(raw: Any) -> list[dict]:
    """
    Normalise /openapi/instrument/option/contracts response.
    Returns a list of dicts with 'symbol', 'strike', 'type' keys.
    """
    data = raw.get("data") or raw
    if isinstance(data, dict):
        data = data.get("items") or data.get("list") or data.get("contracts") or []
    if not isinstance(data, list):
        return []

    contracts = []
    for item in data:
        sym = item.get("option_symbol") or item.get("symbol") or item.get("optionSymbol", "")
        strike = _to_float(
            item.get("strike_price") or item.get("strikePrice") or item.get("strike")
        )
        raw_type = (
            item.get("option_type") or item.get("right") or item.get("contract_type", "CALL")
        ).upper()
        contract_type = "CALL" if raw_type in ("CALL", "C") else "PUT"

        if sym and strike is not None:
            contracts.append({
                "symbol": sym,
                "strike": strike,
                "type":   contract_type,
                "delta":  _to_float(item.get("delta", 0.0)),
            })

    return contracts


def _parse_option_snapshots(raw: Any) -> dict[str, dict]:
    """
    Normalise /openapi/market-data/option/snapshot response.
    Returns a dict keyed by option_symbol.
    """
    data = raw.get("data") or raw
    if isinstance(data, dict):
        data = data.get("items") or data.get("list") or []
    if not isinstance(data, list):
        return {}

    result: dict[str, dict] = {}
    for item in data:
        sym = item.get("option_symbol") or item.get("symbol") or item.get("optionSymbol", "")
        if not sym:
            continue
        bid = _to_float(item.get("bid", item.get("bid_price", 0.0))) or 0.0
        ask = _to_float(item.get("ask", item.get("ask_price", 0.0))) or 0.0
        result[sym] = {
            "bid":           bid,
            "ask":           ask,
            "mid":           (bid + ask) / 2,
            "volume":        _to_int(item.get("volume", 0)),
            "open_interest": _to_int(item.get("open_interest", item.get("openInterest", 0))),
            "iv":            _to_float(item.get("implied_volatility", item.get("iv", 0.0))),
        }

    return result


def _parse_candles(raw: Any) -> list[dict]:
    """
    Normalise /openapi/market-data/stock/bars response.
    SDK returns: { "data": { "list": [...] } } or { "data": [...] }
    Sorted oldest → newest.
    """
    data = raw.get("data") or raw
    if isinstance(data, dict):
        data = data.get("list") or data.get("bars") or data.get("candles") or []
    if not isinstance(data, list):
        return []

    candles = []
    for item in data:
        if isinstance(item, dict):
            candle = {
                "timestamp": _to_int(item.get("timestamp", item.get("time", 0))),
                "open":  _to_float(item.get("open",  item.get("o"))),
                "high":  _to_float(item.get("high",  item.get("h"))),
                "low":   _to_float(item.get("low",   item.get("l"))),
                "close": _to_float(item.get("close", item.get("c"))),
                "volume": _to_int(item.get("volume", item.get("v", 0))),
            }
            candles.append(candle)
        elif isinstance(item, (list, tuple)) and len(item) >= 6:
            candles.append({
                "timestamp": _to_int(item[0]),
                "open":  _to_float(item[1]),
                "high":  _to_float(item[2]),
                "low":   _to_float(item[3]),
                "close": _to_float(item[4]),
                "volume": _to_int(item[5]),
            })

    candles.sort(key=lambda c: c["timestamp"])
    logger.info("Parsed %d minute candles", len(candles))
    return candles


def _parse_order_response(raw: Any) -> dict:
    data = raw.get("data") or raw
    if isinstance(data, list):
        data = data[0] if data else {}
    return {
        "orderId":      str(data.get("order_id", data.get("orderId", ""))),
        "status":       data.get("status", data.get("order_status", "UNKNOWN")),
        "filled_price": _to_float(data.get("avg_filled_price", data.get("avgFilledPrice"))),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raise_if_error(resp: Any, label: str) -> None:
    """Raise WebullAPIError if the HTTP response is not 2xx."""
    if resp.status_code not in (200, 201, 204):
        raise WebullAPIError(
            f"{label} returned HTTP {resp.status_code}: {resp.text[:300]}"
        )


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0


def _validate_ticker(ticker: str) -> None:
    if ticker not in VALID_TICKERS:
        raise ValueError(
            f"'{ticker}' is not a supported ticker. "
            f"Must be one of: {sorted(VALID_TICKERS)}"
        )


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            "Add it as a GitHub Secret and expose it in your workflow env: block."
        )
    return value


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class WebullAPIError(Exception):
    """Raised when a Webull API call fails."""


# ---------------------------------------------------------------------------
# Convenience: today's 0DTE expiry date string
# ---------------------------------------------------------------------------

def today_expiry() -> str:
    """Return today's date as 'YYYY-MM-DD' in Eastern time."""
    eastern = timezone(timedelta(hours=-5))
    return datetime.now(tz=eastern).strftime("%Y-%m-%d")
