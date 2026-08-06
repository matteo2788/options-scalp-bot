"""
webull_client.py
----------------
Single point of contact for all Webull Open API calls.

All endpoints, authentication, retry logic, and response parsing live
here so that the rest of the bot can stay free of HTTP boilerplate.

Credentials are read exclusively from environment variables (set as
GitHub Secrets in CI):
    WEBULL_APP_ID
    WEBULL_APP_SECRET
    WEBULL_PAPER_ACCOUNT_ID

Reference endpoints (Webull Open API v1):
    GET  /quotes/tickerSnapshot        – real-time quote + volume
    GET  /options/chain                – 0DTE options chain (OI, volume, greeks)
    GET  /quotes/tickerChart/query     – 1-min OHLCV candles for momentum
    POST /trade/paper/order            – paper order placement (logging only)
"""

import os
import time
import logging
import hashlib
import hmac
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://openapi.webull.com/openapi"
API_VERSION = "v1"

# Tickers the bot is authorised to trade
VALID_TICKERS = {"TSLA", "NVDA", "SPY"}

# Webull instrument IDs (required by the API for options chain queries).
# These are the standard Webull tickerIds for the three underlyings.
TICKER_IDS = {
    "TSLA": "913255598",
    "NVDA": "913323282",
    "SPY":  "913243251",
}

# Request timeouts (seconds)
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 15

# Retry config: retry on 429 / 5xx, backoff between attempts
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5  # waits 0s, 1.5s, 3s between retries

# ---------------------------------------------------------------------------
# WebullClient
# ---------------------------------------------------------------------------


class WebullClient:
    """Thin, authenticated wrapper around the Webull Open API."""

    def __init__(self) -> None:
        self.app_id = self._require_env("WEBULL_APP_ID")
        self.app_secret = self._require_env("WEBULL_APP_SECRET")
        self.paper_account_id = self._require_env("WEBULL_PAPER_ACCOUNT_ID")

        self.session = self._build_session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_env(name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise EnvironmentError(
                f"Required environment variable '{name}' is not set. "
                "Add it as a GitHub Secret and expose it in your workflow env: block."
            )
        return value

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _sign_request(self, path: str, timestamp: str) -> str:
        """
        HMAC-SHA256 signature expected by the Webull Open API.

        Signature string: <app_id>\n<timestamp>\n<path>
        The result is hex-encoded and passed as the 'sign' header.
        """
        message = f"{self.app_id}\n{timestamp}\n{path}"
        sig = hmac.new(
            self.app_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return sig

    def _headers(self, path: str) -> dict[str, str]:
        """Build authentication headers for every request."""
        timestamp = str(int(time.time() * 1000))  # milliseconds
        return {
            "Content-Type": "application/json",
            "appId": self.app_id,
            "timestamp": timestamp,
            "sign": self._sign_request(path, timestamp),
        }

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        """
        Execute a signed GET request and return parsed JSON.

        Raises:
            WebullAPIError: on non-2xx responses or network failures.
        """
        url = f"{BASE_URL}/{API_VERSION}{path}"
        headers = self._headers(path)
        try:
            resp = self.session.get(
                url,
                headers=headers,
                params=params or {},
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            resp.raise_for_status()
            data = resp.json()
            self._check_api_error(data, path)
            return data
        except requests.RequestException as exc:
            raise WebullAPIError(f"GET {path} failed: {exc}") from exc

    def _post(self, path: str, body: dict) -> Any:
        """
        Execute a signed POST request and return parsed JSON.

        Raises:
            WebullAPIError: on non-2xx responses or network failures.
        """
        url = f"{BASE_URL}/{API_VERSION}{path}"
        headers = self._headers(path)
        try:
            resp = self.session.post(
                url,
                headers=headers,
                json=body,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            resp.raise_for_status()
            data = resp.json()
            self._check_api_error(data, path)
            return data
        except requests.RequestException as exc:
            raise WebullAPIError(f"POST {path} failed: {exc}") from exc

    @staticmethod
    def _check_api_error(data: Any, path: str) -> None:
        """
        Webull returns HTTP 200 even for business-logic errors.
        Check for the 'code' / 'msg' envelope that signals an API-level failure.
        """
        if isinstance(data, dict):
            code = data.get("code") or data.get("retCode") or data.get("result_code")
            if code and str(code) not in ("0", "200", "success", ""):
                msg = data.get("msg") or data.get("message") or str(data)
                raise WebullAPIError(f"API error on {path}: [{code}] {msg}")

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_ticker_snapshot(self, ticker: str) -> dict:
        """
        GET /quotes/tickerSnapshot

        Returns a dict with at minimum:
            {
                "symbol":         str,
                "last":           float,   # current price
                "volume":         int,     # shares traded today so far
                "avgVolume20D":   float,   # 20-day average daily volume
                "vwap":           float,   # today's VWAP
                "open":           float,
                "high":           float,
                "low":            float,
                "bid":            float,
                "ask":            float,
            }

        Notes
        -----
        The relative-volume calculation in ticker_picker.py needs both
        ``volume`` (intraday so far) and ``avgVolume20D``.  The avg is the
        full-day 20-day average; the scorer normalises for time of day.
        """
        _validate_ticker(ticker)
        ticker_id = TICKER_IDS[ticker]
        path = "/quotes/tickerSnapshot"
        raw = self._get(path, params={"tickerId": ticker_id})
        return _parse_snapshot(raw, ticker)

    def get_options_chain(self, ticker: str, expiry_date: str) -> list[dict]:
        """
        GET /options/chain

        Returns the full 0DTE options chain as a list of contract dicts:
            [
                {
                    "strike":       float,
                    "type":         "CALL" | "PUT",
                    "open_interest": int,
                    "volume":       int,
                    "bid":          float,
                    "ask":          float,
                    "mid":          float,   # computed: (bid+ask)/2
                    "iv":           float,   # implied volatility
                    "delta":        float,
                    "symbol":       str,     # OCC symbol e.g. TSLA240101C00300000
                },
                ...
            ]

        Parameters
        ----------
        ticker      : underlying ticker symbol ("TSLA", "NVDA", "SPY")
        expiry_date : "YYYY-MM-DD" — today's date for 0DTE
        """
        _validate_ticker(ticker)
        ticker_id = TICKER_IDS[ticker]
        path = "/options/chain"
        raw = self._get(
            path,
            params={
                "tickerId": ticker_id,
                "expireDate": expiry_date,
                "count": -1,   # -1 = return full chain
            },
        )
        return _parse_options_chain(raw)

    def get_minute_candles(self, ticker: str, count: int = 10) -> list[dict]:
        """
        GET /quotes/tickerChart/query

        Returns the most recent ``count`` one-minute OHLCV candles as a
        list ordered oldest → newest:
            [
                {
                    "timestamp": int,    # Unix ms
                    "open":      float,
                    "high":      float,
                    "low":       float,
                    "close":     float,
                    "volume":    int,
                },
                ...
            ]

        The last element in the list is the most recent completed candle.
        entry_timer.py and direction_engine.py use the last 3 candles for
        momentum analysis.
        """
        _validate_ticker(ticker)
        ticker_id = TICKER_IDS[ticker]
        path = "/quotes/tickerChart/query"
        raw = self._get(
            path,
            params={
                "tickerId": ticker_id,
                "type": "m1",       # 1-minute bars
                "count": count,
                "extendTrading": 0, # regular session only
            },
        )
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
        POST /trade/paper/order

        Places a simulated order in the Webull paper account.  This is
        called for logging/record-keeping purposes only — the bot does NOT
        rely on Webull to manage the position or track exits.

        Parameters
        ----------
        ticker        : underlying e.g. "TSLA"
        option_symbol : OCC option symbol e.g. "TSLA240101C00300000"
        side          : "BUY" | "SELL"
        quantity      : number of contracts (default 1)
        order_type    : "MKT" (market) — always MKT for scalp entries

        Returns a dict with at minimum:
            {
                "orderId": str,
                "status":  str,   # "FILLED" | "PENDING" | etc.
                "filled_price": float | None,
            }
        """
        _validate_ticker(ticker)
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be 'BUY' or 'SELL', got '{side}'")

        path = "/trade/paper/order"
        body = {
            "accountId": self.paper_account_id,
            "action": side,
            "assetType": "OPTION",
            "tickerSymbol": ticker,
            "optionSymbol": option_symbol,
            "qty": quantity,
            "orderType": order_type,
            "timeInForce": "DAY",
        }
        raw = self._post(path, body)
        return _parse_order_response(raw)

    def get_option_quote(self, option_symbol: str) -> dict:
        """
        GET /quotes/tickerSnapshot for a specific option contract.

        Used by position_monitor.py to poll the mid-price of the open
        position every 60 seconds without re-fetching the full chain.

        Returns:
            {
                "symbol": str,
                "bid":    float,
                "ask":    float,
                "mid":    float,   # (bid+ask)/2
                "volume": int,
                "open_interest": int,
            }
        """
        path = "/quotes/tickerSnapshot"
        # Option quotes use the OCC symbol as the tickerSymbol param
        raw = self._get(path, params={"symbol": option_symbol, "type": "option"})
        return _parse_option_quote(raw, option_symbol)


# ---------------------------------------------------------------------------
# Response parsers — isolate Webull's JSON shape from the rest of the bot
# ---------------------------------------------------------------------------

def _parse_snapshot(raw: Any, ticker: str) -> dict:
    """
    Normalise /quotes/tickerSnapshot response to the shape the bot expects.

    Webull returns data nested under a 'data' key in most responses;
    we handle both the wrapped and unwrapped shapes defensively.
    """
    data = raw.get("data") or raw
    if isinstance(data, list):
        # Some endpoints wrap a single item in a list
        data = data[0] if data else {}

    # Helper: try multiple possible field names (API has changed over versions)
    def _field(*names, default=None):
        for n in names:
            v = data.get(n)
            if v is not None:
                return v
        return default

    last = _to_float(_field("close", "last", "latestPrice", "price"))
    volume = _to_int(_field("volume", "vol"))
    avg_volume = _to_float(_field("avgVolume20D", "avgVol20D", "averageVolume"))

    # VWAP is not always present; fall back to last price so the direction
    # engine can still function (it will be less precise but won't crash).
    vwap = _to_float(_field("vwap", "VWAP")) or last

    return {
        "symbol": ticker,
        "last": last,
        "volume": volume,
        "avgVolume20D": avg_volume,
        "vwap": vwap,
        "open": _to_float(_field("open")),
        "high": _to_float(_field("high")),
        "low": _to_float(_field("low")),
        "bid": _to_float(_field("bid", "bidPrice")),
        "ask": _to_float(_field("ask", "askPrice")),
    }


def _parse_options_chain(raw: Any) -> list[dict]:
    """
    Normalise /options/chain response to a flat list of contract dicts.

    Webull returns the chain as two nested lists:
        data.optionList → list of strikes
        each strike has .call and .put sub-objects

    We flatten to a single list of contracts with a 'type' field.
    """
    data = raw.get("data") or raw
    if isinstance(data, dict):
        option_list = data.get("optionList") or data.get("data") or []
    elif isinstance(data, list):
        option_list = data
    else:
        logger.warning("Unexpected options chain shape: %s", type(data))
        return []

    contracts = []
    for row in option_list:
        for side in ("call", "put"):
            contract_raw = row.get(side) or row.get(side.upper())
            if not contract_raw:
                continue
            contract = _parse_single_contract(contract_raw, side.upper())
            if contract:
                contracts.append(contract)

    # Also handle flat list (some API versions return all contracts directly)
    if not contracts and option_list and isinstance(option_list[0], dict):
        if "strike" in option_list[0] or "strikePrice" in option_list[0]:
            for contract_raw in option_list:
                side = contract_raw.get("direction", contract_raw.get("right", "CALL")).upper()
                contract = _parse_single_contract(contract_raw, side)
                if contract:
                    contracts.append(contract)

    logger.info("Parsed %d option contracts from chain", len(contracts))
    return contracts


def _parse_single_contract(raw: dict, side: str) -> Optional[dict]:
    """Parse one call or put contract dict from the chain."""
    strike = _to_float(
        raw.get("strikePrice") or raw.get("strike") or raw.get("strike_price")
    )
    if strike is None:
        return None

    bid = _to_float(raw.get("bid", raw.get("bidPrice", 0.0))) or 0.0
    ask = _to_float(raw.get("ask", raw.get("askPrice", 0.0))) or 0.0
    mid = (bid + ask) / 2 if (bid and ask) else 0.0

    return {
        "strike": strike,
        "type": side,   # "CALL" or "PUT"
        "open_interest": _to_int(raw.get("openInterest", raw.get("oi", 0))),
        "volume": _to_int(raw.get("volume", raw.get("vol", 0))),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "iv": _to_float(raw.get("impliedVolatility", raw.get("iv", 0.0))),
        "delta": _to_float(raw.get("delta", 0.0)),
        "symbol": raw.get("symbol", raw.get("optionSymbol", "")),
    }


def _parse_candles(raw: Any) -> list[dict]:
    """
    Normalise /quotes/tickerChart/query response to a list of OHLCV dicts.

    Webull returns candle data in a 'data' list of tick arrays or objects.
    We normalise to dicts and return oldest-first.
    """
    data = raw.get("data") or raw
    if isinstance(data, dict):
        data = data.get("tickList") or data.get("data") or data.get("candles") or []

    if not isinstance(data, list):
        logger.warning("Unexpected candle data shape: %s", type(data))
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
        elif isinstance(item, (list, tuple)) and len(item) >= 6:
            # Some versions return [timestamp, open, high, low, close, volume]
            candle = {
                "timestamp": _to_int(item[0]),
                "open":  _to_float(item[1]),
                "high":  _to_float(item[2]),
                "low":   _to_float(item[3]),
                "close": _to_float(item[4]),
                "volume": _to_int(item[5]),
            }
        else:
            continue
        candles.append(candle)

    # Ensure oldest-first order
    candles.sort(key=lambda c: c["timestamp"])
    logger.info("Parsed %d minute candles", len(candles))
    return candles


def _parse_order_response(raw: Any) -> dict:
    """Normalise paper order placement response."""
    data = raw.get("data") or raw
    if isinstance(data, list):
        data = data[0] if data else {}

    return {
        "orderId": str(data.get("orderId", data.get("order_id", ""))),
        "status": data.get("status", data.get("orderStatus", "UNKNOWN")),
        "filled_price": _to_float(data.get("avgFilledPrice", data.get("filledPrice"))),
    }


def _parse_option_quote(raw: Any, symbol: str) -> dict:
    """Normalise a single-option quote response for the position monitor."""
    data = raw.get("data") or raw
    if isinstance(data, list):
        data = data[0] if data else {}

    bid = _to_float(data.get("bid", data.get("bidPrice", 0.0))) or 0.0
    ask = _to_float(data.get("ask", data.get("askPrice", 0.0))) or 0.0
    mid = (bid + ask) / 2

    return {
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "volume": _to_int(data.get("volume", 0)),
        "open_interest": _to_int(data.get("openInterest", data.get("oi", 0))),
    }


# ---------------------------------------------------------------------------
# Type-coercion utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class WebullAPIError(Exception):
    """Raised when the Webull API returns an error or the request fails."""


# ---------------------------------------------------------------------------
# Convenience: today's expiry date string (0DTE = today)
# ---------------------------------------------------------------------------

def today_expiry() -> str:
    """Return today's date as 'YYYY-MM-DD' in Eastern time (market timezone)."""
    eastern = timezone(timedelta(hours=-5))  # EST; adjust for EDT if needed
    now_et = datetime.now(tz=eastern)
    return now_et.strftime("%Y-%m-%d")
