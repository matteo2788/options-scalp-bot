"""
webull_client.py
----------------
Single point of contact for all Webull Open API calls.

Base URLs confirmed from developer.webull.com/apis/docs/sdk:
    Production : https://api.webull.com
    Sandbox    : https://api.sandbox.webull.com

Actual endpoint paths (from the API reference):
    GET  /openapi/market-data/stock/snapshot   – real-time quote + volume
    GET  /openapi/market-data/option/snapshot  – option quote (bid/ask/mid)
    GET  /openapi/market-data/stock/bars       – 1-min OHLCV candles
    GET  /openapi/market-data/option/chain     – 0DTE options chain
    POST /openapi/trade/paper/order            – paper order placement
    POST /openapi/auth/token/create            – token creation

Authentication (from developer.webull.com/apis/docs/authentication/signature):
    Required headers on every request:
        x-app-key              : your App Key
        x-timestamp            : ISO-8601 UTC timestamp  e.g. 2025-03-19T10:00:00Z
        x-signature-algorithm  : HMAC-SHA1
        x-signature-version    : 1.0
        x-signature-nonce      : unique UUID per request
        x-signature            : HMAC-SHA1 hex digest

Credentials stored as GitHub Secrets (never hardcoded):
    WEBULL_APP_KEY            – your App Key (not "App ID")
    WEBULL_APP_SECRET         – your App Secret
    WEBULL_PAPER_ACCOUNT_ID   – DEN92WP9
    DISCORD_WEBHOOK_ALERTS    – webhook URL for #alerts
    DISCORD_WEBHOOK_LOG       – webhook URL for #trade-log
"""

import hashlib
import hmac
import logging
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Production host — confirmed at developer.webull.com/apis/docs/sdk
BASE_URL = "https://api.webull.com"

# For local testing against the sandbox, set env var WEBULL_USE_SANDBOX=true
SANDBOX_BASE_URL = "https://api.sandbox.webull.com"

# Tickers the bot is authorised to trade
VALID_TICKERS = {"TSLA", "NVDA", "SPY"}

# Webull instrument IDs (required by the options-chain endpoint)
TICKER_IDS = {
    "TSLA": "913255598",
    "NVDA": "913323282",
    "SPY":  "913243251",
}

# Request timeouts (seconds)
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 15

# Retry on 429 / 5xx with exponential back-off
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5   # delays: 0 s, 1.5 s, 3 s

# ---------------------------------------------------------------------------
# WebullClient
# ---------------------------------------------------------------------------


class WebullClient:
    """Thin, authenticated wrapper around the Webull Open API."""

    def __init__(self) -> None:
        self.app_key    = self._require_env("WEBULL_APP_KEY")
        self.app_secret = self._require_env("WEBULL_APP_SECRET")
        self.paper_account_id = self._require_env("WEBULL_PAPER_ACCOUNT_ID")

        # Allow sandbox override for local testing without changing code
        use_sandbox = os.environ.get("WEBULL_USE_SANDBOX", "").lower() in ("1", "true", "yes")
        self.base_url = SANDBOX_BASE_URL if use_sandbox else BASE_URL
        if use_sandbox:
            logger.info("WebullClient using SANDBOX environment: %s", self.base_url)
        else:
            logger.info("WebullClient using PRODUCTION environment: %s", self.base_url)

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

    def _sign(self, timestamp: str, nonce: str, path: str, body_str: str = "") -> str:
        """
        HMAC-SHA1 signature per developer.webull.com/apis/docs/authentication/signature

        Signature string components (joined with newlines):
            <timestamp>\\n<nonce>\\n<path>\\n<body_str>

        body_str is the raw JSON body for POST requests; empty for GETs.
        The result is hex-encoded and sent as the 'x-signature' header.
        """
        message = f"{timestamp}\n{nonce}\n{path}\n{body_str}"
        sig = hmac.new(
            self.app_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha1,
        ).hexdigest()
        return sig

    def _headers(self, path: str, body_str: str = "") -> dict[str, str]:
        """
        Build the required authentication headers for every request.

        Per developer.webull.com/apis/docs/authentication/signature the
        mandatory headers are:
            x-app-key             : App Key credential
            x-timestamp           : ISO-8601 UTC (e.g. 2025-03-19T10:00:00Z)
            x-signature-algorithm : HMAC-SHA1
            x-signature-version   : 1.0
            x-signature-nonce     : unique UUID per request
            x-signature           : computed HMAC-SHA1 hex digest
        """
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        nonce = str(uuid.uuid4())
        signature = self._sign(timestamp, nonce, path, body_str)
        return {
            "Content-Type":          "application/json",
            "x-app-key":             self.app_key,
            "x-timestamp":           timestamp,
            "x-signature-algorithm": "HMAC-SHA1",
            "x-signature-version":   "1.0",
            "x-signature-nonce":     nonce,
            "x-signature":           signature,
        }

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        """
        Execute a signed GET request and return parsed JSON.

        Raises:
            WebullAPIError: on non-2xx responses or network failures.
        """
        url = f"{self.base_url}{path}"
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
        import json as _json
        url = f"{self.base_url}{path}"
        body_str = _json.dumps(body, separators=(",", ":"))
        headers = self._headers(path, body_str)
        try:
            resp = self.session.post(
                url,
                headers=headers,
                data=body_str,
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
        Webull sometimes returns HTTP 200 with a business-logic error code.
        Check the common envelope fields that signal an API-level failure.
        """
        if isinstance(data, dict):
            code = (
                data.get("code")
                or data.get("retCode")
                or data.get("result_code")
                or data.get("error_code")
            )
            if code and str(code) not in ("0", "200", "success", ""):
                msg = (
                    data.get("msg")
                    or data.get("message")
                    or data.get("error_message")
                    or str(data)
                )
                raise WebullAPIError(f"API error on {path}: [{code}] {msg}")

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_ticker_snapshot(self, ticker: str) -> dict:
        """
        GET /openapi/market-data/stock/snapshot

        Returns a normalised dict:
            {
                "symbol":       str,
                "last":         float,   # latest trade price
                "volume":       int,     # shares traded today so far
                "avgVolume20D": float,   # 20-day average daily volume
                "vwap":         float,   # today's VWAP
                "open":         float,
                "high":         float,
                "low":          float,
                "bid":          float,
                "ask":          float,
            }
        """
        _validate_ticker(ticker)
        path = "/openapi/market-data/stock/snapshot"
        # The API accepts ticker symbols directly via the 'symbols' param
        # and also accepts category to disambiguate (US_STOCK for equities)
        raw = self._get(path, params={
            "symbols":  ticker,
            "category": "US_STOCK",
            "extend_hour_required":  "false",
            "overnight_required":    "false",
        })
        return _parse_snapshot(raw, ticker)

    def get_options_chain(self, ticker: str, expiry_date: str) -> list[dict]:
        """
        GET /openapi/market-data/option/chain

        Returns the 0DTE options chain as a flat list of contract dicts:
            [
                {
                    "strike":        float,
                    "type":          "CALL" | "PUT",
                    "open_interest": int,
                    "volume":        int,
                    "bid":           float,
                    "ask":           float,
                    "mid":           float,   # (bid+ask)/2
                    "iv":            float,
                    "delta":         float,
                    "symbol":        str,     # OCC option symbol
                },
                ...
            ]

        Parameters
        ----------
        ticker      : underlying symbol ("TSLA", "NVDA", "SPY")
        expiry_date : "YYYY-MM-DD" — today's date for 0DTE
        """
        _validate_ticker(ticker)
        path = "/openapi/market-data/option/chain"
        raw = self._get(path, params={
            "symbol":      ticker,
            "expiry_date": expiry_date,
            "category":    "US_STOCK_OPTION",
        })
        return _parse_options_chain(raw)

    def get_minute_candles(self, ticker: str, count: int = 10) -> list[dict]:
        """
        GET /openapi/market-data/stock/bars

        Returns the most recent ``count`` one-minute OHLCV candles,
        sorted oldest → newest:
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
        """
        _validate_ticker(ticker)
        path = "/openapi/market-data/stock/bars"
        raw = self._get(path, params={
            "symbol":       ticker,
            "category":     "US_STOCK",
            "granularity":  "M1",    # 1-minute bars
            "count":        count,
            "type":         "1",     # regular session
        })
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
        POST /openapi/trade/paper/order

        Places a simulated order in the Webull paper account.  Used for
        logging/record-keeping — the bot manages exits independently.

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

        path = "/openapi/trade/paper/order"
        body = {
            "account_id":     self.paper_account_id,
            "action":         side,
            "asset_type":     "OPTION",
            "ticker_symbol":  ticker,
            "option_symbol":  option_symbol,
            "qty":            quantity,
            "order_type":     order_type,
            "time_in_force":  "DAY",
        }
        raw = self._post(path, body)
        return _parse_order_response(raw)

    def get_option_quote(self, option_symbol: str) -> dict:
        """
        GET /openapi/market-data/option/snapshot

        Polls the bid/ask/mid of a single open option contract.
        Used by position_monitor.py every 60 seconds.

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
        path = "/openapi/market-data/option/snapshot"
        raw = self._get(path, params={
            "symbols":  option_symbol,
            "category": "US_STOCK_OPTION",
        })
        return _parse_option_quote(raw, option_symbol)


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_snapshot(raw: Any, ticker: str) -> dict:
    """
    Normalise /openapi/market-data/stock/snapshot response.

    The API returns data under a 'data' key, which may be a list
    (batch endpoint) or a single dict.
    """
    data = raw.get("data") or raw
    if isinstance(data, list):
        # Batch snapshot wraps each ticker in a list entry
        data = data[0] if data else {}

    def _field(*names, default=None):
        for n in names:
            v = data.get(n)
            if v is not None:
                return v
        return default

    last  = _to_float(_field("close", "latest_price", "last_price", "latestPrice"))
    vol   = _to_int(_field("volume", "vol"))
    avg20 = _to_float(_field("avg_volume_20d", "avgVolume20D", "avg_vol_20d", "avgVol20D"))
    vwap  = _to_float(_field("vwap", "VWAP")) or last

    return {
        "symbol":       ticker,
        "last":         last,
        "volume":       vol,
        "avgVolume20D": avg20,
        "vwap":         vwap,
        "open":         _to_float(_field("open")),
        "high":         _to_float(_field("high")),
        "low":          _to_float(_field("low")),
        "bid":          _to_float(_field("bid", "bid_price")),
        "ask":          _to_float(_field("ask", "ask_price")),
    }


def _parse_options_chain(raw: Any) -> list[dict]:
    """
    Normalise /openapi/market-data/option/chain response to a flat list.

    The chain is typically returned as:
        { "data": { "call": [...], "put": [...] } }
    or as a list of strike objects each with .call/.put sub-objects.
    """
    data = raw.get("data") or raw

    contracts = []

    # Shape 1: { "call": [...], "put": [...] }
    if isinstance(data, dict) and ("call" in data or "put" in data):
        for side in ("call", "put"):
            side_list = data.get(side) or []
            for item in side_list:
                c = _parse_single_contract(item, side.upper())
                if c:
                    contracts.append(c)
        return contracts

    # Shape 2: list of strike-level objects with nested call/put
    if isinstance(data, list):
        for row in data:
            for side in ("call", "put"):
                sub = row.get(side) or row.get(side.upper())
                if isinstance(sub, dict):
                    c = _parse_single_contract(sub, side.upper())
                    if c:
                        contracts.append(c)
            # Shape 3: flat list where each item is already a contract
            if "strike_price" in row or "strikePrice" in row or "strike" in row:
                side_raw = row.get("side") or row.get("type") or row.get("right") or "CALL"
                c = _parse_single_contract(row, str(side_raw).upper())
                if c:
                    contracts.append(c)

    logger.info("Parsed %d option contracts from chain", len(contracts))
    return contracts


def _parse_single_contract(raw: dict, side: str) -> Optional[dict]:
    """Parse one call/put contract dict."""
    strike = _to_float(
        raw.get("strike_price") or raw.get("strikePrice") or raw.get("strike")
    )
    if strike is None:
        return None

    bid = _to_float(raw.get("bid", raw.get("bid_price", 0.0))) or 0.0
    ask = _to_float(raw.get("ask", raw.get("ask_price", 0.0))) or 0.0
    mid = (bid + ask) / 2 if (bid and ask) else 0.0

    return {
        "strike":        strike,
        "type":          side,
        "open_interest": _to_int(raw.get("open_interest", raw.get("openInterest", 0))),
        "volume":        _to_int(raw.get("volume", raw.get("vol", 0))),
        "bid":           bid,
        "ask":           ask,
        "mid":           mid,
        "iv":            _to_float(raw.get("implied_volatility", raw.get("impliedVolatility", 0.0))),
        "delta":         _to_float(raw.get("delta", 0.0)),
        "symbol":        raw.get("symbol", raw.get("option_symbol", raw.get("optionSymbol", ""))),
    }


def _parse_candles(raw: Any) -> list[dict]:
    """
    Normalise /openapi/market-data/stock/bars response to a list of OHLCV dicts,
    sorted oldest → newest.
    """
    data = raw.get("data") or raw
    if isinstance(data, dict):
        data = (
            data.get("bars")
            or data.get("candles")
            or data.get("tickList")
            or data.get("list")
            or []
        )

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

    candles.sort(key=lambda c: c["timestamp"])
    logger.info("Parsed %d minute candles", len(candles))
    return candles


def _parse_order_response(raw: Any) -> dict:
    """Normalise paper order placement response."""
    data = raw.get("data") or raw
    if isinstance(data, list):
        data = data[0] if data else {}

    return {
        "orderId":      str(data.get("order_id", data.get("orderId", ""))),
        "status":       data.get("status", data.get("order_status", "UNKNOWN")),
        "filled_price": _to_float(data.get("avg_filled_price", data.get("avgFilledPrice"))),
    }


def _parse_option_quote(raw: Any, symbol: str) -> dict:
    """Normalise a single-option quote response for the position monitor."""
    data = raw.get("data") or raw
    if isinstance(data, list):
        data = data[0] if data else {}

    bid = _to_float(data.get("bid", data.get("bid_price", 0.0))) or 0.0
    ask = _to_float(data.get("ask", data.get("ask_price", 0.0))) or 0.0
    mid = (bid + ask) / 2

    return {
        "symbol":        symbol,
        "bid":           bid,
        "ask":           ask,
        "mid":           mid,
        "volume":        _to_int(data.get("volume", 0)),
        "open_interest": _to_int(data.get("open_interest", data.get("openInterest", 0))),
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
# Convenience helper: today's 0DTE expiry date string
# ---------------------------------------------------------------------------

def today_expiry() -> str:
    """Return today's date as 'YYYY-MM-DD' in Eastern time."""
    eastern = timezone(timedelta(hours=-5))   # EST; CI cron accounts for EDT
    return datetime.now(tz=eastern).strftime("%Y-%m-%d")
