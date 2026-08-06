"""
position_monitor.py
-------------------
After a trade entry alert fires, this module runs as the scalp_monitor
GitHub Actions workflow.  It polls the open position's mid-price every
60 seconds and fires an exit alert when any of three conditions are met:

    1. Take Profit  — mid >= entry × 1.25
    2. Stop Loss    — mid <= entry × 0.85
    3. Time Exit    — time >= 3:45pm ET (hard close, never hold past)

Non-negotiable rules enforced here (spec §10):
    - Hard time exit at 3:45pm regardless of P&L.
    - Uses mid-price for both entry recording and exit detection.

State is read from and written to data/active_trade.json (spec §2.3,
§8).  When an exit condition fires, this file is cleared and the trade
is appended to data/trade_log.csv.
"""

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# Allow running directly (python -m monitor.position_monitor) as well as
# being imported from main.py.  Adjust the import path accordingly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scanner.webull_client import WebullClient, WebullAPIError
from alerts.discord import DiscordAlerter, build_tp_hit_alert, build_sl_hit_alert, build_time_exit_alert

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

EASTERN = timezone(timedelta(hours=-5))

POLL_INTERVAL_SECS = 60

# Hard time-exit: 3:45pm ET (spec §8, §10)
TIME_EXIT_HOUR = 15
TIME_EXIT_MINUTE = 45

# TP / SL multipliers — must match strike_selector.py
TP_MULTIPLIER = 1.25
SL_MULTIPLIER = 0.85

# State files (relative to repo root)
ACTIVE_TRADE_PATH = _REPO_ROOT / "data" / "active_trade.json"
TRADE_LOG_PATH = _REPO_ROOT / "data" / "trade_log.csv"

CSV_HEADERS = [
    "date",
    "ticker",
    "direction",
    "strike",
    "option_symbol",
    "entry_price",
    "entry_time",
    "exit_price",
    "exit_time",
    "exit_reason",
    "pnl_pct",
    "score",
]


# ── Main entry point ─────────────────────────────────────────────────────────

def run_monitor(trade: Optional[dict] = None) -> None:
    """
    Main monitoring loop.

    Parameters
    ----------
    trade : dict (optional)
        Active trade state dict.  If None, reads from active_trade.json.
        Keys expected:
            ticker, direction, strike, option_symbol,
            entry_price, take_profit, stop_loss, entry_time, score
    """
    if trade is None:
        trade = _load_active_trade()
        if trade is None:
            logger.error("No active_trade.json found — nothing to monitor. Exiting.")
            return

    client = WebullClient()
    alerter = DiscordAlerter()

    ticker = trade["ticker"]
    direction = trade["direction"]
    strike = float(trade["strike"])
    option_symbol = trade["option_symbol"]
    entry_price = float(trade["entry_price"])
    take_profit = float(trade["take_profit"])
    stop_loss = float(trade["stop_loss"])
    entry_time_str = trade.get("entry_time", "")
    score = trade.get("score", 0)

    logger.info(
        "Monitor started | %s %s $%.0f | entry=%.2f | TP=%.2f | SL=%.2f",
        ticker, direction, strike, entry_price, take_profit, stop_loss,
    )

    poll_count = 0

    while True:
        now_et = _now_eastern()
        poll_count += 1

        # ── Hard time exit check ───────────────────────────────────────────
        if _is_time_exit(now_et):
            logger.info("TIME EXIT triggered at %s ET", _fmt_time(now_et))

            try:
                quote = client.get_option_quote(option_symbol)
                exit_price = quote["mid"]
            except WebullAPIError as exc:
                logger.warning("Could not fetch final quote: %s — using entry_price", exc)
                exit_price = entry_price

            pnl_pct = _calc_pnl_pct(entry_price, exit_price)
            msg = build_time_exit_alert(
                ticker=ticker,
                direction=direction,
                strike=strike,
                option_symbol=option_symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                entry_time=entry_time_str,
                exit_time=_fmt_time(now_et),
            )
            alerter.send_alert(msg)
            _record_trade(trade, exit_price, _fmt_time(now_et), "TIME_EXIT", pnl_pct)
            _clear_active_trade()
            logger.info("Monitor done (TIME EXIT). P&L: %+.2f%%", pnl_pct)
            return

        # ── Fetch current mid-price ────────────────────────────────────────
        try:
            quote = client.get_option_quote(option_symbol)
            current_mid = quote["mid"]
        except WebullAPIError as exc:
            logger.warning(
                "Poll %d: failed to fetch quote for %s: %s — will retry next cycle",
                poll_count, option_symbol, exc,
            )
            time.sleep(POLL_INTERVAL_SECS)
            continue

        logger.info(
            "Poll %d | %s [%s] mid=%.2f | TP=%.2f | SL=%.2f | time=%s ET",
            poll_count, ticker, direction, current_mid,
            take_profit, stop_loss, _fmt_time(now_et),
        )

        # ── Take Profit ────────────────────────────────────────────────────
        if current_mid >= take_profit:
            pnl_pct = _calc_pnl_pct(entry_price, current_mid)
            exit_price = current_mid
            elapsed = _calc_hold_time(entry_time_str, now_et)

            msg = build_tp_hit_alert(
                ticker=ticker,
                direction=direction,
                strike=strike,
                option_symbol=option_symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                hold_time=elapsed,
            )
            alerter.send_alert(msg)
            _record_trade(trade, exit_price, _fmt_time(now_et), "TP_HIT", pnl_pct)
            _clear_active_trade()
            logger.info("Monitor done (TP HIT). P&L: %+.2f%%", pnl_pct)
            return

        # ── Stop Loss ──────────────────────────────────────────────────────
        if current_mid <= stop_loss:
            pnl_pct = _calc_pnl_pct(entry_price, current_mid)
            exit_price = current_mid
            elapsed = _calc_hold_time(entry_time_str, now_et)

            msg = build_sl_hit_alert(
                ticker=ticker,
                direction=direction,
                strike=strike,
                option_symbol=option_symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                hold_time=elapsed,
            )
            alerter.send_alert(msg)
            _record_trade(trade, exit_price, _fmt_time(now_et), "SL_HIT", pnl_pct)
            _clear_active_trade()
            logger.info("Monitor done (SL HIT). P&L: %+.2f%%", pnl_pct)
            return

        time.sleep(POLL_INTERVAL_SECS)


# ── State management ─────────────────────────────────────────────────────────

def _load_active_trade() -> Optional[dict]:
    """Load and parse active_trade.json. Returns None if file is empty or missing."""
    if not ACTIVE_TRADE_PATH.exists():
        return None
    try:
        text = ACTIVE_TRADE_PATH.read_text().strip()
        if not text:
            return None
        return json.loads(text)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load active_trade.json: %s", exc)
        return None


def _clear_active_trade() -> None:
    """Clear active_trade.json after a position exits (spec §8)."""
    try:
        ACTIVE_TRADE_PATH.write_text("{}")
        logger.info("active_trade.json cleared.")
    except OSError as exc:
        logger.error("Failed to clear active_trade.json: %s", exc)


def write_active_trade(trade: dict) -> None:
    """Write trade state to active_trade.json (called by main.py on entry)."""
    try:
        ACTIVE_TRADE_PATH.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_TRADE_PATH.write_text(json.dumps(trade, indent=2))
        logger.info("active_trade.json written.")
    except OSError as exc:
        logger.error("Failed to write active_trade.json: %s", exc)


# ── Trade logging ─────────────────────────────────────────────────────────────

def _record_trade(
    trade: dict,
    exit_price: float,
    exit_time: str,
    exit_reason: str,
    pnl_pct: float,
) -> None:
    """Append completed trade to trade_log.csv."""
    row = {
        "date": datetime.now(tz=EASTERN).strftime("%Y-%m-%d"),
        "ticker": trade.get("ticker", ""),
        "direction": trade.get("direction", ""),
        "strike": trade.get("strike", ""),
        "option_symbol": trade.get("option_symbol", ""),
        "entry_price": trade.get("entry_price", ""),
        "entry_time": trade.get("entry_time", ""),
        "exit_price": round(exit_price, 2),
        "exit_time": exit_time,
        "exit_reason": exit_reason,
        "pnl_pct": round(pnl_pct, 2),
        "score": trade.get("score", ""),
    }

    try:
        TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = TRADE_LOG_PATH.exists() and TRADE_LOG_PATH.stat().st_size > 0

        with TRADE_LOG_PATH.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info("Trade logged to trade_log.csv: %s", row)
    except OSError as exc:
        logger.error("Failed to write trade_log.csv: %s", exc)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _now_eastern() -> datetime:
    return datetime.now(tz=EASTERN)


def _is_time_exit(now_et: datetime) -> bool:
    """Return True if it's time for the hard close (>= 3:45pm ET)."""
    close_time = now_et.replace(
        hour=TIME_EXIT_HOUR,
        minute=TIME_EXIT_MINUTE,
        second=0,
        microsecond=0,
    )
    return now_et >= close_time


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%I:%M%p").lstrip("0")


def _calc_pnl_pct(entry: float, current: float) -> float:
    if entry <= 0:
        return 0.0
    return ((current - entry) / entry) * 100


def _calc_hold_time(entry_time_str: str, now_et: datetime) -> str:
    """Return human-readable hold duration, e.g. '34 minutes'."""
    try:
        entry_dt = datetime.strptime(entry_time_str, "%I:%M%p").replace(
            year=now_et.year, month=now_et.month, day=now_et.day, tzinfo=EASTERN
        )
        delta_minutes = int((now_et - entry_dt).total_seconds() / 60)
        return f"{delta_minutes} minutes"
    except (ValueError, TypeError):
        return "unknown"


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_monitor()
