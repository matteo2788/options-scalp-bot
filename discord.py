"""
alerts/discord.py
-----------------
Formats and sends all Discord messages for the scalp bot.

Two channels are used (spec §2.2, §7):
    #alerts   — entry alerts, exit alerts (TP / SL / time)
    #trade-log — daily scan summary, no-trade log

All messages are sent as plain text via the Discord Incoming Webhook API.
Webhook URLs are read from environment variables:
    DISCORD_WEBHOOK_ALERTS  — for #alerts channel
    DISCORD_WEBHOOK_LOG     — for #trade-log channel

Message formats exactly match spec §7.1–§7.3.
"""

import logging
import os
from datetime import datetime, timezone, timedelta

import requests

logger = logging.getLogger(__name__)

EASTERN = timezone(timedelta(hours=-5))

# Discord will reject payloads > 2000 characters. Our messages are well
# under this limit, but guard against edge-case long ticker names.
MAX_MESSAGE_LENGTH = 1900

# Retry config for webhook delivery
_WEBHOOK_TIMEOUT = 10  # seconds
_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# DiscordAlerter — the main class used by other modules
# ---------------------------------------------------------------------------

class DiscordAlerter:
    """Send formatted messages to the configured Discord channels."""

    def __init__(self) -> None:
        self._alerts_url = _require_env("DISCORD_WEBHOOK_ALERTS")
        self._log_url = _require_env("DISCORD_WEBHOOK_LOG")

    def send_alert(self, message: str) -> bool:
        """Send to #alerts channel. Returns True on success."""
        return _post_webhook(self._alerts_url, message)

    def send_log(self, message: str) -> bool:
        """Send to #trade-log channel. Returns True on success."""
        return _post_webhook(self._log_url, message)


# ---------------------------------------------------------------------------
# Message builders — each returns a ready-to-send string
# ---------------------------------------------------------------------------

def build_entry_alert(
    ticker: str,
    direction: str,
    strike: float,
    option_symbol: str,
    entry_price: float,
    take_profit: float,
    stop_loss: float,
    score: int,
    reason: str,
    entry_time: str,
) -> str:
    """
    Spec §7.1 — Entry Alert (#alerts channel).

    Example output:
        🔥 0DTE SCALP ALERT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📌 Ticker:      TSLA
        📋 Contract:    TSLA $320 CALL  │  0DTE
        💰 Entry Price: $2.40  (mid-price at fill)
        🎯 Take Profit: $3.00  (+25%)
        🛑 Stop Loss:   $2.04  (-15%)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        📊 Score:       87/100
        📈 Direction:   CALL
        🧠 Reason:      OI call-heavy above + price above VWAP
        ⏰ Entry Time:  9:47am EST
    """
    contract_label = f"{ticker} ${strike:.0f} {direction}  │  0DTE"
    tp_pct = (take_profit / entry_price - 1) * 100
    sl_pct = (stop_loss / entry_price - 1) * 100
    direction_emoji = "📈" if direction == "CALL" else "📉"

    return (
        f"🔥 0DTE SCALP ALERT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 Ticker:      {ticker}\n"
        f"📋 Contract:    {contract_label}\n"
        f"💰 Entry Price: ${entry_price:.2f}  (mid-price at fill)\n"
        f"🎯 Take Profit: ${take_profit:.2f}  (+{tp_pct:.0f}%)\n"
        f"🛑 Stop Loss:   ${stop_loss:.2f}  ({sl_pct:.0f}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Score:       {score}/100\n"
        f"{direction_emoji} Direction:   {direction}\n"
        f"🧠 Reason:      {reason}\n"
        f"⏰ Entry Time:  {entry_time} EST"
    )


def build_tp_hit_alert(
    ticker: str,
    direction: str,
    strike: float,
    option_symbol: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    hold_time: str,
) -> str:
    """
    Spec §7.2 — TP Hit Exit Alert (#alerts channel).

    Example output:
        ✅ TP HIT — TSLA $320 CALL
        Entry: $2.40  →  Exit: $3.00
        Result: +25.0% 🔥
        Time held: 34 minutes
    """
    contract_label = f"{ticker} ${strike:.0f} {direction}"
    pnl_sign = "+" if pnl_pct >= 0 else ""

    return (
        f"✅ TP HIT — {contract_label}\n"
        f"Entry: ${entry_price:.2f}  →  Exit: ${exit_price:.2f}\n"
        f"Result: {pnl_sign}{pnl_pct:.1f}% 🔥\n"
        f"Time held: {hold_time}"
    )


def build_sl_hit_alert(
    ticker: str,
    direction: str,
    strike: float,
    option_symbol: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    hold_time: str,
) -> str:
    """
    Spec §7.2 — SL Hit Exit Alert (#alerts channel).

    Example output:
        🛑 SL HIT — TSLA $320 CALL
        Entry: $2.40  →  Exit: $2.04
        Result: -15.0%
        Time held: 12 minutes
    """
    contract_label = f"{ticker} ${strike:.0f} {direction}"
    pnl_sign = "+" if pnl_pct >= 0 else ""

    return (
        f"🛑 SL HIT — {contract_label}\n"
        f"Entry: ${entry_price:.2f}  →  Exit: ${exit_price:.2f}\n"
        f"Result: {pnl_sign}{pnl_pct:.1f}%\n"
        f"Time held: {hold_time}"
    )


def build_time_exit_alert(
    ticker: str,
    direction: str,
    strike: float,
    option_symbol: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    entry_time: str,
    exit_time: str,
) -> str:
    """
    Spec §7.2 — Time Exit Alert (#alerts channel, at 3:45pm).

    Example output:
        ⏳ TIME EXIT — TSLA $320 CALL
        Entry: $2.40  →  Exit: $2.61
        Result: +8.75%  (closed at 3:45pm to avoid theta decay)
    """
    contract_label = f"{ticker} ${strike:.0f} {direction}"
    pnl_sign = "+" if pnl_pct >= 0 else ""

    return (
        f"⏳ TIME EXIT — {contract_label}\n"
        f"Entry: ${entry_price:.2f}  →  Exit: ${exit_price:.2f}\n"
        f"Result: {pnl_sign}{pnl_pct:.2f}%  (closed at 3:45pm to avoid theta decay)"
    )


def build_no_trade_log(
    winner_ticker: str,
    score: int,
    reason: str,
) -> str:
    """
    Spec §7.3 — No-Trade Daily Log (#trade-log channel).

    Example output:
        📋 DAILY SCAN — Thu Aug 06 2026
        Winner: TSLA (Score: 71/100)
        Status: NO TRADE — entry conditions never met within window
        Window: 9:30am – 10:15am EST
        Reason: ATM spread too wide (14% of mid) throughout window
    """
    now_et = datetime.now(tz=EASTERN)
    date_str = now_et.strftime("%a %b %d %Y")

    return (
        f"📋 DAILY SCAN — {date_str}\n"
        f"Winner: {winner_ticker} (Score: {score}/100)\n"
        f"Status: NO TRADE — entry conditions never met within window\n"
        f"Window: 9:30am – 10:15am EST\n"
        f"Reason: {reason}"
    )


def build_direction_conflict_log(
    winner_ticker: str,
    score: int,
    conflict_reason: str,
) -> str:
    """
    Logged to #trade-log when signals conflict (spec §5).
    The bot enters the entry window but CALL/PUT signals don't agree.
    """
    now_et = datetime.now(tz=EASTERN)
    date_str = now_et.strftime("%a %b %d %Y")

    return (
        f"📋 DAILY SCAN — {date_str}\n"
        f"Winner: {winner_ticker} (Score: {score}/100)\n"
        f"Status: NO TRADE — directional signals conflict\n"
        f"Reason: {conflict_reason}"
    )


def build_spread_gate_log(
    winner_ticker: str,
    score: int,
    spread_pct: float,
) -> str:
    """
    Logged to #trade-log when spread gate blocks the strike (spec §10).
    Entry conditions were met but no eligible ATM strike found.
    """
    now_et = datetime.now(tz=EASTERN)
    date_str = now_et.strftime("%a %b %d %Y")

    return (
        f"📋 DAILY SCAN — {date_str}\n"
        f"Winner: {winner_ticker} (Score: {score}/100)\n"
        f"Status: NO TRADE — ATM spread gate failed\n"
        f"Reason: best available spread {spread_pct:.1f}% of mid (limit 10%)"
    )


# ---------------------------------------------------------------------------
# Low-level webhook sender
# ---------------------------------------------------------------------------

def _post_webhook(url: str, message: str) -> bool:
    """
    POST a message to a Discord Incoming Webhook URL.

    Retries up to _MAX_RETRIES times on 429 (rate limit) or 5xx errors.
    Returns True on success, False on permanent failure.
    """
    # Trim to Discord's limit with an indicator
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH - 3] + "..."

    payload = {"content": message}

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=_WEBHOOK_TIMEOUT,
            )
            if resp.status_code in (200, 204):
                logger.info("Discord message delivered (attempt %d).", attempt)
                return True
            elif resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 1.0)
                logger.warning("Discord rate-limited. Retrying in %.1fs…", retry_after)
                import time
                time.sleep(float(retry_after) + 0.1)
            else:
                logger.warning(
                    "Discord webhook returned %d: %s (attempt %d)",
                    resp.status_code, resp.text[:200], attempt,
                )
        except requests.RequestException as exc:
            logger.error("Discord webhook request failed (attempt %d): %s", attempt, exc)

    logger.error("Discord message delivery failed after %d attempts.", _MAX_RETRIES)
    return False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            "Add it as a GitHub Secret and include it in your workflow env: block."
        )
    return value
