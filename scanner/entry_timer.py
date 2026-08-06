"""
entry_timer.py
--------------
Smart entry timing — spec §4.

The bot does NOT fire at 9:30am. Instead this module polls every 60
seconds from 9:30am and checks four conditions before it will authorise
entry.  If all four conditions are met simultaneously, it returns
``EntryDecision(ok=True)``.  If they are never all true before 10:15am,
it returns ``EntryDecision(ok=False, reason="NO TRADE TODAY")``.

Four conditions (spec §4):
    1. Past 9:35am                    — first 5-min candle has closed
    2. 1-min volume > 1.5× average    — move has participation
    3. ATM spread < 10% of mid        — scalp-friendly spread
    4. OI skew unchanged              — setup hasn't reversed since scoring

Non-negotiable rules enforced here (spec §10):
    - Maximum wait window: 9:30am – 10:15am EST
    - One trade per day maximum (caller must ensure scanner shuts down
      after this module returns ok=True)
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

from .webull_client import WebullClient, today_expiry
from .ticker_picker import _find_atm_strike

logger = logging.getLogger(__name__)

# Eastern time (EST = UTC-5; EDT = UTC-4 in summer — adjust cron in CI)
EASTERN = timezone(timedelta(hours=-5))

# Polling interval (seconds)
POLL_INTERVAL = 60

# Entry window (EST)
ENTRY_OPEN_HOUR, ENTRY_OPEN_MINUTE = 9, 30
ENTRY_CLOSE_HOUR, ENTRY_CLOSE_MINUTE = 10, 15

# First 5-min candle closes at 9:35am; do not fire before then
MIN_ENTRY_HOUR, MIN_ENTRY_MINUTE = 9, 35

# Volume confirmation threshold (spec §4)
VOLUME_MULTIPLIER = 1.5

# ATM spread gate (spec §4 & §10)
MAX_SPREAD_PCT = 10.0  # percent of mid-price


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EntryDecision:
    ok: bool
    reason: str = ""            # human-readable for Discord log
    spot_price: float = 0.0
    entry_time: Optional[datetime] = None
    condition_log: list[dict] = None   # one row per poll for debugging

    def __post_init__(self):
        if self.condition_log is None:
            self.condition_log = []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def wait_for_entry(
    client: WebullClient,
    ticker: str,
    initial_skew_direction: str,   # "CALL-HEAVY" | "PUT-HEAVY" from ticker_picker
    verbose: bool = True,
) -> EntryDecision:
    """
    Poll every 60 seconds until all four entry conditions are met or the
    window expires (10:15am EST).

    Parameters
    ----------
    client               : authenticated WebullClient
    ticker               : the winning ticker from ticker_picker
    initial_skew_direction : the OI skew direction recorded at scan time;
                             condition 4 checks this hasn't reversed.
    verbose              : log each poll's condition status to logger.info

    Returns
    -------
    EntryDecision with ok=True if entry is authorised, ok=False if window expired.
    """
    expiry = today_expiry()
    decision = EntryDecision(ok=False)
    last_fail_reason = "unknown"

    logger.info(
        "Entry timer started for %s — waiting until 9:35am, window closes 10:15am ET",
        ticker,
    )

    while True:
        now_et = _now_eastern()

        # ── Hard window close ──────────────────────────────────────────────
        window_close = now_et.replace(
            hour=ENTRY_CLOSE_HOUR,
            minute=ENTRY_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )
        if now_et >= window_close:
            decision.ok = False
            decision.reason = (
                f"NO TRADE TODAY — entry conditions never met within window. "
                f"Last block: {last_fail_reason}"
            )
            logger.info("Entry window expired at %s", _fmt_time(now_et))
            return decision

        # ── Condition 1: past 9:35am ──────────────────────────────────────
        earliest_entry = now_et.replace(
            hour=MIN_ENTRY_HOUR,
            minute=MIN_ENTRY_MINUTE,
            second=0,
            microsecond=0,
        )
        if now_et < earliest_entry:
            wait_secs = (earliest_entry - now_et).total_seconds()
            logger.info(
                "Waiting for 9:35am candle close — sleeping %.0fs", wait_secs
            )
            time.sleep(min(wait_secs, POLL_INTERVAL))
            continue

        # ── Fetch fresh data ───────────────────────────────────────────────
        try:
            snapshot = client.get_ticker_snapshot(ticker)
            chain = client.get_options_chain(ticker, expiry)
            candles = client.get_minute_candles(ticker, count=5)
        except Exception as exc:
            logger.warning("Data fetch error during entry timer: %s — retrying", exc)
            time.sleep(POLL_INTERVAL)
            continue

        spot = snapshot["last"]
        atm_strike = _find_atm_strike(chain, spot)

        # ── Condition 2: volume confirmation ──────────────────────────────
        vol_ok, vol_ratio = _check_volume(snapshot, candles)

        # ── Condition 3: ATM spread tightness ─────────────────────────────
        spread_ok, spread_pct = _check_atm_spread(chain, atm_strike)

        # ── Condition 4: OI skew unchanged ────────────────────────────────
        skew_ok, current_skew = _check_skew_unchanged(chain, spot, initial_skew_direction)

        # ── Log this poll ──────────────────────────────────────────────────
        poll_record = {
            "time": _fmt_time(now_et),
            "spot": spot,
            "vol_ok": vol_ok,
            "vol_ratio": round(vol_ratio, 2),
            "spread_ok": spread_ok,
            "spread_pct": round(spread_pct, 2),
            "skew_ok": skew_ok,
            "current_skew": current_skew,
        }
        decision.condition_log.append(poll_record)

        if verbose:
            logger.info(
                "[%s] spot=%.2f | vol_ratio=%.2fx (%s) | spread=%.1f%% (%s) | skew=%s (%s)",
                _fmt_time(now_et),
                spot,
                vol_ratio,
                "✓" if vol_ok else "✗",
                spread_pct,
                "✓" if spread_ok else "✗",
                current_skew,
                "✓" if skew_ok else "✗",
            )

        # ── All four conditions met? ───────────────────────────────────────
        if vol_ok and spread_ok and skew_ok:
            decision.ok = True
            decision.spot_price = spot
            decision.entry_time = now_et
            decision.reason = (
                f"Entry authorised at {_fmt_time(now_et)} ET | "
                f"vol_ratio={vol_ratio:.2f}x | spread={spread_pct:.1f}% | skew={current_skew}"
            )
            logger.info("ALL ENTRY CONDITIONS MET — %s", decision.reason)
            return decision

        # Track what blocked us for the NO TRADE log
        blocks = []
        if not vol_ok:
            blocks.append(f"vol_ratio={vol_ratio:.2f}x < {VOLUME_MULTIPLIER}x")
        if not spread_ok:
            blocks.append(f"ATM spread too wide ({spread_pct:.1f}% of mid)")
        if not skew_ok:
            blocks.append(f"OI skew reversed (was {initial_skew_direction}, now {current_skew})")
        last_fail_reason = "; ".join(blocks)

        logger.info("Conditions not met — sleeping %ds. Block: %s", POLL_INTERVAL, last_fail_reason)
        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Individual condition checks
# ---------------------------------------------------------------------------

def _check_volume(snapshot: dict, candles: list[dict]) -> tuple[bool, float]:
    """
    Condition 2: current 1-min volume > 1.5× average 1-min volume.

    We estimate average 1-min volume from the earlier candles in our
    window (excluding the most recent, which may still be forming).
    Falls back to avgVolume20D / 390 if candle history is too short.
    """
    recent_candles = candles[:-1] if len(candles) > 1 else candles
    if len(recent_candles) >= 2:
        avg_1min_vol = sum(c["volume"] for c in recent_candles) / len(recent_candles)
    else:
        # Fallback: 20-day avg / 390 session minutes
        avg_20d = snapshot.get("avgVolume20D") or 0
        avg_1min_vol = avg_20d / 390 if avg_20d else 0

    if avg_1min_vol <= 0:
        return False, 0.0

    current_vol = candles[-1]["volume"] if candles else 0
    ratio = current_vol / avg_1min_vol
    return ratio >= VOLUME_MULTIPLIER, ratio


def _check_atm_spread(chain: list[dict], atm_strike: float) -> tuple[bool, float]:
    """
    Condition 3: ATM bid/ask spread < 10% of mid.

    Checks both CALL and PUT at the ATM strike; requires BOTH to be tight
    (we don't yet know direction, so we need both sides to be tradeable).
    """
    if not chain or atm_strike == 0.0:
        return False, 100.0

    atm_contracts = [c for c in chain if c["strike"] == atm_strike]
    if not atm_contracts:
        return False, 100.0

    spread_pcts = []
    for c in atm_contracts:
        mid = c.get("mid", 0.0)
        if mid <= 0:
            spread_pcts.append(100.0)
            continue
        spread = c["ask"] - c["bid"]
        spread_pcts.append(spread / mid * 100)

    if not spread_pcts:
        return False, 100.0

    max_spread = max(spread_pcts)   # use the wider side as the gate
    return max_spread < MAX_SPREAD_PCT, max_spread


def _check_skew_unchanged(
    chain: list[dict], spot: float, initial_skew: str
) -> tuple[bool, str]:
    """
    Condition 4: the OI skew direction at current market data matches the
    skew direction measured during ticker scoring.

    "EVEN" is treated as inconclusive — if the initial skew was directional
    but has since become even, we return False (setup has weakened).
    If the initial skew was "EVEN", we accept any state (no directional bias
    was required).
    """
    if not chain or spot <= 0:
        return False, "UNKNOWN"

    strikes = sorted({c["strike"] for c in chain})
    if not strikes:
        return False, "UNKNOWN"

    step = min(
        (strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1)),
        default=1.0,
    )
    window = step * 5

    nearby = [c for c in chain if abs(c["strike"] - spot) <= window]
    call_oi = sum(c["open_interest"] for c in nearby if c["type"] == "CALL")
    put_oi = sum(c["open_interest"] for c in nearby if c["type"] == "PUT")

    if call_oi == 0 and put_oi == 0:
        current_skew = "EVEN"
    elif call_oi >= put_oi:
        ratio = call_oi / put_oi if put_oi else float("inf")
        current_skew = "CALL-HEAVY" if ratio >= 1.2 else "EVEN"
    else:
        ratio = put_oi / call_oi if call_oi else float("inf")
        current_skew = "PUT-HEAVY" if ratio >= 1.2 else "EVEN"

    if initial_skew == "EVEN":
        # No directional requirement — always passes
        return True, current_skew

    return current_skew == initial_skew, current_skew


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now_eastern() -> datetime:
    return datetime.now(tz=EASTERN)


def _fmt_time(dt: datetime) -> str:
    return dt.strftime("%I:%M:%S%p").lstrip("0")
