"""
direction_engine.py
-------------------
Determines trade direction (CALL or PUT) using a two-signal confirmation
model — spec §5.

BOTH signals must agree. If they conflict or either is inconclusive,
there is no trade.  This is the single most important discipline rule
in the system.

Signal 1 — OI Skew (spec §5.1):
    Heavy CALL OI above price → call wall repels price → PUT
    Heavy PUT OI below price  → put wall supports price → CALL
    Even OI                   → SIGNAL INCONCLUSIVE

Signal 2 — Price Momentum (spec §5.2):
    Price above VWAP + last 3 candles closing higher → CALL
    Price below VWAP + last 3 candles closing lower  → PUT
    Mixed / VWAP crossing                            → SIGNAL INCONCLUSIVE
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .webull_client import WebullClient, today_expiry
from .ticker_picker import _find_atm_strike

logger = logging.getLogger(__name__)

# OI skew ratio threshold — must exceed this to be "clear" (not EVEN)
SKEW_RATIO_THRESHOLD = 1.2

# Minimum candles required for momentum signal
MIN_CANDLES = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DirectionResult:
    """
    Outcome of the two-signal confirmation.

    direction : "CALL" | "PUT" | None
        None means no trade (signals conflict or inconclusive).
    reason    : human-readable explanation for Discord alert / log.
    signal_oi : "CALL" | "PUT" | "INCONCLUSIVE"
    signal_momentum : "CALL" | "PUT" | "INCONCLUSIVE"
    """
    direction: Optional[str]
    reason: str
    signal_oi: str
    signal_momentum: str

    @property
    def tradeable(self) -> bool:
        return self.direction is not None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def determine_direction(
    client: WebullClient,
    ticker: str,
    spot: float,
    vwap: float,
    candles: list[dict],
) -> DirectionResult:
    """
    Run both signals and return the agreed direction (or None).

    Parameters
    ----------
    client  : authenticated WebullClient
    ticker  : the winning ticker
    spot    : current spot price (from the entry-time snapshot)
    vwap    : today's VWAP (from the entry-time snapshot)
    candles : list of 1-min OHLCV dicts, oldest→newest (from entry_timer data)

    Returns
    -------
    DirectionResult — caller must check ``.tradeable`` before proceeding.
    """
    expiry = today_expiry()

    # ── Signal 1: OI Skew ─────────────────────────────────────────────────
    try:
        chain = client.get_options_chain(ticker, expiry)
        signal_oi, oi_reason = _signal_oi_skew(chain, spot)
    except Exception as exc:
        logger.error("OI skew signal failed: %s", exc)
        signal_oi = "INCONCLUSIVE"
        oi_reason = f"API error: {exc}"

    logger.info("Signal 1 (OI Skew): %s — %s", signal_oi, oi_reason)

    # ── Signal 2: Price Momentum ───────────────────────────────────────────
    signal_momentum, mom_reason = _signal_price_momentum(spot, vwap, candles)
    logger.info("Signal 2 (Momentum): %s — %s", signal_momentum, mom_reason)

    # ── Two-signal confirmation ────────────────────────────────────────────
    if signal_oi == "INCONCLUSIVE" or signal_momentum == "INCONCLUSIVE":
        inconclusive_signals = []
        if signal_oi == "INCONCLUSIVE":
            inconclusive_signals.append(f"OI Skew ({oi_reason})")
        if signal_momentum == "INCONCLUSIVE":
            inconclusive_signals.append(f"Momentum ({mom_reason})")

        reason = "NO TRADE — inconclusive signal(s): " + "; ".join(inconclusive_signals)
        logger.info(reason)
        return DirectionResult(
            direction=None,
            reason=reason,
            signal_oi=signal_oi,
            signal_momentum=signal_momentum,
        )

    if signal_oi != signal_momentum:
        reason = (
            f"NO TRADE — signals conflict: "
            f"OI says {signal_oi} ({oi_reason}), "
            f"Momentum says {signal_momentum} ({mom_reason})"
        )
        logger.info(reason)
        return DirectionResult(
            direction=None,
            reason=reason,
            signal_oi=signal_oi,
            signal_momentum=signal_momentum,
        )

    # Both agree — TRADE
    direction = signal_oi  # "CALL" or "PUT"
    reason = (
        f"OI {oi_reason} + {mom_reason}"
    )
    logger.info("DIRECTION CONFIRMED: %s — %s", direction, reason)

    return DirectionResult(
        direction=direction,
        reason=reason,
        signal_oi=signal_oi,
        signal_momentum=signal_momentum,
    )


# ---------------------------------------------------------------------------
# Signal 1 — OI Skew
# ---------------------------------------------------------------------------

def _signal_oi_skew(chain: list[dict], spot: float) -> tuple[str, str]:
    """
    Spec §5.1:
        Heavy CALL OI *above* current price → market makers will defend call
        wall → price is repelled down → direction: PUT
        Heavy PUT OI *below* current price  → put wall is support → direction: CALL
        Even OI                             → INCONCLUSIVE

    We split the chain into 'above spot' and 'below spot' buckets,
    then compare their OI concentrations to determine which wall is heavier.
    """
    if not chain or spot <= 0:
        return "INCONCLUSIVE", "empty chain"

    call_oi_above = sum(
        c["open_interest"]
        for c in chain
        if c["type"] == "CALL" and c["strike"] > spot
    )
    put_oi_below = sum(
        c["open_interest"]
        for c in chain
        if c["type"] == "PUT" and c["strike"] < spot
    )

    # Also gather total OI for context
    total_oi = sum(c["open_interest"] for c in chain)
    if total_oi == 0:
        return "INCONCLUSIVE", "zero total OI"

    call_above_pct = call_oi_above / total_oi
    put_below_pct = put_oi_below / total_oi

    # Determine which wall is heavier and compute the dominance ratio
    if call_above_pct == 0 and put_below_pct == 0:
        return "INCONCLUSIVE", "no directional OI concentration"

    if call_above_pct >= put_below_pct:
        if put_below_pct == 0:
            ratio = float("inf")
        else:
            ratio = call_above_pct / put_below_pct
        heavier = "CALL"
    else:
        if call_above_pct == 0:
            ratio = float("inf")
        else:
            ratio = put_below_pct / call_above_pct
        heavier = "PUT"

    if ratio < SKEW_RATIO_THRESHOLD:
        return (
            "INCONCLUSIVE",
            f"OI evenly distributed (call-above={call_above_pct:.1%}, "
            f"put-below={put_below_pct:.1%})",
        )

    if heavier == "CALL":
        # Call wall above → price repelled → PUT trade
        direction = "PUT"
        reason = (
            f"OI call-heavy above spot "
            f"(call-above={call_above_pct:.1%} vs put-below={put_below_pct:.1%})"
        )
    else:
        # Put wall below → support → CALL trade
        direction = "CALL"
        reason = (
            f"OI put-heavy below spot "
            f"(put-below={put_below_pct:.1%} vs call-above={call_above_pct:.1%})"
        )

    return direction, reason


# ---------------------------------------------------------------------------
# Signal 2 — Price Momentum
# ---------------------------------------------------------------------------

def _signal_price_momentum(
    spot: float, vwap: float, candles: list[dict]
) -> tuple[str, str]:
    """
    Spec §5.2:
        Price above VWAP + last 3 1-min candles closing higher → CALL
        Price below VWAP + last 3 1-min candles closing lower  → PUT
        Mixed candles or price crossing VWAP repeatedly        → INCONCLUSIVE

    "Closing higher" means close > open (green candle).
    "Closing lower"  means close < open (red candle).

    Both the VWAP condition and the candle condition must agree.
    """
    if len(candles) < MIN_CANDLES:
        return (
            "INCONCLUSIVE",
            f"insufficient candle data ({len(candles)} < {MIN_CANDLES} required)",
        )

    last3 = candles[-3:]

    # Candle directions
    directions = []
    for c in last3:
        if c["close"] > c["open"]:
            directions.append("up")
        elif c["close"] < c["open"]:
            directions.append("down")
        else:
            directions.append("flat")

    up_count = directions.count("up")
    down_count = directions.count("down")

    all_up = up_count == 3
    all_down = down_count == 3
    majority_up = up_count >= 2
    majority_down = down_count >= 2

    # VWAP relationship
    above_vwap = spot > vwap
    below_vwap = spot < vwap

    # Both sub-signals must agree
    if all_up or majority_up:
        candle_signal = "CALL"
    elif all_down or majority_down:
        candle_signal = "PUT"
    else:
        candle_signal = "INCONCLUSIVE"

    if above_vwap:
        vwap_signal = "CALL"
        vwap_desc = f"price {spot:.2f} above VWAP {vwap:.2f}"
    elif below_vwap:
        vwap_signal = "PUT"
        vwap_desc = f"price {spot:.2f} below VWAP {vwap:.2f}"
    else:
        vwap_signal = "INCONCLUSIVE"
        vwap_desc = f"price {spot:.2f} at VWAP {vwap:.2f}"

    if candle_signal == "INCONCLUSIVE" or vwap_signal == "INCONCLUSIVE":
        return (
            "INCONCLUSIVE",
            f"mixed signals: candles={candle_signal} ({up_count} up / {down_count} down), "
            f"VWAP={vwap_signal} ({vwap_desc})",
        )

    if candle_signal != vwap_signal:
        return (
            "INCONCLUSIVE",
            f"conflicting: candles say {candle_signal} but {vwap_desc}",
        )

    # Both agree
    candle_desc = "all 3 candles" if (all_up or all_down) else "2/3 candles"
    direction_word = "higher" if candle_signal == "CALL" else "lower"
    reason = f"{vwap_desc} + {candle_desc} closing {direction_word}"

    return candle_signal, reason
