"""
strike_selector.py
------------------
Selects the ATM strike on the 0DTE options chain and computes TP/SL
levels — spec §6.

Strike Selection Logic (spec §6):
    1. Pull full 0DTE chain (already done by caller — passed in)
    2. Get current spot price
    3. Find strike closest to spot (ATM)
    4. Check: bid/ask spread of this strike < 10% of mid?
          YES → use this strike
          NO  → try next closest strike (one step ITM)
    5. Record mid-price as official entry price
    6. Calculate TP and SL:
          TP = entry_price × 1.25   (+25%)
          SL = entry_price × 0.85   (-15%)

Non-negotiable rules enforced here (spec §10):
    - Spread gate: > 10% of mid at any point → wait or skip
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Spread gate threshold (spec §6 and §10)
MAX_ATM_SPREAD_PCT = 10.0

# TP / SL multipliers (spec §6)
TP_MULTIPLIER = 1.25   # +25%
SL_MULTIPLIER = 0.85   # -15%


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class StrikeSelection:
    """
    Everything the position monitor and Discord alert need.

    Fields
    ------
    ok              : False if no eligible strike found (spread gate failed)
    ticker          : underlying symbol
    direction       : "CALL" | "PUT"
    strike          : selected strike price
    option_symbol   : OCC symbol string (e.g. "TSLA240101C00300000")
    entry_price     : mid-price at the time of selection
    take_profit     : entry_price × 1.25
    stop_loss       : entry_price × 0.85
    bid             : bid at selection time
    ask             : ask at selection time
    spread_pct      : (ask - bid) / mid × 100
    reason          : human-readable note
    """
    ok: bool
    ticker: str
    direction: str          # "CALL" | "PUT"
    strike: float = 0.0
    option_symbol: str = ""
    entry_price: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread_pct: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def select_strike(
    chain: list[dict],
    spot: float,
    ticker: str,
    direction: str,
) -> StrikeSelection:
    """
    Find the best tradeable ATM strike for the given direction.

    Parameters
    ----------
    chain     : full 0DTE options chain from webull_client.get_options_chain()
    spot      : current spot price of the underlying
    ticker    : underlying symbol ("TSLA", "NVDA", "SPY")
    direction : "CALL" or "PUT"

    Returns
    -------
    StrikeSelection with ok=True if a valid strike was found, ok=False otherwise.
    """
    if direction not in ("CALL", "PUT"):
        raise ValueError(f"direction must be 'CALL' or 'PUT', got '{direction}'")

    # Filter to the correct contract type
    filtered = [c for c in chain if c["type"] == direction]
    if not filtered:
        return StrikeSelection(
            ok=False,
            ticker=ticker,
            direction=direction,
            reason=f"No {direction} contracts found in chain",
        )

    # Sort all available strikes by proximity to spot (ascending distance)
    unique_strikes = sorted(
        {c["strike"] for c in filtered},
        key=lambda s: abs(s - spot),
    )

    # Try ATM first, then fall back to ITM (one step closer to in-the-money)
    for strike in unique_strikes[:2]:
        contract = _get_contract(filtered, strike)
        if contract is None:
            continue

        bid = contract["bid"]
        ask = contract["ask"]
        mid = contract.get("mid") or ((bid + ask) / 2 if (bid and ask) else 0.0)

        if mid <= 0:
            logger.warning(
                "Strike %.2f has zero mid-price (bid=%.2f, ask=%.2f) — skipping",
                strike, bid, ask,
            )
            continue

        spread = ask - bid
        spread_pct = (spread / mid) * 100

        if spread_pct >= MAX_ATM_SPREAD_PCT:
            logger.info(
                "Strike %.2f spread too wide: %.1f%% (limit %.1f%%) — trying next",
                strike, spread_pct, MAX_ATM_SPREAD_PCT,
            )
            continue

        # This strike passes the spread gate
        entry_price = mid
        take_profit = round(entry_price * TP_MULTIPLIER, 2)
        stop_loss = round(entry_price * SL_MULTIPLIER, 2)

        option_symbol = (
            contract.get("symbol")
            or _build_occ_symbol(ticker, strike, direction)
        )

        logger.info(
            "Selected %s %s $%.0f | entry=%.2f | TP=%.2f | SL=%.2f | spread=%.1f%%",
            ticker, direction, strike, entry_price, take_profit, stop_loss, spread_pct,
        )

        return StrikeSelection(
            ok=True,
            ticker=ticker,
            direction=direction,
            strike=strike,
            option_symbol=option_symbol,
            entry_price=round(entry_price, 2),
            take_profit=take_profit,
            stop_loss=stop_loss,
            bid=bid,
            ask=ask,
            spread_pct=round(spread_pct, 2),
            reason=f"ATM strike selected | spread={spread_pct:.1f}% of mid",
        )

    # No acceptable strike found
    tried = unique_strikes[:2]
    return StrikeSelection(
        ok=False,
        ticker=ticker,
        direction=direction,
        reason=(
            f"Spread gate failed for all tried strikes "
            f"({[f'{s:.0f}' for s in tried]}) — "
            f"spread > {MAX_ATM_SPREAD_PCT}% of mid"
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_contract(contracts: list[dict], strike: float) -> Optional[dict]:
    """Return the contract dict for a specific strike, or None."""
    matches = [c for c in contracts if c["strike"] == strike]
    return matches[0] if matches else None


def _build_occ_symbol(ticker: str, strike: float, direction: str) -> str:
    """
    Build a standard OCC option symbol from components.

    Format: TICKER + YYMMDD + C/P + 8-digit strike (×1000, zero-padded)
    Example: TSLA240101C00320000 = TSLA, 2024-01-01, Call, $320 strike

    Used as a fallback when the Webull API doesn't return the symbol string.
    The expiry date embedded here is 'today' (0DTE).
    """
    from datetime import datetime, timezone, timedelta
    eastern = timezone(timedelta(hours=-5))
    today = datetime.now(tz=eastern).strftime("%y%m%d")
    cp = "C" if direction == "CALL" else "P"
    strike_int = int(strike * 1000)
    return f"{ticker}{today}{cp}{strike_int:08d}"
