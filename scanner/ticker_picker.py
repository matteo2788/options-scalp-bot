"""
ticker_picker.py
----------------
Scores TSLA, NVDA, and SPY on five factors (0–100 points total) and
returns the single best ticker to trade today.

Scoring factors (spec §3.1):
    1. Relative Volume          30 pts
    2. OI Concentration near ATM 25 pts
    3. Put/Call OI Skew         20 pts
    4. Price Momentum (1-min)   15 pts
    5. ATM Option Liquidity     10 pts

Tiebreaker (spec §3): TSLA > NVDA > SPY
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

from .webull_client import WebullClient, today_expiry

logger = logging.getLogger(__name__)

TICKERS = ["TSLA", "NVDA", "SPY"]

# Tiebreaker priority: lower index = higher priority
TIEBREAKER = {"TSLA": 0, "NVDA": 1, "SPY": 2}

# The number of one-minute candles to pull for momentum scoring
CANDLE_COUNT = 10

# Eastern time offset (EST). The scoring engine runs at 9:25am ET so we
# use EST = UTC-5. During EDT (summer) this is UTC-4; the GitHub Actions
# cron is expressed in UTC and accounts for this separately.
EASTERN = timezone(timedelta(hours=-5))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    """Full scoring detail for one ticker — used in Discord alert payload."""
    ticker: str
    total: int = 0

    rel_volume_score: int = 0
    oi_concentration_score: int = 0
    put_call_skew_score: int = 0
    momentum_score: int = 0
    atm_liquidity_score: int = 0

    # Raw data for logging and the Discord alert "reason" line
    rel_volume_ratio: float = 0.0
    atm_strike: float = 0.0
    spot_price: float = 0.0
    skew_direction: str = ""        # "CALL-HEAVY" | "PUT-HEAVY" | "EVEN"
    skew_ratio: float = 0.0
    momentum_candles: list = field(default_factory=list)
    atm_spread_pct: float = 0.0
    vwap: float = 0.0

    def compute_total(self) -> None:
        self.total = (
            self.rel_volume_score
            + self.oi_concentration_score
            + self.put_call_skew_score
            + self.momentum_score
            + self.atm_liquidity_score
        )

    def reason_summary(self) -> str:
        """Short human-readable reason string for the Discord alert."""
        parts = []
        if self.skew_direction == "CALL-HEAVY":
            parts.append("OI call-heavy above")
        elif self.skew_direction == "PUT-HEAVY":
            parts.append("OI put-heavy below")

        candles = self.momentum_candles[-3:] if len(self.momentum_candles) >= 3 else []
        if all(c["close"] > c["open"] for c in candles):
            parts.append("price above VWAP" if self.spot_price >= self.vwap else "momentum bullish")
        elif all(c["close"] < c["open"] for c in candles):
            parts.append("price below VWAP" if self.spot_price < self.vwap else "momentum bearish")

        return " + ".join(parts) if parts else "composite score winner"


@dataclass
class PickerResult:
    winner: str
    score: int
    breakdown: ScoreBreakdown
    all_scores: dict[str, ScoreBreakdown] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def pick_ticker(client: Optional[WebullClient] = None) -> PickerResult:
    """
    Score all three tickers and return the winner.

    Parameters
    ----------
    client : WebullClient (optional)
        If not provided, a new instance is created.  Pass one in to reuse
        an existing authenticated session.

    Returns
    -------
    PickerResult with .winner, .score, and full .breakdown
    """
    if client is None:
        client = WebullClient()

    expiry = today_expiry()
    logger.info("Ticker picker running — expiry date: %s", expiry)

    all_scores: dict[str, ScoreBreakdown] = {}

    for ticker in TICKERS:
        logger.info("Scoring %s …", ticker)
        try:
            breakdown = _score_ticker(client, ticker, expiry)
            all_scores[ticker] = breakdown
            logger.info(
                "%s total=%d  (rel_vol=%d, oi_conc=%d, skew=%d, mom=%d, liq=%d)",
                ticker,
                breakdown.total,
                breakdown.rel_volume_score,
                breakdown.oi_concentration_score,
                breakdown.put_call_skew_score,
                breakdown.momentum_score,
                breakdown.atm_liquidity_score,
            )
        except Exception as exc:
            # If one ticker's data is unavailable, score it 0 and continue
            logger.error("Failed to score %s: %s — assigning 0 points", ticker, exc)
            all_scores[ticker] = ScoreBreakdown(ticker=ticker, total=0)

    winner = _select_winner(all_scores)
    logger.info("WINNER: %s  score=%d/100", winner, all_scores[winner].total)

    return PickerResult(
        winner=winner,
        score=all_scores[winner].total,
        breakdown=all_scores[winner],
        all_scores=all_scores,
    )


# ---------------------------------------------------------------------------
# Per-ticker scoring
# ---------------------------------------------------------------------------

def _score_ticker(client: WebullClient, ticker: str, expiry: str) -> ScoreBreakdown:
    """Fetch data and compute all five scoring factors for one ticker."""
    bd = ScoreBreakdown(ticker=ticker)

    # --- Fetch data -------------------------------------------------------
    snapshot = client.get_ticker_snapshot(ticker)
    chain = client.get_options_chain(ticker, expiry)
    candles = client.get_minute_candles(ticker, count=CANDLE_COUNT)

    spot = snapshot["last"]
    bd.spot_price = spot
    bd.vwap = snapshot.get("vwap") or spot

    if not chain:
        logger.warning("%s: empty options chain — all option scores = 0", ticker)
    if not candles:
        logger.warning("%s: no candle data — momentum score = 0", ticker)

    # --- Factor 1: Relative Volume ----------------------------------------
    bd.rel_volume_score, bd.rel_volume_ratio = _score_relative_volume(snapshot)

    # --- Factor 2: OI Concentration near ATM ------------------------------
    bd.oi_concentration_score, bd.atm_strike = _score_oi_concentration(chain, spot)

    # --- Factor 3: Put/Call OI Skew ---------------------------------------
    bd.put_call_skew_score, bd.skew_direction, bd.skew_ratio = _score_put_call_skew(
        chain, spot
    )

    # --- Factor 4: Price Momentum -----------------------------------------
    bd.momentum_score, bd.momentum_candles = _score_momentum(candles)

    # --- Factor 5: ATM Option Liquidity -----------------------------------
    bd.atm_liquidity_score, bd.atm_spread_pct = _score_atm_liquidity(
        chain, bd.atm_strike
    )

    bd.compute_total()
    return bd


# ---------------------------------------------------------------------------
# Factor 1 — Relative Volume (30 pts)
# ---------------------------------------------------------------------------

def _score_relative_volume(snapshot: dict) -> tuple[int, float]:
    """
    Compare intraday volume-so-far to the expected volume at this time of
    day, calculated from the 20-day average daily volume.

    Time-normalised expected volume: avg_20d × (minutes_elapsed / 390)
    390 = total regular-session minutes (9:30am – 4:00pm)

    Thresholds (spec §3.2):
        > 2.0x → 30 pts
        > 1.5x → 20 pts
        > 1.2x → 10 pts
        < 1.2x →  0 pts
    """
    volume = snapshot.get("volume") or 0
    avg_volume_20d = snapshot.get("avgVolume20D") or 0

    if avg_volume_20d <= 0:
        logger.warning("avgVolume20D unavailable — relative volume score = 0")
        return 0, 0.0

    # Minutes elapsed since 9:30am ET
    now_et = datetime.now(tz=EASTERN)
    session_start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    elapsed_minutes = max((now_et - session_start).total_seconds() / 60, 1)

    # Expected volume at this point in the session
    expected_volume = avg_volume_20d * (elapsed_minutes / 390.0)

    if expected_volume <= 0:
        return 0, 0.0

    ratio = volume / expected_volume

    if ratio > 2.0:
        score = 30
    elif ratio > 1.5:
        score = 20
    elif ratio > 1.2:
        score = 10
    else:
        score = 0

    return score, round(ratio, 2)


# ---------------------------------------------------------------------------
# Factor 2 — OI Concentration near ATM (25 pts)
# ---------------------------------------------------------------------------

def _score_oi_concentration(chain: list[dict], spot: float) -> tuple[int, float]:
    """
    Total OI within ±2 strikes of spot / total chain OI.

    Spec §3.2:
        top 2 strikes > 40% of total chain OI → 25 pts
        scaled proportionally below 40%

    Returns (score, atm_strike)
    """
    if not chain or spot <= 0:
        return 0, 0.0

    atm_strike = _find_atm_strike(chain, spot)

    strikes = sorted({c["strike"] for c in chain})
    if not strikes:
        return 0, atm_strike

    # Determine strike step (smallest gap between adjacent strikes)
    step = min(
        (strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1)),
        default=1.0,
    )
    near_atm_window = step * 2  # ±2 strikes from ATM

    total_oi = sum(c["open_interest"] for c in chain)
    if total_oi == 0:
        return 0, atm_strike

    near_atm_oi = sum(
        c["open_interest"]
        for c in chain
        if abs(c["strike"] - atm_strike) <= near_atm_window
    )

    concentration_pct = near_atm_oi / total_oi

    # 25 pts at 40%; scale linearly, floor at 0
    score = min(25, int(round(25 * (concentration_pct / 0.40))))
    score = max(0, score)

    return score, atm_strike


# ---------------------------------------------------------------------------
# Factor 3 — Put/Call OI Skew (20 pts)
# ---------------------------------------------------------------------------

def _score_put_call_skew(
    chain: list[dict], spot: float
) -> tuple[int, str, float]:
    """
    Call OI / Put OI within ±5 strikes of spot.

    Spec §3.2:
        ratio > 1.5:1 in either direction → 20 pts
        ratio 1.2–1.5:1                  → 10 pts
        near even (<1.2)                 →  0 pts

    Returns (score, direction_string, ratio)
    direction_string: "CALL-HEAVY" | "PUT-HEAVY" | "EVEN"
    """
    if not chain or spot <= 0:
        return 0, "EVEN", 1.0

    strikes = sorted({c["strike"] for c in chain})
    if not strikes:
        return 0, "EVEN", 1.0

    step = min(
        (strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1)),
        default=1.0,
    )
    window = step * 5

    nearby = [c for c in chain if abs(c["strike"] - spot) <= window]

    call_oi = sum(c["open_interest"] for c in nearby if c["type"] == "CALL")
    put_oi = sum(c["open_interest"] for c in nearby if c["type"] == "PUT")

    if put_oi == 0 and call_oi == 0:
        return 0, "EVEN", 1.0

    if put_oi == 0:
        ratio = call_oi  # extremely call-heavy
        direction = "CALL-HEAVY"
    elif call_oi == 0:
        ratio = put_oi   # extremely put-heavy
        direction = "PUT-HEAVY"
    else:
        if call_oi >= put_oi:
            ratio = call_oi / put_oi
            direction = "CALL-HEAVY"
        else:
            ratio = put_oi / call_oi
            direction = "PUT-HEAVY"

    if ratio >= 1.5:
        score = 20
    elif ratio >= 1.2:
        score = 10
    else:
        score = 0
        direction = "EVEN"

    return score, direction, round(ratio, 2)


# ---------------------------------------------------------------------------
# Factor 4 — Price Momentum (15 pts)
# ---------------------------------------------------------------------------

def _score_momentum(candles: list[dict]) -> tuple[int, list[dict]]:
    """
    Evaluate the last 3 one-minute candles for directional momentum.

    Spec §3.2:
        All 3 same direction + increasing body size → 15 pts
        2/3 same direction                          → 10 pts
        Mixed                                       →  0 pts

    A candle is "bullish" if close > open, "bearish" if close < open.
    "Increasing body size" means each candle's body (|close-open|) is
    larger than the previous candle's body.

    Returns (score, last_3_candles)
    """
    if len(candles) < 3:
        logger.warning("Not enough candles for momentum scoring (%d available)", len(candles))
        return 0, candles

    last3 = candles[-3:]  # [oldest, middle, newest]

    directions = []
    bodies = []
    for c in last3:
        body = c["close"] - c["open"]
        directions.append("up" if body > 0 else "down" if body < 0 else "flat")
        bodies.append(abs(body))

    # Remove flat candles — treat them as ambiguous
    non_flat = [d for d in directions if d != "flat"]
    up_count = non_flat.count("up")
    down_count = non_flat.count("down")

    same_direction = max(up_count, down_count) >= 2
    all_same = max(up_count, down_count) == 3

    increasing_size = all_same and (bodies[0] <= bodies[1] <= bodies[2])

    if all_same and increasing_size:
        score = 15
    elif same_direction:
        score = 10
    else:
        score = 0

    return score, last3


# ---------------------------------------------------------------------------
# Factor 5 — ATM Option Liquidity (10 pts)
# ---------------------------------------------------------------------------

def _score_atm_liquidity(chain: list[dict], atm_strike: float) -> tuple[int, float]:
    """
    Bid/ask spread of the ATM contract as % of mid-price.

    Spec §3.2:
        spread < 5% of mid  → 10 pts
        spread 5–10% of mid →  5 pts
        spread > 10% of mid →  0 pts

    We look at both CALL and PUT ATM contracts and use the tighter one
    (since the bot may trade either direction).

    Returns (score, spread_pct_of_better_side)
    """
    if not chain or atm_strike == 0.0:
        return 0, 100.0

    atm_contracts = [c for c in chain if c["strike"] == atm_strike]
    if not atm_contracts:
        return 0, 100.0

    spread_pcts = []
    for c in atm_contracts:
        mid = c.get("mid", 0.0)
        if mid <= 0:
            continue
        spread = c["ask"] - c["bid"]
        spread_pcts.append(spread / mid * 100)

    if not spread_pcts:
        return 0, 100.0

    best_spread_pct = min(spread_pcts)

    if best_spread_pct < 5.0:
        score = 10
    elif best_spread_pct <= 10.0:
        score = 5
    else:
        score = 0

    return score, round(best_spread_pct, 2)


# ---------------------------------------------------------------------------
# Winner selection with tiebreaker
# ---------------------------------------------------------------------------

def _select_winner(all_scores: dict[str, ScoreBreakdown]) -> str:
    """
    Return the ticker with the highest score.
    Tiebreaker: TSLA > NVDA > SPY (spec §3, higher volatility preferred).
    """
    return max(
        TICKERS,
        key=lambda t: (all_scores[t].total, -TIEBREAKER[t]),
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _find_atm_strike(chain: list[dict], spot: float) -> float:
    """Return the strike in the chain closest to the current spot price."""
    strikes = sorted({c["strike"] for c in chain})
    if not strikes:
        return 0.0
    return min(strikes, key=lambda s: abs(s - spot))
