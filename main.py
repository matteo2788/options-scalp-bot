"""
scanner/main.py
---------------
Orchestrates all phases of the 0DTE options scalp bot.

This is the entry point run by scalp_scanner.yml at 9:25am ET, Mon–Fri.

Execution flow (spec §3–§6, §10):
    Phase 1 — Ticker Picker       : score TSLA/NVDA/SPY, pick winner
    Phase 2 — Smart Entry Timing  : poll 9:30–10:15am until 4 conditions met
    Phase 3 — Direction Decision  : confirm CALL or PUT (both signals must agree)
    Phase 4 — Strike Selection    : find ATM strike, apply spread gate
    Phase 5 — Alert + State Write : send Discord entry alert, write active_trade.json
    (Phase 6 — Position Monitor   : launched as a separate GH Actions workflow)

Non-negotiable rules (spec §10):
    - One trade per day maximum — scanner shuts down the moment a trade fires.
    - No trade is always valid — bot logs and exits cleanly if conditions aren't met.
    - Both signals must agree — OI skew + momentum required; conflict = no trade.
    - Spread gate enforced — > 10% spread = skip or wait.
    - Paper trading only — all orders go to Webull paper account DEN92WP9.
"""

import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure repo root is on the path when run as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scanner.webull_client import WebullClient, today_expiry
from scanner.ticker_picker import pick_ticker
from scanner.entry_timer import wait_for_entry
from scanner.direction_engine import determine_direction
from scanner.strike_selector import select_strike
from monitor.position_monitor import write_active_trade
from alerts.discord import (
    DiscordAlerter,
    build_entry_alert,
    build_no_trade_log,
    build_direction_conflict_log,
    build_spread_gate_log,
)

# ---------------------------------------------------------------------------
# Logging setup — GitHub Actions streams stdout/stderr so basicConfig is fine
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

EASTERN = timezone(timedelta(hours=-5))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Full scanner run.  Exits cleanly in all terminal states (trade fired,
    no-trade, any unexpected exception).  Never raises to the GH Actions
    runner — errors are logged and alerted, then the process exits 0 so
    the workflow step is marked complete rather than failed.
    """
    now_et = datetime.now(tz=EASTERN)
    logger.info(
        "═══ Options Scalp Bot — Scanner starting at %s ET ═══",
        now_et.strftime("%I:%M%p").lstrip("0"),
    )

    client = WebullClient()
    alerter = DiscordAlerter()

    # ──────────────────────────────────────────────────────────────────────
    # Phase 1: Ticker Picker
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Phase 1: Scoring tickers …")
    try:
        pick_result = pick_ticker(client=client)
    except Exception as exc:
        logger.exception("Ticker picker failed with unexpected error: %s", exc)
        alerter.send_log(
            f"⚠️ SCANNER ERROR — Phase 1 (Ticker Picker) failed: {exc}\n"
            f"Bot shutting down. Manual review required."
        )
        return

    winner = pick_result.winner
    score = pick_result.score
    breakdown = pick_result.breakdown
    initial_skew = breakdown.skew_direction

    logger.info(
        "Phase 1 complete: winner=%s  score=%d/100  skew=%s",
        winner, score, initial_skew,
    )

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2: Smart Entry Timing
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Phase 2: Waiting for entry conditions on %s …", winner)
    try:
        entry_decision = wait_for_entry(
            client=client,
            ticker=winner,
            initial_skew_direction=initial_skew,
            verbose=True,
        )
    except Exception as exc:
        logger.exception("Entry timer failed with unexpected error: %s", exc)
        alerter.send_log(
            f"⚠️ SCANNER ERROR — Phase 2 (Entry Timer) failed: {exc}\n"
            f"Bot shutting down."
        )
        return

    if not entry_decision.ok:
        # No trade today — window expired without all conditions being met
        reason = entry_decision.reason
        logger.info("Phase 2: NO TRADE — %s", reason)

        # Strip the "NO TRADE TODAY — " prefix for the Reason: field
        reason_short = reason.replace("NO TRADE TODAY — ", "").replace(
            "entry conditions never met within window. Last block: ", ""
        )
        msg = build_no_trade_log(winner, score, reason_short)
        alerter.send_log(msg)
        logger.info("No-trade log sent to Discord. Exiting cleanly.")
        return

    spot = entry_decision.spot_price
    entry_time_str = (
        entry_decision.entry_time.strftime("%I:%M%p").lstrip("0")
        if entry_decision.entry_time
        else "unknown"
    )
    logger.info(
        "Phase 2 complete: entry authorised at %s | spot=%.2f",
        entry_time_str, spot,
    )

    # Refresh VWAP from a live snapshot at entry time (more accurate than
    # the stale value from scoring 30+ minutes ago)
    try:
        entry_snapshot = client.get_ticker_snapshot(winner)
        vwap = entry_snapshot.get("vwap") or spot
    except Exception:
        vwap = breakdown.vwap or spot

    # Get fresh candles for the direction engine
    try:
        entry_candles = client.get_minute_candles(winner, count=10)
    except Exception:
        entry_candles = breakdown.momentum_candles

    # ──────────────────────────────────────────────────────────────────────
    # Phase 3: Direction Decision (CALL or PUT)
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Phase 3: Determining trade direction …")
    try:
        dir_result = determine_direction(
            client=client,
            ticker=winner,
            spot=spot,
            vwap=vwap,
            candles=entry_candles,
        )
    except Exception as exc:
        logger.exception("Direction engine failed: %s", exc)
        alerter.send_log(
            f"⚠️ SCANNER ERROR — Phase 3 (Direction Engine) failed: {exc}\n"
            f"Bot shutting down."
        )
        return

    if not dir_result.tradeable:
        # Signals conflict or inconclusive — no trade
        logger.info("Phase 3: NO TRADE — %s", dir_result.reason)
        msg = build_direction_conflict_log(winner, score, dir_result.reason)
        alerter.send_log(msg)
        logger.info("Direction conflict log sent to Discord. Exiting cleanly.")
        return

    direction = dir_result.direction
    logger.info("Phase 3 complete: direction=%s", direction)

    # ──────────────────────────────────────────────────────────────────────
    # Phase 4: Strike Selection
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Phase 4: Selecting ATM strike for %s %s …", winner, direction)
    expiry = today_expiry()
    try:
        chain = client.get_options_chain(winner, expiry)
        selection = select_strike(chain, spot, winner, direction)
    except Exception as exc:
        logger.exception("Strike selector failed: %s", exc)
        alerter.send_log(
            f"⚠️ SCANNER ERROR — Phase 4 (Strike Selector) failed: {exc}\n"
            f"Bot shutting down."
        )
        return

    if not selection.ok:
        # Spread gate blocked all strikes
        logger.info("Phase 4: NO TRADE — %s", selection.reason)
        msg = build_spread_gate_log(winner, score, selection.spread_pct)
        alerter.send_log(msg)
        logger.info("Spread gate log sent to Discord. Exiting cleanly.")
        return

    logger.info(
        "Phase 4 complete: %s $%.0f %s | entry=%.2f | TP=%.2f | SL=%.2f",
        winner, selection.strike, direction,
        selection.entry_price, selection.take_profit, selection.stop_loss,
    )

    # ──────────────────────────────────────────────────────────────────────
    # Phase 5: Paper Order + Alert + State Write
    # ──────────────────────────────────────────────────────────────────────
    logger.info("Phase 5: Firing entry alert and writing state …")

    # Log the paper order with Webull (for record-keeping — spec §2.1)
    try:
        order = client.place_paper_order(
            ticker=winner,
            option_symbol=selection.option_symbol,
            side="BUY",
            quantity=1,
        )
        logger.info(
            "Paper order placed: orderId=%s  status=%s  filled=%.2f",
            order.get("orderId", "n/a"),
            order.get("status", "n/a"),
            order.get("filled_price") or 0.0,
        )
    except Exception as exc:
        # Non-fatal — we still fire the alert; the paper order is cosmetic
        logger.warning("Paper order placement failed (non-fatal): %s", exc)

    # Build and send the entry alert to #alerts
    reason_text = dir_result.reason or breakdown.reason_summary()
    entry_msg = build_entry_alert(
        ticker=winner,
        direction=direction,
        strike=selection.strike,
        option_symbol=selection.option_symbol,
        entry_price=selection.entry_price,
        take_profit=selection.take_profit,
        stop_loss=selection.stop_loss,
        score=score,
        reason=reason_text,
        entry_time=entry_time_str,
    )
    alert_ok = alerter.send_alert(entry_msg)
    if not alert_ok:
        logger.warning("Entry alert Discord delivery failed — trade state still written.")

    # Write active_trade.json so the monitor workflow can pick up state
    active_trade = {
        "ticker": winner,
        "direction": direction,
        "strike": selection.strike,
        "option_symbol": selection.option_symbol,
        "entry_price": selection.entry_price,
        "take_profit": selection.take_profit,
        "stop_loss": selection.stop_loss,
        "entry_time": entry_time_str,
        "score": score,
        "expiry": expiry,
    }
    write_active_trade(active_trade)

    logger.info(
        "═══ Scanner complete — trade entered. Monitor will take over. ═══"
    )

    # ONE TRADE PER DAY (spec §10): scanner exits here.
    # The scalp_monitor.yml workflow will now be triggered by the
    # presence of active_trade.json (or by a workflow_run trigger).


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Scanner interrupted by user.")
    except Exception as exc:
        logger.exception("Unhandled exception in scanner: %s", exc)
        sys.exit(1)
