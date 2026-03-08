"""
Execution CLI — cron-ready script for live order execution.

Designed to run at 15:50 WIB daily (or via cron/scheduler).
Scans for trade signals, validates through failsafes, and
submits bracket orders through the broker adapter.

Usage:
    python -m scripts.execute
    python -m scripts.execute --dry-run
    python -m scripts.execute --tickers ASII BBCA
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

from config.settings import (
    DEFAULT_CAPITAL,
    EXECUTION_SCHEDULE_WIB,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
)
from core.alerts import fire_trade_alert
from core.bracket_order import BracketOrderManager
from core.broker import SimulatedBroker
from core.database import ParquetStore
from core.failsafes import DrawdownHaltError, FatFingerError, FailsafeGuard
from core.indicators import atr
from core.portfolio import Portfolio
from core.regime import MarketRegime
from core.scanner import MasterScanner

logger = logging.getLogger("scripts.execute")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "IHSG Live Execution Engine — "
            f"Scheduled for {EXECUTION_SCHEDULE_WIB} WIB"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.execute\n"
            "  python -m scripts.execute --dry-run\n"
            "  python -m scripts.execute --tickers ASII BBCA TLKM\n"
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific tickers to scan (default: all stored)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate everything without actually placing orders",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help=f"Account capital (default: {DEFAULT_CAPITAL:,.0f})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_handlers = [logging.StreamHandler()]
    try:
        from pathlib import Path
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_handlers.append(
            logging.FileHandler(log_dir / "execution.log", encoding="utf-8")
        )
    except Exception:
        pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=log_handlers,
    )

    now = datetime.now()
    logger.info("=" * 60)
    logger.info("IHSG EXECUTION ENGINE")
    logger.info("=" * 60)
    logger.info("Time:     %s", now.strftime("%Y-%m-%d %H:%M:%S WIB"))
    logger.info("Mode:     %s", "DRY RUN" if args.dry_run else "LIVE")
    logger.info("Capital:  IDR %s", f"{args.capital:,.0f}")
    logger.info("=" * 60)

    # ── 1. Initialize components ──
    store = ParquetStore()
    broker = SimulatedBroker(initial_balance=args.capital)
    guard = FailsafeGuard()
    bracket_mgr = BracketOrderManager(broker, guard)

    # ── 2. Check daily drawdown before doing anything ──
    try:
        guard.check_daily_drawdown(broker)
    except DrawdownHaltError as e:
        logger.critical("EXECUTION HALTED: %s", e)
        sys.exit(1)

    # ── 3. Fetch regime ──
    logger.info("[1/4] Fetching market regime...")
    regime_obj = MarketRegime()
    regime = regime_obj.get_snapshot()
    logger.info("Regime: %s", regime)

    # ── 4. Scan for signals ──
    logger.info("[2/4] Scanning for trade signals...")
    scanner = MasterScanner(store)

    if args.tickers:
        tickers = [t.replace(".JK", "") for t in args.tickers]
    else:
        tickers = store.list_tickers()

    if not tickers:
        logger.error("No tickers available. Run scripts.ingest first.")
        sys.exit(1)

    scan_result = scanner.scan_universe(
        tickers, check_earnings=False, regime=regime
    )

    logger.info(
        "Scan results: Avoid=%d | Wait=%d | Trade=%d",
        len(scan_result.avoid),
        len(scan_result.wait),
        len(scan_result.trade),
    )

    if not scan_result.trade:
        logger.info("No trade signals. Nothing to execute.")
        logger.info("Execution complete.")
        return

    # ── 5. Process trade signals ──
    logger.info("[3/4] Processing %d trade signals...", len(scan_result.trade))

    orders_placed = 0
    orders_blocked = 0

    for trade_entry in scan_result.trade:
        ticker = trade_entry.ticker
        entry_price = trade_entry.price

        logger.info(
            "Processing: %s | %s | IDR %,.0f",
            ticker, trade_entry.signal, entry_price,
        )

        # Get ATR for bracket order calculation
        df = store.load(ticker)
        if df is None or len(df) < 20:
            logger.warning("Skipping %s — insufficient data.", ticker)
            continue

        atr_series = atr(df)
        if len(atr_series) == 0 or pd.isna(atr_series.iloc[-1]):
            logger.warning("Skipping %s — ATR unavailable.", ticker)
            continue
        current_atr = float(atr_series.iloc[-1])

        try:
            if args.dry_run:
                # Dry run: just log what WOULD happen
                from core.risk import RiskManager
                rm = RiskManager()
                sl = rm.calculate_stop_loss(entry_price, current_atr)
                tp = entry_price + (current_atr * 3.0)
                adj_risk = rm.adjust_risk_for_regime(2.0, regime.regime.value)
                shares = rm.calculate_position_size(
                    args.capital, entry_price, sl, adj_risk
                )
                logger.info(
                    "  [DRY RUN] %s: BUY %d shares @ IDR %,.0f | "
                    "SL: IDR %,.0f | TP: IDR %,.0f",
                    ticker, shares, entry_price, sl, tp,
                )
                orders_placed += 1
            else:
                result = bracket_mgr.create_bracket(
                    ticker=ticker,
                    entry_price=entry_price,
                    atr_value=current_atr,
                    regime=regime.regime.value,
                    capital=args.capital,
                )
                if result.entry.status == "FILLED":
                    orders_placed += 1
                    fire_trade_alert(
                        ticker=ticker,
                        signal=trade_entry.signal,
                        price=entry_price,
                        details={
                            "order_id": result.entry.order_id,
                            "position_size": result.entry.shares,
                            "stop_loss": result.stop_loss.price,
                        },
                    )
                else:
                    orders_blocked += 1
                    logger.warning(
                        "  Order REJECTED for %s: %s",
                        ticker, result.entry.message,
                    )

        except FatFingerError as e:
            logger.critical("FAT FINGER BLOCKED: %s", e)
            orders_blocked += 1
        except DrawdownHaltError as e:
            logger.critical("DRAWDOWN HALT: %s", e)
            break
        except Exception as e:
            logger.error("Error processing %s: %s", ticker, e)
            orders_blocked += 1

    # ── 6. Summary ──
    logger.info("[4/4] Execution summary")
    logger.info("  Orders placed:  %d", orders_placed)
    logger.info("  Orders blocked: %d", orders_blocked)
    logger.info("=" * 60)
    logger.info("Execution complete.")


# Need pandas for ATR check
import pandas as pd  # noqa: E402

if __name__ == "__main__":
    main()
