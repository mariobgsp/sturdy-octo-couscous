"""
Backtest CLI — Run the backtesting engine from the command line.

Usage:
    python -m scripts.backtest
    python -m scripts.backtest --tickers BBCA BBRI ASII
    python -m scripts.backtest --capital 200000000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from config.settings import (
    BACKTEST_INITIAL_CAPITAL,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
)
from core.backtester import Backtester
from core.backtest_report import (
    generate_report_card,
    print_report_card,
)
from core.database import ParquetStore

logger = logging.getLogger("scripts.backtest")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IHSG Backtest Engine — The Truth Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.backtest\n"
            "  python -m scripts.backtest --tickers BBCA BBRI ASII TLKM UNVR\n"
            "  python -m scripts.backtest --capital 200000000\n"
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific tickers to backtest (default: all stored data)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=BACKTEST_INITIAL_CAPITAL,
        help=f"Starting capital in IDR (default: {BACKTEST_INITIAL_CAPITAL:,.0f})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

    store = ParquetStore()

    # Determine tickers
    if args.tickers:
        tickers = [t.replace(".JK", "") for t in args.tickers]
    else:
        tickers = store.list_tickers()

    if not tickers:
        logger.error("No tickers available. Run scripts.ingest first.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("IHSG BACKTEST ENGINE")
    logger.info("=" * 60)
    logger.info("Tickers:  %d", len(tickers))
    logger.info("Capital:  IDR %s", f"{args.capital:,.0f}")
    logger.info("=" * 60)

    start_time = time.time()

    # Create backtester and run with train/test split
    bt = Backtester(store, capital=args.capital)
    train_result, test_result = bt.run_with_split(tickers)

    elapsed = time.time() - start_time

    # Generate report cards
    train_card = generate_report_card(
        trades=train_result.trades,
        initial_capital=train_result.initial_capital,
        label=train_result.label,
        start_date=train_result.start_date,
        end_date=train_result.end_date,
        total_days=train_result.total_days,
    )

    test_card = generate_report_card(
        trades=test_result.trades,
        initial_capital=test_result.initial_capital,
        label=test_result.label,
        start_date=test_result.start_date,
        end_date=test_result.end_date,
        total_days=test_result.total_days,
    )

    # Print the report card
    print_report_card(train=train_card, test=test_card)

    logger.info("Backtest completed in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
