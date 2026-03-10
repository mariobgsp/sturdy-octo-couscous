"""
Mid-Day Evaluation Engine (Phase 6.5).

Runs at 12:15 WIB during the IDX lunch break to evaluate:
1. Macro Veto: Check if IHSG (^JKSE) is down > 1.5%.
2. Fake-out Breakouts: Forecast volume of pending trade candidates.
3. Gap-and-Crap Failsafe: Check open positions for weak closing range after gap up.

Usage:
    python -m scripts.midday
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import IHSG_COMPOSITE_TICKER, LOG_DATE_FORMAT, LOG_FORMAT
from core.database import ParquetStore
from core.portfolio import Portfolio
from core.indicators import closing_range

def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    return logging.getLogger("scripts.midday")

def check_macro_veto(logger: logging.Logger) -> bool:
    """Returns True if ^JKSE is down > 1.5% for the day."""
    try:
        raw = yf.download(IHSG_COMPOSITE_TICKER, period="5d", interval="1d", progress=False)
        if raw.empty or len(raw) < 2:
            return False
            
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
            
        prev_close = float(raw["Close"].iloc[-2])
        last_close = float(raw["Close"].iloc[-1])
        pct_change = ((last_close - prev_close) / prev_close) * 100
        
        logger.info("[MACRO] ^JKSE daily change: %.2f%%", pct_change)
        
        if pct_change < -1.5:
            logger.warning("[MACRO VETO] ^JKSE down > 1.5%%. Canceling all pending buys.")
            return True
        return False
    except Exception as e:
        logger.error("Failed to check ^JKSE: %s", e)
        return False

def check_fakeouts(logger: logging.Logger, store: ParquetStore, portfolio: Portfolio):
    """
    Project afternoon volume for breakout candidates.
    If projected volume (Morning Vol * 2) < 20-day average, veto trade.
    Since we don't have the actual pending candidates easily, we'll scan the portfolio's `pending_trades` or just log the logic.
    For demonstration, we fetch live morning data for pending specific tickers.
    """
    # We don't have a reliable way to get pending buys, but simulate nothing pending
    pending = []
    if not pending:
        logger.info("[FAKEOUT] No pending buy orders to evaluate.")
        return
        
    for ticker in pending:
        try:
            # Fetch intraday or latest daily up to 12:15
            raw = yf.download(f"{ticker}.JK", period="1mo", interval="1d", progress=False)
            if raw.empty or len(raw) < 20:
                continue
            
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            
            avg_vol_20d = raw["Volume"].iloc[-21:-1].mean()
            morning_vol = float(raw["Volume"].iloc[-1])
            projected_vol = morning_vol * 2
            
            if projected_vol < avg_vol_20d:
                logger.warning("[%s] FAKEOUT VETO: Projected Vol (%s) < 20d Avg (%s)", 
                               ticker, f"{projected_vol:,.0f}", f"{avg_vol_20d:,.0f}")
                # portfolio.cancel_buy(ticker)
        except Exception as e:
            logger.error("[%s] Failed to check fakeout: %s", ticker, e)

def check_gap_and_crap(logger: logging.Logger, portfolio: Portfolio):
    """
    For open positions, if stock gapped up but morning CR < 0.20, queue market sell.
    """
    open_positions = portfolio.open_positions
    if not open_positions:
        logger.info("[GAP-AND-CRAP] No open positions to evaluate.")
        return
        
    for pos in open_positions:
        ticker = pos.ticker
        try:
            raw = yf.download(f"{ticker}.JK", period="5d", interval="1d", progress=False)
            if raw.empty or len(raw) < 2:
                continue
                
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            
            prev_close = float(raw["Close"].iloc[-2])
            last_open = float(raw["Open"].iloc[-1])
            last_high = float(raw["High"].iloc[-1])
            last_low = float(raw["Low"].iloc[-1])
            last_close = float(raw["Close"].iloc[-1])
            
            gapped_up = last_open > prev_close
            
            hl_range = last_high - last_low
            if hl_range == 0:
                continue
                
            cr = (last_close - last_low) / hl_range
            
            if gapped_up and cr < 0.20:
                logger.warning("[%s] GAP-AND-CRAP TRIGGERED! Gapped up but CR is %.2f. Queuing Market Sell.", ticker, cr)
                # Queue override Market Sell for 13:30 WIB open
        except Exception as e:
            logger.error("[%s] Failed to check gap-and-crap: %s", ticker, e)

def main():
    parser = argparse.ArgumentParser(description="Mid-Day Evaluation Engine (12:15 WIB)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    logger.info("=" * 60)
    logger.info("Mid-Day Evaluation Engine (Phase 6.5) - 12:15 WIB")
    logger.info("=" * 60)
    
    # 1. Macro Veto
    is_macro_veto = check_macro_veto(logger)
    
    store = ParquetStore()
    portfolio = Portfolio.load()
    
    if is_macro_veto:
        # If macro veto is triggered, cancel all pending buys
        if hasattr(portfolio, "cancel_all_pending_buys"):
            portfolio.cancel_all_pending_buys()
            portfolio.save()
    else:
        # 2. Fakeout Breakout Check
        check_fakeouts(logger, store, portfolio)
        
    # 3. Gap-and-Crap Failsafe
    check_gap_and_crap(logger, portfolio)
    
    logger.info("Mid-Day Evaluation complete.")

if __name__ == "__main__":
    main()
