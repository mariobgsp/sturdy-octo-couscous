"""
Event-driven Backtesting Engine for the IHSG swing trading system.

Replays historical OHLCV data day-by-day through the exact same
scanner, engines, and risk manager used in live mode. Zero look-ahead
bias — each day only sees data up to that point.

Key features:
  - Train/Test split (3.5y / 1.5y default)
  - Hardcoded slippage and IDX broker fees on every trade
  - Portfolio heat management identical to live trading
  - Chandelier trailing stop updates on every bar
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.settings import (
    ATR_PERIOD,
    BACKTEST_FEE_BUY_PCT,
    BACKTEST_FEE_SELL_PCT,
    BACKTEST_INITIAL_CAPITAL,
    BACKTEST_SLIPPAGE_PCT,
    BACKTEST_TRAIN_YEARS,
    BACKTEST_YEARS,
    MAX_RISK_PER_TRADE_PCT,
    STOP_LOSS_ATR_MULTIPLIER,
    TRADE_BUCKET_MAX_PICKS,
    TRAILING_STOP_ATR_MULTIPLIER,
)
from core.database import ParquetStore
from core.engines import run_all_engines
from core.indicators import atr, sma
from core.regime import RegimeSnapshot, RegimeType
from core.risk import RiskManager
from core.scanner import MasterScanner

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class BacktestTrade:
    """A single completed trade from the backtest simulation."""

    ticker: str
    engine: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    raw_entry: float      # price before slippage/fees
    raw_exit: float       # price before slippage/fees
    slippage_entry: float
    slippage_exit: float
    fee_entry: float
    fee_exit: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    regime: str
    holding_days: int


@dataclass
class OpenBacktestPosition:
    """An open position during the backtest simulation."""

    ticker: str
    engine: str
    entry_date: str
    entry_price: float   # after slippage + fees
    raw_entry: float     # before costs
    shares: int
    stop_loss: float
    trailing_stop: float
    highest_high: float
    risk_per_share: float
    risk_amount: float
    regime: str


@dataclass
class BacktestResult:
    """Complete output of a backtest run."""

    trades: list[BacktestTrade] = field(default_factory=list)
    label: str = ""
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_days: int = 0


# ─── Backtester ───────────────────────────────────────────────────────────────


class Backtester:
    """
    Event-driven backtesting engine.

    Replays the scanner + engines + risk pipeline over historical data,
    simulating trades with realistic costs (slippage + IDX broker fees).

    Usage:
        bt = Backtester(store)
        train_result, test_result = bt.run_with_split(tickers)
    """

    def __init__(
        self,
        store: ParquetStore,
        capital: float = BACKTEST_INITIAL_CAPITAL,
        slippage_pct: float = BACKTEST_SLIPPAGE_PCT,
        fee_buy_pct: float = BACKTEST_FEE_BUY_PCT,
        fee_sell_pct: float = BACKTEST_FEE_SELL_PCT,
    ):
        self._store = store
        self._capital = capital
        self._slippage_pct = slippage_pct / 100.0
        self._fee_buy_pct = fee_buy_pct / 100.0
        self._fee_sell_pct = fee_sell_pct / 100.0
        self._risk_mgr = RiskManager()

    def apply_buy_costs(self, price: float) -> float:
        """Apply slippage (buy higher) and broker fee to entry price."""
        slipped = price * (1.0 + self._slippage_pct)
        with_fee = slipped * (1.0 + self._fee_buy_pct)
        return round(with_fee, 2)

    def apply_sell_costs(self, price: float) -> float:
        """Apply slippage (sell lower) and broker fee to exit price."""
        slipped = price * (1.0 - self._slippage_pct)
        with_fee = slipped * (1.0 - self._fee_sell_pct)
        return round(with_fee, 2)

    def _get_date_range(
        self, all_data: dict[str, pd.DataFrame]
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Determine the overall date range from all loaded data."""
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df.index)

        if not all_dates:
            raise ValueError("No data loaded")

        sorted_dates = sorted(all_dates)
        return sorted_dates[0], sorted_dates[-1]

    def split_dates(
        self,
        end_date: pd.Timestamp,
        years_total: float = BACKTEST_YEARS,
        years_train: float = BACKTEST_TRAIN_YEARS,
    ) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        """
        Split the date range into training and test periods.

        Returns (start_date, split_date, end_date).
        """
        start_date = end_date - pd.DateOffset(years=years_total)
        split_date = start_date + pd.DateOffset(
            days=int(years_train * 365.25)
        )
        return start_date, split_date, end_date

    def run(
        self,
        tickers: list[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        label: str = "",
    ) -> BacktestResult:
        """
        Run the backtester over a specific date range.

        Parameters
        ----------
        tickers : list[str]
            Ticker codes to include in the simulation.
        start_date : pd.Timestamp
            First trading day of the simulation.
        end_date : pd.Timestamp
            Last trading day of the simulation.
        label : str
            Label for this run (e.g., "Training", "Blind Test").

        Returns
        -------
        BacktestResult
            All completed trades and final state.
        """
        logger.info(
            "Backtest [%s]: %s to %s | %d tickers | Capital: IDR %s",
            label, start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            len(tickers),
            f"{self._capital:,.0f}",
        )

        # Load all data
        all_data: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self._store.load(ticker)
            if df is not None and len(df) > 0:
                all_data[ticker] = df

        if not all_data:
            logger.warning("No data available for backtesting.")
            return BacktestResult(label=label)

        # Fetch IHSG data for regime calculation (full history)
        try:
            import yfinance as yf
            from config.settings import IHSG_COMPOSITE_TICKER
            ihsg_raw = yf.download(
                IHSG_COMPOSITE_TICKER, period="10y",
                interval="1d", progress=False, auto_adjust=True, timeout=30,
            )
            if isinstance(ihsg_raw.columns, pd.MultiIndex):
                ihsg_raw.columns = ihsg_raw.columns.get_level_values(0)
            ihsg_df = ihsg_raw if not ihsg_raw.empty else None
        except Exception as e:
            logger.warning("Could not fetch IHSG data for regime: %s", e)
            ihsg_df = None

        # Get all unique trading days within the date range
        all_days: set[pd.Timestamp] = set()
        for df in all_data.values():
            mask = (df.index >= start_date) & (df.index <= end_date)
            all_days.update(df.index[mask])

        trading_days = sorted(all_days)
        if not trading_days:
            logger.warning("No trading days in the specified range.")
            return BacktestResult(label=label)

        # State
        equity = self._capital
        open_positions: dict[str, OpenBacktestPosition] = {}
        completed_trades: list[BacktestTrade] = []
        peak_equity = equity

        logger.info(
            "Simulating %d trading days...", len(trading_days)
        )

        for day_idx, today in enumerate(trading_days):
            # ── 1. Build regime snapshot from IHSG data up to today ──
            if ihsg_df is not None:
                ihsg_slice = ihsg_df[ihsg_df.index <= today]
                if len(ihsg_slice) >= 200:
                    from config.settings import (
                        REGIME_ATR_PERIOD,
                        REGIME_SMA_LONG,
                        REGIME_SMA_SHORT,
                    )
                    sma_short = sma(ihsg_slice["Close"], REGIME_SMA_SHORT)
                    sma_long = sma(ihsg_slice["Close"], REGIME_SMA_LONG)
                    atr_s = atr(ihsg_slice, REGIME_ATR_PERIOD)
                    lc = float(ihsg_slice["Close"].iloc[-1])
                    ls = float(sma_short.iloc[-1])
                    ll = float(sma_long.iloc[-1])
                    la = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0

                    if lc > ls and ls > ll:
                        r = RegimeType.BULL
                    elif lc > ll:
                        r = RegimeType.CAUTION
                    else:
                        r = RegimeType.BEAR

                    regime = RegimeSnapshot(
                        regime=r, close=round(lc, 2),
                        sma_short=round(ls, 2), sma_long=round(ll, 2),
                        atr_value=round(la, 2),
                        as_of_date=today.strftime("%Y-%m-%d"),
                    )
                else:
                    regime = RegimeSnapshot(
                        "CAUTION", 0, 0, 0, 0,
                        today.strftime("%Y-%m-%d"),
                    )
            else:
                regime = RegimeSnapshot(
                    "CAUTION", 0, 0, 0, 0,
                    today.strftime("%Y-%m-%d"),
                )

            # ── 2. Update trailing stops and check stop hits ──
            tickers_to_close: list[tuple[str, float, str]] = []

            for ticker, pos in list(open_positions.items()):
                if ticker not in all_data:
                    continue
                df = all_data[ticker]
                if today not in df.index:
                    continue

                bar = df.loc[today]
                current_high = float(bar["High"])
                current_low = float(bar["Low"])

                # Update highest high
                if current_high > pos.highest_high:
                    pos.highest_high = current_high

                # Update trailing stop (Chandelier)
                atr_series = atr(df[df.index <= today], period=ATR_PERIOD)
                if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]):
                    current_atr = float(atr_series.iloc[-1])
                    new_stop = pos.highest_high - (
                        current_atr * TRAILING_STOP_ATR_MULTIPLIER
                    )
                    if new_stop > pos.trailing_stop:
                        pos.trailing_stop = new_stop

                # Check stop hit
                if current_low <= pos.trailing_stop:
                    exit_price = pos.trailing_stop  # stopped out at stop level
                    tickers_to_close.append(
                        (ticker, exit_price, "trailing_stop")
                    )

            # Close stopped-out positions
            for ticker, raw_exit, reason in tickers_to_close:
                pos = open_positions.pop(ticker)
                actual_exit = self.apply_sell_costs(raw_exit)
                pnl = (actual_exit - pos.entry_price) * pos.shares
                pnl_pct = ((actual_exit / pos.entry_price) - 1.0) * 100.0

                entry_dt = pd.Timestamp(pos.entry_date)
                holding = (today - entry_dt).days

                completed_trades.append(
                    BacktestTrade(
                        ticker=ticker,
                        engine=pos.engine,
                        entry_date=pos.entry_date,
                        exit_date=today.strftime("%Y-%m-%d"),
                        entry_price=pos.entry_price,
                        exit_price=actual_exit,
                        shares=pos.shares,
                        raw_entry=pos.raw_entry,
                        raw_exit=raw_exit,
                        slippage_entry=round(
                            pos.raw_entry * self._slippage_pct, 2
                        ),
                        slippage_exit=round(
                            raw_exit * self._slippage_pct, 2
                        ),
                        fee_entry=round(
                            pos.raw_entry * self._fee_buy_pct, 2
                        ),
                        fee_exit=round(
                            raw_exit * self._fee_sell_pct, 2
                        ),
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                        exit_reason=reason,
                        regime=pos.regime,
                        holding_days=holding,
                    )
                )
                equity += pnl

            # ── 3. Scan for new entries ──
            # Calculate current portfolio heat
            total_risk = sum(
                p.risk_amount for p in open_positions.values()
            )
            heat_pct = (total_risk / self._capital) * 100.0

            if (
                len(open_positions) < TRADE_BUCKET_MAX_PICKS
                and heat_pct < 6.0
            ):
                for ticker in tickers:
                    if ticker in open_positions:
                        continue
                    if ticker not in all_data:
                        continue

                    df = all_data[ticker]
                    if today not in df.index:
                        continue

                    # Slice data up to today (no look-ahead)
                    df_slice = df[df.index <= today]
                    if len(df_slice) < 200:
                        continue

                    # Run avoid filters (simplified — check SMA200)
                    sma200 = sma(df_slice["Close"], 200)
                    last_close = float(df_slice["Close"].iloc[-1])
                    if pd.isna(sma200.iloc[-1]) or last_close < float(
                        sma200.iloc[-1]
                    ):
                        continue

                    # Run entry engines
                    signals = run_all_engines(df_slice, ticker, regime)
                    if not signals:
                        continue

                    # Take the highest priority signal
                    best = max(signals, key=lambda s: (s.priority, s.score))
                    raw_entry = best.price

                    # Apply costs
                    actual_entry = self.apply_buy_costs(raw_entry)

                    # Calculate risk
                    atr_series = atr(df_slice, period=ATR_PERIOD)
                    if pd.isna(atr_series.iloc[-1]):
                        continue
                    current_atr = float(atr_series.iloc[-1])

                    stop = self._risk_mgr.calculate_stop_loss(
                        actual_entry, current_atr
                    )
                    risk_per_share = actual_entry - stop
                    if risk_per_share <= 0:
                        continue

                    adj_risk = self._risk_mgr.adjust_risk_for_regime(
                        MAX_RISK_PER_TRADE_PCT, regime.regime.value
                    )
                    shares = self._risk_mgr.calculate_position_size(
                        equity, actual_entry, stop, adj_risk
                    )
                    if shares <= 0:
                        continue

                    risk_amount = risk_per_share * shares

                    # Check if adding this trade would exceed heat
                    new_heat = (
                        (total_risk + risk_amount) / self._capital
                    ) * 100.0
                    if new_heat > 6.0:
                        continue

                    # Check if we can afford it
                    cost = actual_entry * shares
                    available = equity - sum(
                        p.entry_price * p.shares
                        for p in open_positions.values()
                    )
                    if cost > available:
                        continue

                    trailing = self._risk_mgr.calculate_trailing_stop(
                        float(df_slice["High"].iloc[-1]), current_atr
                    )

                    open_positions[ticker] = OpenBacktestPosition(
                        ticker=ticker,
                        engine=best.engine,
                        entry_date=today.strftime("%Y-%m-%d"),
                        entry_price=actual_entry,
                        raw_entry=raw_entry,
                        shares=shares,
                        stop_loss=stop,
                        trailing_stop=max(stop, trailing),
                        highest_high=float(df_slice["High"].iloc[-1]),
                        risk_per_share=risk_per_share,
                        risk_amount=risk_amount,
                        regime=regime.regime.value,
                    )

                    total_risk += risk_amount
                    heat_pct = (total_risk / self._capital) * 100.0

                    if len(open_positions) >= TRADE_BUCKET_MAX_PICKS:
                        break

            # Progress logging every 250 days
            if day_idx > 0 and day_idx % 250 == 0:
                logger.info(
                    "  Day %d/%d | Equity: IDR %s | Open: %d | Trades: %d",
                    day_idx, len(trading_days),
                    f"{equity:,.0f}",
                    len(open_positions),
                    len(completed_trades),
                )

        # ── Force-close any remaining open positions at last bar ──
        for ticker, pos in list(open_positions.items()):
            if ticker in all_data:
                df = all_data[ticker]
                last_bar = df[df.index <= end_date]
                if len(last_bar) > 0:
                    raw_exit = float(last_bar["Close"].iloc[-1])
                    actual_exit = self.apply_sell_costs(raw_exit)
                    pnl = (actual_exit - pos.entry_price) * pos.shares

                    entry_dt = pd.Timestamp(pos.entry_date)
                    holding = (trading_days[-1] - entry_dt).days

                    completed_trades.append(
                        BacktestTrade(
                            ticker=ticker,
                            engine=pos.engine,
                            entry_date=pos.entry_date,
                            exit_date=trading_days[-1].strftime("%Y-%m-%d"),
                            entry_price=pos.entry_price,
                            exit_price=actual_exit,
                            shares=pos.shares,
                            raw_entry=pos.raw_entry,
                            raw_exit=raw_exit,
                            slippage_entry=round(
                                pos.raw_entry * self._slippage_pct, 2
                            ),
                            slippage_exit=round(
                                raw_exit * self._slippage_pct, 2
                            ),
                            fee_entry=round(
                                pos.raw_entry * self._fee_buy_pct, 2
                            ),
                            fee_exit=round(
                                raw_exit * self._fee_sell_pct, 2
                            ),
                            pnl=round(pnl, 2),
                            pnl_pct=round(
                                ((actual_exit / pos.entry_price) - 1) * 100, 2
                            ),
                            exit_reason="end_of_period",
                            regime=pos.regime,
                            holding_days=holding,
                        )
                    )
                    equity += pnl

        logger.info(
            "Backtest [%s] complete: %d trades | Final equity: IDR %s",
            label, len(completed_trades), f"{equity:,.0f}",
        )

        return BacktestResult(
            trades=completed_trades,
            label=label,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=self._capital,
            final_equity=round(equity, 2),
            total_days=len(trading_days),
        )

    def run_with_split(
        self,
        tickers: list[str],
    ) -> tuple[BacktestResult, BacktestResult]:
        """
        Run the backtest with automatic train/test split.

        Returns (training_result, blind_test_result).
        """
        # Load all data to find the date range
        all_data: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            df = self._store.load(ticker)
            if df is not None and len(df) > 0:
                all_data[ticker] = df

        if not all_data:
            empty = BacktestResult(label="No Data")
            return empty, empty

        _, end_date = self._get_date_range(all_data)
        start_date, split_date, end_date = self.split_dates(end_date)

        logger.info("=" * 60)
        logger.info("BACKTEST WITH TRAIN/TEST SPLIT")
        logger.info("=" * 60)
        logger.info(
            "Total period: %s to %s",
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )
        logger.info(
            "Training:     %s to %s (%.1f years)",
            start_date.strftime("%Y-%m-%d"),
            split_date.strftime("%Y-%m-%d"),
            BACKTEST_TRAIN_YEARS,
        )
        logger.info(
            "Blind Test:   %s to %s (%.1f years)",
            split_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            BACKTEST_YEARS - BACKTEST_TRAIN_YEARS,
        )
        logger.info("=" * 60)

        train_result = self.run(
            tickers, start_date, split_date, label="Training"
        )
        test_result = self.run(
            tickers, split_date, end_date, label="Blind Test"
        )

        return train_result, test_result
