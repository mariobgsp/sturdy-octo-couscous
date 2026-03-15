"""
Master Scanner Engine — the triage system.

Scans the entire IHSG universe and categorizes every stock into
one of three buckets:

  AVOID  — Stocks that fail liquidity, price, or trend filters.
  WAIT   — Stocks forming setups (consolidation, FVG approach).
  TRADE  — Stocks that triggered an entry condition today (max 5).

All scanning is performed against locally-cached Parquet data.
No live API calls are made during the scan (except for the
optional earnings proximity check on surviving candidates,
and the one-time IHSG composite fetch for regime detection).
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import (
    ADTV_LOOKBACK,
    ADTV_MIN_IDR,
    ATR_PERIOD,
    CONSOLIDATION_MAX_RANGE_PCT,
    CONSOLIDATION_WINDOW,
    EARNINGS_PROXIMITY_HOURS,
    FVG_ATR_PROXIMITY,
    PENNY_STOCK_THRESHOLD,
    SMA_200_LOOKBACK,
    TRADE_BUCKET_MAX_PICKS,
    WYCKOFF_PHASE_B_VAR_LOOKBACK,
    WYCKOFF_PHASE_B_VAR_PERCENTILE,
    VSA_SQUAT_VOL_RATIO,
    VSA_SQUAT_EFFICIENCY_PERCENTILE,
    VSA_SQUAT_PERCENTILE_LOOKBACK,
)
from core.database import ParquetStore
from core.engines import EntrySignal, run_all_engines
from core.indicators import (
    adtv,
    atr,
    cvd,
    detect_fvg,
    efficiency_ratio,
    is_tight_consolidation,
    rolling_percentile,
    rsi,
    sma,
    volume_ratio,
)
from core.regime import MarketRegime, RegimeSnapshot, RegimeType
from core.risk import RiskManager

logger = logging.getLogger(__name__)


# ─── Result Data Structures ──────────────────────────────────────────────────


@dataclass
class AvoidEntry:
    """A stock placed in the Avoid bucket."""

    ticker: str
    reason: str


@dataclass
class WaitEntry:
    """A stock placed in the Wait bucket."""

    ticker: str
    condition: str  # "tight_consolidation" or "fvg_approach"
    details: dict = field(default_factory=dict)


@dataclass
class TradeEntry:
    """A stock placed in the Trade bucket."""

    ticker: str
    signal: str  # Entry signal description
    score: float  # Ranking score (higher = better)
    price: float
    details: dict = field(default_factory=dict)


@dataclass
class ScanResult:
    """Complete output of a full universe scan."""

    avoid: list[AvoidEntry] = field(default_factory=list)
    wait: list[WaitEntry] = field(default_factory=list)
    trade: list[TradeEntry] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # tickers with no data
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary of scan results."""
        return (
            f"Avoid: {len(self.avoid)} | "
            f"Wait: {len(self.wait)} | "
            f"Trade: {len(self.trade)} | "
            f"Skipped: {len(self.skipped)}"
        )


# ─── Master Scanner ──────────────────────────────────────────────────────────


class MasterScanner:
    """
    Tri-bucket triage scanner for the IHSG universe.

    Processes stocks through a sequential filter pipeline:
    1. Avoid filters (drop illiquid, penny, downtrend, earnings)
    2. Wait filters (flag consolidation, FVG approach)
    3. Trade bucket (entry signal evaluation + ranking)

    Usage:
        store = ParquetStore()
        scanner = MasterScanner(store)
        result = scanner.scan_universe(tickers)
        print(result.summary())
    """

    def __init__(self, store: ParquetStore) -> None:
        self._store = store
        self._risk_manager = RiskManager()
        self._earnings_lock = threading.Lock()

    # ── Main Scan Pipeline ────────────────────────────────────────────────

    def scan_universe(
        self,
        tickers: list[str],
        check_earnings: bool = False,
        regime: RegimeSnapshot | None = None,
    ) -> ScanResult:
        """
        Scan all tickers through the Avoid -> Wait -> Trade pipeline.

        Parameters
        ----------
        tickers : list[str]
            Ticker codes to scan (without .JK suffix).
        check_earnings : bool
            If True, check for upcoming earnings/corporate actions
            via yfinance (requires network). Only applied to stocks
            surviving the first 3 Avoid filters.
        regime : RegimeSnapshot | None
            Current market regime. If None, fetches IHSG composite
            automatically.

        Returns
        -------
        ScanResult
            Categorized stocks with reasons and details.
        """
        result = ScanResult()
        total = len(tickers)
        avoid_counts: dict[str, int] = {
            "low_adtv": 0,
            "penny_stock": 0,
            "below_sma200": 0,
            "earnings_proximity": 0,
            "insufficient_data": 0,
        }

        # Fetch regime if not provided
        if regime is None:
            try:
                regime = MarketRegime().get_snapshot()
            except Exception as e:
                logger.warning("Could not fetch regime, using CAUTION: %s", e)
                regime = RegimeSnapshot(
                    regime=RegimeType.CAUTION,
                    close=0, sma_short=0, sma_long=0, atr_value=0,
                    as_of_date="fallback",
                )

        # Candidates that pass all Avoid filters
        trade_candidates: list[TradeEntry] = []

        logger.info("Scanning %d tickers... Regime: %s", total, regime.regime.value)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._process_ticker, ticker, regime, check_earnings): ticker
                for ticker in tickers
            }

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                ticker = futures[future]
                completed += 1
                if completed % 100 == 0:
                    logger.info("  [%d/%d] Processing...", completed, total)

                try:
                    res = future.result()
                    status = res["status"]
                    if status == "skipped":
                        result.skipped.append(res["ticker"])
                    elif status == "avoid":
                        result.avoid.append(res["entry"])
                        avoid_counts[res["avoid_reason_key"]] += 1
                    elif status == "wait":
                        result.wait.append(res["entry"])
                    elif status == "trade":
                        trade_candidates.append(res["entry"])
                except Exception as e:
                    logger.error("[%s] Error processing ticker: %s", ticker, e)

        # ── RANK AND SELECT TOP TRADE PICKS ───────────────────────────
        trade_candidates.sort(key=lambda e: e.score, reverse=True)
        result.trade = trade_candidates[:TRADE_BUCKET_MAX_PICKS]

        # If there are candidates beyond the max, they go to Wait
        for overflow in trade_candidates[TRADE_BUCKET_MAX_PICKS:]:
            result.wait.append(
                WaitEntry(
                    ticker=overflow.ticker,
                    condition="trade_overflow",
                    details={"signal": overflow.signal, "score": overflow.score},
                )
            )

        # Stats
        result.stats = {
            "total_scanned": total,
            "total_with_data": total - len(result.skipped),
            "avoid_breakdown": avoid_counts,
            "regime": regime.regime.value,
            "regime_detail": str(regime),
        }

        logger.info("Scan complete. %s", result.summary())
        return result

    def _process_ticker(self, ticker: str, regime: RegimeSnapshot, check_earnings: bool) -> dict:
        """Process a single ticker, returning a dict of results."""
        df = self._store.load(ticker)
        if df is None or df.empty:
            return {"status": "skipped", "ticker": ticker}

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # ── AVOID FILTERS ────────────────────────────────────────
        avoid_reason = self._run_avoid_filters(df, ticker)
        if avoid_reason:
            key = "insufficient_data"
            for k in ["low_adtv", "penny_stock", "below_sma200", "earnings_proximity", "insufficient_data"]:
                if k in avoid_reason:
                    key = k
                    break
            return {
                "status": "avoid",
                "entry": AvoidEntry(ticker=ticker, reason=avoid_reason),
                "avoid_reason_key": key
            }

        # ── EARNINGS PROXIMITY ───────────────────────────────────
        if check_earnings:
            with self._earnings_lock:
                has_earnings = self._filter_earnings_proximity(ticker)
                time.sleep(1.0)  # Rate limit calendar lookups
            if has_earnings:
                return {
                    "status": "avoid",
                    "entry": AvoidEntry(ticker=ticker, reason="earnings_proximity: event within 48h"),
                    "avoid_reason_key": "earnings_proximity"
                }

        # ── WAIT & TRADE BUCKETS ──────────────────────────────────────
        wait_entry = self._run_wait_filters(df, ticker)
        if wait_entry:
            return {"status": "wait", "entry": wait_entry}

        trade_entry = self._run_engines(df, ticker, regime)
        if trade_entry:
            return {"status": "trade", "entry": trade_entry}

        return {"status": "none", "ticker": ticker}

    # ══════════════════════════════════════════════════════════════════════
    # AVOID FILTERS
    # ══════════════════════════════════════════════════════════════════════

    def _run_avoid_filters(self, df: pd.DataFrame, ticker: str) -> str | None:
        """
        Run all Avoid filters sequentially.
        Returns the reason string if the stock should be avoided, else None.
        """
        # Need sufficient data for SMA(200)
        if len(df) < SMA_200_LOOKBACK:
            return "insufficient_data: <200 bars"

        # Filter 1: ADTV < IDR 10 Billion
        reason = self._filter_low_adtv(df)
        if reason:
            return reason

        # Filter 2: Penny stock (Close < IDR 100)
        reason = self._filter_penny_stock(df)
        if reason:
            return reason

        # Filter 3: Below 200-day SMA (downtrend)
        reason = self._filter_below_sma200(df)
        if reason:
            return reason

        return None

    def _filter_low_adtv(self, df: pd.DataFrame) -> str | None:
        """Drop stocks with ADTV under IDR 10 Billion."""
        adtv_series = adtv(df, period=ADTV_LOOKBACK)
        last_adtv = adtv_series.iloc[-1]

        if pd.isna(last_adtv) or last_adtv < ADTV_MIN_IDR:
            adtv_b = last_adtv / 1e9 if not pd.isna(last_adtv) else 0
            return f"low_adtv: IDR {adtv_b:.1f}B < {ADTV_MIN_IDR/1e9:.0f}B minimum"
        return None

    def _filter_penny_stock(self, df: pd.DataFrame) -> str | None:
        """Drop stocks priced under IDR 100."""
        last_close = df["Close"].iloc[-1]
        if pd.isna(last_close) or last_close < PENNY_STOCK_THRESHOLD:
            return f"penny_stock: IDR {last_close:.0f} < {PENNY_STOCK_THRESHOLD:.0f}"
        return None

    def _filter_below_sma200(self, df: pd.DataFrame) -> str | None:
        """Drop stocks trading below their 200-day SMA (downtrend)."""
        sma200 = sma(df["Close"], SMA_200_LOOKBACK)
        last_close = df["Close"].iloc[-1]
        last_sma = sma200.iloc[-1]

        if pd.isna(last_sma):
            return "insufficient_data: SMA(200) not yet computed"

        if last_close < last_sma:
            pct_below = ((last_sma - last_close) / last_sma) * 100
            return f"below_sma200: price {pct_below:.1f}% below SMA(200)"
        return None

    def _filter_earnings_proximity(self, ticker: str) -> bool:
        """
        Check if the stock has earnings or corporate actions within 48 hours.

        Uses yfinance's calendar property (requires network call).
        Returns True if the stock should be AVOIDED.
        """
        try:
            yf_ticker = yf.Ticker(f"{ticker}.JK")
            cal = yf_ticker.calendar

            if cal is None or (isinstance(cal, pd.DataFrame) and cal.empty):
                return False
            if isinstance(cal, dict) and not cal:
                return False

            now = datetime.now(timezone.utc)
            threshold = now + timedelta(hours=EARNINGS_PROXIMITY_HOURS)

            # Calendar can be a DataFrame or dict depending on yfinance version
            if isinstance(cal, pd.DataFrame):
                for col in cal.columns:
                    for val in cal[col]:
                        if isinstance(val, (datetime, pd.Timestamp)):
                            event_dt = pd.Timestamp(val)
                            if event_dt.tzinfo is None:
                                event_dt = event_dt.tz_localize("UTC")
                            if now <= event_dt <= threshold:
                                logger.debug(
                                    "[%s] Earnings event at %s (within %dh)",
                                    ticker, event_dt, EARNINGS_PROXIMITY_HOURS,
                                )
                                return True
            elif isinstance(cal, dict):
                for key, val in cal.items():
                    if isinstance(val, (datetime, pd.Timestamp)):
                        event_dt = pd.Timestamp(val)
                        if event_dt.tzinfo is None:
                            event_dt = event_dt.tz_localize("UTC")
                        if now <= event_dt <= threshold:
                            logger.debug(
                                "[%s] Earnings event '%s' at %s (within %dh)",
                                ticker, key, event_dt, EARNINGS_PROXIMITY_HOURS,
                            )
                            return True
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, (datetime, pd.Timestamp)):
                                event_dt = pd.Timestamp(item)
                                if event_dt.tzinfo is None:
                                    event_dt = event_dt.tz_localize("UTC")
                                if now <= event_dt <= threshold:
                                    return True

            return False

        except Exception as e:
            logger.debug(
                "[%s] Could not check earnings calendar: %s", ticker, e
            )
            return False  # Fail open — don't penalize if calendar unavailable

    # ══════════════════════════════════════════════════════════════════════
    # WAIT FILTERS
    # ══════════════════════════════════════════════════════════════════════

    def _run_wait_filters(
        self, df: pd.DataFrame, ticker: str
    ) -> WaitEntry | None:
        """
        Check if the stock matches a Wait condition.
        Returns a WaitEntry if matched, else None.
        """
        # Check 1: Tight 14-day consolidation
        consolidating, range_pct = is_tight_consolidation(
            df, window=CONSOLIDATION_WINDOW, max_range_pct=CONSOLIDATION_MAX_RANGE_PCT
        )
        if consolidating:
            return WaitEntry(
                ticker=ticker,
                condition="tight_consolidation",
                details={
                    "window": CONSOLIDATION_WINDOW,
                    "range_pct": round(range_pct, 2),
                    "price": round(float(df["Close"].iloc[-1]), 2),
                },
            )

        # Check 2: Approaching an unfilled bullish FVG
        fvg_entry = self._check_fvg_approach(df, ticker)
        if fvg_entry:
            return fvg_entry

        # Check 3: Wyckoff Phase B Accumulation
        # 60-day rolling variance of close drops into historical bottom 10th percentile
        # AND CVD is strictly positive
        if len(df) >= max(WYCKOFF_PHASE_B_VAR_LOOKBACK * 2, 60):
            var_60d = df["Close"].rolling(window=WYCKOFF_PHASE_B_VAR_LOOKBACK).var()
            var_pctile = rolling_percentile(var_60d, window=252)
            cvd_series = cvd(df)
            
            last_var_pctile = float(var_pctile.iloc[-1]) if not pd.isna(var_pctile.iloc[-1]) else 100.0
            last_cvd = float(cvd_series.iloc[-1]) if not pd.isna(cvd_series.iloc[-1]) else 0.0
            
            if last_var_pctile <= WYCKOFF_PHASE_B_VAR_PERCENTILE and last_cvd > 0.0:
                return WaitEntry(
                    ticker=ticker,
                    condition="wyckoff_phase_b",
                    details={
                        "var_percentile": round(last_var_pctile, 2),
                        "cvd": round(last_cvd, 2),
                        "price": round(float(df["Close"].iloc[-1]), 2),
                    },
                )

        # Check 4: Smart Money Proxy / VSA (Squat Candles)
        # Volume > 200% of 20-day average, Efficiency Ratio in bottom 10th percentile
        if len(df) >= max(VSA_SQUAT_PERCENTILE_LOOKBACK * 2, 60):
            vol_ratio = volume_ratio(df, period=20)
            eff_ratio = efficiency_ratio(df)
            eff_pctile = rolling_percentile(eff_ratio, window=VSA_SQUAT_PERCENTILE_LOOKBACK)
            
            last_vol_ratio = float(vol_ratio.iloc[-1]) if not pd.isna(vol_ratio.iloc[-1]) else 0.0
            last_eff_pctile = float(eff_pctile.iloc[-1]) if not pd.isna(eff_pctile.iloc[-1]) else 100.0
            
            if last_vol_ratio > VSA_SQUAT_VOL_RATIO and last_eff_pctile <= VSA_SQUAT_EFFICIENCY_PERCENTILE:
                return WaitEntry(
                    ticker=ticker,
                    condition="vsa_squat_candle",
                    details={
                        "volume_ratio": round(last_vol_ratio, 2),
                        "efficiency_percentile": round(last_eff_pctile, 2),
                        "price": round(float(df["Close"].iloc[-1]), 2),
                    },
                )

        return None

    def _check_fvg_approach(
        self, df: pd.DataFrame, ticker: str
    ) -> WaitEntry | None:
        """
        Check if the current price is approaching an unfilled bullish FVG.

        "Approaching" = current price is within 1x ATR of the FVG's
        upper boundary (gap_high), and price is above the FVG zone.
        """
        fvgs = detect_fvg(df)

        # Only consider unfilled bullish FVGs from the last 60 bars
        recent_bullish = [
            g for g in fvgs
            if g.gap_type == "bullish"
            and not g.filled
            and (df.index[-1] - g.date).days <= 90  # ~60 trading days
        ]

        if not recent_bullish:
            return None

        atr_series = atr(df, period=ATR_PERIOD)
        current_atr = atr_series.iloc[-1]
        last_close = float(df["Close"].iloc[-1])

        if pd.isna(current_atr) or current_atr <= 0:
            return None

        for gap in recent_bullish:
            # Price is above the FVG zone and within ATR proximity of dropping into it
            distance_to_gap = last_close - gap.gap_high
            if 0 < distance_to_gap <= (FVG_ATR_PROXIMITY * current_atr):
                return WaitEntry(
                    ticker=ticker,
                    condition="fvg_approach",
                    details={
                        "fvg_date": gap.date.strftime("%Y-%m-%d"),
                        "gap_high": round(gap.gap_high, 2),
                        "gap_low": round(gap.gap_low, 2),
                        "gap_size": round(gap.gap_size, 2),
                        "distance": round(distance_to_gap, 2),
                        "atr": round(float(current_atr), 2),
                        "price": round(last_close, 2),
                    },
                )

        return None

    # ══════════════════════════════════════════════════════════════════════
    # TRADE BUCKET (Phase 3 Engine Integration)
    # ══════════════════════════════════════════════════════════════════════

    def _run_engines(
        self,
        df: pd.DataFrame,
        ticker: str,
        regime: RegimeSnapshot,
    ) -> TradeEntry | None:
        """
        Run all Phase 3 entry engines against a single stock.

        Engines (evaluated in priority order):
        1. FVG Pullback (priority 3) -- BULL/CAUTION only
        2. Momentum Breakout (priority 2) -- BULL only
        3. Buying on Weakness (priority 1) -- any regime

        Returns the highest-priority signal as a TradeEntry,
        enriched with Phase 4 risk details.
        """
        signals = run_all_engines(df, ticker, regime)

        if not signals:
            return None

        # Pick the highest-priority signal (ties broken by score)
        best = max(signals, key=lambda s: (s.priority, s.score))

        from core.predictor import SyntheticFlowPredictor, VolatilityProjector
        from core.indicators import closing_range

        # Phase 5: Predictive Veto
        predicted_return = SyntheticFlowPredictor().predict_next_return(df)
        if predicted_return is not None and predicted_return < 0:
            logger.debug("[%s] VETOED %s: Predictor forecasts negative return (%.2f%%)", ticker, best.engine, predicted_return * 100)
            return None

        # Enrich with risk details
        details = dict(best.details)
        
        # Phase 5 Enriched Details
        if predicted_return is not None:
            details["predicted_return"] = round(predicted_return, 4)
            
        boundaries = VolatilityProjector.project(df)
        if boundaries:
            details["projected_upper"] = round(boundaries[0], 2)
            details["projected_lower"] = round(boundaries[1], 2)
            
        cr_series = closing_range(df)
        last_cr = float(cr_series.iloc[-1]) if not pd.isna(cr_series.iloc[-1]) else 0.5
        details["closing_range"] = round(last_cr, 2)

        try:
            from config.settings import ATR_PERIOD, DEFAULT_CAPITAL
            atr_series = atr(df, period=ATR_PERIOD)
            last_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0

            if last_atr > 0:
                risk = self._risk_manager.calculate_trade_risk(
                    ticker=ticker,
                    entry_price=best.price,
                    atr_value=last_atr,
                    capital=DEFAULT_CAPITAL,
                    regime=regime.regime.value,
                )
                details["stop_loss"] = risk.stop_loss
                details["trailing_stop"] = risk.trailing_stop
                details["position_size"] = risk.position_size
                details["risk_amount"] = risk.risk_amount
                details["risk_pct"] = risk.risk_pct
                details["regime_risk_pct"] = risk.regime_adjusted_risk_pct
        except Exception as e:
            logger.debug("[%s] Could not calculate risk: %s", ticker, e)

        return TradeEntry(
            ticker=ticker,
            signal=best.engine,
            score=best.score,
            price=best.price,
            details=details,
        )
