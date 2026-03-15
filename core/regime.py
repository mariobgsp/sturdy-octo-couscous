"""
IHSG Composite Market Regime Filter.

Determines the overall market environment by analyzing the IHSG
composite index (^JKSE). The regime classification acts as a
master switch that gates which entry engines may fire.

Regime classifications:
  BULL    — Close > SMA(50) > SMA(200) — all engines active
  CAUTION — Close > SMA(200) but not a full bull — FVG + B.O.W. only
  BEAR    — Close < SMA(200) — only B.O.W. engine active
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import yfinance as yf

from config.settings import (
    IHSG_COMPOSITE_TICKER,
    REGIME_ATR_PERIOD,
    REGIME_SMA_LONG,
    REGIME_SMA_SHORT,
)
from core.indicators import atr, sma, hurst_exponent

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime classifications."""

    BULL = "BULL"
    CAUTION = "CAUTION"
    BEAR = "BEAR"


# Which engines are allowed in each regime
_ENGINE_PERMISSIONS: dict[RegimeType, set[str]] = {
    RegimeType.BULL: {"fvg_pullback", "momentum_breakout", "volume_climax_reversal"},
    RegimeType.CAUTION: {"wyckoff_spring", "volume_climax_reversal"},
    RegimeType.BEAR: set(),
}


@dataclass
class RegimeSnapshot:
    """Immutable snapshot of the current market regime state."""

    regime: RegimeType
    close: float
    sma_short: float
    sma_long: float
    atr_value: float
    hurst_value: float
    as_of_date: str

    def allows_engine(self, engine_name: str) -> bool:
        """Check if a specific engine is permitted under this regime."""
        return engine_name in _ENGINE_PERMISSIONS.get(self.regime, set())

    def __str__(self) -> str:
        return (
            f"Regime: {self.regime.value} | "
            f"Close: {self.close:,.0f} | "
            f"SMA({REGIME_SMA_SHORT}): {self.sma_short:,.0f} | "
            f"SMA({REGIME_SMA_LONG}): {self.sma_long:,.0f} | "
            f"ATR({REGIME_ATR_PERIOD}): {self.atr_value:,.0f} | "
            f"Hurst(100): {self.hurst_value:.2f} | "
            f"As-of: {self.as_of_date}"
        )


class MarketRegime:
    """
    Fetches IHSG composite data and classifies the market regime.

    This class makes a single yfinance call to download recent
    ^JKSE data, then computes SMA(50), SMA(200), and ATR(14)
    to determine the current regime state.

    Usage:
        regime = MarketRegime()
        snapshot = regime.get_snapshot()
        print(snapshot)
        if snapshot.allows_engine("fvg_pullback"):
            ...
    """

    def __init__(self, period: str = "1y") -> None:
        """
        Initialize and fetch IHSG composite data.

        Parameters
        ----------
        period : str
            yfinance period to download (default '1y').
            Must be long enough for SMA(200) — '1y' provides ~250 bars.
        """
        self._df: pd.DataFrame | None = None
        self._snapshot: RegimeSnapshot | None = None
        self._fetch(period)

    def _fetch(self, period: str) -> None:
        """Download ^JKSE data and compute regime indicators."""
        try:
            logger.info(
                "Fetching IHSG composite (%s) for regime analysis...",
                IHSG_COMPOSITE_TICKER,
            )
            raw = yf.download(
                IHSG_COMPOSITE_TICKER,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=True,
                timeout=30,
            )

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            if raw.empty or len(raw) < REGIME_SMA_LONG:
                logger.error(
                    "Insufficient IHSG data: got %d bars, need %d for SMA(%d).",
                    len(raw), REGIME_SMA_LONG, REGIME_SMA_LONG,
                )
                # Fallback to CAUTION if data is insufficient
                self._snapshot = RegimeSnapshot(
                    regime=RegimeType.CAUTION,
                    close=0, sma_short=0, sma_long=0, atr_value=0, hurst_value=0.5,
                    as_of_date="N/A (insufficient data)",
                )
                return

            self._df = raw

            # Compute indicators
            sma_short = sma(raw["Close"], REGIME_SMA_SHORT)
            sma_long = sma(raw["Close"], REGIME_SMA_LONG)
            atr_series = atr(raw, REGIME_ATR_PERIOD)

            last_close = float(raw["Close"].iloc[-1])
            last_sma_short = float(sma_short.iloc[-1])
            last_sma_long = float(sma_long.iloc[-1])
            last_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
            as_of = raw.index[-1].strftime("%Y-%m-%d")

            # Calculate Hurst exponent on last 100 days
            if len(raw) >= 100:
                hurst_val = hurst_exponent(raw["Close"].tail(100), max_lag=20)
            else:
                hurst_val = 0.5
                
            # Classify regime based on Hurst Exponent
            if 0.45 <= hurst_val <= 0.55:
                regime = RegimeType.BEAR
            elif hurst_val < 0.45:
                regime = RegimeType.CAUTION
            else:
                regime = RegimeType.BULL

            self._snapshot = RegimeSnapshot(
                regime=regime,
                close=round(last_close, 2),
                sma_short=round(last_sma_short, 2),
                sma_long=round(last_sma_long, 2),
                atr_value=round(last_atr, 2),
                hurst_value=round(hurst_val, 2),
                as_of_date=as_of,
            )

            logger.info("Market regime: %s", self._snapshot)

        except Exception as e:
            logger.error("Failed to fetch IHSG composite: %s", e)
            # Fallback to CAUTION on error — conservative but not fully frozen
            self._snapshot = RegimeSnapshot(
                regime=RegimeType.CAUTION,
                close=0, sma_short=0, sma_long=0, atr_value=0, hurst_value=0.5,
                as_of_date=f"ERROR: {e}",
            )

    def get_snapshot(self) -> RegimeSnapshot:
        """Return the current regime snapshot."""
        assert self._snapshot is not None, "Regime not initialized"
        return self._snapshot

    @property
    def status(self) -> RegimeType:
        """Shortcut for the current regime classification."""
        return self.get_snapshot().regime
