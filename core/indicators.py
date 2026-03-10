"""
Reusable technical indicator calculations for IHSG analysis.

All functions operate on Pandas Series or DataFrames and return
new columns/values — they never mutate the input. These are the
building blocks used by the scanner, engines, and backtester.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class FairValueGap:
    """Represents a single Fair Value Gap identified in price data."""

    date: pd.Timestamp
    gap_type: str  # "bullish" or "bearish"
    gap_high: float  # upper boundary of the gap
    gap_low: float  # lower boundary of the gap
    filled: bool = False  # whether price has revisited the gap

    @property
    def gap_size(self) -> float:
        """Absolute size of the gap in price units."""
        return abs(self.gap_high - self.gap_low)

    @property
    def midpoint(self) -> float:
        """Midpoint of the gap zone."""
        return (self.gap_high + self.gap_low) / 2.0


# ─── Moving Averages ─────────────────────────────────────────────────────────


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Parameters
    ----------
    series : pd.Series
        Price series (typically Close).
    period : int
        Lookback window in bars.

    Returns
    -------
    pd.Series
        SMA values, with NaN for the first (period - 1) bars.
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Parameters
    ----------
    series : pd.Series
        Price series.
    period : int
        Span for the EMA calculation.

    Returns
    -------
    pd.Series
        EMA values.
    """
    return series.ewm(span=period, adjust=False).mean()


# ─── Volatility ──────────────────────────────────────────────────────────────


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range — measures daily volatility.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: High, Low, Close.
    period : int
        Smoothing period (default 14).

    Returns
    -------
    pd.Series
        ATR values.
    """
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (upper_band, middle_band, lower_band)
    """
    middle = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std()
    upper = middle + (num_std * rolling_std)
    lower = middle - (num_std * rolling_std)
    return upper, middle, lower


# ─── Momentum Oscillators ────────────────────────────────────────────────────


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Uses the Wilder smoothing method (exponential moving average).

    Parameters
    ----------
    series : pd.Series
        Price series (typically Close).
    period : int
        RSI lookback period (default 14).

    Returns
    -------
    pd.Series
        RSI values (0–100).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing (equivalent to EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ─── Volume & Liquidity ──────────────────────────────────────────────────────


def adtv(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Average Daily Trading Value (Close x Volume).

    This is the primary liquidity metric for the IDX market.
    Stocks with ADTV < IDR 10B are considered illiquid.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Close, Volume.
    period : int
        Lookback window (default 20 trading days).

    Returns
    -------
    pd.Series
        ADTV values in IDR.
    """
    daily_value = df["Close"] * df["Volume"]
    return daily_value.rolling(window=period, min_periods=max(1, period // 2)).mean()


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Current volume as a ratio of the N-day average volume.

    A ratio > 1.5 indicates significantly above-average volume.

    Returns
    -------
    pd.Series
        Volume ratio values (1.0 = average).
    """
    avg_vol = df["Volume"].rolling(window=period, min_periods=max(1, period // 2)).mean()
    return df["Volume"] / avg_vol.replace(0, np.nan)


# ─── Price Structure ─────────────────────────────────────────────────────────


def detect_fvg(df: pd.DataFrame) -> list[FairValueGap]:
    """
    Detect all Fair Value Gaps in the price data.

    A **bullish FVG** exists when candle[i]'s Low > candle[i-2]'s High,
    creating an unfilled gap between the two candles (candle[i-1] is the
    "impulse" candle that caused the gap).

    A **bearish FVG** exists when candle[i]'s High < candle[i-2]'s Low.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close with DatetimeIndex.

    Returns
    -------
    list[FairValueGap]
        All detected FVGs, newest first.
    """
    gaps: list[FairValueGap] = []

    if len(df) < 3:
        return gaps

    highs = df["High"].values
    lows = df["Low"].values
    dates = df.index

    for i in range(2, len(df)):
        # Bullish FVG: candle[i] low > candle[i-2] high
        if lows[i] > highs[i - 2]:
            gaps.append(
                FairValueGap(
                    date=dates[i],
                    gap_type="bullish",
                    gap_high=float(lows[i]),    # upper edge = current candle's low
                    gap_low=float(highs[i - 2]),  # lower edge = 2-back candle's high
                )
            )

        # Bearish FVG: candle[i] high < candle[i-2] low
        elif highs[i] < lows[i - 2]:
            gaps.append(
                FairValueGap(
                    date=dates[i],
                    gap_type="bearish",
                    gap_high=float(lows[i - 2]),  # upper edge = 2-back candle's low
                    gap_low=float(highs[i]),       # lower edge = current candle's high
                )
            )

    # Mark FVGs as filled if price has revisited the zone
    closes = df["Close"].values
    for gap in gaps:
        gap_idx = df.index.get_loc(gap.date)
        if gap_idx + 1 < len(df):
            subsequent_lows = lows[gap_idx + 1:]
            subsequent_highs = highs[gap_idx + 1:]

            if gap.gap_type == "bullish":
                # Filled if price drops into the gap zone
                if len(subsequent_lows) > 0 and np.min(subsequent_lows) <= gap.gap_high:
                    gap.filled = True
            else:
                # Filled if price rises into the gap zone
                if len(subsequent_highs) > 0 and np.max(subsequent_highs) >= gap.gap_low:
                    gap.filled = True

    # Return newest first
    gaps.reverse()
    return gaps


def is_tight_consolidation(
    df: pd.DataFrame,
    window: int = 14,
    max_range_pct: float = 10.0,
) -> tuple[bool, float]:
    """
    Check if the stock is in a tight consolidation pattern.

    A stock is consolidating when the price range (highest high - lowest low)
    over the lookback window is less than max_range_pct of the current price.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: High, Low, Close.
    window : int
        Number of trading days to check (default 14).
    max_range_pct : float
        Maximum range as a percentage of price (default 10%).

    Returns
    -------
    tuple[bool, float]
        (is_consolidating, range_pct)
    """
    if len(df) < window:
        return False, 0.0

    recent = df.tail(window)
    highest = recent["High"].max()
    lowest = recent["Low"].min()
    last_close = df["Close"].iloc[-1]

    if last_close <= 0:
        return False, 0.0

    range_pct = ((highest - lowest) / last_close) * 100.0
    return range_pct <= max_range_pct, float(range_pct)


def stoch_rsi(
    series: pd.Series,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic RSI (%K and %D).

    Parameters
    ----------
    series : pd.Series
        Price series (usually Close).
    period : int
        RSI and Stochastic lookback period.
    smooth_k : int
        Periods to smooth %K (usually 3).
    smooth_d : int
        Periods to smooth %D (usually 3).

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (stoch_rsi_k, stoch_rsi_d) as 0-100 scale series.
    """
    # 1. Calculate normal RSI
    rsi_vals = rsi(series, period)

    # 2. Calculate Stochastic of the RSI
    # StochRSI = (RSI - min(RSI)) / (max(RSI) - min(RSI))
    rolling_min = rsi_vals.rolling(window=period).min()
    rolling_max = rsi_vals.rolling(window=period).max()
    
    # Avoid division by zero
    stoch_rsi_raw = (rsi_vals - rolling_min) / (rolling_max - rolling_min).replace(0, 1e-10)
    
    # Scale to 0-100
    stoch_rsi_raw *= 100.0

    # 3. Apply smoothing for %K and %D
    stoch_k = stoch_rsi_raw.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=smooth_d).mean()

    return stoch_k, stoch_d


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD).

    Parameters
    ----------
    series : pd.Series
        Price series (usually Close).
    fast : int
        Fast EMA period.
    slow : int
        Slow EMA period.
    signal : int
        Signal line EMA period.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (macd_line, signal_line, macd_histogram)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    macd_hist = macd_line - signal_line

    return macd_line, signal_line, macd_hist


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate the rolling rank percentile of the current value within its historical window.
    
    Returns 0-100 scale.
    """
    # Uses Pandas built-in rolling rank with pct=True
    return series.rolling(window=window).rank(pct=True) * 100.0


def cvd(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative Volume Delta (CVD).
    
    Positive volume when close > prev_close, negative when close < prev_close.
    """
    prev_close = df["Close"].shift(1)
    direction = np.sign(df["Close"] - prev_close)
    vol_delta = df["Volume"] * direction
    return vol_delta.cumsum()


def efficiency_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Efficiency Ratio = (High - Low) / Volume
    
    Measures how much volume is required to move the price.
    """
    hl_spread = df["High"] - df["Low"]
    vol = df["Volume"].replace(0, np.nan)
    return hl_spread / vol


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate the Hurst Exponent of a time series.
    
    H ~ 0.5: Random Walk
    H < 0.5: Mean Reverting
    H > 0.5: Trending
    """
    vals = series.values
    if len(vals) < max_lag * 2:
        return 0.5  # default to random walk if not enough data
        
    lags = range(2, max_lag)
    
    # Calculate the standard deviation of the difference
    tau = []
    for lag in lags:
        diff = np.subtract(vals[lag:], vals[:-lag])
        std = np.std(diff)
        tau.append(np.sqrt(std) if std > 0 else 1e-8)
        
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0] * 2.0


def closing_range(df: pd.DataFrame) -> pd.Series:
    """Closing Range = (Close - Low) / (High - Low)"""
    hl_range = (df["High"] - df["Low"]).replace(0, np.nan)
    return (df["Close"] - df["Low"]) / hl_range


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow"""
    mfv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mf_vol = mfv * df["Volume"]
    return mf_vol.rolling(window=period).sum() / df["Volume"].rolling(window=period).sum()


def vpt(df: pd.DataFrame) -> pd.Series:
    """Volume Price Trend"""
    prev_close = df["Close"].shift(1).replace(0, np.nan)
    vpt_change = df["Volume"] * (df["Close"] - prev_close) / prev_close
    return vpt_change.cumsum()


def roc(series: pd.Series, period: int = 3) -> pd.Series:
    """Rate of Change"""
    prev = series.shift(period).replace(0, np.nan)
    return (series - prev) / prev
