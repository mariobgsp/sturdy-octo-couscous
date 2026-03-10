"""
Global configuration constants for the IHSG Swing Trading Application.

All tunable parameters are centralized here to avoid magic numbers
scattered across the codebase. Paths are resolved relative to the
project root so the app works regardless of the working directory.
"""

from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ohlcv"
LOG_DIR = PROJECT_ROOT / "logs"

# ─── Data Ingestion ───────────────────────────────────────────────────────────

# Default historical period to download (5 years for backtesting headroom)
DEFAULT_PERIOD: str = "5y"

# Default data interval
DEFAULT_INTERVAL: str = "1d"

# ─── Rate Limiting (Yahoo Finance) ────────────────────────────────────────────

# Number of tickers to download before inserting a batch pause
RATE_LIMIT_BATCH_SIZE: int = 50

# Seconds to pause after each batch
RATE_LIMIT_PAUSE_SECONDS: float = 5.0

# Seconds to pause between individual ticker downloads
INTER_REQUEST_DELAY: float = 2.0

# Maximum retry attempts on transient errors (429, timeout, etc.)
MAX_RETRIES: int = 3

# Base wait time (seconds) for exponential backoff on retries
RETRY_BASE_WAIT: float = 30.0

# ─── Data Cleaning ────────────────────────────────────────────────────────────

# Volume spikes beyond this many standard deviations are capped
VOLUME_SPIKE_STD_THRESHOLD: float = 5.0

# Rolling window (trading days) for volume anomaly detection
VOLUME_ROLLING_WINDOW: int = 60

# Maximum consecutive missing trading days to forward-fill
MAX_FORWARD_FILL_DAYS: int = 5

# Minimum price change ratio to flag a potential stock split
# e.g., 0.40 means a 40%+ single-day move triggers split detection
SPLIT_DETECTION_THRESHOLD: float = 0.40

# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ─── Phase 2: Scanner Thresholds ─────────────────────────────────────────────

# Minimum Average Daily Trading Value (IDR) to survive the Avoid filter
ADTV_MIN_IDR: float = 10_000_000_000  # IDR 10 Billion

# ADTV lookback period in trading days
ADTV_LOOKBACK: int = 20

# Minimum stock price (IDR) — below this is considered a penny stock
PENNY_STOCK_THRESHOLD: float = 100.0

# SMA lookback for trend filter
SMA_200_LOOKBACK: int = 200

# Tight consolidation detection
CONSOLIDATION_WINDOW: int = 14  # trading days
CONSOLIDATION_MAX_RANGE_PCT: float = 10.0  # max High-Low range as % of price

# Fair Value Gap — how close price must be to an FVG zone (in ATR multiples)
FVG_ATR_PROXIMITY: float = 1.0

# ATR period used across the scanner
ATR_PERIOD: int = 14

# Maximum number of picks in the Trade bucket
TRADE_BUCKET_MAX_PICKS: int = 5

# Earnings proximity window (hours) for the Avoid filter
EARNINGS_PROXIMITY_HOURS: int = 48

# ─── Phase 3: Market Regime & Entry Engines ──────────────────────────────────

# IHSG composite index ticker on Yahoo Finance
IHSG_COMPOSITE_TICKER: str = "^JKSE"

# Regime filter SMA periods
REGIME_SMA_SHORT: int = 50
REGIME_SMA_LONG: int = 200
REGIME_ATR_PERIOD: int = 14

# Engine 1: FVG Pullback
FVG_LOW_VOLUME_RATIO: float = 0.8  # volume must be < 80% of avg for valid pullback

# Engine 2: Momentum Breakout
BREAKOUT_CONSOLIDATION_DAYS: int = 20
BREAKOUT_MAX_SPREAD_PCT: float = 5.0  # max price spread over consolidation window
BREAKOUT_VOLUME_THRESHOLD: float = 1.5  # volume must be > 150% of average

# Engine 3: Buying on Weakness (B.O.W.)
BOW_RSI_THRESHOLD: float = 25.0  # RSI below this = extreme capitulation
BOW_BOLLINGER_PERIOD: int = 20
BOW_BOLLINGER_STD: float = 2.0
BOW_VOLUME_CLIMAX_RATIO: float = 2.0  # volume must be > 200% of average

# New B.O.W. Alternative Validation Parameters
BOW_STOCH_RSI_PERIOD: int = 14
BOW_STOCH_RSI_OVERSOLD: float = 20.0
BOW_MACD_FAST: int = 12
BOW_MACD_SLOW: int = 26
BOW_MACD_SIGNAL: int = 9
BOW_MACD_DIVERGENCE_LOOKBACK: int = 20  # Lookback period for price lower low

# Engine 4: Wyckoff Phase C Spring
WYCKOFF_SPRING_LOOKBACK: int = 60
WYCKOFF_SPRING_VOLUME_RATIO: float = 2.0

# Scanner Wait: Wyckoff Phase B Accumulation
WYCKOFF_PHASE_B_VAR_LOOKBACK: int = 60
WYCKOFF_PHASE_B_VAR_PERCENTILE: float = 10.0

# Scanner Wait: VSA Squat Candle
VSA_SQUAT_VOL_RATIO: float = 2.0
VSA_SQUAT_EFFICIENCY_PERCENTILE: float = 10.0
VSA_SQUAT_PERCENTILE_LOOKBACK: int = 60

# Predictor: Ridge Regression
RIDGE_CV_SPLITS: int = 5
RIDGE_LOOKAHEAD: int = 5
RIDGE_TRAIN_WINDOW: int = 252 # 1 year data roughly


# ─── Phase 4: Risk Management ────────────────────────────────────────────────

# Default starting capital (IDR)
DEFAULT_CAPITAL: float = 100_000_000  # IDR 100 Million

# Maximum risk per trade as % of total capital
MAX_RISK_PER_TRADE_PCT: float = 2.0

# Initial stop-loss distance in ATR multiples below entry
STOP_LOSS_ATR_MULTIPLIER: float = 1.5

# Chandelier trailing stop in ATR multiples from highest high
TRAILING_STOP_ATR_MULTIPLIER: float = 2.0
TRAILING_STOP_ATR_PERIOD: int = 14

# Maximum total portfolio heat (sum of all open risks as % of capital)
MAX_PORTFOLIO_HEAT_PCT: float = 6.0

# Maximum number of simultaneous open positions
MAX_OPEN_POSITIONS: int = 5

# Regime-adjusted risk multipliers (applied to MAX_RISK_PER_TRADE_PCT)
REGIME_RISK_MULTIPLIER: dict[str, float] = {
    "BULL": 1.0,      # full 2%
    "CAUTION": 0.5,   # 1%
    "BEAR": 0.25,     # 0.5%
}

# IDX lot size (shares must be bought in multiples of this)
IDX_LOT_SIZE: int = 100

# ─── Phase 6: Backtesting ─────────────────────────────────────────────────────

BACKTEST_YEARS: int = 5
BACKTEST_TRAIN_YEARS: float = 3.5
BACKTEST_SLIPPAGE_PCT: float = 0.15     # 0.15% per side
BACKTEST_FEE_BUY_PCT: float = 0.15      # IDX broker buy fee
BACKTEST_FEE_SELL_PCT: float = 0.25     # IDX broker sell fee (includes tax)
BACKTEST_INITIAL_CAPITAL: float = 5_000_000
MAX_DRAWDOWN_THRESHOLD: float = 15.0    # Report card red flag
MIN_PROFIT_FACTOR: float = 1.5          # Report card threshold

# ─── Phase 7: Live Execution ──────────────────────────────────────────────────

FAT_FINGER_MAX_SHARES: int = 1000           # Hard limit per order
FAT_FINGER_MAX_VALUE_IDR: float = 2_500_000  # Hard limit per order value
DAILY_DRAWDOWN_HALT_PCT: float = 3.0          # Halt trading if account drops 3%
EXECUTION_SCHEDULE_WIB: str = "15:50"         # Generate execution list at this time
BRACKET_ORDER_TP_ATR_MULTIPLIER: float = 3.0  # Take-profit at 3x ATR from entry

