# IHSG Swing Trading System (v2.0.0)

A highly rigorous, accuracy-first swing trading application specifically designed for the Indonesian Stock Exchange (IHSG).

This system automates daily data ingestion, technical scanning, market regime detection, entry signal generation, risk management, backtesting, and live execution through a structured, 7-phase architecture.

## 🏗️ Architecture

1. **Infrastructure & Data Engineering (`core/ingestion.py`, `core/data_cleaner.py`)**
   - Ingests daily OHLCV data for 591 curated IHSG tickers.
   - Handles splits, NaN gaps, and extreme volume spikes.
   - Stores data locally in optimized Parquet format (`data/ohlcv/`).
   - Rate-limited `yfinance` integration to prevent IP bans.

2. **Master Scanner Engine (`core/scanner.py`)**
   - Scans the universe of stocks through a strict "Tri-Bucket" triage pipeline:
     - **Avoid Bucket**: Filters out illiquid stocks (< IDR 2B ADTV), penny stocks (< IDR 50), stocks below their SMA(200), and stocks reporting earnings within 48 hours.
     - **Wait Bucket**: Parks stocks setting up for a trade (e.g., tight consolidation, approaching Fair Value Gaps, or overflow signals).
     - **Trade Bucket**: The highest confidence, ready-to-execute signals (max 5 per day).

3. **Market Regime & Entry Engines (`core/regime.py`, `core/engines.py`)**
   - **Regime Filter**: Classifies the broader IHSG composite (`^JKSE`) as `BULL`, `CAUTION`, or `BEAR` based on its position relative to SMA(50) and SMA(200). Acts as a master permission switch.
   - **Entry Engines**:
     - _Priority 1: Buying on Weakness (B.O.W.)_ — Triggers on extreme capitulation (RSI < 25, below lower Bollinger Band, StochRSI cross, or MACD Divergence) followed by a massive volume influx (>200%). Active in all regimes.
     - _Priority 2: Momentum Breakout_ — Triggers on a high-volume breakout from a 20-day tight consolidation. Active in BULL only.
     - _Priority 3: FVG Pullback_ — Triggers on a low-volume pullback into an unfilled bullish Fair Value Gap. Active in BULL and CAUTION.

4. **Risk Management Engine (`core/risk.py`, `core/portfolio.py`)**
   - Deterministic ATR-based mathematics.
   - Calculates Stop-Loss (1.5x ATR) and Chandelier Trailing Stops (2.0x ATR).
   - Dynamic position sizing forcing equal risk across trades (default 2% of capital), rounding to IDX lot sizes (100 shares).
   - **Portfolio Heat tracking**: Caps maximum simultaneous open risk at 6% of total capital.
   - Regime-adjusted risk: Halves position sizes in CAUTION (1%), quarters them in BEAR (0.5%).

5. **Daily Output Dashboard (`core/report.py`, `scripts/daily.py`, `core/alerts.py`)**
   - Unified daily workflow combining all the above into a single execution step.
   - Generates formatted console reports and a dark-themed `.html` dashboard showing portfolio heat and actionable Trade Cards.
   - CRITICAL-level alert logging for new signals.

6. **Backtesting Engine (`core/backtester.py`, `core/backtest_report.py`)**
   - Event-driven replay through the same scanner + engines + risk manager. Zero look-ahead bias.
   - Automatic 3.5-year Training / 1.5-year Blind Test split.
   - Hardcoded slippage (0.15% per side) and IDX broker fees (0.15% buy, 0.25% sell).
   - **Report Card**: Win Rate, Expected Value, Max Drawdown (⚠️ >15%), Sharpe Ratio, Profit Factor (⚠️ <1.5), Win/Loss Streaks.

7. **Live Execution & Failsafes (`core/broker.py`, `core/failsafes.py`, `core/bracket_order.py`)**
   - Abstract broker adapter with `SimulatedBroker` for testing.
   - **Fat Finger Guard**: Hard limit of 50,000 shares / IDR 50M per order.
   - **Daily Drawdown Breaker**: Halts all trading if the account drops 3% in a single day.
   - **Bracket Orders**: Sends Buy + Stop-Loss + Take-Profit (3x ATR) simultaneously.
   - Cron-ready execution CLI scheduled for 15:50 WIB.

## 🚀 Usage

### 1. Unified Daily Workflow (Start Here)

```powershell
python -m scripts.daily
```

_Options:_

- `--tickers ASII BBCA TLKM` : Scan specific tickers only.
- `--check-earnings` : Fetch upcoming earnings dates (slower).
- `--no-html` : Skip generating the HTML dashboard.
- `--capital 200000000` : Custom portfolio capital.

### 2. Backtesting

```powershell
python -m scripts.backtest
python -m scripts.backtest --tickers BBCA BBRI ASII TLKM UNVR
python -m scripts.backtest --capital 200000000
```

### 3. Live Execution

```powershell
python -m scripts.execute --dry-run    # Simulate without placing orders
python -m scripts.execute              # Live mode (requires broker setup)
```

### 4. Individual Subsystems

```powershell
python -m scripts.ingest               # Phase 1: Ingest OHLCV data
python -m scripts.scan                 # Phase 2: Raw scanner
python -m scripts.regime               # Phase 3: Market regime check
python -m scripts.risk --ticker ASII   # Phase 4: Risk calculator
```

## 🛠️ GitHub Actions Automation

The system is configured to run entirely hands-off via GitHub Actions:

- **Schedule**: Automatically runs the `daily.py` workflow at 09:30 WIB every weekday.
- **Caching**: Parquet OHLCV data is cached across runs to vastly speed up ingestion.
- **Artifacts**: HTML reports and Alert Logs are uploaded as workflow artifacts (30-day retention).
- **Persistence**: Portfolio state (`data/portfolio.json`) is automatically committed back to the repository.

To enable, simply push code to the `main` branch. You can also trigger a manual run with custom tickers from the "Actions" tab.

## 📦 Requirements

- Python 3.12+
- `pandas >= 2.0`
- `numpy >= 1.24`
- `yfinance >= 0.2.36`
- `pyarrow >= 14.0` (for Parquet storage)

Install via:

```powershell
pip install -r requirements.txt
```
