import sys
import logging
from pathlib import Path
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.scanner import MasterScanner
from core.database import ParquetStore
from core.regime import MarketRegime

def setup_logging():
    logging.basicConfig(level=logging.DEBUG)

def main():
    setup_logging()
    
    # 1. Fetch test data and store it
    store = ParquetStore()
    ticker = "BBCA.JK"
    print(f"Fetching 1y data for {ticker}...")
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    
    # Ensure MultiIndex columns are flattened for scanner logic
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Needs Date in DatetimeIndex
    store.save("BBCA", df)
    
    # 2. Test Regime Filter (Hurst)
    print("Testing Regime filter...")
    regime = MarketRegime(period="1y")
    snapshot = regime.get_snapshot()
    print("Regime Snapshot:")
    print(snapshot)
    
    # 3. Test Master Scanner
    print("Testing Master Scanner...")
    scanner = MasterScanner(store)
    
    # Force run check
    result = scanner.scan_universe(["BBCA"], check_earnings=False, regime=snapshot)
    
    print("\nScan Results:")
    print("Avoid:", [a.reason for a in result.avoid])
    print("Wait:", [(w.condition, w.details) for w in result.wait])
    print("Trade:", [(t.signal, t.score, t.details) for t in result.trade])
    print("Stats:", result.stats)

if __name__ == "__main__":
    main()
