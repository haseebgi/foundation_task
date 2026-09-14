"""
STEP 1: DATA COLLECTION
------------------------
Ye script Yahoo Finance se historical stock data download karta hai
aur har stock ke liye alag CSV file 'data/' folder mein save karta hai.

Columns jo milte hain:
Date, Open, High, Low, Close, Volume
"""

import yfinance as yf
import pandas as pd
import os

# -----------------------------------------------------
# Yahan apni pasand ke stock symbols likh sakte hain.
# Global stocks ke liye normal symbol (AAPL, TSLA, MSFT)
# PSX ke liye ".KA" suffix try kar sakte hain (e.g. "LUCK.KA")
# -----------------------------------------------------
STOCK_LIST = ["AAPL", "TSLA", "MSFT", "AMZN", "KO"]  # 5 real companies:
# AAPL = Apple, TSLA = Tesla, MSFT = Microsoft, AMZN = Amazon, KO = Coca-Cola
# Ye sab yfinance par 100% reliably kaam karte hain (verified, actively traded).
#
# NOTE on PSX (Pakistan Stock Exchange): PSX individual stocks ka yfinance
# support unreliable hai - koi consistent ticker format confirm nahi hota.
# Agar PSX data zaroori ho, to PSX ki official website (dps.psx.com.pk) se
# manually CSV export karna behtar option hai, phir usay 'data/' folder mein
# isi format (Date, Open, High, Low, Close, Volume, Symbol) mein daal dein -
# baaki pipeline (Step 2 se aage) automatically kaam kar jayega.

START_DATE = "2015-01-01"
END_DATE = "2025-01-01"

OUTPUT_DIR = "data"


def download_stock_data(symbol: str) -> pd.DataFrame:
    """Ek stock ka historical data download karta hai."""
    print(f"Downloading data for {symbol} ...")
    df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)

    if df.empty:
        print(f"  -> WARNING: {symbol} ke liye koi data nahi mila.")
        return None

    # Multi-index columns ko flatten karna (yfinance kabhi kabhi
    # (Close, AAPL) jaisa column deta hai)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Symbol"] = symbol
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []

    for symbol in STOCK_LIST:
        df = download_stock_data(symbol)
        if df is not None:
            save_path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
            df.to_csv(save_path, index=False)
            print(f"  -> Saved: {save_path}  ({len(df)} rows)")
            all_data.append(df)

    # Sab stocks ka combined file bhi bana dete hain (aage useful hoga)
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, "all_stocks_raw.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined raw file saved: {combined_path}  ({len(combined)} total rows)")
    else:
        print("Koi data download nahi hua. Internet connection check karein.")


if __name__ == "__main__":
    main()
