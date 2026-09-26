import pandas as pd
import psxdata

tickers = ['LUCK', 'OGDC', 'PPL', 'ENGRO', 'HUBC', 'HBL', 'MCB', 'UBL', 'TRG', 'SYS']

print("[INFO] Fetching real historical stock data from PSX...")

all_data = []

for ticker in tickers:
    try:
        print(f"Fetching data for {ticker}...")
        # psxdata ka sahi function call
        df = psxdata.stocks(ticker, start="2024-01-01") 
        if df is not None and not df.empty:
            df['Ticker'] = ticker
            all_data.append(df)
    except Exception as e:
        print(f"[WARNING] Could not fetch data for {ticker}: {e}")

if all_data:
    final_prices_df = pd.concat(all_data, ignore_index=True)
    print("\n[SUCCESS] Real Price Data fetched successfully!")
    print(final_prices_df.head())
    
    # Isay CSV file mein save kar lete hain
    final_prices_df.to_csv("psx_historical_prices.csv", index=False)
    print("[INFO] Data saved to 'psx_historical_prices.csv'")
else:
    print("[ERROR] Failed to fetch data.")