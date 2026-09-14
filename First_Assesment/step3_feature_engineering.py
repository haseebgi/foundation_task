"""
STEP 3: FEATURE ENGINEERING
-------------------------------
Yahan hum raw price/volume data se wo features banate hain jo ACTUALLY
risk ko measure karte hain. Ye features hi ML model ke "inputs" (X) banenge.

Features banaye ja rahe hain (har stock ke liye, rolling window ke sath):
  - daily_return_pct       : din ka % return
  - volatility_20d         : pichle 20 din ke returns ka std dev (short-term risk)
  - volatility_60d         : pichle 60 din ke returns ka std dev (long-term risk)
  - ma_20                  : 20-day moving average
  - ma_50                  : 50-day moving average
  - price_vs_ma50_pct      : current price moving average se kitna upar/neeche hai
  - max_drawdown_pct       : ab tak ka sabse bada peak-se-gira hua %  (rolling 252 din)
  - avg_volume_20d         : pichle 20 din ka average trading volume
  - volume_volatility_20d  : volume ka std dev / mean  (unstable volume ka signal)
  - annualized_volatility  : volatility_20d ko saal ke hisaab se scale kiya hua (%)

Input:  data/all_stocks_clean.csv
Output: data/all_stocks_features.csv
"""

import pandas as pd
import numpy as np

INPUT_PATH = "data/all_stocks_clean.csv"
OUTPUT_PATH = "data/all_stocks_features.csv"

TRADING_DAYS_PER_YEAR = 252


def compute_max_drawdown(prices: pd.Series, window: int = 252) -> pd.Series:
    """Rolling window ke andar, price apne peak se max kitna gira - percentage mein (negative value)."""
    rolling_max = prices.rolling(window, min_periods=20).max()
    drawdown = (prices - rolling_max) / rolling_max * 100
    return drawdown.rolling(window, min_periods=20).min()


def engineer_features_for_stock(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()

    # 1. Daily return %
    df["daily_return_pct"] = df["Close"].pct_change() * 100

    # 2. Rolling volatility (std dev of daily returns)
    df["volatility_20d"] = df["daily_return_pct"].rolling(20, min_periods=10).std()
    df["volatility_60d"] = df["daily_return_pct"].rolling(60, min_periods=20).std()

    # 3. Moving averages
    df["ma_20"] = df["Close"].rolling(20, min_periods=10).mean()
    df["ma_50"] = df["Close"].rolling(50, min_periods=20).mean()
    df["price_vs_ma50_pct"] = (df["Close"] - df["ma_50"]) / df["ma_50"] * 100

    # 4. Maximum drawdown (rolling 1-year window)
    df["max_drawdown_pct"] = compute_max_drawdown(df["Close"], window=TRADING_DAYS_PER_YEAR)

    # 5. Volume based features
    df["avg_volume_20d"] = df["Volume"].rolling(20, min_periods=10).mean()
    df["volume_volatility_20d"] = (
        df["Volume"].rolling(20, min_periods=10).std() /
        df["Volume"].rolling(20, min_periods=10).mean()
    )

    # 6. Annualized volatility (% terms) -> easy to compare across stocks
    df["annualized_volatility"] = df["volatility_20d"] * np.sqrt(TRADING_DAYS_PER_YEAR)

    return df


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])

    all_features = []
    for symbol, group in df.groupby("Symbol"):
        print(f"Engineering features for {symbol} ...")
        feat_df = engineer_features_for_stock(group)
        all_features.append(feat_df)

    result = pd.concat(all_features, ignore_index=True)

    # Rolling windows ki wajah se shuru ke kuch rows mein NaN aayenge - unhe drop karte hain
    before = len(result)
    result = result.dropna(subset=[
        "volatility_20d", "volatility_60d", "ma_50", "max_drawdown_pct"
    ])
    print(f"\nRows dropped due to rolling-window warmup (NaN): {before - len(result)}")

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Feature data saved: {OUTPUT_PATH}  ({len(result)} rows)")
    print("\nSample features:")
    cols_to_show = ["Date", "Symbol", "Close", "annualized_volatility", "max_drawdown_pct", "volume_volatility_20d"]
    print(result[cols_to_show].tail())


if __name__ == "__main__":
    main()
