"""
STEP 1b: SAMPLE DATA GENERATOR (fallback)
-------------------------------------------
Ye script sirf tab use karein jab real Yahoo Finance data na mil raha ho
(jaise is sandbox mein, jahan internet restricted hai).

Ye realistic dikhne wala synthetic stock data banata hai — random walk +
volatility + volume patterns ke saath — taake pura pipeline (preprocessing,
feature engineering, ML model) develop aur test kiya ja sake.

Jab aap apne computer par real yfinance data le lenge (step1_data_collection.py se),
to bas 'data/' folder replace kar dena — baaki pipeline same rahega.
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "data"

# Har stock ki apni "personality" — kis rate se badhta hai, kitna volatile hai
STOCK_PROFILES = {
    "STOCK_A_STABLE":   {"drift": 0.0004, "volatility": 0.010, "base_price": 150},  # Low risk
    "STOCK_B_MODERATE": {"drift": 0.0006, "volatility": 0.020, "base_price": 80},   # Medium risk
    "STOCK_C_VOLATILE": {"drift": 0.0008, "volatility": 0.045, "base_price": 40},   # High risk
    "STOCK_D_STABLE2":  {"drift": 0.0003, "volatility": 0.012, "base_price": 200},  # Low risk
    "STOCK_E_RISKY":    {"drift": -0.0002, "volatility": 0.055, "base_price": 25},  # High risk
}

START_DATE = "2015-01-01"
END_DATE = "2025-01-01"


def generate_stock(symbol: str, drift: float, volatility: float, base_price: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=START_DATE, end=END_DATE)  # business days only
    n = len(dates)

    # Daily returns as random walk (Geometric Brownian Motion style)
    daily_returns = rng.normal(loc=drift, scale=volatility, size=n)
    price_path = base_price * np.cumprod(1 + daily_returns)

    close = price_path
    open_ = close * (1 + rng.normal(0, volatility * 0.3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, volatility * 0.4, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, volatility * 0.4, n)))
    volume = rng.integers(low=int(1e5), high=int(5e6), size=n).astype(float)

    # Volatile stocks -> occasional volume spikes (news/panic days)
    spike_days = rng.choice(n, size=max(1, n // 40), replace=False)
    volume[spike_days] *= rng.uniform(2, 5, size=len(spike_days))

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_.round(2),
        "High": high.round(2),
        "Low": low.round(2),
        "Close": close.round(2),
        "Volume": volume.round(0),
        "Symbol": symbol,
    })
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data = []

    for i, (symbol, profile) in enumerate(STOCK_PROFILES.items()):
        print(f"Generating synthetic data for {symbol} ...")
        df = generate_stock(symbol, profile["drift"], profile["volatility"], profile["base_price"], seed=42 + i)
        save_path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
        df.to_csv(save_path, index=False)
        print(f"  -> Saved: {save_path}  ({len(df)} rows)")
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, "all_stocks_raw.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined raw file saved: {combined_path}  ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
