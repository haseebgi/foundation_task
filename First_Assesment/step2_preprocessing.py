"""
STEP 2: DATA PREPROCESSING
----------------------------
Is script mein hum raw stock data ko clean karte hain:
  1. Missing values handle karna
  2. Duplicate rows remove karna
  3. Data types fix karna (Date ko proper datetime banana)
  4. Symbol + Date ke hisaab se sort karna (zaroori for feature engineering)
  5. Negative/zero price jaisi invalid values check karna

Input:  data/all_stocks_raw.csv
Output: data/all_stocks_clean.csv
"""

import pandas as pd
import numpy as np

INPUT_PATH = "data/all_stocks_raw.csv"
OUTPUT_PATH = "data/all_stocks_clean.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)

    # 1. Date ko proper datetime type mein convert karna
    df["Date"] = pd.to_datetime(df["Date"])

    # 2. Duplicate rows hatana (same Symbol + Date)
    df = df.drop_duplicates(subset=["Symbol", "Date"])
    print(f"Duplicates removed: {initial_rows - len(df)} rows")

    # 3. Missing values check karna
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"Missing values found: {missing_before} -> forward-fill kar rahe hain (per stock)")
        # Har stock ke andar forward-fill (pichle din ki value se fill)
        df = df.sort_values(["Symbol", "Date"])
        df = df.groupby("Symbol", group_keys=False).apply(lambda g: g.ffill().bfill())
    else:
        print("Koi missing values nahi mile.")

    # 4. Invalid / negative price rows hatana
    price_cols = ["Open", "High", "Low", "Close"]
    invalid_mask = (df[price_cols] <= 0).any(axis=1)
    if invalid_mask.sum() > 0:
        print(f"Invalid (<=0) price rows hatayi gayin: {invalid_mask.sum()}")
        df = df[~invalid_mask]

    # 5. Final sort - Symbol ke andar Date ke hisaab se (ascending)
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    return df


def main():
    df = load_data(INPUT_PATH)
    clean_df = clean_data(df)

    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nClean data saved: {OUTPUT_PATH}  ({len(clean_df)} rows, {clean_df['Symbol'].nunique()} stocks)")
    print("\nSample:")
    print(clean_df.head())


if __name__ == "__main__":
    main()
