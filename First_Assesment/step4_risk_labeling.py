"""
STEP 4: RISK LABELING
------------------------
ML model ko supervised training ke liye "ground truth" label chahiye hota hai.
Is script mein hum ek RULE-BASED composite Risk Score (0-100) banate hain,
jo teen cheezon ko combine karta hai:

  1. Annualized Volatility     (kitna price upar-neeche hota hai)
  2. Max Drawdown               (worst-case loss from peak)
  3. Volume Volatility          (trading volume kitna unstable hai)

Har component ko 0-100 scale par normalize (min-max) karte hain, phir
weighted average leke final Risk Score banate hain:

    Risk Score = 0.5 * volatility_score
               + 0.3 * drawdown_score
               + 0.2 * volume_score

Phir Risk Score ko category mein classify karte hain:
    0-30   -> Low Risk
    31-60  -> Medium Risk
    61-100 -> High Risk

NOTE: Ye labeling formula ek starting point hai. Real-world project mein
isay financial domain expert se validate karwana chahiye, ya historical
"actual loss" data se refine karna chahiye.

Input:  data/all_stocks_features.csv
Output: data/all_stocks_labeled.csv
"""

import pandas as pd
import numpy as np

INPUT_PATH = "data/all_stocks_features.csv"
OUTPUT_PATH = "data/all_stocks_labeled.csv"


def min_max_scale(series: pd.Series) -> pd.Series:
    """Series ko 0-100 range mein scale karta hai."""
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series(50, index=series.index)  # sab same ho to neutral 50 de do
    return (series - min_val) / (max_val - min_val) * 100


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    # Drawdown negative hota hai (e.g. -40%), isay positive "severity" mein convert karna
    drawdown_severity = df["max_drawdown_pct"].abs()

    volatility_score = min_max_scale(df["annualized_volatility"])
    drawdown_score = min_max_scale(drawdown_severity)
    volume_score = min_max_scale(df["volume_volatility_20d"])

    df["volatility_score"] = volatility_score
    df["drawdown_score"] = drawdown_score
    df["volume_score"] = volume_score

    df["risk_score"] = (
        0.5 * volatility_score +
        0.3 * drawdown_score +
        0.2 * volume_score
    ).round(1)

    # Clip 0-100 ke andar (safety)
    df["risk_score"] = df["risk_score"].clip(0, 100)

    def classify(score):
        if score <= 30:
            return "Low Risk"
        elif score <= 60:
            return "Medium Risk"
        else:
            return "High Risk"

    df["risk_level"] = df["risk_score"].apply(classify)

    return df


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    labeled_df = compute_risk_score(df)

    labeled_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Labeled data saved: {OUTPUT_PATH}  ({len(labeled_df)} rows)")

    print("\nRisk level distribution:")
    print(labeled_df["risk_level"].value_counts())

    print("\nAverage risk score per stock (latest available snapshot):")
    latest = labeled_df.sort_values("Date").groupby("Symbol").tail(1)
    print(latest[["Symbol", "risk_score", "risk_level"]].sort_values("risk_score", ascending=False))


if __name__ == "__main__":
    main()
