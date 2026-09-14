"""
STEP 7: STOCK COMPARISON
----------------------------
Sab stocks ka latest risk score/level ek table mein compare karta hai,
Risk Score ke hisaab se sort karke (highest risk sabse upar).

Input:  data/all_stocks_labeled.csv, models/*.pkl
Output: data/comparison_table.csv  + console table
"""

import pandas as pd
from step6_predict_and_explain import load_artifacts, predict_risk

DATA_PATH = "data/all_stocks_labeled.csv"
OUTPUT_PATH = "data/comparison_table.csv"


def build_comparison_table() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    model, scaler, feature_columns, model_name = load_artifacts()

    rows = []
    for symbol in sorted(df["Symbol"].unique()):
        result = predict_risk(symbol, df, model, scaler, feature_columns)
        rows.append({
            "Stock": result["symbol"],
            "Risk Score": result["risk_score"],
            "Risk Level": result["risk_level_predicted"],
            "Model Confidence (%)": result["confidence_pct"],
            "Top Factor": result["explanation"][0] if result["explanation"] else "N/A",
        })

    comparison_df = pd.DataFrame(rows).sort_values("Risk Score", ascending=False).reset_index(drop=True)
    comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))
    return comparison_df


def main():
    comparison_df = build_comparison_table()
    comparison_df.to_csv(OUTPUT_PATH, index=False)

    print("STOCK RISK COMPARISON (sorted: highest risk first)")
    print("=" * 90)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
