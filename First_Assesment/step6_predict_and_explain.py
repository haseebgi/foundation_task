"""
STEP 6: RISK PREDICTION + EXPLANATION
------------------------------------------
Trained model ko load karke, har stock ke LATEST data point ke liye:
  - Risk Score (0-100)          [Step 4 formula se, ground truth]
  - Predicted Risk Level          [ML model se]
  - Model confidence              [predict_proba se]
  - Contributing factors          [kaunsa component - volatility/drawdown/volume -
                                    sabse zyada is score mein contribute kar raha]

Ye ek reusable function 'predict_risk()' provide karta hai jo Step 7
(comparison) aur Step 8 (dashboard) dono use karenge.

Input:  data/all_stocks_labeled.csv, models/*.pkl
Output: console pe printed report (har stock ke liye)
"""

import pandas as pd
import joblib
import os

DATA_PATH = "data/all_stocks_labeled.csv"
MODEL_DIR = "models"


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))
    return model, scaler, feature_columns, model_name


def explain_risk(row: pd.Series) -> list:
    """
    Row ke volatility_score / drawdown_score / volume_score dekh kar
    top contributing factors ki human-readable list banata hai.
    """
    components = {
        "High volatility (price bohot upar-neeche hota hai)": row["volatility_score"],
        "Large historical price drop (high drawdown)": row["drawdown_score"],
        "Unstable / spiky trading volume": row["volume_score"],
    }
    # Sirf wo factors dikhao jo meaningfully high hain (> 40 threshold)
    significant = {k: v for k, v in components.items() if v > 40}
    if not significant:
        significant = components  # agar koi high na ho, sab dikha do context ke liye

    sorted_factors = sorted(significant.items(), key=lambda x: x[1], reverse=True)
    return [f"{name} (contribution score: {score:.0f}/100)" for name, score in sorted_factors]


def predict_risk(symbol: str, df: pd.DataFrame, model, scaler, feature_columns) -> dict:
    stock_data = df[df["Symbol"] == symbol].sort_values("Date")
    if stock_data.empty:
        return None

    latest_row = stock_data.iloc[-1]

    X = pd.DataFrame([latest_row[feature_columns]], columns=feature_columns)
    X_scaled = scaler.transform(X)

    predicted_level = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    confidence = max(proba) * 100

    return {
        "symbol": symbol,
        "date": latest_row["Date"],
        "risk_score": latest_row["risk_score"],          # ground-truth formula score
        "risk_level_actual": latest_row["risk_level"],     # ground-truth label
        "risk_level_predicted": predicted_level,            # model prediction
        "confidence_pct": round(confidence, 1),
        "explanation": explain_risk(latest_row),
    }


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    model, scaler, feature_columns, model_name = load_artifacts()

    print(f"Using model: {model_name}\n")

    for symbol in sorted(df["Symbol"].unique()):
        result = predict_risk(symbol, df, model, scaler, feature_columns)

        print("=" * 55)
        print(f"Stock: {result['symbol']}   (as of {result['date'].date()})")
        print(f"Risk Score: {result['risk_score']}/100")
        print(f"Risk Level: {result['risk_level_predicted']}  (model confidence: {result['confidence_pct']}%)")
        print("Possible contributing factors:")
        for factor in result["explanation"]:
            print(f"  - {factor}")
        print()


if __name__ == "__main__":
    main()
