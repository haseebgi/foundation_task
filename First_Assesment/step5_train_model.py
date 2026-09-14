"""
STEP 5: MACHINE LEARNING MODEL (Training + Comparison)
----------------------------------------------------------
Is script mein hum 4 alag ML algorithms train karte hain aur compare
karte hain ke risk_level (Low/Medium/High) predict karne mein kaunsa best hai:

    - Random Forest
    - Decision Tree
    - Logistic Regression
    - Support Vector Machine (SVM)

Features (X) wahi hain jo Step 3 mein banaye the.
Target (y) wo risk_level hai jo Step 4 mein label kiya tha.

IMPORTANT: Train/Test split TIME-BASED hai (random shuffle nahi) - kyunki
stock data time series hai, aur future data se past predict karna "data
leakage" hoga jo real duniya mein galat results dega.

Best model 'models/best_model.pkl' mein save ho jata hai, taake Step 6
(prediction) mein use ho sake.

Input:  data/all_stocks_labeled.csv
Output: models/best_model.pkl, models/scaler.pkl, models/feature_columns.pkl
"""

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

INPUT_PATH = "data/all_stocks_labeled.csv"
MODEL_DIR = "models"

# Features jo model ko dikhengi (X). Note: risk_score/risk_level/scores
# is list mein NAHI hain kyunki wo target se seedha derive hui hain (leakage se bachne ke liye)
FEATURE_COLUMNS = [
    "daily_return_pct",
    "volatility_20d",
    "volatility_60d",
    "price_vs_ma50_pct",
    "max_drawdown_pct",
    "avg_volume_20d",
    "volume_volatility_20d",
    "annualized_volatility",
]

TARGET_COLUMN = "risk_level"


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    """Har stock ke liye: pehle 80% (time ke hisaab se) train, aakhri 20% test."""
    train_parts, test_parts = [], []
    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date")
        split_idx = int(len(group) * (1 - test_size))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])
    return pd.concat(train_parts), pd.concat(test_parts)


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    train_df, test_df = time_based_split(df)
    print(f"Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    # Scaling zaroori hai especially Logistic Regression aur SVM ke liye
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }

    results = {}
    trained_models = {}

    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)

    for name, model in models.items():
        # Random Forest / Decision Tree scale-independent hain, but consistent rakhte hain
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        results[name] = {"accuracy": acc, "f1_score": f1}
        trained_models[name] = model

        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score (weighted): {f1:.4f}")

    # Best model chuno (F1 score ke basis par - imbalanced classes ke liye zyada fair hai)
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = trained_models[best_name]

    print("\n" + "=" * 55)
    print(f"BEST MODEL: {best_name}")
    print("=" * 55)
    print(classification_report(y_test, best_model.predict(X_test_scaled)))

    # Feature importance (agar model support karta ho)
    if hasattr(best_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": best_model.feature_importances_
        }).sort_values("importance", ascending=False)
        print("\nFeature Importance (best model):")
        print(importance_df.to_string(index=False))

    # Sab kuch save karo taake Step 6 mein use ho sake
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    joblib.dump(best_name, os.path.join(MODEL_DIR, "best_model_name.pkl"))

    print(f"\nModel saved to '{MODEL_DIR}/' folder. Best model: {best_name}")


if __name__ == "__main__":
    main()
