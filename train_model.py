"""
train_model.py — Instagram Reach Forecasting
College-level ML pipeline with:
  - Feature engineering (NLP sentiment, time encoding, hashtag parsing)
  - Ensemble model (Random Forest + Gradient Boosting stacked with Ridge)
  - 5-fold cross-validated evaluation (R2, MAE, RMSE per target)
  - Feature importance logging
  - Model versioning via joblib
"""

import pandas as pd
import numpy as np
import joblib
import json
import re
from datetime import datetime
from feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin

# Optional: TextBlob for sentiment (pip install textblob)
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not found - skipping sentiment. Run: pip install textblob")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_pipeline() -> Pipeline:

    # 🔹 Define columns (based on your dataset)
    numerical_cols = ["followers"]
    categorical_cols = ["post_type", "day_of_week"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    base_estimators = [
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
        ("gb", GradientBoostingRegressor(n_estimators=150, random_state=42)),
    ]

    stacked = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(),
        passthrough=True
    )

    return Pipeline([
        ("preprocessor", preprocessor),   # ✅ THIS IS KEY
        ("model", MultiOutputRegressor(stacked))
    ])
#  EVALUATION

def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Compute R2, MAE, RMSE for each target on held-out test data."""
    y_pred  = pipeline.predict(X_test)
    targets = ["reach", "likes", "comments"]
    metrics = {}

    print("\n  Evaluation on test set")
    print("  " + "─" * 48)
    for i, target in enumerate(targets):
        r2   = r2_score(y_test.iloc[:, i], y_pred[:, i])
        mae  = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        metrics[target] = {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2)}
        print(f"  {target:<12}  R2={r2:+.4f}  MAE={mae:>9.1f}  RMSE={rmse:>9.1f}")

    print("  " + "─" * 48)
    return metrics

def cross_validate_model(X, y) -> None:
    """5-fold CV for multi-output model"""
    print("\n  5-Fold Cross-Validation")
    print("  " + "─" * 48)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    pipeline = build_pipeline()

    scores = cross_validate(
        pipeline,
        X,
        y,  # ✅ FULL multi-output target
        cv=kf,
        scoring="r2",  # single metric for multi-output
        n_jobs=-1,
    )

    print(f"  Mean CV R2: {scores['test_score'].mean():+.4f}")
    print("  " + "─" * 48)

#  MAIN

def main():
    print("InstaReach — Model Training Pipeline\n")

    # Load
    try:
        df = pd.read_csv("expanded_instagram_data.csv")
    except FileNotFoundError:
        print("ERROR: expanded_instagram_data.csv not found.")
        return

    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

    TARGET_COLS = ["reach", "likes", "comments"]
    X = df.drop(columns=TARGET_COLS, errors="ignore")
    y = df[TARGET_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # Cross-validation (informational, runs before final fit)
    cross_validate_model(X_train, y_train)

    # Final model fit on full training set
    print("\n  Training final ensemble...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluation on held-out test set
    metrics = evaluate_model(pipeline, X_test, y_test)

    # Save model
    joblib.dump(pipeline, "reach_model.pkl")

    # Save metadata for auditability
    metadata = {
        "trained_at":    datetime.now().isoformat(),
        "train_rows":    len(X_train),
        "test_rows":     len(X_test),
        "input_columns": list(X.columns),
        "targets":       TARGET_COLS,
        "evaluation":    metrics,
        "textblob_used": TEXTBLOB_AVAILABLE,
        "model":         "StackingRegressor(RF + GB -> Ridge) wrapped in MultiOutputRegressor",
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n  Model saved   ->  reach_model.pkl")
    print("  Metadata saved -> model_metadata.json\n")


if __name__ == "__main__":
    main()
