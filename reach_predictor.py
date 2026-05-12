"""
reach_predictor.py — Instagram Reach Forecasting
"""

import sys
import json
import math
import joblib
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer
#  VALIDATION
REQUIRED_FIELDS = ["followers", "caption", "hashtags", "postTime", "region"]
FIELD_RULES = {
    "followers":      {"type": int,   "min": 0,     "max": 100_000_000},
    "engagementGoal": {"type": int,   "min": 0,     "max": 100_000_000, "optional": True},
}

def validate_input(data: dict) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in data or data[field] is None or data[field] == "":
            errors.append(f"Missing required field: {field}")

    for field, rules in FIELD_RULES.items():
        if rules.get("optional") and (field not in data or data[field] is None):
            continue
        if field in data and data[field] is not None:
            try:
                val = rules["type"](data[field])
                if val < rules["min"] or val > rules["max"]:
                    errors.append(f"{field} must be between {rules['min']} and {rules['max']}")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a number")

    return errors

#  CONFIDENCE INTERVAL

def confidence_interval(pipeline, X: pd.DataFrame, target_idx: int,
                         point_estimate: float, z: float = 1.645) -> dict:
    """
    Estimate an 80% confidence interval using the variance across
    individual trees in the underlying Random Forest estimator.

    z=1.645 for 80%, z=1.96 for 95%.
    Works by accessing the per-tree predictions of the first base
    estimator (RF) inside the MultiOutputRegressor -> StackingRegressor.
    """
    try:
        # Drill into: Pipeline -> model (MultiOutputRegressor) -> estimators_[target_idx]
        # -> estimators_[0] (RF inside StackingRegressor)
        multi_output = pipeline.named_steps["model"]
        stacked      = multi_output.estimators_[target_idx]
        rf           = stacked.estimators_[0]   # RandomForestRegressor

        # Transform X through engineer + scaler (everything before the model step)
        X_transformed = pipeline[:-1].transform(X)

        tree_preds = np.array([tree.predict(X_transformed)[0] for tree in rf.estimators_])
        std        = tree_preds.std()
        margin     = z * std

        return {
            "lower": max(0, int(point_estimate - margin)),
            "upper": int(point_estimate + margin),
        }
    except Exception:
        # Graceful fallback: ±15% of point estimate
        margin = point_estimate * 0.15
        return {
            "lower": max(0, int(point_estimate - margin)),
            "upper": int(point_estimate + margin),
        }

#  QUICK TIPS (rule-based, shown alongside prediction)

def generate_tips(data: dict) -> list[str]:
    tips = []
    import re

    hashtag_str   = data.get("hashtags", "")
    hashtag_count = len([x for x in re.split(r"[,\s]+", hashtag_str) if x.startswith("#")])
    caption       = data.get("caption", "")
    post_time     = data.get("postTime", "")

    if hashtag_count < 5:
        tips.append(f"Add more hashtags — you have {hashtag_count}, aim for 5–11.")
    elif hashtag_count > 15:
        tips.append("Too many hashtags can look spammy — try 5–11 for best results.")

    if len(caption) < 50:
        tips.append("A longer, engaging caption typically improves reach.")

    if post_time:
        try:
            hour = pd.to_datetime(post_time).hour
            if not (18 <= hour <= 21):
                tips.append("Consider posting between 6 PM and 9 PM for peak engagement.")
        except Exception:
            pass

    region = data.get("region", "").upper()
    if region not in {"US", "UK", "AU", "CA", "GB"}:
        tips.append("Posts targeting high-engagement regions (US, UK, AU) tend to perform better.")

    return tips
#  MAIN
def main():
    # 1. Parse input
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided. Pass a JSON string as the first argument."}))
        sys.exit(1)

    try:
        input_data = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}))
        sys.exit(1)

    # 2. Validate
    errors = validate_input(input_data)
    if errors:
        print(json.dumps({"error": "Validation failed", "details": errors}))
        sys.exit(1)

    # 3. Load model
    try:
        pipeline = joblib.load("reach_model.pkl")
    except FileNotFoundError:
        print(json.dumps({"error": "Model file not found. Run train_model.py first."}))
        sys.exit(1)

    # 4. Predict
    try:
        # Build a single-row DataFrame — same column names used during training
        input_df   = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]

        reach    = max(0, int(prediction[0]))
        likes    = max(0, int(prediction[1]))
        comments = max(0, int(prediction[2]))

        # 5. Confidence intervals (80%)
        reach_ci    = confidence_interval(pipeline, input_df, 0, reach)
        likes_ci    = confidence_interval(pipeline, input_df, 1, likes)
        comments_ci = confidence_interval(pipeline, input_df, 2, comments)

        # 6. Engagement rate estimate
        followers      = max(1, int(input_data.get("followers", 1)))
        engagement_rate = round((likes + comments) / followers * 100, 2)

        # 7. Tips
        tips = generate_tips(input_data)

        output = {
            "reach":    reach,
            "likes":    likes,
            "comments": comments,
            "confidence_intervals": {
                "reach":    reach_ci,
                "likes":    likes_ci,
                "comments": comments_ci,
            },
            "engagement_rate_pct": engagement_rate,
            "tips": tips,
        }

        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({"error": f"Prediction failed: {str(e)}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()