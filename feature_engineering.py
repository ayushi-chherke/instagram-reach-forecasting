# feature_engineering.py

import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that extracts richer features from raw inputs.
    Plugs directly into the sklearn Pipeline so it works at both
    training time and prediction time automatically.
    """

    def fit(self, X, y=None):
        return self  # stateless transformer

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # --- Caption NLP features ---
        if "caption" in df.columns:
            df["caption_length"]     = df["caption"].fillna("").apply(len)
            df["caption_word_count"] = df["caption"].fillna("").apply(lambda t: len(t.split()))
            df["has_question"]       = df["caption"].fillna("").apply(lambda t: int("?" in t))
            df["has_exclamation"]    = df["caption"].fillna("").apply(lambda t: int("!" in t))

            if TEXTBLOB_AVAILABLE:
                df["sentiment_polarity"]     = df["caption"].fillna("").apply(
                    lambda t: TextBlob(t).sentiment.polarity)
                df["sentiment_subjectivity"] = df["caption"].fillna("").apply(
                    lambda t: TextBlob(t).sentiment.subjectivity)
            else:
                df["sentiment_polarity"]     = 0.0
                df["sentiment_subjectivity"] = 0.0

            df.drop(columns=["caption"], inplace=True)

        # --- Hashtag features ---
        if "hashtags" in df.columns:
            df["hashtags_count"] = df["hashtags"].fillna("").apply(
                lambda h: len([x for x in re.split(r"[,\s]+", h) if x.startswith("#")]))
            df["has_hashtags"] = (df["hashtags_count"] > 0).astype(int)
            df.drop(columns=["hashtags"], inplace=True)

        # --- Temporal features with cyclical encoding ---
        if "postTime" in df.columns:
            def parse_time(val):
                if pd.isna(val) or val == "":
                    return pd.Timestamp.now()
                try:
                    return pd.to_datetime(val)
                except Exception:
                    return pd.Timestamp.now()

            dt   = df["postTime"].apply(parse_time)
            hour = dt.dt.hour
            dow  = dt.dt.dayofweek   # 0=Mon, 6=Sun

            # Cyclical encoding so hour 23 and hour 0 are close (not 23 apart)
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            df["dow_sin"]  = np.sin(2 * np.pi * dow  / 7)
            df["dow_cos"]  = np.cos(2 * np.pi * dow  / 7)

            # Domain-knowledge binary flags
            df["is_peak_hour"] = hour.apply(lambda h: int(h in range(18, 22))).values
            df["is_weekend"]   = (dow >= 5).astype(int).values

            df.drop(columns=["postTime"], inplace=True)

        # --- Region engagement tier ---
        HIGH_ENGAGEMENT = {"US", "UK", "AU", "CA", "GB"}
        if "region" in df.columns:
            df["region_tier"] = df["region"].fillna("").str.upper().apply(
                lambda r: 2 if r in HIGH_ENGAGEMENT else (1 if r else 0))
            df.drop(columns=["region"], inplace=True)

        # --- Log-transform followers (right-skewed distribution) ---
        if "followers" in df.columns:
            df["followers_log"] = np.log1p(df["followers"].fillna(0))
            df.drop(columns=["followers"], inplace=True)

        # --- Log-transform engagement goal ---
        if "engagementGoal" in df.columns:
            df["engagement_goal_log"] = np.log1p(df["engagementGoal"].fillna(0))
            df.drop(columns=["engagementGoal"], inplace=True)

              # --- Log-transform engagement goal ---
        if "engagementGoal" in df.columns:
            df["engagement_goal_log"] = np.log1p(df["engagementGoal"].fillna(0))
            df.drop(columns=["engagementGoal"], inplace=True)

        # ✅ ADD THIS SECTION HERE ↓↓↓

        # --- Post type encoding ---
        if "post_type" in df.columns:
            post_type_map = {"image": 0, "video": 1, "reel": 2}
            df["post_type_encoded"] = df["post_type"].map(post_type_map).fillna(0)
            df.drop(columns=["post_type"], inplace=True)

        # --- Day of week encoding ---
        if "day_of_week" in df.columns:
            day_map = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2,
                "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
            }
            dow = df["day_of_week"].map(day_map).fillna(0)

            df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
            df["is_weekend"] = (dow >= 5).astype(int)

            df.drop(columns=["day_of_week"], inplace=True)

        # --- time_of_day encoding ---
        if "time_of_day" in df.columns:
            hour = df["time_of_day"]

            df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            df["is_peak_hour"] = hour.apply(lambda h: int(h in range(18, 22)))

            df.drop(columns=["time_of_day"], inplace=True)

        return df

