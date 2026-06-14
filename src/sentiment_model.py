"""
src/ml/sentiment_model.py

Core sentiment classifier for warehouse/frontline worker reviews.
Uses Sentence-BERT embeddings + Gradient Boosting classifier.

This is your original model — preserved as-is.
The pipeline feeds it cleaned, labeled data via the unified schema.
"""

import joblib
import numpy as np 
import pandas as pd 
from textblob import TextBlob 
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import re

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import KEYWORD_THEMES 


class WorkforceSentimentModel:
    """
    Three-class sentiment classifier for employee reviews.
    Labels: positive, neutral, negative
    """

    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,!?-]", "", text)
        return text.strip()

    def extract_features(self, text: str) -> dict:
        """Extract embeddings + metadata features from a single review."""
        cleaned = self.clean_text(text)

        # Sentence-BERT embedding
        embedding = self.encoder.encode([cleaned])[0]

        # TextBlob sentiment features
        blob = TextBlob(cleaned)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Keyword theme counts
        text_lower = cleaned.lower()
        theme_counts = {
            f"{theme}_keywords": sum(1 for kw in keywords if kw in text_lower)
            for theme, keywords in KEYWORD_THEMES.items()
        }

        return {
            "embedding": embedding,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "review_length": len(cleaned.split()),
            **theme_counts,
        }

    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """
        Convert a labeled DataFrame into X, y arrays for sklearn.

        Args:
            df: DataFrame with 'text' and 'sentiment_label' columns.

        Returns:
            (X, y) tuple ready for classifier.fit()
        """
        embeddings = []
        metadata = []

        for _, row in df.iterrows():
            features = self.extract_features(row["text"])
            embeddings.append(features.pop("embedding"))
            metadata.append([
                features["polarity"],
                features["subjectivity"],
                features["review_length"],
                features.get("burnout_keywords", 0),
                features.get("physical_demands_keywords", 0),
                features.get("management_issues_keywords", 0),
                features.get("workload_keywords", 0),
                features.get("environment_keywords", 0),
            ])

        X = np.hstack([np.array(embeddings), np.array(metadata)])
        y = self.label_encoder.fit_transform(df["sentiment_label"].values)

        self.feature_names = (
            [f"emb_{i}" for i in range(embeddings[0].shape[0])]
            + [
                "polarity", "subjectivity", "review_length",
                "burnout_keywords", "physical_demands_keywords",
                "management_issues_keywords", "workload_keywords",
                "environment_keywords",
            ]
        )

        return X, y

    def train(self, df: pd.DataFrame, **gb_kwargs) -> "WorkforceSentimentModel":
        """
        Train the Gradient Boosting classifier on a labeled DataFrame.

        Args:
            df: DataFrame with 'text' and 'sentiment_label' columns.
            **gb_kwargs: Extra keyword args for GradientBoostingClassifier.

        Returns:
            self (for chaining).
        """
        X, y = self.prepare_training_data(df)

        default_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        default_params.update(gb_kwargs)

        self.classifier = GradientBoostingClassifier(**default_params)
        self.classifier.fit(X, y)
        return self

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single review text.

        Returns:
            dict with sentiment_label, confidence, class_probabilities,
                  uncertainty_flag, confidence_gap, themes_detected
        """
        if self.classifier is None:
            raise RuntimeError("Model not loaded. Call load() or train() first.")

        features = self.extract_features(text)
        embedding = features.pop("embedding")

        meta = np.array([[
            features["polarity"],
            features["subjectivity"],
            features["review_length"],
            features["burnout_keywords"],
            features["physical_demands_keywords"],
            features["management_issues_keywords"],
            features["workload_keywords"],
            features["environment_keywords"],
        ]])

        X = np.hstack([embedding.reshape(1, -1), meta])
        proba = self.classifier.predict_proba(X)[0]
        classes = self.label_encoder.classes_
        label_idx = np.argmax(proba)
        label = classes[label_idx]
        confidence = float(proba[label_idx])

        sorted_proba = sorted(proba, reverse=True)
        confidence_gap = float(sorted_proba[0] - sorted_proba[1])

        themes = [
            theme for theme, keywords in KEYWORD_THEMES.items()
            if any(kw in text.lower() for kw in keywords)
        ]

        return {
            "sentiment_label": label,
            "confidence": round(confidence, 4),
            "class_probabilities": {
                cls: round(float(p), 4) for cls, p in zip(classes, proba)
            },
            "uncertainty_flag": confidence < 0.5,
            "confidence_gap": round(confidence_gap, 4),
            "themes_detected": themes,
        }

    # Maps KEYWORD_THEMES keys → BAT-12 / OLBI subscale domain names.
    # Used in analyze_debrief to translate detected themes into the same
    # domain vocabulary that _boosted_categories_from_scales expects.
    _THEME_TO_DOMAIN = {
        "burnout":            "exhaustion",
        "physical_demands":   "exhaustion",         # Physical fatigue → BAT-12 exhaustion
        "workload":           "cognitive_impairment",
        "management_issues":  "disengagement",       # Management frustration → OLBI disengagement
        "environment":        "emotional_impairment",
    }

    def analyze_debrief(self, text: str) -> dict:
        """
        Analyze a shift debrief entry for sentiment and burnout domain signals.

        Extends predict() by mapping detected keyword themes to BAT-12 / OLBI
        subscale domain names. The domain_signals dict uses normalized_0_10
        compatible keys so it can be passed directly to
        _boosted_categories_from_scales in intervention_recommender.py without
        requiring a full psychometric assessment.

        Args:
            text: Raw shift debrief string.

        Returns:
            All predict() fields plus:
                has_debrief (bool)
                domain_signals: {domain_name: {"normalized_0_10": float}}
        """
        if not text or not text.strip():
            return {
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "class_probabilities": {},
                "uncertainty_flag": True,
                "confidence_gap": 0.0,
                "themes_detected": [],
                "domain_signals": {},
                "has_debrief": False,
            }

        result = self.predict(text)
        result["has_debrief"] = True

        # Accumulate domain signal strength from detected themes.
        # Each theme hit contributes 0.5 to its domain (max 1.0 before sentiment scaling).
        raw_signals: dict[str, float] = {}
        for theme in result.get("themes_detected", []):
            domain = self._THEME_TO_DOMAIN.get(theme)
            if domain:
                raw_signals[domain] = min(1.0, raw_signals.get(domain, 0.0) + 0.5)

        # Amplify when the overall sentiment is negative — a negative debrief
        # combined with burnout language is a stronger signal than either alone.
        multiplier = 1.3 if result["sentiment_label"] == "negative" else 1.0

        # Convert to normalized_0_10 format so the output is structurally
        # compatible with ScaleSubscore dicts from burnout_scales.py.
        result["domain_signals"] = {
            domain: {"normalized_0_10": round(min(10.0, strength * multiplier * 10), 2)}
            for domain, strength in raw_signals.items()
        }

        return result

    def save(self, path: str):
        joblib.dump({
            "classifier": self.classifier,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "WorkforceSentimentModel":
        instance = cls()
        data = joblib.load(path)
        instance.classifier = data["classifier"]
        instance.label_encoder = data["label_encoder"]
        instance.feature_names = data["feature_names"]
        return instance
