"""Rating prediction service backed by the trained scikit-learn artifact."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from app.config import Settings, get_settings
from app.transforms import apply_low_rating_override, rating_inverse_transform_inv

# VADER is expensive to initialise (reads lexicon from disk), so cache it once.
_VADER: SentimentIntensityAnalyzer | None = None

def _get_vader() -> SentimentIntensityAnalyzer:
    global _VADER
    if _VADER is None:
        _VADER = SentimentIntensityAnalyzer()
    return _VADER


class RatingService:
    """Loads the trained two-stage rating model and predicts star ratings."""

    def __init__(
        self,
        settings: Settings | None = None,
        model: Any | None = None,
        feature_cols: list[str] | None = None,
        low_classifier: Any | None = None,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ):
        self.settings = settings or get_settings()
        self.model = model
        self.feature_cols = feature_cols
        self.low_classifier = low_classifier
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        all_provided = all([
            self.model is not None,
            self.feature_cols is not None,
            self.low_classifier is not None,
            self.low_threshold is not None,
            self.high_threshold is not None,
        ])
        if not all_provided:
            try:
                import joblib
            except ModuleNotFoundError as exc:
                raise RuntimeError("joblib is required for local serving. Run uv sync.") from exc

        if self.model is None:
            try:
                self.model = joblib.load(self.settings.rating_model_path)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load the trained rating model from "
                    f"{self.settings.rating_model_path}."
                ) from exc

        if self.feature_cols is None:
            try:
                self.feature_cols = joblib.load(self.settings.rating_feature_cols_path)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load the Colab-downloaded rating feature columns from "
                    f"{self.settings.rating_feature_cols_path}."
                ) from exc

        if self.low_classifier is None:
            try:
                self.low_classifier = joblib.load(self.settings.rating_low_classifier_path)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load the low-rating classifier from "
                    f"{self.settings.rating_low_classifier_path}."
                ) from exc

        if self.low_threshold is None:
            try:
                self.low_threshold = float(
                    joblib.load(self.settings.rating_classifier_low_threshold_path)
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load the classifier low threshold from "
                    f"{self.settings.rating_classifier_low_threshold_path}."
                ) from exc

        if self.high_threshold is None:
            try:
                self.high_threshold = float(
                    joblib.load(self.settings.rating_classifier_high_threshold_path)
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load the classifier high threshold from "
                    f"{self.settings.rating_classifier_high_threshold_path}."
                ) from exc

    def predict_from_features(self, features: dict[str, Any]) -> float:
        """Predict a star rating in the 1.0-5.0 range from feature values.

        Uses two-stage inference:
          1. Ridge regressor predicts fine-grained rating.
          2. LogisticRegression classifier predicts probability of low (1-3 stars).
          3. If classifier confidence exceeds threshold, prediction is capped at 3.
        """
        features = self._augment_with_text_signals(features)

        text = str(features.get("text", "") or "").lower()
        tokens = re.findall(r"[a-z0-9']+", text)

        NEGATIVE_WORDS = {
            "bad", "poor", "terrible", "awful", "horrible", "broken", "broke",
            "break", "defect", "defective", "crooked", "cracked", "scratch",
            "scratches", "flimsy", "damaged", "damage", "smoke", "stopped",
            "dead", "failed", "failure", "unusable", "useless", "waste",
            "wasted", "overpriced", "worthless", "pricey", "refund", "return",
            "returned", "disappointed", "disappointing", "never", "worst",
            "cheaply", "problem", "problems", "issue", "issues", "overheat",
            "overheating", "corrupted", "crash", "crashed",
            "hot", "loud", "slow", "buggy", "glitch", "glitching", "glitches",
            "junk", "garbage", "trash", "sucks", "suck", "abysmal", "tore",
            "tear", "tears", "ripped", "rip", "heat", "nightmare",
            "disgusted", "disgusting", "crap", "faulty", "died",
        }
        COMPLAINT_PHRASES = [
            "stopped working", "does not work", "did not work", "won't work",
            "not worth", "broke after", "broke within", "nothing but problems",
            "do not buy", "do not purchase", "waste of money", "would not recommend",
            "fell apart", "came apart", "stopped charging",
            "cracked screen", "screen cracked", "screen broke",
            "never again", "rip off", "garbage", "junk", "trash",
            "started glitching", "broke immediately",
            "too hot", "gets too hot", "loud fan", "fan is loud",
            "over heat", "over heating", "so slow",
            "poor quality", "terrible quality", "broke easily",
            "torn seam", "fabric tore", "bent frame",
            "no repair", "no warranty",
            "doesn't last", "didn't last",
        ]

        neg_hits = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        features["negativity_ratio"] = min((neg_hits / max(len(tokens), 1)) * 10, 1.0)

        phrase_hits = sum(1 for p in COMPLAINT_PHRASES if p in text)
        features["complaint_phrases"] = min(phrase_hits / 3.0, 1.0)

        raw_text = str(features.get("text", "") or "")
        words = raw_text.split()
        caps_words = sum(
            1 for w in words
            if len(w) >= 3 and sum(1 for c in w if c.isupper()) >= len(w) * 0.7
        )
        caps_ratio = caps_words / max(len(words), 1)
        excl_bonus = min(raw_text.count("!") / 5.0, 0.3)
        emoji_bonus = 0.1 if any(c in raw_text for c in "🤬😤😡😠💩👎😶") else 0.0
        features["caps_intensity"] = min(caps_ratio + excl_bonus + emoji_bonus, 1.0)

        title_text = str(features.get("product_name", features.get("text", "")) or "")[:200]
        sia = _get_vader()
        if title_text.strip():
            try:
                title_compound = sia.polarity_scores(title_text)["compound"]
                features["title_compound"] = title_compound
            except Exception:
                features["title_compound"] = 0.0
                title_compound = 0.0
        else:
            features["title_compound"] = 0.0
            title_compound = 0.0

        review_text = str(features.get("text", "") or "").lower()
        features["has_but_flag"] = 1.0 if " but " in review_text else 0.0
        try:
            review_compound = sia.polarity_scores(review_text)["compound"]
        except Exception:
            review_compound = 0.0
        features["title_review_sentiment_gap"] = abs(title_compound - review_compound)

        row = {
            col: features.get(col, "" if col == "text" else 0)
            for col in self.feature_cols
        }
        input_df = pd.DataFrame([row])

        raw_prediction = self.model.predict(input_df)[0]
        rating = float(rating_inverse_transform_inv(np.array([raw_prediction]))[0])

        low_probs = self.low_classifier.predict_proba(input_df)
        low_prob = float(low_probs[0][1])
        rating = apply_low_rating_override(
            rating,
            low_prob,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )

        return float(np.clip(rating, 1, 5))

    @staticmethod
    def _augment_with_text_signals(features: dict[str, Any]) -> dict[str, Any]:
        """Align API text inputs with the complaint-aware training features."""
        text = str(features.get("text", "") or "").lower()
        if not text.strip():
            return features

        tokens = set(re.findall(r"[a-z0-9']+", text))
        adjusted = dict(features)
        complaint_groups = {
            "quality_signal": {
                "broken", "broke", "defective", "crooked", "cracked", "damaged",
                "smoke", "stopped", "dead", "failed", "unusable", "useless",
                "flimsy", "scratch", "scratches", "glitch", "glitching",
                "crashed", "corrupted", "junk", "garbage", "trash",
                "tore", "torn", "ripped", "nightmare", "died",
            },
            "service_signal": {
                "refund", "return", "returned", "replacement", "warranty",
                "seller", "support", "technician", "company", "late", "unhelpful",
            },
            "value_signal": {
                "overpriced", "expensive", "waste", "wasted", "overrated",
                "worthless", "pricey", "cheaply", "abysmal",
            },
            "usability_signal": {
                "difficult", "hard", "confusing", "unusable", "stopped", "failed",
                "problem", "problems", "issue", "issues", "hot", "overheat",
                "overheating", "charge", "charging", "buggy", "slow",
                "loud", "glitch", "glitching",
            },
        }
        phrase_penalties = {
            "not worth": "value_signal",
            "stopped working": "quality_signal",
            "does not work": "quality_signal",
            "won't work": "quality_signal",
            "refused refund": "service_signal",
        }

        for feature_name, keywords in complaint_groups.items():
            hits = len(tokens.intersection(keywords))
            hits += sum(
                1 for phrase, target in phrase_penalties.items()
                if target == feature_name and phrase in text
            )
            if hits:
                current = float(adjusted.get(feature_name, 0) or 0)
                penalty = min(1.5, 0.35 * hits)
                adjusted[feature_name] = max(-1.0, min(current, current - penalty))

        negative_hits = sum(
            len(tokens.intersection(words)) for words in complaint_groups.values()
        )
        if negative_hits:
            current_strictness = float(adjusted.get("strictness", 0) or 0)
            adjusted["strictness"] = min(
                1.0, max(current_strictness, 0.15 + (0.08 * negative_hits))
            )

        return adjusted
