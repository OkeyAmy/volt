"""Rating prediction service backed by the trained scikit-learn artifact."""

from __future__ import annotations

import re
from typing import Any

from app.config import Settings, get_settings
from app.transforms import apply_low_rating_override, rating_inverse_transform_inv

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "horrible", "broken", "broke",
    "break", "defect", "defective", "crooked", "cracked", "scratch",
    "scratches", "flimsy", "damaged", "damage", "smoke", "stopped",
    "dead", "failed", "failure", "unusable", "useless", "waste",
    "wasted", "overpriced", "worthless", "pricey", "refund", "return",
    "returned", "disappointed", "disappointing", "never", "worst",
    "cheaply", "problem", "problems", "issue", "issues", "overheat",
    "overheating", "corrupted", "crash", "crashed", "hot", "loud",
    "slow", "buggy", "glitch", "glitching", "glitches", "junk",
    "garbage", "trash", "abysmal", "tore", "tear", "tears", "ripped",
    "rip", "heat", "nightmare", "disgusted", "disgusting", "crap",
    "faulty", "died", "late", "delayed", "weak", "fake",
}

POSITIVE_WORDS = {
    "amazing", "best", "comfortable", "durable", "easy", "excellent",
    "fast", "great", "love", "loved", "perfect", "premium", "reliable",
    "smooth", "solid", "strong", "useful", "value", "worth", "quality",
    "clean", "fresh", "quick", "responsive", "sturdy", "satisfied",
}

COMPLAINT_PHRASES = [
    "stopped working", "does not work", "did not work", "won't work",
    "not worth", "broke after", "broke within", "nothing but problems",
    "do not buy", "do not purchase", "waste of money", "would not recommend",
    "fell apart", "came apart", "stopped charging", "cracked screen",
    "screen cracked", "screen broke", "never again", "rip off", "started glitching",
    "broke immediately", "too hot", "gets too hot", "loud fan", "fan is loud",
    "over heat", "over heating", "so slow", "poor quality", "terrible quality",
    "broke easily", "torn seam", "fabric tore", "bent frame", "no repair",
    "no warranty", "doesn't last", "didn't last", "late delivery",
]


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
        self.using_heuristic = False

        all_provided = all([
            self.model is not None,
            self.feature_cols is not None,
            self.low_classifier is not None,
            self.low_threshold is not None,
            self.high_threshold is not None,
        ])
        if not all_provided and not self._artifacts_exist():
            if self.settings.heuristic_fallback:
                self.using_heuristic = True
                return
            raise RuntimeError(
                "Rating artifacts are missing and VOLT_HEURISTIC_FALLBACK=false. "
                "Run the training scripts or enable the fallback."
            )

        if not all_provided:
            try:
                import joblib
            except ModuleNotFoundError as exc:
                if self.settings.heuristic_fallback:
                    self.using_heuristic = True
                    return
                raise RuntimeError("joblib is required for local serving. Run uv sync.") from exc

        try:
            if self.model is None:
                self.model = joblib.load(self.settings.rating_model_path)
            if self.feature_cols is None:
                self.feature_cols = joblib.load(self.settings.rating_feature_cols_path)
            if self.low_classifier is None:
                self.low_classifier = joblib.load(self.settings.rating_low_classifier_path)
            if self.low_threshold is None:
                self.low_threshold = float(
                    joblib.load(self.settings.rating_classifier_low_threshold_path)
                )
            if self.high_threshold is None:
                self.high_threshold = float(
                    joblib.load(self.settings.rating_classifier_high_threshold_path)
                )
        except Exception as exc:
            if self.settings.heuristic_fallback:
                self.using_heuristic = True
                return
            raise RuntimeError(
                "Unable to load one or more trained rating artifacts."
            ) from exc

    def _artifacts_exist(self) -> bool:
        required = [
            self.settings.rating_model_path,
            self.settings.rating_feature_cols_path,
            self.settings.rating_low_classifier_path,
            self.settings.rating_classifier_low_threshold_path,
            self.settings.rating_classifier_high_threshold_path,
        ]
        return all(path.exists() for path in required)

    def predict_from_features(self, features: dict[str, Any]) -> float:
        """Predict a star rating in the 1.0-5.0 range from feature values.

        Uses two-stage inference:
          1. Ridge regressor predicts fine-grained rating.
          2. LogisticRegression classifier predicts probability of low (1-3 stars).
          3. If classifier confidence exceeds threshold, prediction is capped at 3.
        """
        features = self._augment_with_text_signals(features)
        if self.using_heuristic:
            return self._heuristic_rating(features)

        text = str(features.get("text", "") or "").lower()
        tokens = re.findall(r"[a-z0-9']+", text)

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
        title_compound = self._simple_sentiment(title_text)
        features["title_compound"] = title_compound

        review_text = str(features.get("text", "") or "").lower()
        features["has_but_flag"] = 1.0 if " but " in review_text else 0.0
        review_compound = self._simple_sentiment(review_text)
        features["title_review_sentiment_gap"] = abs(title_compound - review_compound)

        row = {
            col: features.get(col, "" if col == "text" else 0)
            for col in self.feature_cols
        }

        import pandas as pd

        input_df = pd.DataFrame([row])

        raw_prediction = self.model.predict(input_df)[0]
        rating = float(rating_inverse_transform_inv([raw_prediction])[0])

        low_probs = self.low_classifier.predict_proba(input_df)
        low_prob = float(low_probs[0][1])
        rating = apply_low_rating_override(
            rating,
            low_prob,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )

        return _clip(float(rating), 1.0, 5.0)

    @staticmethod
    def _simple_sentiment(text: str) -> float:
        tokens = re.findall(r"[a-z0-9']+", str(text or "").lower())
        if not tokens:
            return 0.0
        positive = sum(1 for token in tokens if token in POSITIVE_WORDS)
        negative = sum(1 for token in tokens if token in NEGATIVE_WORDS)
        phrase_penalty = sum(1 for phrase in COMPLAINT_PHRASES if phrase in text.lower())
        total = positive + negative + phrase_penalty
        if total == 0:
            return 0.0
        return _clip((positive - negative - phrase_penalty) / total, -1.0, 1.0)

    def _heuristic_rating(self, features: dict[str, Any]) -> float:
        text = str(features.get("text", "") or "")
        sentiment = self._simple_sentiment(text)
        complaint_hits = sum(1 for phrase in COMPLAINT_PHRASES if phrase in text.lower())

        quality = float(features.get("quality_signal", 0) or 0)
        service = float(features.get("service_signal", 0) or 0)
        value = float(features.get("value_signal", 0) or 0)
        usability = float(features.get("usability_signal", 0) or 0)
        price_level = float(features.get("price_level", 1) or 1)

        budget = float(features.get("budget_sensitivity", 0.5) or 0.5)
        service_sensitivity = float(features.get("service_sensitivity", 0.5) or 0.5)
        quality_sensitivity = float(features.get("quality_sensitivity", 0.65) or 0.65)
        strictness = float(features.get("strictness", 0.5) or 0.5)

        aspect_values = [
            float(features.get("aspect_quality", 0) or 0),
            float(features.get("aspect_price", 0) or 0),
            float(features.get("aspect_service", 0) or 0),
            float(features.get("aspect_value", 0) or 0),
            float(features.get("aspect_usability", 0) or 0),
            float(features.get("aspect_delivery", 0) or 0),
        ]
        aspect_score = sum(aspect_values) / len(aspect_values)

        weighted_fit = (
            (quality * (0.55 + quality_sensitivity))
            + (service * (0.35 + service_sensitivity))
            + (value * (0.35 + budget))
            + (usability * 0.75)
        ) / 4.0
        budget_penalty = max(0.0, price_level - 1.0) * budget * 0.45
        low_price_boost = max(0.0, 1.0 - price_level) * budget * 0.25
        complaint_penalty = min(1.3, complaint_hits * 0.35 + strictness * max(0.0, -sentiment))

        rating = (
            3.45
            + (weighted_fit * 1.35)
            + (aspect_score * 0.55)
            + (sentiment * 0.8)
            + low_price_boost
            - budget_penalty
            - complaint_penalty
        )
        return round(_clip(rating, 1.0, 5.0), 2)

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
