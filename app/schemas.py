"""Pydantic schemas for Volt API requests and responses."""

from __future__ import annotations

from enum import IntEnum
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class Tone(IntEnum):
    brief = 0
    polite = 1
    casual = 2
    direct = 3
    detailed = 4
    angry = 5


class PriceLevel(IntEnum):
    low = 0
    medium = 1
    high = 2


class AspectSentiment(IntEnum):
    negative = -1
    neutral = 0
    positive = 1


class FeatureModel(StrictBaseModel):
    def to_feature_dict(self) -> dict[str, Any]:
        dump = getattr(self, "model_dump", None)
        if dump is not None:
            data = dump()
        else:
            data = self.dict()
        return {
            key: int(value) if isinstance(value, IntEnum) else value
            for key, value in data.items()
        }


class PersonaFeatures(FeatureModel):
    budget_sensitivity: float = Field(..., ge=0.0, le=1.0)
    service_sensitivity: float = Field(..., ge=0.0, le=1.0)
    quality_sensitivity: float = Field(..., ge=0.0, le=1.0)
    strictness: float = Field(..., ge=0.0, le=1.0)
    tone: Tone
    review_length: int = Field(..., ge=1, le=5000)


class ProductFeatures(FeatureModel):
    product_name: str | None = Field(default=None, max_length=300)
    category: str | None = Field(default=None, max_length=120)
    description: str | None = Field(default=None, max_length=5000)
    quality_signal: float = Field(..., ge=-1.0, le=1.0)
    service_signal: float = Field(..., ge=-1.0, le=1.0)
    value_signal: float = Field(..., ge=-1.0, le=1.0)
    usability_signal: float = Field(..., ge=-1.0, le=1.0)
    price_level: PriceLevel
    aspect_quality: AspectSentiment
    aspect_price: AspectSentiment
    aspect_service: AspectSentiment
    aspect_value: AspectSentiment
    aspect_usability: AspectSentiment
    aspect_delivery: AspectSentiment
    product_text: str | None = Field(default=None, max_length=5000)

    def to_feature_dict(self) -> dict[str, Any]:
        data = super().to_feature_dict()
        text_parts = [
            data.get("product_name") or "",
            data.get("category") or "",
            data.get("description") or "",
            data.get("product_text") or "",
        ]
        data["text"] = " ".join(part.strip() for part in text_parts if part.strip())
        return data


class HealthResponse(BaseModel):
    status: str


POSITIVE_HINTS = {
    "amazing",
    "best",
    "comfortable",
    "durable",
    "easy",
    "excellent",
    "fast",
    "great",
    "love",
    "perfect",
    "premium",
    "reliable",
    "smooth",
    "solid",
    "strong",
    "useful",
    "value",
    "worth",
}

NEGATIVE_HINTS = {
    "bad",
    "broken",
    "cheap",
    "complaint",
    "cracked",
    "defective",
    "delayed",
    "difficult",
    "disappointed",
    "expensive",
    "failed",
    "flimsy",
    "late",
    "poor",
    "problem",
    "refund",
    "slow",
    "terrible",
    "unreliable",
    "waste",
    "weak",
    "worst",
}


def _extra(model: BaseModel, *keys: str) -> Any:
    extras = getattr(model, "model_extra", None) or {}
    for key in keys:
        if key in extras:
            return extras[key]
    return None


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, FeatureModel):
        return value.to_feature_dict()
    if isinstance(value, BaseModel):
        dump = getattr(value, "model_dump", None)
        return dump() if dump is not None else value.dict()
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return {"description": value}
    return {}


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(str(part) for part in value.values() if part is not None)
    return str(value or "")


def _float(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))


def _enum_value(value: Any, enum_cls: type[IntEnum], default: IntEnum) -> int:
    if isinstance(value, enum_cls):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in enum_cls.__members__:
            return int(enum_cls[normalized])
        try:
            return int(enum_cls(int(normalized)))
        except (TypeError, ValueError):
            return int(default)
    try:
        return int(enum_cls(int(value)))
    except (TypeError, ValueError):
        return int(default)


def _keyword_score(text: str) -> float:
    tokens = set(re.findall(r"[a-z0-9']+", text.lower()))
    positive = len(tokens.intersection(POSITIVE_HINTS))
    negative = len(tokens.intersection(NEGATIVE_HINTS))
    if positive == negative == 0:
        return 0.0
    return max(-1.0, min(1.0, (positive - negative) / max(positive + negative, 1)))


def _persona_from_text(text: str) -> dict[str, Any]:
    lower = text.lower()
    budget_words = {"budget", "cheap", "affordable", "student", "price", "cost", "deal"}
    service_words = {"delivery", "support", "refund", "seller", "warranty", "service"}
    quality_words = {"quality", "durable", "premium", "reliable", "strong", "last"}
    strict_words = {"strict", "picky", "careful", "detailed", "complain", "high standard"}

    def has_any(words: set[str]) -> bool:
        return any(word in lower for word in words)

    tone = Tone.detailed if "detail" in lower else Tone.casual
    if any(word in lower for word in ("angry", "frustrated", "annoyed")):
        tone = Tone.angry
    elif any(word in lower for word in ("direct", "straight")):
        tone = Tone.direct
    elif "brief" in lower:
        tone = Tone.brief
    elif any(word in lower for word in ("polite", "calm")):
        tone = Tone.polite

    return {
        "budget_sensitivity": 0.85 if has_any(budget_words) else 0.5,
        "service_sensitivity": 0.8 if has_any(service_words) else 0.45,
        "quality_sensitivity": 0.9 if has_any(quality_words) else 0.65,
        "strictness": 0.8 if has_any(strict_words) else 0.5,
        "tone": int(tone),
        "review_length": 90 if tone == Tone.detailed else 55,
    }


def normalize_persona(value: Any) -> dict[str, Any]:
    if isinstance(value, PersonaFeatures):
        return value.to_feature_dict()

    data = _dump(value)
    text = _as_text(value)
    inferred = _persona_from_text(text)
    return {
        "budget_sensitivity": _float(
            data.get("budget_sensitivity", inferred["budget_sensitivity"]), 0.5, 0.0, 1.0
        ),
        "service_sensitivity": _float(
            data.get("service_sensitivity", inferred["service_sensitivity"]), 0.5, 0.0, 1.0
        ),
        "quality_sensitivity": _float(
            data.get("quality_sensitivity", inferred["quality_sensitivity"]), 0.65, 0.0, 1.0
        ),
        "strictness": _float(data.get("strictness", inferred["strictness"]), 0.5, 0.0, 1.0),
        "tone": _enum_value(data.get("tone", inferred["tone"]), Tone, Tone.casual),
        "review_length": int(_float(data.get("review_length", inferred["review_length"]), 60, 1, 5000)),
    }


def normalize_product(value: Any) -> dict[str, Any]:
    if isinstance(value, ProductFeatures):
        return value.to_feature_dict()

    data = _dump(value)
    text = _as_text(value)
    product_name = (
        data.get("product_name")
        or data.get("name")
        or data.get("title")
        or data.get("item_name")
        or "Product"
    )
    category = data.get("category") or data.get("domain") or "General"
    description = data.get("description") or data.get("details") or data.get("product_text") or text
    keyword_score = _keyword_score(" ".join([str(product_name), str(category), str(description)]))

    price_level = data.get("price_level")
    if price_level is None:
        price_text = str(data.get("price", "")).lower()
        if any(word in price_text for word in ("premium", "expensive", "high")):
            price_level = PriceLevel.high
        elif any(word in price_text for word in ("cheap", "low", "budget", "affordable")):
            price_level = PriceLevel.low
        else:
            price_level = PriceLevel.medium

    signal = keyword_score or 0.2
    aspect_default = 1 if signal > 0.25 else -1 if signal < -0.25 else 0
    return {
        "product_name": str(product_name),
        "category": str(category),
        "description": str(description),
        "quality_signal": _float(data.get("quality_signal", signal), signal, -1.0, 1.0),
        "service_signal": _float(data.get("service_signal", 0.0), 0.0, -1.0, 1.0),
        "value_signal": _float(data.get("value_signal", signal), signal, -1.0, 1.0),
        "usability_signal": _float(data.get("usability_signal", signal), signal, -1.0, 1.0),
        "price_level": _enum_value(price_level, PriceLevel, PriceLevel.medium),
        "aspect_quality": _enum_value(data.get("aspect_quality", aspect_default), AspectSentiment, AspectSentiment.neutral),
        "aspect_price": _enum_value(data.get("aspect_price", aspect_default), AspectSentiment, AspectSentiment.neutral),
        "aspect_service": _enum_value(data.get("aspect_service", 0), AspectSentiment, AspectSentiment.neutral),
        "aspect_value": _enum_value(data.get("aspect_value", aspect_default), AspectSentiment, AspectSentiment.neutral),
        "aspect_usability": _enum_value(data.get("aspect_usability", aspect_default), AspectSentiment, AspectSentiment.neutral),
        "aspect_delivery": _enum_value(data.get("aspect_delivery", 0), AspectSentiment, AspectSentiment.neutral),
        "product_text": str(data.get("product_text") or description),
        "text": " ".join(
            part.strip()
            for part in [str(product_name), str(category), str(description), str(data.get("product_text") or "")]
            if part and part.strip()
        ),
    }


class GenerateReviewRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    persona_features: PersonaFeatures | None = None
    product_features: ProductFeatures | None = None
    persona: Any | None = None
    product: Any | None = None
    user_persona: Any | None = None
    product_details: Any | None = None

    def persona_feature_dict(self) -> dict[str, Any]:
        source = (
            self.persona_features
            or self.persona
            or self.user_persona
            or _extra(self, "user", "profile")
        )
        return normalize_persona(source)

    def product_feature_dict(self) -> dict[str, Any]:
        source = (
            self.product_features
            or self.product
            or self.product_details
            or _extra(self, "item", "item_details")
        )
        return normalize_product(source)


class GeneratedReview(BaseModel):
    rating: float = Field(..., ge=1.0, le=5.0)
    review: str = Field(..., min_length=1)
    reasoning_summary: str = Field(..., min_length=1)


class CounterfactualResult(BaseModel):
    change: str
    new_rating: float
    rating_shift: float


class CounterfactualResponse(BaseModel):
    original_rating: float
    counterfactuals: list[CounterfactualResult]
    regret_risk: float
    robustness: float


class GenerateReviewResponse(BaseModel):
    predicted_rating: float
    rounded_rating: int
    counterfactuals: CounterfactualResponse
    generated_review: GeneratedReview


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    persona_features: PersonaFeatures | None = None
    persona: Any | None = None
    user_persona: Any | None = None
    top_k: int = Field(default=10, ge=1, le=50)

    def persona_feature_dict(self) -> dict[str, Any]:
        source = (
            self.persona_features
            or self.persona
            or self.user_persona
            or _extra(self, "user", "profile")
        )
        return normalize_persona(source)


class RecommendationItem(BaseModel):
    product_id: str
    product_name: str
    category: str | None = None
    predicted_rating: float
    ranker_score: float
    regret_risk: float
    robustness: float
    final_score: float
    reason: str


class RecommendResponse(BaseModel):
    recommendations: list[RecommendationItem]
