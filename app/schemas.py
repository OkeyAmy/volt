"""Pydantic schemas for Volt API requests and responses."""

from __future__ import annotations

from enum import IntEnum
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
            data.pop("product_name") or "",
            data.pop("category") or "",
            data.pop("description") or "",
            data.pop("product_text") or "",
        ]
        data["text"] = " ".join(part.strip() for part in text_parts if part.strip())
        return data


class HealthResponse(BaseModel):
    status: str


class GenerateReviewRequest(StrictBaseModel):
    persona_features: PersonaFeatures
    product_features: ProductFeatures


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


class RecommendRequest(StrictBaseModel):
    persona_features: PersonaFeatures
    top_k: int = Field(default=10, ge=1, le=50)


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
