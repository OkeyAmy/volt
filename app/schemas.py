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
    budget_sensitivity: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="How price-conscious is this user? 0 = doesn't care about price, 1 = extreme budget hunter. High values make the system favour cheaper products and penalise poor value.",
    )
    service_sensitivity: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="How much does the user care about customer service? 0 = service doesn't matter, 1 = bad service alone ruins the experience.",
    )
    quality_sensitivity: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="How much does the user care about product quality? 0 = quality doesn't matter, 1 = any flaw is unacceptable.",
    )
    strictness: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Overall pickiness. 0 = easy to please, 1 = hard to ever satisfy. Acts as a general severity multiplier across all dimensions.",
    )
    tone: Tone = Field(
        ...,
        description="Writing style: 0 = brief, 1 = polite, 2 = casual, 3 = direct, 4 = detailed, 5 = angry.",
    )
    review_length: int = Field(
        ...,
        ge=1, le=5000,
        description="How many words the generated review should be. Typical values: 15-30 for short, 50-100 for detailed.",
    )


class ProductFeatures(FeatureModel):
    product_name: str | None = Field(
        default=None, max_length=300,
        description="Product name, e.g. 'Bluetooth Speaker X200'. Used as context for review generation.",
    )
    category: str | None = Field(
        default=None, max_length=120,
        description="Product category, e.g. 'Electronics', 'Home', 'Books & Stationery'. Affects review language.",
    )
    description: str | None = Field(
        default=None, max_length=5000,
        description="Short product description. Combined with product_text for feature extraction.",
    )
    quality_signal: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Perceived build quality from review text. -1 = terrible quality, 0 = neutral, +1 = excellent quality.",
    )
    service_signal: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Perceived customer service quality from review text. -1 = horrible service, 0 = neutral, +1 = great service.",
    )
    value_signal: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Perceived value for money. -1 = overpriced/rip-off, 0 = fair price, +1 = amazing deal.",
    )
    usability_signal: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Perceived ease of use. -1 = impossible to use, 0 = average, +1 = very intuitive.",
    )
    price_level: PriceLevel = Field(
        ...,
        description="Price tier: 0 = low (budget), 1 = medium (mid-range), 2 = high (premium/luxury).",
    )
    aspect_quality: AspectSentiment = Field(
        ...,
        description="Review sentiment about product quality: -1 = negative, 0 = neutral, 1 = positive.",
    )
    aspect_price: AspectSentiment = Field(
        ...,
        description="Review sentiment about pricing: -1 = negative/too expensive, 0 = neutral, 1 = positive/good value.",
    )
    aspect_service: AspectSentiment = Field(
        ...,
        description="Review sentiment about customer service: -1 = negative, 0 = neutral, 1 = positive.",
    )
    aspect_value: AspectSentiment = Field(
        ...,
        description="Review sentiment about value for money: -1 = negative/overpriced, 0 = neutral, 1 = positive/good deal.",
    )
    aspect_usability: AspectSentiment = Field(
        ...,
        description="Review sentiment about ease of use: -1 = negative/hard to use, 0 = neutral, 1 = positive/easy to use.",
    )
    aspect_delivery: AspectSentiment = Field(
        ...,
        description="Review sentiment about delivery/shipping: -1 = negative/late, 0 = neutral, 1 = positive/fast.",
    )
    product_text: str | None = Field(
        default=None, max_length=5000,
        description="Raw review text or product description used for dynamic feature extraction (negativity, complaints, sentiment gaps). The model reads this to infer signals. Example: 'The battery drains fast and the screen has a scratch.'",
    )

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
