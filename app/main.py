"""FastAPI entry point for Volt local serving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import Depends, FastAPI, Request

from app.agents.review_generator import ReviewGenerator
from app.config import Settings, get_settings
from app.schemas import (
    GenerateReviewRequest,
    GenerateReviewResponse,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
)
from app.services.counterfactual_service import CounterfactualService
from app.services.rating_service import RatingService
from app.services.recommendation_service import RecommendationService


class RatingPredictor(Protocol):
    def predict_from_features(self, features: dict[str, Any]) -> float:
        ...


class CounterfactualRunner(Protocol):
    def run_counterfactuals(self, features: dict[str, Any]) -> dict[str, Any]:
        ...


class Recommender(Protocol):
    def recommend(
        self, persona: dict[str, Any], top_k: int = 10
    ) -> list[dict[str, Any]]:
        ...


class ReviewWriter(Protocol):
    def generate(
        self,
        persona: dict[str, Any],
        product: dict[str, Any],
        rating: float,
        counterfactuals: dict[str, Any],
    ) -> dict[str, Any]:
        ...


@dataclass
class ServiceContainer:
    """Lazy service holder so imports do not require local artifacts."""

    settings: Settings
    rating_service: RatingPredictor | None = None
    counterfactual_service: CounterfactualRunner | None = None
    recommendation_service: Recommender | None = None
    review_generator: ReviewWriter | None = None

    def rating(self) -> RatingPredictor:
        if self.rating_service is None:
            self.rating_service = RatingService(self.settings)
        return self.rating_service

    def counterfactual(self) -> CounterfactualRunner:
        if self.counterfactual_service is None:
            self.counterfactual_service = CounterfactualService(self.rating())
        return self.counterfactual_service

    def recommendation(self) -> Recommender:
        if self.recommendation_service is None:
            self.recommendation_service = RecommendationService(
                self.rating(), self.counterfactual(), self.settings
            )
        return self.recommendation_service

    def review(self) -> ReviewWriter:
        if self.review_generator is None:
            self.review_generator = ReviewGenerator(self.settings)
        return self.review_generator


def create_app(container: ServiceContainer | None = None) -> FastAPI:
    """Create a dependency-friendly FastAPI app."""

    app = FastAPI(title="Volt API")
    app.state.services = container or ServiceContainer(get_settings())

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/presets", summary="Get example personas and products to copy-paste into Task A / Task B")
    def presets() -> dict:
        """Return ready-to-use examples that users can copy-paste into the API.

        Each preset shows meaningful parameter values with real-world scenarios.
        Use these to understand what values to provide or as starting points.
        """
        return {
            "personas": [
                {
                    "name": "Budget Betty — penny-pincher who just wants things cheap",
                    "values": {
                        "budget_sensitivity": 0.9,
                        "service_sensitivity": 0.3,
                        "quality_sensitivity": 0.2,
                        "strictness": 0.2,
                        "tone": 2,
                        "review_length": 20,
                    },
                    "explanation": "High budget_sensitivity means price is everything. Low quality_sensitivity + low strictness = easy to please as long as it is cheap.",
                },
                {
                    "name": "Quality Quinn — picky perfectionist who only accepts the best",
                    "values": {
                        "budget_sensitivity": 0.2,
                        "service_sensitivity": 0.6,
                        "quality_sensitivity": 0.95,
                        "strictness": 0.8,
                        "tone": 4,
                        "review_length": 80,
                    },
                    "explanation": "Very high quality_sensitivity + strictness → notices every flaw and writes long detailed reviews. Budget? Does not matter.",
                },
                {
                    "name": "Average Adam — typical casual reviewer",
                    "values": {
                        "budget_sensitivity": 0.5,
                        "service_sensitivity": 0.5,
                        "quality_sensitivity": 0.5,
                        "strictness": 0.4,
                        "tone": 2,
                        "review_length": 40,
                    },
                    "explanation": "Everything at 0.5 = balanced, moderate expectations. Good default starting point.",
                },
                {
                    "name": "Frustrated Frank — angry customer, everything is terrible",
                    "values": {
                        "budget_sensitivity": 0.7,
                        "service_sensitivity": 0.9,
                        "quality_sensitivity": 0.9,
                        "strictness": 0.95,
                        "tone": 5,
                        "review_length": 60,
                    },
                    "explanation": "Maxed strictness + angry tone + high everything = a scathing review with low predicted rating.",
                },
                {
                    "name": "Easy Eddie — loves everything, never complains",
                    "values": {
                        "budget_sensitivity": 0.1,
                        "service_sensitivity": 0.1,
                        "quality_sensitivity": 0.1,
                        "strictness": 0.05,
                        "tone": 1,
                        "review_length": 15,
                    },
                    "explanation": "Everything near zero except tone=1 (polite). This persona almost always gives 5 stars.",
                },
            ],
            "products": [
                {
                    "name": "Cheap Bluetooth Speaker — poor quality, great price",
                    "values": {
                        "product_name": "SoundBox Mini Bluetooth Speaker",
                        "category": "Electronics",
                        "quality_signal": 0.2,
                        "service_signal": 0.0,
                        "value_signal": 0.8,
                        "usability_signal": 0.7,
                        "price_level": 0,
                        "aspect_quality": 0,
                        "aspect_price": 1,
                        "aspect_service": 0,
                        "aspect_value": 1,
                        "aspect_usability": 1,
                        "aspect_delivery": 0,
                        "product_text": "The sound is okay for the price but the build feels cheap and the battery does not last long.",
                    },
                    "explanation": "Low quality_signal (0.2) but high value_signal (0.8) → product is cheap and that is its only appeal. price_level=0 (budget).",
                },
                {
                    "name": "Premium Headphones — excellent but expensive",
                    "values": {
                        "product_name": "AudioPro ANC Wireless Headphones",
                        "category": "Electronics",
                        "quality_signal": 0.9,
                        "service_signal": 0.5,
                        "value_signal": 0.3,
                        "usability_signal": 0.85,
                        "price_level": 2,
                        "aspect_quality": 1,
                        "aspect_price": -1,
                        "aspect_service": 0,
                        "aspect_value": 0,
                        "aspect_usability": 1,
                        "aspect_delivery": 0,
                        "product_text": "Incredible sound quality and noise cancellation but you will pay a premium for it.",
                    },
                    "explanation": "High quality_signal (0.9), high price_level (2=premium), low value_signal (0.3) → great product, expensive, questionable value.",
                },
                {
                    "name": "Terrible Customer Service Experience",
                    "values": {
                        "product_name": "QuickFix Phone Repair Service",
                        "category": "Electronics",
                        "quality_signal": 0.3,
                        "service_signal": -0.8,
                        "value_signal": 0.2,
                        "usability_signal": 0.5,
                        "price_level": 1,
                        "aspect_quality": 0,
                        "aspect_price": 0,
                        "aspect_service": -1,
                        "aspect_value": 0,
                        "aspect_usability": 1,
                        "aspect_delivery": -1,
                        "product_text": "The repair itself was fine but the staff were rude and it took three weeks longer than promised.",
                    },
                    "explanation": "Very low service_signal (-0.8) + negative aspect_service + negative delivery → service-sensitivity personas will hate this.",
                },
            ],
        }

    @app.post("/task-a/generate-review", response_model=GenerateReviewResponse)
    def generate_review(
        req: GenerateReviewRequest,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        """Task A: predict rating, run counterfactuals, and generate review."""

        persona = req.persona_features.to_feature_dict()
        product = req.product_features.to_feature_dict()
        features = {**persona, **product}
        predicted = services.rating().predict_from_features(features)
        counterfactuals = services.counterfactual().run_counterfactuals(features)
        review = services.review().generate(
            persona=persona,
            product=product,
            rating=predicted,
            counterfactuals=counterfactuals,
        )

        return {
            "predicted_rating": round(predicted, 2),
            "rounded_rating": round(predicted),
            "counterfactuals": counterfactuals,
            "generated_review": review,
        }

    @app.post("/task-b/recommend", response_model=RecommendResponse)
    def recommend(
        req: RecommendRequest,
        services: ServiceContainer = Depends(get_services),
    ) -> dict[str, Any]:
        """Task B: get personalized product recommendations for a persona."""

        return {
            "recommendations": services.recommendation().recommend(
                req.persona_features.to_feature_dict(), req.top_k
            )
        }

    return app


def get_services(request: Request) -> ServiceContainer:
    return request.app.state.services


app = create_app()
