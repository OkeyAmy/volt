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
