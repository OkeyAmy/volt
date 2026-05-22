"""Gemini-backed review generation for Task A."""

from __future__ import annotations

import json
from typing import Any

from app.config import Settings, get_settings


REVIEW_PROMPT = """Generate a realistic Amazon product review.
Match the predicted rating sentiment. Do not mention AI or counterfactuals.

Persona: {persona}
Product: {product}
Predicted rating: {rating}/5
Counterfactual insight: {counterfactuals}

Return JSON only:
{{"rating": 1-5, "review": "text", "reasoning_summary": "text"}}"""


class ReviewGenerationError(RuntimeError):
    """Raised when Gemini cannot produce a valid review payload."""


class ReviewGenerator:
    """Generates natural-language reviews from model predictions."""

    def __init__(self, settings: Settings | None = None, model: Any | None = None):
        self.settings = settings or get_settings()
        self.model = model

    def _get_model(self) -> Any:
        if self.model is not None:
            return self.model
        if not self.settings.gemini_api_key:
            raise ReviewGenerationError(
                "GEMINI_API_KEY is required to generate reviews."
            )

        import google.generativeai as genai

        genai.configure(api_key=self.settings.gemini_api_key)
        self.model = genai.GenerativeModel(self.settings.gemini_model_name)
        return self.model

    def generate(
        self,
        persona: dict[str, Any],
        product: dict[str, Any],
        rating: float,
        counterfactuals: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a review and fail clearly if Gemini returns invalid JSON."""

        prompt = REVIEW_PROMPT.format(
            persona=json.dumps(persona, indent=2),
            product=json.dumps(product, indent=2),
            rating=round(rating, 2),
            counterfactuals=json.dumps(counterfactuals, indent=2),
        )
        response = self._get_model().generate_content(prompt)
        raw = getattr(response, "text", "").strip()
        parsed = self._parse_json(raw)
        return self._validate_review_payload(parsed)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else ""
            raw = raw.rsplit("```", 1)[0].strip()
            if raw.startswith("json"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else ""

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReviewGenerationError(
                "Gemini returned invalid JSON for review generation."
            ) from exc

        if not isinstance(parsed, dict):
            raise ReviewGenerationError("Gemini JSON response must be an object.")
        return parsed

    @staticmethod
    def _validate_review_payload(parsed: dict[str, Any]) -> dict[str, Any]:
        required = {"rating", "review", "reasoning_summary"}
        missing = required.difference(parsed)
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ReviewGenerationError(
                f"Gemini JSON response missing fields: {missing_fields}"
            )

        try:
            rating = float(parsed["rating"])
        except (TypeError, ValueError) as exc:
            raise ReviewGenerationError(
                "Gemini JSON response field 'rating' must be numeric."
            ) from exc
        if not 1 <= rating <= 5:
            raise ReviewGenerationError(
                "Gemini JSON response field 'rating' must be between 1 and 5."
            )

        review = parsed["review"]
        if not isinstance(review, str) or not review.strip():
            raise ReviewGenerationError(
                "Gemini JSON response field 'review' must be a non-empty string."
            )

        summary = parsed["reasoning_summary"]
        if not isinstance(summary, str) or not summary.strip():
            raise ReviewGenerationError(
                "Gemini JSON response field 'reasoning_summary' must be a non-empty string."
            )

        return {
            "rating": rating,
            "review": review.strip(),
            "reasoning_summary": summary.strip(),
        }
