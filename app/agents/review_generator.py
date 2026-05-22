"""Gemini-backed review generation for Task A."""

from __future__ import annotations

import json
from urllib import request
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

    def _generate_raw(self, prompt: str) -> str:
        if self.model is not None:
            response = self.model.generate_content(prompt)
            return getattr(response, "text", "").strip()

        if not self.settings.enable_llm_generation or not self.settings.gemini_api_key:
            raise ReviewGenerationError(
                "Gemini generation is not configured."
            )

        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.gemini_model_name}:generateContent"
            f"?key={self.settings.gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "responseMimeType": "application/json",
            },
        }
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=25) as response:
            data = json.loads(response.read().decode("utf-8"))

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ReviewGenerationError("Gemini response did not contain text.") from exc

    def generate(
        self,
        persona: dict[str, Any],
        product: dict[str, Any],
        rating: float,
        counterfactuals: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a review using Gemini when configured, otherwise deterministic text."""

        prompt = REVIEW_PROMPT.format(
            persona=json.dumps(persona, indent=2),
            product=json.dumps(product, indent=2),
            rating=round(rating, 2),
            counterfactuals=json.dumps(counterfactuals, indent=2),
        )
        try:
            raw = self._generate_raw(prompt)
            parsed = self._parse_json(raw)
            return self._validate_review_payload(parsed)
        except Exception:
            return self._fallback_review(persona, product, rating)

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

    @staticmethod
    def _fallback_review(
        persona: dict[str, Any],
        product: dict[str, Any],
        rating: float,
    ) -> dict[str, Any]:
        rounded = max(1, min(5, round(float(rating))))
        name = product.get("product_name") or product.get("category") or "the product"
        category = product.get("category") or "item"
        budget = float(persona.get("budget_sensitivity", 0.5) or 0.5)
        quality = float(persona.get("quality_sensitivity", 0.65) or 0.65)
        service = float(persona.get("service_sensitivity", 0.5) or 0.5)
        strictness = float(persona.get("strictness", 0.5) or 0.5)
        tone = int(persona.get("tone", 2) or 2)

        persona_bits = []
        if budget >= 0.7:
            persona_bits.append("price")
        if quality >= 0.7:
            persona_bits.append("durability")
        if service >= 0.7:
            persona_bits.append("delivery and support")
        if not persona_bits:
            persona_bits.append("overall usefulness")

        if rounded >= 5:
            review = (
                f"{name} really met my expectations. The {category.lower()} feels reliable, "
                f"easy to use, and worth the money. I would gladly buy it again."
            )
        elif rounded == 4:
            review = (
                f"I had a good experience with {name}. It does the important things well, "
                f"especially around {persona_bits[0]}, though there is still a little room "
                "for improvement."
            )
        elif rounded == 3:
            review = (
                f"{name} is okay, but it is not perfect. I like some parts of it, "
                f"but as someone who checks {', '.join(persona_bits)}, I noticed a few "
                "trade-offs before I could fully recommend it."
            )
        elif rounded == 2:
            review = (
                f"I wanted to like {name}, but it fell short for my needs. The main issue "
                f"is that it does not give enough confidence on {', '.join(persona_bits)}, "
                "so I would only consider it if there were no better options."
            )
        else:
            review = (
                f"{name} was disappointing. It missed the basics I care about and did not "
                "feel like a good use of money."
            )

        if tone == 0:
            review = review.split(".")[0] + "."
        elif tone == 3 and rounded <= 3:
            review += " I would not choose it again without clear improvements."
        elif tone == 4:
            review += (
                " The rating reflects both the product signals and how sensitive this "
                "persona is to small issues."
            )
        elif tone == 5 and rounded <= 2:
            review += " This was frustrating."

        summary = (
            f"Generated a {rounded}-star review from persona priorities "
            f"({', '.join(persona_bits)}) with strictness {strictness:.2f}."
        )
        return {
            "rating": float(rounded),
            "review": review,
            "reasoning_summary": summary,
        }
