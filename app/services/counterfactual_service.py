"""Counterfactual sensitivity analysis for predicted ratings."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class CounterfactualService:
    """Measures rating sensitivity to product feature changes."""

    def __init__(self, rating_service: Any):
        self.rating_service = rating_service

    def run_counterfactuals(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run perturbations and return regret-risk and robustness scores."""

        original = float(self.rating_service.predict_from_features(features))
        tests = []

        perturbations = [
            ("price_level", 1, "price increased"),
            ("service_signal", -0.5, "service worsened"),
            ("usability_signal", -0.5, "usability worsened"),
            ("value_signal", -0.5, "value decreased"),
            ("quality_signal", 0.5, "quality improved"),
        ]

        for feature_name, delta, label in perturbations:
            if feature_name not in features:
                continue

            modified = deepcopy(features)
            if feature_name == "price_level":
                modified[feature_name] = min(2, modified[feature_name] + delta)
            else:
                shifted = modified[feature_name] + delta
                modified[feature_name] = float(max(-1, min(1, shifted)))

            new_rating = float(self.rating_service.predict_from_features(modified))
            tests.append(
                {
                    "change": label,
                    "new_rating": round(new_rating, 2),
                    "rating_shift": round(new_rating - original, 2),
                }
            )

        negative_shifts = [
            abs(test["rating_shift"]) for test in tests if test["rating_shift"] < 0
        ]
        regret_risk = min(1.0, sum(negative_shifts) / 5.0)

        return {
            "original_rating": round(original, 2),
            "counterfactuals": tests,
            "regret_risk": round(regret_risk, 2),
            "robustness": round(1.0 - regret_risk, 2),
        }
