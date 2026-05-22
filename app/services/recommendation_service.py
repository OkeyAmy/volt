"""Product recommendation service for Task B."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.config import Settings, get_settings


PRODUCT_FEATURE_DEFAULTS = {
    "product_name": "",
    "category": "",
    "description": "",
    "quality_signal": 0,
    "service_signal": 0,
    "value_signal": 0,
    "usability_signal": 0,
    "price_level": 1,
    "aspect_quality": 0,
    "aspect_price": 0,
    "aspect_service": 0,
    "aspect_value": 0,
    "aspect_usability": 0,
    "aspect_delivery": 0,
    "text": "",
}

# How many worker threads to use when scoring products in parallel.
_N_WORKERS = 8


DEFAULT_CATALOG = [
    {
        "product_id": "volt_power_bank_20000",
        "product_name": "20,000mAh Fast-Charge Power Bank",
        "category": "Electronics",
        "description": "Long battery backup for students, commuters, and frequent power cuts.",
        "quality_signal": 0.75,
        "service_signal": 0.35,
        "value_signal": 0.85,
        "usability_signal": 0.8,
        "price_level": 1,
        "aspect_quality": 1,
        "aspect_price": 1,
        "aspect_service": 0,
        "aspect_value": 1,
        "aspect_usability": 1,
        "aspect_delivery": 0,
    },
    {
        "product_id": "volt_budget_android",
        "product_name": "Budget Android Phone",
        "category": "Phones",
        "description": "Affordable phone with good battery life and basic camera performance.",
        "quality_signal": 0.45,
        "service_signal": 0.2,
        "value_signal": 0.9,
        "usability_signal": 0.65,
        "price_level": 0,
        "aspect_quality": 0,
        "aspect_price": 1,
        "aspect_service": 0,
        "aspect_value": 1,
        "aspect_usability": 1,
        "aspect_delivery": 0,
    },
    {
        "product_id": "volt_premium_headphones",
        "product_name": "Noise-Cancelling Wireless Headphones",
        "category": "Audio",
        "description": "Premium sound, comfortable pads, and strong noise reduction.",
        "quality_signal": 0.9,
        "service_signal": 0.35,
        "value_signal": 0.35,
        "usability_signal": 0.85,
        "price_level": 2,
        "aspect_quality": 1,
        "aspect_price": -1,
        "aspect_service": 0,
        "aspect_value": 0,
        "aspect_usability": 1,
        "aspect_delivery": 0,
    },
    {
        "product_id": "volt_rice_cooker",
        "product_name": "Compact Rice Cooker",
        "category": "Kitchen",
        "description": "Easy rice, beans, and stew prep for small kitchens and hostel rooms.",
        "quality_signal": 0.65,
        "service_signal": 0.25,
        "value_signal": 0.75,
        "usability_signal": 0.9,
        "price_level": 1,
        "aspect_quality": 1,
        "aspect_price": 1,
        "aspect_service": 0,
        "aspect_value": 1,
        "aspect_usability": 1,
        "aspect_delivery": 0,
    },
    {
        "product_id": "volt_campus_backpack",
        "product_name": "Water-Resistant Campus Backpack",
        "category": "Fashion",
        "description": "Durable laptop bag with padded straps and many compartments.",
        "quality_signal": 0.8,
        "service_signal": 0.2,
        "value_signal": 0.7,
        "usability_signal": 0.75,
        "price_level": 1,
        "aspect_quality": 1,
        "aspect_price": 0,
        "aspect_service": 0,
        "aspect_value": 1,
        "aspect_usability": 1,
        "aspect_delivery": 0,
    },
    {
        "product_id": "volt_skincare_set",
        "product_name": "Gentle Skincare Starter Set",
        "category": "Beauty",
        "description": "Mild cleanser and moisturizer for sensitive skin.",
        "quality_signal": 0.6,
        "service_signal": 0.3,
        "value_signal": 0.45,
        "usability_signal": 0.65,
        "price_level": 1,
        "aspect_quality": 1,
        "aspect_price": 0,
        "aspect_service": 0,
        "aspect_value": 0,
        "aspect_usability": 1,
        "aspect_delivery": 0,
    },
    {
        "product_id": "volt_jollof_spice",
        "product_name": "Jollof Rice Spice Pack",
        "category": "Food",
        "description": "Affordable spice blend for quick party-style jollof rice.",
        "quality_signal": 0.55,
        "service_signal": 0.15,
        "value_signal": 0.85,
        "usability_signal": 0.7,
        "price_level": 0,
        "aspect_quality": 0,
        "aspect_price": 1,
        "aspect_service": 0,
        "aspect_value": 1,
        "aspect_usability": 1,
        "aspect_delivery": 0,
    },
    {
        "product_id": "volt_movie_bundle",
        "product_name": "Nollywood Movie Night Bundle",
        "category": "Entertainment",
        "description": "Curated comedy and drama films for a relaxed weekend.",
        "quality_signal": 0.7,
        "service_signal": 0.25,
        "value_signal": 0.65,
        "usability_signal": 0.8,
        "price_level": 1,
        "aspect_quality": 1,
        "aspect_price": 0,
        "aspect_service": 0,
        "aspect_value": 1,
        "aspect_usability": 1,
        "aspect_delivery": 0,
    },
]


class RecommendationService:
    """Ranks catalog products for a persona using model scores and robustness."""

    def __init__(
        self,
        rating_service: Any,
        counterfactual_service: Any,
        settings: Settings | None = None,
        catalog: Any | None = None,
        ranker: Any | None = None,
        ranker_cols: list[str] | None = None,
    ):
        self.rating_service = rating_service
        self.counterfactual_service = counterfactual_service
        self.settings = settings or get_settings()
        self.using_default_catalog = False
        self.using_heuristic_ranker = False
        if catalog is None:
            if self.settings.product_catalog_path.exists():
                try:
                    import pandas as pd

                    catalog = pd.read_parquet(self.settings.product_catalog_path)
                except Exception as exc:
                    if not self.settings.heuristic_fallback:
                        raise RuntimeError(
                            "Unable to load the Colab-downloaded product catalog from "
                            f"{self.settings.product_catalog_path}."
                        ) from exc
                    catalog = DEFAULT_CATALOG
                    self.using_default_catalog = True
            else:
                if not self.settings.heuristic_fallback:
                    raise RuntimeError(
                        "Product catalog is missing and VOLT_HEURISTIC_FALLBACK=false."
                    )
                catalog = DEFAULT_CATALOG
                self.using_default_catalog = True
        self.catalog = catalog

        if ranker is None:
            if self.settings.ranker_model_path.exists():
                try:
                    import joblib

                    ranker = joblib.load(self.settings.ranker_model_path)
                except Exception as exc:
                    if not self.settings.heuristic_fallback:
                        raise RuntimeError(
                            "Unable to load the Colab-downloaded ranker model from "
                            f"{self.settings.ranker_model_path}."
                        ) from exc
                    self.using_heuristic_ranker = True
            else:
                if not self.settings.heuristic_fallback:
                    raise RuntimeError(
                        "Ranker model is missing and VOLT_HEURISTIC_FALLBACK=false."
                    )
                self.using_heuristic_ranker = True
        self.ranker = ranker

        self.ranker_cols = (
            ranker_cols
            if ranker_cols is not None
            else ([] if self.using_heuristic_ranker else self._load_ranker_cols())
        )

    # Number of top candidates to run full counterfactual analysis on.
    _DEEP_CANDIDATES = 200

    def recommend(
        self, persona: dict[str, Any], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Score and rank catalog products for a persona.

        Uses a two-phase approach for efficiency:
        Phase 1 scores all products with rating + ranker only.
        Phase 2 runs counterfactuals on the top candidates where they
        can affect the final ranking.
        """
        limit = max(1, int(top_k))
        n_deep = max(limit * 2, self._DEEP_CANDIDATES)

        catalog_items = list(self._iter_catalog())
        quick_rows = [None] * len(catalog_items)
        with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            fut_map = {}
            for idx, product in enumerate(catalog_items):
                fut = pool.submit(self._score_quick, persona, product)
                fut_map[fut] = idx
            for fut in as_completed(fut_map):
                quick_rows[fut_map[fut]] = fut.result()

        quick_rows = [r for r in quick_rows if r is not None]
        if not quick_rows:
            return []

        quick_rows.sort(
            key=lambda row: (row["predicted_rating"], row["_ranker_raw"]),
            reverse=True,
        )

        deep_candidates = quick_rows[:n_deep]
        deep_idx_map = {r["product_id"]: r for r in deep_candidates}

        with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            fut_map = {}
            for row in deep_candidates:
                fut = pool.submit(self._augment_deep, row, persona)
                fut_map[fut] = row["product_id"]
            for fut in as_completed(fut_map):
                pid = fut_map[fut]
                result = fut.result()
                if result is not None:
                    deep_idx_map[pid].update(result)

        self._apply_final_scores(deep_candidates)
        deep_candidates.sort(key=lambda r: r["final_score"], reverse=True)
        ranked = deep_candidates[:limit]
        for item in ranked:
            item.pop("_ranker_raw", None)
            item.pop("_product_data", None)
        return ranked

    # ------------------------------------------------------------------
    # Per-product scoring helpers (called from thread pool)
    # ------------------------------------------------------------------

    def _score_quick(
        self, persona: dict[str, Any], product: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Phase 1: rating + ranker only (2 calls)."""
        try:
            features = self._build_features(persona, product)
            predicted_rating = self.rating_service.predict_from_features(features)
            ranker_score = self._ranker_score(features)
            return {
                "product_id": str(product.get("product_id", product.get("item_id", ""))),
                "product_name": str(product.get("product_name", "")),
                "category": str(product.get("category", "")) or None,
                "_product_data": product,  # kept for Phase 2 counterfactuals
                "_ranker_raw": ranker_score,
                "predicted_rating": round(predicted_rating, 2),
                "ranker_score": round(ranker_score, 3),
                # Neutral defaults — overwritten in Phase 2 for top candidates
                "regret_risk": 0.0,
                "robustness": 1.0,
                "reason": f"Predicted {predicted_rating:.1f}/5 before counterfactual analysis.",
            }
        except Exception:
            return None

    def _augment_deep(
        self, row: dict[str, Any], persona: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Phase 2: run counterfactuals for a top candidate."""
        try:
            product = row.get("_product_data", row)
            features = self._build_features(persona, product)
            counterfactuals = self.counterfactual_service.run_counterfactuals(features)
            return {
                "regret_risk": counterfactuals["regret_risk"],
                "robustness": counterfactuals["robustness"],
                "reason": self._reason(
                    persona,
                    product,
                    row["predicted_rating"],
                    counterfactuals,
                ),
            }
        except Exception:
            return None

    @staticmethod
    def _build_features(persona: dict[str, Any], product: Any) -> dict[str, Any]:
        features = dict(persona)
        for column, default in PRODUCT_FEATURE_DEFAULTS.items():
            features[column] = product.get(column, default)
        if not features["text"]:
            features["text"] = " ".join(
                str(product.get(key, ""))
                for key in ("product_name", "category", "description")
                if product.get(key)
            )
        return features

    def _iter_catalog(self):
        if hasattr(self.catalog, "iterrows"):
            for _, product in self.catalog.iterrows():
                yield product.to_dict() if hasattr(product, "to_dict") else dict(product)
            return
        for product in self.catalog:
            yield product.to_dict() if hasattr(product, "to_dict") else dict(product)

    def _ranker_score(self, features: dict[str, Any]) -> float:
        if self.using_heuristic_ranker or self.ranker is None:
            return self._heuristic_ranker_score(features)

        import pandas as pd

        return float(
            self.ranker.predict(
                pd.DataFrame([{col: features.get(col, 0) for col in self.ranker_cols}])
            )[0]
        )

    @staticmethod
    def _heuristic_ranker_score(features: dict[str, Any]) -> float:
        quality = float(features.get("quality_signal", 0) or 0)
        service = float(features.get("service_signal", 0) or 0)
        value = float(features.get("value_signal", 0) or 0)
        usability = float(features.get("usability_signal", 0) or 0)
        price_level = float(features.get("price_level", 1) or 1)
        budget = float(features.get("budget_sensitivity", 0.5) or 0.5)
        service_sensitivity = float(features.get("service_sensitivity", 0.5) or 0.5)
        quality_sensitivity = float(features.get("quality_sensitivity", 0.65) or 0.65)

        return (
            (quality * (1.0 + quality_sensitivity))
            + (service * (0.5 + service_sensitivity))
            + (value * (0.75 + budget))
            + (usability * 0.8)
            - max(0.0, price_level - 1.0) * budget
        )

    @staticmethod
    def _reason(
        persona: dict[str, Any],
        product: Any,
        predicted_rating: float,
        counterfactuals: dict[str, Any],
    ) -> str:
        strengths = []
        if persona.get("budget_sensitivity", 0) >= 0.7 and product.get("price_level", 1) <= 1:
            strengths.append("fits a budget-conscious buyer")
        if persona.get("quality_sensitivity", 0) >= 0.7 and product.get("quality_signal", 0) > 0.6:
            strengths.append("has strong quality signals")
        if persona.get("service_sensitivity", 0) >= 0.7 and product.get("service_signal", 0) > 0.2:
            strengths.append("has acceptable service signals")
        if product.get("usability_signal", 0) > 0.7:
            strengths.append("looks easy to use")
        if not strengths:
            strengths.append("is the best overall match in the catalog")

        return (
            f"Predicted {predicted_rating:.1f}/5 because it "
            f"{', '.join(strengths[:3])}. "
            f"Robustness {counterfactuals['robustness']}, "
            f"regret risk {counterfactuals['regret_risk']}."
        )

    def _load_ranker_cols(self) -> list[str]:
        try:
            import joblib
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "joblib is required for local serving. Install requirements.txt."
            ) from exc

        try:
            return joblib.load(self.settings.ranker_feature_cols_path)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the Colab-downloaded ranker feature columns from "
                f"{self.settings.ranker_feature_cols_path}."
            ) from exc

    @staticmethod
    def _apply_final_scores(rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        raw_scores = [row["_ranker_raw"] for row in rows]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score

        for row in rows:
            if score_range == 0:
                normalized_ranker = 0.0
            else:
                normalized_ranker = (row["_ranker_raw"] - min_score) / score_range

            final_score = (
                0.50 * (row["predicted_rating"] / 5.0)
                + 0.30 * normalized_ranker
                + 0.20 * row["robustness"]
                - 0.20 * row["regret_risk"]
            )
            row["final_score"] = round(final_score, 3)
