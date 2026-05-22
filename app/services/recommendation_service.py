"""Product recommendation service for Task B."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.config import Settings, get_settings


PRODUCT_FEATURE_DEFAULTS = {
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
        if catalog is None:
            try:
                import pandas as pd

                catalog = pd.read_parquet(self.settings.product_catalog_path)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load the Colab-downloaded product catalog from "
                    f"{self.settings.product_catalog_path}."
                ) from exc
        self.catalog = catalog

        if ranker is None:
            try:
                import joblib
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "joblib is required for local serving. Install requirements.txt."
                ) from exc

            try:
                ranker = joblib.load(self.settings.ranker_model_path)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load the Colab-downloaded ranker model from "
                    f"{self.settings.ranker_model_path}."
                ) from exc
        self.ranker = ranker

        self.ranker_cols = (
            ranker_cols
            if ranker_cols is not None
            else self._load_ranker_cols()
        )

    # Number of top candidates to run full counterfactual analysis on.
    _DEEP_CANDIDATES = 200

    def recommend(
        self, persona: dict[str, Any], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Score and rank catalog products for a persona.

        Uses a two-phase approach for efficiency:

        **Phase 1** — score *all* products with rating + ranker only
        (2 model calls per product, ~38 ms each, ~80 s sequential).
        This gives a preliminary ranking for every product.

        **Phase 2** — run full counterfactuals (6 calls) on the top
        *N* candidates where it matters for the final ranking.
        """
        limit = max(1, int(top_k))
        n_deep = max(limit * 2, self._DEEP_CANDIDATES)

        # ── Phase 1: quick score everything ──────────────────────────
        catalog_items = list(self.catalog.iterrows())
        quick_rows = [None] * len(catalog_items)
        with ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            fut_map = {}
            for idx, product in catalog_items:
                prod_dict = product.to_dict() if hasattr(product, "to_dict") else dict(product)
                fut = pool.submit(self._score_quick, persona, prod_dict)
                fut_map[fut] = idx
            for fut in as_completed(fut_map):
                quick_rows[fut_map[fut]] = fut.result()

        quick_rows = [r for r in quick_rows if r is not None]
        if not quick_rows:
            return []

        # Preliminary ranking by predicted rating alone.
        quick_rows.sort(key=lambda r: r["predicted_rating"], reverse=True)

        # ── Phase 2: deep score top candidates ───────────────────────
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

        # ── Final ranking (only deep-scored rows can be top-K) ───────
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
            import pandas as pd

            features = self._build_features(persona, product)
            predicted_rating = self.rating_service.predict_from_features(features)
            ranker_score = float(
                self.ranker.predict(
                    pd.DataFrame(
                        [{col: features.get(col, 0) for col in self.ranker_cols}]
                    )
                )[0]
            )
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
                "reason": (
                    f"Predicted {predicted_rating:.1f}/5. "
                    f"Robustness TBD, regret risk TBD."
                ),
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
                "reason": (
                    f"Predicted {row['predicted_rating']:.1f}/5. "
                    f"Robustness {counterfactuals['robustness']}, "
                    f"regret risk {counterfactuals['regret_risk']}."
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
            features["text"] = product.get("product_name", "")
        return features

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
