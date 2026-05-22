"""Product recommendation service for Task B."""

from __future__ import annotations

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
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "pandas is required for local serving. Install requirements.txt."
                ) from exc

            try:
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

    def recommend(
        self, persona: dict[str, Any], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Score and rank catalog products for a persona."""

        import pandas as pd

        rows = []
        for _, product in self.catalog.iterrows():
            features = self._build_features(persona, product)
            predicted_rating = self.rating_service.predict_from_features(features)
            counterfactuals = self.counterfactual_service.run_counterfactuals(features)
            ranker_score = float(
                self.ranker.predict(
                    pd.DataFrame(
                        [{col: features.get(col, 0) for col in self.ranker_cols}]
                    )
                )[0]
            )

            rows.append(
                {
                    "product_id": str(product.get("product_id", product.get("item_id", ""))),
                    "product_name": str(product.get("product_name", "")),
                    "category": str(product.get("category", "")) or None,
                    "_ranker_raw": ranker_score,
                    "predicted_rating": round(predicted_rating, 2),
                    "ranker_score": round(ranker_score, 3),
                    "regret_risk": counterfactuals["regret_risk"],
                    "robustness": counterfactuals["robustness"],
                    "reason": (
                        f"Predicted {predicted_rating:.1f}/5. "
                        f"Robustness {counterfactuals['robustness']}, "
                        f"regret risk {counterfactuals['regret_risk']}."
                    ),
                }
            )

        self._apply_final_scores(rows)
        limit = max(0, int(top_k))
        ranked = sorted(rows, key=lambda item: item["final_score"], reverse=True)[:limit]
        for item in ranked:
            item.pop("_ranker_raw", None)
        return ranked

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
