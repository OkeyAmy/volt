"""Configuration and path helpers for the local serving app."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Runtime settings for local serving.

    Paths are intentionally resolved without checking for file existence so
    importing the app remains safe before Colab artifacts have been downloaded.
    """

    project_root: Path
    data_dir: Path
    processed_data_dir: Path
    artifacts_dir: Path
    product_catalog_path: Path
    rating_model_path: Path
    rating_feature_cols_path: Path
    rating_low_classifier_path: Path
    rating_classifier_low_threshold_path: Path
    rating_classifier_high_threshold_path: Path
    ranker_model_path: Path
    ranker_feature_cols_path: Path
    gemini_api_key: str | None
    gemini_model_name: str
    heuristic_fallback: bool
    enable_llm_generation: bool


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load environment-backed settings once per process."""

    load_dotenv(PROJECT_ROOT / ".env")
    data_dir = PROJECT_ROOT / "data"
    processed_data_dir = data_dir / "processed"
    artifacts_dir = PROJECT_ROOT / "artifacts"

    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        processed_data_dir=processed_data_dir,
        artifacts_dir=artifacts_dir,
        product_catalog_path=processed_data_dir / "product_catalog.parquet",
        rating_model_path=artifacts_dir / "rating_model.pkl",
        rating_feature_cols_path=artifacts_dir / "rating_feature_cols.pkl",
        rating_low_classifier_path=artifacts_dir / "rating_low_classifier.pkl",
        rating_classifier_low_threshold_path=artifacts_dir / "rating_classifier_low_threshold.pkl",
        rating_classifier_high_threshold_path=artifacts_dir / "rating_classifier_high_threshold.pkl",
        ranker_model_path=artifacts_dir / "ranker_model.pkl",
        ranker_feature_cols_path=artifacts_dir / "ranker_feature_cols.pkl",
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        heuristic_fallback=os.getenv("VOLT_HEURISTIC_FALLBACK", "true").lower()
        not in {"0", "false", "no"},
        enable_llm_generation=os.getenv("VOLT_ENABLE_LLM", "true").lower()
        not in {"0", "false", "no"},
    )
