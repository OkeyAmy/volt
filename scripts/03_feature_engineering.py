# ============================================================
# SCRIPT 03: Feature Engineering Compatibility Check
# Environment: Google Colab
# Input: data/processed/rating_features.parquet
# Output: data/processed/rating_features.parquet
# Note: Script 02 now writes the final feature table directly.
# ============================================================

from pathlib import Path
import json

import pandas as pd


PROC_DIR = Path("data/processed")
FEATURE_PATHS = {
    "train": PROC_DIR / "rating_features_train.parquet",
    "test": PROC_DIR / "rating_features_test.parquet",
    "compatibility": PROC_DIR / "rating_features.parquet",
}
REPORT_PATH = PROC_DIR / "feature_validation_report.json"
MIN_VALID_ROWS = 90
REQUIRED_COLUMNS = [
    "review_id",
    "user_id",
    "item_id",
    "category",
    "text",
    "stars",
    "budget_sensitivity",
    "service_sensitivity",
    "quality_sensitivity",
    "strictness",
    "quality_signal",
    "service_signal",
    "value_signal",
    "usability_signal",
    "price_level",
    "tone",
    "aspect_quality",
    "aspect_price",
    "aspect_service",
    "aspect_value",
    "aspect_usability",
    "aspect_delivery",
    "review_length",
    "negativity_ratio",
    "complaint_phrases",
    "caps_intensity",
    "title_compound",
    "has_but_flag",
    "title_review_sentiment_gap",
]
FEATURE_COLUMNS = REQUIRED_COLUMNS[6:]
RANGE_RULES = {
    "stars": (1.0, 5.0),
    "budget_sensitivity": (0.0, 1.0),
    "service_sensitivity": (0.0, 1.0),
    "quality_sensitivity": (0.0, 1.0),
    "strictness": (0.0, 1.0),
    "quality_signal": (-1.0, 1.0),
    "service_signal": (-1.0, 1.0),
    "value_signal": (-1.0, 1.0),
    "usability_signal": (-1.0, 1.0),
    "price_level": (0.0, 2.0),
    "tone": (0.0, 5.0),
    "aspect_quality": (-1.0, 1.0),
    "aspect_price": (-1.0, 1.0),
    "aspect_service": (-1.0, 1.0),
    "aspect_value": (-1.0, 1.0),
    "aspect_usability": (-1.0, 1.0),
    "aspect_delivery": (-1.0, 1.0),
    "review_length": (1.0, 5000.0),
    "negativity_ratio": (0.0, 1.0),
    "complaint_phrases": (0.0, 1.0),
    "caps_intensity": (0.0, 1.0),
    "title_compound": (-1.0, 1.0),
    "has_but_flag": (0.0, 1.0),
    "title_review_sentiment_gap": (0.0, 2.0),
}


def validate_feature_table(name, path, min_rows):
    """Validate one feature parquet and return summary metadata."""
    if not path.exists():
        raise RuntimeError(
            f"{path} does not exist. Run scripts/02_feature_extraction.py before script 03."
        )

    print(f"Loading {name} feature table produced by script 02...")
    df = pd.read_parquet(path)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise RuntimeError(f"{name} feature table missing required columns: {missing_columns}")

    if len(df) < min_rows:
        raise RuntimeError(f"{name} feature table must contain at least {min_rows} rows.")

    if df[FEATURE_COLUMNS].isna().any().any():
        bad_columns = df[FEATURE_COLUMNS].columns[df[FEATURE_COLUMNS].isna().any()].tolist()
        raise RuntimeError(f"{name} feature table contains NaN values in: {bad_columns}")

    range_violations = {}
    for column, (lower, upper) in RANGE_RULES.items():
        below = int((df[column] < lower).sum())
        above = int((df[column] > upper).sum())
        if below or above:
            range_violations[column] = {"below": below, "above": above}
    if range_violations:
        raise RuntimeError(f"{name} feature table has range violations: {range_violations}")

    print(f"{name} feature table: {df.shape[0]} rows x {df.shape[1]} columns")
    return {
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "rating_distribution": {
            str(key): int(value)
            for key, value in df["stars"].value_counts().sort_index().items()
        },
    }


report = {
    name: validate_feature_table(
        name,
        path,
        MIN_VALID_ROWS if name != "test" else max(30, MIN_VALID_ROWS // 2),
    )
    for name, path in FEATURE_PATHS.items()
}
with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print(f"Saved validation report: {REPORT_PATH}")
print("SCRIPT 03 COMPLETE")
