# ============================================================
# SCRIPT 07: Model Diagnostics And Sampling Audit
# Environment: Local or Google Colab
# Input: artifacts/rating_model.pkl
#        artifacts/rating_feature_cols.pkl
#        data/processed/rating_features_train.parquet
#        data/processed/rating_features_test.parquet
# Output: data/processed/model_sample_audit.csv
#         data/processed/model_sample_summary.json
# ============================================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROC_DIR = Path("data/processed")
ARTIFACT_DIR = Path("artifacts")
SAMPLE_PER_RATING = 3


def split_metrics(scored):
    """Return overall and per-rating diagnostics for one scored split."""
    correct_mask = scored["rounded_prediction"] == scored["stars"]
    correct_count = int(correct_mask.sum())
    total_count = int(len(scored))
    incorrect_count = total_count - correct_count

    per_rating = {}
    for rating, group in scored.groupby("stars"):
        rating_correct = int((group["rounded_prediction"] == group["stars"]).sum())
        rating_total = int(len(group))
        per_rating[str(float(rating))] = {
            "count": rating_total,
            "correct": rating_correct,
            "incorrect": rating_total - rating_correct,
            "pass_rate": float(rating_correct / rating_total) if rating_total else 0.0,
            "sample_count": int(min(SAMPLE_PER_RATING, rating_total)),
            "mae": float(group["absolute_error"].mean()),
            "prediction_mean": float(group["predicted_rating"].mean()),
        }

    return {
        "rows": total_count,
        "correct": correct_count,
        "incorrect": incorrect_count,
        "pass_rate": float(correct_count / total_count) if total_count else 0.0,
        "rating_distribution": {
            str(float(label)): int(count)
            for label, count in scored["stars"].value_counts().sort_index().items()
        },
        "rmse": float(mean_squared_error(scored["stars"], scored["predicted_rating"]) ** 0.5),
        "mae": float(mean_absolute_error(scored["stars"], scored["predicted_rating"])),
        "rounded_accuracy": float(correct_count / total_count) if total_count else 0.0,
        "per_rating": per_rating,
    }


def _inverse_transform_inv(y_pred):
    """Inverse transform: map predictions back to 1-5 scale."""
    raw = 6 - (1.0 / np.clip(np.asarray(y_pred, dtype=float), 0.01, 10) - 0.1)
    return np.clip(raw, 1, 5)


def score_split(model, feature_cols, path):
    """Load one feature split and attach model predictions."""
    df = pd.read_parquet(path).copy()
    df["predicted_rating"] = _inverse_transform_inv(model.predict(df[feature_cols]))
    df["rounded_prediction"] = np.rint(df["predicted_rating"]).clip(1, 5)
    df["absolute_error"] = (df["predicted_rating"] - df["stars"]).abs()
    return df


print("Loading rating model and feature columns...")
rating_model = joblib.load(ARTIFACT_DIR / "rating_model.pkl")
feature_cols = joblib.load(ARTIFACT_DIR / "rating_feature_cols.pkl")

splits = {
    "train": PROC_DIR / "rating_features_train.parquet",
    "test": PROC_DIR / "rating_features_test.parquet",
}

sample_frames = []
summary = {}
for split_name, split_path in splits.items():
    print(f"Scoring {split_name} split from {split_path}...")
    scored_split = score_split(rating_model, feature_cols, split_path)
    summary[split_name] = split_metrics(scored_split)
    for _, group in scored_split.groupby("stars"):
        sample_frames.append(
            group.sort_values("absolute_error", ascending=False)
            .head(SAMPLE_PER_RATING)
            .assign(split=split_name)
        )

sample_cols = [
    "split",
    "review_id",
    "user_id",
    "item_id",
    "category",
    "stars",
    "predicted_rating",
    "rounded_prediction",
    "absolute_error",
    "text",
    "quality_signal",
    "service_signal",
    "value_signal",
    "usability_signal",
    "strictness",
    "tone",
    "review_length",
]
sample_audit = pd.concat(sample_frames, ignore_index=True)[sample_cols]
audit_path = PROC_DIR / "model_sample_audit.csv"
summary_path = PROC_DIR / "model_sample_summary.json"
sample_audit.to_csv(audit_path, index=False)
with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

print(json.dumps(summary, indent=2))
for split_name, split_summary in summary.items():
    print(
        f"\n{split_name.upper()} label pass/fail "
        f"(rounded prediction == true stars): "
        f"{split_summary['correct']}/{split_summary['rows']} passed, "
        f"{split_summary['incorrect']} failed "
        f"({split_summary['pass_rate']:.1%})"
    )
    for rating, rating_stats in sorted(
        split_summary["per_rating"].items(), key=lambda item: float(item[0])
    ):
        print(
            f"  stars={rating}: "
            f"{rating_stats['correct']}/{rating_stats['count']} correct, "
            f"{rating_stats['incorrect']} wrong"
        )
print(f"Saved: {audit_path}")
print(f"Saved: {summary_path}")
print("SCRIPT 07 COMPLETE")
