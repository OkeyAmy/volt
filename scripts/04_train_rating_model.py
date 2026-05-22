# ============================================================
# SCRIPT 04: Train Rating Model (Two-Stage: Ridge + Low-Rating Classifier)
# Environment: Google Colab
# Input: data/processed/rating_features_train.parquet
#        data/processed/rating_features_test.parquet
# Output: artifacts/rating_model.pkl              (Ridge regressor)
#         artifacts/rating_low_classifier.pkl      (low-rating binary classifier)
#         artifacts/rating_classifier_threshold.pkl (override threshold)
#         artifacts/rating_feature_cols.pkl
#         artifacts/metrics_rating.json
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_DATA_PATH = Path("data/processed/rating_features_train.parquet")
TEST_DATA_PATH = Path("data/processed/rating_features_test.parquet")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

FEATURE_COLS = [
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
    "text",
]
NUMERIC_FEATURE_COLS = [col for col in FEATURE_COLS if col != "text"]

# ============================================================
# Transforms (shared with app/transforms.py)
# ============================================================


def _inverse_transform(y):
    """Transform target: 1/(6-star+0.1) spreads out low ratings."""
    return 1.0 / (6 - np.asarray(y, dtype=float) + 0.1)


def _inverse_transform_inv(y_pred):
    """Inverse transform: map predictions back to 1-5 scale."""
    raw = 6 - (1.0 / np.clip(np.asarray(y_pred, dtype=float), 0.01, 10) - 0.1)
    return np.clip(raw, 1, 5)


# ============================================================
# Metrics helpers
# ============================================================


def regression_metrics(y_true, predictions):
    """Return bounded rating regression metrics."""
    clipped = np.clip(predictions, 1, 5)
    rounded = np.rint(clipped).clip(1, 5)
    per_rating = {}
    for rating in sorted(pd.Series(y_true).unique()):
        mask = pd.Series(y_true).to_numpy() == rating
        per_rating[str(float(rating))] = {
            "count": int(mask.sum()),
            "prediction_mean": float(np.mean(clipped[mask])),
            "mae": float(np.mean(np.abs(clipped[mask] - pd.Series(y_true).to_numpy()[mask]))),
        }
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, clipped))),
        "mae": float(mean_absolute_error(y_true, clipped)),
        "rounded_accuracy": float(np.mean(rounded == pd.Series(y_true).to_numpy())),
        "macro_rating_mae": float(np.mean([item["mae"] for item in per_rating.values()])),
        "per_rating": per_rating,
    }


def class_distribution(values):
    """Return JSON-safe class counts for rating labels."""
    counts = pd.Series(values).value_counts().sort_index()
    return {str(float(label)): int(count) for label, count in counts.items()}


# ============================================================
# Model builders
# ============================================================


def _make_preprocessor():
    """Build the standard ColumnTransformer for rating features.

    Shared between the Ridge regressor and the low-rating classifier.
    Both models must see the same transformed feature space.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURE_COLS),
            (
                "text",
                TfidfVectorizer(
                    max_features=12000,
                    ngram_range=(1, 2),
                    min_df=3,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
                "text",
            ),
        ],
        sparse_threshold=0.3,
    )


def _make_ridge_pipeline():
    """Build the Stage-1 Ridge regressor pipeline."""
    return make_pipeline(
        _make_preprocessor(),
        RidgeCV(alphas=[0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]),
    )


def _make_classifier_pipeline():
    """Build the Stage-2 low-rating binary classifier pipeline.

    Uses the same preprocessor so the feature space aligns with Ridge.
    class_weight='balanced' counteracts the heavy 4/5-star class majority.
    """
    return make_pipeline(
        _make_preprocessor(),
        LogisticRegression(
            C=200,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_SEED,
        ),
    )


# ============================================================
# Find optimal override threshold on training data
# ============================================================


def _find_override_threshold_cv(X_train_df, y_train_series):
    """Find tiered override thresholds via 5-fold CV.

    Returns (low_threshold, high_threshold, cv_accuracy) where:
      prob > high  → cap at 2     (top tier)
      prob > low   → cap at 3     (standard tier)

    Grid searches low in [0.55, 0.85] and high in [low, 1.00).
    (1.0, 1.0) acts as a "no override" sentinel baseline.
    """
    y_train_arr = y_train_series.to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    low_candidates = [round(t, 2) for t in np.arange(0.55, 0.86, 0.03)]
    high_candidates = [round(t, 2) for t in np.arange(0.85, 1.00, 0.03)]
    # sentinel key (1.0, 1.0) = "no override" baseline
    fold_scores: dict[tuple[float, float], list[float]] = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y_train_arr)):
        X_ft, X_fv = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
        y_ft = y_train_arr[train_idx]
        y_fv = y_train_arr[val_idx]

        ridge_fold = _make_ridge_pipeline()
        ridge_fold.fit(X_ft, _inverse_transform(y_ft))
        ridge_val = np.rint(
            np.clip(_inverse_transform_inv(ridge_fold.predict(X_fv)), 1, 5)
        ).clip(1, 5)

        clf_fold = _make_classifier_pipeline()
        clf_fold.fit(X_ft, (y_ft <= 3).astype(int))
        prob_val = clf_fold.predict_proba(X_fv)[:, 1]

        for lc in low_candidates:
            for hc in high_candidates:
                if hc < lc:
                    continue
                key = (lc, hc)
                if key not in fold_scores:
                    fold_scores[key] = []

                p = ridge_val.copy()
                p[prob_val > hc] = np.minimum(p[prob_val > hc], 2)
                mask_mid = (prob_val > lc) & (prob_val <= hc)
                p[mask_mid] = np.minimum(p[mask_mid], 3)
                fold_scores[key].append(float(np.mean(p == y_fv)))

        # sentinel: (1.0, 1.0) triggers no overrides → baseline
        key_none = (1.0, 1.0)
        if key_none not in fold_scores:
            fold_scores[key_none] = []
        fold_scores[key_none].append(
            float(np.mean(ridge_val == y_fv))
        )

    mean_scores = {k: float(np.mean(v)) for k, v in fold_scores.items()}
    best_key = max(mean_scores, key=mean_scores.get)
    return float(best_key[0]), float(best_key[1]), mean_scores[best_key]


# ============================================================
# Main training
# ============================================================

print("Loading train and holdout test feature tables...")
train_df_full = pd.read_parquet(TRAIN_DATA_PATH)
test_df = pd.read_parquet(TEST_DATA_PATH)
print(f"  Train features: {train_df_full.shape}")
print(f"  Test features: {test_df.shape}")

train_df_full["text"] = train_df_full["text"].fillna("").astype(str)
test_df["text"] = test_df["text"].fillna("").astype(str)

X_train = train_df_full[FEATURE_COLS]
y_train = train_df_full["stars"]
y_train_transformed = _inverse_transform(y_train)

X_test = test_df[FEATURE_COLS]
y_test = test_df["stars"]

train_distribution = class_distribution(y_train)
test_distribution = class_distribution(y_test)
print(f"  Train distribution: {train_distribution}")
print(f"  Test distribution: {test_distribution}")

# -----------------------------------------------------------
# Stage 1: Ridge regressor (predicts fine-grained rating)
# -----------------------------------------------------------
print("\nTraining Stage 1 — Ridge regressor with inverse target transform...")
ridge_model = _make_ridge_pipeline()
ridge_model.fit(X_train, y_train_transformed)
best_alpha = ridge_model.named_steps["ridgecv"].alpha_
print(f"  RidgeCV selected alpha={best_alpha:.4f}")

train_pred_raw = ridge_model.predict(X_train)
test_pred_raw = ridge_model.predict(X_test)
train_pred = _inverse_transform_inv(train_pred_raw)
test_pred = _inverse_transform_inv(test_pred_raw)

train_metrics = regression_metrics(y_train, train_pred)
test_metrics = regression_metrics(y_test, test_pred)

# -----------------------------------------------------------
# Stage 2: Low-rating binary classifier
# -----------------------------------------------------------
y_low_train = (y_train <= 3).astype(int)
y_low_test = (y_test <= 3).astype(int)

print("\nTraining Stage 2 — Low-rating binary classifier (LR, C=200, balanced)...")
clf_model = _make_classifier_pipeline()
clf_model.fit(X_train, y_low_train)

clf_train_prob = clf_model.predict_proba(X_train)[:, 1]
clf_test_prob = clf_model.predict_proba(X_test)[:, 1]

classifier_train_acc = float(np.mean((clf_train_prob > 0.5) == y_low_train.to_numpy()))
classifier_test_acc = float(np.mean((clf_test_prob > 0.5) == y_low_test.to_numpy()))
print(f"  Classifier train accuracy (p>0.5): {classifier_train_acc:.4f}")
print(f"  Classifier test accuracy  (p>0.5): {classifier_test_acc:.4f}")

# -----------------------------------------------------------
# Find optimal override threshold
# -----------------------------------------------------------
print("\nFinding optimal override thresholds...")
low_threshold, high_threshold, cv_acc = _find_override_threshold_cv(
    X_train, y_train
)
print(f"  CV best: low={low_threshold:.2f} high={high_threshold:.2f} (cv acc: {cv_acc:.4f})")

ridge_test_rounded = np.rint(np.clip(test_pred, 1, 5)).clip(1, 5)
final_test_pred = ridge_test_rounded.copy()
high_mask = clf_test_prob > high_threshold
final_test_pred[high_mask] = np.minimum(final_test_pred[high_mask], 2)
mid_mask = (clf_test_prob > low_threshold) & (clf_test_prob <= high_threshold)
final_test_pred[mid_mask] = np.minimum(final_test_pred[mid_mask], 3)

override_count_high = int(high_mask.sum())
override_count_mid = int(mid_mask.sum())
override_count = override_count_high + override_count_mid
test_override_metrics = regression_metrics(y_test, final_test_pred)

print(f"\n  Test overrides: {override_count_high} cap-to-2 + {override_count_mid} cap-to-3 = {override_count} total")
print(f"  Accuracy break-down:")
pre_stage = (ridge_test_rounded == y_test.to_numpy()).sum()
post_stage = (final_test_pred == y_test.to_numpy()).sum()
print(f"    Ridge only:  {pre_stage}/{len(y_test)} = {pre_stage/len(y_test)*100:.2f}%")
print(f"    Two-stage:   {post_stage}/{len(y_test)} = {post_stage/len(y_test)*100:.2f}%")

# -----------------------------------------------------------
# Assemble metrics
# -----------------------------------------------------------
metrics = {
    "train_rmse": train_metrics["rmse"],
    "train_mae": train_metrics["mae"],
    "test_rmse": test_override_metrics["rmse"],
    "test_mae": test_override_metrics["mae"],
    "test_macro_rating_mae": test_override_metrics["macro_rating_mae"],
    "test_rounded_accuracy": test_override_metrics["rounded_accuracy"],
    "rating_rmse": test_override_metrics["rmse"],
    "rating_mae": test_override_metrics["mae"],
    "rmse_generalization_gap": float(
        test_override_metrics["rmse"] - train_metrics["rmse"]
    ),
    "num_train": int(len(X_train)),
    "num_test": int(len(X_test)),
    "model_type": (
        "Two-stage: TF-IDF bigram(1,2) text 12k-features + StandardScaler numeric "
        " + RidgeCV for fine-grained rating + LR(C=200) low-rating classifier override"
    ),
    "ridge_best_alpha": float(best_alpha),
    "classifier_C": 200,
    "classifier_low_threshold": low_threshold,
    "classifier_high_threshold": high_threshold,
    "override_count_test": override_count,
    "override_count_cap2": override_count_high,
    "override_count_cap3": override_count_mid,
    "ridge_only_accuracy": float(
        (ridge_test_rounded == y_test.to_numpy()).sum() / len(y_test)
    ),
    "imbalance_strategy": (
        "Inverse target transform 1/(6-star+0.1) for Ridge regressor + "
        "class_weight='balanced' for low-rating LogisticRegression classifier. "
        "Tiered inference: clf_prob > high → cap at 2, clf_prob > low → cap at 3."
    ),
    "train_distribution": train_distribution,
    "test_distribution": test_distribution,
    "text_max_features": 12000,
    "text_ngram_range": [1, 2],
    "text_min_df": 3,
    "numeric_feature_count": len(NUMERIC_FEATURE_COLS),
    "target_transform": "inverse 1/(6-star+0.1)",
    "test_per_rating": test_override_metrics["per_rating"],
    "random_seed": RANDOM_SEED,
}

joblib.dump(ridge_model, ARTIFACT_DIR / "rating_model.pkl")
joblib.dump(clf_model, ARTIFACT_DIR / "rating_low_classifier.pkl")
joblib.dump(low_threshold, ARTIFACT_DIR / "rating_classifier_low_threshold.pkl")
joblib.dump(high_threshold, ARTIFACT_DIR / "rating_classifier_high_threshold.pkl")
joblib.dump(FEATURE_COLS, ARTIFACT_DIR / "rating_feature_cols.pkl")
with open(ARTIFACT_DIR / "metrics_rating.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print(f"\nSaved artifacts (trained on {len(X_train)} rows):")
print(f"  {ARTIFACT_DIR / 'rating_model.pkl'}")
print(f"  {ARTIFACT_DIR / 'rating_low_classifier.pkl'}")
print(f"  {ARTIFACT_DIR / 'rating_classifier_low_threshold.pkl'}")
print(f"  {ARTIFACT_DIR / 'rating_classifier_high_threshold.pkl'}")
print(f"  {ARTIFACT_DIR / 'rating_feature_cols.pkl'}")
print(f"  {ARTIFACT_DIR / 'metrics_rating.json'}")

print("\n" + "=" * 60)
print("Retraining production model on ALL data (train + test)...")
print("=" * 60)

all_df = pd.concat([train_df_full, test_df], ignore_index=True)
all_df["text"] = all_df["text"].fillna("").astype(str)
X_all = all_df[FEATURE_COLS]
y_all = all_df["stars"]
y_all_transformed = _inverse_transform(y_all)

ridge_prod = _make_ridge_pipeline()
ridge_prod.fit(X_all, y_all_transformed)
prod_alpha = ridge_prod.named_steps["ridgecv"].alpha_

clf_prod = _make_classifier_pipeline()
clf_prod.fit(X_all, (y_all <= 3).astype(int))

prod_low, prod_high, prod_cv_acc = _find_override_threshold_cv(X_all, y_all)
print(f"  RidgeCV alpha: {prod_alpha:.4f}")
print(f"  Classifier CV thresholds: low={prod_low:.2f} high={prod_high:.2f} (cv acc: {prod_cv_acc:.4f})")

joblib.dump(ridge_prod, ARTIFACT_DIR / "rating_model.pkl")
joblib.dump(clf_prod, ARTIFACT_DIR / "rating_low_classifier.pkl")
joblib.dump(prod_low, ARTIFACT_DIR / "rating_classifier_low_threshold.pkl")
joblib.dump(prod_high, ARTIFACT_DIR / "rating_classifier_high_threshold.pkl")

print(f"\nOverwrote artifacts with production model (trained on {len(X_all)} rows):")
print(f"  {ARTIFACT_DIR / 'rating_model.pkl'}")
print(f"  {ARTIFACT_DIR / 'rating_low_classifier.pkl'}")
print(f"  {ARTIFACT_DIR / 'rating_classifier_low_threshold.pkl'}")
print(f"  {ARTIFACT_DIR / 'rating_classifier_high_threshold.pkl'}")

metrics["production_trained_on_all_data"] = True
metrics["production_num_rows"] = len(X_all)
metrics["production_ridge_alpha"] = float(prod_alpha)
metrics["production_classifier_low_threshold"] = prod_low
metrics["production_classifier_high_threshold"] = prod_high
metrics["production_cv_accuracy"] = prod_cv_acc
with open(ARTIFACT_DIR / "metrics_rating.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print(f"\n  {ARTIFACT_DIR / 'metrics_rating.json'} (updated)")

print("\n" + "=" * 60)
print("EVALUATION (trained on train split only):")
print(f"Ridge-only  test accuracy: "
      f"{(ridge_test_rounded == y_test.to_numpy()).sum()}/{len(y_test)} "
      f"= {pre_stage/len(y_test)*100:.2f}%")
print(f"Two-stage   test accuracy: "
      f"{(final_test_pred == y_test.to_numpy()).sum()}/{len(y_test)} "
      f"= {post_stage/len(y_test)*100:.2f}%")
print(f"Tiered thresholds: low={low_threshold:.2f} high={high_threshold:.2f} "
      f"({override_count_high} cap-to-2 + {override_count_mid} cap-to-3 = {override_count} overrides)")
print(f"PRODUCTION model retrained on all {len(X_all)} examples")
print("=" * 60)

for rating, stats in sorted(
    test_override_metrics["per_rating"].items(), key=lambda x: float(x[0])
):
    print(
        f"  stars={rating}: count={stats['count']}, "
        f"pred_mean={stats['prediction_mean']:.2f}, mae={stats['mae']:.3f}"
    )
print("SCRIPT 04 COMPLETE")
