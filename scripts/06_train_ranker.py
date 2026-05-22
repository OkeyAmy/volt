# ============================================================
# SCRIPT 06: Train Recommendation Ranker
# Environment: Google Colab
# Input: data/processed/rating_features_train.parquet
#        data/processed/rating_features_test.parquet
# Output: artifacts/ranker_model.pkl
#         artifacts/ranker_feature_cols.pkl
#         artifacts/metrics_ranker.json
# ============================================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import ndcg_score


TRAIN_DATA_PATH = Path("data/processed/rating_features_train.parquet")
TEST_DATA_PATH = Path("data/processed/rating_features_test.parquet")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
NUM_NEGATIVES = 3

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
    "aspect_quality",
    "aspect_price",
    "aspect_service",
    "aspect_value",
    "aspect_usability",
    "aspect_delivery",
]


def stars_to_relevance(stars):
    """Convert star rating to relevance score for ranking."""
    if stars >= 4.5:
        return 3
    if stars >= 3.5:
        return 2
    if stars >= 2.5:
        return 1
    return 0


def build_ranking_pairs(df, split_name, rng):
    """Create positive and sampled-negative ranking rows for each user interaction."""
    rank_rows = []
    all_indices = np.array(df.index)
    for i, row in df.iterrows():
        user_query_id = f"{split_name}_user_{row['user_id']}_{i}"

        positive = row.copy()
        positive["query_user_id"] = user_query_id
        positive["relevance"] = stars_to_relevance(row["stars"])
        rank_rows.append(positive)

        candidate_indices = all_indices[all_indices != i]
        if len(candidate_indices) == 0:
            continue
        sample_size = min(NUM_NEGATIVES, len(candidate_indices))
        negative_indices = rng.choice(candidate_indices, size=sample_size, replace=False)
        for negative_index in negative_indices:
            negative = df.loc[negative_index].copy()
            negative["query_user_id"] = user_query_id
            negative["relevance"] = 0
            rank_rows.append(negative)

    return pd.DataFrame(rank_rows).sort_values("query_user_id")


def group_sizes(df):
    """Return query group sizes in the current sorted query order."""
    return df.groupby("query_user_id", sort=False).size().to_list()


def mean_ndcg_at_10(metric_df):
    """Compute mean NDCG@10 over queries with at least one relevant item."""
    scores = []
    for _, group in metric_df.groupby("query_user_id"):
        y_true = group["relevance"].to_numpy()
        y_score = group["prediction"].to_numpy()
        if len(group) > 1 and np.any(y_true > 0):
            scores.append(ndcg_score([y_true], [y_score], k=10))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_ranker(ranker_model, df):
    """Predict scores and return mean NDCG@10 for a ranking dataframe."""
    predictions = ranker_model.predict(df[FEATURE_COLS])
    metric_df = df[["query_user_id", "relevance"]].copy()
    metric_df["prediction"] = predictions
    return mean_ndcg_at_10(metric_df)


print("Loading train and holdout test feature tables...")
train_features = pd.read_parquet(TRAIN_DATA_PATH).reset_index(drop=True)
test_features = pd.read_parquet(TEST_DATA_PATH).reset_index(drop=True)
print(f"  Train features: {train_features.shape}")
print(f"  Test features: {test_features.shape}")

print("\nBuilding ranking pairs without test leakage...")
rng = np.random.default_rng(RANDOM_SEED)
train_df = build_ranking_pairs(train_features, "train", rng)
test_df = build_ranking_pairs(test_features, "test", rng)
print(f"  Train ranking pairs: {len(train_df)}")
print(f"  Test ranking pairs: {len(test_df)}")

print("\nTraining lean sklearn ranking scorer...")
ranker = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=15,
    l2_regularization=0.2,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=20,
    random_state=RANDOM_SEED,
)
ranker.fit(train_df[FEATURE_COLS], train_df["relevance"])

train_ndcg = evaluate_ranker(ranker, train_df)
test_ndcg = evaluate_ranker(ranker, test_df)

metrics = {
    "train_ndcg_at_10": train_ndcg,
    "test_ndcg_at_10": test_ndcg,
    "validation_ndcg_at_10": test_ndcg,
    "ndcg_generalization_gap": float(train_ndcg - test_ndcg),
    "train_query_count": int(train_df["query_user_id"].nunique()),
    "test_query_count": int(test_df["query_user_id"].nunique()),
    "train_row_count": int(len(train_df)),
    "test_row_count": int(len(test_df)),
    "num_negative_samples_per_positive": int(NUM_NEGATIVES),
    "model_type": "HistGradientBoostingRegressor relevance scorer",
    "best_iteration": int(getattr(ranker, "n_iter_", 0)),
    "random_seed": int(RANDOM_SEED),
}

joblib.dump(ranker, ARTIFACT_DIR / "ranker_model.pkl")
joblib.dump(FEATURE_COLS, ARTIFACT_DIR / "ranker_feature_cols.pkl")
with open(ARTIFACT_DIR / "metrics_ranker.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print(f"\nNDCG@10 train/test: {train_ndcg:.3f} / {test_ndcg:.3f}")
print(f"\nSaved: {ARTIFACT_DIR / 'ranker_model.pkl'}")
print(f"Saved: {ARTIFACT_DIR / 'ranker_feature_cols.pkl'}")
print(f"Saved: {ARTIFACT_DIR / 'metrics_ranker.json'}")
print("SCRIPT 06 COMPLETE")
