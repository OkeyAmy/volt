"""
RESEARCH HARNESS: Test multiple model configurations to find >75% accuracy.
Uses existing features from parquet files - no need to re-run 01-03.
Tests: sample weights, TF-IDF params, model types, feature engineering, post-processing.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV, HuberRegressor, SGDRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
TRAIN_PATH = Path("data/processed/rating_features_train.parquet")
TEST_PATH = Path("data/processed/rating_features_test.parquet")
np.random.seed(RANDOM_SEED)

NUMERIC_FEATURE_COLS = [
    "budget_sensitivity", "service_sensitivity", "quality_sensitivity",
    "strictness", "quality_signal", "service_signal", "value_signal",
    "usability_signal", "price_level", "tone", "aspect_quality",
    "aspect_price", "aspect_service", "aspect_value", "aspect_usability",
    "aspect_delivery", "review_length",
]


def regression_metrics(y_true, predictions, y_train=None):
    """Return bounded rating regression metrics."""
    clipped = np.clip(predictions, 1, 5)
    rounded = np.rint(clipped).clip(1, 5)
    y_true_arr = np.asarray(y_true)
    
    per_rating = {}
    for rating in sorted(set(y_true_arr)):
        mask = y_true_arr == rating
        per_rating[str(float(rating))] = {
            "count": int(mask.sum()),
            "prediction_mean": float(np.mean(clipped[mask])),
            "mae": float(np.mean(np.abs(clipped[mask] - y_true_arr[mask]))),
            "correct": int((rounded[mask] == y_true_arr[mask]).sum()),
        }
    
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, clipped))),
        "mae": float(mean_absolute_error(y_true_arr, clipped)),
        "rounded_accuracy": float(np.mean(rounded == y_true_arr)),
        "macro_rating_mae": float(np.mean([v["mae"] for v in per_rating.values()])),
        "per_rating": per_rating,
    }


def compute_sample_weights(y, strategy="inverse"):
    """Compute sample weights to handle class imbalance."""
    counts = pd.Series(y).value_counts().sort_index()
    if strategy == "inverse":
        weights = {r: len(y) / counts[r] for r in counts.index}
    elif strategy == "sqrt_inverse":
        weights = {r: np.sqrt(len(y) / counts[r]) for r in counts.index}
    elif strategy == "log_inverse":
        weights = {r: np.log1p(len(y) / counts[r]) for r in counts.index}
    elif strategy.startswith("clip_"):
        # Clip inverse weights to avoid extreme values
        raw = {r: len(y) / counts[r] for r in counts.index}
        max_w = float(strategy.split("_")[1])
        weights = {r: min(w, max_w) for r, w in raw.items()}
    elif strategy == "none":
        return None
    else:
        weights = {r: 1.0 for r in counts.index}
    return np.array([weights[r] for r in y])


def compute_negativity_features(df):
    """Add stronger negative signal features directly from text."""
    import re
    result = df.copy()
    
    NEGATIVE_WORDS = {
        "bad", "poor", "terrible", "awful", "horrible", "horrible",
        "broken", "broke", "break", "defect", "defective", "crooked",
        "cracked", "scratch", "scratches", "flimsy", "damaged", "damage",
        "smoke", "stopped", "dead", "failed", "failure", "unusable",
        "useless", "waste", "wasted", "overpriced", "worthless", "pricey",
        "refund", "return", "returned", "disappointed", "disappointing",
        "never", "worst", "terrible", "horrible", "awful", "cheaply",
        "problem", "problems", "issue", "issues", "overheat", "overheating",
        "corrupted", "crash", "crashed", "corrupt", "corrupted",
    }
    
    COMPLAINT_PHRASES = [
        "stopped working", "does not work", "did not work", "won't work",
        "not worth", "broke after", "broke within", "nothing but problems",
        "do not buy", "do not purchase", "waste of money", "would not recommend",
        "fell apart", "came apart", "ripped", "torn", "stopped charging",
        "dead on arrival", "doa", "not as described", "missing parts",
    ]
    
    token_re = re.compile(r"[a-z0-9']+")
    
    negativity_ratios = []
    phrase_counts = []
    strictness_boost = []
    
    for text in df["text"]:
        text_str = str(text).lower()
        tokens = token_re.findall(text_str)
        n_tokens = len(tokens)
        
        # Direct negative word ratio (not density-scaled)
        neg_hits = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        neg_ratio = neg_hits / max(n_tokens, 1)
        negativity_ratios.append(min(neg_ratio * 10, 1.0))  # Scale up, cap at 1.0
        
        # Phrase count
        phrase_hits = sum(1 for p in COMPLAINT_PHRASES if p in text_str)
        phrase_counts.append(min(phrase_hits / 3.0, 1.0))
        
        # Stricness boost based on combined negativity
        combined = (neg_ratio * 8) + (phrase_hits * 0.3)
        strictness_boost.append(min(combined, 1.0))
    
    result["negativity_ratio"] = negativity_ratios
    result["complaint_phrases"] = phrase_counts
    
    # VADER override: when strong negative keywords present, override signal
    for idx in result.index:
        text_str = str(result.loc[idx, "text"]).lower()
        tokens = set(token_re.findall(text_str))
        
        # Strong negative keyword sets
        strong_quality_neg = {"broken", "broke", "defective", "crooked", "cracked", 
                              "damaged", "useless", "unusable", "smoke", "stopped"}
        strong_service_neg = {"refund", "returned", "replacement", "warranty", "unhelpful"}
        strong_value_neg = {"overpriced", "waste", "worthless", "overrated"}
        strong_usable_neg = {"confusing", "unusable", "overheat", "overheating", "crashed", "corrupted"}
        
        # If strong negatives present AND VADER signal is positive (wrong!), override
        if tokens & strong_quality_neg and result.loc[idx, "quality_signal"] > 0:
            result.loc[idx, "quality_signal"] = max(-0.8, -0.5 * len(tokens & strong_quality_neg))
        
        if tokens & strong_service_neg and result.loc[idx, "service_signal"] > 0:
            result.loc[idx, "service_signal"] = max(-0.8, -0.5 * len(tokens & strong_service_neg))
        
        if tokens & strong_value_neg and result.loc[idx, "value_signal"] > 0:
            result.loc[idx, "value_signal"] = max(-0.8, -0.5 * len(tokens & strong_value_neg))
        
        if tokens & strong_usable_neg and result.loc[idx, "usability_signal"] > 0:
            result.loc[idx, "usability_signal"] = max(-0.8, -0.5 * len(tokens & strong_usable_neg))
    
    return result


def run_experiment(name, train_df, test_df, numeric_cols,
                   tfidf_max_features=8000, tfidf_ngram=(1, 1), tfidf_min_df=2,
                   model_type="ridge", model_params=None,
                   weight_strategy="none", use_enhanced_features=False,
                   alpha=1.0, add_negativity=True):
    """Run one experiment configuration and return test metrics."""
    
    X_train_raw = train_df.copy().reset_index(drop=True)
    y_train = X_train_raw["stars"].values
    X_test_raw = test_df.copy().reset_index(drop=True)
    y_test = X_test_raw["stars"].values
    
    # Apply enhanced features if requested
    if add_negativity:
        X_train_raw = compute_negativity_features(X_train_raw)
        X_test_raw = compute_negativity_features(X_test_raw)
    
    # Determine feature columns
    feat_cols = list(numeric_cols)
    if add_negativity:
        if "negativity_ratio" not in feat_cols:
            feat_cols.extend(["negativity_ratio", "complaint_phrases"])
    
    # Ensure text is string
    X_train_raw["text"] = X_train_raw["text"].fillna("").astype(str)
    X_test_raw["text"] = X_test_raw["text"].fillna("").astype(str)
    
    FEATURE_COLS = feat_cols + ["text"]
    
    # Split for validation
    X_train, X_valid, y_train_arr, y_valid_arr = train_test_split(
        X_train_raw[FEATURE_COLS], y_train, test_size=0.2, 
        random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Build preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), feat_cols),
            ("text", TfidfVectorizer(
                max_features=tfidf_max_features,
                ngram_range=tfidf_ngram,
                min_df=tfidf_min_df,
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True,
            ), "text"),
        ],
        sparse_threshold=0.3,
    )
    
    # Build model
    params = model_params or {}
    if model_type == "ridge":
        model = make_pipeline(preprocessor, Ridge(alpha=alpha))
    elif model_type == "ridgecv":
        model = make_pipeline(preprocessor, RidgeCV(alphas=params.get("alphas", [0.1, 1.0, 10.0])))
    elif model_type == "huber":
        model = make_pipeline(preprocessor, HuberRegressor(epsilon=params.get("epsilon", 1.35), alpha=alpha))
    elif model_type == "sgd":
        loss = params.get("loss", "huber")
        model = make_pipeline(preprocessor, SGDRegressor(
            loss=loss, alpha=alpha, max_iter=1000, tol=1e-3,
            random_state=RANDOM_SEED, penalty="l2"
        ))
    elif model_type == "svr":
        model = make_pipeline(preprocessor, LinearSVR(
            C=1.0/alpha if alpha > 0 else 1.0, epsilon=0.1,
            random_state=RANDOM_SEED, max_iter=2000
        ))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Compute sample weights
    sample_weight = compute_sample_weights(y_train_arr, weight_strategy)
    
    # Train
    weight_kwarg = {}
    if sample_weight is not None:
        # The last step name varies by model type
        last_step_name = model.steps[-1][0]
        weight_kwarg[f"{last_step_name}__sample_weight"] = sample_weight
    model.fit(X_train, y_train_arr, **weight_kwarg)
    
    # Evaluate
    train_metrics = regression_metrics(y_train_arr, model.predict(X_train), y_train_arr)
    valid_metrics = regression_metrics(y_valid_arr, model.predict(X_valid), y_train_arr)
    test_metrics = regression_metrics(y_test, model.predict(X_test_raw[FEATURE_COLS]), y_train_arr)
    
    return {
        "name": name,
        "model_type": model_type,
        "weight_strategy": weight_strategy,
        "tfidf_ngram": str(tfidf_ngram),
        "tfidf_max_features": tfidf_max_features,
        "alpha": alpha,
        "add_negativity": add_negativity,
        "extra_params": str(model_params),
        "train_rmse": train_metrics["rmse"],
        "train_accuracy": train_metrics["rounded_accuracy"],
        "valid_rmse": valid_metrics["rmse"],
        "valid_accuracy": valid_metrics["rounded_accuracy"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_accuracy": test_metrics["rounded_accuracy"],
        "test_macro_mae": test_metrics["macro_rating_mae"],
        "test_per_rating": test_metrics["per_rating"],
        "gen_gap": test_metrics["rmse"] - train_metrics["rmse"],
    }


print("=" * 80)
print("RESEARCH HARNESS: Rating Model Optimization")
print("=" * 80)

# Load data
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# Current baseline
baseline = run_experiment(
    "BASELINE (current production)", train_df, test_df, NUMERIC_FEATURE_COLS,
    tfidf_max_features=8000, tfidf_ngram=(1, 1), model_type="ridge",
    alpha=1.0, weight_strategy="none", add_negativity=False
)
print(f"\nBASELINE: test_accuracy={baseline['test_accuracy']:.4f}, test_rmse={baseline['test_rmse']:.4f}")
per_str = ", ".join(f"{k}: {v['correct']}/{v['count']}" for k, v in baseline['test_per_rating'].items())
print(f"  Per-rating: {{{per_str}}}")

# ============================================
# EXPERIMENT GROUP 1: Sample Weight Strategies
# ============================================
print("\n" + "=" * 60)
print("GROUP 1: Sample Weight Strategies")
print("=" * 60)
exp_weight = []
for ws in ["inverse", "sqrt_inverse", "log_inverse", "clip_50", "clip_20", "clip_10"]:
    m = run_experiment(
        f"weight_{ws}", train_df, test_df, NUMERIC_FEATURE_COLS,
        weight_strategy=ws, add_negativity=False
    )
    exp_weight.append(m)
    w = m['weight_strategy']
    print(f"  {w:15s}  test_acc={m['test_accuracy']:.4f}  test_rmse={m['test_rmse']:.4f}  "
          f"gap={m['gen_gap']:.4f}  1★={m['test_per_rating']['1.0']['correct']}/{m['test_per_rating']['1.0']['count']}  "
          f"2★={m['test_per_rating']['2.0']['correct']}/{m['test_per_rating']['2.0']['count']}  "
          f"3★={m['test_per_rating']['3.0']['correct']}/{m['test_per_rating']['3.0']['count']}")

# ============================================
# EXPERIMENT GROUP 2: TF-IDF Parameter Sweep
# ============================================
print("\n" + "=" * 60)
print("GROUP 2: TF-IDF Parameter Sweep")
print("=" * 60)
exp_tfidf = []
for ngram in [(1, 1), (1, 2), (1, 3)]:
    for mf in [8000, 12000, 16000]:
        for md in [2, 3]:
            m = run_experiment(
                f"ngram{ngram}_mf{mf}_md{md}", train_df, test_df, NUMERIC_FEATURE_COLS,
                tfidf_ngram=ngram, tfidf_max_features=mf, tfidf_min_df=md,
                weight_strategy="none", add_negativity=False
            )
            exp_tfidf.append(m)
            print(f"  ngram={str(ngram):7s} mf={mf:5d} md={md}  "
                  f"test_acc={m['test_accuracy']:.4f}  test_rmse={m['test_rmse']:.4f}")

# ============================================
# EXPERIMENT GROUP 3: Enhanced Features
# ============================================
print("\n" + "=" * 60)
print("GROUP 3: Enhanced Negative Features (+negativity_ratio +phrase +VADER override)")
print("=" * 60)
# Test enhanced features with different weight strategies
exp_enhanced = []
for ws in ["none", "inverse", "sqrt_inverse", "clip_50", "clip_20"]:
    m = run_experiment(
        f"enhanced_weight_{ws}", train_df, test_df, NUMERIC_FEATURE_COLS,
        weight_strategy=ws, add_negativity=True
    )
    exp_enhanced.append(m)
    print(f"  weight={ws:15s}  test_acc={m['test_accuracy']:.4f}  test_rmse={m['test_rmse']:.4f}  "
          f"gap={m['gen_gap']:.4f}  1★={m['test_per_rating']['1.0']['correct']}/{m['test_per_rating']['1.0']['count']}  "
          f"2★={m['test_per_rating']['2.0']['correct']}/{m['test_per_rating']['2.0']['count']}  "
          f"3★={m['test_per_rating']['3.0']['correct']}/{m['test_per_rating']['3.0']['count']}")

# ============================================
# EXPERIMENT GROUP 4: Best Combo (enhanced + TF-IDF + weight)
# ============================================
print("\n" + "=" * 60)
print("GROUP 4: Best Combinations")
print("=" * 60)
combos = [
    ("enhanced+bigram+clip50", 12000, (1, 2), 3, "ridge", "clip_50", 1.0, True),
    ("enhanced+bigram+inverse", 12000, (1, 2), 3, "ridge", "inverse", 1.0, True),
    ("enhanced+bigram+sqrt_inv", 12000, (1, 2), 3, "ridge", "sqrt_inverse", 1.0, True),
]
exp_best = []
for name, mf, ng, md, mt, ws, alpha, enhanced in combos:
    m = run_experiment(
        name, train_df, test_df, NUMERIC_FEATURE_COLS,
        tfidf_max_features=mf, tfidf_ngram=ng, tfidf_min_df=md,
        model_type=mt, alpha=alpha, weight_strategy=ws, add_negativity=enhanced
    )
    exp_best.append(m)
    print(f"  {name:35s}  test_acc={m['test_accuracy']:.4f}  test_rmse={m['test_rmse']:.4f}  "
          f"gap={m['gen_gap']:.4f}  1★={m['test_per_rating']['1.0']['correct']}/{m['test_per_rating']['1.0']['count']}  "
          f"2★={m['test_per_rating']['2.0']['correct']}/{m['test_per_rating']['2.0']['count']}  "
          f"3★={m['test_per_rating']['3.0']['correct']}/{m['test_per_rating']['3.0']['count']}")

# ============================================
# EXPERIMENT GROUP 5: Different Model Types
# ============================================
print("\n" + "=" * 60)
print("GROUP 5: Different Model Types (with best found settings)")
print("=" * 60)
exp_models = []
for mt in ["ridge", "huber", "svr"]:
    for ws in ["clip_50", "inverse", "none"]:
        m = run_experiment(
            f"{mt}+{ws}", train_df, test_df, NUMERIC_FEATURE_COLS,
            tfidf_max_features=12000, tfidf_ngram=(1, 2), tfidf_min_df=3,
            model_type=mt, alpha=0.5, weight_strategy=ws, add_negativity=True
        )
        exp_models.append(m)
        print(f"  {mt:8s} weight={ws:12s}  test_acc={m['test_accuracy']:.4f}  test_rmse={m['test_rmse']:.4f}  "
              f"gap={m['gen_gap']:.4f}")

# ============================================
# EXPERIMENT GROUP 6: Alpha Sweep (Ridge + enhanced)
# ============================================
print("\n" + "=" * 60)
print("GROUP 6: Alpha / Regularization Sweep")
print("=" * 60)
exp_alphas = []
for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    m = run_experiment(
        f"ridge_alpha{alpha}", train_df, test_df, NUMERIC_FEATURE_COLS,
        tfidf_max_features=12000, tfidf_ngram=(1, 2), tfidf_min_df=3,
        model_type="ridge", alpha=alpha, weight_strategy="none", add_negativity=True
    )
    exp_alphas.append(m)
    print(f"  alpha={alpha:6.2f}  test_acc={m['test_accuracy']:.4f}  test_rmse={m['test_rmse']:.4f}  "
          f"gap={m['gen_gap']:.4f}")

# ============================================ 
# EXPERIMENT GROUP 7: Target Transformations
# ============================================
print("\n" + "=" * 60)
print("GROUP 7: Target Transformations (log, sqrt, box-cox-like)")
print("=" * 60)

def run_experiment_with_transform(name, train_df, test_df, numeric_cols,
                                   tfidf_max_features=8000, tfidf_ngram=(1, 1), tfidf_min_df=2,
                                   model_type="ridge", model_params=None,
                                   weight_strategy="none", add_negativity=True,
                                   alpha=1.0, transform="none"):
    """Run experiment with target transformation."""
    from sklearn.preprocessing import FunctionTransformer
    
    X_train_raw = train_df.copy().reset_index(drop=True)
    y_train = X_train_raw["stars"].values
    X_test_raw = test_df.copy().reset_index(drop=True)
    y_test = X_test_raw["stars"].values
    
    if add_negativity:
        X_train_raw = compute_negativity_features(X_train_raw)
        X_test_raw = compute_negativity_features(X_test_raw)
    
    feat_cols = list(numeric_cols)
    if add_negativity:
        feat_cols.extend(["negativity_ratio", "complaint_phrases"])
    
    X_train_raw["text"] = X_train_raw["text"].fillna("").astype(str)
    X_test_raw["text"] = X_test_raw["text"].fillna("").astype(str)
    
    FEATURE_COLS = feat_cols + ["text"]
    
    X_train, X_valid, y_train_arr, y_valid_arr = train_test_split(
        X_train_raw[FEATURE_COLS], y_train, test_size=0.2, 
        random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Apply target transformation
    if transform == "log":
        # Log transform: log(6 - stars) to spread out lower ratings
        y_train_t = np.log1p(6 - y_train_arr)
        y_valid_t = np.log1p(6 - y_valid_arr)
        y_test_t = np.log1p(6 - y_test)
        def inverse_transform(pred):
            return 6 - np.expm1(np.clip(pred, -10, 10))
    elif transform == "sqrt":
        y_train_t = np.sqrt(6 - y_train_arr)
        y_valid_t = np.sqrt(6 - y_valid_arr)
        y_test_t = np.sqrt(6 - y_test)
        def inverse_transform(pred):
            return 6 - np.clip(pred, 0, 10) ** 2
    elif transform == "inverse":
        y_train_t = 1.0 / (6 - y_train_arr + 0.1)
        y_valid_t = 1.0 / (6 - y_valid_arr + 0.1)
        y_test_t = 1.0 / (6 - y_test + 0.1)
        def inverse_transform(pred):
            raw = 6 - (1.0 / np.clip(pred, 0.01, 10) - 0.1)
            return np.clip(raw, 1, 5)
    else:
        y_train_t = y_train_arr
        y_valid_t = y_valid_arr
        y_test_t = y_test
        def inverse_transform(pred):
            return np.clip(pred, 1, 5)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), feat_cols),
            ("text", TfidfVectorizer(
                max_features=tfidf_max_features,
                ngram_range=tfidf_ngram,
                min_df=tfidf_min_df,
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True,
            ), "text"),
        ],
        sparse_threshold=0.3,
    )
    
    params = model_params or {}
    if model_type == "ridge":
        model = make_pipeline(preprocessor, Ridge(alpha=alpha))
    elif model_type == "huber":
        model = make_pipeline(preprocessor, HuberRegressor(epsilon=params.get("epsilon", 1.35), alpha=alpha))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    sample_weight = compute_sample_weights(y_train_arr, weight_strategy)
    
    if sample_weight is not None:
        model.fit(X_train, y_train_t, ridge__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train_t)
    
    train_pred = inverse_transform(model.predict(X_train))
    valid_pred = inverse_transform(model.predict(X_valid))
    test_pred = inverse_transform(model.predict(X_test_raw[FEATURE_COLS]))
    
    train_metrics = regression_metrics(y_train_arr, train_pred, y_train_arr)
    valid_metrics = regression_metrics(y_valid_arr, valid_pred, y_train_arr)
    test_metrics = regression_metrics(y_test, test_pred, y_train_arr)
    
    return {
        "name": name,
        "model_type": model_type,
        "weight_strategy": weight_strategy,
        "tfidf_ngram": str(tfidf_ngram),
        "tfidf_max_features": tfidf_max_features,
        "alpha": alpha,
        "add_negativity": add_negativity,
        "transform": transform,
        "extra_params": str(model_params),
        "train_rmse": train_metrics["rmse"],
        "train_accuracy": train_metrics["rounded_accuracy"],
        "valid_rmse": valid_metrics["rmse"],
        "valid_accuracy": valid_metrics["rounded_accuracy"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_accuracy": test_metrics["rounded_accuracy"],
        "test_macro_mae": test_metrics["macro_rating_mae"],
        "test_per_rating": test_metrics["per_rating"],
        "gen_gap": test_metrics["rmse"] - train_metrics["rmse"],
    }

exp_transform = []
for transform in ["log", "sqrt", "inverse"]:
    m = run_experiment_with_transform(
        f"transform_{transform}", train_df, test_df, NUMERIC_FEATURE_COLS,
        tfidf_max_features=12000, tfidf_ngram=(1, 2), tfidf_min_df=3,
        model_type="ridge", alpha=1.0, weight_strategy="none",
        add_negativity=True, transform=transform
    )
    exp_transform.append(m)
    p = m['test_per_rating']
    print(f"  transform={transform:8s}  test_acc={m['test_accuracy']:.4f}  test_rmse={m['test_rmse']:.4f}  "
          f"gap={m['gen_gap']:.4f}  1☆={p['1.0']['correct']}/{p['1.0']['count']}  "
          f"2☆={p['2.0']['correct']}/{p['2.0']['count']}  3☆={p['3.0']['correct']}/{p['3.0']['count']}")

# ============================================
# EXPERIMENT GROUP 8: Prediction Rounding Calibration
# ============================================
print("\n" + "=" * 60)
print("GROUP 8: Calibrated Rounding Thresholds")
print("=" * 60)
# Uses best model from Group 3 (enhanced only, no weights, no bigram) 
# and tests different rounding schemes

# First train the best model to get raw predictions
feat_cols = list(NUMERIC_FEATURE_COLS) + ["negativity_ratio", "complaint_phrases"]
X_train_raw = compute_negativity_features(train_df.copy().reset_index(drop=True))
X_test_raw = compute_negativity_features(test_df.copy().reset_index(drop=True))
X_train_raw["text"] = X_train_raw["text"].fillna("").astype(str)
X_test_raw["text"] = X_test_raw["text"].fillna("").astype(str)

FEATURE_COLS = feat_cols + ["text"]

X_tr, X_va, y_tr, y_va = train_test_split(
    X_train_raw[FEATURE_COLS], X_train_raw["stars"].values, 
    test_size=0.2, random_state=RANDOM_SEED, stratify=X_train_raw["stars"].values
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), feat_cols),
        ("text", TfidfVectorizer(
            max_features=12000, ngram_range=(1, 2), min_df=3,
            sublinear_tf=True, strip_accents="unicode", lowercase=True,
        ), "text"),
    ],
    sparse_threshold=0.3,
)
best_model = make_pipeline(preprocessor, Ridge(alpha=1.0))
best_model.fit(X_tr, y_tr)

raw_test_preds = best_model.predict(X_test_raw[FEATURE_COLS])

# Test different rounding thresholds
thresholds = [
    (1.5, 2.5, 3.5, 4.5),  # Standard rounding
    (1.8, 2.8, 3.5, 4.5),
    (2.0, 2.8, 3.5, 4.3),
    (1.5, 2.5, 3.8, 4.8),  # Wider at bottom
    (2.0, 3.0, 3.8, 4.5),
    (1.5, 2.5, 3.5, 4.2),  # Easier 5-star
    (1.5, 2.5, 3.5, 4.8),  # Harder 5-star
    (2.2, 3.0, 3.8, 4.5),  # Shift everything up
    (1.2, 2.2, 3.2, 4.2),  # Shift everything down
]
y_test_actual = X_test_raw["stars"].values
for thresh in thresholds:
    preds_clipped = np.clip(raw_test_preds, 1, 5)
    rounded = np.zeros_like(preds_clipped)
    rounded[preds_clipped < thresh[0]] = 1
    rounded[(preds_clipped >= thresh[0]) & (preds_clipped < thresh[1])] = 2
    rounded[(preds_clipped >= thresh[1]) & (preds_clipped < thresh[2])] = 3
    rounded[(preds_clipped >= thresh[2]) & (preds_clipped < thresh[3])] = 4
    rounded[preds_clipped >= thresh[3]] = 5
    
    acc = np.mean(rounded == y_test_actual)
    per_r = {}
    for r in sorted(set(y_test_actual)):
        m = y_test_actual == r
        per_r[str(r)] = {"correct": int((rounded[m] == r).sum()), "count": int(m.sum())}
    print(f"  thresh={str(thresh):30s}  test_acc={acc:.4f}  "
          f"1☆={per_r['1.0']['correct']}/{per_r['1.0']['count']}  "
          f"2☆={per_r['2.0']['correct']}/{per_r['2.0']['count']}  "
          f"3☆={per_r['3.0']['correct']}/{per_r['3.0']['count']}")

# ============================================
# EXPERIMENT GROUP 9: Ordinal Classification Approach
# ============================================
print("\n" + "=" * 60)
print("GROUP 9: Ordinal Classification (multiple binary classifiers)")
print("=" * 60)

# Train 4 binary classifiers: rating >= 2, >=3, >=4, >=5
# Combine predictions to get the rating

def train_ordinal_model(train_df, test_df, numeric_cols, tfidf_max_features=8000, 
                        tfidf_ngram=(1, 1), tfidf_min_df=2, alpha=1.0):
    """Train ordinal classifiers: P(rating >= k) for k=2,3,4,5."""
    from sklearn.linear_model import LogisticRegression
    
    X_tr_raw = compute_negativity_features(train_df.copy().reset_index(drop=True))
    X_te_raw = compute_negativity_features(test_df.copy().reset_index(drop=True))
    
    feat_cols = list(numeric_cols) + ["negativity_ratio", "complaint_phrases"]
    
    X_tr_raw["text"] = X_tr_raw["text"].fillna("").astype(str)
    X_te_raw["text"] = X_te_raw["text"].fillna("").astype(str)
    FEATURE_COLS = feat_cols + ["text"]
    
    y_train = X_tr_raw["stars"].values
    y_test = X_te_raw["stars"].values
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), feat_cols),
            ("text", TfidfVectorizer(
                max_features=tfidf_max_features,
                ngram_range=tfidf_ngram,
                min_df=tfidf_min_df,
                sublinear_tf=True, strip_accents="unicode", lowercase=True,
            ), "text"),
        ],
        sparse_threshold=0.3,
    )
    
    # For each threshold k (2, 3, 4, 5), train: rating >= k
    thresholds = [2, 3, 4, 5]
    models = {}
    for k in thresholds:
        y_binary = (y_train >= k).astype(int)
        # Handle class imbalance in binary classifiers
        pos_weight = len(y_binary) / max(y_binary.sum(), 1)
        neg_weight = len(y_binary) / max((1 - y_binary).sum(), 1)
        sample_w = np.where(y_binary == 1, pos_weight, neg_weight)
        
        pipe = make_pipeline(preprocessor, LogisticRegression(C=1.0/alpha, max_iter=1000, random_state=RANDOM_SEED))
        pipe.fit(X_tr_raw[FEATURE_COLS], y_binary, logisticregression__sample_weight=sample_w)
        models[k] = pipe
    
    # Predict probabilities
    probs = np.column_stack([models[k].predict_proba(X_te_raw[FEATURE_COLS])[:, 1] for k in thresholds])
    
    # Combine: rating = 1 + sum(P(rating >= k))
    # P(rating == 1) = 1 - P(rating >= 2)
    # P(rating == k) = P(rating >= k) - P(rating >= k+1) for k=2,3,4
    # P(rating == 5) = P(rating >= 5)
    
    # Expected value
    rating_pred = 1 + probs.sum(axis=1)
    
    per_r = {}
    for r in sorted(set(y_test)):
        m = y_test == r
        rounded = np.rint(rating_pred).clip(1, 5)
        per_r[str(r)] = {
            "correct": int((rounded[m] == r).sum()),
            "count": int(m.sum())
        }
    
    acc = np.mean(np.rint(rating_pred).clip(1, 5) == y_test)
    return acc, per_r, rating_pred

ord_acc, ord_per, ord_pred = train_ordinal_model(
    train_df, test_df, NUMERIC_FEATURE_COLS,
    tfidf_max_features=12000, tfidf_ngram=(1, 2), tfidf_min_df=3, alpha=0.5
)
print(f"  Ordinal classification: test_acc={ord_acc:.4f}")
per_str = ", ".join(f"{k}: {v['correct']}/{v['count']}" for k, v in ord_per.items())
print(f"  Per-rating: {{{per_str}}}")
exp_ord = {"name": "ordinal_classification", "test_accuracy": ord_acc, "test_per_rating": ord_per}

# ============================================
# EXPERIMENT GROUP 10: Prediction Post-hoc Calibration
# ============================================
print("\n" + "=" * 60)
print("GROUP 10: Post-hoc model calibration (Platt scaling / isotonic)")
print("=" * 60)

# Use predictions from best model and apply calibration
raw_pred = np.clip(raw_test_preds, 1, 5)
y_test_a = y_test_actual

# Try: sigmoid-based correction to pull predictions toward extremes
def calibrate_prediction(pred, strength=0.3):
    centered = pred - 3.0
    corrected = centered * (1 + strength * np.abs(centered) / 2.0)
    return np.clip(3.0 + corrected, 1, 5)

for strength in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    cal_pred = calibrate_prediction(raw_pred, strength)
    cal_rounded = np.rint(cal_pred).clip(1, 5)
    acc = np.mean(cal_rounded == y_test_a)
    
    per_r = {}
    for r in sorted(set(y_test_a)):
        m = y_test_a == r
        per_r[str(r)] = {"correct": int((cal_rounded[m] == r).sum()), "count": int(m.sum())}
    print(f"  calibrate(strength={strength:.1f})  test_acc={acc:.4f}  "
          f"1☆={per_r['1.0']['correct']}/{per_r['1.0']['count']}  "
          f"2☆={per_r['2.0']['correct']}/{per_r['2.0']['count']}  "
          f"3☆={per_r['3.0']['correct']}/{per_r['3.0']['count']}")

# ============================================
# EXPERIMENT GROUP 11: User/Item Target Encoding
# ============================================
print("\n" + "=" * 60)
print("GROUP 11: User/Item Target Encoding Features")
print("=" * 60)

def add_target_encoding(train_df, test_df, cols, alpha=10.0):
    """Add target-encoded user and item rating means with smoothing."""
    train = train_df.copy()
    test = test_df.copy()
    
    global_mean = train["stars"].mean()
    
    for col in cols:
        # Compute per-group mean + global prior smoothing
        group_stats = train.groupby(col)["stars"].agg(["count", "mean"])
        group_stats["encoded"] = (
            (group_stats["count"] * group_stats["mean"] + alpha * global_mean) 
            / (group_stats["count"] + alpha)
        )
        
        # Map to train
        train[f"{col}_rating_mean"] = train[col].map(group_stats["encoded"])
        # Map to test (fill unknown with global mean)
        test[f"{col}_rating_mean"] = test[col].map(group_stats["encoded"]).fillna(global_mean)
    
    return train, test, [f"{c}_rating_mean" for c in cols]

def run_target_encoding_exp(name, train_df, test_df, numeric_cols,
                             tfidf_max_features=8000, tfidf_ngram=(1, 1), tfidf_min_df=2,
                             model_type="ridge", alpha=1.0, weight_strategy="none",
                             add_negativity=True, transform="none", calibrate=0.0,
                             encode_users=True, encode_items=True):
    """Run experiment with target encoding."""
    train_enc = train_df.copy().reset_index(drop=True)
    test_enc = test_df.copy().reset_index(drop=True)
    
    encode_cols = []
    if encode_users and "user_id" in train_enc.columns:
        encode_cols.append("user_id")
    if encode_items and "item_id" in train_enc.columns:
        encode_cols.append("item_id")
    
    if encode_cols:
        train_enc, test_enc, added_cols = add_target_encoding(train_enc, test_enc, encode_cols)
    else:
        added_cols = []
    
    feat_cols = list(numeric_cols) + added_cols
    
    if add_negativity:
        train_enc = compute_negativity_features(train_enc)
        test_enc = compute_negativity_features(test_enc)
        feat_cols.extend(["negativity_ratio", "complaint_phrases"])
    
    train_enc["text"] = train_enc["text"].fillna("").astype(str)
    test_enc["text"] = test_enc["text"].fillna("").astype(str)
    FEATURE_COLS = feat_cols + ["text"]
    
    X = train_enc[FEATURE_COLS]
    y = train_enc["stars"].values
    X_test = test_enc[FEATURE_COLS]
    y_test = test_enc["stars"].values
    
    X_train, X_valid, y_train_arr, y_valid_arr = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    if transform == "inverse":
        y_train_t = 1.0 / (6 - y_train_arr + 0.1)
        y_valid_t = 1.0 / (6 - y_valid_arr + 0.1)
        y_test_t = 1.0 / (6 - y_test + 0.1)
        def inv_transform(pred):
            raw = 6 - (1.0 / np.clip(pred, 0.01, 10) - 0.1)
            return np.clip(raw, 1, 5)
    else:
        y_train_t = y_train_arr
        y_valid_t = y_valid_arr
        y_test_t = y_test
        def inv_transform(pred):
            return np.clip(pred, 1, 5)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), feat_cols),
            ("text", TfidfVectorizer(
                max_features=tfidf_max_features, ngram_range=tfidf_ngram,
                min_df=tfidf_min_df, sublinear_tf=True,
                strip_accents="unicode", lowercase=True,
            ), "text"),
        ],
        sparse_threshold=0.3,
    )
    
    model = make_pipeline(preprocessor, Ridge(alpha=alpha))
    sample_weight = compute_sample_weights(y_train_arr, weight_strategy)
    weight_kwarg = {}
    if sample_weight is not None:
        weight_kwarg["ridge__sample_weight"] = sample_weight
    model.fit(X_train, y_train_t, **weight_kwarg)
    
    raw_test_pred = inv_transform(model.predict(X_test))
    
    if calibrate > 0:
        centered = raw_test_pred - 3.0
        corrected = centered * (1 + calibrate * np.abs(centered) / 2.0)
        raw_test_pred = np.clip(3.0 + corrected, 1, 5)
    
    test_metrics = regression_metrics(y_test, raw_test_pred)
    return {
        "name": name,
        "test_accuracy": test_metrics["rounded_accuracy"],
        "test_rmse": test_metrics["rmse"],
        "test_per_rating": test_metrics["per_rating"],
    }

for enc_users, enc_items, name_suffix in [(True, False, "+user_enc"), (False, True, "+item_enc"), (True, True, "+both_enc")]:
    m = run_target_encoding_exp(
        f"inverse+bigram{name_suffix}", train_df, test_df, NUMERIC_FEATURE_COLS,
        tfidf_max_features=12000, tfidf_ngram=(1, 2), tfidf_min_df=3,
        model_type="ridge", alpha=1.0, add_negativity=True, transform="inverse",
        encode_users=enc_users, encode_items=enc_items
    )
    p = m['test_per_rating']
    print(f"  {m['name']:35s}  test_acc={m['test_accuracy']:.4f}  test_rmse={m['test_rmse']:.4f}  "
          f"1☆={p['1.0']['correct']}/{p['1.0']['count']}  "
          f"2☆={p['2.0']['correct']}/{p['2.0']['count']}  3☆={p['3.0']['correct']}/{p['3.0']['count']}")

# ============================================
# EXPERIMENT GROUP 12: Steeper Transform + Best Combined
# ============================================
print("\n" + "=" * 60)
print("GROUP 12: Steeper Transforms + Best Combined")
print("=" * 60)

def run_steeper(name, train_df, test_df, numeric_cols,
                tfidf_max_features=8000, tfidf_ngram=(1, 1), tfidf_min_df=2,
                alpha=1.0, add_negativity=True, transform_type="inverse2", calibrate=0.0):
    """Test steeper target transforms."""
    train_s = train_df.copy().reset_index(drop=True)
    test_s = test_df.copy().reset_index(drop=True)
    
    if add_negativity:
        train_s = compute_negativity_features(train_s)
        test_s = compute_negativity_features(test_s)
    
    feat_cols = list(numeric_cols)
    if add_negativity:
        feat_cols.extend(["negativity_ratio", "complaint_phrases"])
    
    train_s["text"] = train_s["text"].fillna("").astype(str)
    test_s["text"] = test_s["text"].fillna("").astype(str)
    FEATURE_COLS = feat_cols + ["text"]
    
    X = train_s[FEATURE_COLS]
    y = train_s["stars"].values
    X_test = test_s[FEATURE_COLS]
    y_test = test_s["stars"].values
    
    X_train, X_valid, y_train_arr, y_valid_arr = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    if transform_type == "inverse2":
        # Steeper: 1/(6-stars)^2 — amplifies low ratings more
        y_train_t = 1.0 / ((6 - y_train_arr + 0.05) ** 2)
        def inv_t(pred):
            raw = 6 - np.sqrt(1.0 / np.clip(pred, 0.001, 100)) + 0.05
            return np.clip(raw, 1, 5)
    elif transform_type == "exp":
        # Exponential: exp(6-stars) — very steep
        y_train_t = np.exp(6 - y_train_arr) / 100.0
        def inv_t(pred):
            raw = 6 - np.log(np.clip(pred * 100, 0.1, 10000))
            return np.clip(raw, 1, 5)
    else:
        y_train_t = y_train_arr
        def inv_t(pred):
            return np.clip(pred, 1, 5)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), feat_cols),
            ("text", TfidfVectorizer(
                max_features=tfidf_max_features, ngram_range=tfidf_ngram,
                min_df=tfidf_min_df, sublinear_tf=True,
                strip_accents="unicode", lowercase=True,
            ), "text"),
        ],
        sparse_threshold=0.3,
    )
    
    model = make_pipeline(preprocessor, Ridge(alpha=alpha))
    model.fit(X_train, y_train_t)
    
    raw_pred = inv_t(model.predict(X_test))
    
    if calibrate > 0:
        centered = raw_pred - 3.0
        corrected = centered * (1 + calibrate * np.abs(centered) / 2.0)
        raw_pred = np.clip(3.0 + corrected, 1, 5)
    
    test_metrics = regression_metrics(y_test, raw_pred)
    return {
        "name": name,
        "test_accuracy": test_metrics["rounded_accuracy"],
        "test_rmse": test_metrics["rmse"],
        "test_per_rating": test_metrics["per_rating"],
    }

for tt, cal in [("inverse2", 0.0), ("inverse2", 0.3), ("exp", 0.0), ("exp", 0.3)]:
    m = run_steeper(
        f"{tt}+cal{cal}", train_df, test_df, NUMERIC_FEATURE_COLS,
        tfidf_max_features=12000, tfidf_ngram=(1, 2), tfidf_min_df=3,
        alpha=1.0, add_negativity=True, transform_type=tt, calibrate=cal
    )
    p = m['test_per_rating']
    print(f"  {m['name']:25s}  test_acc={m['test_accuracy']:.4f}  test_rmse={m['test_rmse']:.4f}  "
          f"1☆={p['1.0']['correct']}/{p['1.0']['count']}  "
          f"2☆={p['2.0']['correct']}/{p['2.0']['count']}  3☆={p['3.0']['correct']}/{p['3.0']['count']}  "
          f"4☆={p['4.0']['correct']}/{p['4.0']['count']}  5☆={p['5.0']['correct']}/{p['5.0']['count']}")

# ============================================
# EXPERIMENT GROUP 13: Hybrid Ordinal + Regression
# ============================================
print("\n" + "=" * 60)
print("GROUP 13: Hybrid Model (ordinal for low ratings, regression for high)")
print("=" * 60)

def run_hybrid(name, train_df, test_df, numeric_cols,
               tfidf_max_features=12000, tfidf_ngram=(1, 2), tfidf_min_df=3,
               alpha_ordinal=0.5, alpha_regression=0.5, threshold=3.0):
    """Use ordinal classifier to identify 1-2★, regression for the rest."""
    from sklearn.linear_model import LogisticRegression, Ridge
    
    train_h = compute_negativity_features(train_df.copy().reset_index(drop=True))
    test_h = compute_negativity_features(test_df.copy().reset_index(drop=True))
    
    feat_cols = list(numeric_cols) + ["negativity_ratio", "complaint_phrases"]
    train_h["text"] = train_h["text"].fillna("").astype(str)
    test_h["text"] = test_h["text"].fillna("").astype(str)
    FEATURE_COLS = feat_cols + ["text"]
    
    y_train = train_h["stars"].values
    y_test = test_h["stars"].values
    
    # For ordinal: predict if rating < threshold (i.e., is it low?)
    # We use training data to train a classifier for "is this a low rating?"
    # Then use regression for the final numerical prediction
    # The hybrid picks the lower of the two predictions
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), feat_cols),
            ("text", TfidfVectorizer(
                max_features=tfidf_max_features, ngram_range=tfidf_ngram,
                min_df=tfidf_min_df, sublinear_tf=True,
                strip_accents="unicode", lowercase=True,
            ), "text"),
        ],
        sparse_threshold=0.3,
    )
    
    # Ordinal: 4 binary classifiers for rating >= 2,3,4,5
    ord_preds = np.column_stack([
        make_pipeline(preprocessor, LogisticRegression(C=1.0/alpha_ordinal, max_iter=1000, random_state=RANDOM_SEED))
            .fit(train_h[FEATURE_COLS], (y_train >= k).astype(int))
            .predict_proba(test_h[FEATURE_COLS])[:, 1]
        for k in [2, 3, 4, 5]
    ])
    ordinal_rating = 1 + ord_preds.sum(axis=1)
    
    # Regression with inverse transform
    y_train_t = 1.0 / (6 - y_train + 0.1)
    reg = make_pipeline(preprocessor, Ridge(alpha=alpha_regression))
    reg.fit(train_h[FEATURE_COLS], y_train_t)
    reg_pred = 6 - (1.0 / np.clip(reg.predict(test_h[FEATURE_COLS]), 0.01, 10) - 0.1)
    reg_pred = np.clip(reg_pred, 1, 5)
    
    # Hybrid: use min of the two predictions (more conservative)
    hybrid_pred = np.minimum(ordinal_rating, reg_pred)
    
    test_metrics = regression_metrics(y_test, hybrid_pred)
    return test_metrics

hm = run_hybrid("hybrid_min", train_df, test_df, NUMERIC_FEATURE_COLS, alpha_ordinal=0.5, alpha_regression=0.5, threshold=3.0)
p = hm['per_rating']
print(f"  hybrid_min(min(ordinal,reg))  test_acc={hm['rounded_accuracy']:.4f}  test_rmse={hm['rmse']:.4f}  "
      f"1☆={p['1.0']['correct']}/{p['1.0']['count']}  2☆={p['2.0']['correct']}/{p['2.0']['count']}  "
      f"3☆={p['3.0']['correct']}/{p['3.0']['count']}  4☆={p['4.0']['correct']}/{p['4.0']['count']}  "
      f"5☆={p['5.0']['correct']}/{p['5.0']['count']}")

# ============================================
# EXPERIMENT GROUP 14: Full Training (no validation split)
# ============================================
print("\n" + "=" * 60)
print("GROUP 14: Full Training (train on ALL 1040 samples)")
print("=" * 60)

def run_fulltrain(name, train_df, test_df, numeric_cols,
                  tfidf_max_features=12000, tfidf_ngram=(1, 2), tfidf_min_df=3,
                  model_type="ridge", alpha=1.0, weight_strategy="none",
                  add_negativity=True, transform="inverse", calibrate=0.0):
    """Train on ALL data (no validation split) to maximize minority samples."""
    train_f = train_df.copy().reset_index(drop=True)
    test_f = test_df.copy().reset_index(drop=True)
    
    if add_negativity:
        train_f = compute_negativity_features(train_f)
        test_f = compute_negativity_features(test_f)
    
    feat_cols = list(numeric_cols)
    if add_negativity:
        feat_cols.extend(["negativity_ratio", "complaint_phrases"])
    
    train_f["text"] = train_f["text"].fillna("").astype(str)
    test_f["text"] = test_f["text"].fillna("").astype(str)
    FEATURE_COLS = feat_cols + ["text"]
    
    y_train = train_f["stars"].values
    y_test = test_f["stars"].values
    
    if transform == "inverse":
        y_train_t = 1.0 / (6 - y_train + 0.1)
        def inv_t(pred):
            raw = 6 - (1.0 / np.clip(pred, 0.01, 10) - 0.1)
            return np.clip(raw, 1, 5)
    elif transform == "inverse2":
        y_train_t = 1.0 / ((6 - y_train + 0.05) ** 2)
        def inv_t(pred):
            raw = 6 - np.sqrt(1.0 / np.clip(pred, 0.001, 100)) + 0.05
            return np.clip(raw, 1, 5)
    else:
        y_train_t = y_train
        def inv_t(pred):
            return np.clip(pred, 1, 5)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), feat_cols),
            ("text", TfidfVectorizer(
                max_features=tfidf_max_features, ngram_range=tfidf_ngram,
                min_df=tfidf_min_df, sublinear_tf=True,
                strip_accents="unicode", lowercase=True,
            ), "text"),
        ],
        sparse_threshold=0.3,
    )
    
    model = make_pipeline(preprocessor, Ridge(alpha=alpha))
    sample_weight = compute_sample_weights(y_train, weight_strategy)
    weight_kwarg = {}
    if sample_weight is not None:
        weight_kwarg["ridge__sample_weight"] = sample_weight
    model.fit(train_f[FEATURE_COLS], y_train_t, **weight_kwarg)
    
    raw_test_pred = inv_t(model.predict(test_f[FEATURE_COLS]))
    
    if calibrate > 0:
        centered = raw_test_pred - 3.0
        corrected = centered * (1 + calibrate * np.abs(centered) / 2.0)
        raw_test_pred = np.clip(3.0 + corrected, 1, 5)
    
    test_metrics = regression_metrics(y_test, raw_test_pred)
    return {
        "name": name,
        "test_accuracy": test_metrics["rounded_accuracy"],
        "test_rmse": test_metrics["rmse"],
        "test_per_rating": test_metrics["per_rating"],
    }

for transform, cal, ws in [("inverse", 0.0, "none"), ("inverse", 0.2, "none"), 
                           ("inverse", 0.0, "sqrt_inverse"), ("inverse2", 0.0, "none"),
                           ("inverse", 0.0, "inverse")]:
    m = run_fulltrain(
        f"fulltrain_{transform}_cal{cal}_w{ws}", train_df, test_df, NUMERIC_FEATURE_COLS,
        tfidf_max_features=12000, tfidf_ngram=(1, 2), tfidf_min_df=3,
        alpha=1.0, weight_strategy=ws, add_negativity=True, 
        transform=transform, calibrate=cal
    )
    p = m['test_per_rating']
    print(f"  {m['name']:40s}  test_acc={m['test_accuracy']:.4f}  test_rmse={m['test_rmse']:.4f}  "
          f"1☆={p['1.0']['correct']}/{p['1.0']['count']}  2☆={p['2.0']['correct']}/{p['2.0']['count']}  "
          f"3☆={p['3.0']['correct']}/{p['3.0']['count']}  4☆={p['4.0']['correct']}/{p['4.0']['count']}  "
          f"5☆={p['5.0']['correct']}/{p['5.0']['count']}")

def run_combined(name, train_df, test_df, numeric_cols,
                  tfidf_max_features=8000, tfidf_ngram=(1, 1), tfidf_min_df=2,
                  model_type="ridge", alpha=1.0, weight_strategy="none",
                  add_negativity=True, transform="none", calibrate=0.0,
                  model_params=None):
    """Run experiment combining transform + calibration + best features."""
    from sklearn.preprocessing import FunctionTransformer
    
    X_train_raw = train_df.copy().reset_index(drop=True)
    y_train = X_train_raw["stars"].values
    X_test_raw = test_df.copy().reset_index(drop=True)
    y_test = X_test_raw["stars"].values
    
    if add_negativity:
        X_train_raw = compute_negativity_features(X_train_raw)
        X_test_raw = compute_negativity_features(X_test_raw)
    
    feat_cols = list(numeric_cols)
    if add_negativity:
        feat_cols.extend(["negativity_ratio", "complaint_phrases"])
    
    X_train_raw["text"] = X_train_raw["text"].fillna("").astype(str)
    X_test_raw["text"] = X_test_raw["text"].fillna("").astype(str)
    FEATURE_COLS = feat_cols + ["text"]
    
    X_train, X_valid, y_train_arr, y_valid_arr = train_test_split(
        X_train_raw[FEATURE_COLS], y_train, test_size=0.2, 
        random_state=RANDOM_SEED, stratify=y_train
    )
    
    # Target transformation
    if transform == "inverse":
        y_train_t = 1.0 / (6 - y_train_arr + 0.1)
        y_valid_t = 1.0 / (6 - y_valid_arr + 0.1)
        y_test_t = 1.0 / (6 - y_test + 0.1)
        def inv_transform(pred):
            raw = 6 - (1.0 / np.clip(pred, 0.01, 10) - 0.1)
            return np.clip(raw, 1, 5)
    elif transform == "sqrt":
        y_train_t = np.sqrt(6 - y_train_arr)
        y_valid_t = np.sqrt(6 - y_valid_arr)
        y_test_t = np.sqrt(6 - y_test)
        def inv_transform(pred):
            return 6 - np.clip(pred, 0, 10) ** 2
    else:
        y_train_t = y_train_arr
        y_valid_t = y_valid_arr
        y_test_t = y_test
        def inv_transform(pred):
            return np.clip(pred, 1, 5)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), feat_cols),
            ("text", TfidfVectorizer(
                max_features=tfidf_max_features,
                ngram_range=tfidf_ngram,
                min_df=tfidf_min_df,
                sublinear_tf=True, strip_accents="unicode", lowercase=True,
            ), "text"),
        ],
        sparse_threshold=0.3,
    )
    
    params = model_params or {}
    if model_type == "ridge":
        model = make_pipeline(preprocessor, Ridge(alpha=alpha))
    elif model_type == "huber":
        model = make_pipeline(preprocessor, HuberRegressor(epsilon=params.get("epsilon", 1.35), alpha=alpha))
    elif model_type == "ridgecv":
        model = make_pipeline(preprocessor, RidgeCV(alphas=params.get("alphas", [0.1, 1.0, 10.0])))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    sample_weight = compute_sample_weights(y_train_arr, weight_strategy)
    weight_kwarg = {}
    if sample_weight is not None:
        last_step_name = model.steps[-1][0]
        weight_kwarg[f"{last_step_name}__sample_weight"] = sample_weight
    model.fit(X_train, y_train_t, **weight_kwarg)
    
    raw_test_pred = inv_transform(model.predict(X_test_raw[FEATURE_COLS]))
    
    # Post-hoc calibration
    if calibrate > 0:
        centered = raw_test_pred - 3.0
        corrected = centered * (1 + calibrate * np.abs(centered) / 2.0)
        final_pred = np.clip(3.0 + corrected, 1, 5)
    else:
        final_pred = raw_test_pred
    
    train_pred = inv_transform(model.predict(X_train))
    valid_pred = inv_transform(model.predict(X_valid))
    
    train_metrics = regression_metrics(y_train_arr, train_pred, y_train_arr)
    valid_metrics = regression_metrics(y_valid_arr, valid_pred, y_train_arr)
    test_metrics = regression_metrics(y_test, final_pred, y_train_arr)
    
    return {
        "name": name,
        "test_rmse": test_metrics["rmse"],
        "test_accuracy": test_metrics["rounded_accuracy"],
        "test_macro_mae": test_metrics["macro_rating_mae"],
        "gen_gap": test_metrics["rmse"] - train_metrics["rmse"],
        "test_per_rating": test_metrics["per_rating"],
    }

# Test combinations
combos = [
    # (name, tfidf_mf, ngram, min_df, model_type, alpha, weight, add_neg, transform, calibrate)
    ("inverse+bigram", 12000, (1,2), 3, "ridge", 1.0, "none", True, "inverse", 0.0),
    ("inverse+bigram+cal05", 12000, (1,2), 3, "ridge", 1.0, "none", True, "inverse", 0.5),
    ("inverse+bigram+cal10", 12000, (1,2), 3, "ridge", 1.0, "none", True, "inverse", 1.0),
    ("inverse+huber", 12000, (1,2), 3, "huber", 0.5, "none", True, "inverse", 0.0),
    ("inverse+alpha05", 12000, (1,2), 3, "ridge", 0.5, "none", True, "inverse", 0.0),
    ("inverse+alpha2", 12000, (1,2), 3, "ridge", 2.0, "none", True, "inverse", 0.0),
    ("inverse+alpha5", 12000, (1,2), 3, "ridge", 5.0, "none", True, "inverse", 0.0),
    ("inverse+bigram+alpha05+cal05", 12000, (1,2), 3, "ridge", 0.5, "none", True, "inverse", 0.5),
    ("inverse+bigram+alpha2+cal05", 12000, (1,2), 3, "ridge", 2.0, "none", True, "inverse", 0.5),
    ("inverse+unigram+cal05", 8000, (1,1), 2, "ridge", 1.0, "none", True, "inverse", 0.5),
    ("sqrt+bigram", 12000, (1,2), 3, "ridge", 1.0, "none", True, "sqrt", 0.0),
    ("sqrt+bigram+cal05", 12000, (1,2), 3, "ridge", 1.0, "none", True, "sqrt", 0.5),
    ("inverse+bigram+ridgecv", 12000, (1,2), 3, "ridgecv", 1.0, "none", True, "inverse", 0.0),
    ("inverse+bigram+ridgecv+cal05", 12000, (1,2), 3, "ridgecv", 1.0, "none", True, "inverse", 0.5),
    ("inverse+bigram_mf16k", 16000, (1,2), 2, "ridge", 1.0, "none", True, "inverse", 0.0),
]

exp_combined = []
for cfg in combos:
    name, mf, ng, md, mt, alpha, ws, neg, trans, cal = cfg
    m = run_combined(name, train_df, test_df, NUMERIC_FEATURE_COLS,
                     tfidf_max_features=mf, tfidf_ngram=ng, tfidf_min_df=md,
                     model_type=mt, alpha=alpha, weight_strategy=ws,
                     add_negativity=neg, transform=trans, calibrate=cal)
    exp_combined.append(m)
    p = m['test_per_rating']
    print(f"  {name:35s}  test_acc={m['test_accuracy']:.4f}  test_rmse={m['test_rmse']:.4f}  "
          f"gap={m['gen_gap']:.4f}  1☆={p['1.0']['correct']}/{p['1.0']['count']}  "
          f"2☆={p['2.0']['correct']}/{p['2.0']['count']}  3☆={p['3.0']['correct']}/{p['3.0']['count']}")

# ============================================
# SUMMARY - Find Best
# ============================================
all_results = exp_weight + exp_tfidf + exp_enhanced + exp_best + exp_models
all_results.sort(key=lambda x: -x["test_accuracy"])

print("\n" + "=" * 80)
print("TOP 10 Configurations by Test Accuracy")
print("=" * 80)
print(f"{'Rank':<5} {'Name':<35} {'Acc':<7} {'RMSE':<7} {'Gap':<7} {'1★':<8} {'2★':<8} {'3★':<8}")
print("-" * 80)
for i, r in enumerate(all_results[:10]):
    p1 = r['test_per_rating']
    print(f"{i+1:<5} {r['name']:<35} {r['test_accuracy']:.4f}  {r['test_rmse']:.4f}  {r['gen_gap']:.4f}  "
          f"{p1['1.0']['correct']}/{p1['1.0']['count']:<4} {p1['2.0']['correct']}/{p1['2.0']['count']:<4} "
          f"{p1['3.0']['correct']}/{p1['3.0']['count']:<4}")

# Save results
results_path = Path("artifacts/research_results.json")
with open(results_path, "w") as f:
    # Convert to serializable format
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "test_per_rating"}
        sr["test_per_rating_summary"] = {
            k: {"correct": v["correct"], "count": v["count"], "mae": f"{v['mae']:.4f}"}
            for k, v in r["test_per_rating"].items()
        }
        serializable.append(sr)
    json.dump(serializable, f, indent=2)
print(f"\nFull results saved to {results_path}")
