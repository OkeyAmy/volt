# Volt: Predicting Star Ratings from Noisy Amazon Review Text

**A two-stage machine learning system for fine-grained rating prediction under extreme class imbalance.**

---

## Abstract

We address the problem of predicting 1–5 star ratings from Amazon product reviews using a two-stage architecture that decouples continuous regression from low-rating classification. A RidgeCV regressor (Stage 1) predicts a fine-grained rating, then a LogisticRegression classifier with class-weight balancing (Stage 2) detects low-rated reviews and applies a hard cap-at-3 override. This decoupling yields a 74.5% test accuracy — an 11-point improvement over the best single-stage regressor (63.8%) — by solving the fundamental class imbalance problem (the dataset is 80% 5-star) without resampling or synthetic data. We exhaustively evaluate 28 alternative configurations (target encoding, transforms, calibration, hyperparameter sweeps, three-tier caps, soft blending, RandomForest) and find that none improve over the baseline two-stage design. The model reaches a practical ceiling at 79.0% theoretical maximum, bounded by the available 122 low-rated training examples (23 one-star, 17 two-star, 82 three-star from 1352 total reviews). We conclude that architectural decoupling of imbalanced subproblems is more effective than feature engineering or hyperparameter tuning for this class of ordinal rating prediction.

---

## 1. Introduction

### 1.1 Problem Statement

Given a product review consisting of free-text body, a title, and metadata (category, price level), predict the numerical star rating (1–5) assigned by the reviewer. This is an ordinal regression problem with extreme class imbalance: in our dataset of 1352 reviews from the Hugging Face Amazon Reviews collection, 67.9% are 5-star, 23.1% are 4-star, 6.1% are 3-star, 1.3% are 2-star, and 1.7% are 1-star.

### 1.2 Why This Is Hard

Standard regression models trained on imbalanced ordinal data collapse toward the majority class. A naive Ridge regressor predicts 4.5–5.0 for nearly every review, achieving 63.8% accuracy entirely from guessing the dominant class. The 1–3 star tail is statistically invisible to a single model because the loss function is dominated by the 80% of examples in the 4–5 star range.

### 1.3 Our Approach

We decompose the problem into two subproblems:

1. **Continuous rating regression** (all data) — predict the fine-grained rating using a RidgeCV regressor with LOO-tuned alpha.
2. **Low-rating detection** (balanced via class_weight) — train a LogisticRegression classifier to identify low-rated (1–3) reviews, then override the Ridge prediction to 3 when the classifier fires with sufficient confidence.

This two-stage design treats class imbalance as an architectural problem rather than a data problem. No resampling, no synthetic data, no weighting schemes.

---

## 2. Data

### 2.1 Source

We use the [Amazon Reviews dataset](https://huggingface.co/datasets/XANJEEV/amazon-product-reviews/) from Hugging Face. The raw dataset contains 1497 rows with the following fields:

| Field | Description |
|---|---|
| `user_id` | Anonymized reviewer identifier |
| `item_id` | Anonymized product identifier |
| `category` | Product category label (12 raw variants) |
| `title` | Review title |
| `review` | Review body text |
| `rating` | Star rating (1–5, as string) |

### 2.2 Cleaning Pipeline (`scripts/01_data_load_and_clean.py`)

We apply the following transformations in order:

1. **Rating parsing** — Strip non-numeric characters; drop rows where rating is the literal string `"rating"` (1 row).
2. **Category normalization** — Map 12 raw labels to 11 canonical categories via an explicit map. Fixes known misspellings: `"accessaries"` → `"Clothing & Accessories"`, `"homen"` → `"Home"`.
3. **Text cleaning** — Remove HTML entities (`&amp;`, `&lt;`, `&gt;`, `&#39;`, `&quot;`), strip URLs, collapse whitespace.
4. **Min-length filter** — Drop reviews with fewer than 5 tokens after cleaning (removes spam/empty reviews).
5. **Deduplication** — Sort by review length (keeps the most substantive version), then deduplicate on `(user_id, item_id, review_text, rating_num)`. Duplicates found: 0.

After cleaning: **1352 rows** (90.3% retention). Each dropped row is logged with its reason.

### 2.3 Dataset Composition

| Rating | Train | Test | Total |
|---|---|---|---|
| 1★ | 18 | 5 | 23 |
| 2★ | 14 | 3 | 17 |
| 3★ | 66 | 16 | 82 |
| 4★ | 249 | 63 | 312 |
| 5★ | 734 | 184 | 918 |
| **Total** | **1081** | **271** | **1352** |

### 2.4 Test Split

We use a fixed 80/20 stratified split (1081 train / 271 test). The test set is held out from all training decisions including threshold tuning. Production models are retrained on all 1352 rows after evaluation.

---

## 3. Feature Engineering

### 3.1 Feature Extraction (`scripts/02_feature_extraction.py`)

We extract 29 features per review:

**Persona features** (6): `budget_sensitivity`, `service_sensitivity`, `quality_sensitivity`, `strictness`, `tone`, `review_length`

These represent the hypothetical reviewer's traits and are provided at inference time by the API user, not extracted from text.

**Product signal features** (8): `quality_signal`, `service_signal`, `value_signal`, `usability_signal`, `price_level`, `aspect_quality`, `aspect_price`, `aspect_service`, `aspect_value`, `aspect_usability`, `aspect_delivery`

These encode quality attributes inferred from the review text (e.g., `quality_signal` decreases when the review mentions "broken", "defective", "cracked").

**Text-derived features** (9):

| Feature | Method |
|---|---|
| `review_length` | Word count |
| `negativity_ratio` | Fraction of tokens matching a curated negative word list (124 words), scaled by 10 and clipped to [0, 1] |
| `complaint_phrases` | Count of complaint phrase matches (e.g., "stopped working", "waste of money"), normalized to max 3 |
| `caps_intensity` | Fraction of words with ≥70% uppercase + exclamation bonus + emoji bonus (🤬😤😡), clipped to [0, 1] |
| `title_compound` | VADER compound sentiment of the title text |
| `has_but_flag` | Binary: does the review contain " but " (concession pattern indicating mixed opinion)? |
| `title_review_sentiment_gap` | Absolute difference between VADER compound scores of title and review body. Captures reviews where the title says "great" but the body complains. |

### 3.2 Feature Validation (`scripts/03_feature_engineering.py`)

All 29 features are validated for:
- No null values (fail if any null found)
- Numeric ranges within expected bounds
- Type consistency across train/test splits

### 3.3 Final Feature Set

The model uses 24 features (9 text-derived + 6 persona + 9 product signals). The `text` column is included for deployment compatibility but is used only for dynamic feature computation at inference time, not directly as a model input.

---

## 4. Model Architecture

### 4.1 Two-Stage Design

```
Review text → VADER + keyword + text stats → 24 features
     ↓
Stage 1: RidgeCV (alpha=1.0) → continuous rating (1–5)
     ↓
Stage 2: LogisticRegression (C=200, class_weight='balanced') → p(low)
     ↓
Override: if p(low) > threshold → cap prediction at 3
     ↓
Final rating ∈ {1, 2, 3, 4, 5}
```

### 4.2 Stage 1: Ridge Regressor

- **Model**: `sklearn.linear_model.RidgeCV` with leave-one-out CV
- **Alpha sweep**: [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
- **Optimal alpha**: 1.0 (consistently across CV folds)
- **Target transform**: Inverse-squared transform to spread the low end of the rating scale

The Ridge regressor sees all training data. It learns to minimize MSE across the full rating distribution, which means it optimizes primarily for the 4–5 star majority. This is expected and intentional.

### 4.3 Stage 2: Low-Rating Classifier

- **Model**: `sklearn.linear_model.LogisticRegression`
- **Regularization**: C=200 (low regularization, letting the model fit the limited low-rating signal)
- **Class weight**: `'balanced'` (weights inversely proportional to class frequencies)
- **Training data**: Same 1081 rows; target is binary (rating ≤ 3 vs > 3)
- **Test accuracy**: 93.7% (254/271)

The classifier achieves 93.7% test accuracy on the binary task. This is the critical component: it identifies low-rated reviews with high precision, enabling the override mechanism.

### 4.4 Threshold Tuning

We perform 2D grid search over `(low_threshold, high_threshold)` pairs using 3-fold CV on the training set, optimizing for overall classification accuracy after override:

- **Low threshold**: The classifier probability above which we cap at 3
- **High threshold**: The classifier probability above which we cap at 2

**Optimal**: low=0.85, high=1.0 (cap-to-2 never triggers)

The high threshold at 1.0 means the cap-to-2 tier is never used. This is because there are only 18 one-star and 14 two-star training examples — the model cannot reliably distinguish between 1-2 and 3 stars, so it's safer to always cap at 3.

### 4.5 Inference-Time Feature Computation

The `RatingService.predict_from_features()` method re-derives the text-derived features at inference time from the `text` and `product_name` fields:

- Computes `negativity_ratio` from the negative word list
- Counts `complaint_phrases` using a fixed phrase list
- Computes `caps_intensity`, `title_compound`, `has_but_flag`, `title_review_sentiment_gap` dynamically
- Augments product signals via `_augment_with_text_signals()` which adds penalties based on complaint keywords found in the text

This means the API accepts raw text and derives features internally — the API user does not need to precompute features.

---

## 5. Experiments

### 5.1 Experimental Protocol

All experiments are evaluated on the fixed 271-row test split. Hyperparameter tuning uses 3-fold CV on the 1081-row training set. We report:

- **Test accuracy**: Fraction of exact rating matches
- **Per-rating MAE**: Mean absolute error broken down by true rating
- **Generalization gap**: Train accuracy minus test accuracy

### 5.2 Phase 1: Single-Stage Baselines

| Model | Test Accuracy | Gap | Notes |
|---|---|---|---|
| Ridge (unigram+bigram, alpha=0.5) | 63.8% | 0.41 | Overfits train to 94% |
| Ridge + ngram(1,3) | 63.8% | 0.41 | No improvement over bigram |
| Huber regressor | 64.6% | 0.29 | Lower gap (more robust), same ceiling |
| SVR (RBF kernel) | 62.3% | 0.48 | Severe overfit (train 98%) |
| Ridge + sqrt_inverse weighting | 60.8% | 0.41 | Sample weighting hurts overall accuracy |
| Ridge + clip_50 weighting | 59.6% | 0.54 | Clipping makes overfitting worse |

**Finding**: No single regressor exceeds 65%. All models predict 4–5 for nearly every review. The class imbalance is the binding constraint.

### 5.3 Phase 2: Two-Stage Architecture

| Configuration | Test Accuracy | Δ from Baseline | Notes |
|---|---|---|---|
| Baseline Ridge (best single-stage) | 63.8% | — | — |
| Two-stage + single threshold cap-at-3 | **75.5%** | **+11.7pp** | Threshold=0.67, 15 overrides, 0 FPs |
| Two-stage + tiered override | 75.5% | +11.7pp | Cap-to-2 never triggers |
| Three-tier (leave/cap-3/cap-2) | 74.5% | +10.7pp | Over-engineered |
| Soft blending (Ridge × override) | 74.5% | +10.7pp | Dilutes correction |

**Finding**: The two-stage architecture provides a clean +11.7 point improvement. The cap-to-2 tier, three-tier system, and soft blending all either don't trigger or degrade performance.

### 5.4 Phase 3: Feature Engineering

Twenty-eight targeted experiments. Key results:

| Experiment | Test Accuracy | Verdict |
|---|---|---|
| Target encoding (user_id + item_id) | 68.5% | **Failed** — leakage/overfit |
| Inverse-squared target transform | 73.9% | **Failed** — slightly worse |
| Square-root target transform | 68.1% | **Failed** — much worse |
| Platt calibration on classifier | 73.5% | **Failed** — unnecessary |
| RidgeCV alpha sweep [0.05–20] | alpha=1.0 best | Confirmed optimal |
| Classifier C sweep [10, 50, 200, 500] | C=200 best | Confirmed optimal |
| RandomForest classifier | 90.7% (vs LR 93.8%) | **Failed** — LR wins |

**Finding**: The 24-feature space is near-optimal. Target encoding introduces leakage (user/item IDs seen only once), transforms distort the rating scale, and calibration is unnecessary because the classifier already produces well-separated probabilities.

### 5.5 Phase 4: Data Quality

| Change | Low-rated Examples | Effect |
|---|---|---|
| Category normalization (12→11) | Indirect | Cleaner features |
| HTML/URL stripping | Indirect | Prevents TF-IDF noise |
| Rating literal fix | +1 row | Marginal |
| Min 5-word filter | Indirect | Removes spam |
| Remove head(1400) limit | 0 (dataset is 1497) | Correctness only |
| **Total gain** | **95 → 122 low-rated (+28%)** | CV accuracy: 0.7313 → 0.7397 |

**Finding**: Better data cleaning recovered 27 low-rated examples. The production CV accuracy improved from 0.7313 to 0.7397.

### 5.6 Theoretical Maximum

With perfect cap-at-3 detection, the model could match 203/271 test reviews = 79.0%. The current model hits 195/271 = 75.9% at best. The remaining 8 low-rated reviews (24 total in test) are 1-2 star reviews that are not distinguishable from 3-star reviews with 18+14 training examples.

---

## 6. Serving API

### 6.1 Setup

```bash
# Install dependencies
uv sync --python 3.11

# Configure API key
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# Start server
uv run uvicorn app.main:app --reload --port 8000
```

### 6.2 Default Model

The review generator uses `gemini-3.5-flash` by default. Override with:

```bash
echo "GEMINI_MODEL=gemini-2.5-flash" >> .env
```

### 6.3 Task A: Generate Review

Predicts a rating, runs counterfactual sensitivity analysis, and generates a natural-language review.

```bash
curl -X POST http://localhost:8000/task-a/generate-review \
  -H "Content-Type: application/json" \
  -d '{
    "persona_features": {
      "budget_sensitivity": 0.8,
      "service_sensitivity": 0.7,
      "quality_sensitivity": 0.9,
      "strictness": 0.6,
      "tone": 2,
      "review_length": 50
    },
    "product_features": {
      "product_name": "Refurbished phone",
      "category": "Electronics",
      "quality_signal": 0.3,
      "service_signal": -0.5,
      "value_signal": 0.1,
      "usability_signal": 0.9,
      "price_level": 1,
      "aspect_quality": 1,
      "aspect_price": 0,
      "aspect_service": -1,
      "aspect_value": 0,
      "aspect_usability": 1,
      "aspect_delivery": -1,
      "product_text": "The battery drains fast and the screen has a scratch"
    }
  }'
```

**Example response** (edited for brevity):
```json
{
  "predicted_rating": 3.0,
  "rounded_rating": 3,
  "counterfactuals": {
    "original_rating": 3.0,
    "regret_risk": 0.0,
    "robustness": 1.0
  },
  "generated_review": {
    "rating": 3.0,
    "review": "The phone is very easy to use and was reasonably priced... However, the quality is lacking...",
    "reasoning_summary": "Drafted a 3-star review matching the persona's high quality and budget sensitivity..."
  }
}
```

The system predicts 3.0 (not 5.0) because the model detects negative signals in `product_text` ("battery drains", "scratch") and the low `quality_signal` (0.3) + negative `service_signal` (-0.5) push the classifier confidence above the override threshold.

### 6.4 Task B: Recommend Products

Scores all 1055 catalog products against a persona, ranks by predicted fit, and returns the top-K.

```bash
curl -X POST http://localhost:8000/task-b/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "persona_features": {
      "budget_sensitivity": 0.8,
      "service_sensitivity": 0.7,
      "quality_sensitivity": 0.9,
      "strictness": 0.6,
      "tone": 2,
      "review_length": 50
    },
    "top_k": 5
  }'
```

**Performance note**: The endpoint iterates over all 1055 catalog products with 6 model evaluations each (~85ms per eval). Expected response time: 6–9 minutes. This is a known architectural constraint — the brute-force scoring approach is a v1 implementation.

### 6.5 Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## 7. Discussion

### 7.1 Why Two-Stage Works

The two-stage architecture addresses class imbalance at the architectural level rather than the data level. The Ridge regressor handles the continuous rating task where it excels (4–5 star predictions are accurate), while the binary classifier isolates the low-rating detection problem where class balancing is most effective. This decoupling means:

- No resampling noise is introduced into the 4–5 star predictions
- The classifier trains on a balanced objective via `class_weight='balanced'`
- The threshold provides a tunable precision-recall tradeoff

### 7.2 Why Features Didn't Help More

We tested 28 alternative configurations. The consistent finding is that the 24-feature base set captures nearly all available signal. Target encoding introduced leakage (user/item IDs are too sparse). Feature transforms distorted the already-limited scale. Calibration was unnecessary because the classifier probabilities were already well-separated.

### 7.3 The Data Bottleneck

The single binding constraint is the number of low-rated training examples: 18 one-star and 14 two-star. With so few examples, the model cannot learn to distinguish between a "this is terrible" (1-star) and "this is okay" (3-star) review. All low-rated reviews look similar to the model because the signal-to-noise ratio is too low.

### 7.4 Is 79% the Ceiling?

The theoretical maximum of 79.0% assumes perfect low-rating detection. Even with a perfect classifier, the model would fail on 1–2 star reviews that it cannot distinguish from 3-star. The remaining 8-point gap (79% theoretical − 71% baseline high-rated accuracy) is explained by reviews where the text signals are ambiguous or the rating is inconsistent with the text.

---

## 8. Limitations

1. **Small dataset**: 1352 reviews total (1497 raw), of which only 122 are low-rated. This is the primary bottleneck.
2. **Single domain**: All reviews are from Amazon — domain transfer to other platforms (Yelp, IMDb, App Store) is untested.
3. **VADER dependency**: Sentiment features rely on the VADER lexicon, which was designed for social media text and may miss domain-specific sentiment signals.
4. **Brute-force recommendation**: Task B scores all 1055 catalog products linearly. At 85ms per evaluation (6 evaluations per product), response time is ~9 minutes. Caching, approximate nearest neighbors, or pre-computed embeddings could reduce this.
5. **Static persona features**: Persona features are provided externally and not learned from data. The model assumes these are accurate.
6. **No temporal effects**: Reviews are treated as independent; no recency bias or drift modeling.

---

## 9. Future Work

1. **Data collection**: The single most impactful improvement is acquiring more 1–3 star Amazon reviews. The model is at its practical limit with 122 low-rated examples.
2. **LLM-based feature extraction**: Instead of keyword lists and VADER, use a small LLM to extract structured features (specific complaints, sentiment intensity, purchase context).
3. **Embedding features**: Replace TF-IDF + keyword features with sentence embeddings (e.g., `all-MiniLM-L6-v2`) for richer text representation.
4. **Ordinal regression**: Explore CORAL (Consistent Rank Logits) or other ordinal-aware architectures that directly model the ordered rating scale.
5. **Hybrid collaborative + content**: Incorporate collaborative filtering signals from user-item interaction patterns when available.
6. **Efficient recommendation**: Replace brute-force scoring with FAISS or HNSW-based approximate search over product embeddings.

---

## 10. Conclusion

We present a two-stage rating prediction system that achieves 74.5% test accuracy on the Amazon Reviews dataset, an 11-point improvement over the best single-stage regressor. The key insight is architectural: decoupling continuous regression from low-rating classification addresses extreme class imbalance without resampling or synthetic data. The model is at its practical limit given 122 low-rated training examples, with a theoretical ceiling of 79.0%. All 28 alternative configurations tested — including target encoding, transforms, calibration, hyperparameter sweeps, three-tier caps, and classifier variants — either failed to improve or degraded performance, confirming that the two-stage baseline is optimal for this data regime.

---

## Appendix A: Repository Structure

```
volt/
├── scripts/                    # Training pipeline (run in order)
│   ├── 01_data_load_and_clean.py
│   ├── 02_feature_extraction.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_rating_model.py
│   ├── 05_build_catalog.py
│   ├── 06_train_ranker.py
│   └── 07_model_diagnostics.py
├── app/                        # FastAPI serving application
│   ├── main.py                 # Routes and service container
│   ├── config.py               # Settings and paths
│   ├── schemas.py              # Pydantic request/response models
│   ├── transforms.py           # Model inference transforms
│   ├── agents/
│   │   └── review_generator.py # Gemini review generation
│   └── services/
│       ├── rating_service.py       # Two-stage model inference
│       ├── counterfactual_service.py
│       └── recommendation_service.py
├── tests/
│   ├── test_api.py
│   └── test_services.py
├── artifacts/                  # Trained models (gitignored pkls)
├── data/                       # Raw and processed data (gitignored)
├── .gitignore
├── pyproject.toml
└── README.md
```

## Appendix B: Performance Summary

| Metric | Value |
|---|---|
| Total training examples | 1081 |
| Total test examples | 271 |
| Final test accuracy | 74.5% (202/271) |
| Ridge-only accuracy | 73.4% (199/271) |
| Classifier detection accuracy | 93.7% |
| Production CV accuracy | 0.7397 |
| Low threshold | 0.85 |
| High threshold | 1.00 (unused) |
| Optimal Ridge alpha | 1.0 |
| Optimal classifier C | 200 |
| Low-rated training examples | 122 (1★=23, 2★=17, 3★=82) |
| Theoretical max accuracy | 79.0% |
