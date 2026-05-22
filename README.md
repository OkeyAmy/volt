# ChoiceLens

ChoiceLens is split into two environments:

- **Google Colab is the training factory.** Run the six scripts in `scripts/` in order to download data, extract VADER-based review features, validate the feature table, train models, and export artifacts.
- **Your local machine is the serving showroom.** After Colab finishes, download `data/` and `artifacts/` into this project and run the FastAPI app locally. Local serving does not train models.

## Colab Workflow

Open a new Colab notebook and run this setup cell once:

```python
!pip install -q datasets pandas pyarrow scikit-learn catboost lightgbm joblib nltk python-dotenv tqdm

import os
import nltk
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

nltk.download("vader_lexicon", quiet=True)
print("Environment ready!")
```

Then copy each script into its own Colab cell and run them in this order:

1. `scripts/01_data_load_and_clean.py` creates `data/raw/amazon_reviews_clean.parquet`, `data/processed/train.parquet`, and `data/processed/test.parquet`.
2. `scripts/02_feature_extraction.py` uses VADER, keyword counts, and text statistics to create `data/processed/rating_features.parquet`.
3. `scripts/03_feature_engineering.py` validates the existing feature table for compatibility with the original six-step workflow.
4. `scripts/04_train_rating_model.py` trains CatBoost and creates `artifacts/rating_model.cbm`, `artifacts/rating_feature_cols.pkl`, and `artifacts/metrics_rating.json`.
5. `scripts/05_build_catalog.py` creates `data/processed/product_catalog.parquet`.
6. `scripts/06_train_ranker.py` trains LightGBM and creates `artifacts/ranker_model.pkl`, `artifacts/ranker_feature_cols.pkl`, and `artifacts/metrics_ranker.json`.

After script 6 completes, download the Colab `data/` and `artifacts/` folders and place them at the project root.

## Local Setup With uv

Install the full reproducible environment with `uv`:

```bash
uv sync
```

Create `.env` from `.env.example` and set your Gemini key:

```bash
cp .env.example .env
```

Run the API after the serving app is present:

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the Swagger UI.

## Local Training Workflow

If you want to train outside Colab, run the scripts with `uv` from the project root:

```bash
uv sync
uv run python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
uv run python scripts/01_data_load_and_clean.py
uv run python scripts/02_feature_extraction.py
uv run python scripts/03_feature_engineering.py
uv run python scripts/04_train_rating_model.py
uv run python scripts/05_build_catalog.py
uv run python scripts/06_train_ranker.py
```

The training scripts write cleaned data to `data/`, trained models to `artifacts/`, and metrics JSON files for the solution paper. The rating and ranking metrics include train/test gaps so you can discuss overfitting controls in the submission.

Before submission, verify the app and scripts:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall app scripts tests
```

## Endpoint Usage

Task A predicts how a persona would rate a product, runs counterfactual checks, and generates a review:

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
      "quality_signal": 0.7,
      "service_signal": -0.2,
      "value_signal": 0.1,
      "usability_signal": 0.9,
      "price_level": 1,
      "aspect_quality": 1,
      "aspect_price": 0,
      "aspect_service": -1,
      "aspect_value": 0,
      "aspect_usability": 1,
      "aspect_delivery": -1
    }
  }'
```

Task B returns ranked product recommendations for a persona:

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
    "top_k": 10
  }'
```

## Notes

- Training scripts are Colab-ready and do not import from the local `app/` package.
- `nltk` is required only in the Colab training setup for VADER feature extraction.
- `requirements.txt` is for local serving dependencies only.
- The FastAPI `app/` package is implemented locally and expects Colab-downloaded `data/` and `artifacts/` folders before live serving.
# volt
