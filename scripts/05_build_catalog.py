# ============================================================
# SCRIPT 05: Build Product Catalog
# Environment: Google Colab
# Input: data/processed/rating_features_train.parquet
# Output: data/processed/product_catalog.parquet
# ============================================================

from pathlib import Path

import pandas as pd


FEATURE_PATH = Path("data/processed/rating_features_train.parquet")
OUT_PATH = Path("data/processed/product_catalog.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading feature table...")
df = pd.read_parquet(FEATURE_PATH)

print("Building catalog from unique training items to avoid holdout leakage...")
catalog = df.groupby("item_id").agg({
    "quality_signal": "mean",
    "service_signal": "mean",
    "value_signal": "mean",
    "usability_signal": "mean",
    "price_level": "first",
    "aspect_quality": "mean",
    "aspect_price": "mean",
    "aspect_service": "mean",
    "aspect_value": "mean",
    "aspect_usability": "mean",
    "aspect_delivery": "mean",
    "category": "first",
    "text": "first",
    "stars": "mean",
}).reset_index()

catalog["product_id"] = [f"prod_{idx}" for idx in range(len(catalog))]
catalog["product_name"] = catalog["text"].str[:80] + "..."

catalog.to_parquet(OUT_PATH, index=False)

print(f"Catalog: {len(catalog)} unique products")
print(f"Categories: {catalog['category'].nunique()}")
print(f"Saved to: {OUT_PATH}")
print("SCRIPT 05 COMPLETE")
