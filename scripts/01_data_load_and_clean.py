"""
SCRIPT 01: Data Load, Clean & Normalize
Environment: Google Colab
Uses all rows from the Hugging Face dataset (no row limit).
Cleaning: rating validation, category normalization, text cleaning, dedup.
Output: data/raw/amazon_reviews_clean.parquet
        data/processed/train.parquet
        data/processed/test.parquet
"""
from pathlib import Path
import hashlib
import json
import re

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
TEST_SIZE = 0.2
MIN_REVIEW_WORDS = 5
MAX_REVIEW_CHARS = 10000
INVALID_TEXT_VALUES = {"", "nan", "none", "null", "n/a", "na"}
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Category normalisation map (lowercase key → display name)
# ============================================================
CATEGORY_MAP = {
    "books and stationary": "Books & Stationery",
    "cloth and accessaries": "Clothing & Accessories",
    "electronics": "Electronics",
    "grocery & gourmet food": "Grocery & Gourmet Food",
    "health, wellness & medical supplies": "Health, Wellness & Medical Supplies",
    "pet supplies": "Pet Supplies",
    "sports, outdoors & travel": "Sports, Outdoors & Travel",
    "toys, games & hobbies": "Toys, Games & Hobbies",
    "beauty and personal care": "Beauty & Personal Care",
    "home": "Home",
    "homen": "Home",
    "kitchen": "Kitchen",
}


# ============================================================
# Rating parsing
# ============================================================
def parse_rating(value):
    """Convert rating strings like '5', '5.0', 'five' or 'rating' to 1-5 float."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()

    # Explicit word map
    word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    if text in word_map:
        return word_map[text]

    # "null", "rating", or any non-numeric text → None
    if text in ("null", "rating", "n/a"):
        return None

    for token in text.replace("/", " ").split():
        try:
            number = float(token)
        except ValueError:
            continue
        if 1 <= number <= 5:
            return number
    return None


def parse_date(value):
    """Safely parse date strings; return NaT for anything unparseable."""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text or text.lower() in ("", "nan", "none", "null"):
        return pd.NaT
    try:
        return pd.to_datetime(text, format="mixed", errors="coerce")
    except Exception:
        return pd.NaT


# ============================================================
# Item ID generation
# ============================================================
def make_item_id(row):
    """Create a stable product hash from title and category."""
    title = str(row.get("title", "")).strip().lower()
    category = str(row.get("category", "")).strip().lower()
    item_key = f"{title}||{category}"
    return hashlib.md5(item_key.encode("utf-8")).hexdigest()


# ============================================================
# Text cleaning
# ============================================================
def clean_text(value):
    """Strip HTML, URLs, and normalise whitespace on a text value."""
    if pd.isna(value):
        return ""
    text = str(value)
    # Strip HTML tags and entities
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    # Strip URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def non_placeholder_mask(series):
    """Keep rows whose cleaned text is not an empty placeholder."""
    return ~series.str.lower().isin(INVALID_TEXT_VALUES)


# ============================================================
# Validation guards
# ============================================================
def require_enough_rows(clean_df):
    """Fail clearly when the dataset is too small for a stratified split."""
    if len(clean_df) < 200:
        raise ValueError(
            f"Need at least 200 clean rows for a reliable split, "
            f"but only found {len(clean_df)}."
        )


def require_stratifiable_ratings(clean_df):
    """Fail when rating classes are too sparse for stratified splitting."""
    rating_counts = clean_df["rating_num"].value_counts().sort_index()
    sparse_classes = rating_counts[rating_counts < 2]
    if not sparse_classes.empty:
        raise ValueError(
            "Stratified split requires at least 2 rows per rating class. "
            f"Sparse counts: {sparse_classes.to_dict()}"
        )
    num_classes = len(rating_counts)
    train_rows = int(len(clean_df) * (1 - TEST_SIZE))
    test_rows = len(clean_df) - train_rows
    if train_rows < num_classes or test_rows < num_classes:
        raise ValueError(
            f"Stratified split needs train and test sizes (got {train_rows}, "
            f"{test_rows}) at least as large as rating classes ({num_classes})."
        )


# ============================================================
# Main cleaning pipeline
# ============================================================
print("Loading dataset from Hugging Face...")
dataset = load_dataset("XANJEEV/amazon-product-reviews")
df = dataset["train"].to_pandas()
print(f"  Loaded {len(df)} rows | Columns: {list(df.columns)}")
raw_row_count = len(df)

# --- Step 1: Rating parsing and validation ---
df = df.copy()
df["rating_num"] = df["rating"].apply(parse_rating)
df = df.dropna(subset=["rating_num"])

print(f"  After rating validation: {len(df)} rows "
      f"(dropped {raw_row_count - len(df)} unparseable)")

# --- Step 2: Drop rows with missing key fields ---
df = df.dropna(subset=["user_id", "review", "category", "title"])
print(f"  After dropping null key fields: {len(df)} rows")

# --- Step 3: Category normalisation ---
df["category"] = df["category"].apply(
    lambda x: CATEGORY_MAP.get(str(x).strip().lower(), str(x).strip().title())
)

# --- Step 4: Text cleaning ---
for column in ["user_id", "review", "title"]:
    if column in df.columns:
        df[column] = df[column].apply(clean_text)

# Also clean user column if present
if "user" in df.columns:
    df["user"] = df["user"].apply(clean_text)

# --- Step 5: Remove placeholder text ---
for column in ["user_id", "review", "title"]:
    df = df[non_placeholder_mask(df[column])]

# --- Step 6: Filter short reviews ---
df["review_word_count"] = df["review"].str.split().str.len()
df = df[df["review_word_count"] >= MIN_REVIEW_WORDS]
print(f"  After min {MIN_REVIEW_WORDS}-word filter: {len(df)} rows")

# Step 6b: Cap review length
df["review"] = df["review"].str[:MAX_REVIEW_CHARS]

# --- Step 7: Finalise columns ---
df["rating_num"] = df["rating_num"].astype(float)
df["review_text"] = df["review"]
df["date_parsed"] = df.get("date", pd.Series([pd.NaT] * len(df))).apply(parse_date)
df["item_id"] = df.apply(make_item_id, axis=1)

# --- Step 8: Deduplication ---
dedup_cols = ["user_id", "item_id", "review_text", "rating_num"]
before_dedup = len(df)
df = df.drop_duplicates(subset=dedup_cols)
print(f"  Deduplication removed {before_dedup - len(df)} rows")

# --- Step 9: Sort for reproducibility ---
sort_cols = [c for c in ["date_parsed", "user_id", "item_id"] if c in df.columns]
df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

df = df.drop(columns=["review_word_count"])

require_enough_rows(df)
require_stratifiable_ratings(df)

print(f"\nClean dataset: {len(df)} rows")
print(f"  Users: {df['user_id'].nunique()}")
print(f"  Items: {df['item_id'].nunique()}")
print(f"  Categories: {df['category'].nunique()}")
print(f"  Categories: {sorted(df['category'].unique())}")
print(f"  Rating distribution:\n{df['rating_num'].value_counts().sort_index()}")

clean_path = RAW_DIR / "amazon_reviews_clean.parquet"
df.to_parquet(clean_path, index=False)
print(f"\nSaved: {clean_path}")

# --- Train / test split ---
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=42, stratify=df["rating_num"]
)
train_df.to_parquet(PROC_DIR / "train.parquet", index=False)
test_df.to_parquet(PROC_DIR / "test.parquet", index=False)

# --- Cleaning report ---
total_dropped = raw_row_count - len(df)
dropped_reasons = {
    "unparseable_rating": int(raw_row_count - len(df[df["rating_num"].notna()])),
    "null_key_fields": int(
        len(df[df["rating_num"].notna()]) - len(df.dropna(subset=["user_id", "review", "category", "title"]))
    ),
    "placeholder_or_too_short": int(
        len(df.dropna(subset=["user_id", "review", "category", "title"])) - before_dedup
    ),
    "duplicates": int(before_dedup - len(df)),
}

cleaning_report = {
    "raw_row_count": int(raw_row_count),
    "clean_row_count": int(len(df)),
    "total_dropped": int(total_dropped),
    "dropped_by_reason": dropped_reasons,
    "train_row_count": int(len(train_df)),
    "test_row_count": int(len(test_df)),
    "distinct_users": int(df["user_id"].nunique()),
    "distinct_items": int(df["item_id"].nunique()),
    "distinct_categories": int(df["category"].nunique()),
    "categories": sorted(df["category"].unique()),
    "rating_distribution": {
        str(k): int(v) for k, v in df["rating_num"].value_counts().sort_index().items()
    },
    "cleaning_rules": {
        "min_review_words": MIN_REVIEW_WORDS,
        "max_review_chars": MAX_REVIEW_CHARS,
        "category_normalisation": "canonical map (12 → 10 categories)",
        "text_cleaning": "strip HTML entities, strip URLs, collapse whitespace",
        "dedupe_key": dedup_cols,
        "split": "stratified_by_rating_num",
        "random_seed": 42,
    },
}
with open(PROC_DIR / "cleaning_report.json", "w", encoding="utf-8") as f:
    json.dump(cleaning_report, f, indent=2)

print(f"\nTrain: {len(train_df)} rows | Test: {len(test_df)} rows")
print(f"Saved: {PROC_DIR / 'cleaning_report.json'}")
print("SCRIPT 01 COMPLETE")
