# ============================================================
# SCRIPT 02: VADER Feature Extraction
# Environment: Google Colab
# Input: data/processed/train.parquet and data/processed/test.parquet
# Output: data/processed/rating_features_train.parquet
#         data/processed/rating_features_test.parquet
#         data/processed/rating_features.parquet (train compatibility copy)
# Note: Colab setup must run nltk.download("vader_lexicon", quiet=True)
# ============================================================

from pathlib import Path
import math
import re

from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd


PROC_DIR = Path("data/processed")
TRAIN_PATH = PROC_DIR / "train.parquet"
TEST_PATH = PROC_DIR / "test.parquet"
TRAIN_FEATURE_PATH = PROC_DIR / "rating_features_train.parquet"
TEST_FEATURE_PATH = PROC_DIR / "rating_features_test.parquet"
COMPAT_FEATURE_PATH = PROC_DIR / "rating_features.parquet"
PROC_DIR.mkdir(parents=True, exist_ok=True)

ASPECT_KEYWORDS = {
    "quality": {
        "quality", "durable", "durability", "sturdy", "solid", "fragile",
        "broken", "broke", "defect", "defective", "material", "made",
        "excellent", "poor", "cheaply",
    },
    "price": {
        "price", "priced", "cost", "costs", "expensive", "cheap",
        "affordable", "budget", "deal", "overpriced", "money", "worth",
    },
    "service": {
        "service", "support", "seller", "customer", "refund", "return",
        "replacement", "warranty", "helpful", "unhelpful",
    },
    "value": {
        "value", "worth", "money", "deal", "bargain", "waste", "price",
        "recommend", "recommended",
    },
    "usability": {
        "easy", "easier", "easiest", "difficult", "hard", "use", "using",
        "setup", "install", "installed", "instructions", "convenient",
        "confusing", "works", "worked",
    },
    "delivery": {
        "delivery", "delivered", "shipping", "shipped", "arrived", "late",
        "fast", "slow", "package", "packaging", "damaged", "box",
    },
}

NEGATIVE_ASPECT_KEYWORDS = {
    "quality": {
        "bad", "poor", "terrible", "awful", "fragile", "broken", "broke",
        "break", "defect", "defective", "crooked", "cracked", "scratch",
        "scratches", "flimsy", "damaged", "smoke", "stopped", "dead",
        "failed", "failure", "unusable", "useless", "cheaply",
    },
    "service": {
        "refund", "return", "returned", "replacement", "warranty", "support",
        "technician", "company", "seller", "late", "unhelpful", "complaint",
    },
    "value": {
        "overpriced", "expensive", "waste", "wasted", "overrated", "cheaply",
        "flimsy", "not worth", "worthless", "pricey",
    },
    "usability": {
        "difficult", "hard", "confusing", "unusable", "stopped", "failed",
        "failure", "problem", "problems", "issue", "issues", "hot", "overheat",
        "overheating", "battery", "charge", "charging",
    },
}

LOW_PRICE_WORDS = {
    "cheap", "affordable", "budget", "bargain", "deal", "inexpensive",
}
HIGH_PRICE_WORDS = {"expensive", "overpriced", "costly", "premium", "pricey"}
POLITE_WORDS = {"please", "thanks", "thank", "appreciate", "appreciated"}
TOKEN_RE = re.compile(r"[a-z0-9']+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "awful", "horrible", "broken", "broke",
    "break", "defect", "defective", "crooked", "cracked", "scratch",
    "scratches", "flimsy", "damaged", "damage", "smoke", "stopped",
    "dead", "failed", "failure", "unusable", "useless", "waste",
    "wasted", "overpriced", "worthless", "pricey", "refund", "return",
    "returned", "disappointed", "disappointing", "never", "worst",
    "cheaply", "problem", "problems", "issue", "issues", "overheat",
    "overheating", "corrupted", "crash", "crashed", "corrupt",
    # Data-driven additions from actual 1-3-star reviews
    "hot", "loud", "slow", "buggy", "glitch", "glitching", "glitches",
    "junk", "garbage", "trash", "sucks", "suck", "abysmal", "tore",
    "tear", "tears", "ripped", "rip", "heat", "useless", "nightmare",
    "disgusted", "disgusting", "crapped", "crap", "faulty", "died",
    # Intensity markers common in 1-star reviews
    "horrible", "atrocious", "pathetic", "appalling",
}

COMPLAINT_PHRASES = [
    "stopped working", "does not work", "did not work", "won't work",
    "not worth", "broke after", "broke within", "nothing but problems",
    "do not buy", "do not purchase", "waste of money", "would not recommend",
    "fell apart", "came apart", "ripped", "torn", "stopped charging",
    "dead on arrival", "doa", "not as described", "missing parts",
    # Data-driven additions from actual low-rated reviews
    "cracked screen", "screen cracked", "screen broke", "cracked",
    "never again", "rip off", "garbage", "junk", "trash",
    "started glitching", "started glitch", "broke immediately",
    "too hot", "gets too hot", "loud fan", "fan is loud",
    "over heat", "over heating", "so slow",
    "poor quality", "terrible quality", "broke easily",
    "torn seam", "tore", "tear in", "fabric tore", "bent frame",
    "useless", "worthless", "defective",
    "no repair", "no warranty", "won't charge",
    "doesn't last", "didn't last",
]

TONE_MAP = {
    "brief": 0, "polite": 1, "casual": 2, "direct": 3, "detailed": 4, "angry": 5,
}
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
]

STRONG_QUALITY_NEG = {
    "broken", "broke", "defective", "crooked", "cracked", "damaged",
    "useless", "unusable", "smoke", "stopped", "glitch", "glitching",
    "crashed", "corrupted", "junk", "garbage", "trash", "tore",
    "torn", "ripped", "nightmare", "died",
}
STRONG_SERVICE_NEG = {
    "refund", "returned", "replacement", "warranty", "unhelpful",
    "no repair", "useless",
}
STRONG_VALUE_NEG = {
    "overpriced", "waste", "worthless", "overrated", "rip off",
    "abysmal",
}
STRONG_USABLE_NEG = {
    "confusing", "unusable", "overheat", "overheating", "crashed",
    "corrupted", "glitch", "glitching", "buggy", "slow", "hot",
    "loud", "hard", "difficult",
}


def clamp(value, lower=0.0, upper=1.0):
    """Clip a finite number into an expected numeric feature range."""
    number = float(value)
    if not math.isfinite(number):
        return lower
    return max(lower, min(upper, number))


def tokenize(text):
    """Tokenize review text into lowercase word-like tokens."""
    return TOKEN_RE.findall(str(text).lower())


def keyword_count(tokens, keywords):
    """Count keyword hits in tokenized review text."""
    return sum(1 for token in tokens if token in keywords)


def density_score(count, token_count, multiplier=8.0):
    """Convert sparse keyword counts into a bounded 0-1 feature."""
    if token_count <= 0:
        return 0.0
    return clamp((count / token_count) * multiplier)


def phrase_count(text, phrases):
    """Count simple multi-word phrase hits in normalized lowercase text."""
    lowered = str(text).lower()
    return sum(1 for phrase in phrases if " " in phrase and phrase in lowered)


def relevant_text(text, keywords):
    """Return sentences that mention an aspect, or an empty string if absent."""
    raw_text = str(text)
    sentences = SENTENCE_RE.split(raw_text)
    relevant = []
    for sentence in sentences:
        sentence_tokens = set(tokenize(sentence))
        if sentence_tokens.intersection(keywords):
            relevant.append(sentence)
    return " ".join(relevant)


def vader_compound(analyzer, text):
    """Return VADER compound sentiment for non-empty text."""
    if not str(text).strip():
        return 0.0
    return float(analyzer.polarity_scores(str(text))["compound"])


def aspect_score(analyzer, text, keywords):
    """Score the sentiment of aspect-specific sentences when present."""
    aspect_text = relevant_text(text, keywords)
    if not aspect_text:
        return 0.0, False
    return vader_compound(analyzer, aspect_text), True


def aspect_bucket(score, mentioned):
    """Map aspect sentiment into the old -1/0/1 feature shape."""
    if not mentioned or abs(score) < 0.05:
        return 0
    return 1 if score > 0 else -1


def complaint_pressure(tokens, text, aspect, review_length):
    """Estimate explicit complaint/defect pressure for an aspect."""
    keywords = NEGATIVE_ASPECT_KEYWORDS.get(aspect, set())
    single_words = {word for word in keywords if " " not in word}
    hits = keyword_count(tokens, single_words) + phrase_count(text, keywords)
    return density_score(hits, review_length, multiplier=12.0)


def adjusted_signal(base_score, complaint_score):
    """Penalize sentiment when explicit defect/complaint terms are present."""
    return clamp(base_score - (1.35 * complaint_score), -1.0, 1.0)


def infer_price_level(tokens):
    """Infer low/medium/high price level from price language."""
    low_hits = keyword_count(tokens, LOW_PRICE_WORDS)
    high_hits = keyword_count(tokens, HIGH_PRICE_WORDS)
    if high_hits > low_hits:
        return 2
    if low_hits > high_hits:
        return 0
    return 1


def infer_tone(tokens, review_length, compound, text):
    """Infer the previous tone enum from sentiment and text statistics."""
    if compound <= -0.55 or (str(text).count("!") >= 2 and compound < -0.2):
        return TONE_MAP["angry"]
    if keyword_count(tokens, POLITE_WORDS):
        return TONE_MAP["polite"]
    if review_length >= 80:
        return TONE_MAP["detailed"]
    if review_length <= 15:
        return TONE_MAP["brief"]
    if review_length <= 40:
        return TONE_MAP["direct"]
    return TONE_MAP["casual"]


def infer_strictness(tokens, review_length, compound, text):
    """Infer reviewer strictness without using the target star rating."""
    negative_pressure = max(0.0, -compound)
    complaint_hits = keyword_count(
        tokens,
        {
            "bad", "poor", "terrible", "awful", "broken", "defective",
            "waste", "refund", "return", "disappointed", "unusable",
            "overpriced", "late", "damaged", "hard", "difficult",
        },
    )
    detail_pressure = clamp(review_length / 120.0)
    punctuation_pressure = clamp(str(text).count("!") / 4.0)
    complaint_pressure = density_score(complaint_hits, review_length, multiplier=12.0)
    return clamp(
        (0.45 * negative_pressure)
        + (0.30 * complaint_pressure)
        + (0.15 * detail_pressure)
        + (0.10 * punctuation_pressure)
    )


def override_vader_signal(signal_value, tokens, strong_neg_set, penalty_per_word=0.5):
    """Override VADER signal when strong negative keywords exist but VADER is wrong."""
    hits = tokens.intersection(strong_neg_set)
    if hits and signal_value > -0.3:
        return max(-0.8, -penalty_per_word * len(hits))
    return signal_value


def compute_negativity_ratio(tokens):
    """Compute ratio of negative keywords to total tokens (bounded 0-1)."""
    neg_hits = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    ratio = neg_hits / max(len(tokens), 1)
    return min(ratio * 10, 1.0)


def compute_complaint_phrases(text):
    """Count complaint phrase hits (bounded 0-1)."""
    text_lower = str(text).lower()
    hits = sum(1 for p in COMPLAINT_PHRASES if p in text_lower)
    return min(hits / 3.0, 1.0)


def compute_caps_intensity(text):
    """Measure intensity from ALL CAPS words and strong punctuation."""
    raw = str(text)
    words = raw.split()
    if not words:
        return 0.0
    # Fraction of words that are ALL CAPS (3+ chars, mostly uppercase)
    caps_words = sum(
        1 for w in words
        if len(w) >= 3 and sum(1 for c in w if c.isupper()) >= len(w) * 0.7
    )
    caps_ratio = caps_words / max(len(words), 1)
    # Bonus for excessive exclamation marks and emoji presence
    excl_bonus = min(raw.count("!") / 5.0, 0.3)
    emoji_bonus = 0.1 if any(c in raw for c in "🤬😤😡😠💩👎😶😢😭😒") else 0.0
    return min(caps_ratio + excl_bonus + emoji_bonus, 1.0)


def feature_row(row, analyzer):
    """Build one model-ready feature row from a raw training review."""
    text = row.get("review_text", "")
    title_text = row.get("title", "")
    tokens = tokenize(text)
    token_set = set(tokens)
    review_length = len(tokens)
    overall_score = vader_compound(analyzer, text)

    aspect_scores = {}
    aspect_mentions = {}
    complaint_scores = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        score, mentioned = aspect_score(analyzer, text, keywords)
        aspect_scores[aspect] = score
        aspect_mentions[aspect] = mentioned
        complaint_scores[aspect] = complaint_pressure(tokens, text, aspect, review_length)

    price_hits = keyword_count(
        tokens,
        ASPECT_KEYWORDS["price"] | ASPECT_KEYWORDS["value"],
    )
    service_hits = keyword_count(
        tokens,
        ASPECT_KEYWORDS["service"] | ASPECT_KEYWORDS["delivery"],
    )
    quality_hits = keyword_count(
        tokens,
        ASPECT_KEYWORDS["quality"] | ASPECT_KEYWORDS["usability"],
    )

    quality_signal = adjusted_signal(
        aspect_scores["quality"] if aspect_mentions["quality"] else overall_score,
        complaint_scores["quality"],
    )
    service_signal = adjusted_signal(
        aspect_scores["service"]
        if aspect_mentions["service"]
        else aspect_scores["delivery"],
        max(complaint_scores["service"], complaint_scores["delivery"]),
    )
    value_signal = adjusted_signal(
        aspect_scores["value"] if aspect_mentions["value"] else aspect_scores["price"],
        max(complaint_scores["value"], complaint_scores["price"]),
    )
    usability_signal = adjusted_signal(
        aspect_scores["usability"]
        if aspect_mentions["usability"]
        else overall_score,
        complaint_scores["usability"],
    )

    # VADER override: when strong negative keywords present and VADER is wrong
    quality_signal = override_vader_signal(quality_signal, token_set, STRONG_QUALITY_NEG)
    service_signal = override_vader_signal(service_signal, token_set, STRONG_SERVICE_NEG)
    value_signal = override_vader_signal(value_signal, token_set, STRONG_VALUE_NEG)
    usability_signal = override_vader_signal(usability_signal, token_set, STRONG_USABLE_NEG)

    stars = float(row["rating_num"])

    return {
        "review_id": int(row.name),
        "user_id": row.get("user_id", ""),
        "item_id": row.get("item_id", ""),
        "category": row.get("category", ""),
        "text": text,
        "stars": stars,
        "budget_sensitivity": density_score(price_hits, review_length),
        "service_sensitivity": density_score(service_hits, review_length),
        "quality_sensitivity": density_score(quality_hits, review_length),
        "strictness": infer_strictness(tokens, review_length, overall_score, text),
        "quality_signal": quality_signal,
        "service_signal": service_signal,
        "value_signal": value_signal,
        "usability_signal": usability_signal,
        "price_level": infer_price_level(tokens),
        "tone": infer_tone(tokens, review_length, overall_score, text),
        "aspect_quality": aspect_bucket(aspect_scores["quality"], aspect_mentions["quality"]),
        "aspect_price": aspect_bucket(aspect_scores["price"], aspect_mentions["price"]),
        "aspect_service": aspect_bucket(aspect_scores["service"], aspect_mentions["service"]),
        "aspect_value": aspect_bucket(aspect_scores["value"], aspect_mentions["value"]),
        "aspect_usability": aspect_bucket(aspect_scores["usability"], aspect_mentions["usability"]),
        "aspect_delivery": aspect_bucket(aspect_scores["delivery"], aspect_mentions["delivery"]),
        "review_length": review_length,
        "negativity_ratio": compute_negativity_ratio(tokens),
        "complaint_phrases": compute_complaint_phrases(text),
        "caps_intensity": compute_caps_intensity(text),
        "title_compound": vader_compound(analyzer, title_text),
        "has_but_flag": 1 if " but " in text.lower() else 0,
        "title_review_sentiment_gap": abs(
            vader_compound(analyzer, title_text) - overall_score
        ),
    }


try:
    sentiment_analyzer = SentimentIntensityAnalyzer()
except LookupError as exc:
    raise RuntimeError(
        "VADER lexicon is missing. In Colab setup, run: "
        'nltk.download("vader_lexicon", quiet=True)'
    ) from exc


def extract_features(input_path, output_path, split_name):
    """Extract feature rows for one split and save them to parquet."""
    print(f"Loading {split_name} data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Shape: {df.shape}")

    required_columns = {"review_text", "rating_num", "category", "user_id", "item_id", "title"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise RuntimeError(f"{split_name} data missing required columns: {missing_columns}")

    print(f"Extracting VADER, keyword, and text-statistic features for {split_name}...")
    feature_df = pd.DataFrame(
        [feature_row(row, sentiment_analyzer) for _, row in df.iterrows()]
    )

    missing_features = sorted(set(FEATURE_COLS) - set(feature_df.columns))
    if missing_features:
        raise RuntimeError(f"Feature extraction missing columns: {missing_features}")

    if feature_df[FEATURE_COLS].isna().any().any():
        bad_columns = (
            feature_df[FEATURE_COLS]
            .columns[feature_df[FEATURE_COLS].isna().any()]
            .tolist()
        )
        raise RuntimeError(f"Feature extraction produced NaN values in: {bad_columns}")

    feature_df.to_parquet(output_path, index=False)
    print(f"{split_name} feature table: {feature_df.shape[0]} rows x {feature_df.shape[1]} columns")
    print(f"Saved to: {output_path}")
    return feature_df


train_features = extract_features(TRAIN_PATH, TRAIN_FEATURE_PATH, "train")
test_features = extract_features(TEST_PATH, TEST_FEATURE_PATH, "test")
train_features.to_parquet(COMPAT_FEATURE_PATH, index=False)

print(f"Compatibility feature table saved to: {COMPAT_FEATURE_PATH}")
print(f"Train columns: {list(train_features.columns)}")
print(f"Test columns: {list(test_features.columns)}")
print("SCRIPT 02 COMPLETE")
