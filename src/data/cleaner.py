"""Text cleaning pipeline — produces data/processed/tweets_clean.csv.

Usage
-----
    from src.data.cleaner import clean_dataset
    df_clean = clean_dataset(df_raw)
    df_clean.to_csv("data/processed/tweets_clean.csv", index=False)
"""

import re

import emoji
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline and return the processed DataFrame.

    Input must have at minimum: text, created_at, label (optional).
    Output schema matches DATA_SCHEMA.md processed table.
    """
    df = df.copy()

    df["text_original"] = df["text"].astype(str)
    df["text_clean"] = df["text_original"].apply(_clean_text)

    # Normalise timestamp → UTC, derive hour bucket
    if pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        df["created_at"] = df["created_at"].dt.tz_localize(
            "UTC") if df["created_at"].dt.tz is None else df["created_at"]
    else:
        df["created_at"] = pd.to_datetime(
            df["created_at"], utc=True, errors="coerce")

    df["hour_bucket"] = df["created_at"].dt.floor("h")

    # Engagement signal
    rt = df["retweet_count"] if "retweet_count" in df.columns else 0
    fav = df["favorite_count"] if "favorite_count" in df.columns else 0
    df["engagement"] = rt + fav

    # Ensure label column exists
    if "label" not in df.columns:
        df["label"] = pd.NA

    # Select and order output columns
    keep = ["id", "text_clean", "text_original",
            "created_at", "hour_bucket", "label", "engagement"]
    available = [c for c in keep if c in df.columns]
    df = df[available].reset_index(drop=True)

    _validate_processed(df)
    print(f"Cleaned dataset: {df.shape[0]:,} rows")
    print(df.head(3))
    return df


def make_sample(df: pd.DataFrame, n: int = 100, random_state: int = 42) -> pd.DataFrame:
    """Return a stratified n-row sample (50/50 crisis/normal if labels exist)."""
    if "label" in df.columns and df["label"].notna().all():
        per_class = n // 2
        groups = []
        for lbl in [0, 1]:
            subset = df[df["label"] == lbl]
            groups.append(subset.sample(
                min(per_class, len(subset)), random_state=random_state))
        return pd.concat(groups).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df.sample(min(n, len(df)), random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"http\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ASCII_PUNCT_RE = re.compile(r"[^\x00-\x7F]+")


def _clean_text(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = emoji.replace_emoji(text, replace="")
    text = _HASHTAG_RE.sub(r"\1", text)          # keep the word, drop the #
    text = _NON_ASCII_PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.lower()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_processed(df: pd.DataFrame) -> None:
    assert "text_clean" in df.columns
    assert df["text_clean"].isnull().sum() == 0, "text_clean has nulls"
    if "label" in df.columns and df["label"].notna().all():
        assert df["label"].isin([0, 1]).all(), "label has non-binary values"
    print("Processed validation passed")
