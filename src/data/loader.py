"""Load and validate a Twitter dataset CSV into a normalised DataFrame.

Usage
-----
    from src.data.loader import load_dataset
    df = load_dataset("data/raw/train.csv", dataset="disaster_tweets")
"""

from typing import Optional

import pandas as pd

from src.data.schema import DatasetConfig, DATASET_REGISTRY, REQUIRED_RAW_COLUMNS


def load_dataset(
    path: str,
    dataset: str = "disaster_tweets",
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """Load a raw CSV and apply minimal normalisation defined by DatasetConfig.

    Parameters
    ----------
    path:      Path to the raw CSV file.
    dataset:   Key from DATASET_REGISTRY — selects column mapping rules.
    chunksize: If set, read in chunks (useful for Sentiment140 at 1.6M rows).

    Returns
    -------
    pd.DataFrame with standardised column names: id, text, label (if present),
    created_at (if present), retweet_count, favorite_count, keyword (if present).
    """
    config: DatasetConfig = DATASET_REGISTRY[dataset]

    if chunksize:
        chunks = pd.read_csv(path, chunksize=chunksize, low_memory=False)
        df = pd.concat(
            (_normalize(chunk, config) for chunk in chunks),
            ignore_index=True,
        )
    else:
        df = _normalize(pd.read_csv(path, low_memory=False), config)

    _validate(df)
    print(f"Loaded {dataset}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(df.head(3))
    return df


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _normalize(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Rename columns, remap labels, parse timestamps."""
    rename = {}

    if config.text_col != "text":
        rename[config.text_col] = "text"

    if config.id_col and config.id_col != "id":
        rename[config.id_col] = "id"

    if config.label_col and config.label_col != "label":
        rename[config.label_col] = "label"

    if rename:
        df = df.rename(columns=rename)

    # Remap label values if needed
    if config.label_map and "label" in df.columns:
        df["label"] = df["label"].map(config.label_map)

    # Parse timestamp
    if config.timestamp_col and config.timestamp_col in df.columns:
        df = df.rename(columns={config.timestamp_col: "created_at"})
        df["created_at"] = pd.to_datetime(
            df["created_at"],
            format=config.timestamp_format,
            utc=True,
            errors="coerce",
        )
    elif "created_at" not in df.columns:
        df["created_at"] = pd.NaT

    # Ensure engagement columns exist
    for col in ("retweet_count", "favorite_count"):
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0).astype(int)

    # Decode URL-encoded keyword (Disaster Tweets quirk)
    if "keyword" in df.columns:
        from urllib.parse import unquote
        df["keyword"] = df["keyword"].fillna("").apply(unquote)

    # Drop duplicate tweet IDs
    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset="id").reset_index(drop=True)
        removed = before - len(df)
        if removed:
            print(f"  Removed {removed} duplicate IDs")

    return df


def _validate(df: pd.DataFrame) -> None:
    """Assert minimum required columns exist and text has no nulls."""
    missing = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_text = df["text"].isnull().sum()
    if null_text > 0:
        raise ValueError(f"Found {null_text} null values in 'text' column")

    if "label" in df.columns:
        invalid = ~df["label"].isin([0, 1])
        if invalid.any():
            raise ValueError(
                f"label column has non-binary values: {df.loc[invalid, 'label'].unique()}")

    print("Validation passed")
