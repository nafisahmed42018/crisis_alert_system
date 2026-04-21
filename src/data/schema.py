"""Column definitions and expected dtypes for every dataset variant."""

from dataclasses import dataclass
from typing import Optional


# Minimum columns required after loading any raw dataset
REQUIRED_RAW_COLUMNS = {"text"}

# Columns produced by cleaner.py — these must exist in processed CSV
PROCESSED_COLUMNS = {
    "id",
    "text_clean",
    "text_original",
    "created_at",
    "hour_bucket",
    "label",
    "engagement",
}


@dataclass
class DatasetConfig:
    name: str
    text_col: str
    id_col: Optional[str]
    label_col: Optional[str]
    timestamp_col: Optional[str]
    timestamp_format: Optional[str]   # None → let pandas infer
    retweet_col: Optional[str]
    favorite_col: Optional[str]
    label_map: Optional[dict]         # remap raw label values → {0, 1}


DISASTER_TWEETS = DatasetConfig(
    name="disaster_tweets",
    text_col="text",
    id_col="id",
    label_col="target",
    timestamp_col=None,               # not present in Kaggle dataset
    timestamp_format=None,
    retweet_col=None,
    favorite_col=None,
    label_map=None,                   # already 0/1
)

SENTIMENT140 = DatasetConfig(
    name="sentiment140",
    text_col="text",
    id_col="ids",
    label_col="target",
    timestamp_col="date",
    timestamp_format="%a %b %d %H:%M:%S PDT %Y",
    retweet_col=None,
    favorite_col=None,
    label_map={0: 0, 4: 1},
)

CLIMATE_CHANGE = DatasetConfig(
    name="climate_change",
    text_col="message",
    id_col="tweetid",
    label_col="sentiment",
    timestamp_col=None,
    timestamp_format=None,
    retweet_col=None,
    favorite_col=None,
    label_map={2: 0, 1: 0, 0: 1, -1: 1},  # treat anti/denial as crisis signal
)

DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "disaster_tweets": DISASTER_TWEETS,
    "sentiment140": SENTIMENT140,
    "climate_change": CLIMATE_CHANGE,
}
