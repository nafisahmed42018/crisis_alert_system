"""Unit tests for src/data/cleaner.py — runs without any CSV on disk."""

import pandas as pd
import pytest

from src.data.cleaner import _clean_text, clean_dataset, make_sample


# ---------------------------------------------------------------------------
# _clean_text
# ---------------------------------------------------------------------------

def test_removes_urls():
    assert "http" not in _clean_text("Check this http://example.com out")


def test_removes_mentions():
    assert "@user" not in _clean_text("Hello @user how are you")


def test_removes_emoji():
    result = _clean_text("Great news 🔥🚨")
    assert "🔥" not in result
    assert "🚨" not in result


def test_keeps_hashtag_word():
    result = _clean_text("This is a #flood event")
    assert "flood" in result
    assert "#" not in result


def test_lowercases():
    assert _clean_text("FLOOD WARNING") == "flood warning"


# ---------------------------------------------------------------------------
# clean_dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "text": [
            "Check http://t.co/abc @alice #flood 🚨",
            "Normal tweet here",
            "Another @bob mention",
        ],
        "created_at": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"], utc=True),
        "label": [1, 0, 1],
        "retweet_count": [10, 0, 5],
        "favorite_count": [2, 3, 1],
    })


def test_clean_dataset_schema(raw_df):
    result = clean_dataset(raw_df)
    for col in ["id", "text_clean", "text_original", "created_at", "hour_bucket", "label", "engagement"]:
        assert col in result.columns, f"Missing column: {col}"


def test_engagement_sum(raw_df):
    result = clean_dataset(raw_df)
    assert result.loc[0, "engagement"] == 12


def test_hour_bucket_type(raw_df):
    result = clean_dataset(raw_df)
    assert pd.api.types.is_datetime64_any_dtype(result["hour_bucket"])


# ---------------------------------------------------------------------------
# make_sample
# ---------------------------------------------------------------------------

def test_make_sample_size(raw_df):
    clean = clean_dataset(raw_df)
    sample = make_sample(clean, n=2)
    assert len(sample) == 2
