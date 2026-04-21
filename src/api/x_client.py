"""X (Twitter) API v2 client — search recent tweets by keyword."""

from __future__ import annotations
import os
from typing import List, Optional
from dataclasses import dataclass

import tweepy
from dotenv import load_dotenv

load_dotenv()

_BEARER = os.getenv("X_BEARER_TOKEN", "")


@dataclass
class RawTweet:
    id:         str
    text:       str
    author_id:  str
    created_at: str
    public_metrics: dict   # retweet_count, like_count, reply_count, impression_count


def get_client() -> Optional[tweepy.Client]:
    if not _BEARER:
        return None
    return tweepy.Client(bearer_token=_BEARER, wait_on_rate_limit=False)


def search_recent(
    keywords: List[str],
    max_results: int = 50,
    exclude_retweets: bool = True,
    lang: str = "en",
) -> List[RawTweet]:
    """Search recent tweets matching any of the given keywords.

    Returns [] if no bearer token is configured or on API errors.
    max_results is clamped to [10, 100] by the X API.
    """
    client = get_client()
    if client is None:
        return []

    query_parts = [f'"{kw}"' for kw in keywords]
    query = "(" + " OR ".join(query_parts) + f') lang:{lang}'
    if exclude_retweets:
        query += " -is:retweet"

    max_results = max(10, min(100, max_results))

    try:
        response = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["created_at", "author_id", "public_metrics"],
        )
    except tweepy.TweepyException:
        return []

    if not response.data:
        return []

    results: List[RawTweet] = []
    for tweet in response.data:
        metrics = tweet.public_metrics or {}
        results.append(RawTweet(
            id=str(tweet.id),
            text=tweet.text,
            author_id=str(tweet.author_id),
            created_at=str(tweet.created_at),
            public_metrics=metrics,
        ))
    return results
