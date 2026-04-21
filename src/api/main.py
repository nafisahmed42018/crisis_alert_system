"""FastAPI backend for the Crisis Alert System.

Run from project root:
    py -3.13 -m uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations
from src.api.x_client import search_recent, RawTweet
from src.api.ai_recommendation import build_recommendation_ai
from src.alerts.alert_engine import AlertEngine
from src.models.ensemble import CrisisEnsemble
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timezone
import uuid
import time
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Crisis Alert API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-loaded ensemble (loaded once, then cached)
# ---------------------------------------------------------------------------

_ensemble: Optional[CrisisEnsemble] = None
_alert_engine = AlertEngine(min_level="LOW")


def get_ensemble() -> CrisisEnsemble:
    global _ensemble
    if _ensemble is None:
        _ensemble = CrisisEnsemble.load(
            bert_path=str(ROOT / "outputs/models/bert_v1"),
            lda_path=str(ROOT / "outputs/models/lda_v1"),
            lstm_path=str(ROOT / "outputs/models/lstm_v1"),
        )
    return _ensemble


# ---------------------------------------------------------------------------
# In-memory alert store (recent 500 alerts)
# ---------------------------------------------------------------------------

_alert_history: list[dict] = []
_MAX_HISTORY = 500


def _store_alert(record: dict) -> None:
    _alert_history.append(record)
    if len(_alert_history) > _MAX_HISTORY:
        _alert_history.pop(0)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str
    demo_mode: bool = True


class FetchRequest(BaseModel):
    keywords:       List[str]
    max_results:    int = 50
    demo_mode:      bool = True


class ScoreResult(BaseModel):
    id:                 str
    text:               str
    bert_score:         float
    lstm_score:         float
    lda_score:          float
    crisis_probability: float
    alert_level:        str
    crisis_type:        str
    escalation_path:    List[str]
    escalation_timing:  str
    recommended_actions: List[str]
    stakeholders:       List[str]
    sentiment_trajectory: str
    predicted_peak:     Optional[str]
    source:             str = "manual"
    created_at:         str = ""
    public_metrics:     dict = {}
    timestamp:          str = ""


class FetchResponse(BaseModel):
    keywords:    List[str]
    fetched:     int
    results:     List[ScoreResult]
    x_api_live:  bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _score_texts(
    texts: list[str],
    demo_mode: bool,
    sources: list[dict] | None = None,
    search_keywords: list[str] | None = None,
) -> list[ScoreResult]:
    ensemble = get_ensemble()
    df = ensemble.predict_df(texts, demo_mode=demo_mode)

    # Fan out AI recommendation calls in parallel for all rows
    rows = list(df.iterrows())
    recs = await asyncio.gather(*[
        build_recommendation_ai(
            text=row["text"],
            alert_level=row["alert_level"],
            bert_score=float(row["bert_score"]),
            lstm_score=float(row["lstm_score"]),
            lda_score=float(row["lda_score"]),
            crisis_probability=float(row["crisis_probability"]),
            search_keywords=search_keywords,
        )
        for _, row in rows
    ])

    results: list[ScoreResult] = []
    for (i, row), rec in zip(rows, recs):
        src = sources[i] if sources else {}
        record = ScoreResult(
            id=src.get("id", str(uuid.uuid4())),
            text=row["text"],
            bert_score=round(float(row["bert_score"]), 4),
            lstm_score=round(float(row["lstm_score"]), 4),
            lda_score=round(float(row["lda_score"]), 4),
            crisis_probability=round(float(row["crisis_probability"]), 4),
            alert_level=row["alert_level"],
            crisis_type=rec.crisis_type,
            escalation_path=rec.escalation_path,
            escalation_timing=rec.escalation_timing,
            recommended_actions=rec.recommended_actions,
            stakeholders=rec.stakeholders,
            sentiment_trajectory=rec.sentiment_trajectory,
            predicted_peak=rec.predicted_peak,
            source=src.get("source", "manual"),
            created_at=src.get("created_at", ""),
            public_metrics=src.get("public_metrics", {}),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        results.append(record)
        _store_alert(record.model_dump())

    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/api/analyze", response_model=ScoreResult)
async def analyze(req: AnalyzeRequest):
    """Score a single tweet text through the ensemble."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    results = await _score_texts([req.text.strip()], demo_mode=req.demo_mode)
    return results[0]


@app.post("/api/fetch-and-analyze", response_model=FetchResponse)
async def fetch_and_analyze(req: FetchRequest):
    """Fetch recent tweets from X API matching keywords, then score them."""
    raw_tweets: list[RawTweet] = search_recent(
        keywords=req.keywords,
        max_results=min(req.max_results, 100),
    )

    x_api_live = len(raw_tweets) > 0

    if raw_tweets:
        texts = [t.text for t in raw_tweets]
        sources = [
            {
                "id": t.id,
                "source": "x_api",
                "created_at": t.created_at,
                "public_metrics": t.public_metrics,
            }
            for t in raw_tweets
        ]
    else:
        # Fallback: synthesize demo tweets from keywords
        texts = [
            f"Breaking: {kw} situation reported — emergency teams responding" for kw in req.keywords[:5]]
        sources = [{"id": str(uuid.uuid4()), "source": "demo"} for _ in texts]

    results = await _score_texts(texts, demo_mode=req.demo_mode, sources=sources, search_keywords=req.keywords)
    return FetchResponse(
        keywords=req.keywords,
        fetched=len(raw_tweets),
        results=results,
        x_api_live=x_api_live,
    )


@app.get("/api/alerts", response_model=List[ScoreResult])
def get_alerts(
    level:  Optional[str] = Query(
        None, description="Filter by alert level: LOW/MEDIUM/HIGH/CRITICAL"),
    limit:  int = Query(50, ge=1, le=500),
):
    """Return recent alert history (in-memory, most recent first)."""
    items = list(reversed(_alert_history))
    if level:
        items = [a for a in items if a["alert_level"] == level.upper()]
    return items[:limit]


@app.delete("/api/alerts")
def clear_alerts():
    """Clear alert history."""
    _alert_history.clear()
    return {"cleared": True}
