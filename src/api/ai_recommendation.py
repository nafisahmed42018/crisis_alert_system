"""AI-powered recommendation engine using OpenAI.

For MEDIUM / HIGH / CRITICAL alerts, calls GPT to generate a tweet-specific
escalation matrix and action plan.  Falls back to the static engine for LOW
alerts or when the OpenAI API is unavailable / not configured.
"""

from __future__ import annotations
from src.api.recommendation import (
    Recommendation,
    build_recommendation,
    classify_crisis_type,
    _TRAJECTORY_MAP,
    _STAKEHOLDERS,
)

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Levels that get AI-generated recommendations
_AI_LEVELS = {"MEDIUM", "HIGH", "CRITICAL"}

_SYSTEM_PROMPT = """\
You are a crisis management specialist. You will be given a social media post \
flagged as a potential crisis, along with its crisis type, severity level, and \
the monitoring keywords that surfaced it.

Your task is to produce a SPECIFIC, ACTIONABLE response plan tailored to the \
exact content of the post — not generic templates.

Respond with a JSON object containing exactly these four keys:
{
  "escalation_path": [list of 3–5 specific notification channels, e.g. "SMS to Regional Crisis Manager", "Slack #incidents", "Press release via PR team"],
  "escalation_timing": "one concise sentence on urgency and timeframe",
  "recommended_actions": [list of 4–6 concrete imperative actions to take immediately],
  "stakeholders": [list of 6–10 stakeholders that MUST include both internal roles AND location-specific external authorities]
}

Stakeholder rules (critical):
- Always start with the relevant internal roles (CEO, Legal, Communications, etc.).
- Then add location-specific external stakeholders inferred from the tweet text and keywords:
    e.g. if the crisis is in California → "California Governor's Office of Emergency Services", "FEMA Region 9"
    e.g. if it involves a coastline → "US Coast Guard District Commander", "NOAA Response Team"
    e.g. if it involves public health → "Local County Health Department", "CDC Regional Director"
- Name the actual agency / role — never write "Local Authorities" or "Relevant Government Body".
- Total list should be 6–10 stakeholders.

Other rules:
- Base everything on the tweet content and keywords, not just the crisis type label.
- escalation_path must name real channels (Slack, SMS, email, press release, hotline, etc.).
- recommended_actions must be imperative sentences starting with a verb.
- Return only valid JSON — no markdown, no explanation outside the JSON.
"""


def _build_user_prompt(
    text: str,
    alert_level: str,
    crisis_type: str,
    bert_score: float,
    lstm_score: float,
    lda_score: float,
    crisis_probability: float,
    search_keywords: list[str],
    base_stakeholders: list[str],
) -> str:
    keywords_str = ", ".join(
        search_keywords) if search_keywords else "not specified"
    stakeholders_str = ", ".join(
        base_stakeholders) if base_stakeholders else "none defined"
    return (
        f"Tweet: {text}\n\n"
        f"Alert level: {alert_level}\n"
        f"Crisis type: {crisis_type.replace('_', ' ').title()}\n"
        f"Crisis probability: {crisis_probability:.1%}\n"
        f"Model scores — BERT: {bert_score:.2f}, LSTM: {lstm_score:.2f}, LDA: {lda_score:.2f}\n\n"
        f"Search keywords that surfaced this tweet: {keywords_str}\n"
        f"Base internal stakeholders from our system: {stakeholders_str}\n\n"
        "Using the tweet text and search keywords, identify the geographic location or "
        "jurisdiction of this crisis and add the appropriate location-specific external "
        "authorities to the stakeholders list."
    )


async def build_recommendation_ai(
    text: str,
    alert_level: str,
    bert_score: float,
    lstm_score: float,
    lda_score: float,
    crisis_probability: float,
    lda_topic_words: Optional[list[str]] = None,
    search_keywords: Optional[list[str]] = None,
) -> Recommendation:
    """Return an AI-generated Recommendation for MEDIUM/HIGH/CRITICAL alerts.

    Falls back to the static engine for LOW alerts or if OpenAI is unavailable.
    """
    crisis_type = classify_crisis_type(text, lda_topic_words)
    trajectory, peak = _TRAJECTORY_MAP.get(alert_level, ("STABILIZING", None))

    if alert_level not in _AI_LEVELS:
        return build_recommendation(
            text=text,
            alert_level=alert_level,
            bert_score=bert_score,
            lstm_score=lstm_score,
            lda_score=lda_score,
            lda_topic_words=lda_topic_words,
        )

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning(
            "[ai_recommendation] OPENAI_API_KEY not set — falling back to static recommendations.")
        return build_recommendation(
            text=text,
            alert_level=alert_level,
            bert_score=bert_score,
            lstm_score=lstm_score,
            lda_score=lda_score,
            lda_topic_words=lda_topic_words,
        )

    try:
        # imported lazily so missing package doesn't break startup
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.3,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        text=text,
                        alert_level=alert_level,
                        crisis_type=crisis_type,
                        bert_score=bert_score,
                        lstm_score=lstm_score,
                        lda_score=lda_score,
                        crisis_probability=crisis_probability,
                        search_keywords=search_keywords or [],
                        base_stakeholders=_STAKEHOLDERS.get(crisis_type, []),
                    ),
                },
            ],
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        logger.info(
            "[ai_recommendation] OpenAI generated recommendation for %s alert.", alert_level)
        return Recommendation(
            crisis_type=crisis_type,
            escalation_path=data.get("escalation_path", []),
            escalation_timing=data.get("escalation_timing", ""),
            recommended_actions=data.get("recommended_actions", []),
            stakeholders=data.get(
                "stakeholders", _STAKEHOLDERS.get(crisis_type, [])),
            sentiment_trajectory=trajectory,
            predicted_peak=peak,
        )

    except Exception as exc:
        logger.error("[ai_recommendation] OpenAI call failed (%s: %s) — using static fallback.", type(
            exc).__name__, exc)
        return build_recommendation(
            text=text,
            alert_level=alert_level,
            bert_score=bert_score,
            lstm_score=lstm_score,
            lda_score=lda_score,
            lda_topic_words=lda_topic_words,
        )
