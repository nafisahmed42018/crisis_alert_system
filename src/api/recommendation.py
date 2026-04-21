"""Recommendation engine: maps crisis context to actions and escalation paths."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import re


# ---------------------------------------------------------------------------
# Crisis type classifier — uses LDA topic words + BERT score to label the
# type of crisis so recommendations can be domain-specific.
# ---------------------------------------------------------------------------

_TOPIC_PATTERNS: list[tuple[str, list[str]]] = [
    ("environmental_disaster", [
        "fire", "flood", "earthquake", "hurricane", "wildfire", "tsunami",
        "explosion", "spill", "leak", "contamination", "toxic", "radiation",
        "oil", "chemical", "disaster", "emergency", "evacuation",
    ]),
    ("corporate_misconduct", [
        "fraud", "corruption", "scandal", "misconduct", "lawsuit", "sued",
        "investigation", "ceo", "executive", "company", "corporate",
        "breach", "cover", "illegal", "unethical", "negligence",
    ]),
    ("health_hazard", [
        "outbreak", "virus", "pandemic", "epidemic", "disease", "illness",
        "hospital", "patient", "death", "fatality", "infection", "toxic",
        "contaminated", "water", "food", "recall", "health", "medical",
    ]),
    ("policy_regulatory", [
        "government", "law", "regulation", "policy", "ban", "legislation",
        "senate", "congress", "parliament", "vote", "ruling", "court",
        "compliance", "fine", "penalty", "mandate", "order",
    ]),
    ("market_financial", [
        "stock", "market", "crash", "bankrupt", "bankruptcy", "default",
        "recession", "inflation", "layoff", "job", "unemployment",
        "economy", "financial", "investment", "fund", "bitcoin", "crypto",
    ]),
    ("civil_unrest", [
        "protest", "riot", "violence", "attack", "shooting", "bomb",
        "terror", "unrest", "demonstration", "clash", "strike", "march",
        "police", "military", "conflict", "war",
    ]),
]

_ESCALATION_MATRIX: dict[str, dict[str, list[str]]] = {
    "CRITICAL": {
        "channels": ["Real-time Dashboard", "SMS to Crisis Manager", "Slack #crisis-alerts", "Executive Brief"],
        "timing":   "Immediate — within 15 minutes",
    },
    "HIGH": {
        "channels": ["Real-time Dashboard", "Slack #alerts", "Email to Response Team"],
        "timing":   "Urgent — within 1 hour",
    },
    "MEDIUM": {
        "channels": ["Real-time Dashboard", "Daily Email Digest"],
        "timing":   "Monitor — within 4 hours",
    },
    "LOW": {
        "channels": ["Real-time Dashboard"],
        "timing":   "Routine — end-of-day report",
    },
}

_ACTION_MATRIX: dict[str, list[str]] = {
    "environmental_disaster": [
        "Contact Environmental Response Team",
        "Prepare Public Statement",
        "Notify Regulatory Authorities",
        "Activate Media Relations",
        "Monitor Social Sentiment Every 15 min",
    ],
    "corporate_misconduct": [
        "Contact Legal Team",
        "Brief Executive Leadership",
        "Prepare Crisis Communication",
        "Engage PR / Communications Director",
        "Monitor Stakeholder Sentiment",
    ],
    "health_hazard": [
        "Notify Public Health Officials",
        "Issue Safety Advisory",
        "Contact Medical Response Team",
        "Coordinate with Regulatory Bodies",
        "Provide Verified Information to Media",
    ],
    "policy_regulatory": [
        "Contact Government Relations Team",
        "Prepare Policy Position Statement",
        "Engage Advocacy / Lobbying Partners",
        "Monitor Legislative Timeline",
        "Brief Compliance Team",
    ],
    "market_financial": [
        "Contact Investor Relations",
        "Prepare Financial Disclosure",
        "Brief Financial Analysts",
        "Monitor Market Impact",
        "Coordinate with CFO",
    ],
    "civil_unrest": [
        "Contact Security Team",
        "Issue Staff Safety Advisory",
        "Coordinate with Local Authorities",
        "Prepare Crisis Communication",
        "Monitor Situation Every 30 min",
    ],
    "general": [
        "Monitor Situation Closely",
        "Prepare Response Briefing",
        "Alert Relevant Department Heads",
        "Document Incident Timeline",
    ],
}

_STAKEHOLDERS: dict[str, list[str]] = {
    "environmental_disaster": ["CEO", "Environmental Officer", "Communications Director", "Legal Team", "Operations"],
    "corporate_misconduct":   ["CEO", "Legal Team", "Communications Director", "Board of Directors"],
    "health_hazard":          ["CEO", "Chief Medical Officer", "Communications Director", "Operations"],
    "policy_regulatory":      ["CEO", "General Counsel", "Government Relations", "Compliance Officer"],
    "market_financial":       ["CEO", "CFO", "Investor Relations", "Board of Directors"],
    "civil_unrest":           ["CEO", "Head of Security", "Communications Director", "HR Director"],
    "general":                ["CEO", "Communications Director", "Department Head"],
}


@dataclass
class Recommendation:
    crisis_type:     str
    escalation_path: list[str]
    escalation_timing: str
    recommended_actions: list[str]
    stakeholders:    list[str]
    sentiment_trajectory: str   # ESCALATING / STABILIZING / DE_ESCALATING
    predicted_peak:  Optional[str]


def classify_crisis_type(text: str, lda_topic_words: Optional[List[str]] = None) -> str:
    """Infer crisis type from text + optional LDA topic words."""
    tokens = re.findall(r"\b\w+\b", (text + " " + " ".join(lda_topic_words or [])).lower())
    token_set = set(tokens)

    scores: dict[str, int] = {}
    for crisis_type, keywords in _TOPIC_PATTERNS:
        scores[crisis_type] = sum(1 for kw in keywords if kw in token_set)

    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "general"


def infer_trajectory(bert_score: float, lstm_score: float, lda_score: float) -> tuple[str, str]:
    """Return (trajectory_label, predicted_peak) based on model scores."""
    combined = 0.4 * bert_score + 0.4 * lstm_score + 0.2 * lda_score

    if lstm_score > 0.65 and bert_score > 0.65:
        return "ESCALATING", "4–8 hours"
    if combined > 0.55:
        return "STABILIZING", "1–2 hours"
    return "DE_ESCALATING", None


def build_recommendation(
    text: str,
    alert_level: str,
    bert_score: float,
    lstm_score: float,
    lda_score: float,
    lda_topic_words: Optional[List[str]] = None,
) -> Recommendation:
    crisis_type   = classify_crisis_type(text, lda_topic_words)
    escalation    = _ESCALATION_MATRIX[alert_level]
    actions       = _ACTION_MATRIX.get(crisis_type, _ACTION_MATRIX["general"])
    stakeholders  = _STAKEHOLDERS.get(crisis_type, _STAKEHOLDERS["general"])
    trajectory, peak = infer_trajectory(bert_score, lstm_score, lda_score)

    return Recommendation(
        crisis_type=crisis_type,
        escalation_path=escalation["channels"],
        escalation_timing=escalation["timing"],
        recommended_actions=actions,
        stakeholders=stakeholders,
        sentiment_trajectory=trajectory,
        predicted_peak=peak,
    )
