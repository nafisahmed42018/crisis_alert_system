"""Alert data structures."""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List
import json


ALERT_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

LEVEL_COLORS = {
    "LOW":      "#6c757d",
    "MEDIUM":   "#ffc107",
    "HIGH":     "#fd7e14",
    "CRITICAL": "#e63946",
}


@dataclass
class CrisisAlert:
    alert_id:          str
    level:             str
    crisis_probability: float
    bert_score:        float
    lstm_score:        float
    lda_score:         float
    trigger_text:      str
    timestamp:         str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    top_tweets:        List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
