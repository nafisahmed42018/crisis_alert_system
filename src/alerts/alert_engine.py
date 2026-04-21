"""Alert engine — converts ensemble results into structured CrisisAlert objects.

Usage
-----
    from src.alerts.alert_engine import AlertEngine
    engine  = AlertEngine()
    alerts  = engine.process(results_df)
    for alert in alerts:
        print(alert.to_json())
        engine.save(alert, "outputs/alerts/")
"""

import json
import os
import uuid
from typing import List

import pandas as pd

from src.alerts.alert_schema import CrisisAlert, ALERT_LEVELS


class AlertEngine:
    def __init__(self, min_level: str = "MEDIUM"):
        """Only emit alerts at or above min_level."""
        assert min_level in ALERT_LEVELS
        self._min_idx = ALERT_LEVELS.index(min_level)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, results_df: pd.DataFrame) -> List[CrisisAlert]:
        """Generate one alert per qualifying row (alert_level >= min_level).

        results_df must have columns produced by CrisisEnsemble.predict_df().
        """
        qualifying = results_df[
            results_df["alert_level"].apply(
                lambda lvl: ALERT_LEVELS.index(lvl) >= self._min_idx
            )
        ].copy()

        alerts = []
        for _, row in qualifying.iterrows():
            alert = CrisisAlert(
                alert_id=str(uuid.uuid4())[:8],
                level=row["alert_level"],
                crisis_probability=row["crisis_probability"],
                bert_score=row["bert_score"],
                lstm_score=row["lstm_score"],
                lda_score=row["lda_score"],
                trigger_text=row["text"][:280],
            )
            alerts.append(alert)

        return alerts

    def aggregate(self, results_df: pd.DataFrame) -> CrisisAlert:
        """Produce a single summary alert for a batch of tweets (e.g. one hour window).

        Uses the maximum crisis_probability tweet as the trigger.
        """
        worst = results_df.loc[results_df["crisis_probability"].idxmax()]
        top5 = (results_df
                .nlargest(5, "crisis_probability")["text"]
                .tolist())

        return CrisisAlert(
            alert_id=str(uuid.uuid4())[:8],
            level=worst["alert_level"],
            crisis_probability=worst["crisis_probability"],
            bert_score=worst["bert_score"],
            lstm_score=worst["lstm_score"],
            lda_score=worst["lda_score"],
            trigger_text=worst["text"][:280],
            top_tweets=top5,
        )

    def save(self, alert: CrisisAlert, output_dir: str = "outputs/alerts/") -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(
            output_dir, f"alert_{alert.alert_id}_{alert.level}.json")
        with open(path, "w") as f:
            f.write(alert.to_json())
        return path

    @staticmethod
    def summary_table(alerts: List[CrisisAlert]) -> pd.DataFrame:
        rows = [a.to_dict() for a in alerts]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df[["alert_id", "level", "crisis_probability", "bert_score",
                   "lstm_score", "lda_score", "timestamp"]].sort_values(
            "crisis_probability", ascending=False
        ).reset_index(drop=True)
