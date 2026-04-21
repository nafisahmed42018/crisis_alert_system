"""Weighted ensemble — combines BERT, LSTM, and LDA crisis scores.

Usage
-----
    from src.models.ensemble import CrisisEnsemble
    ensemble = CrisisEnsemble.load(
        bert_path = "outputs/models/bert_v1",
        lda_path  = "outputs/models/lda_v1",
        lstm_path = "outputs/models/lstm_v1",
    )
    results = ensemble.predict(texts, timestamps=None)
    # returns list of EnsembleResult dataclasses
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.models.bert_classifier import BERTClassifier
from src.models.lda_analyzer import LDAAnalyzer
from src.models.lstm_detector import LSTMDetector

# Ensemble weights (must sum to 1.0)
W_BERT = 0.40
W_LSTM = 0.20
W_LDA = 0.40

# Alert thresholds (full pipeline — all 3 models active with real timestamps)
THRESHOLD_CRITICAL = 0.85
THRESHOLD_HIGH = 0.70
THRESHOLD_MEDIUM = 0.50

# Demo thresholds — used when LSTM is neutral (no timestamps available)
# Max possible score when LSTM=0.5: 0.4*BERT + 0.2 + 0.2*LDA ≈ 0.64
THRESHOLD_CRITICAL_DEMO = 0.62
THRESHOLD_HIGH_DEMO = 0.55
THRESHOLD_MEDIUM_DEMO = 0.45


@dataclass
class EnsembleResult:
    text:              str
    bert_score:        float
    lstm_score:        float
    lda_score:         float
    crisis_probability: float
    alert_level:       str   # LOW / MEDIUM / HIGH / CRITICAL


def _alert_level(prob: float, demo_mode: bool = False) -> str:
    tc = THRESHOLD_CRITICAL_DEMO if demo_mode else THRESHOLD_CRITICAL
    th = THRESHOLD_HIGH_DEMO if demo_mode else THRESHOLD_HIGH
    tm = THRESHOLD_MEDIUM_DEMO if demo_mode else THRESHOLD_MEDIUM
    if prob > tc:
        return "CRITICAL"
    if prob > th:
        return "HIGH"
    if prob > tm:
        return "MEDIUM"
    return "LOW"


class CrisisEnsemble:
    def __init__(self, bert: BERTClassifier, lda: LDAAnalyzer, lstm: LSTMDetector):
        self.bert = bert
        self.lda = lda
        self.lstm = lstm

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        texts:      List[str],
        timestamps: Optional[pd.Series] = None,
        demo_mode:  bool = False,
    ) -> List[EnsembleResult]:
        """Score each tweet and return structured EnsembleResult objects.

        Parameters
        ----------
        texts:      Raw or cleaned tweet texts.
        timestamps: Optional pd.Series of datetime values aligned to texts.
                    If None or missing, LSTM score defaults to 0.5 (neutral).
        """
        bert_scores = self.bert.predict(texts)
        lda_scores = self.lda.predict(texts)

        if timestamps is not None and self.lstm._hour_scores is not None:
            lstm_scores = self.lstm.predict_tweets(timestamps)
        else:
            lstm_scores = [0.5] * len(texts)

        results = []
        for i, text in enumerate(texts):
            b = bert_scores[i]
            l = lstm_scores[i]
            d = lda_scores[i]
            prob = W_BERT * b + W_LSTM * l + W_LDA * d
            results.append(EnsembleResult(
                text=text,
                bert_score=round(b, 4),
                lstm_score=round(l, 4),
                lda_score=round(d, 4),
                crisis_probability=round(prob, 4),
                alert_level=_alert_level(prob, demo_mode=demo_mode),
            ))
        return results

    def predict_df(
        self,
        texts:      List[str],
        timestamps: Optional[pd.Series] = None,
        demo_mode:  bool = False,
    ) -> pd.DataFrame:
        """Convenience wrapper — returns results as a DataFrame."""
        results = self.predict(texts, timestamps, demo_mode=demo_mode)
        return pd.DataFrame([r.__dict__ for r in results])

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        bert_path: str = "outputs/models/bert_v1",
        lda_path:  str = "outputs/models/lda_v1",
        lstm_path: str = "outputs/models/lstm_v1",
    ) -> "CrisisEnsemble":
        print("Loading BERT...")
        bert = BERTClassifier.load(bert_path)
        print("Loading LDA...")
        lda = LDAAnalyzer.load(lda_path)
        print("Loading LSTM...")
        lstm = LSTMDetector.load(lstm_path)
        return cls(bert=bert, lda=lda, lstm=lstm)
