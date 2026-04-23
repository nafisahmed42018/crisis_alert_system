"""Train BERT + LDA on Kaggle Disaster Tweets, then run ensemble.

Dataset: data/raw/disaster/tweets.csv  (vstepanenko/disaster-tweets mirror)
Columns: id, keyword, location, text, target
Labels:  1=real disaster (crisis), 0=not disaster (normal)
Balancing: 2,114 per class (matches minority class size)

Run:
    py -3.13 run_restore.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import re
from sklearn.metrics import roc_auc_score, accuracy_score

# ---------------------------------------------------------------------------
# 1. Load & prepare
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1 -- Prepare Disaster Tweets dataset")
print("=" * 60)

RAW = "data/raw/disaster/tweets.csv"
df  = pd.read_csv(RAW)
print(f"Loaded {len(df):,} rows")
print(df["target"].value_counts().sort_index().to_string())

df = df.rename(columns={"target": "label"})

def _clean(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+",  "", text)
    text = re.sub(r"@\w+",     "", text)
    text = re.sub(r"#(\w+)",   r"\1", text)
    text = re.sub(r"[^\w\s]",  " ", text)
    text = re.sub(r"\s+",      " ", text)
    return text.lower().strip()

df["text_clean"] = df["text"].apply(_clean)
df = df[["id", "text_clean", "label"]]
df = df[df["text_clean"].str.len() > 5].reset_index(drop=True)

# Balance: use full minority class size
crisis = df[df["label"] == 1]
normal = df[df["label"] == 0]
n = min(len(crisis), len(normal))
df_bal = pd.concat([
    crisis.sample(n, random_state=42),
    normal.sample(n, random_state=42),
]).sample(frac=1, random_state=42).reset_index(drop=True)

df_bal.to_csv("./data/processed/tweets_clean.csv", index=False)
print(f"\nBalanced: {len(df_bal):,} rows  ({df_bal['label'].mean():.1%} crisis)")
print("Saved -> data/processed/tweets_clean.csv")

# ---------------------------------------------------------------------------
# 2. Train BERT
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 2 -- Train BERT")
print("=" * 60)

from src.models.bert_classifier import BERTClassifier
clf = BERTClassifier()
clf.train(
    data_path  = "data/processed/tweets_clean.csv",
    text_col   = "text_clean",
    label_col  = "label",
    output_dir = "outputs/models/bert_v1",
    epochs     = 3,
    batch_size = 32,
)

bert_scores = clf.predict(df_bal["text_clean"].tolist())
df_bal["bert_score"] = bert_scores
df_bal.to_csv("data/processed/tweets_bert_scores.csv", index=False)
print("Saved -> data/processed/tweets_bert_scores.csv")

roc = roc_auc_score(df_bal["label"], df_bal["bert_score"])
acc = accuracy_score(df_bal["label"], (df_bal["bert_score"] >= 0.5).astype(int))
print(f"BERT  ROC-AUC: {roc:.3f}  Accuracy: {acc:.1%}")

# ---------------------------------------------------------------------------
# 3. Train LDA
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 3 -- Train LDA")
print("=" * 60)

from src.models.lda_analyzer import LDAAnalyzer
lda = LDAAnalyzer()
lda.fit(
    texts  = df_bal["text_clean"].tolist(),
    labels = df_bal["label"].tolist(),
)
lda.save("outputs/models/lda_v1")

lda_scores = lda.predict(df_bal["text_clean"].tolist())
df_bal["lda_score"] = lda_scores
df_bal.to_csv("data/processed/tweets_lda_scores.csv", index=False)
print("Saved -> data/processed/tweets_lda_scores.csv")

roc_lda = roc_auc_score(df_bal["label"], df_bal["lda_score"])
print(f"LDA   ROC-AUC: {roc_lda:.3f}")

# ---------------------------------------------------------------------------
# 4. Ensemble scoring
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("STEP 4 -- Ensemble scoring")
print("=" * 60)

from src.models.ensemble     import CrisisEnsemble
from src.alerts.alert_engine import AlertEngine
import os

ensemble = CrisisEnsemble.load(
    bert_path="outputs/models/bert_v1",
    lda_path ="outputs/models/lda_v1",
    lstm_path="outputs/models/lstm_v1",
)

result = ensemble.predict_df(df_bal["text_clean"].tolist(), demo_mode=True)
result["true_label"] = df_bal["label"].values
result.to_csv("data/processed/tweets_ensemble_scores.csv", index=False)
print("Saved -> data/processed/tweets_ensemble_scores.csv")

engine = AlertEngine(min_level="HIGH")
alerts = engine.process(result)
os.makedirs("outputs/alerts", exist_ok=True)
for a in alerts[:5]:
    with open(f"outputs/alerts/{a.alert_id}.json", "w") as f:
        f.write(a.to_json())
print(f"{len(alerts)} HIGH+ alerts -- top 5 saved to outputs/alerts/")

roc_ens = roc_auc_score(result["true_label"], result["crisis_probability"])
acc_ens = accuracy_score(result["true_label"], (result["crisis_probability"] >= 0.5).astype(int))
print(f"\nEnsemble ROC-AUC: {roc_ens:.3f}  Accuracy: {acc_ens:.1%}")
print("\nDone.")
