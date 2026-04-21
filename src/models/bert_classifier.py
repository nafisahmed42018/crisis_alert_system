"""DistilBERT binary classifier — crisis (1) vs normal (0).

Train
-----
    from src.models.bert_classifier import BERTClassifier
    clf = BERTClassifier()
    clf.train("data/processed/tweets_clean.csv")
    clf.save("outputs/models/bert_v1")

Predict
-------
    clf = BERTClassifier.load("outputs/models/bert_v1")
    scores = clf.predict(["oil spill reported in gulf", "I love Mondays"])
    # [0.91, 0.08]  — float crisis probability per tweet
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 128
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class TweetDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_len: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class BERTClassifier:
    def __init__(self, model_name: str = MODEL_NAME, max_length: int = MAX_LENGTH):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer  = None
        self.model      = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data_path: str,
        text_col:  str = "text_clean",
        label_col: str = "label",
        test_size: float = 0.15,
        epochs:    int   = 3,
        batch_size:int   = 16,
        output_dir:str   = "outputs/models/bert_v1",
    ) -> dict:
        df = pd.read_csv(data_path)
        df = df.dropna(subset=[text_col, label_col])
        texts  = df[text_col].tolist()
        labels = df[label_col].astype(int).tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size,
            random_state=RANDOM_SEED, stratify=labels,
        )
        print(f"Train: {len(X_train)} | Val: {len(X_val)}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )

        # Class weights to handle 57/43 imbalance
        class_counts = np.bincount(labels)
        weights = torch.tensor(
            len(labels) / (2.0 * class_counts), dtype=torch.float
        )

        train_ds = TweetDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_ds   = TweetDataset(X_val,   y_val,   self.tokenizer, self.max_length)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=50,
            seed=RANDOM_SEED,
            report_to="none",
        )

        trainer = _WeightedTrainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=_compute_metrics,
            class_weights=weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()
        self.model = trainer.model

        # Final evaluation
        preds = self.predict([t for t in X_val])
        pred_labels = (np.array(preds) >= 0.5).astype(int)
        print("\n" + classification_report(y_val, pred_labels, target_names=["Normal","Crisis"]))

        self.save(output_dir)
        return {"val_size": len(X_val), "model_dir": output_dir}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """Return list of crisis probabilities (0.0–1.0) for each tweet."""
        self.model.eval()
        device = next(self.model.parameters()).device
        scores = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            scores.extend(probs.tolist())

        return scores

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BERTClassifier":
        obj = cls()
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model     = AutoModelForSequenceClassification.from_pretrained(path)
        obj.model.eval()
        return obj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = self._class_weights.to(logits.device) if self._class_weights is not None else None
        loss = torch.nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, preds, average="binary")}
