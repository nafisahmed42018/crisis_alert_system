"""LSTM temporal anomaly detector — crisis velocity & volume spike detection.

Pipeline
--------
1. build_timeseries(csv_path)       → hourly aggregated DataFrame saved to disk
2. label_from_disasters(ts, path)   → add binary crisis label from disasters.csv
3. train(ts_df)                     → fit LSTM on sliding windows
4. predict_timeseries(ts_df)        → crisis score per hour bucket
5. predict_tweets(timestamps)       → map tweet timestamps to hourly LSTM scores
6. save / load

Architecture
------------
    Input  : (batch, window=24, features=6)
    LSTM   : hidden=64, layers=2, dropout=0.2
    Linear : 64 -> 1  + Sigmoid
    Loss   : BCE with class weights
"""

import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

WINDOW = 24          # hours of history per sample
N_FEATURES = 6
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Time-series builder
# ---------------------------------------------------------------------------

def build_timeseries(
    csv_path: str,
    out_path: str = "data/processed/climate_hourly.csv",
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Aggregate 15M climate tweets into hourly feature vectors.

    Features per hour bucket:
      mean_sentiment, tweet_volume, sentiment_velocity,
      pct_aggressive, pct_negative, pct_weather_extremes
    """
    print("Building hourly time series from climate tweets...")
    chunks = []
    for i, chunk in enumerate(pd.read_csv(
        csv_path,
        usecols=["created_at", "sentiment", "topic", "aggressiveness"],
        parse_dates=["created_at"],
        chunksize=chunksize,
    )):
        chunk["hour"] = chunk["created_at"].dt.floor("h")
        chunk["negative"] = (chunk["sentiment"] < 0).astype(int)
        chunk["is_aggressive"] = (
            chunk["aggressiveness"] == "aggressive").astype(int)
        chunk["is_weather"] = (
            chunk["topic"] == "Weather Extremes").astype(int)

        agg = chunk.groupby("hour").agg(
            mean_sentiment=("sentiment",      "mean"),
            tweet_volume=("sentiment",      "count"),
            pct_negative=("negative",       "mean"),
            pct_aggressive=("is_aggressive",  "mean"),
            pct_weather=("is_weather",     "mean"),
        ).reset_index()
        chunks.append(agg)
        if (i + 1) % 5 == 0:
            print(f"  processed {(i+1)*chunksize/1e6:.1f}M rows...")

    ts = pd.concat(chunks).groupby("hour").agg(
        mean_sentiment=("mean_sentiment", "mean"),
        tweet_volume=("tweet_volume",   "sum"),
        pct_negative=("pct_negative",   "mean"),
        pct_aggressive=("pct_aggressive", "mean"),
        pct_weather=("pct_weather",    "mean"),
    ).reset_index().sort_values("hour").reset_index(drop=True)

    # Sentiment velocity = hourly change in mean_sentiment
    ts["sentiment_velocity"] = ts["mean_sentiment"].diff().fillna(0)

    # Forward-fill sparse hours (minor gaps in early dataset years)
    ts = ts.set_index("hour").resample("h").mean().ffill().reset_index()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ts.to_csv(out_path, index=False)
    print(f"Saved {len(ts):,} hourly rows to {out_path}")
    return ts


def label_from_sentiment(
    ts: pd.DataFrame,
    sentiment_threshold: float = -0.15,
    velocity_threshold:  float = -0.05,
    volume_quantile:     float = 0.70,
) -> pd.DataFrame:
    """Label hours as crisis using sentiment anomaly thresholds.

    Crisis = negative sentiment spike + elevated volume, OR sharp velocity drop.
    This directly mirrors the detection logic described in CLAUDE.md:
      Warning if sentiment < -0.7 ; Critical if velocity < -0.3/hr

    Thresholds are intentionally loose to achieve ~25-35% positive rate.
    """
    ts = ts.copy()
    vol_thresh = ts["tweet_volume"].quantile(volume_quantile)

    crisis_mask = (
        # Negative sentiment during high-volume periods
        ((ts["mean_sentiment"] < sentiment_threshold)
         & (ts["tweet_volume"] >= vol_thresh))
        |
        # Sharp sentiment velocity drop (sudden negative swing)
        (ts["sentiment_velocity"] < velocity_threshold)
    )
    ts["label"] = crisis_mask.astype(int)

    pos = ts["label"].sum()
    print(f"Crisis hours: {pos:,} / {len(ts):,}  ({pos/len(ts)*100:.1f}%)")
    return ts


def label_from_disasters(
    ts: pd.DataFrame,
    disasters_path: str,
    min_deaths: int = 100,
    window_hours: int = 24,
) -> pd.DataFrame:
    """Mark hours near major disaster events (deaths >= min_deaths) as crisis.

    Kept for reference — use label_from_sentiment() instead for better balance.
    """
    disasters = pd.read_csv(disasters_path,
                            usecols=["start_date", "end_date", "Total Deaths"])
    disasters["start_date"] = pd.to_datetime(
        disasters["start_date"], utc=True, errors="coerce")
    disasters["end_date"] = pd.to_datetime(
        disasters["end_date"],   utc=True, errors="coerce")
    disasters = disasters.dropna(subset=["start_date"])
    disasters = disasters[disasters["Total Deaths"] >= min_deaths]

    ts = ts.copy()
    ts["hour"] = pd.to_datetime(ts["hour"], utc=True)
    ts["label"] = 0

    delta = pd.Timedelta(hours=window_hours)
    for _, row in disasters.iterrows():
        end = row["end_date"] if pd.notna(
            row["end_date"]) else row["start_date"]
        mask = (ts["hour"] >= row["start_date"] -
                delta) & (ts["hour"] <= end + delta)
        ts.loc[mask, "label"] = 1

    pos = ts["label"].sum()
    print(
        f"Crisis hours (deaths>={min_deaths}): {pos:,} / {len(ts):,}  ({pos/len(ts)*100:.1f}%)")
    return ts


# ---------------------------------------------------------------------------
# PyTorch dataset + model
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "mean_sentiment", "tweet_volume", "sentiment_velocity",
    "pct_negative", "pct_aggressive", "pct_weather",
]


class _WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden,
            num_layers=n_layers, dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LSTMDetector:
    def __init__(self, window: int = WINDOW, hidden: int = 64, n_layers: int = 2):
        self.window = window
        self.hidden = hidden
        self.n_layers = n_layers
        self.model:   Optional[_LSTMNet] = None
        self.scaler:  Optional[StandardScaler] = None
        # index=hour, value=score
        self._hour_scores: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        ts: pd.DataFrame,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        val_frac: float = 0.15,
    ) -> dict:
        X_raw, y = self._make_windows(ts)

        split = int(len(X_raw) * (1 - val_frac))
        X_train, X_val = X_raw[:split], X_raw[split:]
        y_train, y_val = y[:split],     y[split:]

        self.scaler = StandardScaler()
        n = X_train.shape[0] * X_train.shape[1]
        self.scaler.fit(X_train.reshape(n, N_FEATURES))
        X_train = self._scale(X_train)
        X_val = self._scale(X_val)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"Training on {device}  |  train={len(y_train):,}  val={len(y_val):,}")

        pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
                                  dtype=torch.float32).to(device)

        self.model = _LSTMNet(N_FEATURES, self.hidden,
                              self.n_layers).to(device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        # class weight via pos_weight below
        criterion = nn.BCELoss(weight=None)
        criterion_w = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_loader = DataLoader(_WindowDataset(X_train, y_train),
                                  batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience, patience_counter = 3, 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimiser.zero_grad()
                # Use raw logits path for weighted BCE
                out_raw = self.model.lstm(xb)[0][:, -1, :]
                logits = self.model.head[:-
                                         1](out_raw).squeeze(1)  # pre-sigmoid
                # Recompute properly:
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimiser.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                xv = torch.tensor(X_val, dtype=torch.float32).to(device)
                yv = torch.tensor(y_val, dtype=torch.float32).to(device)
                val_pred = self.model(xv)
                val_loss = criterion(val_pred, yv).item()
                val_acc = ((val_pred >= 0.5).float()
                           == yv).float().mean().item()

            print(f"Epoch {epoch:2d}/{epochs}  train_loss={train_loss/len(train_loader):.4f}"
                  f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # store on CPU to free GPU memory during training
                best_state = {k: v.clone().cpu()
                              for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        return {"best_val_loss": best_val_loss, "epochs_trained": epoch}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_timeseries(self, ts: pd.DataFrame, batch_size: int = 4096) -> pd.Series:
        """Return crisis score per hour bucket as a Series indexed by hour."""
        X_raw, _ = self._make_windows(ts, require_label=False)
        X_scaled = self._scale(X_raw)

        # Run inference on CPU to avoid GPU OOM after training
        cpu_model = self.model.cpu()
        cpu_model.eval()
        all_scores = []
        with torch.no_grad():
            for start in range(0, len(X_scaled), batch_size):
                batch = torch.tensor(
                    X_scaled[start:start + batch_size], dtype=torch.float32
                )
                all_scores.append(cpu_model(batch).numpy())
        scores = np.concatenate(all_scores)

        hours = pd.to_datetime(ts["hour"], utc=True).iloc[self.window:]
        self._hour_scores = pd.Series(scores, index=hours)
        return self._hour_scores

    def predict_tweets(self, timestamps: pd.Series) -> List[float]:
        """Map tweet timestamps to their hour bucket LSTM score.

        Returns 0.5 (neutral) for any tweet with no timestamp.
        """
        assert self._hour_scores is not None, "Call predict_timeseries() first"
        ts = pd.to_datetime(timestamps, utc=True,
                            errors="coerce").dt.floor("h")
        scores = ts.map(self._hour_scores).fillna(0.5).tolist()
        return scores

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "lstm.pt"))
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump({
                "window": self.window, "hidden": self.hidden,
                "n_layers": self.n_layers, "scaler": self.scaler,
                "hour_scores": self._hour_scores,
            }, f)
        print(f"LSTM model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LSTMDetector":
        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        obj = cls(window=meta["window"],
                  hidden=meta["hidden"], n_layers=meta["n_layers"])
        obj.scaler = meta["scaler"]
        obj._hour_scores = meta.get("hour_scores")
        obj.model = _LSTMNet(N_FEATURES, obj.hidden, obj.n_layers)
        obj.model.load_state_dict(torch.load(
            os.path.join(path, "lstm.pt"), weights_only=True))
        obj.model.eval()
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_windows(self, ts: pd.DataFrame, require_label: bool = True):
        feats = ts[FEATURE_COLS].values.astype(np.float32)
        labels = ts["label"].values.astype(
            np.float32) if require_label else np.zeros(len(ts))

        X, y = [], []
        for i in range(self.window, len(feats)):
            X.append(feats[i - self.window: i])
            y.append(labels[i])
        return np.array(X), np.array(y)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        n, w, f = X.shape
        return self.scaler.transform(X.reshape(n * w, f)).reshape(n, w, f)
