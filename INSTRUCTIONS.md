# Running the Crisis Alert System — Step-by-Step

## Prerequisites

- Python 3.13 (`py -3.13` on Windows)
- Node.js 18+ and npm
- GPU optional but recommended (CUDA 12.8 used in development)

---

## One-Time Setup

### 1. Install Python dependencies

```bash
py -3.13 -m pip install -r requirements.txt
py -3.13 -m pip install fastapi uvicorn tweepy
```

### 2. Install Node.js dependencies (for Next.js app)

```bash
cd webapp
npm install
cd ..
```

### 3. Configure API keys

Edit `.env` in the project root:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_api_key
X_BEARER_TOKEN=your_twitter_bearer_token
```

The `X_BEARER_TOKEN` enables live tweet fetching from the X/Twitter API.
If omitted, the app falls back to demo mode (synthesised tweets from keywords).

---

## Train the Models (required before running either app)

Run this **once** to train BERT + LDA and generate all model files:

```bash
py -3.13 run_restore.py
```

This script does the following in order:

1. Loads `data/raw/disaster/tweets.csv` (11,370 Kaggle Disaster Tweets)
2. Cleans text and balances to 4,228 rows (50% crisis / 50% normal)
3. Trains **DistilBERT** for 3 epochs → saves to `outputs/models/bert_v1/`
4. Trains **LDA** (10 topics, Gensim) → saves to `outputs/models/lda_v1/`
5. Loads the pre-trained **LSTM** from `outputs/models/lstm_v1/`
6. Runs the full ensemble and saves scored CSVs to `data/processed/`

Expected training time: ~2 minutes on GPU, ~15 minutes on CPU.

> **Important:** Do not have the FastAPI server running while this script runs.
> The server memory-maps model files, which blocks overwriting them on Windows.
> Kill the server first (`Ctrl+C` in its terminal), retrain, then restart.

---

## Option A — Next.js Web App (recommended)

This is the full interactive dashboard with live X/Twitter tweet fetching.

### Step 1: Start the FastAPI backend

```bash
py -3.13 -m uvicorn src.api.main:app --reload --port 8000
```

The API loads models lazily on the first request (a few seconds the first time).

### Step 2: Start the Next.js frontend

In a **separate terminal**:

```bash
cd webapp
npm run dev
```

Open **http://localhost:3000** in your browser.

### Pages

| Page         | URL         | What it does                                                                   |
| ------------ | ----------- | ------------------------------------------------------------------------------ |
| Dashboard    | `/`         | KPI strip, alert distribution chart, recent alerts. Auto-refreshes every 10 s. |
| Analyzer     | `/analyzer` | Paste any tweet text → get BERT/LSTM/LDA scores + recommendation card          |
| Fetch Tweets | `/fetch`    | Enter keywords → fetches live tweets from X API → runs ensemble on all of them |
| Alerts       | `/alerts`   | Full alert history, filterable by level (CRITICAL / HIGH / MEDIUM / LOW)       |

---

## Option B — Streamlit Dashboard (legacy)

The original Streamlit dashboard is still available but does not include the
recommendation engine or live tweet fetching.

```bash
py -3.13 -m streamlit run src/dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

## What Are the Notebooks For?

The notebooks are **course deliverables / exploration tools**, not required to run the app.
You do **not** need to run them to use the system.

| Notebook                    | Purpose                                                               |
| --------------------------- | --------------------------------------------------------------------- |
| `01_exploration.ipynb`      | EDA — label distribution, tweet lengths, top keywords, word clouds    |
| `02_bert_sentiment.ipynb`   | BERT model development — training curves, confusion matrix, ROC curve |
| `03_lstm_anomaly.ipynb`     | LSTM anomaly detection on Climate Change Twitter time series          |
| `04_lda_topics.ipynb`       | LDA topic modelling — coherence scores, topic word clouds             |
| `05_ensemble_demo.ipynb`    | End-to-end pipeline demo — ensemble scoring, alert generation         |
| `06_final_submission.ipynb` | Full course submission notebook — all models + results in one place   |

To run a notebook, open it in VS Code or JupyterLab and run all cells.
The kernel must use Python 3.13.

---

## Do I Need to Run `src/models/` Files Directly?

No. The files under `src/models/` are library code (classes), not scripts.
They are imported by:

- `run_restore.py` — training pipeline
- `src/api/main.py` — FastAPI backend (loaded at runtime)
- The notebooks — for individual model exploration

You never need to run them directly.

---

## File Roles at a Glance

```
run_restore.py          ← Run once to train all models
src/api/main.py         ← FastAPI backend (start with uvicorn)
src/api/x_client.py     ← Tweepy wrapper for X/Twitter API
src/api/recommendation.py ← Crisis type classifier + action suggestions
src/models/
  bert_classifier.py    ← DistilBERT fine-tune + inference
  lstm_detector.py      ← LSTM anomaly detector
  lda_analyzer.py       ← Gensim LDA topic scorer
  ensemble.py           ← Weighted combiner (0.4×BERT + 0.4×LSTM + 0.2×LDA)
src/dashboard/app.py    ← Legacy Streamlit dashboard
webapp/                 ← Next.js app (npm run dev)
data/raw/disaster/      ← Training data (Kaggle Disaster Tweets)
data/processed/         ← Cleaned CSVs + model scores (auto-generated)
outputs/models/         ← Saved model weights (auto-generated by run_restore.py)
outputs/alerts/         ← Alert JSON files (auto-generated)
```

---

## Typical First-Run Sequence

```bash
# 1. Install dependencies (once)
py -3.13 -m pip install -r requirements.txt
py -3.13 -m pip install fastapi uvicorn tweepy
cd webapp && npm install && cd ..

# 2. Train models (once, or after deleting outputs/models/)
py -3.13 run_restore.py

# 3. Start backend
py -3.13 -m uvicorn src.api.main:app --reload --port 8000

# 4. Start frontend (new terminal)
cd webapp && npm run dev

# 5. Open http://localhost:3000
```
