# Crisis Alert System — AML-3203

**Course**: AML-3203 Business Applications of ML in Social Media  
**Topic**: Social Media Crisis Detection using BERT + LSTM + LDA Ensemble

---

## What This System Does

Detects crises in social media using an ensemble of three complementary ML models:

| Model | Architecture | Dataset | Weight |
|---|---|---|---|
| **BERT** | DistilBERT fine-tuned | Kaggle Disaster Tweets (7.6K) | 40% |
| **LSTM** | 2-layer LSTM, hidden=64 | Climate Change Twitter (15.8M) | 40% |
| **LDA** | Gensim LDA, k=5 topics | Kaggle Disaster Tweets (7.6K) | 20% |

```
crisis_probability = 0.40 × BERT + 0.40 × LSTM + 0.20 × LDA
```

---

## Project Structure

```
sentinment-analysis/
├── data/
│   ├── raw/               # Original Kaggle CSVs
│   ├── processed/         # Cleaned & scored datasets
│   └── samples/           # 100-row stratified sample
├── notebooks/
│   ├── 01_exploration.ipynb      # EDA
│   ├── 02_bert_sentiment.ipynb   # BERT training & eval
│   ├── 03_lstm_anomaly.ipynb     # LSTM temporal detection
│   ├── 04_lda_topics.ipynb       # LDA topic modelling
│   ├── 05_ensemble_demo.ipynb    # Ensemble + alerts demo
│   └── 06_final_submission.ipynb # Full end-to-end walkthrough
├── src/
│   ├── data/
│   │   ├── schema.py      # DatasetConfig registry
│   │   ├── loader.py      # load_dataset()
│   │   └── cleaner.py     # clean_dataset(), make_sample()
│   ├── models/
│   │   ├── bert_classifier.py   # BERTClassifier
│   │   ├── lstm_detector.py     # LSTMDetector
│   │   ├── lda_analyzer.py      # LDAAnalyzer
│   │   └── ensemble.py          # CrisisEnsemble
│   ├── alerts/
│   │   ├── alert_schema.py      # CrisisAlert dataclass
│   │   └── alert_engine.py      # AlertEngine
│   └── dashboard/
│       ├── app.py               # Streamlit dashboard
│       └── components.py        # Plotly chart helpers
├── outputs/
│   ├── models/            # Saved model checkpoints
│   ├── charts/            # Training & evaluation charts
│   └── alerts/            # Sample alert JSON files
├── tests/
│   └── test_cleaner.py    # Unit tests
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note**: BERT and LSTM training require PyTorch with CUDA. Install CUDA-enabled PyTorch separately:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
> ```

### 2. Download datasets

**Kaggle Disaster Tweets** → place at `data/raw/train.csv`  
Source: https://www.kaggle.com/competitions/nlp-getting-started

**Climate Change Twitter** → place at `data/raw/twitter_sentiment_data.csv`  
Source: https://www.kaggle.com/datasets/deffro/the-climate-change-twitter-dataset

### 3. Run notebooks in order

```
01_exploration.ipynb      ← EDA (no model training)
02_bert_sentiment.ipynb   ← Fine-tune DistilBERT (~1 min on GPU)
03_lstm_anomaly.ipynb     ← Inspect LSTM scores
04_lda_topics.ipynb       ← LDA topic inspection
05_ensemble_demo.ipynb    ← Ensemble + alert demo
06_final_submission.ipynb ← Full pipeline walkthrough
```

### 4. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## Results

| Model | ROC-AUC | Notes |
|---|---|---|
| BERT | ~0.910 | DistilBERT, fine-tuned 3 epochs |
| LSTM | 0.856 | Trained on Climate Twitter time series |
| LDA | 0.605 | k=5 topics, crisis topics [1,2,3] |
| **Ensemble** | **0.926** | Weighted: 40% BERT + 40% LSTM + 20% LDA |

**Ensemble Accuracy**: 88% on Disaster Tweets test set  
**Alerts on full dataset**: 2,645 HIGH+CRITICAL alerts out of 7,613 tweets

### Alert Levels

| Level | Threshold | Action |
|---|---|---|
| CRITICAL | > 85% | Immediate escalation |
| HIGH | > 70% | Urgent review |
| MEDIUM | > 50% | Monitor closely |
| LOW | ≤ 50% | No action required |

*Demo mode uses lower thresholds (0.62/0.55/0.45) since LSTM defaults to 0.5 when no timestamps are available.*

---

## Architecture Details

### BERT (40%)
- Base: `distilbert-base-uncased`
- Fine-tuned for binary classification (crisis vs normal)
- Input: raw tweet text (max 128 tokens)
- Output: crisis probability in [0, 1]

### LSTM (40%)
- Trained on 116,726 hourly windows from Climate Change Twitter (2006–2019)
- Labeling: `crisis = negative_sentiment + high_volume`
- Input: sliding 24-hour window of [mean_sentiment, tweet_volume, pct_negative, pct_aggressive, pct_weather]
- Output: hourly crisis probability; mapped to tweets via timestamp lookup

### LDA (20%)
- k=5 topics discovered via coherence-optimised Gensim LDA
- Crisis topics identified by correlation with disaster labels
- Output: fraction of token weight in crisis topics

---

## Tests

```bash
python -m pytest tests/ -v
```
