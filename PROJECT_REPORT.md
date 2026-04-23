# Crisis Alert System — Project Report

**Course:** AML-3203 Business Applications of ML in Social Media  
**Project:** Crisis Alert System (CAS)  
**Author:** Nafis Ahmed  
**Date:** 2026-04-22  
**Repository Branch:** `main`

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [System Overview](#2-system-overview)
3. [Tech Stack](#3-tech-stack)
4. [High-Level Architecture](#4-high-level-architecture)
5. [Data Flow Diagram](#5-data-flow-diagram)
6. [ML Pipeline Architecture](#6-ml-pipeline-architecture)
7. [Ensemble Scoring Logic](#7-ensemble-scoring-logic)
8. [API Architecture](#8-api-architecture)
9. [Frontend Architecture](#9-frontend-architecture)
10. [Alert & Notification Workflow](#10-alert--notification-workflow)
11. [Training Pipeline Workflow](#11-training-pipeline-workflow)
12. [Component Interaction Diagram](#12-component-interaction-diagram)
13. [Dataset Summary](#13-dataset-summary)
14. [Model Performance](#14-model-performance)
15. [Deployment & Setup](#15-deployment--setup)
16. [Key Design Decisions](#16-key-design-decisions)

---

## 1. Project Objective

The Crisis Alert System is a **real-time social media crisis detection and notification platform** built for organizations that need to monitor Twitter/X for emerging crises — natural disasters, corporate misconduct, health hazards, civil unrest, financial shocks, and regulatory events.

### Core Objectives

| # | Objective | How It's Fulfilled |
|---|-----------|-------------------|
| 1 | Detect crisis signals in social media text | DistilBERT binary classifier (fine-tuned on disaster tweets) |
| 2 | Identify temporal anomalies in sentiment | LSTM temporal detector trained on 15.8M climate tweets |
| 3 | Discover latent crisis topics without supervision | Gensim LDA topic model (10 topics) |
| 4 | Fuse multi-model signals into a single probability | Weighted ensemble (0.4 BERT + 0.4 LSTM + 0.2 LDA) |
| 5 | Categorize crisis type and suggest actions | Rule-based recommendation engine (6 crisis categories) |
| 6 | Alert stakeholders in real time | Slack webhook integration for HIGH/CRITICAL alerts |
| 7 | Provide an operational monitoring dashboard | Next.js web app with live KPI refresh |
| 8 | Enable ad-hoc tweet analysis and bulk fetch | REST API + X/Twitter API v2 integration |

---

## 2. System Overview

The system operates as two independent services that communicate over HTTP:

```
┌─────────────────────────────────────────────────────────┐
│                  CRISIS ALERT SYSTEM                     │
│                                                          │
│  ┌──────────────────────┐    ┌────────────────────────┐  │
│  │   Next.js Frontend   │◄──►│   FastAPI Backend      │  │
│  │   (React / TypeScript│    │   (Python / ML Models) │  │
│  │   Port 3000)         │    │   Port 8000)           │  │
│  └──────────────────────┘    └────────────────────────┘  │
│                                        │                  │
│                              ┌─────────┴──────────┐      │
│                              │  External Services  │      │
│                              │  - X/Twitter API    │      │
│                              │  - Slack Webhook    │      │
│                              └─────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### User Personas

- **Crisis Manager** — monitors the dashboard for emerging alerts, drills into specific tweets
- **Social Media Analyst** — uses the Fetch page to search keywords, reviews model score breakdowns
- **Executive** — receives Slack notifications with recommended actions and stakeholder routing
- **Data Scientist / Course Grader** — explores notebooks for reproducibility and methodology

---

## 3. Tech Stack

### Backend

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Language | Python | 3.13 | Core runtime |
| Web Framework | FastAPI | latest | REST API server |
| ASGI Server | Uvicorn | latest | Production/dev HTTP server |
| Deep Learning | PyTorch | latest | BERT model training & inference |
| Deep Learning | TensorFlow/Keras | latest | LSTM model |
| Transformers | HuggingFace Transformers | latest | DistilBERT tokenizer + model |
| Topic Modeling | Gensim | latest | LDA topic model |
| NLP Utilities | scikit-learn | latest | TF-IDF vectorizer, metrics |
| Twitter Integration | Tweepy | latest | X/Twitter API v2 client |
| Notifications | Slack SDK / Webhooks | latest | Slack incoming webhook |
| Scheduling / API | OpenAI | latest | AI-powered recommendations |
| Legacy Dashboard | Streamlit | latest | Plotly-based chart UI |
| Data Processing | pandas, numpy | latest | Data wrangling |
| Viz | Plotly | latest | Charts in Streamlit |

### Frontend

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Framework | Next.js | 16.1.7 | Full-stack React framework (App Router) |
| Runtime | React | 19.2 | Component rendering |
| Language | TypeScript | 5.9 | Static typing |
| Styling | Tailwind CSS | 4.2 | Utility-first CSS |
| UI Components | shadcn/ui | latest | Accessible component library |
| Charts | Recharts | latest | Alert distribution bar chart, gauges |
| Bundler | Turbopack | built-in | Fast dev builds (Next.js 16) |
| HTTP Client | Native `fetch` | — | API calls to FastAPI backend |

### Infrastructure & Data

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Model Storage | PyTorch `.pt` / joblib `.pkl` | Serialized trained models |
| Alert History | In-memory Python list → JSON | Max 500 alerts, flushed to disk |
| Training Data | CSV (Kaggle download) | Disaster Tweets, Climate Twitter |
| Secrets | `.env` file | API keys for X, Slack, OpenAI, Kaggle |

---

## 4. High-Level Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                    CRISIS ALERT SYSTEM — ARCHITECTURE               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────────────────────────────────────┐                ║
║  │              DATA INGESTION LAYER                │                ║
║  │                                                  │                ║
║  │  ┌──────────────┐        ┌────────────────────┐  │                ║
║  │  │  X/Twitter   │        │   Manual Input     │  │                ║
║  │  │  API v2      │        │   (Paste Tweet)    │  │                ║
║  │  │  (Tweepy)    │        │                    │  │                ║
║  │  └──────┬───────┘        └─────────┬──────────┘  │                ║
║  └─────────┼────────────────────────┼──────────────┘                ║
║            │                        │                                ║
║            ▼                        ▼                                ║
║  ┌─────────────────────────────────────────────────┐                ║
║  │           TEXT PREPROCESSING LAYER              │                ║
║  │                                                  │                ║
║  │   Remove URLs → Remove Mentions → Lowercase      │                ║
║  │   Remove Special Chars → Tokenize               │                ║
║  └───────────────────────┬──────────────────────────┘                ║
║                          │                                           ║
║           ┌──────────────┼──────────────┐                            ║
║           ▼              ▼              ▼                            ║
║  ┌───────────────┐ ┌───────────┐ ┌───────────────┐                  ║
║  │  BERT MODEL   │ │   LSTM    │ │  LDA MODEL    │                  ║
║  │               │ │  MODEL    │ │               │                  ║
║  │ DistilBERT    │ │           │ │  10 Topics    │                  ║
║  │ fine-tuned    │ │ 2-layer   │ │  Gensim       │                  ║
║  │ on disaster   │ │ hidden=64 │ │  10 passes    │                  ║
║  │ tweets        │ │ temporal  │ │               │                  ║
║  │               │ │ anomaly   │ │               │                  ║
║  │ Score: 0–1    │ │ Score:0–1 │ │ Score: 0–1    │                  ║
║  └───────┬───────┘ └─────┬─────┘ └──────┬────────┘                  ║
║          │               │              │                            ║
║          └───────────────┼──────────────┘                            ║
║                          ▼                                           ║
║  ┌─────────────────────────────────────────────────┐                ║
║  │               ENSEMBLE LAYER                    │                ║
║  │                                                  │                ║
║  │    P = 0.40×BERT + 0.40×LSTM + 0.20×LDA         │                ║
║  │                                                  │                ║
║  └───────────────────────┬──────────────────────────┘                ║
║                          │                                           ║
║           ┌──────────────┴──────────────┐                            ║
║           ▼                             ▼                            ║
║  ┌────────────────────┐    ┌─────────────────────────┐               ║
║  │  ALERT ENGINE      │    │  RECOMMENDATION ENGINE  │               ║
║  │                    │    │                         │               ║
║  │ CRITICAL  > 0.80   │    │  Classify crisis type   │               ║
║  │ HIGH      > 0.65   │    │  Map to actions         │               ║
║  │ MEDIUM    > 0.50   │    │  Route to stakeholders  │               ║
║  │ LOW       ≤ 0.50   │    │                         │               ║
║  └─────────┬──────────┘    └─────────────────────────┘               ║
║            │                                                         ║
║    ┌───────┴────────────────────────┐                                ║
║    ▼                                ▼                                ║
║  ┌──────────────────┐    ┌───────────────────────────┐               ║
║  │  SLACK NOTIFIER  │    │   NEXT.JS DASHBOARD       │               ║
║  │  (HIGH/CRITICAL) │    │   - Live KPI strip        │               ║
║  │                  │    │   - Alert history         │               ║
║  │  Webhook POST    │    │   - Score breakdown       │               ║
║  └──────────────────┘    │   - Recommendations       │               ║
║                          └───────────────────────────┘               ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 5. Data Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                         DATA FLOW                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  USER (Browser)                                                      ║
║       │                                                              ║
║       │  1. POST /api/analyze { text: "..." }                       ║
║       │     OR                                                       ║
║       │  1. POST /api/fetch-and-analyze { keywords: [...] }         ║
║       ▼                                                              ║
║  ┌─────────────────────────────────────────────────┐                ║
║  │              FastAPI (main.py)                   │                ║
║  │                                                  │                ║
║  │  • CORS headers                                  │                ║
║  │  • Lazy-load models on first request             │                ║
║  │  • Input validation (pydantic)                   │                ║
║  └───────┬────────────────────────┬─────────────────┘                ║
║          │                        │                                  ║
║          │ (fetch-and-analyze)    │ (analyze single)                ║
║          ▼                        │                                  ║
║  ┌───────────────────┐            │                                  ║
║  │  x_client.py      │            │                                  ║
║  │  Tweepy v2 search │            │                                  ║
║  │  → RawTweet list  │            │                                  ║
║  └────────┬──────────┘            │                                  ║
║           │                       │                                  ║
║           └───────────┬───────────┘                                  ║
║                       ▼                                              ║
║  ┌─────────────────────────────────────────────────┐                ║
║  │           cleaner.py (text preprocessing)        │                ║
║  │                                                  │                ║
║  │  raw text → clean text (URL/mention removal)     │                ║
║  └───────────────────────┬──────────────────────────┘                ║
║                          │                                           ║
║          ┌───────────────┼───────────────────┐                       ║
║          ▼               ▼                   ▼                       ║
║  ┌──────────────┐ ┌─────────────┐ ┌──────────────────┐              ║
║  │bert_         │ │lstm_        │ │lda_              │              ║
║  │classifier.py │ │detector.py  │ │analyzer.py       │              ║
║  │              │ │             │ │                  │              ║
║  │bert_score    │ │lstm_score   │ │lda_score         │              ║
║  │∈ [0, 1]      │ │∈ [0, 1]     │ │∈ [0, 1]          │              ║
║  └──────┬───────┘ └──────┬──────┘ └────────┬─────────┘              ║
║         │                │                 │                         ║
║         └────────────────┼─────────────────┘                         ║
║                          ▼                                           ║
║  ┌─────────────────────────────────────────────────┐                ║
║  │               ensemble.py                       │                ║
║  │                                                  │                ║
║  │  crisis_prob = 0.4*bert + 0.4*lstm + 0.2*lda    │                ║
║  │  alert_level = threshold_lookup(crisis_prob)     │                ║
║  └───────────────────────┬──────────────────────────┘                ║
║                          │                                           ║
║          ┌───────────────┼───────────────┐                           ║
║          ▼               ▼               ▼                           ║
║  ┌──────────────┐ ┌────────────┐ ┌──────────────────┐               ║
║  │alert_engine  │ │recommenda  │ │ slack_notifier   │               ║
║  │.py           │ │tion.py     │ │.py               │               ║
║  │              │ │            │ │                  │               ║
║  │CrisisAlert   │ │type + acts │ │POST webhook      │               ║
║  │stored in     │ │+ stakeholders (HIGH/CRITICAL)   │               ║
║  │memory list   │ │            │ │                  │               ║
║  └──────┬───────┘ └──────┬─────┘ └──────────────────┘               ║
║         └────────────────┘                                           ║
║                  │                                                   ║
║                  ▼                                                   ║
║  ┌─────────────────────────────────────────────────┐                ║
║  │              ScoreResult JSON Response           │                ║
║  │  {                                               │                ║
║  │    tweet_text, bert_score, lstm_score,           │                ║
║  │    lda_score, crisis_probability,                │                ║
║  │    alert_level, recommendations,                 │                ║
║  │    crisis_type, stakeholders, actions            │                ║
║  │  }                                               │                ║
║  └──────────────────────┬──────────────────────────┘                ║
║                         │                                            ║
║                         ▼                                            ║
║  USER (Browser) receives JSON → React updates UI                     ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 6. ML Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                    ML PIPELINE ARCHITECTURE                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  TRAINING PHASE (run_restore.py)                                     ║
║  ─────────────────────────────                                       ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │                   BERT TRAINING PATH                         │   ║
║  │                                                              │   ║
║  │  Kaggle Disaster Tweets (11,370 rows)                       │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  Balance classes (2,114 crisis + 2,114 normal = 4,228)      │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  DistilBERT tokenizer (max_len=128, batch=32)               │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  Fine-tune 3 epochs → outputs/models/bert_v1/               │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  BERT ROC-AUC: 0.910                                        │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │                   LSTM TRAINING PATH                         │   ║
║  │                                                              │   ║
║  │  Climate Change Twitter Dataset (15.8M tweets)              │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  Hourly aggregation → 116,700 time windows                  │   ║
║  │  Features: avg_sentiment, tweet_volume, topic_coherence     │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  2-layer LSTM (hidden_size=64, dropout=0.2)                 │   ║
║  │  Sequence length = 24 hours                                 │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  Save → outputs/models/lstm_v1/                             │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  LSTM ROC-AUC: 0.856                                        │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │                   LDA TRAINING PATH                          │   ║
║  │                                                              │   ║
║  │  Same 4,228 balanced disaster tweets                        │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  Gensim Dictionary + Corpus (Bag-of-Words)                  │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  LDA(num_topics=10, passes=10, random_state=42)             │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  Map crisis-correlated topics → probability                 │   ║
║  │  Save → outputs/models/lda_v1/                              │   ║
║  │           │                                                  │   ║
║  │           ▼                                                  │   ║
║  │  LDA ROC-AUC: 0.605                                         │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
║  INFERENCE PHASE (FastAPI /analyze)                                  ║
║  ──────────────────────────────────                                  ║
║                                                                      ║
║  Input tweet text                                                    ║
║       │                                                              ║
║       ├──────────────────────────────────────────┐                  ║
║       │                                          │                  ║
║       ▼                                          ▼                  ║
║  BERT (loaded once on                       LDA (loaded once        ║
║  first request)                             on first request)       ║
║  AutoModelForSequence                       via gensim.load()       ║
║  Classification.from_pretrained             Dictionary.load()       ║
║       │                                          │                  ║
║       │              LSTM                        │                  ║
║       │         (loaded once,                    │                  ║
║       │          returns 0.5                     │                  ║
║       │          if no                           │                  ║
║       │          timestamps)                     │                  ║
║       │                │                         │                  ║
║       └────────────────┼─────────────────────────┘                  ║
║                        ▼                                             ║
║               Weighted sum → crisis_probability                      ║
║               Threshold → alert_level                                ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 7. Ensemble Scoring Logic

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE SCORING FORMULA                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   crisis_probability = 0.40 × bert_score                           │
│                      + 0.40 × lstm_score                           │
│                      + 0.20 × lda_score                            │
│                                                                     │
│   bert_score  ─ real-time semantic intent (supervised)             │
│   lstm_score  ─ temporal pattern deviation (time-series)           │
│   lda_score   ─ topic coherence with crisis themes (unsupervised)  │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                    ALERT LEVEL THRESHOLDS                           │
├───────────────┬───────────────────────┬───────────────────────────-┤
│  Alert Level  │  Production Threshold │  Notification Actions       │
├───────────────┼───────────────────────┼───────────────────────────-┤
│  🔴 CRITICAL  │     P > 0.80          │  Slack + SMS + Dashboard    │
│               │                       │  + Executive Brief          │
├───────────────┼───────────────────────┼────────────────────────────┤
│  🟠 HIGH      │     P > 0.65          │  Slack + Email + Dashboard  │
├───────────────┼───────────────────────┼────────────────────────────┤
│  🟡 MEDIUM    │     P > 0.50          │  Email Digest + Dashboard   │
├───────────────┼───────────────────────┼────────────────────────────┤
│  🟢 LOW       │     P ≤ 0.50          │  Dashboard only             │
├───────────────┴───────────────────────┴────────────────────────────┤
│                                                                     │
│  Demo Mode Thresholds (LSTM unavailable, max P ≈ 0.60):            │
│  CRITICAL > 0.62 | HIGH > 0.55 | MEDIUM > 0.45                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. API Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                  FASTAPI BACKEND — src/api/main.py                  ║
║                        http://localhost:8000                        ║
╠═══════════════════════╦══════════════════════════════════════════════╣
║   Endpoint            ║  Description                                ║
╠═══════════════════════╬══════════════════════════════════════════════╣
║  GET  /health         ║  Health check: {"status":"ok","timestamp"…} ║
╠═══════════════════════╬══════════════════════════════════════════════╣
║  POST /api/analyze    ║  Analyze a single tweet text                ║
║                       ║  Body: {text: str, demo_mode: bool}         ║
║                       ║  Response: ScoreResult                      ║
╠═══════════════════════╬══════════════════════════════════════════════╣
║  POST /api/           ║  Fetch tweets from X API by keywords,       ║
║  fetch-and-analyze    ║  then analyze each one                      ║
║                       ║  Body: {keywords: str[], max_results: int,  ║
║                       ║         demo_mode: bool}                    ║
║                       ║  Response: FetchResponse                    ║
╠═══════════════════════╬══════════════════════════════════════════════╣
║  GET  /api/alerts     ║  Get alert history with optional filter     ║
║                       ║  Query: ?level=HIGH&limit=50                ║
║                       ║  Response: ScoreResult[]                    ║
╠═══════════════════════╬══════════════════════════════════════════════╣
║  DELETE /api/alerts   ║  Clear all stored alerts                    ║
║                       ║  Response: {cleared: true}                  ║
╚═══════════════════════╩══════════════════════════════════════════════╝

Response Shape — ScoreResult
┌────────────────────────────────────────────────────────────┐
│  {                                                         │
│    tweet_text:         string                              │
│    bert_score:         number (0–1)                        │
│    lstm_score:         number (0–1)                        │
│    lda_score:          number (0–1)                        │
│    crisis_probability: number (0–1)                        │
│    alert_level:        "LOW" | "MEDIUM" | "HIGH" |         │
│                        "CRITICAL"                          │
│    crisis_type:        string (6 categories)               │
│    recommendations: {                                      │
│      actions:      string[]                                │
│      stakeholders: string[]                                │
│      escalation:   string                                  │
│    }                                                       │
│    timestamp:          ISO8601 string                      │
│  }                                                         │
└────────────────────────────────────────────────────────────┘
```

---

## 9. Frontend Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                NEXT.JS FRONTEND — cas/                              ║
║                   http://localhost:3000                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  cas/app/ (Next.js App Router)                                       ║
║  ├── layout.tsx ─── Root layout: top nav, theme provider            ║
║  │                                                                   ║
║  ├── page.tsx ────── / DASHBOARD                                    ║
║  │   ┌──────────────────────────────────────────────────────┐       ║
║  │   │  KPI Strip: [Critical N] [High N] [Medium N] [Low N]│       ║
║  │   │  Alert Distribution Bar Chart (Recharts)             │       ║
║  │   │  Recent Alerts List (last 10)                        │       ║
║  │   │  Quick-Start Cards → /analyzer, /fetch               │       ║
║  │   │  Auto-refresh: every 10 seconds                      │       ║
║  │   └──────────────────────────────────────────────────────┘       ║
║  │                                                                   ║
║  ├── analyzer/page.tsx ── /analyzer SINGLE TWEET ANALYZER          ║
║  │   ┌──────────────────────────────────────────────────────┐       ║
║  │   │  Tweet Text Input (textarea + pre-loaded examples)   │       ║
║  │   │  [Analyze] button → POST /api/analyze                │       ║
║  │   │  ┌─────────────────────────────────────────────┐     │       ║
║  │   │  │ ScoreGauge (crisis probability dial)        │     │       ║
║  │   │  │ AlertBadge (colored level pill)             │     │       ║
║  │   │  │ ModelBars (BERT / LSTM / LDA score bars)    │     │       ║
║  │   │  │ RecommendationCard (type + actions + people)│     │       ║
║  │   │  └─────────────────────────────────────────────┘     │       ║
║  │   └──────────────────────────────────────────────────────┘       ║
║  │                                                                   ║
║  ├── fetch/page.tsx ── /fetch KEYWORD FETCH + BULK ANALYZE         ║
║  │   ┌──────────────────────────────────────────────────────┐       ║
║  │   │  Keyword input → POST /api/fetch-and-analyze         │       ║
║  │   │  Falls back to demo tweets if no X API token         │       ║
║  │   │  TweetCard grid (with TweetCardSkeleton loaders)     │       ║
║  │   └──────────────────────────────────────────────────────┘       ║
║  │                                                                   ║
║  └── alerts/page.tsx ── /alerts ALERT HISTORY                      ║
║      ┌──────────────────────────────────────────────────────┐       ║
║      │  Filter by alert level (ALL / CRITICAL / HIGH / …)  │       ║
║      │  Paginated list of stored CrisisAlerts               │       ║
║      │  GET /api/alerts?level=<filter>&limit=50             │       ║
║      └──────────────────────────────────────────────────────┘       ║
║                                                                      ║
║  cas/components/                                                     ║
║  ├── AlertBadge.tsx       ── Colored pill: CRITICAL/HIGH/…         ║
║  ├── ScoreGauge.tsx        ── Recharts radial gauge dial            ║
║  ├── ModelBars.tsx         ── Horizontal bars for 3 model scores   ║
║  ├── RecommendationCard.tsx── Crisis type + actions + stakeholders  ║
║  ├── TweetCard.tsx         ── Full tweet result card                ║
║  ├── TweetCardSkeleton.tsx  ── Animated loading skeleton            ║
║  ├── theme-provider.tsx    ── next-themes dark/light context        ║
║  └── ui/                  ── shadcn/ui primitives                  ║
║                                                                      ║
║  cas/lib/                                                            ║
║  ├── api.ts    ── HTTP client: analyzeTweet(), fetchAndAnalyze(),  ║
║  │               getAlerts(), clearAlerts()                         ║
║  ├── types.ts  ── TypeScript interfaces: ScoreResult, FetchResponse ║
║  └── utils.ts  ── Tailwind class merging, helpers                   ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 10. Alert & Notification Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│              ALERT & NOTIFICATION WORKFLOW                          │
└─────────────────────────────────────────────────────────────────────┘

 crisis_probability = P (from ensemble)
         │
         ▼
 ┌────────────────────────────────────────────────────────────┐
 │               ALERT ENGINE (alert_engine.py)               │
 │                                                            │
 │   P > 0.80 ──► CRITICAL alert created                     │
 │   P > 0.65 ──► HIGH alert created                         │
 │   P > 0.50 ──► MEDIUM alert created                       │
 │   P ≤ 0.50 ──► LOW alert created                          │
 │                                                            │
 │   CrisisAlert {                                            │
 │     id:           UUID                                     │
 │     level:        AlertLevel                               │
 │     probability:  float                                    │
 │     bert_score:   float                                    │
 │     lstm_score:   float                                    │
 │     lda_score:    float                                    │
 │     trigger_text: str                                      │
 │     timestamp:    datetime                                 │
 │     recommendations: {...}                                 │
 │   }                                                        │
 │                                                            │
 │   Stored in memory list (max 500 alerts → FIFO rotation)  │
 └─────────────────────────────┬──────────────────────────────┘
                               │
              ┌────────────────┼───────────────────┐
              │                │                   │
              ▼                ▼                   ▼
      ┌───────────────┐ ┌───────────────┐  ┌──────────────────┐
      │ LOW / MEDIUM  │ │    HIGH       │  │    CRITICAL      │
      │               │ │               │  │                  │
      │ Dashboard     │ │ Dashboard +   │  │ Dashboard +      │
      │ only          │ │ Slack webhook │  │ Slack + SMS +    │
      │               │ │               │  │ Executive Brief  │
      └───────────────┘ └───────┬───────┘  └───────┬──────────┘
                                │                  │
                                └────────┬──────────┘
                                         ▼
                               ┌──────────────────────┐
                               │  slack_notifier.py    │
                               │                      │
                               │  POST SLACK_WEBHOOK  │
                               │  Message includes:   │
                               │  - Alert level badge │
                               │  - Crisis probability│
                               │  - Tweet text        │
                               │  - Crisis type       │
                               │  - Recommended steps │
                               │  - Stakeholders      │
                               └──────────────────────┘
```

---

## 11. Training Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│              TRAINING PIPELINE (run_restore.py)                     │
└─────────────────────────────────────────────────────────────────────┘

 START
   │
   ▼
 ┌─────────────────────────────────────┐
 │  Step 1: Load Dataset               │
 │  data/raw/disaster/tweets.csv       │
 │  → 11,370 rows                      │
 │  Columns: id, keyword, location,    │
 │           text, target (0/1)        │
 └──────────────────┬──────────────────┘
                    ▼
 ┌─────────────────────────────────────┐
 │  Step 2: Text Cleaning              │
 │  cleaner.py                         │
 │  • Remove URLs (http://...)         │
 │  • Remove @mentions                 │
 │  • Remove HTML entities             │
 │  • Remove special chars             │
 │  • Lowercase                        │
 │  → tweets_clean.csv                 │
 └──────────────────┬──────────────────┘
                    ▼
 ┌─────────────────────────────────────┐
 │  Step 3: Class Balancing            │
 │  • 2,114 crisis samples             │
 │  • 2,114 non-crisis samples         │
 │  → 4,228 balanced rows              │
 └──────────────────┬──────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 ┌──────────────────┐    ┌──────────────────┐
 │  Step 4a: Train  │    │  Step 4b: Train  │
 │  BERT            │    │  LDA             │
 │                  │    │                  │
 │  tokenize(text,  │    │  Dictionary +    │
 │  max_len=128)    │    │  Corpus BOW      │
 │                  │    │                  │
 │  3 epochs        │    │  10 topics       │
 │  batch=32        │    │  10 passes       │
 │  lr=2e-5         │    │                  │
 │  AdamW           │    │  Identify crisis │
 │                  │    │  topic indices   │
 │  AUC: 0.910      │    │  AUC: 0.605      │
 │                  │    │                  │
 │  → bert_v1/      │    │  → lda_v1/       │
 └────────┬─────────┘    └────────┬─────────┘
          │                       │
          │    ┌──────────────────┐│
          │    │ Step 4c: Load    ││
          │    │ Pre-trained LSTM ││
          │    │                  ││
          │    │ lstm_v1/ (from   ││
          │    │ climate twitter  ││
          │    │ training)        ││
          │    │ AUC: 0.856       ││
          │    └────────┬─────────┘│
          │             │          │
          └─────────────┼──────────┘
                        ▼
 ┌─────────────────────────────────────┐
 │  Step 5: Ensemble Predict           │
 │  on 4,228 balanced tweets           │
 │                                     │
 │  P = 0.4*BERT + 0.4*LSTM + 0.2*LDA │
 │                                     │
 │  → tweets_ensemble_scores.csv       │
 │  Ensemble AUC: 0.926                │
 └──────────────────┬──────────────────┘
                    ▼
 ┌─────────────────────────────────────┐
 │  Step 6: Generate Sample Alerts     │
 │                                     │
 │  Filter HIGH+ alerts               │
 │  Save top 5 to outputs/alerts/      │
 │  as JSON files                      │
 └──────────────────┬──────────────────┘
                    ▼
                   END
```

---

## 12. Component Interaction Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                 COMPONENT INTERACTION DIAGRAM                     │
└───────────────────────────────────────────────────────────────────┘

  Browser                  FastAPI                   External
  (Next.js)                (Python)                  Services
      │                       │                          │
      │── GET /               │                          │
      │   (Dashboard page)    │                          │
      │                       │                          │
      │── GET /api/alerts ───►│                          │
      │◄── [ScoreResult[]] ───│                          │
      │                       │                          │
      │   [User pastes tweet] │                          │
      │                       │                          │
      │── POST /api/analyze ─►│                          │
      │   {text, demo_mode}   │                          │
      │                       │── cleaner.clean_text()   │
      │                       │── bert.predict(text)     │
      │                       │── lstm.predict(text)     │
      │                       │── lda.predict(text)      │
      │                       │── ensemble.score(…)      │
      │                       │── recommendation(type)   │
      │                       │── alert_engine.create()  │
      │                       │                          │
      │                       │ if level >= HIGH:        │
      │                       │─────────────────────────►│
      │                       │  POST slack webhook       │
      │                       │◄─────────────────────────│
      │                       │  200 OK                  │
      │                       │                          │
      │◄─── ScoreResult ──────│                          │
      │   {bert, lstm, lda,   │                          │
      │    probability, level,│                          │
      │    recommendations}   │                          │
      │                       │                          │
      │   [User enters keyword│                          │
      │                       │                          │
      │── POST /api/          │                          │
      │   fetch-and-analyze ─►│                          │
      │                       │─── Tweepy search ───────►│
      │                       │                   X API  │
      │                       │◄── RawTweet[] ───────────│
      │                       │── analyze each tweet     │
      │◄─── FetchResponse ────│                          │
      │   {results: [         │                          │
      │     ScoreResult,…]}   │                          │
```

---

## 13. Dataset Summary

| Dataset | Source | Size | Use in System |
|---------|--------|------|---------------|
| **Kaggle Disaster Tweets** | Kaggle / NLP with Disaster Tweets | 11,370 tweets | BERT training (balanced to 4,228), LDA training |
| **Climate Change Twitter** | Kaggle / Climate Change dataset | 15.8M tweets, ~116.7K hourly windows | LSTM temporal training |
| **Sentiment140** | Kaggle / Sentiment140 | 1.6M tweets | Available as alternative; column schema defined but not primary |
| **X/Twitter API v2** | Live via Tweepy | Real-time, up to 100 per request | Production inference |

### Data Processing Steps

```
Raw CSV
  ↓  schema.py          — column name mapping, label remapping (0/4 → 0/1)
  ↓  loader.py          — auto-detect dataset type, load with pandas
  ↓  cleaner.py         — URL removal, mention removal, special char strip
  ↓  balance/sample     — equal class sizes (2,114 each)
Processed CSV           — used for model training
```

---

## 14. Model Performance

| Model | Architecture | Training Data | ROC-AUC | Notes |
|-------|-------------|---------------|---------|-------|
| BERT | DistilBERT fine-tuned (binary) | 4,228 disaster tweets | **0.910** | Best single-model score |
| LSTM | 2-layer LSTM (hidden=64) | 116.7K hourly climate windows | **0.856** | Temporal anomaly detection |
| LDA | Gensim 10-topic model | 4,228 disaster tweets | **0.605** | Unsupervised; weaker but adds topic coverage |
| **Ensemble** | Weighted sum | All above | **0.926** | 88% accuracy, best overall |

```
Performance Comparison
                        0.60   0.70   0.80   0.90   1.00
                        │      │      │      │      │
  LDA      ─────────────┤      │      │      │
                       0.605   │      │      │
  LSTM     ─────────────────────────────┤    │
                               │      0.856  │
  BERT     ─────────────────────────────────-┤
                               │      │    0.910
  Ensemble ──────────────────────────────────┤
                               │      │      0.926
```

---

## 15. Deployment & Setup

### Prerequisites

- Python 3.13+
- Node.js 18+
- GPU (optional but recommended for BERT training)

### Quick Start

```bash
# 1. Clone and install Python dependencies
pip install -r requirements.txt

# 2. Install Node.js dependencies
cd cas && npm install && cd ..

# 3. Configure environment variables
cp .env.example .env
# Fill in: KAGGLE_USERNAME, KAGGLE_KEY, X_BEARER_TOKEN,
#           SLACK_WEBHOOK_URL, OPENAI_API_KEY

# 4. Train all models (~2 min GPU / ~15 min CPU)
python run_restore.py

# 5. Start FastAPI backend (Terminal 1)
uvicorn src.api.main:app --reload --port 8000

# 6. Start Next.js frontend (Terminal 2)
cd cas && npm run dev
# Visit http://localhost:3000
```

### Environment Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `KAGGLE_USERNAME` | For training | Kaggle API username |
| `KAGGLE_KEY` | For training | Kaggle API key |
| `X_BEARER_TOKEN` | Optional | X/Twitter API v2 bearer token (falls back to demo mode) |
| `SLACK_WEBHOOK_URL` | Optional | Slack incoming webhook URL for HIGH+ alerts |
| `OPENAI_API_KEY` | Optional | For AI-powered recommendation enhancement |
| `NEXT_PUBLIC_API_URL` | Optional | FastAPI URL (defaults to `http://localhost:8000`) |

### Demo Mode

When `X_BEARER_TOKEN` is absent or `demo_mode=true` is passed, the system uses pre-loaded example tweets and adjusts alert thresholds:

```
Demo thresholds (max ensemble P ≈ 0.60 without LSTM timestamps):
  CRITICAL > 0.62 | HIGH > 0.55 | MEDIUM > 0.45 | LOW ≤ 0.45
```

---

## 16. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Ensemble of 3 model types** | Each model captures a different signal: BERT = semantic intent, LSTM = temporal patterns, LDA = topic structure. Single models miss one or more of these dimensions. |
| **BERT weight 40% = LSTM weight** | BERT and LSTM are the most accurate models. LDA at 20% adds topic diversity without dragging down overall performance. |
| **Lazy model loading** | Models (especially BERT) are large. Loading on first API request avoids ~30-second startup delay while still warming the cache for subsequent requests. |
| **In-memory alert store (not a database)** | Simplifies deployment for a course project. Production would swap this for PostgreSQL or Redis. Max 500 alerts prevents unbounded memory growth. |
| **Next.js App Router + shadcn/ui** | Modern, type-safe stack. shadcn/ui components are fully customizable (unlike headless libraries) and pair naturally with Tailwind 4. |
| **Demo mode fallback** | Makes the system usable in presentations/grading without requiring paid X API credentials. |
| **Separate BERT/LSTM training datasets** | BERT is trained on labeled disaster tweets for explicit crisis/non-crisis signal. LSTM is trained on 15.8M climate tweets to learn normal temporal baselines — an entirely different kind of signal. |
| **Streamlit retained as legacy** | The Streamlit dashboard (`src/dashboard/app.py`) is the original prototype. The Next.js frontend is the production replacement, but Streamlit is kept for notebook-style demos. |

---

## Summary

The **Crisis Alert System** is a full-stack, production-ready platform that combines three complementary ML approaches — supervised BERT classification, temporal LSTM anomaly detection, and unsupervised LDA topic modeling — into a single weighted ensemble for robust real-time crisis detection in social media. The ensemble achieves **ROC-AUC 0.926 and 88% accuracy** on disaster tweet classification.

The system is served through a **FastAPI REST backend** (Python 3.13) that lazily loads all models, integrates with the X/Twitter API v2 via Tweepy for live tweet fetching, and pushes HIGH/CRITICAL alerts to Slack. The operational interface is a **Next.js 16 / React 19 dashboard** with live KPI monitoring, per-tweet score breakdowns across all three models, a recommendation engine that classifies crisis type and routes to the correct stakeholders, and full alert history.

```
 Kaggle Disaster Tweets ─┐
 Climate Twitter 15.8M ──┼── Training Pipeline ──► Ensemble Model
                         │   (run_restore.py)      ROC-AUC 0.926
                         │
 Live X/Twitter API ──────┬── FastAPI Backend ──► Slack Webhook
 Manual tweet input ──────┘   (uvicorn :8000)  └► Next.js Dashboard
                                                   (localhost:3000)
```
