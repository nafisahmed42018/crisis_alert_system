"""Streamlit Crisis Alert Dashboard.

Run from the project root:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of cwd
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.models.ensemble      import CrisisEnsemble
from src.alerts.alert_engine  import AlertEngine
from src.dashboard.components import (
    alert_badge, gauge_chart, score_bar_chart,
    alert_distribution_chart, score_histogram,
    top_crisis_table, timeseries_chart, LEVEL_COLORS, LEVEL_EMOJI,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "Crisis Alert System",
    page_icon  = "🚨",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa; border-radius: 8px;
        padding: 16px; text-align: center;
    }
    .metric-label { font-size: 0.85rem; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 1.8rem; font-weight: bold; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data / model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models (BERT + LDA + LSTM)...")
def load_ensemble():
    return CrisisEnsemble.load(
        bert_path = str(ROOT / "outputs/models/bert_v1"),
        lda_path  = str(ROOT / "outputs/models/lda_v1"),
        lstm_path = str(ROOT / "outputs/models/lstm_v1"),
    )


@st.cache_data(show_spinner="Loading scored dataset...")
def load_scored_df():
    path = ROOT / "data/processed/tweets_ensemble_scores.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading LSTM time series...")
def load_lstm_scores():
    path = ROOT / "data/processed/climate_lstm_scores.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🚨 Crisis Alert System")
    st.caption("Business Applications of ML in Social Media")
    st.divider()

    st.markdown("**Pipeline**")
    st.markdown("""
- 🔵 **BERT** — Crisis intent (40%)
- 🟢 **LSTM** — Temporal anomaly (20%)
- 🟣 **LDA** — Topic shift (40%)
""")
    st.divider()

    st.markdown("**Alert Thresholds** *(demo mode)*")
    for level, thr in [("CRITICAL", ">62%"), ("HIGH", ">55%"),
                       ("MEDIUM",   ">45%"), ("LOW",  "≤45%")]:
        st.markdown(f"{LEVEL_EMOJI[level]} **{level}** — {thr}")

    st.divider()
    st.caption("AML-3203 Course Project")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Live Analyzer",
    "📊 Alert Dashboard",
    "🤖 Model Performance",
    "ℹ️  About",
])

# ===========================================================================
# TAB 1 — Live Analyzer
# ===========================================================================

with tab1:
    st.header("Live Tweet Crisis Analyzer")
    st.caption("Enter any tweet text to get an instant crisis probability score.")

    col_input, col_examples = st.columns([3, 1])

    with col_examples:
        st.markdown("**Quick examples**")
        example_tweets = {
            "Wildfire 🔴":   "Massive wildfire destroys thousands of homes, evacuation ordered across three counties",
            "Oil spill 🔴":  "Oil spill reported near Gulf coast, marine life at critical risk, emergency teams deployed",
            "Flood 🟠":      "Flash flood warnings issued for low-lying areas, rivers approaching record levels",
            "Normal 🟢":     "Just had the best weekend camping trip, nature is so beautiful this time of year",
            "Sports 🟢":     "What a game last night! Cannot believe that final score, absolute thriller",
        }
        chosen = st.radio("Load example", list(example_tweets.keys()), index=0)

    with col_input:
        tweet_text = st.text_area(
            "Tweet text",
            value=example_tweets[chosen],
            height=120,
            placeholder="Type or paste a tweet here...",
        )
        analyze_btn = st.button("🔍 Analyze Tweet", type="primary", use_container_width=True)

    if analyze_btn and tweet_text.strip():
        with st.spinner("Running ensemble..."):
            ensemble = load_ensemble()
            results  = ensemble.predict_df([tweet_text.strip()], demo_mode=True)
            row      = results.iloc[0]

        st.divider()

        # Alert badge
        st.markdown(alert_badge(row["alert_level"], row["crisis_probability"]),
                    unsafe_allow_html=True)
        st.markdown("")

        col_gauge, col_bars, col_scores = st.columns([2, 2, 1])

        with col_gauge:
            st.plotly_chart(
                gauge_chart(row["crisis_probability"], row["alert_level"]),
                use_container_width=True,
            )

        with col_bars:
            st.markdown("**Model Contributions**")
            st.plotly_chart(
                score_bar_chart(row["bert_score"], row["lstm_score"], row["lda_score"]),
                use_container_width=True,
            )

        with col_scores:
            st.markdown("**Scores**")
            st.metric("BERT",     f"{row['bert_score']:.3f}")
            st.metric("LSTM",     f"{row['lstm_score']:.3f}", help="0.5 = neutral (no timestamp)")
            st.metric("LDA",      f"{row['lda_score']:.3f}")
            st.metric("Ensemble", f"{row['crisis_probability']:.3f}")

        # Alert engine
        engine = AlertEngine(min_level="MEDIUM")
        alerts = engine.process(results)
        if alerts:
            with st.expander("📋 Alert JSON"):
                st.code(alerts[0].to_json(), language="json")

# ===========================================================================
# TAB 2 — Alert Dashboard
# ===========================================================================

with tab2:
    st.header("Alert Dashboard")
    st.caption("Ensemble scores over the full Kaggle Disaster Tweets dataset (7,613 tweets).")

    df = load_scored_df()

    if df.empty:
        st.warning("Run the ensemble pipeline first to generate scored data.")
    else:
        # KPI row
        total    = len(df)
        n_crit   = (df["alert_level"] == "CRITICAL").sum()
        n_high   = (df["alert_level"] == "HIGH").sum()
        n_medium = (df["alert_level"] == "MEDIUM").sum()
        avg_prob = df["crisis_probability"].mean()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Tweets",   f"{total:,}")
        k2.metric("🔴 CRITICAL",   f"{n_crit:,}",   f"{n_crit/total*100:.1f}%")
        k3.metric("🟠 HIGH",       f"{n_high:,}",   f"{n_high/total*100:.1f}%")
        k4.metric("🟡 MEDIUM",     f"{n_medium:,}", f"{n_medium/total*100:.1f}%")
        k5.metric("Avg Crisis Prob", f"{avg_prob:.3f}")

        st.divider()

        col_bar, col_hist = st.columns(2)
        with col_bar:
            st.plotly_chart(alert_distribution_chart(df), use_container_width=True)
        with col_hist:
            st.plotly_chart(score_histogram(df), use_container_width=True)

        st.divider()
        st.subheader("Top Crisis Tweets")
        level_filter = st.selectbox("Filter by alert level",
                                    ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
        filtered = df if level_filter == "ALL" else df[df["alert_level"] == level_filter]
        top = top_crisis_table(filtered, n=20)
        st.dataframe(
            top.style.background_gradient(subset=["crisis_probability"], cmap="Reds"),
            use_container_width=True,
        )

        st.divider()
        st.subheader("LSTM Climate Signal Over Time")
        lstm_df = load_lstm_scores()
        if not lstm_df.empty:
            n_days = st.slider("Days to show", 30, 365*5, 365, step=30)
            st.plotly_chart(timeseries_chart(lstm_df, n_days=n_days), use_container_width=True)

# ===========================================================================
# TAB 3 — Model Performance
# ===========================================================================

with tab3:
    st.header("Model Performance")

    df = load_scored_df()

    if not df.empty and "true_label" in df.columns:
        from sklearn.metrics import roc_auc_score, classification_report
        import numpy as np

        y_true = df["true_label"].astype(int)
        y_prob = df["crisis_probability"]
        y_pred = (y_prob >= 0.5).astype(int)

        roc = roc_auc_score(y_true, y_prob)
        acc = (y_pred == y_true).mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ensemble ROC-AUC", f"{roc:.3f}")
        m2.metric("Accuracy",         f"{acc:.1%}")
        m3.metric("BERT ROC-AUC",     "0.910")
        m4.metric("LSTM ROC-AUC",     "0.856")

        st.divider()

        # Score distributions per model
        col_a, col_b, col_c = st.columns(3)
        for col, score_col, name, color in [
            (col_a, "bert_score",  "BERT (40%)",  "#e63946"),
            (col_b, "lstm_score",  "LSTM (20%)",  "#457b9d"),
            (col_c, "lda_score",   "LDA (40%)",   "#2a9d8f"),
        ]:
            fig = go.Figure()
            for lbl, c in [(0, "#adb5bd"), (1, color)]:
                subset = df[df["true_label"] == lbl][score_col]
                fig.add_trace(go.Histogram(
                    x=subset, name="Normal" if lbl == 0 else "Crisis",
                    marker_color=c, opacity=0.7, nbinsx=40, histnorm="probability density",
                ))
            fig.update_layout(
                barmode="overlay", height=280,
                title=f"{name} Score Distribution",
                xaxis_title="Score", yaxis_title="Density",
                legend=dict(x=0.7, y=0.95), margin=dict(t=40, b=20),
            )
            col.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Ensemble score distribution
        fig = go.Figure()
        for lbl, c, name in [(0,"#457b9d","Normal"),(1,"#e63946","Crisis")]:
            subset = df[df["true_label"]==lbl]["crisis_probability"]
            fig.add_trace(go.Histogram(
                x=subset, name=name, marker_color=c,
                opacity=0.7, nbinsx=50, histnorm="probability density",
            ))
        fig.add_vline(x=0.5, line_dash="dash", line_color="black",
                      annotation_text="threshold=0.5")
        fig.update_layout(
            barmode="overlay", height=320,
            title="Ensemble Score Distribution by True Label",
            xaxis_title="Crisis Probability", yaxis_title="Density",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Run the ensemble pipeline first to see performance metrics.")

    # Static charts from outputs/charts/
    st.divider()
    st.subheader("Training Charts")
    charts_dir = ROOT / "outputs/charts"
    chart_files = {
        "BERT Validation":        "07_bert_validation.png",
        "BERT Score Distribution":"08_bert_score_dist.png",
        "LDA Coherence":          "09_lda_coherence.png",
        "LDA Topics":             "10_lda_topics.png",
        "LSTM Validation":        "13_lstm_validation.png",
    }
    cols = st.columns(2)
    for i, (title, fname) in enumerate(chart_files.items()):
        path = charts_dir / fname
        if path.exists():
            cols[i % 2].image(str(path), caption=title, use_container_width=True)

# ===========================================================================
# TAB 4 — About
# ===========================================================================

with tab4:
    st.header("About — Crisis Alert System")
    st.markdown("""
**Course**: AML-3203 Business Applications of ML in Social Media

### What This System Does

Detects social media crises using an ensemble of three ML models:

| Model | Architecture | Dataset | Weight |
|---|---|---|---|
| **BERT** | DistilBERT fine-tuned | Disaster Tweets (7.6K) | 40% |
| **LSTM** | 2-layer LSTM, hidden=64 | Climate Twitter (15.8M) | 20% |
| **LDA** | Gensim LDA, k=5 topics | Disaster Tweets (7.6K) | 40% |

### Ensemble Formula

```
crisis_probability = 0.40 × BERT + 0.20 × LSTM + 0.40 × LDA
```

### Alert Levels

| Level | Threshold | Action |
|---|---|---|
| 🔴 CRITICAL | > 85% (62% demo) | Immediate escalation |
| 🟠 HIGH     | > 70% (55% demo) | Urgent review |
| 🟡 MEDIUM   | > 50% (45% demo) | Monitor closely |
| 🟢 LOW      | ≤ 50%            | No action required |

### Datasets

- **Kaggle Disaster Tweets** — 7,613 hand-labeled crisis/normal tweets
- **Climate Change Twitter** — 15.8M timestamped tweets (2006–2019) with continuous sentiment scores

### Performance

| Metric | Value |
|---|---|
| Ensemble ROC-AUC | 0.926 |
| Ensemble Accuracy | 88% |
| BERT ROC-AUC | ~0.91 |
| LSTM ROC-AUC | 0.856 |
| LDA ROC-AUC | 0.605 |
""")
