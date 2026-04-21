"""Reusable chart and UI helpers for the Streamlit dashboard."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LEVEL_COLORS = {
    "LOW":      "#6c757d",
    "MEDIUM":   "#ffc107",
    "HIGH":     "#fd7e14",
    "CRITICAL": "#e63946",
}

LEVEL_EMOJI = {
    "LOW":      "🟢",
    "MEDIUM":   "🟡",
    "HIGH":     "🟠",
    "CRITICAL": "🔴",
}


def alert_badge(level: str, prob: float) -> str:
    """Return HTML for a coloured alert badge."""
    color = LEVEL_COLORS[level]
    emoji = LEVEL_EMOJI[level]
    return (
        f'<div style="display:inline-block;padding:10px 22px;border-radius:8px;'
        f'background:{color};color:white;font-size:1.4rem;font-weight:bold;">'
        f'{emoji} {level} &nbsp;|&nbsp; {prob:.1%}</div>'
    )


def gauge_chart(prob: float, level: str) -> go.Figure:
    """Plotly gauge showing crisis probability."""
    color = LEVEL_COLORS[level]
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = prob * 100,
        delta = {"reference": 50, "suffix": "%"},
        number= {"suffix": "%", "font": {"size": 36}},
        title = {"text": "Crisis Probability", "font": {"size": 16}},
        gauge = {
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  50], "color": "#e9ecef"},
                {"range": [50, 70], "color": "#fff3cd"},
                {"range": [70, 85], "color": "#ffe0cc"},
                {"range": [85,100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.85,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def score_bar_chart(bert: float, lstm: float, lda: float) -> go.Figure:
    """Horizontal bar chart showing individual model scores."""
    models = ["LDA (20%)", "LSTM (40%)", "BERT (40%)"]
    scores = [lda, lstm, bert]
    colors = ["#2a9d8f", "#457b9d", "#e63946"]

    fig = go.Figure(go.Bar(
        x=scores, y=models,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(
        height=200,
        xaxis=dict(range=[0, 1], title="Score"),
        margin=dict(t=10, b=30, l=10, r=60),
        showlegend=False,
    )
    return fig


def alert_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of alert level counts."""
    level_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = df["alert_level"].value_counts().reindex(level_order, fill_value=0)
    fig = go.Figure(go.Bar(
        x=level_order,
        y=counts.values,
        marker_color=[LEVEL_COLORS[l] for l in level_order],
        text=counts.values,
        textposition="outside",
    ))
    fig.update_layout(
        height=320,
        yaxis_title="Tweet count",
        title="Alert Level Distribution",
        margin=dict(t=40, b=20),
    )
    return fig


def score_histogram(df: pd.DataFrame) -> go.Figure:
    """Crisis probability histogram coloured by alert level."""
    level_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    fig = go.Figure()
    for level in level_order:
        subset = df[df["alert_level"] == level]["crisis_probability"]
        fig.add_trace(go.Histogram(
            x=subset, name=level,
            marker_color=LEVEL_COLORS[level],
            opacity=0.75, nbinsx=50,
        ))
    for thr, lbl in [(0.45,"MEDIUM"),(0.55,"HIGH"),(0.62,"CRITICAL")]:
        fig.add_vline(x=thr, line_dash="dash", line_color="black",
                      annotation_text=lbl, annotation_position="top right",
                      annotation_font_size=10)
    fig.update_layout(
        barmode="stack", height=320,
        xaxis_title="Crisis Probability",
        yaxis_title="Count",
        title="Score Distribution by Alert Level",
        margin=dict(t=40, b=20),
    )
    return fig


def top_crisis_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-n highest scoring tweets as a display DataFrame."""
    cols = ["text", "crisis_probability", "bert_score", "lstm_score",
            "lda_score", "alert_level"]
    available = [c for c in cols if c in df.columns]
    return (df[available]
            .nlargest(n, "crisis_probability")
            .reset_index(drop=True)
            .round(3))


def timeseries_chart(scores_df: pd.DataFrame, n_days: int = 365) -> go.Figure:
    """Plot recent LSTM hourly crisis scores as a time series."""
    df = scores_df.copy()
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    df = df.sort_values("hour").tail(n_days * 24)
    daily = df.set_index("hour")["lstm_score"].resample("D").mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["hour"], y=daily["lstm_score"],
        mode="lines", line=dict(color="#457b9d", width=1.5),
        fill="tozeroy", fillcolor="rgba(69,123,157,0.15)",
        name="Daily avg LSTM score",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                  annotation_text="crisis threshold", annotation_position="bottom right")
    fig.update_layout(
        height=300,
        xaxis_title="Date",
        yaxis_title="LSTM Crisis Score",
        yaxis=dict(range=[0, 1]),
        title=f"LSTM Crisis Signal — Last {n_days} Days",
        margin=dict(t=40, b=20),
    )
    return fig
