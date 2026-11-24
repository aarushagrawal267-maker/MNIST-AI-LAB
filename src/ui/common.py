"""Cached loaders and chart builders shared by all Streamlit pages."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .. import model_manager as mm
from ..config import EDA_STATS_PATH, EXPERIMENTS_PATH, LAB_NAME, MODEL_NAME, SUMMARY_PATH
from . import theme as T


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading MNIST from local cache…")
def get_splits():
    from ..data_loader import get_splits as _gs

    return _gs()


def safe_splits():
    """Return splits or render a friendly error with manual-download instructions."""
    from ..data_loader import DatasetUnavailableError

    try:
        return get_splits()
    except DatasetUnavailableError as exc:
        st.error("The MNIST dataset is not available yet.")
        st.code(str(exc))
        st.stop()


@st.cache_resource(show_spinner="Loading model…")
def _load_predictor(key: str, mtime: float):
    from ..prediction import Predictor

    return Predictor(key)


def get_predictor(key: str | None = None):
    key = key or mm.production_key()
    if key is None:
        return None
    path = mm.abs_path(mm.get_entry(key)["path"])
    return _load_predictor(key, path.stat().st_mtime if path.exists() else 0.0)


def get_model(key: str):
    return get_predictor(key).model


def _mtime(p) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _json(path_str: str, mtime: float):
    from pathlib import Path

    p = Path(path_str)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def eda_stats() -> dict | None:
    return _json(str(EDA_STATS_PATH), _mtime(EDA_STATS_PATH))


def summary() -> dict | None:
    return _json(str(SUMMARY_PATH), _mtime(SUMMARY_PATH))


def metrics(key: str, split: str) -> dict | None:
    from ..evaluate import metrics_path

    p = metrics_path(key, split)
    return _json(str(p), _mtime(p))


@st.cache_data(show_spinner=False)
def _preds(key: str, split: str, mtime: float):
    from ..evaluate import load_predictions

    return load_predictions(key, split)


def predictions(key: str, split: str):
    from ..evaluate import predictions_path

    return _preds(key, split, _mtime(predictions_path(key, split)))


def experiments():
    import pandas as pd

    if not EXPERIMENTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(EXPERIMENTS_PATH)


def models(include_custom: bool = True) -> dict[str, dict[str, Any]]:
    ms = mm.list_models()
    if not include_custom:
        ms = {k: v for k, v in ms.items() if not v.get("custom")}
    return ms


def label(key: str) -> str:
    return mm.display_name(key)


def require_models() -> dict[str, dict[str, Any]]:
    ms = models()
    if not ms or mm.production_key() is None:
        st.warning(
            f"No trained models were found. Train them from the **Model Training** page or run "
            f"`python -m src.train --all` (≈20–40 min on a laptop CPU)."
        )
        st.stop()
    return ms


def model_select(label_text: str, keys: list[str] | None = None, key: str | None = None,
                 default: str | None = None, keras_only: bool = False, sidebar: bool = False) -> str:
    ms = models()
    keys = keys or [k for k, v in ms.items() if (v["family"] == "keras" or not keras_only)]
    prod = mm.production_key()
    default = default or (prod if prod in keys else keys[0])
    target = st.sidebar if sidebar else st
    return target.selectbox(label_text, keys, index=keys.index(default) if default in keys else 0,
                            format_func=label, key=key)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def prob_bar(proba: np.ndarray, height: int = 300, title: str | None = "Probability for every digit") -> go.Figure:
    proba = np.asarray(proba, dtype=float)
    top = int(proba.argmax())
    colors = [T.SERIES[0] if i == top else T.NEUTRAL for i in range(10)]
    text = [f"{p * 100:.1f}%" if i == top else "" for i, p in enumerate(proba)]
    fig = go.Figure(go.Bar(
        x=[str(i) for i in range(10)], y=proba * 100, marker=dict(color=colors, cornerradius=4),
        text=text, textposition="outside", textfont=dict(color=T.TEXT),
        hovertemplate="digit %{x}<br>probability %{y:.3f}%<extra></extra>", cliponaxis=False,
    ))
    fig.update_layout(yaxis=dict(title="probability (%)", range=[0, 112], ticksuffix=""),
                      xaxis=dict(title="digit", type="category"), showlegend=False)
    return T.style_fig(fig, height=height, title=title)


def confusion_heatmap(cm, normalize: bool = False, height: int = 520, title: str = "Confusion matrix") -> go.Figure:
    cm = np.asarray(cm, dtype=float)
    z = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    text = [[(f"{v * 100:.1f}" if normalize else f"{int(v)}") if v > 0 else "" for v in row] for row in z]
    zmax = z.max() if not normalize else 1.0
    # Log-ish emphasis so off-diagonal errors stay visible next to large diagonals
    zc = np.sqrt(z / zmax) if zmax > 0 else z
    fig = go.Figure(go.Heatmap(
        z=zc, x=[str(i) for i in range(10)], y=[str(i) for i in range(10)], text=text, texttemplate="%{text}",
        textfont=dict(size=11), customdata=z,
        colorscale=[[i / (len(T.SEQ_BLUE) - 1), c] for i, c in enumerate(T.SEQ_BLUE)], showscale=False,
        hovertemplate=("actual %{y} → predicted %{x}<br>" + ("%{customdata:.2%} of class" if normalize else "%{customdata:.0f} images") + "<extra></extra>"),
        xgap=2, ygap=2,
    ))
    fig.update_layout(xaxis=dict(title="predicted digit", side="bottom", showgrid=False, type="category"),
                      yaxis=dict(title="actual digit", autorange="reversed", showgrid=False, type="category"))
    return T.style_fig(fig, height=height, title=title)


def history_figs(history: dict) -> tuple[go.Figure, go.Figure, go.Figure | None]:
    ep = list(range(1, len(history.get("loss", [])) + 1))
    acc = go.Figure()
    acc.add_scatter(x=ep, y=history.get("accuracy", []), name="training", mode="lines+markers",
                    line=dict(color=T.SERIES[0], width=2), marker=dict(size=7))
    acc.add_scatter(x=ep, y=history.get("val_accuracy", []), name="validation", mode="lines+markers",
                    line=dict(color=T.SERIES[1], width=2), marker=dict(size=7))
    acc.update_layout(xaxis=dict(title="epoch", dtick=1), yaxis_title="accuracy", hovermode="x unified")
    loss = go.Figure()
    loss.add_scatter(x=ep, y=history.get("loss", []), name="training", mode="lines+markers",
                     line=dict(color=T.SERIES[0], width=2), marker=dict(size=7))
    loss.add_scatter(x=ep, y=history.get("val_loss", []), name="validation", mode="lines+markers",
                     line=dict(color=T.SERIES[1], width=2), marker=dict(size=7))
    loss.update_layout(xaxis=dict(title="epoch", dtick=1), yaxis_title="loss (cross-entropy)", hovermode="x unified")
    lr = None
    if history.get("learning_rate"):
        lr = go.Figure(go.Scatter(x=ep, y=history["learning_rate"], mode="lines+markers", line=dict(color=T.SERIES[2], width=2, shape="hv"),
                                  marker=dict(size=7), name="learning rate", hovertemplate="epoch %{x}<br>lr %{y:.2e}<extra></extra>"))
        lr.update_layout(xaxis=dict(title="epoch", dtick=1), yaxis=dict(title="learning rate", type="log", exponentformat="e"))
    return (T.style_fig(acc, 320, "Accuracy"), T.style_fig(loss, 320, "Loss"),
            T.style_fig(lr, 320, "Learning-rate schedule") if lr else None)


def digit_heatmap(img: np.ndarray, title: str = "", height: int = 380, show_values: bool = True) -> go.Figure:
    img = np.asarray(img)
    text = [[str(int(v)) if v > 0 else "" for v in row] for row in img] if show_values else None
    fig = go.Figure(go.Heatmap(z=img, colorscale="gray", showscale=False, text=text, texttemplate="%{text}" if show_values else None,
                               textfont=dict(size=7, color="#ff9db5"), hovertemplate="row %{y}, col %{x}<br>value %{z}<extra></extra>"))
    fig.update_layout(yaxis=dict(autorange="reversed", showgrid=False, scaleanchor="x", constrain="domain"),
                      xaxis=dict(showgrid=False, constrain="domain"))
    return T.style_fig(fig, height=height, title=title, margin=dict(l=30, r=10, t=40, b=30))


def heat_image(img: np.ndarray, colorscale="Inferno", height: int = 260, title: str = "", zmid=None) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=np.asarray(img), colorscale=colorscale, showscale=False, zmid=zmid,
                               hovertemplate="row %{y}, col %{x}<br>%{z:.3f}<extra></extra>"))
    fig.update_layout(yaxis=dict(autorange="reversed", showgrid=False, showticklabels=False, scaleanchor="x", showline=False),
                      xaxis=dict(showgrid=False, showticklabels=False, showline=False))
    return T.style_fig(fig, height=height, title=title, margin=dict(l=4, r=4, t=36 if title else 4, b=4))


def plot(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, config=T.PLOTLY_CONFIG, key=key)


def sidebar_status() -> None:
    prod = mm.production_key()
    with st.sidebar:
        if prod:
            e = mm.get_entry(prod)
            va = e.get("val_accuracy")
            st.markdown(
                f'<div class="purf-card" style="padding:14px 16px"><span class="badge trax">⚡ {MODEL_NAME} online</span>'
                f'<p style="margin-top:10px;font-size:.82rem">{T.esc(e.get("display_name", prod))}<br>'
                f'validation accuracy <b style="color:white">{va * 100:.2f}%</b></p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<span class="badge bad">{MODEL_NAME} not trained yet</span>', unsafe_allow_html=True)
