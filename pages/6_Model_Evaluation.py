"""Model Evaluation — confusion matrix, classification report, ROC and precision-recall analysis."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluate import evaluate_model, pr_curves, roc_curves
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Modeling",
    "Model Evaluation",
    "Detailed metrics for any trained model. Choose the <b>validation</b> split (what model selection used) or the "
    "<b>test</b> split (the final held-out score).",
)
C.require_models()
c1, c2, _ = st.columns([1.6, 1.2, 1])
key = c1.selectbox("Model", list(C.models()), format_func=C.label, key="eval_model",
                   index=list(C.models()).index(C.mm.production_key()) if C.mm.production_key() in C.models() else 0)
split = c2.radio("Split", ["test", "val"], horizontal=True, format_func={"test": "Test (final)", "val": "Validation (selection)"}.get)

m = C.metrics(key, split)
pr = C.predictions(key, split)
if m is None or pr is None:
    st.info("This model has not been evaluated on this split yet.")
    if st.button("Evaluate now", type="primary"):
        with st.spinner("Evaluating…"):
            evaluate_model(key, split, splits=C.safe_splits())
        st.rerun()
    st.stop()
proba, y = pr

T.metric_row([
    T.metric_card("Accuracy", f"{m['accuracy'] * 100:.2f}%", f"{m['n_errors']} errors / {m['n_samples']:,}", "🎯", T.BRAND),
    T.metric_card("Precision", f"{m['precision_macro'] * 100:.2f}%", "macro average", "", T.SERIES[0]),
    T.metric_card("Recall", f"{m['recall_macro'] * 100:.2f}%", "macro average", "", T.SERIES[2]),
    T.metric_card("F1-score", f"{m['f1_macro'] * 100:.2f}%", "macro average", "", T.SERIES[3]),
    T.metric_card("ROC AUC", f"{m['roc_auc_ovr_macro']:.4f}", "one-vs-rest, macro", "", T.BRAND_2),
    T.metric_card("Log loss", f"{m['log_loss']:.4f}", f"top-2 acc. {m['top2_accuracy'] * 100:.2f}%", "", T.SERIES[4]),
])

T.section("Confusion matrix", "rows = actual digit, columns = predicted digit")
a, b = st.columns([1.35, 1])
with a:
    norm = st.toggle("Normalise by row (% of each actual digit)", value=False)
    C.plot(C.confusion_heatmap(m["confusion_matrix"], normalize=norm, height=560, title=None), key="cm")
    st.caption("Colour uses a square-root scale so that small off-diagonal error counts remain visible next to the "
               "large diagonal.")
with b:
    T.section("Per-digit accuracy")
    pca = np.array(m["per_class_accuracy"]) * 100
    fig = go.Figure(go.Bar(x=[str(i) for i in range(10)], y=100 - pca, marker=dict(color=T.SERIES[0], cornerradius=3),
                           customdata=pca, hovertemplate="digit %{x}<br>accuracy %{customdata:.2f}%<br>error %{y:.2f}%<extra></extra>"))
    worst = int(np.argmin(pca))
    fig.add_annotation(x=str(worst), y=100 - pca[worst], text=f"hardest: {worst}", showarrow=False, yshift=14,
                       font=dict(color=T.TEXT_2))
    fig.update_layout(xaxis=dict(title="digit", type="category"), yaxis_title="error rate (%)")
    C.plot(T.style_fig(fig, 300), key="pca")
    from src.evaluate import top_confusions

    tc = top_confusions(m["confusion_matrix"], 6)
    st.markdown("**Most frequent confusions**")
    st.dataframe(pd.DataFrame([{"actual": t["actual"], "predicted as": t["predicted"], "count": t["count"]} for t in tc]),
                 hide_index=True, use_container_width=True)

T.section("Classification report")
rep = m["classification_report"]
rows = [{"digit": str(i), "precision": rep[str(i)]["precision"], "recall": rep[str(i)]["recall"],
         "f1-score": rep[str(i)]["f1-score"], "support": int(rep[str(i)]["support"]),
         "accuracy": m["per_class_accuracy"][i]} for i in range(10)]
for avg in ("macro avg", "weighted avg"):
    rows.append({"digit": avg, "precision": rep[avg]["precision"], "recall": rep[avg]["recall"],
                 "f1-score": rep[avg]["f1-score"], "support": int(rep[avg]["support"]), "accuracy": None})
fmt = st.column_config.ProgressColumn(min_value=0.9 if m["accuracy"] > 0.95 else 0.0, max_value=1.0, format="%.4f")
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
             column_config={"precision": fmt, "recall": fmt, "f1-score": fmt, "accuracy": st.column_config.NumberColumn(format="%.4f")})

# ------------------------------------------------------------------ curves
T.section("ROC & precision-recall", "one-vs-rest per digit")
c1, c2 = st.columns([1, 1])
focus = c1.selectbox("Highlight digit", list(range(10)), index=int(np.argmin(m["per_class_accuracy"])))
zoom = c2.toggle("Zoom into the top-left corner", value=True,
                 help="High-accuracy models hug the corner; zooming reveals the differences.")
roc = roc_curves(y, proba)
prc = pr_curves(y, proba)
a, b = st.columns(2)
with a:
    fig = go.Figure()
    for c in range(10):
        if c == focus:
            continue
        fig.add_scatter(x=roc[c]["fpr"], y=roc[c]["tpr"], mode="lines", line=dict(color=T.NEUTRAL, width=1.2),
                        name=f"{c}", showlegend=False, hovertemplate=f"digit {c} (AUC {roc[c]['auc']:.5f})<br>FPR %{{x:.4f}}<br>TPR %{{y:.4f}}<extra></extra>")
    fig.add_scatter(x=roc[focus]["fpr"], y=roc[focus]["tpr"], mode="lines", line=dict(color=T.BRAND_2, width=2.5),
                    name=f"digit {focus} · AUC {roc[focus]['auc']:.5f}",
                    hovertemplate=f"digit {focus}<br>FPR %{{x:.4f}}<br>TPR %{{y:.4f}}<extra></extra>")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.AXIS, width=1), showlegend=False, hoverinfo="skip")
    fig.update_layout(xaxis=dict(title="false positive rate", range=[0, 0.05] if zoom else [0, 1]),
                      yaxis=dict(title="true positive rate", range=[0.85, 1.005] if zoom else [0, 1.02]),
                      legend=dict(y=0.02, yanchor="bottom", x=0.98, xanchor="right"))
    C.plot(T.style_fig(fig, 420, "ROC curves (grey = other digits)"), key="roc")
with b:
    fig = go.Figure()
    for c in range(10):
        if c == focus:
            continue
        fig.add_scatter(x=prc[c]["recall"], y=prc[c]["precision"], mode="lines", line=dict(color=T.NEUTRAL, width=1.2),
                        showlegend=False, hovertemplate=f"digit {c} (AP {prc[c]['ap']:.5f})<br>recall %{{x:.4f}}<br>precision %{{y:.4f}}<extra></extra>")
    fig.add_scatter(x=prc[focus]["recall"], y=prc[focus]["precision"], mode="lines", line=dict(color=T.BRAND_2, width=2.5),
                    name=f"digit {focus} · AP {prc[focus]['ap']:.5f}")
    fig.update_layout(xaxis=dict(title="recall", range=[0.85, 1.005] if zoom else [0, 1.02]),
                      yaxis=dict(title="precision", range=[0.85, 1.005] if zoom else [0, 1.02]),
                      legend=dict(y=0.02, yanchor="bottom", x=0.02, xanchor="left"))
    C.plot(T.style_fig(fig, 420, "Precision-recall curves"), key="prc")
auc_df = pd.DataFrame({"digit": list(range(10)), "ROC AUC": [roc[c]["auc"] for c in range(10)],
                       "average precision": [prc[c]["ap"] for c in range(10)]})
with st.expander("AUC / AP table"):
    st.dataframe(auc_df, hide_index=True, use_container_width=True,
                 column_config={"ROC AUC": st.column_config.NumberColumn(format="%.6f"),
                                "average precision": st.column_config.NumberColumn(format="%.6f")})

T.section("Confidence calibration", "is a 90%-confident prediction right 90% of the time?")
conf = proba.max(1)
correct = proba.argmax(1) == y
bins = np.array([0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0001])
ids = np.digitize(conf, bins) - 1
rows = []
for i in range(len(bins) - 1):
    sel = ids == i
    if sel.sum():
        rows.append({"bin": f"{bins[i]:.3g}–{min(bins[i + 1], 1):.3g}", "mean conf": conf[sel].mean() * 100,
                     "accuracy": correct[sel].mean() * 100, "n": int(sel.sum())})
cal = pd.DataFrame(rows)
a, b = st.columns([1.3, 1])
with a:
    fig = go.Figure()
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=T.AXIS, width=1), name="perfect calibration")
    fig.add_scatter(x=cal["mean conf"], y=cal["accuracy"], mode="lines+markers", name="model",
                    line=dict(color=T.SERIES[0], width=2), marker=dict(size=9, line=dict(width=2, color=T.SURFACE)),
                    customdata=cal["n"], hovertemplate="mean confidence %{x:.2f}%<br>accuracy %{y:.2f}%<br>%{customdata:,} images<extra></extra>")
    fig.update_layout(xaxis_title="mean confidence in bin (%)", yaxis_title="actual accuracy (%)")
    C.plot(T.style_fig(fig, 340), key="cal")
with b:
    st.dataframe(cal, hide_index=True, use_container_width=True,
                 column_config={"mean conf": st.column_config.NumberColumn(format="%.2f"),
                                "accuracy": st.column_config.NumberColumn(format="%.2f")})
st.download_button("Download metrics (JSON)", json.dumps(m, indent=2).encode(), f"{key}_{split}_metrics.json",
                   "application/json", icon=":material/download:")
