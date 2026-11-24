"""Model Comparison — every algorithm side by side, plus a head-to-head duel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import model_manager as mm
from src.config import MODEL_NAME
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Modeling",
    "Model Comparison",
    "Logistic regression, k-nearest neighbours, a support vector machine, a fully-connected neural network and "
    "convolutional networks — compared on identical data. Validation results drove model selection; test results "
    "are the final, held-out score and are shown separately.",
)
C.require_models()
splits = C.safe_splits()
show_custom = st.toggle("Include models trained in the app", value=False)
ms = C.models(include_custom=show_custom)
prod = mm.production_key()

rows = []
for k, v in ms.items():
    mv, mt = C.metrics(k, "val"), C.metrics(k, "test")
    if not mv:
        continue
    rows.append({
        "key": k, "model": C.label(k), "family": "deep learning" if v["family"] == "keras" else "classical ML",
        "architecture": v.get("architecture", ""), "params": v.get("params") or np.nan,
        "train_samples": v.get("train_samples"), "train_time_s": v.get("train_time_s") or np.nan,
        "val_acc": mv["accuracy"], "val_prec": mv["precision_macro"], "val_rec": mv["recall_macro"], "val_f1": mv["f1_macro"],
        "val_loss": mv["log_loss"],
        "test_acc": mt["accuracy"] if mt else np.nan, "test_prec": mt["precision_macro"] if mt else np.nan,
        "test_rec": mt["recall_macro"] if mt else np.nan, "test_f1": mt["f1_macro"] if mt else np.nan,
        "ms_per_img": (mt or mv).get("inference_ms_per_image", np.nan), "latency_ms": (mt or mv).get("latency_ms_single", np.nan),
    })
df = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
if df.empty:
    st.info("Run `python -m src.train --all` to evaluate the models.")
    st.stop()

best = df.iloc[0]
T.metric_row([
    T.metric_card("Best validation acc.", f"{best.val_acc * 100:.2f}%", T.esc(best.model), "🏆", T.BRAND),
    T.metric_card("Its test accuracy", f"{best.test_acc * 100:.2f}%" if not np.isnan(best.test_acc) else "—", "10,000 held-out images", "🏁", T.BRAND_2),
    T.metric_card("Fastest inference", f"{df.ms_per_img.min():.3f} ms", T.esc(df.loc[df.ms_per_img.idxmin(), 'model']) + " · per image, batched", "⚡", T.SERIES[2]),
    T.metric_card("Models compared", str(len(df)), "same splits, same metrics", "🧪", T.SERIES[3]),
])

T.section("Validation results", "used for model selection")
pct = st.column_config.NumberColumn(format="%.2f%%")
v = df[["model", "family", "architecture", "val_acc", "val_prec", "val_rec", "val_f1", "val_loss", "train_time_s", "params"]].copy()
for c in ("val_acc", "val_prec", "val_rec", "val_f1"):
    v[c] *= 100
st.dataframe(v, hide_index=True, width="stretch", column_config={
    "val_acc": st.column_config.NumberColumn("accuracy", format="%.2f%%"), "val_prec": st.column_config.NumberColumn("precision (macro)", format="%.2f%%"),
    "val_rec": st.column_config.NumberColumn("recall (macro)", format="%.2f%%"), "val_f1": st.column_config.NumberColumn("F1 (macro)", format="%.2f%%"),
    "val_loss": st.column_config.NumberColumn("log loss", format="%.4f"), "train_time_s": st.column_config.NumberColumn("training time (s)", format="%.1f"),
    "params": st.column_config.NumberColumn("parameters", format="%d")})
T.section("Final test results", "held-out 10,000 images — reported, never used for tuning")
t = df[["model", "test_acc", "test_prec", "test_rec", "test_f1", "ms_per_img", "latency_ms"]].copy()
for c in ("test_acc", "test_prec", "test_rec", "test_f1"):
    t[c] *= 100
st.dataframe(t, hide_index=True, width="stretch", column_config={
    "test_acc": st.column_config.NumberColumn("accuracy", format="%.2f%%"), "test_prec": st.column_config.NumberColumn("precision (macro)", format="%.2f%%"),
    "test_rec": st.column_config.NumberColumn("recall (macro)", format="%.2f%%"), "test_f1": st.column_config.NumberColumn("F1 (macro)", format="%.2f%%"),
    "ms_per_img": st.column_config.NumberColumn("inference ms / image (batched)", format="%.4f"),
    "latency_ms": st.column_config.NumberColumn("single-image latency (ms)", format="%.2f")})
st.download_button("Download comparison (CSV)", df.to_csv(index=False).encode(), "purf_model_comparison.csv", "text/csv",
                   icon=":material/download:")

# ------------------------------------------------------------------ charts
T.section("Visual comparison")
d = df.sort_values("val_acc")
a, b = st.columns(2)
with a:
    fig = go.Figure()
    fig.add_bar(y=d.model, x=(1 - d.val_acc) * 100, name="validation", orientation="h", marker=dict(color=T.SERIES[0], cornerradius=3),
                hovertemplate="%{y}<br>validation error %{x:.2f}%<extra></extra>")
    fig.add_bar(y=d.model, x=(1 - d.test_acc) * 100, name="test", orientation="h", marker=dict(color=T.SERIES[1], cornerradius=3),
                hovertemplate="%{y}<br>test error %{x:.2f}%<extra></extra>")
    fig.update_layout(barmode="group", xaxis_title="error rate (%) — lower is better", yaxis_title="", bargroupgap=0.1)
    C.plot(T.style_fig(fig, 420, "Error rate"), key="err")
with b:
    fig = go.Figure()
    for r in d.sort_values("val_acc", ascending=False).itertuples():
        is_prod = r.key == prod
        fig.add_scatter(x=[r.ms_per_img], y=[r.test_acc * 100 if not np.isnan(r.test_acc) else r.val_acc * 100],
                        mode="markers+text" if is_prod else "markers", text=[f"⚡ {MODEL_NAME}"] if is_prod else None,
                        textposition="bottom center", textfont=dict(color=T.TEXT, size=12),
                        marker=dict(size=14, color=T.model_color(r.key), line=dict(width=2, color=T.SURFACE)), name=r.model,
                        hovertemplate=f"{r.model}<br>%{{x:.4f}} ms/image<br>test accuracy %{{y:.2f}}%<extra></extra>")
    fig.update_layout(xaxis=dict(title="inference time per image (ms, log scale)", type="log"), yaxis_title="test accuracy (%)",
                      legend=dict(orientation="v", x=1.02, xanchor="left", y=1, yanchor="top", font=dict(size=11)),
                      margin=dict(r=10))
    C.plot(T.style_fig(fig, 420, "Accuracy vs speed"), key="tradeoff")
a, b = st.columns(2)
with a:
    fig = go.Figure(go.Bar(y=d.model, x=d.train_time_s, orientation="h", marker=dict(color=T.SERIES[0], cornerradius=3),
                           customdata=d.train_samples, hovertemplate="%{y}<br>%{x:.1f}s on %{customdata:,} images<extra></extra>"))
    fig.update_layout(xaxis=dict(title="training time (s, log scale)", type="log"), yaxis_title="")
    C.plot(T.style_fig(fig, 360, "Training time (CPU)"), key="ttime")
with b:
    metric = st.radio("Metric", ["test_f1", "test_prec", "test_rec"], horizontal=True,
                      format_func={"test_f1": "F1", "test_prec": "precision", "test_rec": "recall"}.get)
    fig = go.Figure(go.Bar(y=d.model, x=(1 - d[metric]) * 100, orientation="h", marker=dict(color=T.SERIES[2], cornerradius=3),
                           hovertemplate="%{y}<br>1 − score = %{x:.2f}%<extra></extra>"))
    fig.update_layout(xaxis_title=f"1 − macro {metric.split('_')[1]} on test (%) — lower is better", yaxis_title="")
    C.plot(T.style_fig(fig, 320, "Macro-averaged score gap"), key="macro")

# ------------------------------------------------------------------ duel
T.section("Head-to-head", "pick any two models")
keys = list(df.key)
c1, c2 = st.columns(2)
ka = c1.selectbox("Model A", keys, index=0, format_func=C.label)
kb = c2.selectbox("Model B", keys, index=min(len(keys) - 1, keys.index("logistic_regression") if "logistic_regression" in keys else 1), format_func=C.label)
ra, rb = df.set_index("key").loc[ka], df.set_index("key").loc[kb]
T.metric_row([
    T.metric_card("Δ test accuracy", f"{(ra.test_acc - rb.test_acc) * 100:+.2f} pp", "A − B", "", T.BRAND),
    T.metric_card("Δ test F1", f"{(ra.test_f1 - rb.test_f1) * 100:+.2f} pp", "A − B", "", T.BRAND_2),
    T.metric_card("Inference speed", f"{max(ra.ms_per_img, rb.ms_per_img) / min(ra.ms_per_img, rb.ms_per_img):.1f}×",
                  f"model {'A' if ra.ms_per_img < rb.ms_per_img else 'B'} is faster (batched)", ""),
    T.metric_card("Test errors", f"{int(round((1 - ra.test_acc) * 10000))} vs {int(round((1 - rb.test_acc) * 10000))}", "A vs B (of 10,000)", ""),
])
pa, pb = C.predictions(ka, "test"), C.predictions(kb, "test")
ma, mb = C.metrics(ka, "test"), C.metrics(kb, "test")
if pa and pb and ma and mb:
    l, r = st.columns([1.2, 1])
    with l:
        fa = [ma["classification_report"][str(i)]["f1-score"] * 100 for i in range(10)]
        fb = [mb["classification_report"][str(i)]["f1-score"] * 100 for i in range(10)]
        fig = go.Figure()
        fig.add_bar(x=[str(i) for i in range(10)], y=[100 - x for x in fa], name="A · " + C.label(ka), marker=dict(color=T.SERIES[0], cornerradius=3))
        fig.add_bar(x=[str(i) for i in range(10)], y=[100 - x for x in fb], name="B · " + C.label(kb), marker=dict(color=T.SERIES[1], cornerradius=3))
        fig.update_layout(barmode="group", xaxis_title="digit", yaxis_title="1 − F1 (%) — lower is better")
        C.plot(T.style_fig(fig, 360, "Per-digit F1 gap"), key="duel_f1")
    with r:
        ya = pa[0].argmax(1)
        yb = pb[0].argmax(1)
        y = pa[1]
        both = int(((ya == y) & (yb == y)).sum())
        only_a = np.where((ya == y) & (yb != y))[0]
        only_b = np.where((ya != y) & (yb == y))[0]
        neither = int(((ya != y) & (yb != y)).sum())
        st.markdown(T.card("Agreement on the test set",
                           f"Both correct: <b style='color:white'>{both:,}</b><br>Only A correct: <b style='color:white'>{len(only_a):,}</b>"
                           f"<br>Only B correct: <b style='color:white'>{len(only_b):,}</b><br>Both wrong: <b style='color:white'>{neither:,}</b>",
                           "🤝"), unsafe_allow_html=True)
        if len(only_a):
            st.caption("Digits only model A got right")
            sel = only_a[:8]
            T.image_grid([T.image_tile(splits.x_test[i], f"{y[i]} · B said {yb[i]}", upscale=3) for i in sel], columns=4)

T.section("Why convolutional networks suit images")
c = st.columns(3)
c[0].markdown(T.card("Local patterns", "Each 3×3 filter looks at a small neighbourhood, so the network learns stroke "
                     "edges, curves and line endings — the building blocks of digits.", "🔍"), unsafe_allow_html=True)
c[1].markdown(T.card("Weight sharing", "The same filter slides over the whole image: a loop is recognised wherever it "
                     "appears, with far fewer parameters than a dense layer on raw pixels.", "♻️"), unsafe_allow_html=True)
c[2].markdown(T.card("Spatial hierarchy", "Pooling builds tolerance to small shifts and stacks simple features into "
                     "complex ones. Flattened-pixel models (logistic regression, k-NN, SVM, ANN) see 784 unrelated numbers "
                     "and must learn spatial structure from scratch.", "🏗️"), unsafe_allow_html=True)
