"""Error Analysis — what does the model get wrong, and why?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.evaluate import top_confusions
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Insight",
    "Error Analysis",
    "Every misclassified image, with what the model believed. Confusion patterns below are measured from the "
    "model's actual predictions, not assumed.",
)
C.require_models()
splits = C.safe_splits()
c1, c2 = st.columns([1.6, 1])
key = c1.selectbox("Model", list(C.models()), format_func=C.label, key="err_model",
                   index=list(C.models()).index(C.mm.production_key()) if C.mm.production_key() in C.models() else 0)
split = c2.radio("Split", ["test", "val"], horizontal=True, format_func={"test": "Test", "val": "Validation"}.get)
pr = C.predictions(key, split)
if pr is None:
    st.info("Evaluate this model first (Model Evaluation page).")
    st.stop()
proba, y = pr
x = splits.x_test if split == "test" else splits.x_val
pred = proba.argmax(1)
conf = proba.max(1)
wrong = np.where(pred != y)[0]
m = C.metrics(key, split)

T.metric_row([
    T.metric_card("Misclassified", f"{len(wrong):,}", f"of {len(y):,} images ({len(wrong) / len(y) * 100:.2f}%)"),
    T.metric_card("Mean confidence when wrong", f"{conf[wrong].mean() * 100:.1f}%" if len(wrong) else "—", ""),
    T.metric_card("Mean confidence when right", f"{conf[pred == y].mean() * 100:.2f}%", ""),
    T.metric_card("Confidently wrong", f"{int((conf[wrong] > 0.9).sum())}", "errors with > 90% confidence"),
])

T.section("Which digits get confused", "measured from this model's predictions")
a, b = st.columns([1, 1.2])
with a:
    tc = top_confusions(m["confusion_matrix"], 10)
    if tc:
        lab = [f"{t['actual']} → {t['predicted']}" for t in tc][::-1]
        fig = go.Figure(go.Bar(y=lab, x=[t["count"] for t in tc][::-1], orientation="h", marker=dict(color=T.SERIES[0], cornerradius=3),
                               hovertemplate="actual %{y}<br>%{x} images<extra></extra>"))
        fig.update_layout(xaxis_title="number of errors", yaxis_title="actual → predicted")
        C.plot(T.style_fig(fig, 380, "Top confusion pairs"), key="pairs")
with b:
    cm = np.array(m["confusion_matrix"], dtype=float)
    np.fill_diagonal(cm, 0)
    fig = go.Figure(go.Heatmap(z=cm, x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
                               colorscale=[[i / (len(T.SEQ_BLUE) - 1), c] for i, c in enumerate(T.SEQ_BLUE)],
                               text=[[str(int(v)) if v else "" for v in r] for r in cm], texttemplate="%{text}", showscale=False,
                               xgap=2, ygap=2, hovertemplate="actual %{y} → predicted %{x}: %{z:.0f}<extra></extra>"))
    fig.update_layout(yaxis=dict(title="actual", autorange="reversed", showgrid=False, type="category"), xaxis=dict(title="predicted", showgrid=False, type="category"))
    C.plot(T.style_fig(fig, 380, "Errors only (diagonal removed)"), key="offdiag")
if tc:
    sym = {}
    for t in top_confusions(m["confusion_matrix"], 90):
        pair = tuple(sorted((t["actual"], t["predicted"])))
        sym[pair] = sym.get(pair, 0) + t["count"]
    top_pairs = sorted(sym.items(), key=lambda kv: -kv[1])[:3]
    st.markdown(T.card("Observed pattern",
                       "The most-confused pairs (both directions combined) are "
                       + ", ".join(f"<b style='color:var(--text)'>{p[0]}↔{p[1]}</b> ({n})" for p, n in top_pairs)
                       + ". Compare with the class-similarity analysis on the Dataset Explorer: confusions concentrate on "
                         "digits whose shapes overlap when written loosely."), unsafe_allow_html=True)

T.section("Error gallery", "filter the misclassified images")
f1, f2, f3, f4 = st.columns([1, 1, 1.6, 1])
act = f1.multiselect("Actual digit", list(range(10)))
prd = f2.multiselect("Predicted digit", list(range(10)))
lo, hi = f3.slider("Confidence of the wrong prediction (%)", 0, 100, (0, 100))
sort = f4.selectbox("Sort by", ["most confident", "least confident", "index"])
sel = wrong.copy()
if act:
    sel = sel[np.isin(y[sel], act)]
if prd:
    sel = sel[np.isin(pred[sel], prd)]
sel = sel[(conf[sel] * 100 >= lo) & (conf[sel] * 100 <= hi)]
if sort == "most confident":
    sel = sel[np.argsort(-conf[sel])]
elif sort == "least confident":
    sel = sel[np.argsort(conf[sel])]
st.caption(f"{len(sel)} matching images")
per_page = 24
pages = max(1, int(np.ceil(len(sel) / per_page)))
page = st.number_input("Page", 1, pages, 1) if pages > 1 else 1
show = sel[(page - 1) * per_page: page * per_page]
tiles = []
for i in show:
    order = np.argsort(-proba[i])
    alts = " · ".join(f"{d}: {proba[i][d] * 100:.1f}%" for d in order[1:3])
    tiles.append(T.image_tile(
        x[i], f"actual <b style='color:#8fd9b0'>{y[i]}</b> · predicted <b style='color:#f0a0a2'>{pred[i]}</b>"
              f"<br>{conf[i] * 100:.1f}% confident<br><span style='color:var(--muted)'>alt {alts}</span><br>#{i}",
        upscale=4, bad=True))
if tiles:
    T.image_grid(tiles, columns=6)
else:
    st.info("No errors match these filters.")

if len(show):
    T.section("Inspect one error")
    pick = st.selectbox("Image", list(show), format_func=lambda i: f"#{i} — actual {y[i]}, predicted {pred[i]}")
    a, b = st.columns([1, 2])
    with a:
        st.markdown(T.image_tile(x[pick], f"actual {y[pick]} · predicted {pred[pick]}", upscale=10, bad=True), unsafe_allow_html=True)
        if C.models()[key]["family"] == "keras" and st.button("Explain with XAI", icon=":material/visibility:"):
            st.session_state["xai_custom"] = {"image": x[pick].astype(np.float32) / 255.0, "label": int(y[pick]),
                                              "source": f"{split} image #{pick}"}
            st.session_state["xai_model"] = key
            st.switch_page("pages/8_Explainable_AI.py")
    with b:
        C.plot(C.prob_bar(proba[pick], 320, title="Class probabilities"), key="err_prob")

T.section("Error rate by confidence")
bins = [0, 50, 70, 90, 99, 100.0001]
cat = pd.cut(conf * 100, bins, right=False, labels=["<50%", "50–70%", "70–90%", "90–99%", "≥99%"])
dfc = pd.DataFrame({"bin": cat, "wrong": pred != y}).groupby("bin", observed=False).agg(n=("wrong", "size"), errors=("wrong", "sum"))
dfc["error rate %"] = dfc.errors / dfc.n.clip(lower=1) * 100
fig = go.Figure(go.Bar(x=dfc.index.astype(str), y=dfc["error rate %"], marker=dict(color=T.SERIES[0], cornerradius=3),
                       customdata=np.stack([dfc.n, dfc.errors], 1),
                       hovertemplate="confidence %{x}<br>error rate %{y:.1f}%<br>%{customdata[1]} errors / %{customdata[0]} images<extra></extra>"))
fig.update_layout(xaxis_title="model confidence", yaxis_title="error rate (%)")
C.plot(T.style_fig(fig, 300), key="errconf")
st.caption("Low-confidence predictions are far more likely to be wrong — confidence is a useful (imperfect) warning signal.")
