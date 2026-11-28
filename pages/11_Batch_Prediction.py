"""Batch Prediction — many images (or ZIP archives) at once, with CSV export."""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.config import MODEL_NAME
from src.prediction import expand_uploads
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Playground",
    "Batch Prediction",
    "Upload many digit images — or ZIP archives of them — and get a prediction, confidence and runner-up for each, "
    "ready to export as CSV. Invalid files are skipped and reported, never crash the batch.",
)
C.require_models()
splits = C.safe_splits()

c1, c2 = st.columns([2, 1.2])
with c1:
    files = st.file_uploader("Images or ZIP archives", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True)
with c2:
    model_key = C.model_select("Model", key="batch_model")
    invert = st.radio("Polarity", ["auto", "yes", "no"], horizontal=True)
    st.caption("Need test data? Download a ZIP of 24 real MNIST test digits:")

    @st.cache_data
    def demo_zip(seed: int = 0) -> bytes:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(splits.y_test), 24, replace=False)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for k, i in enumerate(idx):
                img = 255 - np.kron(splits.x_test[i], np.ones((8, 8), dtype=np.uint8))
                b = io.BytesIO()
                Image.fromarray(img).save(b, format="PNG")
                zf.writestr(f"digits/{k:02d}_label{splits.y_test[i]}.png", b.getvalue())
            zf.writestr("digits/readme.txt", "not an image - will be reported as unsupported")
        return buf.getvalue()

    st.download_button("Demo ZIP (24 digits)", demo_zip(), "purf_demo_digits.zip", "application/zip", icon=":material/folder_zip:")

if not files:
    st.markdown(
        f"<div class='purf-card purf-empty' style='padding:48px'>{T.blank28('28×28 · empty')}"
        "<h4 style='margin:0'>Upload several images or a ZIP to begin</h4><p>PNG · JPG · JPEG · ZIP (max 500 files, 50 MB extracted)</p></div>",
        unsafe_allow_html=True)
    st.stop()

items, rejected = expand_uploads([(f.name, f.getvalue()) for f in files])
with st.spinner(f"{MODEL_NAME} is reading {len(items)} image(s)…"):
    rows = C.get_predictor(model_key).predict_batch(items, invert=invert) if items else []
rows += rejected
ok = [r for r in rows if r["status"] == "ok"]
bad = [r for r in rows if r["status"] != "ok"]

T.metric_row([
    T.metric_card("Files received", str(len(rows)), "after ZIP expansion"),
    T.metric_card("Predicted", str(len(ok)), "valid images"),
    T.metric_card("Skipped", str(len(bad)), "invalid / unsupported"),
    T.metric_card("Mean confidence", f"{np.mean([r['confidence'] for r in ok]) * 100:.1f}%" if ok else "—", ""),
])

if ok:
    T.section("Results")
    df = pd.DataFrame([{
        "preview": T.png_data_uri(r["processed"], 2), "file": r["file"], "prediction": r["prediction"],
        "confidence": r["confidence"], "runner-up": r["second_choice"], "runner-up prob.": r["second_prob"],
        "certainty": r["certainty"], "warnings": r["warnings"],
    } for r in ok])
    st.dataframe(df, hide_index=True, use_container_width=True, row_height=58, column_config={
        "preview": st.column_config.ImageColumn("28×28 input", width="small"),
        "confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.4f"),
        "runner-up prob.": st.column_config.NumberColumn(format="%.4f"),
    })
    export = pd.DataFrame([{
        "file": r["file"], "status": r["status"], "prediction": r.get("prediction"), "confidence": r.get("confidence"),
        "second_choice": r.get("second_choice"), "second_prob": r.get("second_prob"),
        "third_choice": r.get("third_choice"), "third_prob": r.get("third_prob"), "certainty": r.get("certainty"),
        "warnings": r.get("warnings", ""), "error": r.get("error", ""),
        **({f"p_{d}": float(r["probabilities"][d]) for d in range(10)} if r["status"] == "ok" else {}),
    } for r in rows])
    st.download_button("Export results (CSV)", export.to_csv(index=False).encode(), "purf_batch_predictions.csv", "text/csv",
                       type="primary", icon=":material/download:")
    a, b = st.columns(2)
    with a:
        counts = np.bincount([r["prediction"] for r in ok], minlength=10)
        fig = go.Figure(go.Bar(x=[str(i) for i in range(10)], y=counts, marker=dict(color=T.SERIES[0], cornerradius=3),
                               hovertemplate="digit %{x}: %{y} images<extra></extra>"))
        fig.update_layout(xaxis=dict(title="predicted digit", type="category"), yaxis_title="images")
        C.plot(T.style_fig(fig, 300, "Predicted digits"), key="b_counts")
    with b:
        fig = go.Figure(go.Histogram(x=[r["confidence"] * 100 for r in ok], nbinsx=20,
                                     marker=dict(color=T.SERIES[0], line=dict(width=1, color=T.SURFACE)),
                                     hovertemplate="%{x}%: %{y} images<extra></extra>"))
        fig.update_layout(xaxis_title="confidence (%)", yaxis_title="images")
        C.plot(T.style_fig(fig, 300, "Confidence distribution"), key="b_conf")
    T.section("Gallery")
    T.image_grid([T.image_tile(r["processed"], f"<b style='color:var(--text)'>{r['prediction']}</b> · {r['confidence'] * 100:.1f}%"
                               f"<br><span style='color:#7f89b8'>{T.esc(r['file'].split('/')[-1][:22])}</span>", upscale=3)
                  for r in ok[:48]], columns=8)

if bad:
    T.section("Skipped files")
    st.dataframe(pd.DataFrame([{"file": r["file"], "reason": r["error"]} for r in bad]), hide_index=True, use_container_width=True)
