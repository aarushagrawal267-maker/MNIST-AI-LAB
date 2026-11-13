"""Image Preprocessing — tensors for training, and the real-world digit pipeline."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODEL_NAME
from src.image_processor import ImageProcessingError, naive_resize, preprocess_digit
from src.preprocessing import describe, normalize, one_hot, to_cnn_input, to_flat_input
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Data",
    "Image Preprocessing",
    "Two pipelines live here: the tensor preparation applied to MNIST before training, and the real-world pipeline "
    "that turns canvas drawings, photos and scans into MNIST-format digits.",
)
splits = C.safe_splits()
tab1, tab2 = st.tabs(["MNIST → model tensors", "Real-world image → MNIST format"])

with tab1:
    c1, c2 = st.columns([1, 3])
    idx = c1.number_input("Training image index", 0, len(splits.y_train) - 1, 7)
    raw = splits.x_train[idx]
    lab = int(splits.y_train[idx])
    norm = normalize(raw)
    cnn = to_cnn_input(raw)
    flat = to_flat_input(raw)
    with c2:
        st.markdown(
            f"Label **{lab}**. Pixel values are scaled from 0–255 to 0–1 (`x / 255`), then reshaped to "
            "`(N, 28, 28, 1)` for convolutional networks or flattened to `(N, 784)` for classical models.")
    cols = st.columns(3)
    for col, (title, arr, scale) in zip(cols, [("Original (uint8)", raw, raw), ("Normalised (float32)", norm, norm),
                                               ("CNN tensor (28×28×1)", cnn[0, ..., 0], cnn[0, ..., 0])]):
        with col:
            C.plot(C.heat_image(arr, "gray", 300, title=title), key=f"pp_{title}")
            d = describe(arr if "CNN" not in title else cnn)
            st.caption(f"shape {d['shape']} · {d['dtype']} · min {d['min']:.3g} · max {d['max']:.3g} · mean {d['mean']:.3g}")
    T.section("Flattened vector & label encoding")
    a, b = st.columns([2, 1])
    with a:
        import plotly.graph_objects as go

        fig = go.Figure(go.Scatter(y=flat[0], mode="lines", line=dict(color=T.SERIES[0], width=1.5),
                                   hovertemplate="pixel %{x}<br>%{y:.3f}<extra></extra>"))
        fig.update_layout(xaxis_title="pixel index (row-major)", yaxis_title="value", showlegend=False)
        C.plot(T.style_fig(fig, 260, f"784-dimensional input for logistic regression / k-NN / SVM"), key="flat")
    with b:
        oh = one_hot(np.array([lab]))[0].astype(int)
        st.dataframe(pd.DataFrame([oh], columns=[str(i) for i in range(10)], index=["one-hot"]), use_container_width=True)
        st.caption("Neural networks here use **sparse categorical cross-entropy**, so the integer label is used directly; "
                   "the equivalent one-hot vector is shown for reference.")

with tab2:
    st.markdown(
        "Upload any photo or scan of a single handwritten digit — or use the generated example, a small, "
        "off-centre digit written in dark ink on unevenly lit paper with a stray ink speck.")
    src_col, opt_col = st.columns([2, 1])
    up = src_col.file_uploader("Image (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"], key="pp_up")
    invert = opt_col.radio("Polarity", ["auto", "yes", "no"], horizontal=True,
                           help="auto: detect from the background. yes: dark ink on light paper. no: light ink on dark.")

    @st.cache_data
    def synthetic_photo(seed: int = 3) -> np.ndarray:
        rng = np.random.default_rng(seed)
        h, w = 360, 480
        yy, xx = np.mgrid[0:h, 0:w]
        paper = 225 - 35 * (xx / w) - 15 * (yy / h) + rng.normal(0, 4, (h, w))
        img = np.clip(paper, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.putText(img, "3", (300, 250), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4.2, (40, 40, 55), 9, cv2.LINE_AA)
        cv2.circle(img, (60, 70), 3, (60, 60, 60), -1)
        return img

    source = up.getvalue() if up else synthetic_photo()
    try:
        proc = preprocess_digit(source, invert=invert)
        naive = naive_resize(source)
    except ImageProcessingError as exc:
        st.error(str(exc))
        st.stop()

    order = [("original", "Original"), ("grayscale", "Grayscale"), ("polarity", "Polarity → light on dark"),
             ("contrast", "Background removed + contrast"), ("mask", "Digit mask (noise removed)"),
             ("cropped", "Cropped to digit"), ("resized_20", "Resized to 20 px"), ("boxed_28", "Placed on 28×28"),
             ("final_28", "Centred by mass (final)")]
    tiles = []
    for k, t in order:
        img = proc.stages[k]
        if img.ndim == 3:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
        h, w = img.shape[:2]
        tiles.append(T.image_tile(img, f"<b>{t}</b><br>{w}×{h}", upscale=max(1, int(200 / max(h, w)))))
    T.image_grid(tiles, columns=9)
    for wmsg in proc.warnings:
        st.info(wmsg)

    T.section("Why naive resizing fails", f"both inputs scored by {MODEL_NAME}")
    pred = C.get_predictor()
    if pred is not None:
        p_naive = pred.predict_array(naive)[0]
        p_ours = pred.predict_array(proc.image)[0]
        a, b = st.columns(2)
        for col, (title, img, p) in zip((a, b), [("Naive: whole image squashed to 28×28", naive, p_naive),
                                                 ("PURF pipeline: crop → 20 px → centre by mass", proc.image, p_ours)]):
            with col:
                i1, i2 = st.columns([1, 1.6])
                i1.markdown(T.image_tile(img, title, upscale=6), unsafe_allow_html=True)
                with i2:
                    st.markdown(T.prediction_orb(int(p.argmax()), float(p.max()), "prediction"), unsafe_allow_html=True)
                C.plot(C.prob_bar(p, 240, title=None), key=f"cmp_{title[:5]}")
        st.caption("The naive version keeps the empty margins, so the digit shrinks to a few pixels and sits off-centre — "
                   "far from anything the model saw during training.")
    with st.expander("Pipeline details"):
        st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in proc.info.items()} | {"inverted": proc.inverted, "bbox_xywh": proc.bbox})
