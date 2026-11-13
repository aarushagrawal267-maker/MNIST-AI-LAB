"""UPLOAD YOUR DIGIT — predict from a photo, scan or image file."""
from __future__ import annotations

import io

import numpy as np
import streamlit as st
from PIL import Image

from src.config import MODEL_NAME
from src.image_processor import ImageProcessingError
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Playground",
    "Upload Your Digit",
    f"Upload a picture of a single handwritten digit and {MODEL_NAME} will read it. The image is validated, "
    "converted to grayscale, cropped to the digit, size-normalised, centred and normalised exactly like MNIST.",
)
C.require_models()
splits = C.safe_splits()

with st.expander("What images work best?", expanded=False):
    st.markdown("""
- **Formats:** PNG, JPG or JPEG, up to 10 MB.
- **Content:** exactly **one** digit (0–9), filling a reasonable part of the frame.
- **Background:** plain and uniform — dark ink on white paper *or* white ink on black both work (polarity is
  detected automatically; override it below if needed).
- **Avoid:** lined/grid paper, shadows across the digit, several digits, or heavy text around it.
- Photos are fine: background lighting is removed and small specks of noise are discarded.
""")

c1, c2, c3 = st.columns([2, 1.2, 1.3])
with c1:
    up = st.file_uploader("Drop an image here", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
with c2:
    invert = st.radio("Polarity", ["auto", "yes", "no"], horizontal=True,
                      help="auto: detect from the image border · yes: dark ink on light paper · no: light ink on dark")
    model_key = C.model_select("Model", key="up_model")
with c3:
    st.markdown("**No image at hand?**")
    st.caption("Download a real MNIST test digit (rendered large on white paper) and upload it back.")
    i = int(np.random.default_rng(st.session_state.get("up_seed", 0)).integers(len(splits.y_test)))
    sample = Image.fromarray(255 - np.kron(splits.x_test[i], np.ones((12, 12), dtype=np.uint8)))
    buf = io.BytesIO()
    sample.save(buf, format="PNG")
    if st.download_button(f"Sample digit ({splits.y_test[i]})", buf.getvalue(), f"mnist_test_{i}_label{splits.y_test[i]}.png",
                          "image/png", icon=":material/download:"):
        st.session_state["up_seed"] = st.session_state.get("up_seed", 0) + 1

if up is None:
    st.markdown(
        "<div class='purf-card' style='text-align:center;padding:46px'><div style='font-size:52px;opacity:.55'>🖼️</div>"
        "<h4 style='margin-top:10px'>Upload an image to begin</h4><p>PNG · JPG · JPEG — one digit per image</p></div>",
        unsafe_allow_html=True)
    st.stop()

try:
    result = C.get_predictor(model_key).predict_image(up.getvalue(), invert=invert)
except ImageProcessingError as exc:
    st.error(f"**This image can't be used:** {exc}")
    st.stop()

p = result.processed
left, mid, right = st.columns([1.1, 1, 1.4], gap="large")
with left:
    T.section("Original")
    st.image(up.getvalue(), use_container_width=True)
    st.caption(f"{p.info['input_shape'][1]}×{p.info['input_shape'][0]} px · "
               f"{'inverted (dark ink on light background detected)' if p.inverted else 'kept (light ink on dark background)'}")
with mid:
    T.section("Processed 28×28")
    st.markdown(T.image_tile(p.image, "model input", upscale=10), unsafe_allow_html=True)
with right:
    T.section("Prediction")
    st.markdown(T.prediction_orb(result.digit, result.confidence, f"{result.certainty_label} · {C.label(model_key)}"),
                unsafe_allow_html=True)
    st.write("")
    st.markdown(f"<div class='purf-card'><h4>Top-3 predictions</h4><div style='height:6px'></div>{T.top3_bars(result.top3)}</div>",
                unsafe_allow_html=True)
for w in result.warnings:
    st.info(w, icon=":material/info:")
C.plot(C.prob_bar(result.probabilities, 300), key="up_prob")

T.section("Processing stages")
stages = [("grayscale", "Grayscale"), ("polarity", "Polarity"), ("contrast", "Contrast"), ("mask", "Digit mask"),
          ("cropped", "Cropped"), ("resized_20", "20 px"), ("final_28", "Final 28×28")]
tiles = []
for k_, t_ in stages:
    im = p.stages[k_]
    h, w = im.shape[:2]
    tiles.append(T.image_tile(im, f"<b>{t_}</b><br>{w}×{h}", upscale=max(1, int(200 / max(h, w)))))
T.image_grid(tiles, columns=7)
if st.button("Explain this prediction", icon=":material/visibility:"):
    st.session_state["xai_custom"] = {"image": p.image, "label": None, "source": f"uploaded file {up.name}"}
    st.switch_page("pages/8_Explainable_AI.py")
