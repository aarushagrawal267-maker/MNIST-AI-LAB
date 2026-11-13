"""DRAW YOUR DIGIT — the interactive heart of the lab."""
from __future__ import annotations

import numpy as np
import streamlit as st

from src.config import MODEL_NAME
from src.drawing_canvas import decode_data_url, draw_canvas
from src.image_processor import ImageProcessingError
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Playground",
    "Draw Your Digit",
    f"Draw any digit from 0 to 9 with your mouse, touchpad or pen. Your sketch is cropped, size-normalised, "
    f"centred by mass and converted to a 28×28 image exactly like MNIST, then {MODEL_NAME} returns a probability "
    f"for every digit. All numbers below come straight from the model's softmax output.",
)
C.require_models()

ss = st.session_state
ss.setdefault("draw_history", [])
ss.setdefault("draw_last", None)
ss.setdefault("draw_uid", 0)

ctl1, ctl2, ctl3 = st.columns([1.6, 1, 1.4])
with ctl1:
    model_key = C.model_select("Model", key="draw_model")
with ctl2:
    live = st.toggle("Live prediction", value=True, help="Predict automatically after every stroke.")
with ctl3:
    st.caption("Tips: draw large and centred · one digit at a time · `Enter` predicts, `Ctrl+Z` undoes, `Esc` clears.")

left, right = st.columns([1.05, 1.25], gap="large")
with left:
    event = draw_canvas(key="purf_canvas", brush=22, live=live, size=400)

# --------------------------------------------------------------- inference
result, error = None, None
if event and not event.get("empty") and event.get("image"):
    is_predict = event.get("action") == "predict"
    if is_predict or (live and event.get("action") in ("stroke", "undo", "redo")):
        cache_key = (event["nonce"], model_key)
        last = ss.draw_last
        if last and last["cache_key"] == cache_key:
            result = last["result"]
        else:
            try:
                rgba = decode_data_url(event["image"])
                result = C.get_predictor(model_key).predict_image(rgba)
                ss.draw_last = {"cache_key": cache_key, "result": result, "rgba": rgba}
                if is_predict:
                    ss.draw_uid += 1
                    ss.draw_history.insert(0, {
                        "uid": ss.draw_uid, "img": result.processed.image, "digit": result.digit, "conf": result.confidence,
                        "model": C.label(model_key), "feedback": None,
                    })
                    ss.draw_history = ss.draw_history[:24]
            except ImageProcessingError as exc:
                error = str(exc)
    elif ss.draw_last:
        result = ss.draw_last["result"]

with right:
    if error:
        st.warning(error)
    if result is None:
        st.markdown(
            f"""<div class="purf-card" style="min-height:420px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center">
            <div style="font-size:64px;opacity:.5">✍️</div>
            <h4 style="margin-top:12px">Waiting for your handwriting</h4>
            <p>Draw a digit on the canvas{' — predictions update live' if live else ' and press <b>Predict</b>'}.</p></div>""",
            unsafe_allow_html=True,
        )
    else:
        r = result
        o1, o2 = st.columns([1, 1.15])
        with o1:
            st.markdown(T.prediction_orb(r.digit, r.confidence, f"{r.certainty_label} · {C.label(model_key)}"),
                        unsafe_allow_html=True)
        with o2:
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='purf-card'><h4>Top-3 predictions</h4><div style='height:8px'></div>"
                        f"{T.top3_bars(r.top3)}</div>", unsafe_allow_html=True)
            st.write("")
            st.markdown(
                f"<div class='purf-card' style='padding:14px 18px'><p>Inference <b style='color:white'>{r.latency_ms:.1f} ms</b>"
                f" · 28×28 input · processed on CPU</p></div>", unsafe_allow_html=True)
        C.plot(C.prob_bar(r.probabilities, height=270), key="draw_prob")
        for w in r.warnings:
            st.info(w, icon=":material/info:")

# --------------------------------------------------------------- pipeline view
if result is not None and result.processed is not None:
    p = result.processed
    T.section("From your drawing to MNIST format", "every stage of the preprocessing pipeline")
    stage_info = [
        ("original", "1 · Original canvas", "your drawing (RGBA, 336×336)"),
        ("grayscale", "2 · Grayscale", "luminance only"),
        ("mask", "3 · Digit detection", "Otsu threshold + connected components"),
        ("cropped", "4 · Cropped", f"bounding box {p.bbox[2]}×{p.bbox[3]} px"),
        ("resized_20", "5 · Resized", "longest side → 20 px, aspect kept"),
        ("boxed_28", "6 · Placed 28×28", "pasted on a black frame"),
        ("final_28", "7 · Centred by mass", "matches MNIST's centring"),
    ]
    tiles = []
    for k, title, cap in stage_info:
        img = p.stages[k]
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2).astype(np.uint8)
        h, w = img.shape[:2]
        up = max(1, int(round(200 / max(h, w))))
        tiles.append(T.image_tile(img, f"<b>{title}</b><br>{cap}", upscale=up))
    T.image_grid(tiles, columns=7)
    with st.expander("Model input tensor (28×28 pixel values, 0–255 scale)"):
        C.plot(C.digit_heatmap(p.stages["final_28"], height=520), key="draw_tensor")
        st.caption(f"Shape (1, 28, 28, 1) · float32 · range [0, 1] · ink fraction {p.info['ink_fraction'] * 100:.1f}% of canvas · "
                   f"polarity {'inverted' if p.inverted else 'kept (light ink on dark)'}")
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Explain this prediction", icon=":material/visibility:"):
        ss["xai_custom"] = {"image": p.image, "label": None, "source": "your drawing"}
        st.switch_page("pages/8_Explainable_AI.py")

# --------------------------------------------------------------- history
if ss.draw_history:
    T.section("Your session", "predictions made with the Predict button")
    fb = [h["feedback"] for h in ss.draw_history if h["feedback"] is not None]
    if fb:
        st.caption(f"You marked {len(fb)} prediction(s): {sum(fb)} correct · personal accuracy "
                   f"{sum(fb) / len(fb) * 100:.0f}% — your handwriting, not MNIST, so this differs from test accuracy.")
    cols = st.columns(8)
    for i, h in enumerate(ss.draw_history[:16]):
        with cols[i % 8]:
            tag = "" if h["feedback"] is None else (" ✓" if h["feedback"] else " ✗")
            st.markdown(T.image_tile(h["img"], f"<b style='color:white'>{h['digit']}</b> · {h['conf'] * 100:.1f}%{tag}", upscale=3,
                                     bad=h["feedback"] is False), unsafe_allow_html=True)
            if h["feedback"] is None:
                a, b = st.columns(2)
                if a.button("✓", key=f"ok_{h['uid']}", help="Prediction was correct", use_container_width=True):
                    h["feedback"] = True
                    st.rerun()
                if b.button("✗", key=f"no_{h['uid']}", help="Prediction was wrong", use_container_width=True):
                    h["feedback"] = False
                    st.rerun()
    if st.button("Clear history", icon=":material/delete_sweep:"):
        ss.draw_history = []
        st.rerun()

with st.expander("Why the drawing needs careful preprocessing"):
    st.markdown(
        """
- **MNIST format** — every training digit is light-on-dark, fits a 20×20 box inside a 28×28 frame, and is centred
  by its centre of mass. A model only sees data in that format, so user drawings must be converted to it.
- **Naive resizing fails** — squashing a 336×336 canvas straight down to 28×28 keeps empty margins, shrinks small
  drawings to a few pixels and leaves off-centre digits off-centre. Compare both approaches on the
  *Image Preprocessing* page.
- **Limitations** — the model was trained only on MNIST-style handwriting. Unusual styles (e.g. a crossed 7, a
  very open 4, stylised fonts) or several digits at once can be misread. Confidence is the model's softmax output,
  not a guarantee of correctness.
"""
    )
