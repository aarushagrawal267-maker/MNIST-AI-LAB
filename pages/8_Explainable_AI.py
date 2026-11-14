"""Explainable AI — which pixels drive a prediction?"""
from __future__ import annotations

import numpy as np
import streamlit as st

from src import explainability as X
from src.config import MODEL_NAME
from src.numpy_runtime import HAS_TENSORFLOW
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Insight",
    "Explainable AI",
    "Gradient- and perturbation-based attribution maps show which pixels the network's output is most sensitive to. "
    "They are diagnostic tools — useful for spotting shortcuts and odd behaviour — <b>not proof of how or why</b> the "
    "model reasons.",
)
C.require_models()
splits = C.safe_splits()
if not HAS_TENSORFLOW:
    st.info("TensorFlow is not available on this server's Python version, so gradient-based maps (saliency, "
            "SmoothGrad, Integrated Gradients, Grad-CAM) are disabled. Occlusion sensitivity — which needs no "
            "gradients — still works. Run the lab on Python 3.10–3.13 for every method.", icon=":material/info:")
ss = st.session_state
keras_keys = [k for k, v in C.models().items() if v["family"] == "keras"]
default = ss.get("xai_model") if ss.get("xai_model") in keras_keys else None
c1, c2 = st.columns([1.5, 1.5])
with c1:
    key = C.model_select("Model (neural networks only — gradients required)", keys=keras_keys, default=default, key="xai_m")
custom = ss.get("xai_custom")
sources = (["Sent from another page"] if custom else []) + ["Random test image", "Misclassified test image", "Test image by index"]
with c2:
    src = st.radio("Image source", sources, horizontal=True)

model = C.get_model(key)
pred = C.get_predictor(key)
label = None
if src == "Sent from another page":
    img = np.asarray(custom["image"], dtype=np.float32)
    label = custom.get("label")
    st.caption(f"Image from: {custom.get('source')}")
else:
    if st.button("New image", icon=":material/shuffle:"):
        ss["xai_seed"] = ss.get("xai_seed", 0) + 1
    rng = np.random.default_rng(ss.get("xai_seed", 3))
    if src == "Random test image":
        i = int(rng.integers(len(splits.y_test)))
    elif src == "Misclassified test image":
        pr = C.predictions(key, "test")
        wrong = np.where(pr[0].argmax(1) != pr[1])[0] if pr else np.array([0])
        i = int(rng.choice(wrong))
    else:
        i = int(st.number_input("Test index", 0, len(splits.y_test) - 1, 0))
    img = splits.x_test[i].astype(np.float32) / 255.0
    label = int(splits.y_test[i])
    st.caption(f"Test image #{i}")

proba = pred.predict_array(img)[0]
order = np.argsort(-proba)
a, b, c = st.columns([1, 1, 2])
with a:
    st.markdown(T.image_tile(img, "input (28×28)", upscale=8), unsafe_allow_html=True)
with b:
    st.markdown(T.prediction_orb(int(order[0]), float(proba[order[0]]), "predicted"), unsafe_allow_html=True)
    if label is not None:
        ok = label == int(order[0])
        st.markdown(f"<div style='text-align:center;margin-top:10px'>{T.badge(f'actual {label}', 'good' if ok else 'bad')}</div>",
                    unsafe_allow_html=True)
with c:
    C.plot(C.prob_bar(proba, 290), key="xai_prob")

target = st.selectbox("Explain the score of digit", list(range(10)), index=int(order[0]),
                      help="Default: the predicted digit. Pick another class to ask 'what would make it a …?'")


@st.cache_data(show_spinner="Computing attributions…", max_entries=64)
def _maps(model_key: str, img_bytes: bytes, target: int):
    im = np.frombuffer(img_bytes, dtype=np.float32).reshape(28, 28)
    mdl = C.get_model(model_key)
    sal = sg = ig = cam = None
    ig_err = float("nan")
    if HAS_TENSORFLOW:
        sal = X.saliency_map(mdl, im, target)
        sg = X.smoothgrad(mdl, im, target)
        ig, ig_err = X.integrated_gradients(mdl, im, target)
        cam = X.grad_cam(mdl, im, target)
    occ = X.occlusion_sensitivity(lambda z: C.get_predictor(model_key).predict_array(z), im, target)
    return sal, sg, ig, ig_err, cam, occ


sal, sg, ig, ig_err, cam, occ = _maps(key, img.astype(np.float32).tobytes(), int(target))
T.section("Attribution maps", f"for the class score (logit) of digit {target}")
maps = [
    ("Saliency", sal, False, "|∂logit/∂pixel| — pixels where a tiny change moves the class score most."),
    ("SmoothGrad", sg, False, "Saliency averaged over 32 noisy copies — less speckle."),
    ("Integrated Gradients", ig, True, f"Signed: red raises, blue lowers the score. Completeness error {ig_err * 100:.1f}%."),
    ("Grad-CAM", cam, False, f"Class-weighted activations of the last conv layer ({X.find_last_conv(model)}), upsampled."
     if cam is not None else ("Not available: this model has no convolutional layers." if HAS_TENSORFLOW
                              else "Requires TensorFlow (Python 3.10–3.13).")),
    ("Occlusion", occ, False, "Drop in probability when a 4×4 patch is blacked out (model-agnostic)."),
]
cols = st.columns(5)
for col, (name, mp, signed, desc) in zip(cols, maps):
    with col:
        if mp is None:
            st.markdown(T.card(name, desc, "🚫"), unsafe_allow_html=True)
        else:
            st.markdown(T.image_tile(X.colorize(mp, signed=signed), f"<b>{name}</b>", upscale=8, rgb=True), unsafe_allow_html=True)
        st.caption(desc)
st.caption("Colour scale: black → purple → orange → yellow = increasing attribution (each map rescaled to its own maximum). "
           "Integrated Gradients: blue = negative, dark = 0, red = positive.")

T.section("Overlays", "heatmap blended onto the digit")
ov = [("Saliency", sal), ("SmoothGrad", sg), ("Integrated Gradients |IG|", None if ig is None else np.abs(ig)),
      ("Grad-CAM", cam), ("Occlusion", occ)]
T.image_grid([T.image_tile(X.overlay(img, m_), n, upscale=8, rgb=True) for n, m_ in ov if m_ is not None], columns=5)

if cam is not None:
    T.section("Contrastive view", "Grad-CAM for the top-2 classes")
    second = int(order[1])
    cam2 = X.grad_cam(model, img, second)
    a, b, _, _ = st.columns(4)
    a.markdown(T.image_tile(X.overlay(img, cam), f"evidence for <b>{order[0]}</b> ({proba[order[0]] * 100:.2f}%)", upscale=10, rgb=True),
               unsafe_allow_html=True)
    b.markdown(T.image_tile(X.overlay(img, cam2), f"evidence for <b>{second}</b> ({proba[second] * 100:.2f}%)", upscale=10, rgb=True),
               unsafe_allow_html=True)

with st.expander("How to read these maps — and their limits"):
    st.markdown(f"""
- **Saliency / SmoothGrad** use the gradient of the class logit (pre-softmax score) with respect to each input
  pixel. Logits are used because a ~100%-confident softmax saturates and its gradients vanish. They show
  *sensitivity*, not importance in a causal sense, and are often noisy.
- **Integrated Gradients** accumulates gradients along a straight path from a black image to the input. Attributions
  approximately sum to `score(input) − score(black)`; the reported completeness error measures how good that
  approximation is (64 steps, trapezoidal rule).
- **Grad-CAM** weights the last convolutional feature maps by their pooled gradients. On 28×28 inputs the last conv
  layer is low-resolution (≈3–11 px), so Grad-CAM is coarse by construction.
- **Occlusion** needs no gradients at all — it simply measures how the probability drops when part of the digit
  is hidden.
- Maps are rescaled to [0, 1] for display, so compare *shapes*, not absolute brightness, across images.
- None of these methods proves *why* {MODEL_NAME} made a decision; they highlight where its output is sensitive.
""")
