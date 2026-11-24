"""Home dashboard."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import model_manager as mm
from src.config import LAB_NAME, MODEL_NAME
from src.ui import common as C
from src.ui import theme as T

ms = C.models(include_custom=False)
prod = mm.production_key()
prod_entry = ms.get(prod, {}) if prod else {}
val_acc = prod_entry.get("val_accuracy")
test_m = C.metrics(prod, "test") if prod else None
stats = C.eda_stats()

status = (f'<span class="dot"></span>{MODEL_NAME} online · {T.esc(prod_entry.get("display_name", ""))}'
          if prod else f'<span class="dot"></span>{MODEL_NAME} awaiting training')
st.markdown(
    f"""
<div class="purf-hero">
  <div class="purf-kicker">{status}</div>
  <div class="purf-title">{LAB_NAME}<br/>reads your handwriting.</div>
  <div class="purf-sub">An end-to-end deep-learning laboratory for handwritten digit recognition. Explore 70,000
  MNIST digits, train and compare five families of models, look inside the network with explainable AI — then draw a
  digit and watch <b style="color:white">{MODEL_NAME}</b>, the lab's convolutional neural network, recognise it in milliseconds.</div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")
c1, c2, c3, _ = st.columns([1.2, 1.2, 1.2, 2])
c1.page_link("pages/9_Draw_Your_Digit.py", label="Draw a digit", icon=":material/draw:")
c2.page_link("pages/1_Dataset_Explorer.py", label="Explore the data", icon=":material/dataset:")
c3.page_link("pages/5_Model_Comparison.py", label="Compare models", icon=":material/compare_arrows:")

# ---------------------------------------------------------------- metrics
T.section("Lab at a glance")
n_total = stats["n_total"] if stats else 70000
T.metric_row([
    T.metric_card("Dataset", f"{n_total:,}", "MNIST images · Kaggle", "🗂️", T.SERIES[0]),
    T.metric_card("Classes", "10", "digits 0 – 9", "🔢", T.SERIES[2]),
    T.metric_card("Models trained", str(len(ms)), "5 model families", "🧪", T.SERIES[3]),
    T.metric_card("Validation acc.", f"{val_acc * 100:.2f}%" if val_acc else "—", f"{MODEL_NAME} · model selection", "🎯", T.BRAND),
    T.metric_card("Test acc.", f"{test_m['accuracy'] * 100:.2f}%" if test_m else "—",
                  "held-out 10,000 images" if test_m else "run final evaluation", "🏁", T.BRAND_2),
])

# ---------------------------------------------------------------- leaderboard
if ms:
    T.section("Model leaderboard", "validation error rate — lower is better")
    rows = sorted(ms.items(), key=lambda kv: kv[1]["val_accuracy"])
    names = [v["display_name"] + (f"  ⚡ {MODEL_NAME}" if k == prod else "") for k, v in rows]
    err = [(1 - v["val_accuracy"]) * 100 for _, v in rows]
    test_acc = [(C.metrics(k, "test") or {}).get("accuracy") for k, _ in rows]
    colors = [T.BRAND_2 if k == prod else T.SERIES[0] for k, _ in rows]
    fig = go.Figure(go.Bar(
        y=names, x=err, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[f"{e:.2f}%" for e in err], textposition="outside", textfont=dict(color=T.TEXT_2), cliponaxis=False,
        customdata=[[(1 - e / 100) * 100, (t or 0) * 100] for e, t in zip(err, test_acc)],
        hovertemplate="%{y}<br>validation error %{x:.2f}%<br>validation accuracy %{customdata[0]:.2f}%"
                      "<br>test accuracy %{customdata[1]:.2f}%<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(title="validation error rate (%)", range=[0, max(err) * 1.18]), yaxis=dict(title=""),
                      showlegend=False)
    left, right = st.columns([1.55, 1])
    with left:
        C.plot(T.style_fig(fig, height=360, title=None, margin=dict(l=10, r=30, t=10, b=40)))
    with right:
        best_classical = min((v for k, v in ms.items() if v["family"] == "sklearn"),
                             key=lambda v: 1 - v["val_accuracy"], default=None)
        if best_classical and val_acc:
            reduction = (1 - (1 - val_acc) / (1 - best_classical["val_accuracy"])) * 100
            st.markdown(T.card(
                f"Why {MODEL_NAME} wins",
                f"Convolutions share weights across the image and learn local stroke detectors, so {MODEL_NAME} "
                f"makes <b style='color:white'>{reduction:.0f}% fewer validation errors</b> than the best classical "
                f"model ({T.esc(best_classical['display_name'])}).", "🧠"), unsafe_allow_html=True)
            st.write("")
        st.markdown(T.card(
            "Honest evaluation",
            "Models are selected on a stratified 6,000-image validation split. The original 10,000-image test set "
            "is never used for tuning — it is only scored once, for the final report.", "⚖️"), unsafe_allow_html=True)

# ---------------------------------------------------------------- live strip
if prod:
    T.section(f"{MODEL_NAME}, live", "random digits from the held-out test set")
    splits = C.safe_splits()
    if "home_seed" not in st.session_state:
        st.session_state.home_seed = 7
    if st.button("Shuffle digits", icon=":material/shuffle:"):
        st.session_state.home_seed += 1
    rng = np.random.default_rng(st.session_state.home_seed)
    idx = rng.choice(len(splits.x_test), 12, replace=False)
    pred = C.get_predictor()
    proba = pred.predict_array(splits.x_test[idx])
    tiles = []
    for i, p in zip(idx, proba):
        d, ok = int(p.argmax()), int(p.argmax()) == int(splits.y_test[i])
        tiles.append(T.image_tile(
            splits.x_test[i],
            f"<b style='color:white;font-size:1rem'>{d}</b> · {p.max() * 100:.1f}%<br>"
            + (T.badge("correct", "good") if ok else T.badge(f"label {splits.y_test[i]}", "bad")),
            upscale=4, bad=not ok))
    T.image_grid(tiles, columns=12 if len(tiles) == 12 else 6)

# ---------------------------------------------------------------- pipeline
T.section("How the lab works")
steps = [
    ("01", "Acquire", "MNIST IDX files from Kaggle, cached locally and validated."),
    ("02", "Explore", "Class balance, pixel statistics, handwriting variation."),
    ("03", "Preprocess", "Normalise, reshape; canvas digits re-centred like MNIST."),
    ("04", "Augment", "Small rotations, shifts and zoom for robustness."),
    ("05", "Train", "Logistic regression, k-NN, SVM, ANN and CNNs."),
    ("06", "Tune", "Controlled random search, tracked experiment by experiment."),
    ("07", "Evaluate", "Accuracy, F1, confusion, ROC & PR on held-out data."),
    ("08", "Explain", "Saliency, Integrated Gradients, Grad-CAM, occlusion."),
]
st.markdown('<div class="purf-steps">' + "".join(
    f'<div class="purf-step"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
    for n, t, d in steps) + "</div>", unsafe_allow_html=True)

T.section("Jump in")
links = [
    ("pages/9_Draw_Your_Digit.py", "Draw", ":material/draw:"),
    ("pages/10_Upload_Your_Digit.py", "Upload", ":material/upload_file:"),
    ("pages/11_Batch_Prediction.py", "Batch", ":material/stacks:"),
    ("pages/4_Model_Training.py", "Train", ":material/model_training:"),
    ("pages/7_Error_Analysis.py", "Errors", ":material/bug_report:"),
    ("pages/8_Explainable_AI.py", "Explain", ":material/visibility:"),
]
cols = st.columns(6)
for c, (path, lbl, icon) in zip(cols, links):
    c.page_link(path, label=lbl, icon=icon, width="stretch")
