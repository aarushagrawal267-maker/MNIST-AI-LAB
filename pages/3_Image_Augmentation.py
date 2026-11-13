"""Image Augmentation — interactive preview and the measured effect on validation performance."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.augmentation import augment_images
from src.config import CFG
from src.train import load_history
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Data",
    "Image Augmentation",
    "Augmentation shows the network slightly rotated, shifted and zoomed copies of each digit during training. "
    "Only label-preserving transforms are used — no flips (a mirrored 2 or 5 is not a digit) and no large rotations "
    "(a 6 rotated 180° becomes a 9).",
)
splits = C.safe_splits()
aug = CFG["augmentation"]

c = st.columns(4)
rot = c[0].slider("Rotation (± degrees)", 0, 45, int(aug["rotation_degrees"]))
tr = c[1].slider("Translation (± fraction)", 0.0, 0.3, float(aug["translation"]), 0.01)
zm = c[2].slider("Zoom (± fraction)", 0.0, 0.4, float(aug["zoom"]), 0.01)
digit = c[3].selectbox("Digit", list(range(10)), index=7)
if rot > 20 or tr > 0.15 or zm > 0.25:
    st.warning("Strong settings can push strokes out of the frame or change identity (6↔9). "
               f"Training uses ±{aug['rotation_degrees']}°, ±{aug['translation']:.0%} shift, ±{aug['zoom']:.0%} zoom.")

if st.button("Regenerate", icon=":material/refresh:"):
    st.session_state["aug_seed"] = st.session_state.get("aug_seed", 0) + 1
seed = st.session_state.get("aug_seed", 0)
rng = np.random.default_rng(seed)
ids = rng.choice(np.where(splits.y_train == digit)[0], 4, replace=False)


@st.cache_data(show_spinner="Augmenting…")
def _augment(images, rot, tr, zm, seed):
    return augment_images(images, n_variants=9, rotation_degrees=rot, translation=tr, zoom=zm, seed=seed + 1)


out = _augment(splits.x_train[ids], rot, tr, zm, seed)
for row, i in enumerate(ids):
    tiles = [T.image_tile(splits.x_train[i], "<b>original</b>", upscale=3)]
    tiles += [T.image_tile(out[row * 9 + k], f"variant {k + 1}", upscale=3) for k in range(9)]
    T.image_grid(tiles, columns=10)
    st.write("")

# --------------------------------------------------------------- measured effect
T.section("Does augmentation help?", "same tuned architecture, trained with and without augmentation")
ms = C.models(include_custom=False)
if "cnn_tuned" in ms and "cnn_tuned_aug" in ms:
    a, b = ms["cnn_tuned"], ms["cnn_tuned_aug"]
    ta, tb = C.metrics("cnn_tuned", "test"), C.metrics("cnn_tuned_aug", "test")
    T.metric_row([
        T.metric_card("Val. accuracy · no aug", f"{a['val_accuracy'] * 100:.2f}%", f"loss {a['val_loss']:.4f} · {a['epochs_trained']} epochs", "", T.SERIES[0]),
        T.metric_card("Val. accuracy · augmented", f"{b['val_accuracy'] * 100:.2f}%", f"loss {b['val_loss']:.4f} · {b['epochs_trained']} epochs", "", T.SERIES[1]),
        T.metric_card("Δ validation accuracy", f"{(b['val_accuracy'] - a['val_accuracy']) * 100:+.2f} pp",
                      "augmented − baseline", "", T.BRAND),
        T.metric_card("Training time", f"{a['train_time_s'] / 60:.1f} → {b['train_time_s'] / 60:.1f} min", "no aug → augmented", "", T.BRAND_2),
    ])
    ha, hb = load_history("cnn_tuned"), load_history("cnn_tuned_aug")
    if ha and hb:
        l, r = st.columns(2)
        for col, metric, ttl in ((l, "val_accuracy", "Validation accuracy"), (r, "val_loss", "Validation loss")):
            fig = go.Figure()
            fig.add_scatter(y=ha[metric], x=list(range(1, len(ha[metric]) + 1)), name="no augmentation", mode="lines+markers",
                            line=dict(color=T.SERIES[0], width=2), marker=dict(size=7))
            fig.add_scatter(y=hb[metric], x=list(range(1, len(hb[metric]) + 1)), name="augmented", mode="lines+markers",
                            line=dict(color=T.SERIES[1], width=2), marker=dict(size=7))
            fig.update_layout(xaxis=dict(title="epoch", dtick=1), yaxis_title=metric.replace("_", " "), hovermode="x unified")
            with col:
                C.plot(T.style_fig(fig, 320, ttl), key=metric)
        gap_a = ha["accuracy"][-1] - ha["val_accuracy"][-1]
        gap_b = hb["accuracy"][-1] - hb["val_accuracy"][-1]
        better = b["val_accuracy"] > a["val_accuracy"]
        st.markdown(T.card(
            "Verdict (from the measured results)",
            f"With augmentation the validation accuracy is <b style='color:white'>{b['val_accuracy'] * 100:.2f}%</b> "
            f"versus {a['val_accuracy'] * 100:.2f}% without — augmentation {'improved' if better else 'did not improve'} "
            f"validation performance in this run. Final-epoch train−validation accuracy gap: {gap_a * 100:+.2f} pp "
            f"(no aug) vs {gap_b * 100:+.2f} pp (augmented); training accuracy on augmented images is naturally lower "
            "because those images are harder. "
            + (f"Held-out test accuracy: {ta['accuracy'] * 100:.2f}% vs {tb['accuracy'] * 100:.2f}% (reported only, not used for selection)."
               if ta and tb else ""), "📊"), unsafe_allow_html=True)
else:
    st.info("Run the full training pipeline (`python -m src.train --all`) to produce the with/without augmentation comparison.")
