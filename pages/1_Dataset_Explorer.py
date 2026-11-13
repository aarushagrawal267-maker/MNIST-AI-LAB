"""Dataset Explorer — EDA of the MNIST dataset (Kaggle: hojjatk/mnist-dataset)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_validation import validate_dataset
from src.eda import class_mean_images, class_std_images, compute_stats, image_features
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Data",
    "Dataset Explorer",
    "70,000 grayscale 28×28 images of handwritten digits, loaded from the Kaggle MNIST dataset "
    "(<code>hojjatk/mnist-dataset</code>, original IDX files). Every statistic and insight on this page is computed "
    "live from the data.",
)
splits = C.safe_splits()


@st.cache_resource(show_spinner="Computing dataset statistics…")
def _stats():
    return C.eda_stats() or compute_stats(splits)


@st.cache_resource(show_spinner=False)
def _features():
    x = np.concatenate([splits.x_train, splits.x_val])
    y = np.concatenate([splits.y_train, splits.y_val])
    f = image_features(x)
    df = pd.DataFrame({k: np.asarray(v) for k, v in f.items()})
    df["digit"] = y
    return df, class_mean_images(x, y), class_std_images(x, y)


@st.cache_resource(show_spinner=False)
def _validation():
    class _D:
        x_train = np.concatenate([splits.x_train, splits.x_val])
        y_train = np.concatenate([splits.y_train, splits.y_val])
        x_test, y_test, source = splits.x_test, splits.y_test, splits.source

    return validate_dataset(_D)


stats = _stats()
tabs = st.tabs(["Overview", "Samples & pixels", "Digit gallery", "Distributions", "Similarity & insights"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    T.metric_row([
        T.metric_card("Total images", f"{stats['n_total']:,}", "train + test", "🗂️", T.SERIES[0]),
        T.metric_card("Training", f"{stats['n_train']:,}", "90% of the original 60k", "🏋️", T.SERIES[1]),
        T.metric_card("Validation", f"{stats['n_val']:,}", "stratified 10% split", "🧭", T.SERIES[2]),
        T.metric_card("Test", f"{stats['n_test']:,}", "original, untouched", "🏁", T.SERIES[3]),
    ])
    st.write("")
    T.metric_row([
        T.metric_card("Classes", "10", "digits 0 – 9", "🔢", T.SERIES[4]),
        T.metric_card("Image size", "28 × 28", "784 pixels · 1 channel", "📐", T.SERIES[6]),
        T.metric_card("Pixel range", f"{stats['pixel_min']} – {stats['pixel_max']}", f"mean {stats['pixel_mean']:.1f} · std {stats['pixel_std']:.1f}", "🎚️", T.BRAND),
        T.metric_card("Background", f"{stats['fraction_zero_pixels'] * 100:.1f}%", "of all pixels are exactly 0", "⬛", T.BRAND_2),
    ])
    T.section("Data validation", "automatic integrity checks run on load")
    rep = _validation()
    df = pd.DataFrame(rep["checks"])
    df["status"] = np.where(df["passed"], "✅ pass", "❌ fail")
    st.dataframe(df[["status", "check", "detail"]], hide_index=True, use_container_width=True)
    st.caption(f"Source: **{splits.source}** · cached at `data/processed/mnist.npz` · "
               f"{'all checks passed' if rep['all_passed'] else 'some checks failed'}")

# ------------------------------------------------------------------ samples
with tabs[1]:
    c1, c2, c3 = st.columns([1, 1, 1])
    split = c1.selectbox("Split", ["train", "val", "test"], format_func=lambda s: {"train": "Training", "val": "Validation", "test": "Test"}[s])
    digit = c2.selectbox("Digit", ["all"] + list(range(10)))
    n = c3.slider("Images", 12, 60, 36, step=12)
    x = getattr(splits, f"x_{split}")
    y = getattr(splits, f"y_{split}")
    pool = np.arange(len(y)) if digit == "all" else np.where(y == digit)[0]
    if st.button("New random sample", icon=":material/shuffle:"):
        st.session_state["eda_seed"] = st.session_state.get("eda_seed", 0) + 1
    rng = np.random.default_rng(st.session_state.get("eda_seed", 0))
    idx = rng.choice(pool, n, replace=False)
    T.image_grid([T.image_tile(x[i], f"<b>{y[i]}</b> · #{i}", upscale=3) for i in idx], columns=12)

    T.section("Random image viewer", "pixel intensity heatmap of one image")
    v1, v2 = st.columns([1.2, 1])
    pick = v2.number_input("Image index", 0, len(y) - 1, int(idx[0]))
    with v1:
        C.plot(C.digit_heatmap(x[pick], title=f"Digit {y[pick]} — raw pixel values (0–255)", height=560), key="eda_hm")
    with v2:
        img = x[pick]
        st.markdown(T.image_tile(img, f"label <b>{y[pick]}</b>", upscale=8), unsafe_allow_html=True)
        nz = img[img > 0]
        fig = go.Figure(go.Histogram(x=nz, nbinsx=32, marker=dict(color=T.SERIES[0], line=dict(width=1, color=T.SURFACE)),
                                     hovertemplate="value %{x}<br>%{y} pixels<extra></extra>"))
        fig.update_layout(xaxis_title="pixel value (non-zero only)", yaxis_title="pixels", showlegend=False)
        C.plot(T.style_fig(fig, 260, "Histogram of this image"), key="eda_hist1")
        st.caption(f"{(img == 0).mean() * 100:.1f}% background · {len(nz)} ink pixels · max {img.max()}")

# ------------------------------------------------------------------ gallery
with tabs[2]:
    df_f, means, stds = _features()
    T.section("Every digit, many hands", "12 random training examples per class")
    rng = np.random.default_rng(st.session_state.get("eda_seed", 0) + 99)
    for d in range(10):
        ids = rng.choice(np.where(splits.y_train == d)[0], 12, replace=False)
        T.image_grid([T.image_tile(splits.x_train[i], "", upscale=3) for i in ids], columns=12)
    T.section("Average digit", "mean of all training images per class")
    from src.explainability import colorize

    T.image_grid([T.image_tile(colorize(means[d]), f"<b>{d}</b>", upscale=5, rgb=True) for d in range(10)], columns=10)
    T.section("Where writers disagree", "per-pixel standard deviation — bright = high variation in handwriting")
    T.image_grid([T.image_tile(colorize(stds[d], "viridis"), f"<b>{d}</b>", upscale=5, rgb=True) for d in range(10)], columns=10)

# ------------------------------------------------------------------ distributions
with tabs[3]:
    df_f, _, _ = _features()
    T.section("Class distribution", "share of each split — near-identical shares confirm the stratified split")
    fig = go.Figure()
    for k, (s, name) in enumerate([("train", "training"), ("val", "validation"), ("test", "test")]):
        counts = np.array(stats["class_counts"][s])
        fig.add_bar(x=[str(i) for i in range(10)], y=counts / counts.sum() * 100, name=name,
                    marker=dict(color=T.SERIES[k], cornerradius=3), customdata=counts,
                    hovertemplate=f"{name}<br>digit %{{x}}<br>%{{y:.2f}}%% of split (%{{customdata:,}} images)<extra></extra>")
    fig.update_layout(barmode="group", xaxis_title="digit", yaxis_title="share of split (%)", bargroupgap=0.08)
    C.plot(T.style_fig(fig, 360), key="dist")
    counts = pd.DataFrame({s: stats["class_counts"][s] for s in ("train", "val", "test")})
    counts.index.name = "digit"
    with st.expander("Counts table"):
        st.dataframe(counts.T, use_container_width=True)

    a, b = st.columns(2)
    with a:
        T.section("Pixel intensities", "all non-zero pixels in the training set")
        hist = np.array(stats["pixel_histogram"])
        edges = np.linspace(0, 256, len(hist) + 1)
        fig = go.Figure(go.Bar(x=edges[1:-1], y=hist[1:], width=edges[1] - edges[0] - 1,
                               marker=dict(color=T.SERIES[0]), hovertemplate="≈%{x:.0f}<br>%{y:,} pixels<extra></extra>"))
        fig.update_layout(xaxis_title="pixel value", yaxis_title="pixel count", showlegend=False)
        C.plot(T.style_fig(fig, 340), key="pixhist")
        st.caption(f"The first bin (value 0, {stats['fraction_zero_pixels'] * 100:.1f}% of pixels) is omitted so the "
                   "stroke distribution is visible: most ink is either faint anti-aliasing or near-saturated.")
    with b:
        T.section("Ink per digit", "bright pixels (> 50% intensity) per image")
        fig = go.Figure()
        for d in range(10):
            vals = df_f.loc[df_f.digit == d, "ink_area"].sample(min(1500, (df_f.digit == d).sum()), random_state=0)
            fig.add_box(y=vals, name=str(d), marker=dict(color=T.SERIES[0], size=3), line=dict(width=1.5),
                        boxpoints=False, showlegend=False)
        fig.update_layout(xaxis_title="digit", yaxis_title="ink pixels")
        C.plot(T.style_fig(fig, 340), key="ink")

    a, b = st.columns(2)
    with a:
        T.section("Digit footprint", "mean bounding-box size per class")
        g = df_f.groupby("digit")[["bbox_w", "bbox_h"]].mean()
        fig = go.Figure()
        fig.add_bar(x=g.index.astype(str), y=g.bbox_w, name="width", marker=dict(color=T.SERIES[0], cornerradius=3))
        fig.add_bar(x=g.index.astype(str), y=g.bbox_h, name="height", marker=dict(color=T.SERIES[1], cornerradius=3))
        fig.update_layout(barmode="group", xaxis_title="digit", yaxis_title="pixels")
        C.plot(T.style_fig(fig, 320), key="bbox")
    with b:
        T.section("Positioning", "centre of mass of 3,000 random images")
        s = df_f.sample(3000, random_state=1)
        fig = go.Figure(go.Scattergl(x=s.com_x, y=s.com_y, mode="markers",
                                     marker=dict(size=5, color=T.SERIES[0], opacity=0.35),
                                     hovertemplate="x %{x:.2f}, y %{y:.2f}<extra></extra>"))
        fig.update_layout(xaxis=dict(title="x (px)", range=[11, 17]), yaxis=dict(title="y (px)", range=[17, 11], scaleanchor="x"))
        C.plot(T.style_fig(fig, 320), key="com")
        st.caption(f"Digits are tightly centred around ({stats['com_x_mean']:.2f}, {stats['com_y_mean']:.2f}) with "
                   f"{stats['com_std']:.2f}px spread — the canvas preprocessing reproduces this centring.")

# ------------------------------------------------------------------ similarity
with tabs[4]:
    a, b = st.columns([1.1, 1])
    with a:
        T.section("Similar-looking digits", "cosine similarity of class-average images")
        sim = np.array(stats["similarity_matrix"])
        z = np.where(np.eye(10, dtype=bool), np.nan, sim)
        fig = go.Figure(go.Heatmap(
            z=z, x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
            colorscale=[[i / (len(T.SEQ_BLUE) - 1), c] for i, c in enumerate(T.SEQ_BLUE)],
            text=[[f"{v:.2f}" if i != j else "" for j, v in enumerate(r)] for i, r in enumerate(sim)],
            texttemplate="%{text}", textfont=dict(size=10), xgap=2, ygap=2, showscale=False,
            hovertemplate="%{y} vs %{x}<br>similarity %{z:.3f}<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=False))
        C.plot(T.style_fig(fig, 470), key="sim")
    with b:
        T.section("Most similar pairs")
        _, means, _ = _features()
        for p in stats["similar_pairs"][:5]:
            cols = st.columns([1, 1, 2])
            cols[0].markdown(T.image_tile(means[p["a"]], f"<b>{p['a']}</b>", upscale=3), unsafe_allow_html=True)
            cols[1].markdown(T.image_tile(means[p["b"]], f"<b>{p['b']}</b>", upscale=3), unsafe_allow_html=True)
            cols[2].markdown(f"<div style='padding-top:22px'><span class='badge'>cos {p['cosine_similarity']:.3f}</span></div>",
                             unsafe_allow_html=True)
    T.section("Automatic insights", "generated from the statistics above")
    for line in stats["insights"]:
        st.markdown(f"<div class='purf-card' style='margin-bottom:10px'><p>💡 {line}</p></div>", unsafe_allow_html=True)
    st.markdown(T.card(
        "Why some digits are harder",
        "Digits whose average shapes overlap (high similarity above) or that have many writing styles (high pixel "
        "variability) leave less margin between classes. A 4 with a closed top resembles a 9; a 5 and a 3 share the "
        "lower curve; 7s written with a crossbar or 1s with a serif deviate from the class mean. The <b>Error Analysis</b> "
        "page shows which confusions the trained models actually make.", "🧩"), unsafe_allow_html=True)
