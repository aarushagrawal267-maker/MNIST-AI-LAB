"""Explainable-AI utilities for the Keras models.

Methods
-------
* Vanilla gradient saliency  — |d logit / d pixel|
* SmoothGrad                 — saliency averaged over noisy copies (less noisy maps)
* Integrated Gradients       — path-integrated gradients from a black baseline
                               (satisfies the completeness axiom; we report the check)
* Grad-CAM                   — class-weighted activation map of the last conv layer (CNNs only)
* Occlusion sensitivity      — model-agnostic; works for scikit-learn models too

These maps show which pixels the *model's output is sensitive to*. They are
not proof of causal reasoning and can be noisy or misleading on their own.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .preprocessing import to_cnn_input


def _tf():
    import tensorflow as tf

    return tf


def _norm(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    rng = m.max() - m.min()
    return (m - m.min()) / rng if rng > 1e-12 else np.zeros_like(m)


_LOGIT_CACHE: dict[int, object] = {}


def logit_fn(model):
    """Return a callable producing pre-softmax class scores (logits).

    Explanations are computed on logits, as in the original saliency and Grad-CAM papers:
    when a model is ~100% confident the softmax saturates and probability gradients
    vanish (~1e-4), leaving only noise after normalisation.
    """
    key = id(model)
    if key in _LOGIT_CACHE:
        return _LOGIT_CACHE[key]
    import keras

    tf = _tf()
    last = model.layers[-1]
    act = getattr(last, "activation", None)
    if isinstance(last, keras.layers.Dense) and getattr(act, "__name__", "") == "softmax":
        features = keras.Model(model.inputs[0], last.input)

        def fn(x, training=False):
            return tf.matmul(features(x, training=training), last.kernel) + last.bias
    else:  # pragma: no cover - models without a softmax head
        def fn(x, training=False):
            return model(x, training=training)
    _LOGIT_CACHE[key] = fn
    return fn


def class_score_gradients(model, x: np.ndarray, class_idx: int) -> np.ndarray:
    """Gradients of the class logit w.r.t. a batch of inputs (N, 28, 28, 1)."""
    tf = _tf()
    f = logit_fn(model)
    xt = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(xt)
        score = f(xt, training=False)[:, class_idx]
    return tape.gradient(score, xt).numpy()


def saliency_map(model, image28: np.ndarray, class_idx: int) -> np.ndarray:
    g = class_score_gradients(model, to_cnn_input(image28), class_idx)[0, ..., 0]
    return _norm(np.abs(g))


def smoothgrad(model, image28: np.ndarray, class_idx: int, n: int = 32, noise: float = 0.15, seed: int = 0) -> np.ndarray:
    x = to_cnn_input(image28)
    rng = np.random.default_rng(seed)
    noisy = np.clip(x + rng.normal(0, noise, size=(n,) + x.shape[1:]).astype(np.float32), 0, 1)
    g = class_score_gradients(model, noisy, class_idx)
    return _norm(np.abs(g).mean(axis=0)[..., 0])


def integrated_gradients(model, image28: np.ndarray, class_idx: int, steps: int = 64, baseline: np.ndarray | None = None):
    """Return (attribution map (28,28), completeness_error).

    Completeness: sum(attributions) ≈ F(x) - F(baseline), with F the class logit. The relative error is
    reported so users can judge the approximation quality.
    """
    x = to_cnn_input(image28)
    base = np.zeros_like(x) if baseline is None else to_cnn_input(baseline)
    alphas = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[:, None, None, None]
    path = base + alphas * (x - base)
    grads = class_score_gradients(model, path, class_idx)
    avg = (grads[:-1] + grads[1:]).mean(axis=0) / 2.0  # trapezoidal rule
    attr = ((x - base)[0] * avg)[..., 0]
    f = logit_fn(model)
    f_x = float(np.asarray(f(x, training=False))[0, class_idx])
    f_b = float(np.asarray(f(base, training=False))[0, class_idx])
    err = abs(attr.sum() - (f_x - f_b)) / (abs(f_x - f_b) + 1e-8)
    return attr.astype(np.float32), float(err)


def find_last_conv(model) -> str | None:
    import keras

    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    return None


def grad_cam(model, image28: np.ndarray, class_idx: int, layer_name: str | None = None) -> np.ndarray | None:
    """Grad-CAM heatmap upsampled to 28x28, or None if the model has no conv layers."""
    import cv2
    import keras

    tf = _tf()
    layer_name = layer_name or find_last_conv(model)
    if layer_name is None:
        return None
    last = model.layers[-1]
    grad_model = keras.Model(model.inputs[0], [model.get_layer(layer_name).output, last.input])
    x = tf.convert_to_tensor(to_cnn_input(image28))
    with tf.GradientTape() as tape:
        conv_out, feats = grad_model(x, training=False)
        score = (tf.matmul(feats, last.kernel) + last.bias)[:, class_idx]  # class logit
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # global-average-pooled gradients
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1))[0].numpy()
    cam = cv2.resize(cam, (28, 28), interpolation=cv2.INTER_CUBIC)
    return _norm(np.maximum(cam, 0))


def occlusion_sensitivity(
    predict_fn: Callable[[np.ndarray], np.ndarray], image28: np.ndarray, class_idx: int, patch: int = 4, stride: int = 2
) -> np.ndarray:
    """Drop in class probability when a patch is blacked out (model-agnostic)."""
    img = np.asarray(image28, dtype=np.float32)
    if img.max() > 1:
        img = img / 255.0
    base = float(predict_fn(img[None])[0, class_idx])
    heat = np.zeros_like(img)
    counts = np.zeros_like(img)
    batch, coords = [], []
    for y in range(0, 28 - patch + 1, stride):
        for x in range(0, 28 - patch + 1, stride):
            occ = img.copy()
            occ[y : y + patch, x : x + patch] = 0.0
            batch.append(occ)
            coords.append((y, x))
    probs = predict_fn(np.stack(batch))[:, class_idx]
    for (y, x), p in zip(coords, probs):
        heat[y : y + patch, x : x + patch] += base - p
        counts[y : y + patch, x : x + patch] += 1
    heat = heat / np.maximum(counts, 1)
    return np.clip(heat, 0, None)


DIVERGING = ("#3987e5", "#1b2347", "#e66767")


def colorize(heat: np.ndarray, cmap: str = "inferno", signed: bool = False) -> np.ndarray:
    """Heatmap -> RGB uint8. Signed maps use a blue / neutral / red diverging scale centred on 0."""
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    h = np.asarray(heat, dtype=np.float32)
    if signed:
        lim = float(np.abs(h).max()) or 1.0
        cm = LinearSegmentedColormap.from_list("purf_div", DIVERGING)
        rgb = cm((h / lim + 1) / 2)[..., :3]
    else:
        rgb = matplotlib.colormaps[cmap](_norm(h))[..., :3]
    return (rgb * 255).astype(np.uint8)


def overlay(image28: np.ndarray, heat: np.ndarray, alpha: float = 0.55, cmap: str = "inferno") -> np.ndarray:
    """Blend a heatmap onto a grayscale digit -> RGB uint8 (28, 28, 3)."""
    import matplotlib

    img = np.asarray(image28, dtype=np.float32)
    img = img / 255.0 if img.max() > 1 else img
    colored = matplotlib.colormaps[cmap](_norm(heat))[..., :3]
    base = np.repeat(img[..., None], 3, axis=2)
    return (np.clip((1 - alpha) * base + alpha * colored, 0, 1) * 255).astype(np.uint8)
