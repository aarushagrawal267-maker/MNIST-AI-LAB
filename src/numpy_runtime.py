"""TensorFlow-free inference for PURF's Keras models.

Each Keras model is also exported to ``models/<key>.npz`` (layer specs + weights). When TensorFlow is
unavailable — e.g. a deployment on Python 3.14, for which no TensorFlow release exists yet — the app
loads that file and runs the forward pass in pure NumPy. Predictions match TensorFlow to ~1e-6
(verified by tests/test_numpy_runtime.py).

Supported layers: InputLayer, Conv2D (stride 1, 'same'/'valid'), BatchNormalization, Activation,
MaxPooling2D, Flatten, Dense, Dropout (identity at inference).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HAS_TENSORFLOW: bool = importlib.util.find_spec("tensorflow") is not None


def npz_path(keras_path: str | Path) -> Path:
    return Path(keras_path).with_suffix(".npz")


# --------------------------------------------------------------------------- #
# Export (requires TensorFlow)
# --------------------------------------------------------------------------- #
def export_numpy(model, path: str | Path) -> Path:
    """Write layer specs and weights of a Keras model to an .npz file."""
    specs, arrays = [], {}
    for i, layer in enumerate(model.layers):
        cls = layer.__class__.__name__
        cfg = layer.get_config()
        spec = {"class": cls, "name": layer.name}
        if cls == "Conv2D":
            spec.update(padding=cfg["padding"], strides=list(cfg["strides"]), activation=cfg["activation"],
                        use_bias=cfg["use_bias"])
        elif cls == "Dense":
            spec.update(activation=cfg["activation"], use_bias=cfg["use_bias"])
        elif cls == "BatchNormalization":
            spec.update(epsilon=cfg["epsilon"], scale=cfg["scale"], center=cfg["center"])
        elif cls == "Activation":
            spec.update(activation=cfg["activation"])
        elif cls == "MaxPooling2D":
            spec.update(pool_size=list(cfg["pool_size"]), strides=list(cfg["strides"] or cfg["pool_size"]),
                        padding=cfg["padding"])
        elif cls not in ("InputLayer", "Flatten", "Dropout"):
            raise ValueError(f"Layer type {cls} is not supported by the NumPy runtime")
        weights = layer.get_weights()
        spec["n_weights"] = len(weights)
        for j, w in enumerate(weights):
            arrays[f"w{i}_{j}"] = np.asarray(w, dtype=np.float32)
        specs.append(spec)
    path = Path(path)
    np.savez_compressed(path, __spec__=np.array(json.dumps(specs)), **arrays)
    return path


# --------------------------------------------------------------------------- #
# Inference (NumPy only)
# --------------------------------------------------------------------------- #
def _act(x: np.ndarray, name: str | None) -> np.ndarray:
    if name in (None, "linear"):
        return x
    if name == "relu":
        return np.maximum(x, 0)
    if name == "softmax":
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    if name == "sigmoid":
        return 1 / (1 + np.exp(-x))
    raise ValueError(f"Unsupported activation {name}")


def _conv2d(x: np.ndarray, k: np.ndarray, padding: str) -> np.ndarray:
    kh, kw = k.shape[:2]
    if padding == "same":
        ph, pw = kh // 2, kw // 2
        x = np.pad(x, ((0, 0), (ph, kh - 1 - ph), (pw, kw - 1 - pw), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))  # (N, H', W', C, kh, kw)
    return np.einsum("nhwcij,ijco->nhwo", win, k, optimize=True)


def _maxpool(x: np.ndarray, size, strides) -> np.ndarray:
    ph, pw = size
    sh, sw = strides
    n, h, w, c = x.shape
    oh, ow = (h - ph) // sh + 1, (w - pw) // sw + 1
    win = np.lib.stride_tricks.sliding_window_view(x, (ph, pw), axis=(1, 2))[:, ::sh, ::sw][:, :oh, :ow]
    return win.max(axis=(-2, -1))


class NumpyModel:
    """Drop-in replacement for a Keras model at inference time (``model(x)`` / ``model.predict(x)``)."""

    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as z:
            self.specs = json.loads(str(z["__spec__"]))
            self.weights = {k: z[k] for k in z.files if k != "__spec__"}
        self.path = Path(path)
        self.layers = self.specs

    def _forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        for i, s in enumerate(self.specs):
            w = [self.weights[f"w{i}_{j}"] for j in range(s["n_weights"])]
            c = s["class"]
            if c == "Conv2D":
                if s["strides"] != [1, 1]:
                    raise ValueError("Only stride-1 convolutions are supported")
                x = _conv2d(x, w[0], s["padding"])
                if s["use_bias"]:
                    x = x + w[1]
                x = _act(x, s["activation"])
            elif c == "BatchNormalization":
                idx = 0
                gamma = w[idx] if s["scale"] else 1.0
                idx += int(s["scale"])
                beta = w[idx] if s["center"] else 0.0
                idx += int(s["center"])
                mean, var = w[idx], w[idx + 1]
                x = (x - mean) / np.sqrt(var + s["epsilon"]) * gamma + beta
            elif c == "Activation":
                x = _act(x, s["activation"])
            elif c == "MaxPooling2D":
                x = _maxpool(x, s["pool_size"], s["strides"])
            elif c == "Flatten":
                x = x.reshape(len(x), -1)
            elif c == "Dense":
                x = x @ w[0]
                if s["use_bias"]:
                    x = x + w[1]
                x = _act(x, s["activation"])
            # InputLayer / Dropout: identity at inference
        return x

    def __call__(self, x, training: bool = False) -> np.ndarray:
        return self._forward(np.asarray(x))

    def predict(self, x, batch_size: int = 512, verbose: int = 0) -> np.ndarray:
        x = np.asarray(x)
        return np.concatenate([self._forward(x[i: i + batch_size]) for i in range(0, len(x), batch_size)])

    def count_params(self) -> int:
        return int(sum(v.size for v in self.weights.values()))

    def summary(self, print_fn=print, **_) -> None:
        print_fn(f"NumPy runtime model ({self.path.name}) — {self.count_params():,} parameters")
        for s in self.specs:
            print_fn(f"  {s['class']:<20} {s['name']}")
