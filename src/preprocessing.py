"""Tensor-level preprocessing shared by training, evaluation and inference."""
from __future__ import annotations

import numpy as np

from .config import IMG_SIZE, NUM_CLASSES


def normalize(x: np.ndarray) -> np.ndarray:
    """Scale uint8 pixels 0-255 to float32 0-1 (no-op for already-normalized floats)."""
    x = np.asarray(x)
    if x.dtype == np.uint8 or x.max() > 1.0 + 1e-6:
        return x.astype(np.float32) / 255.0
    return x.astype(np.float32)


def to_cnn_input(x: np.ndarray) -> np.ndarray:
    """(N, 28, 28) or (28, 28) -> (N, 28, 28, 1) float32 in [0, 1]."""
    x = normalize(x)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim == 3:
        x = x[..., None]
    if x.shape[1:] != (IMG_SIZE, IMG_SIZE, 1):
        raise ValueError(f"Expected images of shape (28, 28), got {x.shape[1:]}")
    return x


def to_flat_input(x: np.ndarray) -> np.ndarray:
    """(N, 28, 28) or (28, 28) -> (N, 784) float32 in [0, 1]."""
    x = normalize(x)
    if x.ndim == 2:
        x = x[None, ...]
    x = x.reshape(len(x), -1)
    if x.shape[1] != IMG_SIZE * IMG_SIZE:
        raise ValueError(f"Expected 784 pixels per image, got {x.shape[1]}")
    return x


def prepare(x: np.ndarray, input_kind: str) -> np.ndarray:
    """Prepare images for a model that consumes ``input_kind`` ('image' or 'flat')."""
    if input_kind == "image":
        return to_cnn_input(x)
    if input_kind == "flat":
        return to_flat_input(x)
    raise ValueError(f"Unknown input kind: {input_kind}")


def one_hot(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64).ravel()
    if y.min() < 0 or y.max() >= num_classes:
        raise ValueError("Labels must be in [0, num_classes)")
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x)
    return {
        "shape": tuple(int(s) for s in x.shape),
        "dtype": str(x.dtype),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }
