"""Model evaluation: metrics, curves, timing, persisted reports.

Validation metrics drive model selection. Test metrics are computed only for
final reporting and are stored separately (``*_test.json``).
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from .config import METRICS_DIR, NUM_CLASSES, PREDICTIONS_DIR
from .preprocessing import prepare


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def predict_proba(model, family: str, input_kind: str, images: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Return (N, 10) class probabilities for uint8 or [0,1] images of shape (N, 28, 28)."""
    x = prepare(images, input_kind)
    if family == "keras":
        if len(x) <= 64:
            proba = np.asarray(model(x, training=False))
        else:
            proba = model.predict(x, batch_size=batch_size, verbose=0)
    else:
        proba = model.predict_proba(x)
    proba = np.asarray(proba, dtype=np.float64)
    proba = np.clip(proba, 0.0, 1.0)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba.astype(np.float32)


def measure_inference(model, family: str, input_kind: str, images: np.ndarray, n_batch: int = 2000, n_single: int = 30):
    """Return batch throughput (ms/image) and single-image latency (ms)."""
    xb = images[:n_batch]
    predict_proba(model, family, input_kind, xb[:64])  # warm-up
    t0 = time.perf_counter()
    predict_proba(model, family, input_kind, xb)
    batch_ms = (time.perf_counter() - t0) * 1000 / len(xb)
    times = []
    for i in range(n_single):
        t0 = time.perf_counter()
        predict_proba(model, family, input_kind, images[i : i + 1])
        times.append((time.perf_counter() - t0) * 1000)
    return float(batch_ms), float(np.median(times))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        log_loss,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = proba.argmax(axis=1)
    labels = list(range(NUM_CLASSES))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_acc = (np.diag(cm) / cm.sum(axis=1).clip(min=1)).tolist()
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    try:
        auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    top2 = np.argsort(-proba, axis=1)[:, :2]
    top2_acc = float(np.mean([t in row for t, row in zip(y_true, top2)]))
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),
        "f1_macro": float(f_m),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-7, 1), labels=labels)),
        "roc_auc_ovr_macro": auc,
        "top2_accuracy": top2_acc,
        "n_errors": int((y_pred != y_true).sum()),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def accuracy_ci(n_correct: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for an accuracy of ``n_correct`` out of ``n``."""
    from scipy.stats import binomtest

    if n <= 0:
        raise ValueError("accuracy_ci needs at least one sample")
    ci = binomtest(int(n_correct), int(n)).proportion_ci(confidence_level=confidence, method="wilson")
    return float(ci.low), float(ci.high)


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """Exact (binomial) McNemar test for two classifiers scored on the same images.

    ``only_a`` counts images model A gets right and model B gets wrong, ``only_b`` the reverse.
    A large p-value means the accuracy difference is within what chance disagreement would produce.
    """
    from scipy.stats import binomtest

    y_true, pred_a, pred_b = (np.asarray(v).astype(int) for v in (y_true, pred_a, pred_b))
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("mcnemar_exact needs aligned predictions")
    ok_a, ok_b = pred_a == y_true, pred_b == y_true
    only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p = float(binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "p_value": p}


def roc_curves(y_true: np.ndarray, proba: np.ndarray) -> dict[int, dict[str, Any]]:
    from sklearn.metrics import auc, roc_curve

    out = {}
    for c in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve((y_true == c).astype(int), proba[:, c])
        out[c] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
    return out


def pr_curves(y_true: np.ndarray, proba: np.ndarray) -> dict[int, dict[str, Any]]:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    out = {}
    for c in range(NUM_CLASSES):
        yb = (y_true == c).astype(int)
        prec, rec, _ = precision_recall_curve(yb, proba[:, c])
        out[c] = {"precision": prec, "recall": rec, "ap": float(average_precision_score(yb, proba[:, c]))}
    return out


def top_confusions(cm: np.ndarray | list, k: int = 10) -> list[dict[str, int]]:
    cm = np.asarray(cm)
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append({"actual": i, "predicted": j, "count": int(cm[i, j])})
    return sorted(pairs, key=lambda d: -d["count"])[:k]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def metrics_path(key: str, split: str):
    return METRICS_DIR / f"{key}_{split}.json"


def predictions_path(key: str, split: str):
    return PREDICTIONS_DIR / f"{key}_{split}.npz"


def evaluate_model(key: str, split: str, splits=None, save: bool = True, timing: bool = True) -> dict[str, Any]:
    """Evaluate a registered model on 'val' or 'test' and persist metrics + probabilities."""
    from . import model_manager as mm
    from .data_loader import get_splits

    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    splits = splits or get_splits()
    entry = mm.get_entry(key)
    model = mm.load_model(key)
    x = splits.x_val if split == "val" else splits.x_test
    y = splits.y_val if split == "val" else splits.y_test

    t0 = time.perf_counter()
    proba = predict_proba(model, entry["family"], entry["input_kind"], x)
    total_s = time.perf_counter() - t0
    metrics = compute_metrics(y, proba)
    metrics.update({"model_key": key, "split": split, "predict_seconds_total": total_s})
    if timing:
        batch_ms, single_ms = measure_inference(model, entry["family"], entry["input_kind"], x)
        metrics.update({"inference_ms_per_image": batch_ms, "latency_ms_single": single_ms})
    if save:
        metrics_path(key, split).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.savez_compressed(predictions_path(key, split), proba=proba.astype(np.float32), y_true=y.astype(np.uint8))
    return metrics


def load_metrics(key: str, split: str) -> dict[str, Any] | None:
    p = metrics_path(key, split)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_predictions(key: str, split: str):
    p = predictions_path(key, split)
    if not p.exists():
        return None
    with np.load(p) as npz:
        return npz["proba"], npz["y_true"]
