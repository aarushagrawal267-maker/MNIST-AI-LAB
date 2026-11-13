"""Dataset validation: shapes, dtypes, label ranges, class balance, integrity."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .config import IMG_SIZE, NUM_CLASSES, VALIDATION_REPORT_PATH

EXPECTED = {"x_train": (60000, IMG_SIZE, IMG_SIZE), "x_test": (10000, IMG_SIZE, IMG_SIZE)}


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def validate_dataset(data) -> dict[str, Any]:
    """Run integrity checks on a ``MnistData`` object and return a report."""
    checks: list[dict[str, Any]] = []
    for split in ("train", "test"):
        x = getattr(data, f"x_{split}")
        y = getattr(data, f"y_{split}")
        exp = EXPECTED[f"x_{split}"]
        checks.append(_check(f"{split}: image shape", x.shape == exp, f"{x.shape} (expected {exp})"))
        checks.append(_check(f"{split}: image dtype uint8", x.dtype == np.uint8, str(x.dtype)))
        checks.append(
            _check(f"{split}: label count matches images", len(y) == len(x), f"{len(y)} labels / {len(x)} images")
        )
        checks.append(
            _check(
                f"{split}: pixel range 0-255",
                int(x.min()) >= 0 and int(x.max()) <= 255,
                f"min={int(x.min())}, max={int(x.max())}",
            )
        )
        uniq = sorted(int(v) for v in np.unique(y))
        checks.append(
            _check(f"{split}: labels are digits 0-9", uniq == list(range(NUM_CLASSES)), f"unique labels={uniq}")
        )
        counts = np.bincount(y, minlength=NUM_CLASSES)
        ratio = counts.max() / max(counts.min(), 1)
        checks.append(
            _check(
                f"{split}: no severe class imbalance (max/min < 1.5)",
                ratio < 1.5,
                f"max/min ratio = {ratio:.3f}",
            )
        )
        blank = int((x.reshape(len(x), -1).max(axis=1) == 0).sum())
        checks.append(_check(f"{split}: no blank images", blank == 0, f"{blank} blank images"))

    report = {
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
        "source": getattr(data, "source", "unknown"),
        "class_counts": {
            "train": np.bincount(data.y_train, minlength=NUM_CLASSES).tolist(),
            "test": np.bincount(data.y_test, minlength=NUM_CLASSES).tolist(),
        },
    }
    return report


def assert_valid(data) -> None:
    report = validate_dataset(data)
    if not report["all_passed"]:
        failed = [c for c in report["checks"] if not c["passed"]]
        raise ValueError("MNIST validation failed: " + "; ".join(f"{c['check']} ({c['detail']})" for c in failed))


def save_validation_report(data) -> dict[str, Any]:
    report = validate_dataset(data)
    VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
