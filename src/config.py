"""Central configuration for the PURF lab.

Loads ``config.yaml`` from the project root and exposes resolved paths and
settings. Every other module imports from here so paths work identically on
Windows, macOS and Linux.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Quieten TensorFlow before it is imported anywhere else.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = ROOT_DIR / "config.yaml"

_DEFAULTS: dict[str, Any] = {
    "project": {"lab_name": "PURF", "model_name": "TRAX", "seed": 42},
    "paths": {
        "data_raw": "data/raw",
        "data_processed": "data/processed",
        "models": "models",
        "figures": "outputs/figures",
        "reports": "outputs/reports",
    },
    "data": {
        "kaggle_dataset": "hojjatk/mnist-dataset",
        "validation_size": 0.1,
        "image_size": 28,
        "num_classes": 10,
    },
    "training": {
        "batch_size": 128,
        "epochs": 15,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "early_stopping_patience": 4,
        "reduce_lr_patience": 2,
        "reduce_lr_factor": 0.5,
        "min_lr": 1e-5,
    },
    "classical": {
        "pca_components": 50,
        "svm_train_samples": 20000,
        "knn_neighbors": 3,
        "logreg_max_iter": 300,
    },
    "tuning": {"train_subset": 15000, "epochs": 4, "max_trials": 10},
    "augmentation": {"rotation_degrees": 10, "translation": 0.1, "zoom": 0.1},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Return the merged configuration (defaults overridden by config.yaml)."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
    else:
        user_cfg = {}
    return _deep_merge(_DEFAULTS, user_cfg)


CFG = load_config()

LAB_NAME: str = CFG["project"]["lab_name"]
MODEL_NAME: str = CFG["project"]["model_name"]
SEED: int = int(CFG["project"]["seed"])
IMG_SIZE: int = int(CFG["data"]["image_size"])
NUM_CLASSES: int = int(CFG["data"]["num_classes"])
CLASS_NAMES: list[str] = [str(i) for i in range(NUM_CLASSES)]


def _p(key: str) -> Path:
    path = ROOT_DIR / CFG["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path


DATA_RAW_DIR: Path = _p("data_raw")
DATA_PROCESSED_DIR: Path = _p("data_processed")
MODELS_DIR: Path = _p("models")
FIGURES_DIR: Path = _p("figures")
REPORTS_DIR: Path = _p("reports")
METRICS_DIR: Path = REPORTS_DIR / "metrics"
PREDICTIONS_DIR: Path = REPORTS_DIR / "predictions"
HISTORY_DIR: Path = REPORTS_DIR / "histories"
CUSTOM_MODELS_DIR: Path = MODELS_DIR / "custom"
for _d in (METRICS_DIR, PREDICTIONS_DIR, HISTORY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

REGISTRY_PATH: Path = MODELS_DIR / "registry.json"
EXPERIMENTS_PATH: Path = REPORTS_DIR / "experiments.csv"
EDA_STATS_PATH: Path = REPORTS_DIR / "eda_stats.json"
VALIDATION_REPORT_PATH: Path = REPORTS_DIR / "data_validation.json"
SUMMARY_PATH: Path = REPORTS_DIR / "results_summary.json"


def set_global_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if imported) TensorFlow/Keras for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except Exception:  # pragma: no cover - keras optional at import time
        pass
