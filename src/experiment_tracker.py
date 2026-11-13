"""Lightweight CSV experiment tracker (no external services required)."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import EXPERIMENTS_PATH

COLUMNS = [
    "experiment_id",
    "timestamp",
    "stage",
    "model_key",
    "model_type",
    "architecture",
    "optimizer",
    "learning_rate",
    "batch_size",
    "epochs",
    "epochs_trained",
    "dropout",
    "filters",
    "dense_units",
    "batchnorm",
    "augmentation",
    "train_samples",
    "val_accuracy",
    "val_loss",
    "train_time_s",
    "params",
    "notes",
]


class ExperimentTracker:
    def __init__(self, path: str | Path = EXPERIMENTS_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _next_id(self) -> str:
        df = self.load()
        nums = [int(str(i).split("-")[-1]) for i in df.get("experiment_id", []) if str(i).split("-")[-1].isdigit()]
        return f"EXP-{(max(nums) if nums else 0) + 1:03d}"

    def log(self, **row: Any) -> str:
        row = {k: row.get(k, "") for k in COLUMNS}
        row["experiment_id"] = row["experiment_id"] or self._next_id()
        row["timestamp"] = row["timestamp"] or time.strftime("%Y-%m-%d %H:%M:%S")
        for k in ("val_accuracy", "val_loss", "train_time_s", "learning_rate"):
            if isinstance(row[k], float):
                row[k] = round(row[k], 6)
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow(row)
        return row["experiment_id"]

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_csv(self.path)

    def best(self, stage: str | None = None, metric: str = "val_accuracy") -> pd.Series | None:
        df = self.load()
        if stage:
            df = df[df["stage"] == stage]
        if df.empty:
            return None
        return df.sort_values([metric, "val_loss"], ascending=[False, True]).iloc[0]

    def clear_stage(self, stage: str, model_key: str | None = None) -> None:
        """Remove rows of a stage (optionally only for one model) so re-runs do not duplicate."""
        df = self.load()
        if df.empty:
            return
        mask = df["stage"] == stage
        if model_key is not None:
            mask &= df["model_key"] == model_key
        df[~mask].to_csv(self.path, index=False)
