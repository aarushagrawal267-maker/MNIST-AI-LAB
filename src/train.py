"""Training pipeline and CLI.

Examples
--------
Full pipeline (baselines -> tuning -> TRAX selection -> final evaluation)::

    python -m src.train --all

Single model::

    python -m src.train --model cnn_baseline --epochs 10
    python -m src.train --model svm

Quick smoke run (small subsets, few epochs)::

    python -m src.train --all --quick
"""
from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import CFG, HISTORY_DIR, MODEL_NAME, SEED, SUMMARY_PATH, set_global_seed
from .data_loader import MnistSplits, get_splits, stratified_subset
from .experiment_tracker import ExperimentTracker
from .models import (
    MODEL_CATALOG,
    SKLEARN_BUILDERS,
    CNNConfig,
    baseline_cnn_config,
    build_ann,
    build_cnn,
    count_params,
)
from .preprocessing import to_cnn_input, to_flat_input
from . import model_manager as mm

TRAIN = CFG["training"]


# --------------------------------------------------------------------------- #
# Keras callbacks
# --------------------------------------------------------------------------- #
def _keras_callbacks():
    import keras

    class LearningRateLogger(keras.callbacks.Callback):
        """Record the optimizer learning rate at the end of every epoch."""

        def __init__(self):
            super().__init__()
            self.lrs: list[float] = []

        def on_epoch_end(self, epoch, logs=None):
            lr = self.model.optimizer.learning_rate
            self.lrs.append(float(np.asarray(lr.value if hasattr(lr, "value") else lr)))

    class FunctionCallback(keras.callbacks.Callback):
        """Forward epoch/batch events to plain Python functions (used by the Streamlit UI)."""

        def __init__(self, on_epoch: Callable | None = None, on_batch: Callable | None = None, every: int = 20):
            super().__init__()
            self.on_epoch, self.on_batch, self.every = on_epoch, on_batch, every

        def on_epoch_end(self, epoch, logs=None):
            if self.on_epoch:
                lr = self.model.optimizer.learning_rate
                logs = dict(logs or {})
                logs["learning_rate"] = float(np.asarray(lr.value if hasattr(lr, "value") else lr))
                self.on_epoch(epoch, logs)

        def on_train_batch_end(self, batch, logs=None):
            if self.on_batch and batch % self.every == 0:
                self.on_batch(batch, dict(logs or {}))

    return LearningRateLogger, FunctionCallback


# --------------------------------------------------------------------------- #
# Training primitives
# --------------------------------------------------------------------------- #
def train_keras(
    model,
    splits: MnistSplits,
    save_path: Path | None,
    epochs: int = TRAIN["epochs"],
    batch_size: int = TRAIN["batch_size"],
    augment: bool = False,
    train_subset: int | None = None,
    early_stopping: bool = True,
    reduce_lr: bool = True,
    on_epoch: Callable | None = None,
    on_batch: Callable | None = None,
    verbose: int = 2,
) -> dict[str, Any]:
    """Fit a compiled Keras model with EarlyStopping, ModelCheckpoint and ReduceLROnPlateau."""
    import keras

    from .augmentation import make_dataset

    LearningRateLogger, FunctionCallback = _keras_callbacks()
    x_tr, y_tr = stratified_subset(splits.x_train, splits.y_train, train_subset)
    x_tr, x_val = to_cnn_input(x_tr), to_cnn_input(splits.x_val)
    y_tr, y_val = y_tr.astype("int32"), splits.y_val.astype("int32")

    lr_logger = LearningRateLogger()
    callbacks: list = [lr_logger]
    if early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=TRAIN["early_stopping_patience"], restore_best_weights=True, verbose=0
            )
        )
    if reduce_lr:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=TRAIN["reduce_lr_factor"],
                patience=TRAIN["reduce_lr_patience"],
                min_lr=TRAIN["min_lr"],
                verbose=0,
            )
        )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(str(save_path), monitor="val_loss", save_best_only=True, verbose=0)
        )
    if on_epoch or on_batch:
        callbacks.append(FunctionCallback(on_epoch, on_batch))

    train_ds = make_dataset(x_tr, y_tr, batch_size, augment=augment, shuffle=True)
    val_ds = make_dataset(x_val, y_val, 1024, augment=False, shuffle=False)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*shuffle.*")
        hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=verbose)
    train_time = time.perf_counter() - t0

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    history["learning_rate"] = lr_logger.lrs[: len(history.get("loss", []))]
    # Persist the restored-best weights (EarlyStopping restores them after fit).
    if save_path is not None:
        model.save(save_path)
    best_epoch = int(np.argmin(history["val_loss"]))
    return {
        "model": model,
        "history": history,
        "train_time_s": train_time,
        "epochs_trained": len(history["loss"]),
        "best_epoch": best_epoch + 1,
        "val_accuracy": history["val_accuracy"][best_epoch],
        "val_loss": history["val_loss"][best_epoch],
        "train_samples": int(len(x_tr)),
        "path": save_path,
    }


def train_sklearn(key: str, splits: MnistSplits, train_subset: int | None = None, model=None) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, log_loss

    if train_subset is None and key == "svm":
        train_subset = CFG["classical"]["svm_train_samples"]
    model = model if model is not None else SKLEARN_BUILDERS[key]()
    x_tr, y_tr = stratified_subset(splits.x_train, splits.y_train, train_subset)
    x_tr = to_flat_input(x_tr)
    t0 = time.perf_counter()
    model.fit(x_tr, y_tr)
    train_time = time.perf_counter() - t0
    proba = model.predict_proba(to_flat_input(splits.x_val))
    return {
        "model": model,
        "history": None,
        "train_time_s": train_time,
        "epochs_trained": None,
        "val_accuracy": float(accuracy_score(splits.y_val, proba.argmax(1))),
        "val_loss": float(log_loss(splits.y_val, np.clip(proba, 1e-7, 1), labels=list(range(10)))),
        "train_samples": int(len(x_tr)),
    }


def save_history(key: str, history: dict | None) -> None:
    if history:
        (HISTORY_DIR / f"{key}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_history(key: str) -> dict | None:
    p = HISTORY_DIR / f"{key}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------------------- #
# High-level steps
# --------------------------------------------------------------------------- #
def train_and_register(
    key: str,
    splits: MnistSplits,
    tracker: ExperimentTracker,
    stage: str,
    *,
    display_name: str | None = None,
    architecture: str | None = None,
    model_type: str | None = None,
    cnn_config: CNNConfig | None = None,
    ann_kwargs: dict | None = None,
    epochs: int = TRAIN["epochs"],
    augment: bool = False,
    train_subset: int | None = None,
    custom: bool = False,
    on_epoch: Callable | None = None,
    on_batch: Callable | None = None,
    verbose: int = 2,
    notes: str = "",
) -> dict[str, Any]:
    """Build, train, save, register and track one model."""
    set_global_seed()
    catalog = MODEL_CATALOG.get(key, {})
    model_type = model_type or (
        key if key in SKLEARN_BUILDERS else ("ann" if key == "ann" or ann_kwargs is not None else "cnn")
    )
    hp: dict[str, Any] = {}

    if model_type in SKLEARN_BUILDERS:
        family, input_kind = "sklearn", "flat"
        print(f"\n=== Training {key} ({model_type}) ===")
        res = train_sklearn(model_type, splits, train_subset)
        path = mm.save_model(res["model"], key, family, custom)
        arch = architecture or MODEL_CATALOG[model_type]["architecture"]
        params = 0
    else:
        family, input_kind = "keras", "image"
        if model_type == "ann":
            kw = {"hidden_units": (256, 128), "dropout": 0.2, "learning_rate": TRAIN["learning_rate"],
                  "optimizer": TRAIN["optimizer"], **(ann_kwargs or {})}
            batch_size = int(kw.pop("batch_size", TRAIN["batch_size"]))
            model = build_ann(name=key, **kw)
            hp = {**kw, "batch_size": batch_size, "hidden_units": list(kw["hidden_units"])}
            parts = ["784"]
            for i, u in enumerate(kw["hidden_units"]):
                parts.append(f"Dense{u}")
                if i == 0 and kw["dropout"] > 0:
                    parts.append(f"Dropout{kw['dropout']:g}")
            arch = architecture or " -> ".join(parts) + " -> Softmax10"
        else:
            cfg = cnn_config or baseline_cnn_config()
            model = build_cnn(cfg, name=key)
            batch_size = cfg.batch_size
            hp = cfg.to_dict()
            arch = architecture or cfg.describe()
        print(f"\n=== Training {key}: {arch} (aug={augment}) ===")
        path = mm.model_path(key, family, custom)
        res = train_keras(
            model, splits, path, epochs=epochs, batch_size=batch_size, augment=augment,
            train_subset=train_subset, on_epoch=on_epoch, on_batch=on_batch, verbose=verbose,
        )
        params = count_params(res["model"])
        save_history(key, res["history"])

    entry = {
        "display_name": display_name or catalog.get("display_name", key),
        "family": family,
        "input_kind": input_kind,
        "model_type": model_type,
        "architecture": arch,
        "path": path,
        "params": params,
        "hyperparameters": hp,
        "augmentation": augment,
        "train_samples": res["train_samples"],
        "epochs_trained": res["epochs_trained"],
        "train_time_s": res["train_time_s"],
        "val_accuracy": res["val_accuracy"],
        "val_loss": res["val_loss"],
        "custom": custom,
        "stage": stage,
    }
    mm.register_model(key, entry)
    if not custom:
        tracker.clear_stage(stage, key)
    tracker.log(
        stage=stage,
        model_key=key,
        model_type=model_type,
        architecture=arch,
        optimizer=hp.get("optimizer", ""),
        learning_rate=hp.get("learning_rate", ""),
        batch_size=hp.get("batch_size", ""),
        epochs=epochs if family == "keras" else "",
        epochs_trained=res["epochs_trained"] or "",
        dropout=hp.get("dropout", ""),
        filters="/".join(map(str, hp.get("filters", []))) if hp.get("filters") else "",
        dense_units=hp.get("dense_units", "/".join(map(str, hp.get("hidden_units", []))) if hp.get("hidden_units") else ""),
        batchnorm=hp.get("batchnorm", ""),
        augmentation=augment,
        train_samples=res["train_samples"],
        val_accuracy=res["val_accuracy"],
        val_loss=res["val_loss"],
        train_time_s=res["train_time_s"],
        params=params,
        notes=notes,
    )
    print(f"[{key}] val_acc={res['val_accuracy']:.4f} val_loss={res['val_loss']:.4f} time={res['train_time_s']:.1f}s")
    return {**res, "entry": entry}


SEARCH_SPACE: dict[str, list] = {
    "filters": [[16, 32], [32, 64], [64, 128]],
    "convs_per_block": [1, 2],
    "batchnorm": [False, True],
    "dense_units": [64, 128, 256],
    "dropout": [0.2, 0.3, 0.5],
    "learning_rate": [3e-4, 1e-3, 3e-3],
    "optimizer": ["adam", "rmsprop", "sgd"],
    "batch_size": [64, 128, 256],
    "epochs": [3, 5],
}


def sample_configs(n: int, seed: int = SEED) -> list[tuple[CNNConfig, int]]:
    """Baseline config first, then ``n-1`` unique random draws from SEARCH_SPACE."""
    rng = random.Random(seed)
    base = baseline_cnn_config()
    configs = [(base, CFG["tuning"]["epochs"])]
    seen = {json.dumps(base.to_dict(), sort_keys=True)}
    while len(configs) < n:
        draw = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        epochs = draw.pop("epochs")
        cfg = CNNConfig(**draw)
        sig = json.dumps(cfg.to_dict(), sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            configs.append((cfg, epochs))
    return configs


def run_tuning(splits: MnistSplits, tracker: ExperimentTracker, n_trials: int | None = None,
               train_subset: int | None = None, verbose: int = 0) -> list[dict[str, Any]]:
    """Controlled random search over the CNN family (subset of train, full validation set)."""
    n_trials = n_trials or CFG["tuning"]["max_trials"]
    train_subset = train_subset or CFG["tuning"]["train_subset"]
    tracker.clear_stage("tuning")
    results = []
    for i, (cfg, epochs) in enumerate(sample_configs(n_trials), start=1):
        set_global_seed()
        model = build_cnn(cfg, name=f"trial_{i}")
        print(f"\n--- Trial {i}/{n_trials}: {cfg.describe()} | {cfg.optimizer} lr={cfg.learning_rate} "
              f"bs={cfg.batch_size} epochs={epochs}")
        res = train_keras(model, splits, None, epochs=epochs, batch_size=cfg.batch_size,
                          train_subset=train_subset, early_stopping=False, reduce_lr=False, verbose=verbose)
        params = count_params(model)
        exp_id = tracker.log(
            stage="tuning", model_key=f"trial_{i}", model_type="cnn", architecture=cfg.describe(),
            optimizer=cfg.optimizer, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size,
            epochs=epochs, epochs_trained=res["epochs_trained"], dropout=cfg.dropout,
            filters="/".join(map(str, cfg.filters)), dense_units=cfg.dense_units, batchnorm=cfg.batchnorm,
            augmentation=False, train_samples=res["train_samples"], val_accuracy=res["val_accuracy"],
            val_loss=res["val_loss"], train_time_s=res["train_time_s"], params=params,
            notes="baseline config" if i == 1 else "random search",
        )
        print(f"    {exp_id}: val_acc={res['val_accuracy']:.4f} val_loss={res['val_loss']:.4f} "
              f"({res['train_time_s']:.1f}s, {params:,} params)")
        results.append({"config": cfg, "epochs": epochs, "val_accuracy": res["val_accuracy"],
                        "val_loss": res["val_loss"], "experiment_id": exp_id})
    return results


def evaluate_all(splits: MnistSplits, keys: list[str] | None = None, splits_to_eval=("val", "test")) -> dict:
    from .evaluate import evaluate_model

    keys = keys or [k for k, v in mm.list_models().items() if not v.get("custom")]
    out: dict[str, dict] = {}
    for key in keys:
        out[key] = {}
        for split in splits_to_eval:
            m = evaluate_model(key, split, splits=splits)
            out[key][split] = {k: m[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                 "log_loss", "inference_ms_per_image", "latency_ms_single")}
            mm.update_model(key, **{f"{split}_metrics": out[key][split]})
            print(f"[eval] {key:<22} {split:<4} acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f}")
    return out


def run_pipeline(quick: bool = False, skip_classical: bool = False, skip_tuning: bool = False,
                 final_epochs: int | None = None) -> dict[str, Any]:
    """Complete reproducible pipeline used to produce the shipped models and report."""
    import keras  # noqa: F401  (initialise TF once)

    set_global_seed()
    t_start = time.perf_counter()
    from .data_loader import load_mnist
    from .data_validation import save_validation_report

    data = load_mnist()
    report = save_validation_report(data)
    print(f"[data] source={data.source} validation_passed={report['all_passed']}")
    splits = get_splits(data=data)
    tracker = ExperimentTracker()

    try:
        from .eda import save_eda_report

        save_eda_report(splits)
    except Exception as exc:  # EDA is informative, never blocking
        print(f"[eda] skipped: {exc}")

    sub = 6000 if quick else None
    epochs = 2 if quick else TRAIN["epochs"]
    final_epochs = final_epochs or (2 if quick else 20)

    # 1) Classical baselines ------------------------------------------------
    if not skip_classical:
        for key in ("logistic_regression", "knn", "svm"):
            train_and_register(key, splits, tracker, "baseline",
                               train_subset=min(sub, 3000) if quick else None)

    # 2) ANN and baseline CNN -------------------------------------------------
    train_and_register("ann", splits, tracker, "baseline", epochs=epochs, train_subset=sub,
                       ann_kwargs={"hidden_units": (256, 128), "dropout": 0.2})
    train_and_register("cnn_baseline", splits, tracker, "baseline", epochs=epochs, train_subset=sub,
                       cnn_config=baseline_cnn_config())

    # 3) Hyperparameter search ---------------------------------------------
    best_cfg = baseline_cnn_config()
    if not skip_tuning:
        trials = run_tuning(splits, tracker, n_trials=3 if quick else None,
                            train_subset=3000 if quick else None)
        best = sorted(trials, key=lambda r: (-r["val_accuracy"], r["val_loss"]))[0]
        best_cfg = best["config"]
        print(f"\n[tuning] best trial {best['experiment_id']}: {best_cfg.describe()} "
              f"val_acc={best['val_accuracy']:.4f}")

    # 4) Full training of the tuned architecture, with and without augmentation
    for key, aug, name in (("cnn_tuned", False, "Tuned CNN"), ("cnn_tuned_aug", True, "Tuned CNN + Augmentation")):
        train_and_register(key, splits, tracker, "final", display_name=name, cnn_config=best_cfg,
                           epochs=final_epochs, augment=aug, train_subset=sub,
                           notes="best tuning config, full training")

    # 5) Select TRAX: the CNN candidate with the best *validation* accuracy ----
    models = {k: v for k, v in mm.list_models().items() if not v.get("custom")}
    cnns = {k: v for k, v in models.items() if v.get("model_type") == "cnn"}
    prod = max(cnns, key=lambda k: (cnns[k]["val_accuracy"], -cnns[k]["val_loss"]))
    mm.set_production(prod)
    print(f"\n[select] {MODEL_NAME} = {prod} (val_acc={models[prod]['val_accuracy']:.4f})")

    # 6) Final evaluation (validation + held-out test) ----------------------
    results = evaluate_all(splits)
    summary = {
        "production_model": prod,
        "selection_criterion": "highest validation accuracy (test set never used for selection)",
        "dataset_source": data.source,
        "results": results,
        "pipeline_minutes": (time.perf_counter() - t_start) / 60,
        "quick_mode": quick,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] pipeline finished in {summary['pipeline_minutes']:.1f} min -> {SUMMARY_PATH}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PURF lab training pipeline")
    parser.add_argument("--all", action="store_true", help="run the full pipeline")
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--final-epochs", type=int, default=None)
    parser.add_argument("--model", choices=list(MODEL_CATALOG), help="train a single catalogue model")
    parser.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="evaluate all registered models")
    args = parser.parse_args(argv)

    if args.all:
        run_pipeline(args.quick, args.skip_classical, args.skip_tuning, args.final_epochs)
    elif args.model:
        splits = get_splits()
        train_and_register(args.model, splits, ExperimentTracker(), "manual", epochs=args.epochs,
                           augment=args.augment)
        evaluate_all(splits, [args.model])
    elif args.evaluate:
        evaluate_all(get_splits())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
