"""Model persistence and registry.

Every trained model is saved to ``models/`` (Keras ``.keras`` format or Joblib
``.joblib``) and described in ``models/registry.json``. The registry also
records which model is the production model, branded **TRAX**.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from .config import CUSTOM_MODELS_DIR, MODEL_NAME, MODELS_DIR, REGISTRY_PATH, ROOT_DIR

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _empty() -> dict[str, Any]:
    return {"production_model": None, "models": {}}


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return _empty()
    try:
        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty()
    reg.setdefault("models", {})
    reg.setdefault("production_model", None)
    return reg


def save_registry(reg: dict[str, Any]) -> None:
    with _LOCK:
        tmp = REGISTRY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(reg, indent=2, default=_json_default), encoding="utf-8")
        tmp.replace(REGISTRY_PATH)


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def rel(path: Path) -> str:
    """Store paths relative to the project root using forward slashes (portable)."""
    try:
        return Path(path).resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def abs_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT_DIR / p


def model_path(key: str, family: str, custom: bool = False) -> Path:
    base = CUSTOM_MODELS_DIR if custom else MODELS_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{key}.{'keras' if family == 'keras' else 'joblib'}"


def save_model(model, key: str, family: str, custom: bool = False) -> Path:
    path = model_path(key, family, custom)
    if family == "keras":
        model.save(path)
    else:
        import joblib

        joblib.dump(model, path, compress=3)
    _CACHE.pop(key, None)
    return path


def register_model(key: str, entry: dict[str, Any], make_production: bool = False) -> None:
    reg = load_registry()
    entry = dict(entry)
    entry.setdefault("registered_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    if "path" in entry:
        entry["path"] = rel(Path(entry["path"]))
    reg["models"][key] = {**reg["models"].get(key, {}), **entry}
    if make_production:
        reg["production_model"] = key
    save_registry(reg)


def update_model(key: str, **fields) -> None:
    reg = load_registry()
    if key in reg["models"]:
        reg["models"][key].update(fields)
        save_registry(reg)


def list_models(available_only: bool = True) -> dict[str, dict[str, Any]]:
    models = load_registry()["models"]
    if available_only:
        models = {k: v for k, v in models.items() if abs_path(v["path"]).exists()}
    return models


def get_entry(key: str) -> dict[str, Any]:
    models = load_registry()["models"]
    if key not in models:
        raise KeyError(f"Model '{key}' is not registered. Train it first (python -m src.train).")
    return models[key]


def production_key() -> str | None:
    reg = load_registry()
    key = reg.get("production_model")
    if key and key in reg["models"] and abs_path(reg["models"][key]["path"]).exists():
        return key
    return None


def set_production(key: str) -> None:
    reg = load_registry()
    if key not in reg["models"]:
        raise KeyError(key)
    reg["production_model"] = key
    save_registry(reg)


def display_name(key: str, entry: dict[str, Any] | None = None) -> str:
    entry = entry or load_registry()["models"].get(key, {})
    name = entry.get("display_name", key)
    return f"{MODEL_NAME} · {name}" if key == production_key() else name


def load_model(key: str):
    """Load (and memoise) a registered model."""
    if key in _CACHE:
        return _CACHE[key]
    entry = get_entry(key)
    path = abs_path(entry["path"])
    if not path.exists():
        raise FileNotFoundError(f"Model file missing: {path}")
    if entry["family"] == "keras":
        from .numpy_runtime import HAS_TENSORFLOW, NumpyModel, npz_path

        if HAS_TENSORFLOW:
            import keras

            model = keras.models.load_model(path, compile=False)
        elif npz_path(path).exists():  # TensorFlow unavailable (e.g. Python 3.14): pure-NumPy inference
            model = NumpyModel(npz_path(path))
        else:
            raise RuntimeError(f"TensorFlow is not installed and no NumPy export exists for '{key}'.")
    else:
        import joblib

        model = joblib.load(path)
    _CACHE[key] = model
    return model


def delete_model(key: str) -> None:
    reg = load_registry()
    entry = reg["models"].pop(key, None)
    if entry is None:
        return
    if reg.get("production_model") == key:
        reg["production_model"] = None
    path = abs_path(entry["path"])
    if path.exists() and CUSTOM_MODELS_DIR in path.parents:
        path.unlink()
    _CACHE.pop(key, None)
    save_registry(reg)


def copy_model_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
