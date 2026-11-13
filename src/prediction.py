"""Inference API used by the drawing canvas, image upload and batch pages."""
from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import model_manager as mm
from .evaluate import predict_proba
from .image_processor import (
    SUPPORTED_EXTENSIONS,
    ImageProcessingError,
    ProcessedImage,
    preprocess_digit,
    validate_model_input,
)

MAX_ZIP_FILES = 500
MAX_ZIP_BYTES = 50 * 1024 * 1024


@dataclass
class Prediction:
    digit: int
    confidence: float
    probabilities: np.ndarray            # (10,) sums to 1
    top3: list[tuple[int, float]]
    model_key: str
    latency_ms: float
    processed: ProcessedImage | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def certainty_label(self) -> str:
        if self.confidence >= 0.95:
            return "Very confident"
        if self.confidence >= 0.80:
            return "Confident"
        if self.confidence >= 0.55:
            return "Unsure"
        return "Very unsure"


class Predictor:
    """Wraps a registered model; defaults to the production model (TRAX)."""

    def __init__(self, model_key: str | None = None):
        key = model_key or mm.production_key()
        if key is None:
            raise RuntimeError("No trained model found. Run `python -m src.train --all` first.")
        self.key = key
        self.entry = mm.get_entry(key)
        self.model = mm.load_model(key)

    # -- raw 28x28 inputs ---------------------------------------------------
    def predict_array(self, images: np.ndarray) -> np.ndarray:
        """(N, 28, 28) or (28, 28) images in [0,1] or uint8 -> (N, 10) probabilities."""
        images = np.asarray(images)
        if images.ndim == 2:
            images = images[None]
        if images.ndim != 3 or images.shape[1:] != (28, 28):
            raise ValueError(f"Expected images of shape (N, 28, 28), got {images.shape}")
        return predict_proba(self.model, self.entry["family"], self.entry["input_kind"], images)

    def _wrap(self, proba: np.ndarray, latency_ms: float, processed=None) -> Prediction:
        order = np.argsort(-proba)
        return Prediction(
            digit=int(order[0]),
            confidence=float(proba[order[0]]),
            probabilities=proba,
            top3=[(int(i), float(proba[i])) for i in order[:3]],
            model_key=self.key,
            latency_ms=latency_ms,
            processed=processed,
            warnings=list(processed.warnings) if processed else [],
        )

    def predict_processed(self, image28: np.ndarray, processed: ProcessedImage | None = None) -> Prediction:
        validate_model_input(np.asarray(image28, dtype=np.float32))
        t0 = time.perf_counter()
        proba = self.predict_array(image28)[0]
        return self._wrap(proba, (time.perf_counter() - t0) * 1000, processed)

    # -- arbitrary images --------------------------------------------------
    def predict_image(self, source: Any, invert: str = "auto") -> Prediction:
        processed = preprocess_digit(source, invert=invert)
        return self.predict_processed(processed.image, processed)

    def predict_batch(self, items: Iterable[tuple[str, Any]], invert: str = "auto") -> list[dict[str, Any]]:
        """Predict many (name, source) pairs. Invalid images are reported, not raised."""
        items = list(items)
        names, processed, rows = [], [], []
        for name, src in items:
            try:
                p = preprocess_digit(src, invert=invert)
                names.append(name)
                processed.append(p)
            except ImageProcessingError as exc:
                rows.append({"file": name, "status": "error", "error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive
                rows.append({"file": name, "status": "error", "error": f"Unexpected error: {exc}"})
        if processed:
            t0 = time.perf_counter()
            proba = self.predict_array(np.stack([p.image for p in processed]))
            per_img = (time.perf_counter() - t0) * 1000 / len(processed)
            for name, p, pr in zip(names, processed, proba):
                pred = self._wrap(pr, per_img, p)
                rows.append({
                    "file": name,
                    "status": "ok",
                    "prediction": pred.digit,
                    "confidence": pred.confidence,
                    "second_choice": pred.top3[1][0],
                    "second_prob": pred.top3[1][1],
                    "third_choice": pred.top3[2][0],
                    "third_prob": pred.top3[2][1],
                    "certainty": pred.certainty_label,
                    "warnings": "; ".join(p.warnings),
                    "probabilities": pr,
                    "processed": p.image,
                })
        order = {n: i for i, (n, _) in enumerate(items)}
        rows.sort(key=lambda r: order.get(r["file"], 0))
        return rows


def expand_uploads(files: Iterable[tuple[str, bytes]]) -> tuple[list[tuple[str, bytes]], list[dict[str, str]]]:
    """Expand ZIP archives safely and filter unsupported files.

    Returns (valid_items, rejected_rows).
    """
    items: list[tuple[str, bytes]] = []
    rejected: list[dict[str, str]] = []
    for name, data in files:
        ext = Path(name).suffix.lower()
        if ext == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    members = [m for m in zf.infolist() if not m.is_dir()]
                    if len(members) > MAX_ZIP_FILES:
                        rejected.append({"file": name, "status": "error",
                                         "error": f"ZIP contains more than {MAX_ZIP_FILES} files."})
                        continue
                    if sum(m.file_size for m in members) > MAX_ZIP_BYTES:
                        rejected.append({"file": name, "status": "error",
                                         "error": "ZIP content exceeds 50 MB when extracted."})
                        continue
                    for m in members:
                        inner = Path(m.filename)
                        if inner.name.startswith(".") or "__MACOSX" in inner.parts:
                            continue
                        label = f"{name}/{inner.as_posix()}"
                        if inner.suffix.lower() not in SUPPORTED_EXTENSIONS:
                            rejected.append({"file": label, "status": "error",
                                             "error": f"Unsupported file type '{inner.suffix or 'none'}'."})
                            continue
                        items.append((label, zf.read(m)))
            except zipfile.BadZipFile:
                rejected.append({"file": name, "status": "error", "error": "Corrupted or invalid ZIP archive."})
        elif ext in SUPPORTED_EXTENSIONS:
            items.append((name, data))
        else:
            rejected.append({"file": name, "status": "error",
                             "error": f"Unsupported file type '{ext or 'none'}'. Use PNG, JPG, JPEG or ZIP."})
    return items, rejected
