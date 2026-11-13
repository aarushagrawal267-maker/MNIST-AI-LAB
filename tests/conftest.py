"""Shared pytest fixtures for the PURF lab test-suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

CANVAS = 336
# Simple stroke recipes in normalised coordinates, mimicking what a user draws on the canvas.
STROKES = {
    0: [[(0.5 + 0.2 * np.cos(a), 0.5 + 0.3 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 40)]],
    1: [[(0.5, 0.18), (0.5, 0.82)]],
    2: [[(0.30, 0.30), (0.40, 0.20), (0.58, 0.20), (0.66, 0.32), (0.60, 0.48), (0.30, 0.80), (0.72, 0.80)]],
    3: [[(0.30, 0.25), (0.45, 0.18), (0.62, 0.22), (0.65, 0.35), (0.48, 0.48), (0.65, 0.58), (0.68, 0.72), (0.52, 0.82), (0.32, 0.76)]],
    4: [[(0.58, 0.80), (0.58, 0.18), (0.28, 0.60), (0.75, 0.60)]],
    7: [[(0.25, 0.22), (0.75, 0.22), (0.45, 0.82)]],
}


def draw_digit(digit: int, size: int = CANVAS, thickness: int = 22, offset=(0.0, 0.0), scale: float = 1.0,
               ink: int = 255, bg: int = 0) -> np.ndarray:
    """Render a canvas-like drawing (RGBA uint8) of a digit made of polyline strokes."""
    img = np.full((size, size), bg, np.uint8)
    for stroke in STROKES[digit]:
        pts = np.array([[(0.5 + (x - 0.5) * scale + offset[0]) * size, (0.5 + (y - 0.5) * scale + offset[1]) * size]
                        for x, y in stroke], np.int32)
        cv2.polylines(img, [pts], False, int(ink), thickness, lineType=cv2.LINE_AA)
    rgba = np.dstack([img, img, img, np.full_like(img, 255)])
    return rgba


@pytest.fixture(scope="session")
def splits():
    from src.data_loader import get_splits

    return get_splits()


@pytest.fixture(scope="session")
def predictor():
    from src import model_manager as mm
    from src.prediction import Predictor

    if mm.production_key() is None:
        pytest.skip("No trained production model; run `python -m src.train --all`.")
    return Predictor()
