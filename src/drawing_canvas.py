"""Streamlit wrapper for the custom PURF drawing canvas.

The front-end lives in ``src/components/draw_canvas/index.html`` and speaks the
Streamlit component protocol directly, so it needs no Node.js build step and no
third-party canvas package.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_FRONTEND = Path(__file__).parent / "components" / "draw_canvas"
_component = None


def _get_component():
    global _component
    if _component is None:
        import streamlit.components.v1 as components

        _component = components.declare_component("purf_draw_canvas", path=str(_FRONTEND))
    return _component


def draw_canvas(key: str = "purf_canvas", brush: int = 22, live: bool = True, size: int = 380) -> dict[str, Any] | None:
    """Render the canvas. Returns the latest event dict or None.

    Event fields: ``action`` (stroke/predict/clear/undo/redo), ``nonce``,
    ``empty`` (bool), ``strokes`` (int) and ``image`` (PNG data URL or None).
    """
    return _get_component()(brush=brush, live=live, size=size, key=key, default=None)


def decode_data_url(data_url: str) -> np.ndarray:
    """PNG data URL -> RGBA uint8 array."""
    if not data_url or "," not in data_url:
        raise ValueError("Empty canvas image")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))
