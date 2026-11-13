"""Convert real-world digit images (canvas drawings, photos, scans) into MNIST format.

MNIST digits were produced by: size-normalising each digit to fit a 20x20 box
(preserving aspect ratio, anti-aliased), then placing it in a 28x28 frame so
that its *centre of mass* sits at the frame centre. Light strokes on a dark
background. This module reproduces that recipe so user input matches the
distribution the models were trained on.

Naively resizing a whole 400x400 canvas down to 28x28 would shrink a small
drawing into a few pixels, keep it off-centre and distort its proportions,
which is why each step below exists.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import IMG_SIZE

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DIGIT_BOX = 20  # MNIST digits are fitted into a 20x20 box inside the 28x28 frame


class ImageProcessingError(ValueError):
    """User-facing error explaining why an image cannot be used."""


@dataclass
class ProcessedImage:
    image: np.ndarray                       # (28, 28) float32 in [0, 1]
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    inverted: bool = False
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h in the grayscale image
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_image(source: Any) -> np.ndarray:
    """Load PNG/JPG bytes, a path, a file-like object, a PIL image or an array.

    Returns an RGB(A) or grayscale uint8 numpy array.
    """
    if isinstance(source, np.ndarray):
        arr = source
    elif isinstance(source, Image.Image):
        arr = np.array(_exif_fix(source))
    else:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ImageProcessingError(
                    f"Unsupported file type '{path.suffix}'. Please use PNG, JPG or JPEG."
                )
            data = path.read_bytes()
        elif isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        elif hasattr(source, "read"):
            data = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
        else:
            raise ImageProcessingError(f"Cannot read an image from object of type {type(source).__name__}.")
        if not data:
            raise ImageProcessingError("The file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ImageProcessingError("The image is larger than 10 MB. Please upload a smaller file.")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise ImageProcessingError(
                "The file could not be decoded as an image. Please upload a valid PNG, JPG or JPEG."
            ) from exc
        if img.format not in {"PNG", "JPEG", "MPO"}:
            raise ImageProcessingError(f"Unsupported image format '{img.format}'. Please use PNG, JPG or JPEG.")
        arr = np.array(_exif_fix(img))

    if arr.size == 0 or arr.ndim not in (2, 3):
        raise ImageProcessingError("The image has no pixels or an unsupported number of dimensions.")
    if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
        raise ImageProcessingError("Unsupported number of colour channels.")
    if min(arr.shape[:2]) < 8:
        raise ImageProcessingError("The image is too small (minimum 8x8 pixels).")
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        a = a * 255.0 if a.max() <= 1.0 else a
        arr = np.clip(a, 0, 255).astype(np.uint8)
    return arr


def _exif_fix(img: Image.Image) -> Image.Image:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode in ("P", "LA", "I;16", "I", "F", "CMYK", "1", "L;16"):
        img = img.convert("RGBA" if "A" in img.mode or img.mode == "P" else "RGB")
    return img


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """RGB/RGBA/gray -> gray uint8. Transparent regions become background.

    For RGBA input the composite colour is chosen to contrast with the ink:
    bright strokes on transparency are composited onto black, dark strokes onto white.
    """
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 1:
        return arr[..., 0]
    if arr.shape[2] == 4:
        rgb = arr[..., :3].astype(np.float32)
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        if alpha.min() < 0.99:  # genuinely uses transparency
            lum = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
            opaque = alpha[..., 0] > 0.5
            ink_bright = opaque.any() and lum[opaque].mean() > 127
            bg = 0.0 if ink_bright else 255.0
            rgb = rgb * alpha + bg * (1.0 - alpha)
        return cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def _border_median(gray: np.ndarray) -> float:
    b = max(1, min(gray.shape) // 20)
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    return float(np.median(border))


def center_by_mass(img28: np.ndarray) -> np.ndarray:
    """Shift a 28x28 image so its intensity centre of mass is at the frame centre."""
    total = img28.sum()
    if total <= 0:
        return img28
    ys, xs = np.indices(img28.shape)
    cy = (ys * img28).sum() / total
    cx = (xs * img28).sum() / total
    # MNIST's measured mean centre of mass is (14.0, 14.0) in pixel-index coordinates
    # (see outputs/reports/eda_stats.json), i.e. size/2 — not the geometric (size-1)/2.
    shift_x = img28.shape[1] / 2.0 - cx
    shift_y = img28.shape[0] / 2.0 - cy
    m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img28, m, (img28.shape[1], img28.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


def preprocess_digit(
    source: Any,
    invert: str = "auto",
    min_component_frac: float = 0.05,
    margin_frac: float = 0.04,
) -> ProcessedImage:
    """Full pipeline: any digit image -> MNIST-style (28, 28) float32 in [0, 1].

    Args:
        source: image bytes / path / PIL / numpy array.
        invert: 'auto' (detect from background), 'yes' (dark ink on light paper),
            or 'no' (already light ink on dark background).
        min_component_frac: connected components smaller than this fraction of the
            largest component are treated as noise and removed.
        margin_frac: extra margin around the detected digit before resizing.
    """
    arr = load_image(source)
    stages: dict[str, np.ndarray] = {"original": arr}
    warnings: list[str] = []

    # 1) Grayscale
    gray = to_grayscale(arr)
    stages["grayscale"] = gray

    # 2) Polarity: MNIST is light ink on dark background
    bg_level = _border_median(gray)
    if invert == "auto":
        do_invert = bg_level > 127
    elif invert in ("yes", True):
        do_invert = True
    elif invert in ("no", False):
        do_invert = False
    else:
        raise ValueError("invert must be 'auto', 'yes' or 'no'")
    work = (255 - gray) if do_invert else gray.copy()
    stages["polarity"] = work

    # 3) Contrast normalisation: remove background level, stretch ink to full range
    work_f = cv2.GaussianBlur(work, (3, 3), 0).astype(np.float32) if max(work.shape) > 64 else work.astype(np.float32)
    bg = (255.0 - bg_level) if do_invert else bg_level
    work_f = np.clip(work_f - bg, 0, None)
    hi = float(np.percentile(work_f, 99.5)) if work_f.max() > 0 else 0.0
    if hi < 25:  # less than ~10% contrast between ink and background
        raise ImageProcessingError(
            "No digit was found. The canvas/image looks empty or has very low contrast — "
            "draw a digit (or upload a clearer image) and try again."
        )
    work_f = np.clip(work_f / hi, 0, 1)
    contrast = (work_f * 255).astype(np.uint8)
    stages["contrast"] = contrast

    # 4) Digit region detection: Otsu threshold + connected components
    _, mask = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise ImageProcessingError("No digit strokes were detected in the image.")
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= max(areas.max() * min_component_frac, 4)]
    clean_mask = np.isin(labels, keep).astype(np.uint8) * 255
    if len(keep) < len(areas):
        warnings.append(f"Removed {len(areas) - len(keep)} small noise blob(s).")
    stages["mask"] = clean_mask
    # Keep anti-aliased edges: dilate the mask slightly before applying it
    soft_mask = cv2.dilate(clean_mask, np.ones((3, 3), np.uint8), iterations=1) > 0
    contrast = np.where(soft_mask, contrast, 0).astype(np.uint8)

    ys, xs = np.where(clean_mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = x1 - x0, y1 - y0
    H, W = contrast.shape
    ink_fraction = float((clean_mask > 0).mean())
    if max(w, h) < max(6, 0.03 * max(H, W)):
        warnings.append("The digit is very small; drawing it larger usually improves accuracy.")
    if ink_fraction > 0.45:
        warnings.append(
            "Over 45% of the image is 'ink'. The background may not be uniform or the polarity "
            "may be wrong — try the invert option."
        )
    if len(keep) > 3:
        warnings.append("Several separate shapes were detected; the model expects a single digit.")

    # 5) Crop with a small margin
    m = int(round(margin_frac * max(w, h)))
    x0c, y0c = max(0, x0 - m), max(0, y0 - m)
    x1c, y1c = min(W, x1 + m), min(H, y1 + m)
    crop = contrast[y0c:y1c, x0c:x1c]
    stages["cropped"] = crop

    # 6) Resize preserving aspect ratio so the longest side is 20 px
    ch, cw = crop.shape
    scale = DIGIT_BOX / max(ch, cw)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (nw, nh), interpolation=interp)
    stages["resized_20"] = resized

    # 7) Place on a 28x28 black frame and centre by centre of mass
    frame = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    ox, oy = (IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2
    frame[oy : oy + nh, ox : ox + nw] = resized.astype(np.float32)
    stages["boxed_28"] = frame.astype(np.uint8)
    frame = center_by_mass(frame)

    # 8) Normalise to [0, 1]
    final = np.clip(frame / 255.0, 0.0, 1.0).astype(np.float32)
    stages["final_28"] = (final * 255).astype(np.uint8)

    validate_model_input(final)
    return ProcessedImage(
        image=final,
        stages=stages,
        inverted=bool(do_invert),
        bbox=(int(x0), int(y0), int(w), int(h)),
        warnings=warnings,
        info={
            "background_level": bg_level,
            "ink_fraction": ink_fraction,
            "components_kept": len(keep),
            "scale": scale,
            "input_shape": tuple(int(s) for s in arr.shape),
        },
    )


def validate_model_input(img: np.ndarray) -> None:
    """Validate a preprocessed image right before it reaches a model."""
    if img.shape != (IMG_SIZE, IMG_SIZE):
        raise ImageProcessingError(f"Processed image has shape {img.shape}, expected (28, 28).")
    if not np.isfinite(img).all():
        raise ImageProcessingError("Processed image contains invalid (NaN/inf) values.")
    if img.min() < 0 or img.max() > 1:
        raise ImageProcessingError("Processed image values must lie in [0, 1].")
    if img.max() < 0.1:
        raise ImageProcessingError("Processed image is empty — no digit detected.")


def naive_resize(source: Any) -> np.ndarray:
    """The *wrong* approach (whole image squashed to 28x28) — used for comparison only."""
    gray = to_grayscale(load_image(source))
    if _border_median(gray) > 127:
        gray = 255 - gray
    return cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def image_to_png_bytes(img: np.ndarray, upscale: int = 1) -> bytes:
    """Encode a grayscale array (uint8 or [0,1] float) as PNG bytes."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    if upscale > 1:
        a = cv2.resize(a, (a.shape[1] * upscale, a.shape[0] * upscale), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ImageProcessingError("Could not encode image.")
    return buf.tobytes()
