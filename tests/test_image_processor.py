"""Drawing-canvas / upload preprocessing: grayscale, crop, resize, centre, normalise, validate."""
import io

import numpy as np
import pytest
from PIL import Image

from src.image_processor import ImageProcessingError, center_by_mass, naive_resize, preprocess_digit
from tests.conftest import draw_digit


def _com(img):
    ys, xs = np.indices(img.shape)
    return (xs * img).sum() / img.sum(), (ys * img).sum() / img.sum()


def test_output_format():
    p = preprocess_digit(draw_digit(7))
    assert p.image.shape == (28, 28)
    assert p.image.dtype == np.float32
    assert 0.0 <= p.image.min() and p.image.max() <= 1.0
    assert p.image.max() > 0.5


def test_off_centre_drawing_is_centred():
    p = preprocess_digit(draw_digit(4, offset=(-0.25, -0.22), scale=0.4))
    cx, cy = _com(p.image)
    assert abs(cx - 14) < 1.0 and abs(cy - 14) < 1.0


def test_digit_is_resized_to_20px_box():
    p = preprocess_digit(draw_digit(0, scale=0.3))
    ys, xs = np.where(p.image > 0.2)
    longest = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    assert 18 <= longest <= 22


def test_aspect_ratio_preserved():
    p = preprocess_digit(draw_digit(1))
    ys, xs = np.where(p.image > 0.2)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    assert h >= 18 and w <= 8  # a tall thin "1" stays tall and thin


def test_dark_ink_on_white_is_inverted():
    light = preprocess_digit(draw_digit(7))
    dark = preprocess_digit(draw_digit(7, ink=20, bg=245))
    assert dark.inverted and not light.inverted
    assert np.corrcoef(light.image.ravel(), dark.image.ravel())[0, 1] > 0.95


def test_explicit_polarity_override():
    p = preprocess_digit(draw_digit(7), invert="no")
    assert not p.inverted
    with pytest.raises(ValueError):
        preprocess_digit(draw_digit(7), invert="maybe")


def test_transparent_png_with_black_strokes():
    rgba = draw_digit(7)
    alpha = rgba[..., 0].copy()  # ink where white strokes were
    img = np.dstack([np.zeros_like(alpha)] * 3 + [alpha])  # black strokes on transparent background
    p = preprocess_digit(img)
    ref = preprocess_digit(draw_digit(7))
    assert np.corrcoef(p.image.ravel(), ref.image.ravel())[0, 1] > 0.95


def test_noise_speck_is_ignored():
    rgba = draw_digit(1)
    rgba[5:8, 5:8, :3] = 255  # tiny speck far from the digit
    p = preprocess_digit(rgba)
    assert p.bbox[0] > 50  # bounding box starts at the digit, not the speck


def test_png_bytes_and_jpeg_input():
    for fmt in ("PNG", "JPEG"):
        buf = io.BytesIO()
        Image.fromarray(draw_digit(3)[..., :3]).save(buf, format=fmt)
        assert preprocess_digit(buf.getvalue()).image.shape == (28, 28)


def test_empty_canvas_has_clear_error():
    with pytest.raises(ImageProcessingError, match="empty|No digit"):
        preprocess_digit(np.zeros((336, 336, 4), np.uint8))


def test_invalid_file_has_clear_error():
    with pytest.raises(ImageProcessingError, match="could not be decoded"):
        preprocess_digit(b"definitely not an image")
    with pytest.raises(ImageProcessingError, match="empty"):
        preprocess_digit(b"")


def test_unsupported_extension(tmp_path):
    p = tmp_path / "digit.gif"
    p.write_bytes(b"GIF89a")
    with pytest.raises(ImageProcessingError, match="Unsupported"):
        preprocess_digit(p)


def test_tiny_image_rejected():
    with pytest.raises(ImageProcessingError, match="too small"):
        preprocess_digit(np.zeros((4, 4), np.uint8))


def test_center_by_mass():
    img = np.zeros((28, 28), np.float32)
    img[2:6, 2:6] = 1.0
    cx, cy = _com(center_by_mass(img))
    assert abs(cx - 14) < 0.6 and abs(cy - 14) < 0.6


def test_naive_resize_differs_from_pipeline():
    rgba = draw_digit(4, offset=(-0.25, -0.22), scale=0.4)
    naive = naive_resize(rgba)
    proper = preprocess_digit(rgba).image
    assert naive.shape == proper.shape == (28, 28)
    assert (naive > 0.2).sum() < (proper > 0.2).sum()  # naive shrinks the digit to few pixels
