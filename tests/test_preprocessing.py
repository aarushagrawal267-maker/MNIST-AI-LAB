"""Tensor preprocessing and augmentation."""
import numpy as np
import pytest

from src.augmentation import augment_images
from src.preprocessing import normalize, one_hot, prepare, to_cnn_input, to_flat_input


def test_normalize_range_and_dtype():
    x = np.array([[0, 128, 255]], np.uint8)
    n = normalize(x)
    assert n.dtype == np.float32
    assert n.min() == 0.0 and n.max() == 1.0
    assert np.allclose(normalize(n), n)  # idempotent on [0, 1] floats


def test_cnn_input_shapes():
    assert to_cnn_input(np.zeros((28, 28), np.uint8)).shape == (1, 28, 28, 1)
    assert to_cnn_input(np.zeros((5, 28, 28), np.uint8)).shape == (5, 28, 28, 1)
    with pytest.raises(ValueError):
        to_cnn_input(np.zeros((5, 32, 32), np.uint8))


def test_flat_input_shapes():
    assert to_flat_input(np.zeros((3, 28, 28), np.uint8)).shape == (3, 784)
    assert prepare(np.zeros((2, 28, 28)), "flat").shape == (2, 784)
    assert prepare(np.zeros((2, 28, 28)), "image").shape == (2, 28, 28, 1)
    with pytest.raises(ValueError):
        prepare(np.zeros((2, 28, 28)), "nope")


def test_one_hot():
    oh = one_hot(np.array([0, 3, 9]))
    assert oh.shape == (3, 10)
    assert np.allclose(oh.sum(1), 1)
    assert oh[1, 3] == 1
    with pytest.raises(ValueError):
        one_hot(np.array([10]))


def test_augmentation_shape_and_range(splits):
    out = augment_images(splits.x_train[:4], n_variants=3)
    assert out.shape == (12, 28, 28)
    assert out.min() >= 0 and out.max() <= 1
    assert not np.allclose(out[0], splits.x_train[0] / 255.0)  # actually transformed


def test_augmentation_disabled_is_identity(splits):
    out = augment_images(splits.x_train[:2], rotation_degrees=0, translation=0, zoom=0)
    assert np.allclose(out, splits.x_train[:2] / 255.0, atol=1e-6)
