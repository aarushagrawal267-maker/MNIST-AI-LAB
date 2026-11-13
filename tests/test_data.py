"""Dataset loading, parsing and validation."""
import struct

import numpy as np
import pytest

from src.data_loader import find_idx_files, get_splits, load_mnist, read_idx, stratified_subset
from src.data_validation import validate_dataset


def _write_idx(path, arr):
    codes = {np.dtype(np.uint8): 0x08}
    with open(path, "wb") as fh:
        fh.write(struct.pack(">BBBB", 0, 0, codes[arr.dtype], arr.ndim))
        fh.write(struct.pack(">" + "I" * arr.ndim, *arr.shape))
        fh.write(arr.tobytes())


def test_read_idx_roundtrip(tmp_path):
    imgs = np.random.default_rng(0).integers(0, 256, (5, 28, 28), dtype=np.uint8)
    labels = np.array([0, 1, 2, 3, 9], np.uint8)
    _write_idx(tmp_path / "train-images.idx3-ubyte", imgs)
    _write_idx(tmp_path / "train-labels.idx1-ubyte", labels)
    assert np.array_equal(read_idx(tmp_path / "train-images.idx3-ubyte"), imgs)
    assert np.array_equal(read_idx(tmp_path / "train-labels.idx1-ubyte"), labels)


def test_read_idx_rejects_garbage(tmp_path):
    p = tmp_path / "bad.idx3-ubyte"
    p.write_bytes(b"\x12\x34\x56\x78" + b"\x00" * 20)
    with pytest.raises(ValueError):
        read_idx(p)


def test_read_idx_rejects_truncated(tmp_path):
    p = tmp_path / "t.idx3-ubyte"
    _write_idx(p, np.zeros((2, 28, 28), np.uint8))
    p.write_bytes(p.read_bytes()[:-10])
    with pytest.raises(ValueError, match="expected"):
        read_idx(p)


def test_find_idx_files_needs_all_four(tmp_path):
    _write_idx(tmp_path / "train-images.idx3-ubyte", np.zeros((1, 28, 28), np.uint8))
    assert find_idx_files(tmp_path) is None


def test_mnist_loads_with_expected_dimensions():
    d = load_mnist(verbose=False)
    assert d.x_train.shape == (60000, 28, 28)
    assert d.y_train.shape == (60000,)
    assert d.x_test.shape == (10000, 28, 28)
    assert d.y_test.shape == (10000,)
    assert d.x_train.dtype == np.uint8


def test_labels_are_digits():
    d = load_mnist(verbose=False)
    assert set(np.unique(d.y_train)) == set(range(10))
    assert set(np.unique(d.y_test)) == set(range(10))
    assert d.x_train.min() >= 0 and d.x_train.max() <= 255


def test_validation_report_passes():
    rep = validate_dataset(load_mnist(verbose=False))
    assert rep["all_passed"], [c for c in rep["checks"] if not c["passed"]]


def test_splits_are_stratified_and_disjoint(splits):
    assert len(splits.x_train) == 54000 and len(splits.x_val) == 6000 and len(splits.x_test) == 10000
    assert not set(splits.train_idx) & set(splits.val_idx)
    p_train = np.bincount(splits.y_train, minlength=10) / len(splits.y_train)
    p_val = np.bincount(splits.y_val, minlength=10) / len(splits.y_val)
    assert np.abs(p_train - p_val).max() < 0.002


def test_split_is_deterministic(splits):
    again = get_splits()
    assert np.array_equal(again.val_idx, splits.val_idx)


def test_stratified_subset(splits):
    x, y = stratified_subset(splits.x_train, splits.y_train, 1000)
    assert len(x) == len(y) == 1000
    assert np.bincount(y, minlength=10).min() > 80
