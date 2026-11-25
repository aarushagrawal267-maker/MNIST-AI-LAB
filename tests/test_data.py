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


def test_corrupt_split_cache_is_rebuilt(tmp_path, monkeypatch, splits):
    """A truncated split_indices.npz must not break every page; the same split is recomputed."""
    import src.data_loader as dl

    bad = tmp_path / "split_indices.npz"
    bad.write_bytes(b"PK\x03\x04 truncated")
    monkeypatch.setattr(dl, "SPLIT_PATH", bad)
    s = get_splits()
    np.testing.assert_array_equal(s.val_idx, splits.val_idx)
    np.testing.assert_array_equal(s.train_idx, splits.train_idx)
    with np.load(bad) as npz:  # rewritten as a valid cache
        np.testing.assert_array_equal(npz["val_idx"], splits.val_idx)


def test_split_cache_from_other_dataset_is_ignored(tmp_path, monkeypatch, splits):
    """Cached indices that do not partition the current training set are not reused."""
    import src.data_loader as dl

    stale = tmp_path / "split_indices.npz"
    np.savez(stale, train_idx=np.arange(10), val_idx=np.arange(10, 12), val_size=0.1, seed=dl.SEED)
    monkeypatch.setattr(dl, "SPLIT_PATH", stale)
    s = get_splits()
    assert len(s.train_idx) + len(s.val_idx) == 60000
    assert len(np.intersect1d(s.train_idx, s.val_idx)) == 0


def test_corrupt_dataset_cache_falls_back(tmp_path, monkeypatch):
    """An unreadable mnist.npz cache is ignored instead of raising on every load."""
    import src.data_loader as dl

    good = load_mnist(verbose=False)
    bad = tmp_path / "mnist.npz"
    bad.write_bytes(b"not a zip")
    monkeypatch.setattr(dl, "CACHE_PATH", bad)
    monkeypatch.setattr(dl, "find_idx_files", lambda root: None)
    monkeypatch.setattr(dl, "download_from_kaggle", lambda verbose=True: None)
    monkeypatch.setattr(dl, "download_from_keras", lambda verbose=True: {
        "x_train": good.x_train, "y_train": good.y_train, "x_test": good.x_test, "y_test": good.y_test})
    d = load_mnist(verbose=False)
    assert d.x_train.shape == (60000, 28, 28)
    with np.load(bad) as npz:  # cache rewritten atomically and readable
        assert npz["x_test"].shape == (10000, 28, 28)
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_cache_write_is_thread_safe(tmp_path):
    """Streamlit sessions are threads of one process: concurrent cache writes must not collide."""
    import threading

    from src.data_loader import _atomic_savez

    path, errors = tmp_path / "cache.npz", []
    x = np.arange(200_000, dtype=np.int64)

    def write():
        try:
            for _ in range(10):
                _atomic_savez(path, compressed=True, x=x)
        except Exception as exc:  # noqa: BLE001 - surfaced through the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with np.load(path) as npz:
        np.testing.assert_array_equal(npz["x"], x)
    assert not list(tmp_path.glob("*.tmp"))
