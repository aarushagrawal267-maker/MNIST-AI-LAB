"""MNIST dataset acquisition.

Primary source: the Kaggle dataset ``hojjatk/mnist-dataset`` (original IDX
binary files), downloaded with ``kagglehub``. Fallback: the Keras built-in
``mnist.load_data()`` mirror. Both are cached locally so the download happens
only once.

The original 10,000-image test set is kept completely separate; the
validation set is a stratified split of the 60,000 training images.
"""
from __future__ import annotations

import gzip
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CFG, DATA_PROCESSED_DIR, DATA_RAW_DIR, SEED

CACHE_PATH = DATA_PROCESSED_DIR / "mnist.npz"
SPLIT_PATH = DATA_PROCESSED_DIR / "split_indices.npz"

# Kaggle ships both "train-images.idx3-ubyte" and "train-images-idx3-ubyte".
IDX_FILES: dict[str, list[str]] = {
    "x_train": ["train-images.idx3-ubyte", "train-images-idx3-ubyte"],
    "y_train": ["train-labels.idx1-ubyte", "train-labels-idx1-ubyte"],
    "x_test": ["t10k-images.idx3-ubyte", "t10k-images-idx3-ubyte"],
    "y_test": ["t10k-labels.idx1-ubyte", "t10k-labels-idx1-ubyte"],
}

MANUAL_INSTRUCTIONS = f"""
Could not obtain the MNIST dataset automatically.

Manual setup (any ONE of these works):
  1. Download https://www.kaggle.com/datasets/hojjatk/mnist-dataset (click "Download"),
     unzip it, and copy these four files into:  {DATA_RAW_DIR}
        train-images.idx3-ubyte   train-labels.idx1-ubyte
        t10k-images.idx3-ubyte    t10k-labels.idx1-ubyte
     (the "-idx3-ubyte" spellings and .gz versions are also accepted)
  2. Or download https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
     and place it at:  {Path.home() / '.keras' / 'datasets' / 'mnist.npz'}
Then restart the app or rerun the training command.
""".strip()


class DatasetUnavailableError(RuntimeError):
    """Raised when MNIST cannot be downloaded or found locally."""


@dataclass
class MnistData:
    x_train: np.ndarray  # (60000, 28, 28) uint8
    y_train: np.ndarray  # (60000,) uint8
    x_test: np.ndarray   # (10000, 28, 28) uint8
    y_test: np.ndarray   # (10000,) uint8
    source: str


@dataclass
class MnistSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    source: str


# --------------------------------------------------------------------------- #
# IDX parsing
# --------------------------------------------------------------------------- #
_IDX_DTYPES = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4", 0x0D: ">f4", 0x0E: ">f8"}


def read_idx(path: str | Path) -> np.ndarray:
    """Parse an IDX file (the binary format used by the original MNIST release)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 4:
        raise ValueError(f"{path.name}: file too small to be an IDX file")
    zero1, zero2, dtype_code, ndim = struct.unpack(">BBBB", raw[:4])
    if zero1 != 0 or zero2 != 0 or dtype_code not in _IDX_DTYPES:
        raise ValueError(f"{path.name}: invalid IDX magic number")
    dims = struct.unpack(">" + "I" * ndim, raw[4 : 4 + 4 * ndim])
    dtype = np.dtype(_IDX_DTYPES[dtype_code])
    expected = int(np.prod(dims)) * dtype.itemsize
    payload = raw[4 + 4 * ndim :]
    if len(payload) != expected:
        raise ValueError(f"{path.name}: expected {expected} data bytes, found {len(payload)}")
    return np.frombuffer(payload, dtype=dtype).reshape(dims).astype(dtype.newbyteorder("="))


def find_idx_files(root: str | Path) -> dict[str, Path] | None:
    """Locate the four MNIST IDX files anywhere under ``root``."""
    root = Path(root)
    if not root.exists():
        return None
    found: dict[str, Path] = {}
    for key, names in IDX_FILES.items():
        for name in names:
            hits = [p for p in root.rglob(name) if p.is_file()]
            hits += [p for p in root.rglob(name + ".gz") if p.is_file()]
            if hits:
                found[key] = sorted(hits, key=lambda p: len(p.parts))[0]
                break
    return found if len(found) == 4 else None


def load_from_idx(files: dict[str, Path]) -> dict[str, np.ndarray]:
    return {key: read_idx(path) for key, path in files.items()}


# --------------------------------------------------------------------------- #
# Download strategies
# --------------------------------------------------------------------------- #
def download_from_kaggle(dataset: str | None = None, verbose: bool = True) -> dict[str, Path] | None:
    """Download the Kaggle MNIST dataset and copy its IDX files into data/raw."""
    dataset = dataset or CFG["data"]["kaggle_dataset"]
    try:
        import kagglehub
    except ImportError:
        if verbose:
            print("[data] kagglehub not installed - skipping Kaggle download")
        return None
    try:
        if verbose:
            print(f"[data] Downloading Kaggle dataset '{dataset}' ...")
        downloaded = Path(kagglehub.dataset_download(dataset))
    except Exception as exc:  # network / auth problems
        if verbose:
            print(f"[data] Kaggle download failed: {exc}")
        return None
    files = find_idx_files(downloaded)
    if not files:
        if verbose:
            print("[data] Kaggle download did not contain the expected IDX files")
        return None
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for key, src in files.items():
        dst = DATA_RAW_DIR / IDX_FILES[key][0]
        if src.suffix == ".gz":
            dst = dst.with_suffix(dst.suffix + ".gz")
        if not dst.exists():
            shutil.copyfile(src, dst)
        copied[key] = dst
    return copied


def download_from_keras(verbose: bool = True) -> dict[str, np.ndarray] | None:
    try:
        from keras.datasets import mnist

        if verbose:
            print("[data] Falling back to keras.datasets.mnist ...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
    except Exception as exc:
        if verbose:
            print(f"[data] Keras download failed: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_mnist(force_reload: bool = False, verbose: bool = True) -> MnistData:
    """Load MNIST, downloading and caching it if necessary.

    Order: local cache -> IDX files in data/raw -> Kaggle download -> Keras mirror.
    """
    if CACHE_PATH.exists() and not force_reload:
        with np.load(CACHE_PATH, allow_pickle=False) as npz:
            return MnistData(
                npz["x_train"], npz["y_train"], npz["x_test"], npz["y_test"], str(npz["source"])
            )

    arrays: dict[str, np.ndarray] | None = None
    source = ""
    files = find_idx_files(DATA_RAW_DIR)
    if files:
        arrays, source = load_from_idx(files), "kaggle:hojjatk/mnist-dataset (local IDX files)"
    if arrays is None:
        files = download_from_kaggle(verbose=verbose)
        if files:
            arrays, source = load_from_idx(files), "kaggle:hojjatk/mnist-dataset"
    if arrays is None:
        arrays = download_from_keras(verbose=verbose)
        source = "keras.datasets.mnist"
    if arrays is None:
        raise DatasetUnavailableError(MANUAL_INSTRUCTIONS)

    data = MnistData(
        arrays["x_train"].astype(np.uint8),
        arrays["y_train"].astype(np.uint8),
        arrays["x_test"].astype(np.uint8),
        arrays["y_test"].astype(np.uint8),
        source,
    )
    from .data_validation import assert_valid

    assert_valid(data)
    np.savez_compressed(
        CACHE_PATH,
        x_train=data.x_train,
        y_train=data.y_train,
        x_test=data.x_test,
        y_test=data.y_test,
        source=np.array(source),
    )
    if verbose:
        print(f"[data] Cached MNIST from {source} -> {CACHE_PATH}")
    return data


def get_splits(
    val_size: float | None = None, seed: int = SEED, data: MnistData | None = None
) -> MnistSplits:
    """Return stratified train/validation splits plus the untouched test set."""
    from sklearn.model_selection import train_test_split

    val_size = float(val_size if val_size is not None else CFG["data"]["validation_size"])
    data = data or load_mnist(verbose=False)

    train_idx = val_idx = None
    if SPLIT_PATH.exists():
        with np.load(SPLIT_PATH) as npz:
            if float(npz["val_size"]) == val_size and int(npz["seed"]) == seed:
                train_idx, val_idx = npz["train_idx"], npz["val_idx"]
    if train_idx is None:
        idx = np.arange(len(data.y_train))
        train_idx, val_idx = train_test_split(
            idx, test_size=val_size, random_state=seed, stratify=data.y_train
        )
        train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)
        np.savez(SPLIT_PATH, train_idx=train_idx, val_idx=val_idx, val_size=val_size, seed=seed)

    return MnistSplits(
        x_train=data.x_train[train_idx],
        y_train=data.y_train[train_idx],
        x_val=data.x_train[val_idx],
        y_val=data.y_train[val_idx],
        x_test=data.x_test,
        y_test=data.y_test,
        train_idx=train_idx,
        val_idx=val_idx,
        source=data.source,
    )


def stratified_subset(x: np.ndarray, y: np.ndarray, n: int, seed: int = SEED):
    """Return a class-balanced random subset of size ``n`` (or everything if n >= len)."""
    if n is None or n >= len(y):
        return x, y
    from sklearn.model_selection import train_test_split

    xs, _, ys, _ = train_test_split(x, y, train_size=n, random_state=seed, stratify=y)
    return xs, ys


if __name__ == "__main__":  # python -m src.data_loader
    d = load_mnist()
    s = get_splits(data=d)
    print(f"source={d.source}")
    print(f"train={s.x_train.shape} val={s.x_val.shape} test={s.x_test.shape}")
