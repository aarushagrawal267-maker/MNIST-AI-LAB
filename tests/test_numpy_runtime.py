"""TensorFlow-free runtime: exported NumPy models must reproduce the Keras models exactly."""
import numpy as np
import pytest

from src import model_manager as mm
from src import numpy_runtime as nr
from src.preprocessing import to_cnn_input


def _keras_models():
    return {k: v for k, v in mm.list_models().items() if v["family"] == "keras" and not v.get("custom")}


def test_every_keras_model_has_numpy_export():
    ms = _keras_models()
    if not ms:
        pytest.skip("no trained models")
    for k, e in ms.items():
        assert nr.npz_path(mm.abs_path(e["path"])).exists(), k


@pytest.mark.skipif(not nr.HAS_TENSORFLOW, reason="needs TensorFlow for the reference")
def test_numpy_matches_keras(splits):
    import keras

    x = to_cnn_input(splits.x_test[:500])
    for k, e in _keras_models().items():
        ref = keras.models.load_model(mm.abs_path(e["path"]), compile=False).predict(x, verbose=0)
        out = nr.NumpyModel(nr.npz_path(mm.abs_path(e["path"]))).predict(x)
        assert np.abs(ref - out).max() < 1e-4, k
        assert (ref.argmax(1) == out.argmax(1)).all(), k


def test_loader_falls_back_without_tensorflow(monkeypatch, splits):
    if mm.production_key() is None:
        pytest.skip("no trained models")
    monkeypatch.setattr(nr, "HAS_TENSORFLOW", False)
    mm._CACHE.clear()
    try:
        from src.prediction import Predictor
        from tests.conftest import draw_digit

        p = Predictor()
        assert isinstance(p.model, nr.NumpyModel)
        proba = p.predict_array(splits.x_test[:300])
        assert np.allclose(proba.sum(1), 1, atol=1e-5)
        assert (proba.argmax(1) == splits.y_test[:300]).mean() > 0.97
        assert p.predict_image(draw_digit(7)).digit == 7
    finally:
        mm._CACHE.clear()


def test_opencv_augmentation_fallback(monkeypatch, splits):
    from src import augmentation

    monkeypatch.setattr(nr, "HAS_TENSORFLOW", False)
    out = augmentation.augment_images(splits.x_train[:3], n_variants=4)
    assert out.shape == (12, 28, 28) and out.min() >= 0 and out.max() <= 1
    same = augmentation.augment_images(splits.x_train[:2], rotation_degrees=0, translation=0, zoom=0)
    assert np.allclose(same, splits.x_train[:2] / 255.0, atol=1e-6)
