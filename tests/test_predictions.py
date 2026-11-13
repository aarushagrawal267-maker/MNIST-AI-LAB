"""End-to-end predictions with the trained models (TRAX and all registered models)."""
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from src import model_manager as mm
from src.evaluate import compute_metrics, predict_proba
from src.prediction import Predictor, expand_uploads
from tests.conftest import draw_digit


def _png(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


def test_probabilities_are_valid(predictor, splits):
    p = predictor.predict_array(splits.x_test[:256])
    assert p.shape == (256, 10)
    assert (p >= 0).all() and (p <= 1).all()
    assert np.allclose(p.sum(1), 1, atol=1e-5)
    assert set(p.argmax(1)).issubset(set(range(10)))


def test_trax_accuracy_on_test_sample(predictor, splits):
    p = predictor.predict_array(splits.x_test[:1000])
    assert (p.argmax(1) == splits.y_test[:1000]).mean() > 0.97


def test_wrong_input_shape_rejected(predictor):
    with pytest.raises(ValueError):
        predictor.predict_array(np.zeros((2, 32, 32)))


@pytest.mark.parametrize("digit", [0, 1, 2, 3, 4, 7])
def test_drawn_digits_are_recognised(predictor, digit):
    pred = predictor.predict_image(draw_digit(digit))
    assert pred.digit == digit, f"drew {digit}, model said {pred.digit} ({pred.top3})"
    assert 0 <= pred.confidence <= 1
    assert len(pred.top3) == 3 and pred.top3[0][0] == pred.digit


def test_drawn_digit_robust_to_position_and_size(predictor):
    for offset, scale in [((-0.25, -0.2), 0.45), ((0.2, 0.25), 0.5), ((0, 0), 1.2)]:
        assert predictor.predict_image(draw_digit(7, offset=offset, scale=scale, thickness=14)).digit == 7


def test_batch_prediction_mixed_inputs(predictor):
    items = [("a.png", _png(draw_digit(1))), ("broken.png", b"not an image"), ("b.png", _png(draw_digit(7))),
             ("empty.png", _png(np.zeros((64, 64, 4), np.uint8)))]
    rows = predictor.predict_batch(items)
    assert [r["file"] for r in rows] == ["a.png", "broken.png", "b.png", "empty.png"]
    status = {r["file"]: r for r in rows}
    assert status["a.png"]["prediction"] == 1 and status["b.png"]["prediction"] == 7
    assert status["broken.png"]["status"] == "error" and "decoded" in status["broken.png"]["error"]
    assert status["empty.png"]["status"] == "error"
    for r in rows:
        if r["status"] == "ok":
            assert np.isclose(r["probabilities"].sum(), 1, atol=1e-5)


def test_zip_expansion_filters_files():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("d/one.png", _png(draw_digit(1)))
        zf.writestr("d/notes.txt", "hello")
        zf.writestr("__MACOSX/._one.png", "junk")
    items, rejected = expand_uploads([("set.zip", buf.getvalue()), ("bad.zip", b"PK?"), ("x.bmp", b"..")])
    assert [n for n, _ in items] == ["set.zip/d/one.png"]
    reasons = {r["file"]: r["error"] for r in rejected}
    assert "Unsupported" in reasons["set.zip/d/notes.txt"]
    assert "invalid ZIP" in reasons["bad.zip"]
    assert "Unsupported" in reasons["x.bmp"]


def test_every_registered_model_loads_and_predicts(splits):
    models = {k: v for k, v in mm.list_models().items() if not v.get("custom")}
    if not models:
        pytest.skip("no trained models")
    for key, entry in models.items():
        model = mm.load_model(key)
        p = predict_proba(model, entry["family"], entry["input_kind"], splits.x_val[:300])
        assert p.shape == (300, 10) and np.allclose(p.sum(1), 1, atol=1e-4)
        assert (p.argmax(1) == splits.y_val[:300]).mean() > 0.85, key


def test_compute_metrics_perfect_and_shapes():
    y = np.repeat(np.arange(10), 5)
    proba = np.eye(10)[y] * 0.9 + 0.01
    m = compute_metrics(y, proba)
    assert m["accuracy"] == 1.0 and m["f1_macro"] == 1.0
    assert np.array(m["confusion_matrix"]).shape == (10, 10)
    assert len(m["per_class_accuracy"]) == 10


def test_predictor_rejects_unknown_model():
    with pytest.raises(KeyError):
        Predictor("does_not_exist")
