"""Model architectures, training, persistence and explainability."""
import numpy as np
import pytest

from src.models import CNNConfig, baseline_cnn_config, build_ann, build_cnn, build_logistic_regression, last_conv_layer_name


def test_baseline_cnn_architecture():
    m = build_cnn(baseline_cnn_config())
    assert m.input_shape == (None, 28, 28, 1)
    assert m.output_shape == (None, 10)
    names = [l.__class__.__name__ for l in m.layers]
    assert names.count("Conv2D") == 2 and names.count("MaxPooling2D") == 2 and "Dropout" in names
    assert m.count_params() == 225_034
    assert last_conv_layer_name(m) == "conv2_1"


def test_ann_architecture():
    m = build_ann()
    dense = [l for l in m.layers if l.__class__.__name__ == "Dense"]
    assert [d.units for d in dense] == [256, 128, 10]
    assert m.count_params() == 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10


@pytest.mark.parametrize("cfg", [
    CNNConfig(filters=[16, 32], convs_per_block=2, batchnorm=True, optimizer="sgd"),
    CNNConfig(filters=[32, 64, 128], dense_units=64, optimizer="rmsprop"),
])
def test_cnn_variants_build(cfg):
    m = build_cnn(cfg)
    out = np.asarray(m(np.zeros((2, 28, 28, 1), np.float32)))
    assert out.shape == (2, 10)
    assert np.allclose(out.sum(1), 1, atol=1e-5)


def test_cnn_trains_and_roundtrips(tmp_path, splits):
    from src.train import train_keras

    m = build_cnn(baseline_cnn_config())
    path = tmp_path / "cnn.keras"
    res = train_keras(m, splits, path, epochs=1, batch_size=64, train_subset=3000, verbose=0)
    assert res["val_accuracy"] > 0.8  # far above 10% chance after one short epoch
    assert len(res["history"]["learning_rate"]) == 1
    import keras

    loaded = keras.models.load_model(path, compile=False)
    x = splits.x_val[:16].astype(np.float32)[..., None] / 255.0
    assert np.allclose(np.asarray(res["model"](x)), np.asarray(loaded(x)), atol=1e-5)


def test_sklearn_model_trains_and_roundtrips(tmp_path, splits):
    import joblib

    from src.preprocessing import to_flat_input

    m = build_logistic_regression(max_iter=50)
    x, y = to_flat_input(splits.x_train[:2000]), splits.y_train[:2000]
    m.fit(x, y)
    joblib.dump(m, tmp_path / "lr.joblib")
    m2 = joblib.load(tmp_path / "lr.joblib")
    xv = to_flat_input(splits.x_val[:200])
    assert np.allclose(m.predict_proba(xv), m2.predict_proba(xv))
    assert (m2.predict(xv) == splits.y_val[:200]).mean() > 0.75


def test_explainability_methods(predictor, splits):
    from src import explainability as X

    if predictor.entry["family"] != "keras":
        pytest.skip("explainability requires a Keras model")
    img = splits.x_test[0].astype(np.float32) / 255.0
    c = int(predictor.predict_array(img)[0].argmax())
    for fn in (X.saliency_map, X.smoothgrad):
        mp = fn(predictor.model, img, c)
        assert mp.shape == (28, 28) and mp.min() >= 0 and mp.max() <= 1
    ig, err = X.integrated_gradients(predictor.model, img, c)
    assert ig.shape == (28, 28) and err < 0.15
    cam = X.grad_cam(predictor.model, img, c)
    assert cam is not None and cam.shape == (28, 28)
    occ = X.occlusion_sensitivity(predictor.predict_array, img, c)
    assert occ.shape == (28, 28) and occ.min() >= 0
    assert X.overlay(img, cam).shape == (28, 28, 3)
