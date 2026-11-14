"""Independent verification that TRAX is our own trained model and that it works.

    python -m src.verify

Every check recomputes its result from the model file on disk and the raw dataset;
nothing is read from the saved metrics except for the final "matches saved report" check.
Results are printed and written to outputs/reports/trax_verification.json.
"""
from __future__ import annotations

import hashlib
import json
import time

import cv2
import numpy as np

from . import model_manager as mm
from .config import HISTORY_DIR, MODEL_NAME, REPORTS_DIR, set_global_seed
from .data_loader import get_splits, load_mnist
from .evaluate import load_metrics, predict_proba

OUT = REPORTS_DIR / "trax_verification.json"


def _acc(p, y):
    return float((p.argmax(1) == y).mean())


def _shift(imgs, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return np.stack([cv2.warpAffine(i, m, (28, 28), borderValue=0) for i in imgs])


def _rotate(imgs, deg):
    m = cv2.getRotationMatrix2D((14, 14), deg, 1.0)
    return np.stack([cv2.warpAffine(i, m, (28, 28), borderValue=0) for i in imgs])


def run() -> dict:
    import keras

    from .models import CNNConfig, build_cnn

    checks: list[dict] = []

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)

    key = mm.production_key()
    if key is None:
        raise SystemExit("No production model registered. Run `python -m src.train --all` first.")
    entry = mm.get_entry(key)
    path = mm.abs_path(entry["path"])
    data = load_mnist(verbose=False)
    s = get_splits(data=data)
    print(f"\n{MODEL_NAME} = '{key}' ({entry['display_name']}) -> {mm.rel(path)}\n", flush=True)

    # 1) Provenance --------------------------------------------------------
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    hist = json.loads((HISTORY_DIR / f"{key}.json").read_text()) if (HISTORY_DIR / f"{key}.json").exists() else None
    check("Model file is local and loads from disk",
          path.exists(), f"{mm.rel(path)} · {path.stat().st_size / 1024:.0f} KB · sha256 {sha[:16]}…")
    check("Training history exists for this model (it was trained here, epoch by epoch)",
          hist is not None and len(hist["loss"]) == entry["epochs_trained"],
          f"{len(hist['loss']) if hist else 0} epochs logged; train loss {hist['loss'][0]:.3f} → {hist['loss'][-1]:.4f}, "
          f"val acc {hist['val_accuracy'][0] * 100:.2f}% → {max(hist['val_accuracy']) * 100:.2f}%" if hist else "missing")
    check("Dataset is the Kaggle MNIST release", "kaggle" in data.source, data.source)

    model = keras.models.load_model(path, compile=False)  # fresh load, bypasses any app cache

    # 2) Learned vs untrained ---------------------------------------------
    set_global_seed()
    untrained = build_cnn(CNNConfig(**entry["hyperparameters"]))
    p_untrained = predict_proba(untrained, "keras", "image", s.x_test)
    p = predict_proba(model, "keras", "image", s.x_test)
    acc_test = _acc(p, s.y_test)
    check("Same architecture untrained ≈ random guessing (proves accuracy comes from training)",
          _acc(p_untrained, s.y_test) < 0.2, f"untrained {_acc(p_untrained, s.y_test) * 100:.2f}% vs trained {acc_test * 100:.2f}%")
    w_tr = np.concatenate([w.ravel() for w in model.get_weights()])
    w_un = np.concatenate([w.ravel() for w in untrained.get_weights()])
    check("Weights differ from a fresh initialisation", not np.allclose(w_tr, w_un),
          f"{len(w_tr):,} weights, mean |Δ| = {np.abs(w_tr - w_un).mean():.4f}")

    # 3) Fresh evaluation ---------------------------------------------------
    acc_val = _acc(predict_proba(model, "keras", "image", s.x_val), s.y_val)
    check("Test accuracy recomputed from scratch on all 10,000 held-out images", acc_test > 0.99,
          f"{acc_test * 100:.2f}% ({int((p.argmax(1) != s.y_test).sum())} errors)")
    saved = load_metrics(key, "test")
    check("Recomputed test accuracy matches the saved report", saved and abs(saved["accuracy"] - acc_test) < 1e-9,
          f"recomputed {acc_test * 100:.2f}% vs saved {saved['accuracy'] * 100:.2f}%" if saved else "no saved report")
    check("Validation accuracy recomputed", acc_val > 0.99, f"{acc_val * 100:.2f}% on 6,000 images")
    per_class = [float((p.argmax(1)[s.y_test == c] == c).mean()) for c in range(10)]
    check("Every digit is recognised well (per-class accuracy > 98%)", min(per_class) > 0.98,
          " ".join(f"{c}:{a * 100:.1f}%" for c, a in enumerate(per_class)))

    # 4) No data leakage ---------------------------------------------------
    tr_hash = {hashlib.md5(x.tobytes()).hexdigest() for x in data.x_train}
    dup = sum(hashlib.md5(x.tobytes()).hexdigest() in tr_hash for x in data.x_test)
    check("Test images are not in the training data", dup == 0, f"{dup} of 10,000 test images also appear in the training set")
    check("Validation split is disjoint from the training split",
          len(set(s.train_idx) & set(s.val_idx)) == 0, f"{len(s.train_idx):,} train / {len(s.val_idx):,} val indices, overlap 0")

    # 5) Output validity ---------------------------------------------------
    check("Probabilities valid (0–1, sum to 1, classes 0–9)",
          (p >= 0).all() and (p <= 1).all() and np.allclose(p.sum(1), 1, atol=1e-5) and set(p.argmax(1)) <= set(range(10)),
          f"max |sum−1| = {np.abs(p.sum(1) - 1).max():.2e}")
    p2 = predict_proba(model, "keras", "image", s.x_test[:500])
    check("Deterministic: same input → same output", np.array_equal(p[:500], p2), "500 images predicted twice")
    tmp = REPORTS_DIR / "_verify_roundtrip.keras"
    model.save(tmp)
    p3 = predict_proba(keras.models.load_model(tmp, compile=False), "keras", "image", s.x_test[:500])
    tmp.unlink()
    check("Save → reload gives identical predictions", np.allclose(p[:500], p3, atol=1e-6), "round-trip through .keras")

    # 6) Robustness on perturbed test images (never trained on) ------------
    sub, ysub = s.x_test[:2000], s.y_test[:2000]
    raw_shift = _acc(predict_proba(model, "keras", "image", _shift(sub, 2, 2)), ysub)
    from .image_processor import preprocess_digit

    shifted = _shift(sub[:1000], 3, 2)
    piped = np.stack([preprocess_digit(cv2.resize(i, (280, 280), interpolation=cv2.INTER_CUBIC)).image for i in shifted])
    a_pipe = _acc(predict_proba(model, "keras", "image", piped), ysub[:1000])
    check("Off-centre digits through the app pipeline (re-centred before prediction)", a_pipe > 0.97,
          f"{a_pipe * 100:.2f}% on 1,000 test digits shifted 3 px and upscaled like a canvas drawing")
    print(f"[INFO] Raw-tensor shift sensitivity: fed directly as un-centred 28×28 tensors (bypassing the app's "
          f"preprocessing), digits shifted 2 px score {raw_shift * 100:.2f}%. TRAX was trained without augmentation, so it "
          "relies on MNIST-style centring — which the app always applies. The augmented model is the shift-robust alternative.",
          flush=True)
    rob = {
        "rotate +12°": _rotate(sub, 12),
        "rotate −12°": _rotate(sub, -12),
        "gaussian noise σ=25": np.clip(sub + np.random.default_rng(0).normal(0, 25, sub.shape), 0, 255).astype(np.uint8),
        "thicker strokes (dilate)": np.stack([cv2.dilate(i, np.ones((2, 2), np.uint8)) for i in sub]),
    }
    for name, imgs in rob.items():
        a = _acc(predict_proba(model, "keras", "image", imgs), ysub)
        check(f"Robust to {name}", a > 0.95, f"{a * 100:.2f}% on 2,000 perturbed test images")

    # 7) Canvas-style drawings through the full app pipeline --------------
    from .report import evaluate_drawn_digits

    d = evaluate_drawn_digits()
    check("Synthetic canvas drawings, full preprocessing pipeline", d["accuracy"] >= 0.95,
          f"{d['accuracy'] * 100:.1f}% of {d['n']} drawings (10 digits × 12 position/size/brush variants)")

    # 8) Honest limitation probe ------------------------------------------
    noise = np.random.default_rng(1).integers(0, 256, (200, 28, 28)).astype(np.uint8)
    pn = predict_proba(model, "keras", "image", noise)
    print(f"[INFO] Limitation: on 200 pure-noise images TRAX still answers a digit (mean confidence "
          f"{pn.max(1).mean() * 100:.1f}%) — it has no 'not a digit' class.", flush=True)

    report = {
        "model_key": key, "file": mm.rel(path), "sha256": sha, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_accuracy": acc_test, "val_accuracy": acc_val, "per_class_accuracy": per_class,
        "noise_mean_confidence": float(pn.max(1).mean()), "raw_shift_2px_accuracy": raw_shift,
        "all_passed": all(c["passed"] for c in checks), "checks": checks,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    n_pass = sum(c["passed"] for c in checks)
    print(f"\n{n_pass}/{len(checks)} checks passed -> {OUT}")
    return report


if __name__ == "__main__":
    run()
