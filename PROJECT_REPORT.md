# PURF — Project Report

*Generated 2026-11-25 06:05 by `python -m src.report` from the saved experiment artefacts. All figures below are measured, not estimated.*

## 1. Executive summary

* **Production model (TRAX):** `cnn_tuned` — Tuned CNN: Conv16x2+BN-Pool -> Conv32x2+BN-Pool -> Dense128 -> Dropout0.5 -> Softmax (218,778 parameters).
* **TRAX validation accuracy:** 99.17% · **test accuracy:** 99.51% (49 errors on 10,000 held-out images) · macro F1 (test) 99.50%.
* **Best classical model:** Support Vector Machine — test accuracy 97.82%. TRAX makes 78% fewer test errors.
* **Dataset:** kaggle:hojjatk/mnist-dataset; all 14 integrity checks passed.
* **Canvas-style drawings:** TRAX recognised 100.0% of 120 synthetic stroke drawings rendered at different positions, sizes and brush widths (Section 8).
* **Selection protocol:** highest validation accuracy (test set never used for selection).
* **Compute:** CPU only (x86_64, 4 logical cores); full pipeline 24.2 min (classical models trained in an earlier invocation of the same pipeline).

## 2. Data

* Training split **54,000**, validation **6,000** (stratified 10%, seed 42), test **10,000** (original MNIST test set, untouched).
* Images 28×28 uint8, pixel range 0–255, mean 33.32, std 78.57; 80.9% of pixels are 0.

| digit | train | validation | test |
|---|---|---|---|
| 0 | 5331 | 592 | 980 |
| 1 | 6068 | 674 | 1135 |
| 2 | 5362 | 596 | 1032 |
| 3 | 5518 | 613 | 1010 |
| 4 | 5258 | 584 | 982 |
| 5 | 4879 | 542 | 892 |
| 6 | 5326 | 592 | 958 |
| 7 | 5638 | 627 | 1028 |
| 8 | 5266 | 585 | 974 |
| 9 | 5354 | 595 | 1009 |

**Automatically generated observations:**

* The 60,000-image training set is close to balanced: digit 1 is the most frequent (6,742 images) and digit 5 the least (5,421), a ratio of 1.24. Accuracy is therefore a fair headline metric, but per-class recall is still reported.
* 80.9% of all pixels are exactly 0 (background) and only 0.7% are fully saturated (255) — images are sparse, with anti-aliased stroke edges producing the intermediate grey values.
* Digit 1 uses the least ink (≈60 bright pixels on average) and is the narrowest (mean width 8.2px), while digit 0 uses the most (≈138 px). Ink amount alone carries some class information.
* Digit 2 shows the highest pixel-level variability across writers and digit 1 the lowest, i.e. some digits are written in many more styles than others (slant, loops, open/closed tops).
* The most similar class-average images (cosine similarity) are 4↔9 (0.926), 5↔8 (0.912), 3↔5 (0.898). Pairs that overlap heavily in pixel space are the ones a pixel-based model is most likely to confuse.
* Every digit fits inside a 20×20px box (max observed) and the average centre of mass is at (14.0, 14.0) with spread 0.41px — MNIST digits are size-normalised to 20×20 and mass-centred, which is exactly what the drawing-canvas preprocessing must replicate.

## 3. Baseline models (validation split)

| model | architecture | train images | epochs | train time | val accuracy | val log-loss |
|---|---|---|---|---|---|---|
| Logistic Regression | Multinomial logistic regression on 784 normalised pixels (L-BFGS) | 54,000 | — | 5.7s | 91.88% | 0.2986 |
| K-Nearest Neighbors | PCA(50) -> k-NN (k=3, distance-weighted) | 54,000 | — | 0.3s | 97.72% | 0.1856 |
| Support Vector Machine | PCA(50) -> RBF-kernel SVC (C=5), trained on a 20k stratified subset | 20,000 | — | 18.4s | 98.00% | 0.0725 |
| Neural Network (ANN) | 784 -> Dense256 -> Dropout0.2 -> Dense128 -> Softmax10 | 54,000 | 15 | 30.5s | 98.20% | 0.0681 |
| CNN Baseline | Conv32-Pool -> Conv64-Pool -> Dense128 -> Dropout0.3 -> Softmax | 54,000 | 13 | 108.8s | 98.92% | 0.0371 |

## 4. Hyperparameter search (CNN family)

Seeded random search, 10 trials; each trained on a 15,000-image stratified subset without early stopping and scored on the full 6,000-image validation set. Trial 1 is the brief's baseline configuration.

| ID | architecture | optimizer | lr | batch | epochs | dropout | params | val acc | val loss | time |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-024 | Conv16x2+BN-Pool -> Conv32x2+BN-Pool -> Dense128 -> Dropout0.5 -> Softmax | adam | 0.001 | 128 | 5 | 0.5 | 218,778 | 98.22% | 0.0688 | 34s |
| EXP-018 | Conv64+BN-Pool -> Conv128+BN-Pool -> Dense64 -> Dropout0.2 -> Softmax | adam | 0.0003 | 64 | 3 | 0.2 | 280,586 | 97.95% | 0.0685 | 35s |
| EXP-017 | Conv64-Pool -> Conv128-Pool -> Dense256 -> Dropout0.3 -> Softmax | adam | 0.0003 | 64 | 3 | 0.3 | 896,522 | 97.68% | 0.0823 | 28s |
| EXP-021 | Conv32+BN-Pool -> Conv64+BN-Pool -> Dense128 -> Dropout0.5 -> Softmax | adam | 0.001 | 256 | 5 | 0.5 | 225,322 | 97.65% | 0.0799 | 23s |
| EXP-020 | Conv64x2+BN-Pool -> Conv128x2+BN-Pool -> Dense128 -> Dropout0.2 -> Softmax | rmsprop | 0.0003 | 64 | 3 | 0.2 | 1,064,394 | 97.48% | 0.0985 | 123s |
| EXP-016 | Conv32-Pool -> Conv64-Pool -> Dense128 -> Dropout0.3 -> Softmax | adam | 0.001 | 128 | 4 | 0.3 | 225,034 | 97.47% | 0.0873 | 13s |
| EXP-023 | Conv64-Pool -> Conv128-Pool -> Dense64 -> Dropout0.5 -> Softmax | rmsprop | 0.0003 | 64 | 3 | 0.5 | 280,010 | 95.98% | 0.1315 | 20s |
| EXP-022 | Conv64+BN-Pool -> Conv128+BN-Pool -> Dense64 -> Dropout0.5 -> Softmax | sgd | 0.001 | 256 | 5 | 0.5 | 280,586 | 95.85% | 0.1524 | 51s |
| EXP-019 | Conv64+BN-Pool -> Conv128+BN-Pool -> Dense64 -> Dropout0.3 -> Softmax | rmsprop | 0.003 | 64 | 3 | 0.3 | 280,586 | 94.98% | 0.2006 | 33s |
| EXP-025 | Conv16x2-Pool -> Conv32x2-Pool -> Dense256 -> Dropout0.5 -> Softmax | sgd | 0.0003 | 256 | 3 | 0.5 | 420,602 | 17.05% | 2.2841 | 10s |

**Best trial:** EXP-024 — Conv16x2+BN-Pool -> Conv32x2+BN-Pool -> Dense128 -> Dropout0.5 -> Softmax, adam, lr 0.001, batch 128 → 98.22% after only 5 epochs on the subset.

Per-optimizer range (best / worst validation accuracy): adam: 98.22% / 97.47% (5 trials); rmsprop: 97.48% / 94.98% (3 trials); sgd: 95.85% / 17.05% (2 trials).

Configurations that failed to learn within the short budget: EXP-025 (sgd, lr 0.0003, batch 256). This shows how sensitive CNN training is to the optimizer/learning-rate combination.

> **Methodological note.** In a first run of the search, every batch-normalised trial scored poorly at validation time because Keras' default BatchNorm momentum (0.99) cannot settle its running statistics in the ~200–700 optimisation steps of a tuning trial. The search was stopped, BatchNorm momentum was set to 0.9 and the search was repeated (log of the first run: `outputs/reports/training_log_run1_bn_momentum_issue.txt`). After the fix a batch-normalised architecture won the search.

## 5. Final training and data augmentation

| model | augmentation | epochs (early-stopped) | best epoch | final LR | train time | val accuracy | val log-loss | test accuracy |
|---|---|---|---|---|---|---|---|---|
| Tuned CNN | no | 19 | 15 | 1.3e-04 | 8.0 min | 99.17% | 0.0334 | 99.51% |
| Tuned CNN + Augmentation | yes | 18 | 14 | 1.3e-04 | 7.2 min | 99.03% | 0.0274 | 99.39% |

Augmentation (±10° rotation, ±10% shift, ±10% zoom) changed validation accuracy by **-0.13 percentage points** and validation log-loss from 0.0334 to 0.0274. It therefore **did not improve** validation accuracy in this experiment. Measured training cost: 25.2 s/epoch without vs 23.9 s/epoch with on-the-fly augmentation. The accuracy gap is small: on validation the two models disagree on 52 images (30 only the non-augmented model gets right, 22 only the augmented one), exact McNemar p = 0.33, so the selection between them is effectively a tie. The lower validation log-loss does not carry over to the test set (0.0193 without vs 0.0207 with augmentation), so it is not evidence of consistently better-calibrated probabilities. The augmented model is markedly more robust to un-centred input fed directly to the network (README, *Verifying TRAX*) and remains selectable in the app.

## 6. Model comparison

### 6.1 Validation split (used for model selection)

| model | accuracy | macro precision | macro recall | macro F1 | log-loss | errors |
|---|---|---|---|---|---|---|
| Logistic Regression | 91.88% | 91.81% | 91.79% | 91.79% | 0.2986 | 487 / 6,000 |
| K-Nearest Neighbors | 97.72% | 97.73% | 97.70% | 97.71% | 0.1839 | 137 / 6,000 |
| Support Vector Machine | 98.00% | 97.99% | 97.98% | 97.98% | 0.0725 | 120 / 6,000 |
| Neural Network (ANN) | 98.20% | 98.19% | 98.19% | 98.19% | 0.0680 | 108 / 6,000 |
| CNN Baseline | 98.92% | 98.91% | 98.92% | 98.91% | 0.0371 | 65 / 6,000 |
| Tuned CNN **(TRAX)** | 99.17% | 99.16% | 99.17% | 99.16% | 0.0334 | 50 / 6,000 |
| Tuned CNN + Augmentation | 99.03% | 99.03% | 99.03% | 99.03% | 0.0274 | 58 / 6,000 |

### 6.2 Test split (final, held-out — reported once, never used for selection)

| model | accuracy | macro precision | macro recall | macro F1 | ROC AUC (macro OvR) | errors | train time | ms / image (batch) | latency (1 image) | params |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 92.18% | 92.08% | 92.07% | 92.06% | 0.99369 | 782 / 10,000 | 5.7s | 0.0025 | 0.10 ms | n/a (sklearn) |
| K-Nearest Neighbors | 97.64% | 97.64% | 97.62% | 97.62% | 0.99436 | 236 / 10,000 | 0.3s | 0.1507 | 2.44 ms | n/a (sklearn) |
| Support Vector Machine | 97.82% | 97.82% | 97.80% | 97.80% | 0.99969 | 218 / 10,000 | 18.4s | 0.3111 | 0.75 ms | n/a (sklearn) |
| Neural Network (ANN) | 98.13% | 98.13% | 98.12% | 98.12% | 0.99978 | 187 / 10,000 | 30.5s | 0.0407 | 2.94 ms | 235,146 |
| CNN Baseline | 99.15% | 99.14% | 99.14% | 99.14% | 0.99995 | 85 / 10,000 | 108.8s | 0.0761 | 7.43 ms | 225,034 |
| Tuned CNN **(TRAX)** | 99.51% | 99.51% | 99.50% | 99.50% | 0.99998 | 49 / 10,000 | 478.0s | 0.1106 | 13.71 ms | 218,778 |
| Tuned CNN + Augmentation | 99.39% | 99.39% | 99.39% | 99.39% | 0.99998 | 61 / 10,000 | 430.7s | 0.1050 | 13.72 ms | 218,778 |

**Observations (from the tables above):**

* The linear model is clearly the weakest (92.18% test accuracy): one linear boundary per class cannot capture the variety of handwriting styles.
* Moving from a fully-connected ANN (98.13%) to the brief's baseline CNN (99.15%) cuts test errors from 187 to 85 with a similar parameter budget — the benefit of local receptive fields, weight sharing and pooling.
* TRAX reduces test errors further to 49. Its single-image latency (13.7 ms) is well within interactive limits on a CPU.
* Batched inference is slowest for Support Vector Machine (0.3111 ms/image) and fastest for Logistic Regression (0.0025 ms/image); k-NN trains almost instantly (0.3s, just PCA + indexing) but must keep all training vectors in memory.

### 6.3 Statistical uncertainty

Validation was used for early stopping, checkpoint choice, learning-rate scheduling, the hyperparameter search and the final selection, so validation figures are optimistic by construction; the test figures are the unbiased estimate. All results come from a single seed and a single train/validation split, so run-to-run variance was not measured. 95% Wilson intervals for test accuracy (10,000 images):

| model | test accuracy | 95% CI |
|---|---|---|
| Logistic Regression | 92.18% | 91.64% – 92.69% |
| K-Nearest Neighbors | 97.64% | 97.32% – 97.92% |
| Support Vector Machine | 97.82% | 97.51% – 98.09% |
| Neural Network (ANN) | 98.13% | 97.85% – 98.38% |
| CNN Baseline | 99.15% | 98.95% – 99.31% |
| Tuned CNN **(TRAX)** | 99.51% | 99.35% – 99.63% |
| Tuned CNN + Augmentation | 99.39% | 99.22% – 99.52% |

Paired comparisons on the same images (exact McNemar test; p ≥ 0.05 means the difference is within chance disagreement):

| comparison (A vs B) | split | only A right | only B right | exact McNemar p |
|---|---|---|---|---|
| Tuned CNN vs Tuned CNN + Augmentation | val | 30 | 22 | 0.33 |
| Tuned CNN vs Tuned CNN + Augmentation | test | 35 | 23 | 0.15 |
| Tuned CNN vs CNN Baseline | val | 32 | 17 | 0.044 |
| Tuned CNN vs CNN Baseline | test | 54 | 18 | 2.6e-05 |
| CNN Baseline vs Neural Network (ANN) | val | 63 | 20 | 2.4e-06 |
| CNN Baseline vs Neural Network (ANN) | test | 127 | 25 | 1.2e-17 |

## 7. TRAX in detail (test split)

| digit | precision | recall | F1 | support | errors |
|---|---|---|---|---|---|
| 0 | 0.9939 | 0.9990 | 0.9964 | 980 | 1 |
| 1 | 0.9956 | 0.9991 | 0.9974 | 1135 | 1 |
| 2 | 0.9961 | 0.9961 | 0.9961 | 1032 | 4 |
| 3 | 0.9931 | 0.9970 | 0.9951 | 1010 | 3 |
| 4 | 0.9969 | 0.9969 | 0.9969 | 982 | 3 |
| 5 | 0.9933 | 0.9933 | 0.9933 | 892 | 6 |
| 6 | 0.9989 | 0.9843 | 0.9916 | 958 | 15 |
| 7 | 0.9961 | 0.9942 | 0.9951 | 1028 | 6 |
| 8 | 0.9938 | 0.9938 | 0.9938 | 974 | 6 |
| 9 | 0.9931 | 0.9960 | 0.9946 | 1009 | 4 |

Most frequent confusions (actual → predicted, count): 6→0 (4), 6→5 (4), 4→9 (3), 5→3 (3), 6→1 (3), 2→7 (2), 3→5 (2), 5→9 (2).

Hardest digit: **6** (98.43% recall); easiest: **1** (99.91%). Top-2 accuracy 99.92%; log-loss 0.0193.

Confusion matrix (rows = actual, columns = predicted):

```
         0     1     2     3     4     5     6     7     8     9
  0    979     0     0     0     0     0     0     1     0     0
  1      0  1134     0     1     0     0     0     0     0     0
  2      1     0  1028     1     0     0     0     2     0     0
  3      0     0     0  1007     0     2     0     0     1     0
  4      0     0     0     0   979     0     0     0     0     3
  5      0     0     0     3     0   886     1     0     0     2
  6      4     3     0     1     1     4   943     0     2     0
  7      0     2     2     0     0     0     0  1022     1     1
  8      1     0     2     1     0     0     0     1   968     1
  9      0     0     0     0     2     0     0     0     2  1005
```

Confidence analysis: mean confidence 99.78% on correct vs 83.12% on incorrect predictions; 23 of 49 errors were made with >90% confidence; predictions with confidence ≥ 99% (9,798 images) are 99.88% accurate.

## 8. Drawing-canvas validation

The canvas pipeline (grayscale → polarity → contrast → Otsu + connected components → crop → aspect-preserving resize to 20 px → 28×28 frame → centre-of-mass shift to (14, 14) → normalise) was validated in three ways:

1. **Unit tests** (`tests/test_image_processor.py`): centring of off-centre drawings (within 1 px of (14, 14)), 20-px size normalisation, aspect-ratio preservation, polarity detection, transparent PNGs, noise removal, empty-canvas and invalid-file errors.
2. **Synthetic drawings** (`python -m src.report`): each digit drawn as polylines on a 336×336 black canvas in 12 variants (3 positions × 2 sizes × 2 brush widths) — overall **100.0%** of 120 recognised. Per digit: 0: 100%, 1: 100%, 2: 100%, 3: 100%, 4: 100%, 5: 100%, 6: 100%, 7: 100%, 8: 100%, 9: 100%.
3. **Browser end-to-end test** (Playwright, development only): real mouse strokes drawn on the canvas component in Chromium, Predict clicked, and the rendered prediction read back from the page (results in Section 10).

## 9. Explainability

Saliency, SmoothGrad, Integrated Gradients (64 steps, completeness error reported in the app), Grad-CAM on the last convolutional layer and occlusion sensitivity are implemented in `src/explainability.py` and exercised by `tests/test_models.py::test_explainability_methods`. These maps indicate input sensitivity; they are not causal explanations.

## 10. Verification

Recorded 2026-11-13:

* **Automated tests (pytest)** — 66 passed, 0 failed on 2026-11-25 (tests/: data, preprocessing, image processor, models, predictions, batch CSV export, evaluation statistics, explainability, deploy secret exclusion)
* **Browser: all pages** — 13/13 pages rendered in headless Chromium with no Python exceptions and no JavaScript errors
* **Browser: Draw Your Digit** — digits 0, 1, 2, 3, 4 and 7 drawn with real mouse strokes on the canvas; all 6 predicted correctly by TRAX (100.0% displayed confidence)
* **Browser: Upload Your Digit** — an 800x600 JPEG 'photo' of a dark pen 7 on unevenly lit paper was predicted as 7; a corrupt PNG produced the message 'The file could not be decoded as an image...'
* **Browser: Batch Prediction** — ZIP of 20 MNIST test digits + a .txt file + a corrupt PNG + the photo: 21 predicted (all 21 matched the true labels), 2 skipped with reasons, CSV export available
* **Browser: Model Training** — a CNN trained from the web form (2,000 images, 2 epochs) with live metrics and curves, saved to models/custom/, registered and reloaded via 'Load & verify' (test model removed afterwards)

## 11. Reproducibility

```bash
python -m src.train --all      # full pipeline
python -m src.report           # regenerate this report
```

Seeds are fixed (42) for Python, NumPy and TensorFlow; the validation split indices are cached. Minor numeric differences can still occur across CPUs/TensorFlow builds.
