"""Exploratory data analysis: statistics, figures and data-driven insights.

Every number reported here is computed from the actual dataset; nothing is
hard-coded.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .config import EDA_STATS_PATH, FIGURES_DIR, NUM_CLASSES


def image_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """Per-image descriptors: ink area, intensity, bounding box and centre of mass."""
    xf = x.astype(np.float32) / 255.0
    ink = xf > 0.5
    rows = ink.any(axis=2)
    cols = ink.any(axis=1)
    n = len(x)
    h = np.zeros(n, np.int32)
    w = np.zeros(n, np.int32)
    has_r, has_c = rows.any(1), cols.any(1)
    first_r = rows.argmax(1)
    last_r = rows.shape[1] - 1 - rows[:, ::-1].argmax(1)
    first_c = cols.argmax(1)
    last_c = cols.shape[1] - 1 - cols[:, ::-1].argmax(1)
    h[has_r] = (last_r - first_r + 1)[has_r]
    w[has_c] = (last_c - first_c + 1)[has_c]
    total = xf.sum(axis=(1, 2)) + 1e-8
    yy, xx = np.indices(x.shape[1:])
    com_y = (xf * yy).sum(axis=(1, 2)) / total
    com_x = (xf * xx).sum(axis=(1, 2)) / total
    # Stroke thickness proxy: ink area / skeleton-free perimeter estimate (area per bbox height)
    area = ink.sum(axis=(1, 2))
    thickness = area / np.maximum(h, 1)
    return {
        "ink_area": area,
        "mean_intensity": xf.mean(axis=(1, 2)),
        "bbox_h": h,
        "bbox_w": w,
        "com_x": com_x,
        "com_y": com_y,
        "thickness": thickness,
    }


def class_mean_images(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([x[y == c].astype(np.float32).mean(0) / 255.0 for c in range(NUM_CLASSES)])


def class_std_images(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([x[y == c].astype(np.float32).std(0) / 255.0 for c in range(NUM_CLASSES)])


def mean_image_similarity(means: np.ndarray) -> np.ndarray:
    flat = means.reshape(NUM_CLASSES, -1)
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    return flat @ flat.T


def compute_stats(splits) -> dict[str, Any]:
    x_all = np.concatenate([splits.x_train, splits.x_val])
    y_all = np.concatenate([splits.y_train, splits.y_val])
    feats = image_features(x_all)
    means = class_mean_images(x_all, y_all)
    stds = class_std_images(x_all, y_all)
    sim = mean_image_similarity(means)

    per_class: dict[str, dict[str, float]] = {}
    for c in range(NUM_CLASSES):
        m = y_all == c
        per_class[str(c)] = {
            "count": int(m.sum()),
            "ink_area_mean": float(feats["ink_area"][m].mean()),
            "ink_area_std": float(feats["ink_area"][m].std()),
            "mean_intensity": float(feats["mean_intensity"][m].mean()),
            "bbox_w_mean": float(feats["bbox_w"][m].mean()),
            "bbox_h_mean": float(feats["bbox_h"][m].mean()),
            "thickness_mean": float(feats["thickness"][m].mean()),
            "com_x_std": float(feats["com_x"][m].std()),
            "com_y_std": float(feats["com_y"][m].std()),
            "pixel_variability": float(stds[c].mean()),
        }

    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            pairs.append({"a": i, "b": j, "cosine_similarity": float(sim[i, j])})
    pairs.sort(key=lambda d: -d["cosine_similarity"])

    pix = x_all.ravel()
    hist, _ = np.histogram(pix, bins=32, range=(0, 256))
    stats = {
        "n_train_original": int(len(x_all)),
        "n_train": int(len(splits.x_train)),
        "n_val": int(len(splits.x_val)),
        "n_test": int(len(splits.x_test)),
        "n_total": int(len(x_all) + len(splits.x_test)),
        "image_shape": list(x_all.shape[1:]),
        "pixel_min": int(pix.min()),
        "pixel_max": int(pix.max()),
        "pixel_mean": float(pix.mean()),
        "pixel_std": float(pix.std()),
        "fraction_zero_pixels": float((pix == 0).mean()),
        "fraction_saturated_pixels": float((pix == 255).mean()),
        "pixel_histogram": hist.tolist(),
        "class_counts": {
            "train": np.bincount(splits.y_train, minlength=NUM_CLASSES).tolist(),
            "val": np.bincount(splits.y_val, minlength=NUM_CLASSES).tolist(),
            "test": np.bincount(splits.y_test, minlength=NUM_CLASSES).tolist(),
        },
        "bbox_max_h": int(feats["bbox_h"].max()),
        "bbox_max_w": int(feats["bbox_w"].max()),
        "com_x_mean": float(feats["com_x"].mean()),
        "com_y_mean": float(feats["com_y"].mean()),
        "com_std": float(np.sqrt(feats["com_x"].var() + feats["com_y"].var())),
        "per_class": per_class,
        "similar_pairs": pairs[:8],
        "similarity_matrix": sim.tolist(),
        "source": splits.source,
    }
    stats["insights"] = generate_insights(stats)
    return stats


def generate_insights(s: dict[str, Any]) -> list[str]:
    """Turn computed statistics into plain-language observations."""
    counts = np.array(s["class_counts"]["train"]) + np.array(s["class_counts"]["val"])
    pc = s["per_class"]
    most, least = int(counts.argmax()), int(counts.argmin())
    ink = {int(k): v["ink_area_mean"] for k, v in pc.items()}
    var = {int(k): v["pixel_variability"] for k, v in pc.items()}
    width = {int(k): v["bbox_w_mean"] for k, v in pc.items()}
    top_pairs = s["similar_pairs"][:3]
    ins = [
        f"The {s['n_train_original']:,}-image training set is close to balanced: digit {most} is the most "
        f"frequent ({counts[most]:,} images) and digit {least} the least ({counts[least]:,}), a ratio of "
        f"{counts[most] / counts[least]:.2f}. Accuracy is therefore a fair headline metric, but per-class "
        f"recall is still reported.",
        f"{s['fraction_zero_pixels'] * 100:.1f}% of all pixels are exactly 0 (background) and only "
        f"{s['fraction_saturated_pixels'] * 100:.1f}% are fully saturated (255) — images are sparse, "
        f"with anti-aliased stroke edges producing the intermediate grey values.",
        f"Digit 1 uses the least ink (≈{ink[1]:.0f} bright pixels on average) and is the narrowest "
        f"(mean width {width[1]:.1f}px), while digit {max(ink, key=ink.get)} uses the most "
        f"(≈{max(ink.values()):.0f} px). Ink amount alone carries some class information.",
        f"Digit {max(var, key=var.get)} shows the highest pixel-level variability across writers and "
        f"digit {min(var, key=var.get)} the lowest, i.e. some digits are written in many more styles "
        f"than others (slant, loops, open/closed tops).",
        "The most similar class-average images (cosine similarity) are "
        + ", ".join(f"{p['a']}↔{p['b']} ({p['cosine_similarity']:.3f})" for p in top_pairs)
        + ". Pairs that overlap heavily in pixel space are the ones a pixel-based model is most likely to confuse.",
        f"Every digit fits inside a {s['bbox_max_h']}×{s['bbox_max_w']}px box (max observed) and the average "
        f"centre of mass is at ({s['com_x_mean']:.1f}, {s['com_y_mean']:.1f}) with spread "
        f"{s['com_std']:.2f}px — MNIST digits are size-normalised to 20×20 and mass-centred, which is "
        f"exactly what the drawing-canvas preprocessing must replicate.",
    ]
    return ins


def save_eda_report(splits, figures: bool = True) -> dict[str, Any]:
    stats = compute_stats(splits)
    EDA_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if figures:
        save_figures(splits, stats)
    print(f"[eda] statistics -> {EDA_STATS_PATH}")
    return stats


def save_figures(splits, stats: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("dark_background")
    rng = np.random.default_rng(0)
    bg = "#0b1020"

    # Sample grid
    fig, axes = plt.subplots(6, 12, figsize=(12, 6.4), facecolor=bg)
    idx = rng.choice(len(splits.x_train), 72, replace=False)
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(splits.x_train[i], cmap="magma")
        ax.set_title(int(splits.y_train[i]), fontsize=8, color="#a5b4fc")
        ax.axis("off")
    fig.suptitle("Random MNIST training samples", color="white")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sample_grid.png", dpi=110, facecolor=bg)
    plt.close(fig)

    # One row per digit
    fig, axes = plt.subplots(10, 10, figsize=(8, 8.4), facecolor=bg)
    for c in range(10):
        ids = rng.choice(np.where(splits.y_train == c)[0], 10, replace=False)
        for j, i in enumerate(ids):
            axes[c, j].imshow(splits.x_train[i], cmap="gray")
            axes[c, j].axis("off")
    fig.suptitle("Ten handwriting styles for every digit", color="white")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "digits_by_class.png", dpi=110, facecolor=bg)
    plt.close(fig)

    # Mean images
    means = class_mean_images(np.concatenate([splits.x_train, splits.x_val]),
                              np.concatenate([splits.y_train, splits.y_val]))
    fig, axes = plt.subplots(1, 10, figsize=(13, 1.8), facecolor=bg)
    for c in range(10):
        axes[c].imshow(means[c], cmap="inferno")
        axes[c].set_title(str(c), color="white")
        axes[c].axis("off")
    fig.suptitle("Average image per digit", color="white")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mean_images.png", dpi=110, facecolor=bg)
    plt.close(fig)

    # Label distribution
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=bg)
    ax.set_facecolor(bg)
    width = 0.27
    for k, (split, col) in enumerate(zip(("train", "val", "test"), ("#818cf8", "#22d3ee", "#f472b6"))):
        ax.bar(np.arange(10) + (k - 1) * width, stats["class_counts"][split], width, label=split, color=col)
    ax.set_xticks(range(10))
    ax.set_xlabel("digit")
    ax.set_ylabel("images")
    ax.set_title("Class distribution per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=110, facecolor=bg)
    plt.close(fig)

    # Pixel histogram (non-zero)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=bg)
    ax.set_facecolor(bg)
    nz = splits.x_train[splits.x_train > 0].ravel()
    ax.hist(nz, bins=64, color="#a78bfa")
    ax.set_title("Distribution of non-zero pixel intensities")
    ax.set_xlabel("pixel value")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pixel_histogram.png", dpi=110, facecolor=bg)
    plt.close(fig)

    # Similarity heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=bg)
    sns.heatmap(np.array(stats["similarity_matrix"]), annot=True, fmt=".2f", cmap="mako", ax=ax,
                cbar=False, annot_kws={"size": 7})
    ax.set_title("Cosine similarity of class-average images")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_similarity.png", dpi=110, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":  # python -m src.eda
    from .data_loader import get_splits

    st = save_eda_report(get_splits())
    for line in st["insights"]:
        print("-", line)
