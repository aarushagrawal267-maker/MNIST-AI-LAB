"""Data augmentation with Keras preprocessing layers.

Only label-preserving transformations are used: small rotations, small
translations and slight zoom. Flips are deliberately excluded because they
change digit identity (e.g. a mirrored 2 or 5 is no longer a valid digit).
"""
from __future__ import annotations

import numpy as np

from .config import CFG, SEED


def build_augmenter(
    rotation_degrees: float | None = None,
    translation: float | None = None,
    zoom: float | None = None,
    seed: int = SEED,
):
    import keras
    from keras import layers

    aug = CFG["augmentation"]
    rotation_degrees = aug["rotation_degrees"] if rotation_degrees is None else rotation_degrees
    translation = aug["translation"] if translation is None else translation
    zoom = aug["zoom"] if zoom is None else zoom

    seq = []
    if rotation_degrees > 0:
        seq.append(layers.RandomRotation(rotation_degrees / 360.0, fill_mode="constant", fill_value=0.0, seed=seed))
    if translation > 0:
        seq.append(layers.RandomTranslation(translation, translation, fill_mode="constant", fill_value=0.0, seed=seed))
    if zoom > 0:
        seq.append(layers.RandomZoom((-zoom, zoom), fill_mode="constant", fill_value=0.0, seed=seed))
    if not seq:  # all strengths set to 0 -> pass images through unchanged
        seq.append(layers.Identity())
    return keras.Sequential(seq, name="augmentation")


def augment_images(
    images: np.ndarray,
    n_variants: int = 1,
    rotation_degrees: float | None = None,
    translation: float | None = None,
    zoom: float | None = None,
    seed: int = SEED,
) -> np.ndarray:
    """Return ``n_variants`` augmented copies per image as float32 (N*n, 28, 28) in [0, 1]."""
    from .preprocessing import to_cnn_input

    x = to_cnn_input(images)
    x = np.repeat(x, n_variants, axis=0)
    aug = build_augmenter(rotation_degrees, translation, zoom, seed=seed)
    out = aug(x, training=True)
    return np.clip(np.asarray(out)[..., 0], 0.0, 1.0)


def make_dataset(x, y, batch_size: int, augment: bool = False, shuffle: bool = True, seed: int = SEED):
    """Build a tf.data pipeline (optionally applying augmentation on the fly)."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 20000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment:
        aug = build_augmenter(seed=seed)
        ds = ds.map(lambda a, b: (aug(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
