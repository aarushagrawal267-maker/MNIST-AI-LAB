"""Model zoo: classical ML baselines, a dense ANN and configurable CNNs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import CFG, IMG_SIZE, NUM_CLASSES, SEED

# Static catalogue used by the UI and training CLI.
MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "family": "sklearn",
        "input_kind": "flat",
        "architecture": "Multinomial logistic regression on 784 normalised pixels (L-BFGS)",
        "purpose": "Linear baseline — how far can a single linear decision boundary per class go?",
    },
    "knn": {
        "display_name": "K-Nearest Neighbors",
        "family": "sklearn",
        "input_kind": "flat",
        "architecture": "PCA(50) -> k-NN (k=3, distance-weighted)",
        "purpose": "Similarity-based classification: a digit is what its closest training digits are.",
    },
    "svm": {
        "display_name": "Support Vector Machine",
        "family": "sklearn",
        "input_kind": "flat",
        "architecture": "PCA(50) -> RBF-kernel SVC (C=5), trained on a 20k stratified subset",
        "purpose": "Strong classical non-linear classifier.",
    },
    "ann": {
        "display_name": "Neural Network (ANN)",
        "family": "keras",
        "input_kind": "image",
        "architecture": "784 -> Dense 256 ReLU -> Dropout 0.2 -> Dense 128 ReLU -> Dense 10 Softmax",
        "purpose": "Fully-connected deep learning baseline with no notion of spatial locality.",
    },
    "cnn_baseline": {
        "display_name": "CNN Baseline",
        "family": "keras",
        "input_kind": "image",
        "architecture": "Conv32 -> Pool -> Conv64 -> Pool -> Dense128 -> Dropout 0.3 -> Softmax",
        "purpose": "The reference convolutional architecture from the project brief.",
    },
}


@dataclass
class CNNConfig:
    """Hyperparameters describing a CNN in the searchable family."""

    filters: list[int] = field(default_factory=lambda: [32, 64])
    convs_per_block: int = 1
    kernel_size: int = 3
    batchnorm: bool = False
    dense_units: int = 128
    dropout: float = 0.3
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    batch_size: int = 128

    def describe(self) -> str:
        block = "x2" if self.convs_per_block == 2 else ""
        bn = "+BN" if self.batchnorm else ""
        convs = " -> ".join(f"Conv{f}{block}{bn}-Pool" for f in self.filters)
        return f"{convs} -> Dense{self.dense_units} -> Dropout{self.dropout:g} -> Softmax"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# scikit-learn models
# --------------------------------------------------------------------------- #
def build_logistic_regression(max_iter: int | None = None):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        max_iter=max_iter or CFG["classical"]["logreg_max_iter"], C=0.1, tol=1e-3, random_state=SEED
    )


def build_knn(n_neighbors: int | None = None, pca_components: int | None = None):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("pca", PCA(n_components=pca_components or CFG["classical"]["pca_components"], random_state=SEED)),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors or CFG["classical"]["knn_neighbors"], weights="distance", n_jobs=-1
                ),
            ),
        ]
    )


def build_svm(C: float = 5.0, pca_components: int | None = None):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    return Pipeline(
        [
            ("pca", PCA(n_components=pca_components or CFG["classical"]["pca_components"], random_state=SEED)),
            ("svc", SVC(kernel="rbf", C=C, gamma="scale", probability=True, random_state=SEED)),
        ]
    )


SKLEARN_BUILDERS = {"logistic_regression": build_logistic_regression, "knn": build_knn, "svm": build_svm}


# --------------------------------------------------------------------------- #
# Keras models
# --------------------------------------------------------------------------- #
def get_optimizer(name: str = "adam", learning_rate: float = 1e-3):
    import keras

    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate)
    if name == "adamw":
        return keras.optimizers.AdamW(learning_rate, weight_decay=1e-4)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer '{name}'")


def compile_model(model, optimizer: str = "adam", learning_rate: float = 1e-3):
    model.compile(
        optimizer=get_optimizer(optimizer, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_ann(
    hidden_units: tuple[int, ...] = (256, 128),
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    name: str = "ann",
):
    """Fully connected network. Accepts (28, 28, 1) images and flattens internally."""
    import keras
    from keras import layers

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
    x = layers.Flatten(name="flatten")(inputs)
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)
        if i == 0 and dropout > 0:
            x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="probabilities")(x)
    return compile_model(keras.Model(inputs, outputs, name=name), optimizer, learning_rate)


def build_cnn(config: CNNConfig | None = None, name: str = "cnn"):
    """Build a CNN from the searchable family (functional API -> Grad-CAM friendly)."""
    import keras
    from keras import layers

    cfg = config or CNNConfig()
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
    x = inputs
    for b, filters in enumerate(cfg.filters):
        for c in range(cfg.convs_per_block):
            pad = "same" if cfg.convs_per_block > 1 else "valid"
            x = layers.Conv2D(
                filters,
                cfg.kernel_size,
                padding=pad,
                use_bias=not cfg.batchnorm,
                activation=None if cfg.batchnorm else "relu",
                name=f"conv{b + 1}_{c + 1}",
            )(x)
            if cfg.batchnorm:
                # momentum 0.9 (Keras default 0.99) so running statistics converge within
                # the short tuning budget; otherwise BN models look unfairly bad at inference.
                x = layers.BatchNormalization(momentum=0.9, name=f"bn{b + 1}_{c + 1}")(x)
                x = layers.Activation("relu", name=f"relu{b + 1}_{c + 1}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{b + 1}")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(cfg.dense_units, activation="relu", name="dense")(x)
    if cfg.dropout > 0:
        x = layers.Dropout(cfg.dropout, name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="probabilities")(x)
    return compile_model(keras.Model(inputs, outputs, name=name), cfg.optimizer, cfg.learning_rate)


def baseline_cnn_config() -> CNNConfig:
    """The exact architecture suggested in the project brief."""
    t = CFG["training"]
    return CNNConfig(filters=[32, 64], dense_units=128, dropout=0.3, learning_rate=t["learning_rate"],
                     optimizer=t["optimizer"], batch_size=t["batch_size"])


def count_params(model) -> int:
    try:
        return int(model.count_params())
    except Exception:
        return 0


def last_conv_layer_name(model) -> str | None:
    import keras

    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    return None
