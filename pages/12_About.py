"""About the project."""
from __future__ import annotations

import streamlit as st

from src import model_manager as mm
from src.config import LAB_NAME, MODEL_NAME
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Lab",
    f"About {LAB_NAME}",
    f"{LAB_NAME} is a complete, locally-run handwritten digit recognition platform: data acquisition, exploration, "
    f"preprocessing, augmentation, classical and deep models, tuning, evaluation, explainability and an interactive "
    f"playground — powered by {MODEL_NAME}.",
)

prod = mm.production_key()
if prod:
    e = mm.get_entry(prod)
    tm = C.metrics(prod, "test")
    T.section(f"Meet {MODEL_NAME}")
    a, b = st.columns([1, 1.4])
    with a:
        st.markdown(
            f"<div class='purf-card'><span class='badge trax'>⚡ {MODEL_NAME}</span>"
            f"<h4 style='margin-top:14px'>{T.esc(e['display_name'])}</h4>"
            f"<p><b>Architecture</b><br>{T.esc(e['architecture'])}</p><br>"
            f"<p><b>Parameters</b> {e.get('params', 0):,} · <b>Augmentation</b> {'yes' if e.get('augmentation') else 'no'}<br>"
            f"<b>Validation accuracy</b> {e['val_accuracy'] * 100:.2f}%"
            + (f" · <b>Test accuracy</b> {tm['accuracy'] * 100:.2f}%" if tm else "") +
            f"<br><b>Trained</b> {e.get('epochs_trained')} epochs in {(e.get('train_time_s') or 0) / 60:.1f} min on CPU</p></div>",
            unsafe_allow_html=True)
    with b:
        st.markdown(T.card(
            "How it was chosen",
            f"The pipeline trains five model families, runs a controlled random search over CNN hyperparameters "
            f"(filters, depth, batch-norm, dense size, dropout, optimizer, learning rate, batch size, epochs), retrains the "
            f"best configuration on the full training split with and without augmentation, and promotes the CNN with the "
            f"highest <b>validation</b> accuracy to {MODEL_NAME}. The 10,000-image test set is scored only afterwards.", "🧭"),
            unsafe_allow_html=True)

T.section("System architecture")
st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fillcolor="#141c3d", color="#3b4680", fontcolor="#e8ecff", fontname="Helvetica", fontsize=11];
  edge [color="#7f89b8", arrowsize=0.7];
  kaggle [label="Kaggle\\nhojjatk/mnist-dataset", fillcolor="#1d2a5c"];
  loader [label="data_loader\\nIDX parser + cache"]; val [label="data_validation"]; eda [label="eda"];
  prep [label="preprocessing\\naugmentation"]; models [label="models\\nsklearn + Keras"]; train [label="train\\ncallbacks + tuning"];
  tracker [label="experiment_tracker"]; reg [label="model_manager\\nregistry.json"]; evaluate [label="evaluate\\nmetrics, ROC, PR"];
  xai [label="explainability"]; ip [label="image_processor\\nMNIST-style centring"]; pred [label="prediction"];
  ui [label="Streamlit UI\\n13 pages", fillcolor="#2a1f66"]; canvas [label="drawing_canvas\\ncustom HTML component", fillcolor="#2a1f66"];
  kaggle -> loader -> val; loader -> eda; loader -> prep -> train; models -> train -> reg; train -> tracker;
  reg -> evaluate; reg -> pred; ip -> pred; canvas -> ip; pred -> ui; evaluate -> ui; xai -> ui; eda -> ui; tracker -> ui;
}
""", use_container_width=True)

T.section("Technology")
cols = st.columns(4)
stack = [
    ("Deep learning", "TensorFlow · Keras 3 (CPU)"), ("Machine learning", "scikit-learn"),
    ("Data", "NumPy · Pandas · kagglehub"), ("Images", "OpenCV · Pillow"),
    ("Visualisation", "Plotly · Matplotlib · Seaborn"), ("Web app", "Streamlit + custom canvas component"),
    ("Persistence", "Keras .keras · Joblib"), ("Testing", "Pytest · Playwright (end-to-end)"),
]
for i, (k, v) in enumerate(stack):
    cols[i % 4].markdown(T.card(k, v), unsafe_allow_html=True)
    if i % 4 == 3:
        cols = st.columns(4)

T.section("Limitations")
st.markdown("""
- Trained only on MNIST: centred, size-normalised digits written mostly by US census employees and students.
  Unusual handwriting styles, stylised fonts, or digits written very thick/thin relative to their size can be misread.
- One digit per image. Multi-digit strings would need a segmentation step.
- Confidence scores are softmax outputs; high confidence does not guarantee correctness, especially for inputs that
  are not digits at all (the model must always pick one of 0–9).
- Uploaded photos need a reasonably uniform background; strong shadows or lined paper can confuse digit detection.
- Explainability maps show sensitivity, not causal reasoning.
- Accuracies are for the MNIST test distribution; accuracy on your own handwriting may differ.
""")
T.section("Future improvements")
st.markdown("""
- An explicit "not a digit" / reject option using confidence thresholds or an out-of-distribution detector.
- Multi-digit recognition (segmentation + sequence model).
- Training on EMNIST or user-collected drawings to broaden handwriting styles; on-device fine-tuning from feedback.
- Model ensembling and test-time augmentation; exporting TRAX to TensorFlow Lite / ONNX for mobile and browser use.
""")
T.section("Run it yourself")
st.code("""python -m venv .venv
.venv\\Scripts\\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m src.train --all          # optional: retrain everything (~20–40 min on CPU)
streamlit run app.py""", language="bash")
