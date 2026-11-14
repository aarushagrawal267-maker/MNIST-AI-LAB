"""Model Training — configure, train, monitor, track and manage models."""
from __future__ import annotations

import io
import threading
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import model_manager as mm
from src.config import CFG, DEMO_MAX_EPOCHS, DEMO_MAX_TRAIN_IMAGES, MODEL_NAME, PUBLIC_DEMO
from src.experiment_tracker import ExperimentTracker
from src.numpy_runtime import HAS_TENSORFLOW
from src.models import CNNConfig
from src.train import load_history, train_and_register
from src.ui import common as C
from src.ui import theme as T

T.page_header(
    "Modeling",
    "Model Training",
    "Train new models on your own machine (CPU is fine), watch accuracy and loss evolve epoch by epoch, review every "
    "tracked experiment and manage saved models. Pre-trained models are loaded from disk, so nothing retrains when "
    "you navigate.",
)
splits = C.safe_splits()
ss = st.session_state


@st.cache_resource
def _training_lock() -> threading.Lock:
    """One training job at a time per server (shared by all visitors)."""
    return threading.Lock()


if PUBLIC_DEMO:
    st.info(
        f"**Public demo:** training is limited to {DEMO_MAX_TRAIN_IMAGES:,} images and {DEMO_MAX_EPOCHS} epochs, one job "
        f"at a time. Models you train here are temporary (they disappear when the server restarts) and cannot replace "
        f"{MODEL_NAME}. Run the lab locally for full-size training.", icon=":material/public:")
tab_train, tab_exp, tab_curves, tab_models = st.tabs(["Train a model", "Experiment tracker", "Learning curves", "Saved models"])

# ======================================================================= train
with tab_train:
    pending = ss.get("pending_run")
    if pending and pending.get("status") == "running":
        path = mm.abs_path(pending["path"])
        st.warning(f"Training of **{pending['name']}** was stopped before it finished.")
        if path.exists() and pending["family"] == "keras":
            if st.button("Register best checkpoint", icon=":material/save:"):
                import keras

                from src.evaluate import compute_metrics, predict_proba

                model = keras.models.load_model(path, compile=False)
                pr = predict_proba(model, "keras", "image", splits.x_val)
                m = compute_metrics(splits.y_val, pr)
                mm.register_model(pending["key"], {
                    "display_name": pending["name"], "family": "keras", "input_kind": "image",
                    "model_type": pending["model_type"], "architecture": pending["arch"] + " (stopped early)",
                    "path": path, "params": int(model.count_params()), "hyperparameters": pending["hp"],
                    "augmentation": pending["augment"], "val_accuracy": m["accuracy"], "val_loss": m["log_loss"],
                    "train_time_s": None, "epochs_trained": None, "custom": True, "stage": "user",
                })
                ss["pending_run"] = None
                st.success(f"Checkpoint registered (validation accuracy {m['accuracy'] * 100:.2f}%).")
        if st.button("Dismiss"):
            ss["pending_run"] = None
            st.rerun()

    with st.form("train_form"):
        top = st.columns([1.3, 1.3, 1])
        mtype = top[0].selectbox("Model type", ["cnn", "ann", "logistic_regression", "knn", "svm"],
                                 format_func={"cnn": "Convolutional Neural Network", "ann": "Neural Network (ANN)",
                                              "logistic_regression": "Logistic Regression", "knn": "K-Nearest Neighbors",
                                              "svm": "Support Vector Machine"}.get)
        name = top[1].text_input("Model name", value="My experiment")
        subset_opts = [2000, 5000, 10000, 20000, 30000, 54000]
        if PUBLIC_DEMO:
            subset_opts = [n for n in subset_opts if n <= DEMO_MAX_TRAIN_IMAGES]
        subset = top[2].select_slider("Training images", subset_opts, value=min(10000, subset_opts[-1]),
                                      help="Smaller subsets train faster on a laptop.")
        st.markdown("**Neural network settings** (ignored for classical models)")
        r1 = st.columns(5)
        epochs = r1[0].slider("Epochs", 1, DEMO_MAX_EPOCHS if PUBLIC_DEMO else 30, min(5, DEMO_MAX_EPOCHS) if PUBLIC_DEMO else 5)
        batch = r1[1].selectbox("Batch size", [32, 64, 128, 256], index=2)
        lr = r1[2].select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3, format_func=lambda v: f"{v:g}")
        opt = r1[3].selectbox("Optimizer", ["adam", "adamw", "rmsprop", "sgd"])
        dropout = r1[4].slider("Dropout", 0.0, 0.6, 0.3, 0.05)
        r2 = st.columns(5)
        filters = r2[0].selectbox("Conv filters", ["16/32", "32/64", "64/128", "32/64/128"], index=1)
        convs = r2[1].selectbox("Convs per block", [1, 2])
        bn = r2[2].checkbox("Batch normalisation", value=False)
        dense = r2[3].selectbox("Dense units", [64, 128, 256, 512], index=1)
        augment = r2[4].checkbox("Augmentation", value=False)
        hidden = st.select_slider("ANN hidden layers", ["256", "256/128", "512/256/128"], value="256/128")
        submitted = st.form_submit_button("Start training", type="primary", icon=":material/play_arrow:")

    if not HAS_TENSORFLOW:
        st.info("TensorFlow is not available on this server's Python version: neural networks (CNN, ANN) cannot be "
                "trained here, but the classical models can. All pre-trained models still work for prediction.",
                icon=":material/info:")
    if submitted and not HAS_TENSORFLOW and mtype in ("cnn", "ann"):
        st.error("Training neural networks requires TensorFlow (Python 3.10–3.13). Choose a classical model instead.")
        submitted = False
    lock = _training_lock()
    if submitted and lock.locked():
        st.warning("Another training job is running on this server right now — please try again in a minute.")
        submitted = False
    if submitted:
        key = "custom_" + "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")[:40] + time.strftime("_%H%M%S")
        cfg = CNNConfig(filters=[int(f) for f in filters.split("/")], convs_per_block=convs, batchnorm=bn,
                        dense_units=dense, dropout=dropout, learning_rate=lr, optimizer=opt, batch_size=batch)
        family = "keras" if mtype in ("cnn", "ann") else "sklearn"
        arch = cfg.describe() if mtype == "cnn" else ""
        ss["pending_run"] = {"status": "running", "key": key, "name": name, "family": family, "model_type": mtype,
                             "path": mm.rel(mm.model_path(key, family, custom=True)), "arch": arch,
                             "hp": cfg.to_dict() if mtype == "cnn" else {}, "augment": augment}
        st.button("Stop training", icon=":material/stop:", help="Stops after the current batch; the best checkpoint is kept.")
        prog = st.progress(0.0, text="Preparing data…")
        mcols = st.columns(4)
        ph = [c.empty() for c in mcols]
        chart_l, chart_r = st.columns(2)
        acc_ph, loss_ph = chart_l.empty(), chart_r.empty()
        logs_hist: dict[str, list] = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
        t0 = time.time()
        steps_per_epoch = int(np.ceil(subset / batch))

        def on_epoch(epoch, logs):
            for k in logs_hist:
                logs_hist[k].append(float(logs.get(k, np.nan)))
            prog.progress((epoch + 1) / epochs, text=f"Epoch {epoch + 1}/{epochs} · {time.time() - t0:.0f}s elapsed")
            ph[0].markdown(T.metric_card("Train acc.", f"{logs['accuracy'] * 100:.2f}%", f"epoch {epoch + 1}"), unsafe_allow_html=True)
            ph[1].markdown(T.metric_card("Val. acc.", f"{logs['val_accuracy'] * 100:.2f}%", "", "", T.BRAND_2), unsafe_allow_html=True)
            ph[2].markdown(T.metric_card("Val. loss", f"{logs['val_loss']:.4f}", ""), unsafe_allow_html=True)
            ph[3].markdown(T.metric_card("Learning rate", f"{logs['learning_rate']:.1e}", ""), unsafe_allow_html=True)
            acc, loss, _ = C.history_figs(logs_hist)
            acc_ph.plotly_chart(acc, use_container_width=True, key=f"live_acc_{epoch}")
            loss_ph.plotly_chart(loss, use_container_width=True, key=f"live_loss_{epoch}")

        def on_batch(batch_i, logs):
            done = len(logs_hist["loss"])
            frac = (done + batch_i / steps_per_epoch) / epochs
            prog.progress(min(frac, 1.0), text=f"Epoch {done + 1}/{epochs} · batch {batch_i}/{steps_per_epoch} · "
                                               f"loss {logs.get('loss', 0):.4f} · {time.time() - t0:.0f}s")

        buf = io.StringIO()
        if not lock.acquire(blocking=False):  # acquired right before the guarded block, released in finally
            ss["pending_run"] = None
            st.warning("Another training job just started on this server — please try again in a minute.")
            st.stop()
        try:
            with st.spinner(f"Training {name}…"), redirect_stdout(buf):
                res = train_and_register(
                    key, splits, ExperimentTracker(), "user", display_name=name, model_type=mtype,
                    cnn_config=cfg if mtype == "cnn" else None,
                    ann_kwargs={"hidden_units": tuple(int(h) for h in hidden.split("/")), "dropout": dropout,
                                "learning_rate": lr, "optimizer": opt, "batch_size": batch} if mtype == "ann" else None,
                    epochs=epochs, augment=augment, train_subset=subset, custom=True,
                    on_epoch=on_epoch, on_batch=on_batch, verbose=0, notes="trained from the web app",
                )
        finally:
            lock.release()  # also runs when the visitor stops the job
        ss["pending_run"] = None
        prog.progress(1.0, text="Done")
        from src.evaluate import evaluate_model

        evaluate_model(key, "val", splits=splits, timing=False)
        st.success(f"**{name}** trained in {res['train_time_s']:.1f}s · validation accuracy "
                   f"**{res['val_accuracy'] * 100:.2f}%** · saved to `{mm.rel(mm.model_path(key, 'keras' if mtype in ('cnn', 'ann') else 'sklearn', True))}`")
        st.caption("Find it under *Saved models*; it is also selectable on the drawing and evaluation pages.")

# ======================================================================= experiments
with tab_exp:
    df = C.experiments()
    if df.empty:
        st.info("No experiments logged yet.")
    else:
        stages = sorted(df["stage"].dropna().unique())
        sel = st.multiselect("Stages", stages, default=stages)
        view = df[df["stage"].isin(sel)].copy()
        cols = ["experiment_id", "stage", "model_key", "architecture", "optimizer", "learning_rate", "batch_size",
                "epochs", "epochs_trained", "dropout", "augmentation", "train_samples", "val_accuracy", "val_loss",
                "train_time_s", "params"]
        st.dataframe(
            view[cols].sort_values("val_accuracy", ascending=False), hide_index=True, use_container_width=True,
            column_config={
                "val_accuracy": st.column_config.ProgressColumn("val accuracy", min_value=0.0, max_value=1.0, format="%.4f"),
                "train_time_s": st.column_config.NumberColumn("train time (s)", format="%.1f"),
                "params": st.column_config.NumberColumn("parameters", format="%d"),
            },
        )
        tune = df[df["stage"] == "tuning"].copy()
        if not tune.empty:
            T.section("Hyperparameter search", f"{len(tune)} trials · {int(tune['train_samples'].iloc[0]):,} training images each · full validation set")
            best_id = tune.sort_values(["val_accuracy", "val_loss"], ascending=[False, True]).iloc[0]["experiment_id"]
            tune = tune.sort_values("val_accuracy")
            labels = [f"{r.experiment_id} · {r.optimizer} lr={r.learning_rate:g} bs={int(r.batch_size)}" for r in tune.itertuples()]
            fig = go.Figure(go.Bar(
                y=labels, x=(1 - tune["val_accuracy"]) * 100, orientation="h",
                marker=dict(color=[T.BRAND_2 if e == best_id else T.SERIES[0] for e in tune["experiment_id"]], cornerradius=3),
                customdata=np.stack([tune["architecture"], tune["val_accuracy"] * 100, tune["train_time_s"]], axis=1),
                hovertemplate="%{y}<br>%{customdata[0]}<br>val acc %{customdata[1]:.2f}%<br>%{customdata[2]:.0f}s<extra></extra>",
            ))
            fig.update_layout(xaxis=dict(title="validation error rate (%) — lower is better", type="log"), yaxis_title="")
            C.plot(T.style_fig(fig, 60 + 34 * len(tune)), key="tune_bar")
            st.caption("Log-scaled error axis so both the near-perfect and the failed configurations stay readable. "
                       "Highlighted: the configuration chosen for full training.")
            a, b = st.columns(2)
            with a:
                fig = go.Figure(go.Scatter(x=tune["learning_rate"], y=tune["val_accuracy"] * 100, mode="markers",
                                           marker=dict(size=11, color=T.SERIES[0], line=dict(width=2, color=T.SURFACE)),
                                           text=tune["experiment_id"], hovertemplate="%{text}<br>lr %{x:g}<br>val acc %{y:.2f}%<extra></extra>"))
                fig.update_layout(xaxis=dict(title="learning rate", type="log"), yaxis_title="validation accuracy (%)")
                C.plot(T.style_fig(fig, 300, "Learning rate vs accuracy"), key="lr_sc")
            with b:
                g = tune.groupby("optimizer")["val_accuracy"].max().sort_values() * 100
                fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker=dict(color=T.SERIES[0], cornerradius=3),
                                       hovertemplate="%{y}: best %{x:.2f}%<extra></extra>"))
                fig.update_layout(xaxis_title="best validation accuracy (%)", xaxis_range=[max(0, g.min() - 5), 100])
                C.plot(T.style_fig(fig, 300, "Best result per optimizer"), key="opt_bar")

# ======================================================================= curves
with tab_curves:
    keras_keys = [k for k, v in C.models().items() if v["family"] == "keras" and load_history(k)]
    if not keras_keys:
        st.info("No training histories yet.")
    else:
        k = C.model_select("Model", keys=keras_keys, key="curve_model")
        h = load_history(k)
        e = mm.get_entry(k)
        T.metric_row([
            T.metric_card("Epochs trained", str(len(h["loss"])), "early stopping on val loss"),
            T.metric_card("Best val. accuracy", f"{max(h['val_accuracy']) * 100:.2f}%", "", "", T.BRAND_2),
            T.metric_card("Lowest val. loss", f"{min(h['val_loss']):.4f}", ""),
            T.metric_card("Training time", f"{(e.get('train_time_s') or 0) / 60:.1f} min", f"{e.get('train_samples', 0):,} images"),
        ])
        acc, loss, lrf = C.history_figs(h)
        a, b = st.columns(2)
        with a:
            C.plot(acc, key="c_acc")
        with b:
            C.plot(loss, key="c_loss")
        if lrf:
            C.plot(lrf, key="c_lr")
            st.caption("ReduceLROnPlateau halves the learning rate when validation loss stops improving for 2 epochs; "
                       "EarlyStopping (patience 4) restores the best weights; ModelCheckpoint keeps the best epoch on disk.")

# ======================================================================= saved models
with tab_models:
    ms = C.models()
    prod = mm.production_key()
    rows = []
    for k, v in ms.items():
        rows.append({
            "key": k, "name": v["display_name"], "type": v.get("model_type"), "family": v["family"],
            f"{MODEL_NAME}": "⚡" if k == prod else "", "val accuracy": v.get("val_accuracy"),
            "parameters": v.get("params") or None, "train time (s)": v.get("train_time_s"),
            "file": v["path"], "custom": bool(v.get("custom")),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"val accuracy": st.column_config.NumberColumn(format="%.4f"),
                                "train time (s)": st.column_config.NumberColumn(format="%.1f")})
    a, b, c = st.columns(3)
    with a:
        k = st.selectbox("Model", list(ms), format_func=C.label, key="mgr_model")
    with b:
        st.write("")
        st.write("")
        if st.button("Load & verify", icon=":material/download_done:"):
            t0 = time.time()
            pred = C.get_predictor(k)
            p = pred.predict_array(splits.x_val[:500])
            acc = (p.argmax(1) == splits.y_val[:500]).mean()
            st.success(f"Loaded in {time.time() - t0:.2f}s · sanity accuracy on 500 validation images: {acc * 100:.1f}%")
        if ms[k]["family"] == "keras" and not PUBLIC_DEMO and st.button(f"Use as {MODEL_NAME}", icon=":material/bolt:"):
            mm.set_production(k)
            st.cache_resource.clear()
            st.rerun()
    with c:
        st.write("")
        st.write("")
        if ms[k].get("custom") and not PUBLIC_DEMO and st.button("Delete model", icon=":material/delete:"):
            mm.delete_model(k)
            st.rerun()
    if ms[k]["family"] == "keras":
        with st.expander("Architecture summary"):
            model = C.get_model(k)
            lines = []
            model.summary(print_fn=lambda s, **kw: lines.append(s))
            st.code("\n".join(lines))
    st.caption(f"{MODEL_NAME} is the model used by the drawing, upload and batch pages. The training pipeline selects it "
               "automatically as the CNN with the best validation accuracy; you can override it here.")
