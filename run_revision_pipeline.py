#!/usr/bin/env python3
"""End-to-end revision pipeline for Manuscript #93768.

Outputs:
- results/best_hyperparameters.json
- results/pca_variance_summary.json
- results/pca_cumulative_variance.png
- results/loss_curves_<model>.png
- results/shap_summary_plot.png
- results/cell_level_metrics.csv|json
- results/patient_level_metrics.csv|json
- results/cell_level_predictions.csv
- results/patient_level_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import anndata as ad
except ImportError:
    ad = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
except ImportError:
    tf = None
    keras = None
    layers = None
    regularizers = None


MODEL_NAMES = ["MLP", "CNN", "BiLSTM", "Transformer"]


@dataclass
class TrainedFold:
    model_name: str
    fold_id: int
    held_out_patient: str
    y_true: np.ndarray
    y_proba: np.ndarray
    y_pred: np.ndarray
    history: Dict[str, List[float]]
    best_params: Dict[str, float]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_plot(path: Path, dpi: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def normalize_obs_columns(adata: ad.AnnData, metadata_df: pd.DataFrame | None) -> None:
    obs = adata.obs
    obs_cols_lower = {c.lower(): c for c in obs.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in obs.columns:
                return n
            if n.lower() in obs_cols_lower:
                return obs_cols_lower[n.lower()]
        return None

    patient_col = pick("patient_id", "patient", "Patient_ID", "PatientID")
    sample_col = pick("sample_id", "sample", "Sample_ID", "GEX_Sample_ID")
    timepoint_col = pick("timepoint", "Timepoint", "time_point", "visit")
    response_col = pick("response", "Response", "rcb_response", "label")

    if sample_col is not None and "sample_id" not in obs.columns:
        obs["sample_id"] = obs[sample_col].astype(str)
    if patient_col is not None and "patient_id" not in obs.columns:
        obs["patient_id"] = obs[patient_col].astype(str)
    if timepoint_col is not None and "timepoint" not in obs.columns:
        obs["timepoint"] = obs[timepoint_col].astype(str)
    if response_col is not None and "response" not in obs.columns:
        obs["response"] = obs[response_col].astype(str)

    if metadata_df is not None and "sample_id" in obs.columns:
        md = metadata_df.copy()
        md.columns = [c.strip() for c in md.columns]
        md_low = {c.lower(): c for c in md.columns}

        md_sample = None
        for c in ["sample_id", "gex_sample_id", "sample", "sampleid"]:
            if c in md_low:
                md_sample = md_low[c]
                break

        if md_sample is not None:
            md[md_sample] = md[md_sample].astype(str)
            if "patient_id" not in obs.columns:
                for c in ["patient_id", "patient", "patientid"]:
                    if c in md_low:
                        mp = md_low[c]
                        map_dict = md.set_index(md_sample)[mp].astype(str).to_dict()
                        obs["patient_id"] = obs["sample_id"].map(map_dict)
                        break
            if "timepoint" not in obs.columns:
                for c in ["timepoint", "time_point", "visit"]:
                    if c in md_low:
                        mt = md_low[c]
                        map_dict = md.set_index(md_sample)[mt].astype(str).to_dict()
                        obs["timepoint"] = obs["sample_id"].map(map_dict)
                        break
            if "response" not in obs.columns:
                for c in ["response", "rcb_response", "label"]:
                    if c in md_low:
                        mr = md_low[c]
                        map_dict = md.set_index(md_sample)[mr].astype(str).to_dict()
                        obs["response"] = obs["sample_id"].map(map_dict)
                        break

    if "patient_id" not in obs.columns:
        raise ValueError("Could not derive patient_id from AnnData obs/metadata.")
    if "response" not in obs.columns:
        raise ValueError("Could not derive response labels from AnnData obs/metadata.")

    obs["patient_id"] = obs["patient_id"].astype(str)
    obs["response"] = obs["response"].astype(str)
    if "sample_id" in obs.columns:
        obs["sample_id"] = obs["sample_id"].astype(str)

    if "timepoint" in obs.columns:
        tp = obs["timepoint"].fillna("unknown").astype(str).str.lower()
        out = []
        for v in tp:
            if any(k in v for k in ["baseline", "pre", "pretreat", "pre-treatment"]):
                out.append("baseline")
            elif any(k in v for k in ["on-treatment", "on treatment", "cycle", "surgery", "post-treatment", "post treatment"]):
                out.append("on_treatment")
            elif "recurrence" in v or "recur" in v or "post" in v:
                out.append("post_treatment_or_recurrence")
            else:
                out.append("unknown")
        obs["timepoint_group"] = out
    else:
        obs["timepoint"] = "unknown"
        obs["timepoint_group"] = "unknown"

    response_map = {
        "responder": "Responder",
        "non-responder": "Non-Responder",
        "non_responder": "Non-Responder",
        "nonresponder": "Non-Responder",
        "0": "Responder",
        "1": "Non-Responder",
        "rcb 0": "Responder",
        "rcb i": "Responder",
        "rcb ii": "Non-Responder",
        "rcb iii": "Non-Responder",
    }
    obs["response"] = (
        obs["response"]
        .str.strip()
        .str.lower()
        .map(response_map)
        .fillna(obs["response"])
    )


def stratify_adata(
    adata: ad.AnnData,
    baseline_only: bool,
    exclude_samples: List[str],
) -> ad.AnnData:
    mask = np.ones(adata.n_obs, dtype=bool)
    if baseline_only and "timepoint_group" in adata.obs.columns:
        mask &= adata.obs["timepoint_group"].values == "baseline"
    if exclude_samples and "sample_id" in adata.obs.columns:
        ex = set(s.strip() for s in exclude_samples if s.strip())
        mask &= ~adata.obs["sample_id"].isin(ex).values
    return adata[mask].copy()


def get_matrix(adata: ad.AnnData) -> np.ndarray | sp.spmatrix:
    x = adata.X
    if sp.issparse(x):
        return x.tocsr()
    return np.asarray(x, dtype=np.float32)


def compute_pca_features(
    adata: ad.AnnData,
    max_pcs: int,
    target_variance: float,
    results_dir: Path,
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    x = get_matrix(adata)
    n_samples, n_features = x.shape
    n_comp = max(2, min(max_pcs, n_samples - 1, n_features - 1))

    if sp.issparse(x):
        scaler = StandardScaler(with_mean=False)
        x_scaled = scaler.fit_transform(x)
        reducer = TruncatedSVD(n_components=n_comp, random_state=random_state)
    else:
        scaler = StandardScaler(with_mean=True)
        x_scaled = scaler.fit_transform(x)
        reducer = PCA(n_components=n_comp, random_state=random_state)

    x_pca = reducer.fit_transform(x_scaled)
    var_ratio = np.asarray(reducer.explained_variance_ratio_, dtype=float)
    cum_var = np.cumsum(var_ratio)

    top50_index = min(49, len(cum_var) - 1)
    top50_variance = float(cum_var[top50_index])
    k_target = int(np.searchsorted(cum_var, target_variance) + 1)

    summary = {
        "n_cells": int(n_samples),
        "n_features": int(n_features),
        "n_components_computed": int(len(var_ratio)),
        "top_50_variance_explained": top50_variance,
        "target_variance": float(target_variance),
        "k_for_target_variance": int(k_target),
    }

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(np.arange(1, len(cum_var) + 1), cum_var, linewidth=2)
    plt.axhline(target_variance, linestyle="--", linewidth=1)
    plt.axvline(50, linestyle="--", linewidth=1)
    plt.xlabel("Principal components")
    plt.ylabel("Cumulative explained variance")
    save_plot(results_dir / "pca_cumulative_variance.png")

    with open(results_dir / "pca_variance_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return x_pca, summary


def prepare_features(
    adata: ad.AnnData,
    x_pca: np.ndarray,
    max_pcs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    n_pcs = min(max_pcs, x_pca.shape[1])
    gene_features = x_pca[:, :n_pcs]

    numeric_cols = []
    for col in adata.obs.columns:
        c = col.lower()
        if any(k in c for k in ["tcr", "cdr3", "clon", "entropy", "hydro", "charge", "length", "v_gene", "j_gene"]):
            if pd.api.types.is_numeric_dtype(adata.obs[col]):
                numeric_cols.append(col)

    if numeric_cols:
        extra = adata.obs[numeric_cols].fillna(0.0).to_numpy(dtype=np.float32)
        x_tab = np.hstack([gene_features, extra])
        feature_names = [f"PC{i+1}" for i in range(n_pcs)] + numeric_cols
    else:
        x_tab = gene_features
        feature_names = [f"PC{i+1}" for i in range(n_pcs)]

    x_seq = x_tab.astype(np.float32).reshape(x_tab.shape[0], x_tab.shape[1], 1)

    y_text = adata.obs["response"].astype(str).values
    mask = np.isin(y_text, ["Responder", "Non-Responder"])
    y_text = y_text[mask]
    x_tab = x_tab[mask]
    x_seq = x_seq[mask]
    groups = adata.obs["patient_id"].astype(str).values[mask]
    samples = adata.obs["sample_id"].astype(str).values[mask] if "sample_id" in adata.obs.columns else np.array(["NA"] * len(y_text))

    y = LabelEncoder().fit_transform(y_text).astype(np.int32)
    return x_tab, x_seq, y, groups, samples, feature_names


def build_model(model_name: str, input_shape, params: Dict[str, float]) -> keras.Model:
    lr = float(params.get("learning_rate", 1e-3))
    wd = float(params.get("weight_decay", 1e-4))
    dr = float(params.get("dropout", 0.3))

    if model_name == "MLP":
        inp = keras.Input(shape=(input_shape[0],), name="tabular_input")
        x = inp
        for _ in range(int(params.get("n_layers", 2))):
            x = layers.Dense(int(params.get("hidden_dim", 128)), kernel_regularizer=regularizers.l2(wd))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Dropout(dr)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)

    elif model_name == "CNN":
        inp = keras.Input(shape=input_shape, name="sequence_input")
        x = layers.Conv1D(int(params.get("filters", 64)), int(params.get("kernel_size", 5)), padding="same", activation="relu")(inp)
        x = layers.Conv1D(int(params.get("filters", 64)), int(params.get("kernel_size", 5)), padding="same", activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(dr)(x)
        x = layers.Dense(int(params.get("hidden_dim", 64)), activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)

    elif model_name == "BiLSTM":
        inp = keras.Input(shape=input_shape, name="sequence_input")
        x = layers.Bidirectional(layers.LSTM(int(params.get("lstm_units", 64)), return_sequences=False))(inp)
        x = layers.Dropout(dr)(x)
        x = layers.Dense(int(params.get("hidden_dim", 64)), activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)

    elif model_name == "Transformer":
        inp = keras.Input(shape=input_shape, name="sequence_input")
        embed_dim = int(params.get("embed_dim", 64))
        num_heads = int(params.get("num_heads", 4))
        ff_dim = int(params.get("ff_dim", 128))

        x = layers.Dense(embed_dim)(inp)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dense(embed_dim)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dr)(x)
        x = layers.Dense(int(params.get("hidden_dim", 64)), activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inp, out)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


def model_grid(model_name: str) -> List[Dict[str, float]]:
    if model_name == "MLP":
        grid = {
            "hidden_dim": [64, 128],
            "n_layers": [1, 2],
            "dropout": [0.2, 0.4],
            "learning_rate": [1e-3, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [128],
        }
    elif model_name == "CNN":
        grid = {
            "filters": [32, 64],
            "kernel_size": [3, 5],
            "hidden_dim": [64],
            "dropout": [0.2, 0.4],
            "learning_rate": [1e-3, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [128],
        }
    elif model_name == "BiLSTM":
        grid = {
            "lstm_units": [32, 64],
            "hidden_dim": [64],
            "dropout": [0.2, 0.4],
            "learning_rate": [1e-3, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [128],
        }
    else:
        grid = {
            "embed_dim": [32, 64],
            "num_heads": [2, 4],
            "ff_dim": [64, 128],
            "hidden_dim": [64],
            "dropout": [0.2, 0.4],
            "learning_rate": [1e-3, 5e-4],
            "weight_decay": [1e-4, 1e-3],
            "batch_size": [128],
        }

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]
    return combos


def predict_with_model(model_name: str, model: keras.Model, x_tab: np.ndarray, x_seq: np.ndarray, idx: np.ndarray) -> np.ndarray:
    x = x_tab[idx] if model_name == "MLP" else x_seq[idx]
    return model.predict(x, verbose=0).reshape(-1)


def tune_hyperparameters(
    model_name: str,
    x_tab_train: np.ndarray,
    x_seq_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    max_trials: int,
    epochs: int,
    random_state: int,
) -> Dict[str, float]:
    combos = model_grid(model_name)
    rng = np.random.default_rng(random_state)
    if len(combos) > max_trials:
        combos = [combos[i] for i in rng.choice(len(combos), size=max_trials, replace=False)]

    unique_groups = np.unique(groups_train)
    n_splits = min(3, len(unique_groups))
    if n_splits < 2:
        return combos[0]

    splitter = GroupKFold(n_splits=n_splits)
    best_score = -np.inf
    best_params = combos[0]

    for params in combos:
        fold_scores = []
        for tr_idx, va_idx in splitter.split(x_tab_train, y_train, groups_train):
            model = build_model(
                model_name,
                input_shape=(x_tab_train.shape[1],) if model_name == "MLP" else (x_seq_train.shape[1], x_seq_train.shape[2]),
                params=params,
            )
            x_tr = x_tab_train[tr_idx] if model_name == "MLP" else x_seq_train[tr_idx]
            x_va = x_tab_train[va_idx] if model_name == "MLP" else x_seq_train[va_idx]
            callbacks = [keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True)]
            hist = model.fit(
                x_tr,
                y_train[tr_idx],
                validation_data=(x_va, y_train[va_idx]),
                epochs=max(8, epochs // 2),
                batch_size=int(params.get("batch_size", 128)),
                verbose=0,
                callbacks=callbacks,
            )
            fold_scores.append(float(np.max(hist.history.get("val_auc", [0.0]))))
            keras.backend.clear_session()

        score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) >= 2:
        out["auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        out["auc"] = float("nan")
    return out


def aggregate_histories(histories: List[Dict[str, List[float]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(h.get("loss", [])) for h in histories)
    train_mat = np.full((len(histories), max_len), np.nan)
    val_mat = np.full((len(histories), max_len), np.nan)

    for i, h in enumerate(histories):
        t = np.asarray(h.get("loss", []), dtype=float)
        v = np.asarray(h.get("val_loss", []), dtype=float)
        train_mat[i, : len(t)] = t
        val_mat[i, : len(v)] = v

    epochs = np.arange(1, max_len + 1)
    return epochs, np.nanmean(train_mat, axis=0), np.nanmean(val_mat, axis=0)


def create_shap_plot(
    model_name: str,
    params: Dict[str, float],
    x_tab: np.ndarray,
    x_seq: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    results_dir: Path,
    random_state: int,
) -> None:
    import shap

    rng = np.random.default_rng(random_state)
    model = build_model(
        model_name,
        input_shape=(x_tab.shape[1],) if model_name == "MLP" else (x_seq.shape[1], x_seq.shape[2]),
        params=params,
    )

    x_model = x_tab if model_name == "MLP" else x_seq
    model.fit(
        x_model,
        y,
        validation_split=0.2,
        epochs=15,
        batch_size=int(params.get("batch_size", 128)),
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    n_bg = min(80, len(x_tab))
    n_eval = min(200, len(x_tab))
    bg_idx = rng.choice(len(x_tab), size=n_bg, replace=False)
    ev_idx = rng.choice(len(x_tab), size=n_eval, replace=False)

    if model_name == "MLP":
        background = x_tab[bg_idx]
        explain_x = x_tab[ev_idx]

        def f(z):
            return model.predict(z, verbose=0).reshape(-1)

        explainer = shap.KernelExplainer(f, background)
        shap_values = explainer.shap_values(explain_x, nsamples=100)
        values = shap_values[0] if isinstance(shap_values, list) else shap_values
        plt.figure(figsize=(10, 6))
        shap.summary_plot(values, explain_x, feature_names=feature_names, max_display=20, show=False)

    else:
        background_seq = x_seq[bg_idx]
        explain_seq = x_seq[ev_idx]
        background_flat = background_seq.reshape(background_seq.shape[0], -1)
        explain_flat = explain_seq.reshape(explain_seq.shape[0], -1)

        def f(z_flat):
            z_seq = z_flat.reshape(-1, x_seq.shape[1], x_seq.shape[2])
            return model.predict(z_seq, verbose=0).reshape(-1)

        explainer = shap.KernelExplainer(f, background_flat)
        shap_values = explainer.shap_values(explain_flat, nsamples=100)
        values = shap_values[0] if isinstance(shap_values, list) else shap_values
        seq_features = [f"Feat_{i+1}" for i in range(explain_flat.shape[1])]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(values, explain_flat, feature_names=seq_features, max_display=20, show=False)

    plt.xlabel("SHAP value")
    save_plot(results_dir / "shap_summary_plot.png")
    keras.backend.clear_session()


def run_pipeline(args: argparse.Namespace) -> None:
    if ad is None:
        raise ImportError("anndata is required. Install dependencies with: pip install -r requirements.txt")
    if tf is None or keras is None:
        raise ImportError("tensorflow is required. Install dependencies with: pip install -r requirements.txt")

    set_seed(args.random_state)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)
    metadata_df = pd.read_csv(args.metadata) if args.metadata and Path(args.metadata).exists() else None

    normalize_obs_columns(adata, metadata_df)
    adata = stratify_adata(adata, baseline_only=args.baseline_only, exclude_samples=args.exclude_samples)

    x_pca, pca_summary = compute_pca_features(
        adata=adata,
        max_pcs=max(args.max_pcs, 50),
        target_variance=args.target_variance,
        results_dir=results_dir,
        random_state=args.random_state,
    )

    x_tab, x_seq, y, groups, samples, feature_names = prepare_features(adata, x_pca, max_pcs=args.max_pcs)
    if len(np.unique(groups)) < 2:
        raise ValueError("Need at least 2 unique patients for LOPO cross-validation.")

    logo = GroupKFold(n_splits=len(np.unique(groups)))
    folds_by_model: Dict[str, List[TrainedFold]] = {m: [] for m in MODEL_NAMES}

    for fold_id, (tr_idx, te_idx) in enumerate(logo.split(x_tab, y, groups), start=1):
        held_out = str(np.unique(groups[te_idx])[0])

        x_tab_tr, x_tab_te = x_tab[tr_idx], x_tab[te_idx]
        x_seq_tr, x_seq_te = x_seq[tr_idx], x_seq[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        groups_tr = groups[tr_idx]

        for model_name in MODEL_NAMES:
            best_params = tune_hyperparameters(
                model_name=model_name,
                x_tab_train=x_tab_tr,
                x_seq_train=x_seq_tr,
                y_train=y_tr,
                groups_train=groups_tr,
                max_trials=args.max_trials,
                epochs=args.epochs,
                random_state=args.random_state + fold_id,
            )

            model = build_model(
                model_name,
                input_shape=(x_tab.shape[1],) if model_name == "MLP" else (x_seq.shape[1], x_seq.shape[2]),
                params=best_params,
            )

            x_tr = x_tab_tr if model_name == "MLP" else x_seq_tr
            x_te = x_tab_te if model_name == "MLP" else x_seq_te

            # patient-aware validation subset inside training data
            tr_groups_unique = np.unique(groups_tr)
            val_group = tr_groups_unique[0]
            val_mask = groups_tr == val_group
            fit_mask = ~val_mask
            if fit_mask.sum() < 2 or val_mask.sum() < 2:
                val_mask = np.zeros_like(val_mask, dtype=bool)
                val_mask[: max(2, len(val_mask) // 5)] = True
                fit_mask = ~val_mask

            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
            ]

            history = model.fit(
                x_tr[fit_mask],
                y_tr[fit_mask],
                validation_data=(x_tr[val_mask], y_tr[val_mask]),
                epochs=args.epochs,
                batch_size=int(best_params.get("batch_size", 128)),
                verbose=0,
                callbacks=callbacks,
            ).history

            y_proba = model.predict(x_te, verbose=0).reshape(-1)
            y_pred = (y_proba >= 0.5).astype(int)

            folds_by_model[model_name].append(
                TrainedFold(
                    model_name=model_name,
                    fold_id=fold_id,
                    held_out_patient=held_out,
                    y_true=y_te.copy(),
                    y_proba=y_proba.copy(),
                    y_pred=y_pred.copy(),
                    history={k: list(v) for k, v in history.items()},
                    best_params=dict(best_params),
                )
            )
            keras.backend.clear_session()

    best_hparams = {}
    cell_rows = []
    patient_rows = []

    for model_name, folds in folds_by_model.items():
        params_json = [json.dumps(f.best_params, sort_keys=True) for f in folds]
        mode_params = json.loads(pd.Series(params_json).mode().iloc[0])
        best_hparams[model_name] = {
            "consensus": mode_params,
            "per_fold": [f.best_params for f in folds],
        }

        histories = [f.history for f in folds if "loss" in f.history and "val_loss" in f.history]
        if histories:
            epochs, train_loss, val_loss = aggregate_histories(histories)
            plt.figure(figsize=(7.5, 5.0))
            plt.plot(epochs, train_loss, linewidth=2, label="train_loss")
            plt.plot(epochs, val_loss, linewidth=2, label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(frameon=False)
            save_plot(results_dir / f"loss_curves_{model_name.lower()}.png")

        for fold in folds:
            # cell-level rows
            patient_for_cell = fold.held_out_patient
            for i in range(len(fold.y_true)):
                cell_rows.append(
                    {
                        "model": model_name,
                        "fold": fold.fold_id,
                        "held_out_patient": patient_for_cell,
                        "y_true": int(fold.y_true[i]),
                        "y_pred": int(fold.y_pred[i]),
                        "y_proba": float(fold.y_proba[i]),
                    }
                )

            # patient-level aggregation for this fold
            patient_rows.append(
                {
                    "model": model_name,
                    "fold": fold.fold_id,
                    "patient_id": patient_for_cell,
                    "y_true": int(np.round(np.mean(fold.y_true))),
                    "y_proba": float(np.mean(fold.y_proba)),
                }
            )

    cell_pred_df = pd.DataFrame(cell_rows)
    patient_pred_df = pd.DataFrame(patient_rows)
    patient_pred_df["y_pred"] = (patient_pred_df["y_proba"] >= 0.5).astype(int)

    cell_metrics = []
    patient_metrics = []
    for model_name in MODEL_NAMES:
        c = cell_pred_df[cell_pred_df["model"] == model_name]
        p = patient_pred_df[patient_pred_df["model"] == model_name]
        if len(c) > 0:
            cm = compute_metrics(c["y_true"].to_numpy(), c["y_proba"].to_numpy())
            cm["model"] = model_name
            cell_metrics.append(cm)
        if len(p) > 0:
            pm = compute_metrics(p["y_true"].to_numpy(), p["y_proba"].to_numpy())
            pm["model"] = model_name
            patient_metrics.append(pm)

    cell_metrics_df = pd.DataFrame(cell_metrics)
    patient_metrics_df = pd.DataFrame(patient_metrics)

    cell_pred_df.to_csv(results_dir / "cell_level_predictions.csv", index=False)
    patient_pred_df.to_csv(results_dir / "patient_level_predictions.csv", index=False)
    cell_metrics_df.to_csv(results_dir / "cell_level_metrics.csv", index=False)
    patient_metrics_df.to_csv(results_dir / "patient_level_metrics.csv", index=False)

    with open(results_dir / "cell_level_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cell_metrics_df.to_dict(orient="records"), f, indent=2)
    with open(results_dir / "patient_level_metrics.json", "w", encoding="utf-8") as f:
        json.dump(patient_metrics_df.to_dict(orient="records"), f, indent=2)
    with open(results_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_hparams, f, indent=2)

    # SHAP using best patient-level model
    if patient_metrics_df.empty:
        best_model_name = "MLP"
    else:
        best_model_name = patient_metrics_df.sort_values("auc", ascending=False).iloc[0]["model"]

    create_shap_plot(
        model_name=best_model_name,
        params=best_hparams[best_model_name]["consensus"],
        x_tab=x_tab,
        x_seq=x_seq,
        y=y,
        feature_names=feature_names,
        results_dir=results_dir,
        random_state=args.random_state,
    )

    pipeline_report = {
        "terminology": {
            "section_name_updated_to": "Datasets",
            "modality_description": "single-cell multimodal (scRNA-seq + paired TCR, when available)",
        },
        "stratification": {
            "baseline_only": bool(args.baseline_only),
            "excluded_samples": args.exclude_samples,
        },
        "pca": pca_summary,
    }
    with open(results_dir / "pipeline_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_report, f, indent=2)

    print("Revision pipeline complete.")
    print(f"Outputs written to: {results_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run revised JMIR analysis pipeline.")
    p.add_argument("--h5ad", required=True, help="Path to processed AnnData .h5ad file")
    p.add_argument("--metadata", default="", help="Optional CSV metadata file for sample/patient/timepoint mappings")
    p.add_argument("--results-dir", default="results", help="Directory for output artifacts")
    p.add_argument("--max-pcs", type=int, default=50, help="Number of top PCs to use in modeling")
    p.add_argument("--target-variance", type=float, default=0.85, help="Target cumulative explained variance for reporting")
    p.add_argument("--baseline-only", action="store_true", help="Use baseline-only samples for evaluation")
    p.add_argument("--exclude-samples", nargs="*", default=["S8"], help="Sample IDs to exclude (default: S8)")
    p.add_argument("--epochs", type=int, default=25, help="Maximum training epochs")
    p.add_argument("--max-trials", type=int, default=8, help="Max hyperparameter trials per model/fold")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
