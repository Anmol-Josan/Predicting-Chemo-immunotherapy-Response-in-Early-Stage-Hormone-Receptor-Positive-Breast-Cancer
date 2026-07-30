#!/usr/bin/env python3
"""Leakage-safe revision pipeline for manuscript #93768.

This script is intentionally self-contained so it can run as a Kaggle script or
locally against an existing AnnData object/raw GEO directory.  The supervised
feature path contains gene-expression PCs and TCR encodings only; UMAP is never
used as a predictive feature.

The effective response-prediction unit is the patient.  Cells are retained for
training efficiency, but every outer split is Leave-One-Patient-Out and all
reported uncertainty intervals resample patients, not cells.

Typical Kaggle command:
    python run_revision_pipeline.py --timepoint-mode baseline

Longitudinal sensitivity analysis:
    python run_revision_pipeline.py --timepoint-mode all
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib.util
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RUNTIME_PACKAGES = {
    "joblib": "joblib>=1.3",
    "matplotlib": "matplotlib>=3.7",
    "numpy": "numpy>=1.24",
    "pandas": "pandas>=2.0",
    "scipy": "scipy>=1.10",
    "sklearn": "scikit-learn>=1.3",
    "xgboost": "xgboost>=2.0",
    "tensorflow": "tensorflow>=2.15",
    "shap": "shap>=0.44",
    "optuna": "optuna>=3.6",
}


def ensure_runtime_packages() -> None:
    """Install only missing runtime packages before importing the science stack."""
    missing = [
        requirement
        for module, requirement in RUNTIME_PACKAGES.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return
    print(
        "Installing missing runtime packages: " + ", ".join(missing),
        file=sys.stderr,
        flush=True,
    )
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        *missing,
    ]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Automatic dependency installation failed. Ensure internet access is "
            "enabled or install requirements.txt before running the pipeline."
        ) from exc


ensure_runtime_packages()

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("revision_pipeline")
RANDOM_SEED = 93768
MODEL_NAMES = ("MLP", "CNN", "BiLSTM", "Transformer", "XGBoost")
METRIC_NAMES = ("roc_auc", "accuracy", "precision", "recall", "specificity", "f1", "brier", "log_loss")

SAMPLE_METADATA = pd.DataFrame(
    [
        ("S1", "GSM9061665", "GSM9061687", "PT1", "Baseline", 1, "Responder"),
        ("S2", "GSM9061666", "GSM9061688", "PT1", "Post-Tx", 1, "Responder"),
        ("S3", "GSM9061667", "GSM9061689", "PT1", "Recurrence", 1, "Responder"),
        ("S4", "GSM9061668", "GSM9061690", "PT2", "Baseline", 1, "Responder"),
        ("S5", "GSM9061669", "GSM9061691", "PT2", "Post-Tx", 1, "Responder"),
        ("S6", "GSM9061670", "GSM9061692", "PT3", "Baseline", 0, "Non-Responder"),
        ("S7", "GSM9061671", "GSM9061693", "PT3", "Post-Tx", 0, "Non-Responder"),
        ("S8", "GSM9061672", None, "PT3", "Recurrence", 0, "Non-Responder"),
        ("S9", "GSM9061673", "GSM9061694", "PT4", "Baseline", 0, "Non-Responder"),
        ("S10", "GSM9061674", "GSM9061695", "PT4", "Post-Tx", 0, "Non-Responder"),
        ("S11", "GSM9061675", "GSM9061696", "PT4", "Recurrence", 0, "Non-Responder"),
    ],
    columns=["sample_id", "gex_id", "tcr_id", "patient_id", "timepoint", "y", "response"],
)


@dataclass
class CellData:
    X: Any
    gene_names: np.ndarray
    metadata: pd.DataFrame
    cdr3_tra: np.ndarray
    cdr3_trb: np.ndarray


@dataclass
class FoldFeatureSet:
    X: np.ndarray
    feature_names: List[str]
    gene_transformer: "GenePCATransformer"
    tcr_transformer: "TCRFeatureTransformer"
    pca_variance_ratio: np.ndarray
    pca_loadings_gene_space: np.ndarray


@dataclass
class FitRecord:
    model_name: str
    fold: int
    held_out_patient: str
    params: Dict[str, Any]
    model_path: str
    preprocessor_path: str
    feature_names: List[str]
    test_indices: np.ndarray = field(repr=False)
    test_features: Optional[np.ndarray] = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)


@dataclass
class ModelTaskResult:
    model_name: str
    fold: int
    held_out_patient: str
    params: Dict[str, Any]
    tune_history: Dict[str, List[float]]
    tune_score: float
    model_path: str
    test_indices: np.ndarray = field(repr=False)
    test_probabilities: np.ndarray = field(repr=False)
    resumed: bool = False


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_clean_figure(fig: plt.Figure, path: Path) -> None:
    """Save a title-free JMIR-ready figure at the requested resolution."""
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def clean_sequence(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return "".join(ch for ch in str(value).upper() if ch in "ACDEFGHIKLMNPQRSTVWY")


def physicochemical_features(sequence: str) -> np.ndarray:
    properties = {
        "hydrophobicity": dict(zip("ACDEFGHIKLMNPQRSTVWY", [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, -3.9, 4.5, 3.8, 1.9, -1.6, -3.5, -3.5, -3.5, -4.5, -0.8, -0.7, 4.2, -1.3])),
        "molecular_weight": dict(zip("ACDEFGHIKLMNPQRSTVWY", [89.1, 121.2, 133.1, 147.1, 165.2, 75.1, 155.2, 131.2, 146.2, 149.2, 132.1, 115.1, 146.2, 105.1, 119.1, 204.2, 174.2, 117.1, 181.2, 163.2])),
        "polarity": dict(zip("ACDEFGHIKLMNPQRSTVWY", [8.1, 5.5, 13.0, 12.3, 5.2, 4.0, 10.4, 11.3, 5.2, 4.9, 5.7, 10.5, 10.5, 9.2, 8.6, 5.7, 6.6, 5.7, 5.4, 6.2])),
        "volume": dict(zip("ACDEFGHIKLMNPQRSTVWY", [88.6, 114.1, 111.1, 117.7, 140.0, 78.0, 124.6, 125.1, 126.0, 117.7, 127.6, 105.1, 129.3, 96.1, 100.7, 162.5, 143.8, 108.5, 142.6, 137.3])),
    }
    if not sequence:
        return np.zeros(8, dtype=np.float32)
    values = []
    for name in ("hydrophobicity", "molecular_weight", "polarity", "volume"):
        arr = np.asarray([properties[name][aa] for aa in sequence], dtype=np.float32)
        values.extend((float(arr.mean()), float(arr.std())))
    return np.asarray(values, dtype=np.float32)


def row_normalize_log1p(X: Any, target_sum: float = 1e4) -> Any:
    X = X.tocsr() if sparse.issparse(X) else np.asarray(X, dtype=np.float32)
    if sparse.issparse(X):
        totals = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
        out = X.multiply(scale[:, None]).tocsr().astype(np.float32)
        out.data = np.log1p(out.data).astype(np.float32)
        return out
    totals = X.sum(axis=1, keepdims=True).astype(np.float32)
    out = np.divide(X * target_sum, totals, out=np.zeros_like(X, dtype=np.float32), where=totals > 0)
    return np.log1p(out).astype(np.float32, copy=False)


def chunk_ranges(n: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, chunk_size):
        yield start, min(n, start + chunk_size)


class GenePCATransformer:
    """Training-only HVG selection, scaling, and memory-bounded incremental PCA."""

    def __init__(self, n_top_genes: int = 1500, n_components: int = 50, batch_size: int = 2048, seed: int = RANDOM_SEED):
        self.n_top_genes = n_top_genes
        self.n_components = n_components
        self.batch_size = batch_size
        self.seed = seed
        self.gene_indices_: Optional[np.ndarray] = None
        self.gene_names_: Optional[np.ndarray] = None
        self.scaler_: Optional[StandardScaler] = None
        self.pca_: Optional[IncrementalPCA] = None
        self.variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: Any, gene_names: Sequence[str], train_indices: np.ndarray) -> "GenePCATransformer":
        X_log = row_normalize_log1p(X)
        train = X_log[train_indices]
        if sparse.issparse(train):
            mean = np.asarray(train.mean(axis=0)).ravel()
            mean_sq = np.asarray(train.multiply(train).mean(axis=0)).ravel()
        else:
            mean = np.asarray(train.mean(axis=0)).ravel()
            mean_sq = np.asarray((train * train).mean(axis=0)).ravel()
        variance = np.maximum(mean_sq - mean * mean, 0.0)
        n_genes = min(self.n_top_genes, X.shape[1])
        self.gene_indices_ = np.argsort(variance)[-n_genes:][::-1].astype(int)
        self.gene_names_ = np.asarray(gene_names)[self.gene_indices_]

        self.scaler_ = StandardScaler(with_mean=False, copy=False)
        train_hvg = train[:, self.gene_indices_]
        self.scaler_.fit(train_hvg)
        n_train = len(train_indices)
        n_comp = max(1, min(self.n_components, len(self.gene_indices_), n_train - 1 if n_train > 1 else 1))
        fit_batch_size = max(self.batch_size, n_comp)
        self.pca_ = IncrementalPCA(n_components=n_comp, batch_size=fit_batch_size)
        for start, end in chunk_ranges(n_train, fit_batch_size):
            if end - start < n_comp and n_train >= n_comp:
                start = max(0, end - n_comp)
            batch = self.scaler_.transform(train_hvg[start:end]).toarray() if sparse.issparse(train_hvg) else self.scaler_.transform(train_hvg[start:end])
            self.pca_.partial_fit(batch.astype(np.float32, copy=False))
        self.variance_ratio_ = np.asarray(self.pca_.explained_variance_ratio_, dtype=np.float32)
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.scaler_ is None or self.pca_ is None or self.gene_indices_ is None:
            raise RuntimeError("GenePCATransformer must be fitted before transform")
        X_log = row_normalize_log1p(X)
        out = np.empty((X.shape[0], len(self.variance_ratio_)), dtype=np.float32)
        for start, end in chunk_ranges(X.shape[0], self.batch_size):
            batch = X_log[start:end][:, self.gene_indices_]
            scaled = self.scaler_.transform(batch)
            if sparse.issparse(scaled):
                scaled = scaled.toarray()
            out[start:end] = self.pca_.transform(np.asarray(scaled, dtype=np.float32))
        return out

    def feature_names(self) -> List[str]:
        n_components = 0 if self.variance_ratio_ is None else len(self.variance_ratio_)
        return [f"gene_pc_{i + 1}" for i in range(n_components)]

    def gene_space_loadings(self, n_genes: int) -> np.ndarray:
        """Return PC loadings mapped to the original gene axis."""
        if self.pca_ is None or self.scaler_ is None or self.gene_indices_ is None:
            raise RuntimeError("Transformer is not fitted")
        loadings = np.zeros((self.pca_.n_components_, n_genes), dtype=np.float32)
        scales = np.where(np.asarray(self.scaler_.scale_) == 0, 1.0, np.asarray(self.scaler_.scale_))
        loadings[:, self.gene_indices_] = self.pca_.components_ / scales[None, :]
        return loadings


class TCRFeatureTransformer:
    """Training-only char 3-mer vocabulary plus explicit missingness indicators."""

    def __init__(self, max_kmers: int = 256, k: int = 3):
        self.max_kmers = max_kmers
        self.k = k
        self.tra_vectorizer_: Optional[CountVectorizer] = None
        self.trb_vectorizer_: Optional[CountVectorizer] = None

    def _fit_vectorizer(self, seqs: Sequence[str], train_indices: np.ndarray) -> Optional[CountVectorizer]:
        train_seqs = [clean_sequence(seqs[i]) for i in train_indices]
        if not any(len(seq) >= self.k for seq in train_seqs):
            return None
        vec = CountVectorizer(analyzer="char", ngram_range=(self.k, self.k), lowercase=False, max_features=self.max_kmers, dtype=np.float32)
        vec.fit(train_seqs)
        return vec

    def fit(self, tra: Sequence[str], trb: Sequence[str], train_indices: np.ndarray) -> "TCRFeatureTransformer":
        self.tra_vectorizer_ = self._fit_vectorizer(tra, train_indices)
        self.trb_vectorizer_ = self._fit_vectorizer(trb, train_indices)
        return self

    def _transform_vectorizer(self, vec: Optional[CountVectorizer], seqs: Sequence[str]) -> np.ndarray:
        if vec is None:
            return np.zeros((len(seqs), 0), dtype=np.float32)
        cleaned = [clean_sequence(seq) for seq in seqs]
        return vec.transform(cleaned).toarray().astype(np.float32, copy=False)

    def transform(self, tra: Sequence[str], trb: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
        tra = np.asarray([clean_sequence(v) for v in tra], dtype=object)
        trb = np.asarray([clean_sequence(v) for v in trb], dtype=object)
        tra_k = self._transform_vectorizer(self.tra_vectorizer_, tra)
        trb_k = self._transform_vectorizer(self.trb_vectorizer_, trb)
        tra_phys = np.vstack([physicochemical_features(s) for s in tra]).astype(np.float32)
        trb_phys = np.vstack([physicochemical_features(s) for s in trb]).astype(np.float32)
        indicators = np.column_stack(
            [
                (tra == "").astype(np.float32),
                (trb == "").astype(np.float32),
                ((tra == "") & (trb == "")).astype(np.float32),
            ]
        )
        parts = [tra_k, trb_k, tra_phys, trb_phys, indicators]
        names = ([f"tra_kmer_{x}" for x in (self.tra_vectorizer_.get_feature_names_out() if self.tra_vectorizer_ else [])]
                 + [f"trb_kmer_{x}" for x in (self.trb_vectorizer_.get_feature_names_out() if self.trb_vectorizer_ else [])]
                 + [f"tra_phys_{x}" for x in ("mean_hydrophobicity", "sd_hydrophobicity", "mean_molecular_weight", "sd_molecular_weight", "mean_polarity", "sd_polarity", "mean_volume", "sd_volume")]
                 + [f"trb_phys_{x}" for x in ("mean_hydrophobicity", "sd_hydrophobicity", "mean_molecular_weight", "sd_molecular_weight", "mean_polarity", "sd_polarity", "mean_volume", "sd_volume")]
                 + ["tra_missing", "trb_missing", "tcr_missing_any"])
        return np.hstack(parts).astype(np.float32, copy=False), names


def locate_file(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.rglob(pattern))
    return matches[0] if matches else None


def make_unique_names(values: Sequence[str]) -> np.ndarray:
    """Match Scanpy's make_unique behavior without importing Scanpy."""
    counts: Dict[str, int] = {}
    unique: List[str] = []
    for value in values:
        name = str(value)
        occurrence = counts.get(name, 0)
        unique.append(name if occurrence == 0 else f"{name}-{occurrence}")
        counts[name] = occurrence + 1
    return np.asarray(unique, dtype=object)


def tenx_companion(matrix_path: Path, stem: str) -> Optional[Path]:
    """Locate a barcodes/features file sharing a 10x matrix prefix."""
    prefix = matrix_path.name.split("matrix.mtx", 1)[0]
    candidates = sorted(matrix_path.parent.glob(f"{prefix}{stem}.tsv*"))
    return candidates[0] if candidates else None


def read_10x_sparse(matrix_path: Path) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Read a 10x Matrix Market bundle directly into cells-by-genes CSR."""
    features_path = tenx_companion(matrix_path, "features") or tenx_companion(matrix_path, "genes")
    barcodes_path = tenx_companion(matrix_path, "barcodes")
    if features_path is None or barcodes_path is None:
        raise FileNotFoundError(f"Incomplete 10x bundle beside {matrix_path}")

    opener = gzip.open if matrix_path.suffix == ".gz" else open
    with opener(matrix_path, "rb") as handle:
        matrix = mmread(handle)
    matrix = sparse.csr_matrix(matrix.T, dtype=np.float32)

    features = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None, compression="infer")
    if matrix.shape != (len(barcodes), len(features)):
        raise ValueError(
            f"10x dimensions disagree for {matrix_path}: matrix={matrix.shape}, "
            f"barcodes={len(barcodes)}, features={len(features)}"
        )
    gene_column = 1 if features.shape[1] > 1 else 0
    genes = make_unique_names(features.iloc[:, gene_column].astype(str).tolist())
    return matrix, genes, barcodes.iloc[:, 0].astype(str).to_numpy()


def safe_extract_tar(tar_path: Path, destination: Path) -> None:
    ensure_dir(destination)
    with tarfile.open(tar_path, "r") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        try:
            archive.extractall(destination, filter="data")
        except TypeError:
            archive.extractall(destination)


def download_geo_raw(data_root: Path) -> Path:
    ensure_dir(data_root)
    tar_path = data_root / "GSE300475_RAW.tar"
    extract_path = data_root / "GSE300475_RAW"
    if not tar_path.exists():
        url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE300475&format=file"
        LOGGER.info("Downloading %s", url)
        urllib.request.urlretrieve(url, tar_path)
    if not any(extract_path.rglob("*matrix.mtx*")) if extract_path.exists() else True:
        LOGGER.info("Extracting %s", tar_path)
        safe_extract_tar(tar_path, extract_path)
    return extract_path


def find_input_h5ad(input_path: Optional[Path], project_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
    if input_path and input_path.is_file() and input_path.suffix == ".h5ad":
        candidates.append(input_path)
    roots = [input_path] if input_path and input_path.is_dir() else []
    roots += [project_root, Path("/kaggle/input"), Path("/kaggle/working")]
    for root in roots:
        if root and root.exists():
            candidates += sorted(root.rglob("*.h5ad"))
    return candidates[0] if candidates else None


def normalize_metadata(obs: pd.DataFrame) -> pd.DataFrame:
    obs = obs.copy()
    aliases = {"Patient_ID": "patient_id", "PatientID": "patient_id", "Response": "response", "Timepoint": "timepoint", "sample": "sample_id", "Sample_ID": "sample_id"}
    for source, target in aliases.items():
        if target not in obs.columns and source in obs.columns:
            obs[target] = obs[source]
    if "sample_id" not in obs.columns:
        obs["sample_id"] = "sample_1"
    if "patient_id" not in obs.columns:
        obs["patient_id"] = obs["sample_id"].astype(str)
    if "timepoint" not in obs.columns:
        obs["timepoint"] = "Unknown"
    if "response" not in obs.columns:
        raise ValueError("Input data must contain response or Response metadata")
    response = obs["response"].astype(str).str.lower()
    obs["y"] = response.map({"responder": 1, "non-responder": 0, "nonresponder": 0, "0": 0, "1": 1})
    if obs["y"].isna().any():
        raise ValueError("Response values must be Responder/Non-Responder or binary")
    obs["y"] = obs["y"].astype(int)
    obs["sample_id"] = obs["sample_id"].astype(str)
    obs["patient_id"] = obs["patient_id"].astype(str)
    obs["timepoint"] = obs["timepoint"].astype(str)
    return obs.reset_index(drop=True)


def load_h5ad(path: Path) -> CellData:
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("anndata is required to read .h5ad input") from exc
    adata = ad.read_h5ad(path, backed=None)
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    X = X.tocsr().astype(np.float32) if sparse.issparse(X) else np.asarray(X, dtype=np.float32)
    obs = normalize_metadata(adata.obs)
    tra = obs.get("cdr3_TRA", pd.Series([""] * len(obs))).fillna("").astype(str).to_numpy()
    trb = obs.get("cdr3_TRB", pd.Series([""] * len(obs))).fillna("").astype(str).to_numpy()
    return CellData(X=X, gene_names=np.asarray(adata.var_names.astype(str)), metadata=obs, cdr3_tra=tra, cdr3_trb=trb)


def read_tcr_for_sample(raw_dir: Path, tcr_id: Optional[str], sample_id: str) -> pd.DataFrame:
    if not tcr_id:
        return pd.DataFrame(columns=["barcode", "cdr3_TRA", "cdr3_TRB"])
    path = locate_file(raw_dir, f"*{tcr_id}*{sample_id}*all_contig_annotations*.csv*") or locate_file(raw_dir, f"*{tcr_id}*all_contig_annotations*.csv*")
    if path is None:
        return pd.DataFrame(columns=["barcode", "cdr3_TRA", "cdr3_TRB"])
    header = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = [c for c in ("barcode", "chain", "productive", "high_confidence", "cdr3", "cdr3_aa", "cdr3_amino_acid") if c in header]
    df = pd.read_csv(path, usecols=wanted, low_memory=False)
    if "barcode" not in df.columns or "chain" not in df.columns:
        return pd.DataFrame(columns=["barcode", "cdr3_TRA", "cdr3_TRB"])
    if "productive" in df.columns:
        df = df[df["productive"].astype(str).str.lower().isin(["true", "1", "yes"]) | (df["productive"] == True)]
    if "high_confidence" in df.columns:
        df = df[df["high_confidence"].astype(str).str.lower().isin(["true", "1", "yes"]) | (df["high_confidence"] == True)]
    sequence_col = next((c for c in ("cdr3_amino_acid", "cdr3_aa", "cdr3") if c in df.columns), None)
    if sequence_col is None:
        return pd.DataFrame(columns=["barcode", "cdr3_TRA", "cdr3_TRB"])
    df = df[df["chain"].isin(["TRA", "TRB"])].copy()
    df[sequence_col] = df[sequence_col].fillna("").astype(str)
    out = {}
    for chain in ("TRA", "TRB"):
        part = df[df["chain"] == chain]
        if not part.empty:
            out[chain] = part.groupby("barcode", sort=False)[sequence_col].first()
    barcodes = sorted(set().union(*(series.index for series in out.values())) if out else set())
    result = pd.DataFrame({"barcode": barcodes})
    result["cdr3_TRA"] = result["barcode"].map(out.get("TRA", pd.Series(dtype=str))).fillna("")
    result["cdr3_TRB"] = result["barcode"].map(out.get("TRB", pd.Series(dtype=str))).fillna("")
    return result


def load_raw_directory(raw_dir: Path) -> CellData:
    samples: List[Tuple[Any, np.ndarray, pd.DataFrame]] = []
    for row in SAMPLE_METADATA.itertuples(index=False):
        matrix = locate_file(raw_dir, f"*{row.gex_id}*{row.sample_id}*matrix.mtx*")
        if matrix is None:
            LOGGER.warning("Skipping missing expression matrix for %s", row.sample_id)
            continue
        X, genes, barcodes = read_10x_sparse(matrix)
        obs = pd.DataFrame({"barcode": barcodes})
        tcr = read_tcr_for_sample(raw_dir, row.tcr_id, row.sample_id)
        if not tcr.empty:
            obs = obs.merge(tcr, on="barcode", how="left", sort=False)
        else:
            obs["cdr3_TRA"] = ""
            obs["cdr3_TRB"] = ""
        obs["sample_id"] = row.sample_id
        obs["patient_id"] = row.patient_id
        obs["timepoint"] = row.timepoint
        obs["response"] = row.response
        obs["y"] = row.y
        samples.append((X, genes, obs))
        LOGGER.info("Loaded %s: cells=%d genes=%d", row.sample_id, X.shape[0], X.shape[1])
    if not samples:
        raise FileNotFoundError(f"No 10x expression matrices found under {raw_dir}")

    all_genes = list(dict.fromkeys(gene for _, genes, _ in samples for gene in genes))
    gene_lookup = {gene: index for index, gene in enumerate(all_genes)}
    aligned: List[Any] = []
    observations: List[pd.DataFrame] = []
    for X_sample, genes, obs_sample in samples:
        if len(genes) == len(all_genes) and all(gene == all_genes[i] for i, gene in enumerate(genes)):
            aligned.append(X_sample)
        else:
            coo = X_sample.tocoo()
            remapped_columns = np.fromiter(
                (gene_lookup[genes[column]] for column in coo.col),
                dtype=np.int64,
                count=coo.nnz,
            )
            aligned.append(
                sparse.csr_matrix(
                    (coo.data, (coo.row, remapped_columns)),
                    shape=(X_sample.shape[0], len(all_genes)),
                    dtype=np.float32,
                )
            )
        observations.append(obs_sample)

    X = sparse.vstack(aligned, format="csr", dtype=np.float32)
    obs = normalize_metadata(pd.concat(observations, ignore_index=True))
    tra = obs.get("cdr3_TRA", pd.Series([""] * len(obs))).fillna("").astype(str).to_numpy()
    trb = obs.get("cdr3_TRB", pd.Series([""] * len(obs))).fillna("").astype(str).to_numpy()
    return CellData(X=X, gene_names=np.asarray(all_genes, dtype=object), metadata=obs, cdr3_tra=tra, cdr3_trb=trb)


def make_synthetic_data(seed: int = RANDOM_SEED, cells_per_patient: int = 120, genes: int = 80) -> CellData:
    rng = np.random.default_rng(seed)
    rows = []
    matrices = []
    tra: List[str] = []
    trb: List[str] = []
    for patient, y in (("PT1", 1), ("PT2", 1), ("PT3", 0), ("PT4", 0)):
        signal = np.zeros(genes, dtype=np.float32)
        signal[:6] = (2.0 if y else -1.0)
        counts = rng.poisson(np.exp(rng.normal(1.2, 0.5, size=(cells_per_patient, genes)) + signal)).astype(np.float32)
        matrices.append(sparse.csr_matrix(counts))
        for i in range(cells_per_patient):
            seq = "CASS" + "A" * (i % 4) + "QETQYF"
            tra.append(seq if i % 5 else "")
            trb.append(seq if i % 7 else "")
            rows.append({"sample_id": f"{patient}_S", "patient_id": patient, "timepoint": "Baseline", "response": "Responder" if y else "Non-Responder", "y": y})
    return CellData(sparse.vstack(matrices, format="csr"), np.asarray([f"G{i:04d}" for i in range(genes)]), pd.DataFrame(rows), np.asarray(tra, dtype=object), np.asarray(trb, dtype=object))


def load_data(args: argparse.Namespace, project_root: Path) -> CellData:
    if args.synthetic:
        return make_synthetic_data(args.seed, args.synthetic_cells_per_patient, args.synthetic_genes)
    input_path = Path(args.input_data).expanduser() if args.input_data else None
    h5ad = find_input_h5ad(input_path, project_root)
    if h5ad:
        LOGGER.info("Loading AnnData from %s", h5ad)
        return load_h5ad(h5ad)
    raw_dir = None
    roots = [input_path] if input_path and input_path.is_dir() else []
    roots += [project_root / "Data", project_root / "data", Path("/kaggle/input"), Path("/kaggle/working/Data")]
    for root in roots:
        if root and root.exists():
            matches = list(root.rglob("*matrix.mtx*"))
            if matches:
                raw_dir = root if any(root.glob("*matrix.mtx*")) else matches[0].parent
                break
    if raw_dir is None and args.download:
        raw_dir = download_geo_raw(Path(args.data_dir))
    if raw_dir is None:
        raise FileNotFoundError("No .h5ad or raw 10x directory found. Use --download on Kaggle or --input-data PATH.")
    LOGGER.info("Loading raw 10x data from %s", raw_dir)
    return load_raw_directory(raw_dir)


def filter_timepoints(data: CellData, mode: str) -> CellData:
    if mode == "baseline":
        mask = data.metadata["timepoint"].str.lower().eq("baseline").to_numpy()
    elif mode == "exclude_recurrence":
        mask = ~data.metadata["timepoint"].str.lower().eq("recurrence").to_numpy()
    else:
        mask = np.ones(len(data.metadata), dtype=bool)
    if mask.sum() == 0:
        raise ValueError(f"No cells remain after timepoint filter: {mode}")
    return CellData(data.X[mask], data.gene_names, data.metadata.loc[mask].reset_index(drop=True), data.cdr3_tra[mask], data.cdr3_trb[mask])


def sample_group_indices(groups: Sequence[str], max_per_group: Optional[int], seed: int) -> np.ndarray:
    groups = np.asarray(groups)
    if max_per_group is None or max_per_group <= 0:
        return np.arange(len(groups))
    rng = np.random.default_rng(seed)
    selected = []
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if len(idx) > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        selected.extend(idx.tolist())
    return np.asarray(sorted(selected), dtype=int)


def build_fold_features(data: CellData, train_indices: np.ndarray, args: argparse.Namespace) -> FoldFeatureSet:
    gene = GenePCATransformer(args.n_top_genes, args.n_pcs, args.pca_batch_size, args.seed).fit(data.X, data.gene_names, train_indices)
    gene_X = gene.transform(data.X)
    tcr = TCRFeatureTransformer(args.max_kmers).fit(data.cdr3_tra, data.cdr3_trb, train_indices)
    tcr_X, tcr_names = tcr.transform(data.cdr3_tra, data.cdr3_trb)
    if args.feature_set == "gene_only":
        X = gene_X
        names = gene.feature_names()
    elif args.feature_set == "tcr_only":
        X = tcr_X
        names = tcr_names
    else:
        X = np.hstack([gene_X, tcr_X]).astype(np.float32, copy=False)
        names = gene.feature_names() + tcr_names
    return FoldFeatureSet(X, names, gene, tcr, gene.variance_ratio_, gene.gene_space_loadings(len(data.gene_names)))


def plot_pca_variance(records: List[Tuple[int, np.ndarray]], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for fold, variance in records:
        ax.plot(np.arange(1, len(variance) + 1), np.cumsum(variance), marker="o", ms=2.5, lw=1.1, label=f"Fold {fold}")
    ax.axhline(0.8, color="0.55", lw=0.8, ls="--")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    save_clean_figure(fig, outdir / "pca_explained_variance.png")


def make_inner_splits(indices: np.ndarray, groups: np.ndarray, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups[indices])
    if len(unique) < 2:
        return [(indices, indices)]
    n_splits = min(3, len(unique))
    splitter = GroupKFold(n_splits=n_splits)
    return [(indices[tr], indices[va]) for tr, va in splitter.split(indices, groups=groups[indices])]


def balanced_sample_weights(groups: np.ndarray) -> np.ndarray:
    counts = pd.Series(groups).value_counts().to_dict()
    weights = np.asarray([1.0 / counts[g] for g in groups], dtype=np.float32)
    return weights * (len(weights) / weights.sum())


def keras_available() -> Any:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        for device in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                pass
        return tf, keras, layers
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for MLP/CNN/BiLSTM/Transformer models; install requirements.txt") from exc


def build_keras_model(name: str, n_features: int, params: Mapping[str, Any], seed: int) -> Any:
    _, keras, layers = keras_available()
    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(n_features,), name="features")
    if name == "MLP":
        x = inputs
        for units in params["hidden_units"]:
            x = layers.Dense(units, activation="relu", kernel_regularizer=keras.regularizers.l2(params["l2"]))(x)
            x = layers.Dropout(params["dropout"])(x)
    else:
        sequence_channels = max(1, min(int(params.get("sequence_channels", 1)), n_features))
        sequence_steps = int(np.ceil(n_features / sequence_channels))
        padding = sequence_steps * sequence_channels - n_features
        x = layers.Reshape((n_features, 1))(inputs)
        if padding:
            x = layers.ZeroPadding1D((0, padding))(x)
        x = layers.Reshape((sequence_steps, sequence_channels))(x)
        if name == "CNN":
            x = layers.Conv1D(params["filters"], min(params["kernel_size"], max(1, sequence_steps)), padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling1D()(x)
        elif name == "BiLSTM":
            x = layers.Bidirectional(layers.LSTM(params["lstm_units"], dropout=params["dropout"], recurrent_dropout=0.0))(x)
        elif name == "Transformer":
            d_model = int(params["d_model"])
            heads = int(params["num_heads"])
            x = layers.Dense(d_model)(x)
            attention = layers.MultiHeadAttention(num_heads=heads, key_dim=max(1, d_model // heads), dropout=params["dropout"])(x, x)
            x = layers.LayerNormalization()(x + attention)
            ff = layers.Dense(d_model * 2, activation="relu")(x)
            ff = layers.Dropout(params["dropout"])(ff)
            x = layers.LayerNormalization()(x + layers.Dense(d_model)(ff))
            x = layers.GlobalAveragePooling1D()(x)
        else:
            raise ValueError(name)
    x = layers.Dense(params.get("final_units", 32), activation="relu")(x)
    x = layers.Dropout(params["dropout"])(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]), loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc")])
    return model


def param_candidates(name: str, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    candidates: List[Dict[str, Any]] = []
    for _ in range(8):
        dropout = float(rng.choice([0.1, 0.2, 0.35]))
        lr = float(rng.choice([1e-4, 3e-4, 1e-3]))
        batch = int(rng.choice([128, 256, 512]))
        if name == "MLP":
            candidates.append({"hidden_units": tuple(rng.choice([32, 64, 128], size=2)), "dropout": dropout, "l2": float(rng.choice([1e-6, 1e-4, 1e-3])), "final_units": 32, "learning_rate": lr, "batch_size": batch})
        elif name == "CNN":
            candidates.append({"filters": int(rng.choice([16, 32, 64])), "kernel_size": int(rng.choice([3, 5, 7])), "dropout": dropout, "final_units": 32, "learning_rate": lr, "batch_size": batch})
        elif name == "BiLSTM":
            candidates.append({"lstm_units": int(rng.choice([16, 32, 64])), "dropout": dropout, "final_units": 32, "learning_rate": lr, "batch_size": batch})
        elif name == "Transformer":
            d_model, heads = (64, 4) if rng.random() > 0.5 else (32, 2)
            candidates.append({"d_model": d_model, "num_heads": heads, "dropout": dropout, "final_units": 32, "learning_rate": lr, "batch_size": batch})
        else:
            candidates.append({"n_estimators": int(rng.choice([80, 120, 180])), "max_depth": int(rng.choice([2, 3, 5])), "learning_rate": lr, "subsample": float(rng.choice([0.7, 0.9, 1.0])), "colsample_bytree": float(rng.choice([0.7, 0.9, 1.0])), "reg_lambda": float(rng.choice([1.0, 5.0, 10.0]))})
    return candidates


def optuna_parameters(name: str, trial: Any) -> Dict[str, Any]:
    """Architecture-specific Optuna search spaces used inside each LOPO fold."""
    dropout = trial.suggest_float("dropout", 0.1, 0.35)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    if name == "MLP":
        return {"hidden_units": (trial.suggest_categorical("units_1", [32, 64, 128]), trial.suggest_categorical("units_2", [32, 64, 128])), "dropout": dropout, "l2": trial.suggest_float("l2", 1e-6, 1e-3, log=True), "final_units": 32, "learning_rate": learning_rate, "batch_size": batch_size}
    if name == "CNN":
        return {"filters": trial.suggest_categorical("filters", [16, 32, 64]), "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]), "dropout": dropout, "final_units": 32, "learning_rate": learning_rate, "batch_size": batch_size}
    if name == "BiLSTM":
        return {"lstm_units": trial.suggest_categorical("lstm_units", [16, 32, 64]), "dropout": dropout, "final_units": 32, "learning_rate": learning_rate, "batch_size": batch_size}
    if name == "Transformer":
        d_model = trial.suggest_categorical("d_model", [32, 64])
        heads = 2 if d_model == 32 else trial.suggest_categorical("num_heads", [2, 4])
        return {"d_model": d_model, "num_heads": heads, "dropout": dropout, "final_units": 32, "learning_rate": learning_rate, "batch_size": batch_size}
    return {"n_estimators": trial.suggest_int("n_estimators", 80, 180, step=20), "max_depth": trial.suggest_int("max_depth", 2, 5), "learning_rate": learning_rate, "subsample": trial.suggest_float("subsample", 0.7, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0), "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True)}


def apply_completion_constraints(name: str, params: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Bound architecture cost for CPU-only Kaggle completion runs."""
    params = dict(params)
    if not getattr(args, "completion_mode", False):
        return params
    if name != "XGBoost":
        params["batch_size"] = max(512, int(params.get("batch_size", 512)))
        params["early_stopping_patience"] = 1
    if name == "MLP":
        params["hidden_units"] = tuple(min(64, int(units)) for units in params["hidden_units"])
    elif name == "CNN":
        params["filters"] = min(32, int(params["filters"]))
    elif name == "BiLSTM":
        params["lstm_units"] = min(16, int(params["lstm_units"]))
    elif name == "Transformer":
        params["d_model"] = min(32, int(params["d_model"]))
        params["num_heads"] = min(2, int(params["num_heads"]))
    elif name == "XGBoost":
        params["n_estimators"] = min(80, int(params["n_estimators"]))
        params["max_depth"] = min(3, int(params["max_depth"]))
    return params


def fit_xgboost(X_train: np.ndarray, y_train: np.ndarray, params: Mapping[str, Any], seed: int, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, groups_train: Optional[np.ndarray] = None) -> Tuple[Any, Dict[str, List[float]]]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is required for the XGBoost benchmark") from exc
    device_params: Dict[str, Any] = {"tree_method": "hist", "device": "cpu"}
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=seed, n_jobs=1, **device_params, **dict(params))
    weights = balanced_sample_weights(groups_train) if groups_train is not None else None
    fit_kwargs: Dict[str, Any] = {"sample_weight": weights, "verbose": False}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except (TypeError, ValueError) as first_error:
        fit_kwargs.pop("verbose", None)
        model.fit(X_train, y_train, **fit_kwargs)
    evals = model.evals_result() if X_val is not None and y_val is not None else {}
    return model, {"loss": list(evals.get("validation_0", {}).get("logloss", [])), "val_loss": list(evals.get("validation_0", {}).get("logloss", []))}


def predict_model(model_name: str, model: Any, X: np.ndarray) -> np.ndarray:
    if model_name == "XGBoost":
        return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float32)
    return np.asarray(model.predict(X, verbose=0)).reshape(-1).astype(np.float32)


def fit_neural(X_train: np.ndarray, y_train: np.ndarray, params: Mapping[str, Any], name: str, seed: int, epochs: int, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, groups_train: Optional[np.ndarray] = None) -> Tuple[Any, Dict[str, List[float]]]:
    _, keras, _ = keras_available()
    model = build_keras_model(name, X_train.shape[1], params, seed)
    weights = balanced_sample_weights(groups_train) if groups_train is not None else None
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss" if X_val is not None else "loss", patience=int(params.get("early_stopping_patience", 3)), restore_best_weights=True, min_delta=1e-4)]
    history = model.fit(X_train, y_train, sample_weight=weights, validation_data=(X_val, y_val) if X_val is not None else None, epochs=epochs, batch_size=int(params["batch_size"]), verbose=0, callbacks=callbacks)
    values = {key: [float(x) for x in val] for key, val in history.history.items()}
    return model, {"loss": values.get("loss", []), "val_loss": values.get("val_loss", [])}


def trial_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        return -float(log_loss(y_true, np.clip(probabilities, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except Exception:
        return -float(np.mean((y_true - probabilities) ** 2))


def tune_model(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, outer_train: np.ndarray, args: argparse.Namespace, fold: int) -> Tuple[Dict[str, Any], Dict[str, List[float]], float]:
    tune_indices = sample_group_indices(groups[outer_train], args.tune_cells_per_patient, args.seed + fold)
    outer_tune = outer_train[tune_indices]
    splits = make_inner_splits(outer_tune, groups, args.seed + fold)
    def evaluate(params: Dict[str, Any], trial_id: int) -> Tuple[float, Dict[str, List[float]]]:
        split_scores = []
        histories = []
        for inner_id, (tr_idx, va_idx) in enumerate(splits):
            if len(np.unique(y[tr_idx])) < 2:
                continue
            if name == "XGBoost":
                model, hist = fit_xgboost(X[tr_idx], y[tr_idx], params, args.seed + fold * 100 + trial_id * 10 + inner_id, X[va_idx], y[va_idx], groups[tr_idx])
            else:
                model, hist = fit_neural(X[tr_idx], y[tr_idx], params, name, args.seed + fold * 100 + trial_id * 10 + inner_id, args.tune_epochs, X[va_idx], y[va_idx], groups[tr_idx])
            probabilities = predict_model(name, model, X[va_idx])
            split_scores.append(trial_score(y[va_idx], probabilities))
            histories.append(hist)
            del model
            gc.collect()
        return (float(np.mean(split_scores)) if split_scores else -np.inf), average_histories(histories)

    # Optuna is the primary tuner.  The deterministic randomized fallback keeps
    # the script runnable in minimal environments while preserving grouped folds.
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed + fold))

        def objective(trial: Any) -> float:
            params = apply_completion_constraints(name, optuna_parameters(name, trial), args)
            if name not in ("MLP", "XGBoost"):
                params["sequence_channels"] = args.sequence_channels
            score, history = evaluate(params, trial.number)
            trial.set_user_attr("params", params)
            trial.set_user_attr("history", history)
            return score

        study.optimize(objective, n_trials=args.n_trials, n_jobs=1, show_progress_bar=False)
        best_trial = study.best_trial
        return dict(best_trial.user_attrs["params"]), dict(best_trial.user_attrs["history"]), float(best_trial.value)
    except ImportError:
        LOGGER.warning("Optuna unavailable; using deterministic randomized search fallback for %s", name)

    candidates = param_candidates(name, args.seed + fold)[: args.n_trials]
    candidates = [apply_completion_constraints(name, params, args) for params in candidates]
    for params in candidates:
        if name not in ("MLP", "XGBoost"):
            params["sequence_channels"] = args.sequence_channels
    best_params, best_score, best_history = candidates[0], -np.inf, {"loss": [], "val_loss": []}
    for trial_id, params in enumerate(candidates):
        score, history = evaluate(params, trial_id)
        if score > best_score:
            best_score = score
            best_params = params
            best_history = history
    return best_params, best_history, best_score


def average_histories(histories: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    if not histories:
        return {"loss": [], "val_loss": []}
    out: Dict[str, List[float]] = {}
    for key in ("loss", "val_loss"):
        arrays = [np.asarray(h.get(key, []), dtype=float) for h in histories if h.get(key)]
        if not arrays:
            out[key] = []
            continue
        width = max(len(a) for a in arrays)
        padded = np.full((len(arrays), width), np.nan)
        for i, arr in enumerate(arrays):
            padded[i, : len(arr)] = arr
        out[key] = np.nanmean(padded, axis=0).tolist()
    return out


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) == 2 else np.nan,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
    }
    return metrics


def aggregate_predictions(predictions: pd.DataFrame, unit: str) -> pd.DataFrame:
    group_col = "patient_id" if unit == "patient" else "sample_id"
    return predictions.groupby(group_col, as_index=False).agg(y=("y", "first"), probability=("probability", "mean"), n_cells=("probability", "size"))


def patient_bootstrap(predictions: pd.DataFrame, unit: str, n_bootstrap: int, seed: int) -> Dict[str, Tuple[float, float]]:
    grouped = aggregate_predictions(predictions, unit)
    units = grouped["patient_id" if unit == "patient" else "sample_id"].to_numpy()
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in METRIC_NAMES}
    for _ in range(n_bootstrap):
        sampled = rng.choice(units, size=len(units), replace=True)
        pieces = [grouped.iloc[np.flatnonzero(units == unit_name)] for unit_name in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        current = compute_metrics(boot["y"].to_numpy(), boot["probability"].to_numpy())
        for metric in METRIC_NAMES:
            if np.isfinite(current[metric]):
                values[metric].append(current[metric])
    return {metric: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan) for metric, vals in values.items()}


def export_loss_curves(loss_histories: Dict[str, List[Dict[str, List[float]]]], outdir: Path) -> None:
    for model_name, histories in loss_histories.items():
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for key, color, label in (("loss", "#1f77b4", "Training"), ("val_loss", "#d62728", "Validation")):
            arrays = [np.asarray(h.get(key, []), dtype=float) for h in histories if h.get(key)]
            if not arrays:
                continue
            width = max(len(a) for a in arrays)
            mat = np.full((len(arrays), width), np.nan)
            for i, arr in enumerate(arrays):
                mat[i, : len(arr)] = arr
            mean = np.nanmean(mat, axis=0)
            lo = np.nanmin(mat, axis=0)
            hi = np.nanmax(mat, axis=0)
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=color, lw=1.5, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.set_xlabel("Epoch / boosting iteration")
        ax.set_ylabel("Binary cross-entropy loss")
        ax.legend(frameon=False)
        fig.tight_layout()
        save_clean_figure(fig, outdir / f"loss_curves_{model_name}.png")


def compute_shap_summary(model_name: str, records: List[FitRecord], data: CellData, outdir: Path, args: argparse.Namespace) -> None:
    try:
        import shap
    except ImportError:
        shap = None
        LOGGER.warning("SHAP is not installed; exporting model-native attribution fallback")
    values_by_feature: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None
    gene_rows: List[Dict[str, Any]] = []
    for record in records:
        if record.model is None or record.test_features is None:
            continue
        X = record.test_features[: args.shap_cells]
        if len(X) == 0:
            continue
        background = X[: min(50, len(X))]
        try:
            if shap is None:
                raise RuntimeError("SHAP unavailable")
            if model_name == "XGBoost":
                explainer = shap.TreeExplainer(record.model)
                raw = explainer.shap_values(X)
                shap_values = raw[0] if isinstance(raw, list) else raw
            else:
                explainer = shap.DeepExplainer(record.model, background)
                raw = explainer.shap_values(X, check_additivity=False)
                shap_values = raw[0] if isinstance(raw, list) else raw
            shap_values = np.asarray(shap_values, dtype=np.float32)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 0]
        except Exception as exc:
            LOGGER.warning("SHAP failed for %s fold %s (%s); using permutation-free gradient fallback", model_name, record.fold, exc)
            if model_name == "XGBoost":
                shap_values = np.tile(np.asarray(getattr(record.model, "feature_importances_", np.zeros(X.shape[1]))), (len(X), 1)).astype(np.float32)
            else:
                try:
                    import tensorflow as tf
                    with tf.GradientTape() as tape:
                        tensor = tf.convert_to_tensor(X)
                        tape.watch(tensor)
                        pred = record.model(tensor, training=False)
                    shap_values = np.asarray(tape.gradient(pred, tensor), dtype=np.float32)
                except Exception:
                    continue
        values_by_feature.append(shap_values)
        feature_names = record.feature_names
        preprocessor = joblib.load(record.preprocessor_path)
        loadings = preprocessor["pca_loadings_gene_space"]
        n_pcs = loadings.shape[0]
        if n_pcs and shap_values.shape[1] >= n_pcs:
            pc_attr = shap_values[:, :n_pcs]
            gene_attr = np.abs(pc_attr) @ np.abs(loadings)
            mean_gene = gene_attr.mean(axis=0)
            for gene, score in zip(data.gene_names, mean_gene):
                gene_rows.append({"model": model_name, "fold": record.fold, "gene": gene, "mean_abs_attribution": float(score)})
    if not values_by_feature or feature_names is None:
        return
    values = np.vstack(values_by_feature)
    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(mean_abs)[-25:][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.barh(np.arange(len(order)), mean_abs[order][::-1], color="#4c78a8")
    ax.set_yticks(np.arange(len(order)), [feature_names[i] for i in order][::-1], fontsize=7)
    ax.set_xlabel("Mean absolute attribution")
    fig.tight_layout()
    save_clean_figure(fig, outdir / "shap_summary_plot.png")
    if gene_rows:
        genes = pd.DataFrame(gene_rows).groupby("gene", as_index=False)["mean_abs_attribution"].mean().sort_values("mean_abs_attribution", ascending=False)
        genes.insert(0, "model", model_name)
        genes.to_csv(outdir / "shap_gene_attributions.csv", index=False)


def cohort_exports(data: CellData, outdir: Path) -> None:
    summary = data.metadata.groupby(["patient_id", "response", "timepoint", "sample_id"], as_index=False).agg(n_cells=("y", "size"), productive_tra=("cdr3_tra", lambda s: np.mean([bool(clean_sequence(v)) for v in s])), productive_trb=("cdr3_trb", lambda s: np.mean([bool(clean_sequence(v)) for v in s])))
    summary.to_csv(outdir / "cohort_summary.csv", index=False)


def available_gpu_count() -> int:
    """Detect CUDA devices without importing TensorFlow in the parent process."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = [token.strip() for token in visible.split(",") if token.strip() and token.strip() != "-1"]
        return len(tokens)
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], check=False, capture_output=True, text=True, timeout=10,
        )
        return sum(1 for line in result.stdout.splitlines() if line.lstrip().startswith("GPU "))
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0


def assign_worker_gpu(gpu_count: int) -> None:
    """Pin each loky worker to one GPU before TensorFlow is imported."""
    if gpu_count <= 0:
        return
    try:
        import multiprocessing

        identity = multiprocessing.current_process()._identity
        worker_number = identity[0] - 1 if identity else 0
    except Exception:
        worker_number = 0
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_tokens = [token.strip() for token in current.split(",")] if current else [str(i) for i in range(gpu_count)]
    device_tokens = [token for token in device_tokens if token and token != "-1"]
    if device_tokens:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_tokens[worker_number % len(device_tokens)]


def restore_resume_zip(zip_path: Path, outdir: Path, data_dir: Path) -> None:
    """Restore result artifacts and extracted raw inputs from a Kaggle archive."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Resume archive not found: {zip_path}")
    restored_results = 0
    restored_data = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            normalized = info.filename.replace("\\", "/")
            if info.is_dir():
                continue
            if normalized.startswith("results/"):
                relative = Path(normalized[len("results/") :])
                target = outdir / relative
                category = "results"
            elif normalized.startswith("Data/GSE300475_RAW/"):
                relative = Path(normalized[len("Data/") :])
                target = data_dir / relative
                category = "data"
            else:
                continue
            if not relative.parts or ".." in relative.parts:
                continue
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            ensure_dir(target.parent)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            if category == "results":
                restored_results += 1
            else:
                restored_data += 1
    LOGGER.info(
        "Restored %d result artifacts and %d raw-data files from %s",
        restored_results, restored_data, zip_path,
    )


def load_fold_features(data: CellData, preprocessor_path: Path, args: argparse.Namespace) -> FoldFeatureSet:
    saved = joblib.load(preprocessor_path)
    gene = saved["gene_transformer"]
    tcr = saved["tcr_transformer"]
    gene_X = gene.transform(data.X)
    tcr_X, tcr_names = tcr.transform(data.cdr3_tra, data.cdr3_trb)
    if args.feature_set == "gene_only":
        X = gene_X
        names = gene.feature_names()
    elif args.feature_set == "tcr_only":
        X = tcr_X
        names = tcr_names
    else:
        X = np.hstack([gene_X, tcr_X]).astype(np.float32, copy=False)
        names = list(saved.get("feature_names", gene.feature_names() + tcr_names))
    loadings = np.asarray(saved.get("pca_loadings_gene_space", gene.gene_space_loadings(len(data.gene_names))))
    return FoldFeatureSet(X, list(names), gene, tcr, np.asarray(gene.variance_ratio_), loadings)


def model_artifact_path(outdir: Path, model_name: str, fold: int) -> Path:
    suffix = ".joblib" if model_name == "XGBoost" else ".keras"
    return outdir / "models" / f"{model_name}_fold_{fold}{suffix}"


def load_saved_model(model_name: str, path: Path) -> Any:
    if model_name == "XGBoost":
        return joblib.load(path)
    _, keras, _ = keras_available()
    return keras.models.load_model(path, compile=False)


def task_signature(model_name: str, fold: int, X: np.ndarray, args: argparse.Namespace) -> str:
    settings = {
        "model": model_name,
        "fold": fold,
        "shape": list(X.shape),
        "seed": args.seed,
        "feature_set": args.feature_set,
        "n_trials": args.n_trials,
        "tune_epochs": args.tune_epochs,
        "epochs": args.epochs,
        "sequence_channels": args.sequence_channels,
        "completion_mode": bool(getattr(args, "completion_mode", False)),
        "tune_cells_per_patient": args.tune_cells_per_patient,
        "max_train_cells_per_patient": args.max_train_cells_per_patient,
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()


def load_completed_task(
    model_name: str,
    fold: int,
    held_out: str,
    X: np.ndarray,
    test_idx: np.ndarray,
    outdir: Path,
    args: argparse.Namespace,
    allow_checkpoint: bool,
    allow_legacy_artifact: bool,
) -> Optional[ModelTaskResult]:
    artifact = model_artifact_path(outdir, model_name, fold)
    checkpoint_dir = ensure_dir(outdir / "checkpoints")
    metadata_path = checkpoint_dir / f"{model_name}_fold_{fold}.json"
    predictions_path = checkpoint_dir / f"{model_name}_fold_{fold}.npz"
    signature = task_signature(model_name, fold, X, args)
    if allow_checkpoint and artifact.exists() and metadata_path.exists() and predictions_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("signature") == signature:
                with np.load(predictions_path) as saved:
                    probabilities = np.asarray(saved["test_probabilities"], dtype=np.float32)
                if len(probabilities) == len(test_idx):
                    return ModelTaskResult(
                        model_name, fold, held_out, dict(metadata["params"]),
                        dict(metadata.get("tune_history", {})), float(metadata.get("tune_score", np.nan)),
                        str(artifact), test_idx, probabilities, True,
                    )
        except Exception as exc:
            LOGGER.warning("Ignoring unreadable checkpoint for %s fold %d: %s", model_name, fold, exc)
    if not (allow_legacy_artifact and artifact.exists()):
        return None
    LOGGER.info("Recovering predictions from legacy artifact: %s", artifact)
    try:
        model = load_saved_model(model_name, artifact)
        probabilities = predict_model(model_name, model, X[test_idx])
        if model_name != "XGBoost":
            try:
                _, keras, _ = keras_available()
                keras.backend.clear_session()
            except Exception:
                pass
        del model
        gc.collect()
    except Exception as exc:
        LOGGER.warning("Legacy artifact is incompatible; retraining %s fold %d: %s", model_name, fold, exc)
        return None
    return ModelTaskResult(
        model_name, fold, held_out, {"resumed_from_legacy_artifact": True},
        {"loss": [], "val_loss": []}, np.nan, str(artifact), test_idx, probabilities, True,
    )


def train_model_task(
    model_name: str,
    fold: int,
    held_out: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    outdir_text: str,
    args: argparse.Namespace,
) -> ModelTaskResult:
    """Tune and fit one independent model/fold job inside a worker process."""
    assign_worker_gpu(int(getattr(args, "worker_gpu_count", 0)))
    outdir = Path(outdir_text)
    LOGGER.info("Training %s fold %d (held out %s)", model_name, fold, held_out)
    params, tune_history, tune_score = tune_model(model_name, X, y, groups, train_idx, args, fold)
    full_relative = sample_group_indices(groups[train_idx], args.max_train_cells_per_patient, args.seed + fold + 100)
    full_train_idx = train_idx[full_relative]
    if model_name == "XGBoost":
        model, _ = fit_xgboost(X[full_train_idx], y[full_train_idx], params, args.seed + fold, groups_train=groups[full_train_idx])
    else:
        model, _ = fit_neural(X[full_train_idx], y[full_train_idx], params, model_name, args.seed + fold, args.epochs, groups_train=groups[full_train_idx])
    probabilities = predict_model(model_name, model, X[test_idx])
    artifact = model_artifact_path(outdir, model_name, fold)
    ensure_dir(artifact.parent)
    temporary_artifact = artifact.with_name(artifact.stem + ".tmp" + artifact.suffix)
    if model_name == "XGBoost":
        joblib.dump(model, temporary_artifact)
    else:
        model.save(temporary_artifact, include_optimizer=False)
    os.replace(temporary_artifact, artifact)

    checkpoint_dir = ensure_dir(outdir / "checkpoints")
    predictions_path = checkpoint_dir / f"{model_name}_fold_{fold}.npz"
    temporary_predictions = predictions_path.with_suffix(".tmp.npz")
    with temporary_predictions.open("wb") as handle:
        np.savez_compressed(handle, test_probabilities=probabilities)
    os.replace(temporary_predictions, predictions_path)
    metadata = {
        "signature": task_signature(model_name, fold, X, args),
        "model_name": model_name,
        "fold": fold,
        "held_out_patient": held_out,
        "params": params,
        "tune_history": tune_history,
        "tune_score": tune_score,
    }
    metadata_path = checkpoint_dir / f"{model_name}_fold_{fold}.json"
    temporary_metadata = metadata_path.with_suffix(".tmp.json")
    temporary_metadata.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    if model_name != "XGBoost":
        try:
            _, keras, _ = keras_available()
            keras.backend.clear_session()
        except Exception:
            pass
    del model
    gc.collect()
    return ModelTaskResult(
        model_name, fold, held_out, params, tune_history, tune_score,
        str(artifact), test_idx, probabilities, False,
    )


def run_pipeline(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    project_root = Path(__file__).resolve().parent
    outdir = ensure_dir(Path(args.output_dir).expanduser())
    ensure_dir(outdir / "models")
    ensure_dir(outdir / "preprocessors")
    ensure_dir(outdir / "checkpoints")
    if args.resume_zip:
        restore_resume_zip(Path(args.resume_zip).expanduser(), outdir, Path(args.data_dir).expanduser())
    data = filter_timepoints(load_data(args, project_root), args.timepoint_mode)
    data.metadata["cdr3_tra"] = data.cdr3_tra
    data.metadata["cdr3_trb"] = data.cdr3_trb
    data.metadata["tcr_missing_any"] = [(not clean_sequence(a) and not clean_sequence(b)) for a, b in zip(data.cdr3_tra, data.cdr3_trb)]
    cohort_exports(data, outdir)
    LOGGER.info("Cells=%d genes=%d patients=%d samples=%d mode=%s", data.X.shape[0], data.X.shape[1], data.metadata.patient_id.nunique(), data.metadata.sample_id.nunique(), args.timepoint_mode)

    y = data.metadata["y"].to_numpy(dtype=int)
    groups = data.metadata["patient_id"].to_numpy()
    patients = np.unique(groups)
    if len(patients) < 3:
        raise ValueError("At least three patients are required for nested LOPO/GroupKFold")
    outer_splitter = LeaveOneGroupOut()
    outer_splits = [(train_idx, test_idx, str(groups[test_idx][0])) for train_idx, test_idx in outer_splitter.split(np.zeros(len(y)), y, groups)]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if any(m not in MODEL_NAMES for m in models):
        raise ValueError(f"Unknown model; choose from {MODEL_NAMES}")
    gpu_count = available_gpu_count()
    args.worker_gpu_count = gpu_count
    if args.parallel_jobs > 0:
        parallel_jobs = args.parallel_jobs
    elif gpu_count > 0:
        parallel_jobs = gpu_count
    else:
        parallel_jobs = min(4, max(1, os.cpu_count() or 1))
    if parallel_jobs == 1 and any(m != "XGBoost" for m in models):
        assign_worker_gpu(gpu_count)
        keras_available()
    LOGGER.info("Model/fold parallel jobs=%d detected_gpus=%d", parallel_jobs, gpu_count)
    LOGGER.info(
        "Training budget: completion_mode=%s trials=%d tune_epochs=%d epochs=%d "
        "tune_cells_per_patient=%d train_cells_per_patient=%d sequence_channels=%d",
        bool(getattr(args, "completion_mode", False)),
        args.n_trials,
        args.tune_epochs,
        args.epochs,
        args.tune_cells_per_patient,
        args.max_train_cells_per_patient,
        args.sequence_channels,
    )

    pca_records: List[Tuple[int, np.ndarray]] = []
    variance_rows: List[Dict[str, Any]] = []
    all_fold_metrics: List[Dict[str, Any]] = []
    prediction_frames: Dict[str, List[pd.DataFrame]] = {m: [] for m in models}
    loss_histories: Dict[str, List[Dict[str, List[float]]]] = {m: [] for m in models}
    best_hyperparameters: Dict[str, Any] = {}
    shap_records: Dict[str, List[FitRecord]] = {m: [] for m in models}
    if args.shap_model in ("auto", "all"):
        shap_target = "XGBoost" if "XGBoost" in models else models[0]
    else:
        shap_target = args.shap_model

    fold_contexts: List[Tuple[int, np.ndarray, np.ndarray, str, FoldFeatureSet, Path, bool]] = []
    allow_resume = bool(args.resume or args.resume_zip)
    for fold, (train_idx, test_idx, held_out) in enumerate(outer_splits, start=1):
        LOGGER.info("Outer fold %d/%d: held-out patient=%s", fold, len(outer_splits), held_out)
        preprocessor_path = outdir / "preprocessors" / f"fold_{fold}.joblib"
        reused_preprocessor = False
        if allow_resume and preprocessor_path.exists():
            LOGGER.info("Reusing fitted preprocessor for fold %d", fold)
            try:
                feature_set = load_fold_features(data, preprocessor_path, args)
                reused_preprocessor = True
            except Exception as exc:
                LOGGER.warning("Could not reuse fold %d preprocessor (%s); refitting", fold, exc)
        if not reused_preprocessor:
            tuning_train = sample_group_indices(groups[train_idx], args.max_train_cells_per_patient, args.seed + fold)
            tuning_train = train_idx[tuning_train]
            feature_set = build_fold_features(data, tuning_train, args)
            joblib.dump(
                {
                    "gene_transformer": feature_set.gene_transformer,
                    "tcr_transformer": feature_set.tcr_transformer,
                    "feature_names": feature_set.feature_names,
                    "pca_loadings_gene_space": feature_set.pca_loadings_gene_space,
                },
                preprocessor_path,
            )
        pca_records.append((fold, feature_set.pca_variance_ratio))
        for pc, variance in enumerate(feature_set.pca_variance_ratio, start=1):
            variance_rows.append({"fold": fold, "pc": pc, "explained_variance_ratio": float(variance), "cumulative_explained_variance": float(np.sum(feature_set.pca_variance_ratio[:pc]))})
        fold_contexts.append((fold, train_idx, test_idx, held_out, feature_set, preprocessor_path, reused_preprocessor))

    completed_results: List[ModelTaskResult] = []
    pending_tasks: List[Tuple[str, int, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for fold, train_idx, test_idx, held_out, feature_set, _, reused_preprocessor in fold_contexts:
        for model_name in models:
            completed = load_completed_task(
                model_name, fold, held_out, feature_set.X, test_idx, outdir, args,
                allow_checkpoint=allow_resume and reused_preprocessor,
                allow_legacy_artifact=allow_resume and reused_preprocessor,
            )
            if completed is not None:
                completed_results.append(completed)
            else:
                pending_tasks.append((model_name, fold, held_out, feature_set.X, train_idx, test_idx))

    if pending_tasks:
        # Longest-processing-time ordering keeps slow recurrent/attention jobs
        # from becoming a serial tail after the shorter MLP/XGBoost jobs finish.
        priority = {"BiLSTM": 0, "Transformer": 1, "CNN": 2, "MLP": 3, "XGBoost": 4}
        pending_tasks.sort(key=lambda task: (priority[task[0]], task[1]))
        LOGGER.info(
            "Training %d pending model/fold jobs (%d recovered from checkpoints/artifacts)",
            len(pending_tasks), len(completed_results),
        )
        if parallel_jobs == 1:
            trained_results = [
                train_model_task(model_name, fold, held_out, X, y, groups, train_idx, test_idx, str(outdir), args)
                for model_name, fold, held_out, X, train_idx, test_idx in pending_tasks
            ]
        else:
            # Loky processes isolate TensorFlow state. Limiting each process to
            # one native thread prevents BLAS/TensorFlow oversubscription.
            with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
                trained_results = joblib.Parallel(
                    n_jobs=min(parallel_jobs, len(pending_tasks)),
                    max_nbytes="10M",
                    mmap_mode="r",
                    verbose=10 if args.log_level == "DEBUG" else 0,
                )(
                    joblib.delayed(train_model_task)(
                        model_name, fold, held_out, X, y, groups, train_idx, test_idx, str(outdir), args
                    )
                    for model_name, fold, held_out, X, train_idx, test_idx in pending_tasks
                )
        completed_results.extend(trained_results)
    else:
        LOGGER.info("All model/fold jobs recovered; no retraining required")

    context_by_fold = {fold: (test_idx, feature_set, preprocessor_path) for fold, _, test_idx, _, feature_set, preprocessor_path, _ in fold_contexts}
    for result in sorted(completed_results, key=lambda item: (item.fold, models.index(item.model_name))):
        model_name = result.model_name
        fold = result.fold
        held_out = result.held_out_patient
        test_idx, feature_set, preprocessor_path = context_by_fold[fold]
        test_prob = result.test_probabilities
        best_hyperparameters.setdefault(model_name, []).append({"fold": fold, "held_out_patient": held_out, "params": result.params, "inner_score": result.tune_score, "resumed": result.resumed})
        loss_histories[model_name].append(result.tune_history)
        fold_cell = compute_metrics(y[test_idx], test_prob)
        for metric, value in fold_cell.items():
            all_fold_metrics.append({"model": model_name, "fold": fold, "held_out_patient": held_out, "aggregation": "cell_fold", "metric": metric, "value": value})
        patient_frame = pd.DataFrame({"patient_id": groups[test_idx], "sample_id": data.metadata.iloc[test_idx]["sample_id"].to_numpy(), "y": y[test_idx], "probability": test_prob, "model": model_name, "fold": fold})
        patient_fold = aggregate_predictions(patient_frame, "patient")
        for metric, value in compute_metrics(patient_fold["y"].to_numpy(), patient_fold["probability"].to_numpy()).items():
            all_fold_metrics.append({"model": model_name, "fold": fold, "held_out_patient": held_out, "aggregation": "patient_fold", "metric": metric, "value": value, "n_units": int(len(patient_fold))})
        prediction_frames[model_name].append(patient_frame)
        record = FitRecord(model_name, fold, str(held_out), result.params, result.model_path, str(preprocessor_path), feature_set.feature_names, test_idx, feature_set.X[test_idx], None)
        if model_name == shap_target:
            shap_records[model_name].append(record)

    # Load only the selected attribution models after parallel training. Keeping
    # models out of worker return values avoids expensive Keras serialization.
    for record in shap_records.get(shap_target, []):
        record.model = load_saved_model(record.model_name, Path(record.model_path))

    pd.DataFrame(variance_rows).to_csv(outdir / "pca_variance_by_fold.csv", index=False)
    plot_pca_variance(pca_records, outdir)
    (outdir / "best_hyperparameters.json").write_text(json.dumps(best_hyperparameters, indent=2, default=str), encoding="utf-8")

    metric_rows = list(all_fold_metrics)
    all_predictions: Dict[str, pd.DataFrame] = {}
    for model_name, frames in prediction_frames.items():
        predictions = pd.concat(frames, ignore_index=True)
        all_predictions[model_name] = predictions
        predictions.to_csv(outdir / f"cell_level_predictions_{model_name}.csv.gz", index=False, compression="gzip")
        patient_predictions = aggregate_predictions(predictions, "patient")
        patient_predictions.insert(0, "model", model_name)
        patient_predictions.to_csv(outdir / f"patient_level_predictions_{model_name}.csv", index=False)
        sample_predictions = aggregate_predictions(predictions, "sample")
        sample_predictions.insert(0, "model", model_name)
        sample_predictions.to_csv(outdir / f"sample_level_predictions_{model_name}.csv", index=False)
        for aggregation, frame, unit in (("cell_pooled", predictions, "patient"), ("patient_pooled", patient_predictions, "patient"), ("sample_pooled", sample_predictions, "sample")):
            metric_input = frame if aggregation == "cell_pooled" else frame.rename(columns={"probability": "probability"})
            metrics = compute_metrics(metric_input["y"].to_numpy(), metric_input["probability"].to_numpy())
            ci = patient_bootstrap(predictions, unit, args.bootstrap_replicates, args.seed + 1000) if aggregation != "cell_pooled" else patient_bootstrap(predictions, "patient", args.bootstrap_replicates, args.seed + 1000)
            for metric, value in metrics.items():
                lower, upper = ci[metric]
                metric_rows.append({"model": model_name, "fold": "pooled", "held_out_patient": "all", "aggregation": aggregation, "metric": metric, "value": value, "ci_lower": lower, "ci_upper": upper, "n_units": int(metric_input["patient_id" if aggregation == "patient_pooled" else "sample_id"].nunique()) if aggregation != "cell_pooled" else int(predictions["patient_id"].nunique())})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(outdir / "model_evaluation_metrics.csv", index=False)
    pd.DataFrame([{"model": model, **compute_metrics(df["y"].to_numpy(), df["probability"].to_numpy())} for model, df in all_predictions.items()]).to_csv(outdir / "pooled_cell_metrics.csv", index=False)
    export_loss_curves(loss_histories, outdir)

    compute_shap_summary(shap_target, shap_records[shap_target], data, outdir, args)
    run_metadata = {
        "seed": args.seed,
        "timepoint_mode": args.timepoint_mode,
        "feature_set": args.feature_set,
        "models": models,
        "n_cells": int(data.X.shape[0]),
        "n_genes": int(data.X.shape[1]),
        "n_patients": int(len(patients)),
        "parallel_jobs": parallel_jobs,
        "detected_gpus": gpu_count,
        "fast_mode": bool(args.fast),
        "completion_mode": bool(getattr(args, "completion_mode", False)),
        "n_trials": args.n_trials,
        "tune_epochs": args.tune_epochs,
        "epochs": args.epochs,
        "sequence_channels": args.sequence_channels,
        "pca_top_50_cumulative_by_fold": {str(fold): float(np.sum(var)) for fold, var in pca_records},
        "software": software_versions(),
    }
    (outdir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Pipeline completed. Outputs: %s", outdir.resolve())
    return outdir


def software_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}
    for module_name in ("sklearn", "xgboost", "tensorflow", "shap", "scanpy", "anndata"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = "unavailable"
    return versions


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-data", default=None, help="Existing .h5ad or raw-data directory")
    parser.add_argument("--data-dir", default="/kaggle/working/Data", help="Download/cache directory for GEO raw files")
    parser.add_argument("--download", action="store_true", help="Download GSE300475 when no local input is found")
    parser.add_argument("--output-dir", default="results", help="Artifact directory")
    parser.add_argument("--resume", action="store_true", help="Reuse compatible checkpoints and legacy model artifacts already in output-dir")
    parser.add_argument("--resume-zip", default=None, help="Previous Kaggle results.zip; restores results and extracted raw data before resuming")
    parser.add_argument("--parallel-jobs", type=int, default=0, help="Independent model/fold processes; 0 auto-detects up to 4")
    parser.add_argument("--fast", action="store_true", help="Use a substantially smaller, publication-draft search budget for Kaggle")
    parser.add_argument("--full-budget", action="store_true", help="Disable the automatic bounded CPU completion budget on Kaggle")
    parser.add_argument("--timepoint-mode", choices=("baseline", "all", "exclude_recurrence"), default="baseline")
    parser.add_argument("--feature-set", choices=("combined", "gene_only", "tcr_only"), default="combined")
    parser.add_argument("--models", default=",".join(MODEL_NAMES), help="Comma-separated model names")
    parser.add_argument("--n-top-genes", type=int, default=1500)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--pca-batch-size", type=int, default=2048)
    parser.add_argument("--max-kmers", type=int, default=256)
    parser.add_argument("--max-train-cells-per-patient", type=int, default=10000, help="Cap used for each outer fit to control runtime; 0 uses all cells")
    parser.add_argument("--tune-cells-per-patient", type=int, default=2000, help="Cap used by inner tuning; 0 uses all cells")
    parser.add_argument("--n-trials", type=int, default=4, help="Randomized trials per architecture; every architecture is tuned")
    parser.add_argument("--tune-epochs", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sequence-channels", type=int, default=1, help="Pack adjacent scalar features into channels for CNN/BiLSTM/Transformer; values >1 reduce sequence cost")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--shap-model", choices=("auto", "all", *MODEL_NAMES), default="auto")
    parser.add_argument("--shap-cells", type=int, default=500)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--synthetic", action="store_true", help="Run a small local smoke dataset")
    parser.add_argument("--synthetic-cells-per-patient", type=int, default=120)
    parser.add_argument("--synthetic-genes", type=int, default=80)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    on_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists())
    if not args.resume_zip and on_kaggle:
        packaged_archive = Path.cwd() / "results.zip"
        if packaged_archive.exists():
            args.resume_zip = str(packaged_archive)
        else:
            resume_candidates = sorted(Path("/kaggle/input").rglob("results.zip"))
            if len(resume_candidates) == 1:
                args.resume_zip = str(resume_candidates[0])
    packaged_models = Path(args.output_dir).expanduser() / "models"
    if on_kaggle and packaged_models.exists() and any(packaged_models.iterdir()):
        args.resume = True
    args.completion_mode = bool(on_kaggle and not args.full_budget)
    if args.completion_mode:
        args.max_train_cells_per_patient = 2500
        args.tune_cells_per_patient = 400
        args.n_trials = 1
        args.tune_epochs = 3
        args.epochs = 8
        args.sequence_channels = 16
        args.bootstrap_replicates = min(args.bootstrap_replicates, 500)
        args.shap_cells = min(args.shap_cells, 150)
    elif args.fast:
        args.max_train_cells_per_patient = min(args.max_train_cells_per_patient, 6000) if args.max_train_cells_per_patient > 0 else 6000
        args.tune_cells_per_patient = min(args.tune_cells_per_patient, 1000) if args.tune_cells_per_patient > 0 else 1000
        args.n_trials = min(args.n_trials, 2)
        args.tune_epochs = min(args.tune_epochs, 6)
        args.epochs = min(args.epochs, 18)
        args.sequence_channels = max(args.sequence_channels, 16)
        args.bootstrap_replicates = min(args.bootstrap_replicates, 1000)
        args.shap_cells = min(args.shap_cells, 250)
    # Kaggle script kernels do not provide a convenient command-line argument
    # field in kernel-metadata.json. Auto-enable the public GEO download only
    # in the Kaggle runtime; local runs remain explicit and reproducible.
    if not args.input_data and not args.synthetic and on_kaggle:
        args.download = True
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    started = time.time()
    try:
        run_pipeline(args)
    except Exception:
        LOGGER.exception("Revision pipeline failed")
        return 1
    LOGGER.info("Wall time %.1f minutes", (time.time() - started) / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
