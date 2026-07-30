# Predicting-Chemo-immunotherapy-Response-in-Early-Stage-Hormone-Receptor-Positive-Breast-Cancer

This repository applies unsupervised machine learning to single-cell RNA sequencing (scRNA-seq) data from hormone receptor–positive (HR+) breast cancer patients treated with nab‑paclitaxel and pembrolizumab. The primary aim is to identify immune cell clusters and transcriptomic biomarkers associated with treatment response.

**Project goals**
- Identify cell clusters predictive of immunotherapy response.
- Discover transcriptomic biomarkers (for example, expansion of GZMB+ cytotoxic CD8 T cells and interferon-driven signatures) that distinguish responders from non‑responders.
- Provide reproducible notebooks and helper scripts to reproduce the main analyses.

**Key insights**
- Responders exhibit expansion of GZMB+ cytotoxic CD8 T cells, dynamic TCR clonality, and interferon-driven monocyte and B cell signatures.
- Non-responders tend to display exhausted, static immune states.

**Repository layout**
- `Code/`: legacy exploratory notebooks and supporting scripts
- `run_revision_pipeline.py`: standalone leakage-safe revision pipeline

**Getting started**
1. Create a Python environment (recommended Python 3.9+).
2. Install dependencies:

    pip install -r requirements.txt

3. Run the revised analysis in baseline-only mode:

    python run_revision_pipeline.py --download --timepoint-mode baseline

   Use `--timepoint-mode all` for the longitudinal sensitivity analysis. The script fits HVG selection, scaling, PCA, and TCR vocabularies inside every LOPO training fold; UMAP is not used as a predictive feature.
4. The legacy notebooks remain available for historical figure reproduction: `Code/Main.ipynb` and `Code/hr-cancer.ipynb`.

**Kaggle performance and resume**

The pipeline trains independent model/fold jobs in parallel. In CPU runtimes it
uses up to two single-threaded worker processes to avoid TensorFlow worker
memory loss; in GPU runtimes it uses one
worker per detected GPU. Each completed job writes an atomic checkpoint, so a
Kaggle timeout can be resumed without repeating finished work.

Because this kernel is configured without a GPU, Kaggle uses a bounded expanded
budget: 3 tuning trials, 8 tuning epochs, up to 30 final epochs with early
stopping, 1,200 tuning cells and 6,000 final-fit cells per patient, and packed
sequences for CNN/BiLSTM/Transformer. Use `--full-budget` only on a machine with
enough time for the 50-epoch default.

The primary baseline run now exports Transformer integrated gradients with
signed and absolute feature effects, PC-to-gene back-projection, pathway
attribution enrichment, modality shares, fold-level attribution stability,
two additional initialization seeds per fold, simulated out-of-distribution
stress tests, exact balanced patient-label permutation tests, calibration/ECE,
five fully retrained label-permutation negative controls, decision curves,
threshold sensitivity, patient confusion matrices,
TCR-missingness tests, clonotype diversity summaries, and leakage audits.
Kaggle also runs Transformer sensitivity models for baseline gene-only,
baseline TCR-only, combined data excluding recurrence, and combined data across
all timepoints. `--skip-robustness` or `--skip-sensitivity-suite` can disable
these additions for debugging.

The GitHub workflow extracts the completed artifacts and rebuilds a compact
`results.zip` before pushing the kernel. A script kernel automatically resumes
from both the extracted artifacts and that packaged file. If running manually, upload
`results.zip` as a private Kaggle dataset and attach it to the kernel; the
script also detects exactly one attached file named `results.zip`. For an
explicit path, run:

    python run_revision_pipeline.py --download --timepoint-mode baseline \
      --resume-zip /kaggle/input/YOUR-DATASET/results.zip

For a much faster fresh draft run, add `--fast` (do not add it when continuing
the supplied full-budget archive). This reduces tuning cells, trials, epochs,
SHAP cells, and bootstrap replicates and packs scalar inputs into shorter
sequences for the recurrent/attention models. Use the full defaults for final
reported results. `--parallel-jobs N` overrides automatic CPU/GPU worker
selection. The supplied `kernel-metadata.json` disables Kaggle GPU usage.

**Data**
- Processed data and derived tables are available in `Processed_Data/` and `Output/Processed_Data/`.
- Large raw data files are not included in the repository; follow dataset acquisition instructions inside the notebooks.

**Notebook Execution Time**
The total wall time for all cells with timing measurements in the main analysis notebook (`Code/Main.ipynb`) is approximately 10 hours, 47 minutes, and 31 seconds when executed with GPU acceleration (Kaggle 2x T4 used during development). This includes data loading, processing, clustering, machine learning, and visualization steps.

**Reproducibility notes**
- Notebooks were developed and tested on Kaggle with GPU acceleration; local runs may require more memory/time.
- For long experiments, use a machine with GPU and >=30 GB RAM.

The four-patient response cohort is exploratory. Cell-level metrics are retained for transparency, while patient-level aggregation and patient-clustered uncertainty intervals are the primary validation summaries.

Last updated: July 26, 2026
