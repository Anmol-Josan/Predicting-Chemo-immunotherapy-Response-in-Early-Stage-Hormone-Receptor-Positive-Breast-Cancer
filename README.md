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
