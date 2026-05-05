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
- `Code/`: main analysis notebook (`Main.ipynb`) and supporting scripts

**Getting started**
1. Create a Python environment (recommended Python 3.9+).
2. Install dependencies:

    pip install -r requirements.txt

3. Open and run the main analysis notebook: `Code/Main.ipynb` (this is the primary notebook to run).
4. To run scripted portions of the analysis:

    python Dev/main.py

**Data**
- Processed data and derived tables are available in `Processed_Data/` and `Output/Processed_Data/`.
- Large raw data files are not included in the repository; follow dataset acquisition instructions inside the notebooks.

**Notebook Execution Time**
The total wall time for all cells with timing measurements in the main analysis notebook (`Code/Main.ipynb`) is approximately 10 hours, 47 minutes, and 31 seconds when executed with GPU acceleration (Kaggle 2x T4 used during development). This includes data loading, processing, clustering, machine learning, and visualization steps.

**Reproducibility notes**
- Notebooks were developed and tested on Kaggle with GPU acceleration; local runs may require more memory/time.
- For long experiments, use a machine with GPU and >=30 GB RAM.

Last updated: February 27, 2026
