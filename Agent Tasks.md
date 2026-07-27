# Role & Objective
You are an expert computational biologist and machine learning engineer. You are assisting with code revisions for a manuscript submitted to *JMIR Bioinformatics and Biotechnology* (Manuscript #93768). 

Your goal is to:
1. Refactor the existing codebase to fix methodological issues, missing figures, and model training gaps raised during peer review.
2. Create a new standalone script (`run_revision_pipeline.py`) that executes the entire updated analysis pipeline end-to-end and outputs all required figures, tables, and training artifacts.

---

# Context & Critiques to Address

1. **Hyperparameter Tuning Across All Models:**
   - Previously, hyperparameter tuning was only conducted on the MLP model, making baseline comparisons (CNN, BiLSTM, Transformer) unfair.
   - Requirement: Implement hyperparameter tuning (e.g., using Optuna, Ray Tune, or Grid/Random Search) for **all models**: CNN, BiLSTM, and Transformer, in addition to MLP.

2. **Overfitting & Loss Plots:**
   - Cohort size is small (N=4 patients, ~100,067 single cells). Reviewers noted potential cell-level overfitting.
   - Requirement: Modify model training loops to track epoch-wise training and validation loss/metrics. Save and export clean loss curves (PNG/JPG) for all architectures to prove models are not severely overfitting.

3. **SHAP & Interpretability Plot Generation:**
   - The manuscript text references SHAP (SHapley Additive exPlanations) plots, but the files were missing/not generated.
   - Requirement: Implement a SHAP (or Integrated Gradients) explainer module for feature importance across model predictions and export high-resolution SHAP plots.

4. **Statistical Feature Selection Justification:**
   - The original code used the top 50 Principal Components of log-normalized counts without statistical justification.
   - Requirement: Add a function/script to compute cumulative explained variance ratio across PCs and automatically output the exact percentage of total variance explained by the top 50 PCs (or select $k$ PCs based on a target variance threshold like 80–90%).

5. **Sample & Timepoint Stratification:**
   - The dataset contains baseline, on-treatment, and post-treatment/recurrence samples (including sample S8 without TCR sequencing).
   - Requirement: Add data filtering options to explicitly stratify evaluation between:
     a) Baseline prediction models (pre-treatment samples only).
     b) Patient-level vs. cell-level evaluation splits (to handle the effective sample size limitation of $N=4$ patients).

6. **Correct Terminology in Pipeline Data Loading:**
   - Update dataset ingestion/metadata keys: replace references to "patient recruitment" with "Datasets", and handle single-cell RNA-seq + TCR as single-cell multimodal data rather than misclassifying non-existent modalities.

---

# Key Tasks & Implementation Requirements

### Task 1: Refactor Existing Modules
- **`models/` (or model definition scripts):**
  - Standardize model interfaces for MLP, CNN, BiLSTM, and Transformer models.
  - Implement dynamic hyperparameter configurations (learning rate, weight decay, hidden dimensions, dropout rates, layer counts).
  - Add validation loss tracking at every epoch during Leave-One-Patient-Out (LOPO) cross-validation.
- **`explainability/` (or interpretability logic):**
  - Implement `shap.Explainer` or `shap.DeepExplainer` on the trained models to generate feature summary plots and dot plots.
- **`data/` (or preprocessing scripts):**
  - Add PCA variance ratio logging (e.g., via `sklearn.decomposition.PCA`).
  - Add sample stratification filters (e.g., filtering out recurrence sample S8 or separating baseline vs. post-treatment).

### Task 2: Create `run_revision_pipeline.py`
Create a master execution script (`run_revision_pipeline.py`) that runs the complete pipeline when executed:

1. **Step 1: Data Ingestion & PCA Analysis**
   - Load single-cell expression and TCR data.
   - Run PCA on log-normalized counts, calculate cumulative explained variance for top 50 PCs, and log the summary to console/file.
2. **Step 2: Hyperparameter Tuning & Cross-Validation**
   - Execute hyperparameter tuning for MLP, CNN, BiLSTM, and Transformer architectures using Leave-One-Patient-Out (LOPO) CV.
   - Save best hyperparameter settings to `results/best_hyperparameters.json`.
3. **Step 3: Model Training & Artifact Generation**
   - Train final models using best hyperparameters.
   - Plot and save Training vs. Validation Loss curves for every model architecture (`results/loss_curves_[model_name].png`).
4. **Step 4: Explainability / SHAP Analysis**
   - Compute SHAP values for feature importance and save the SHAP summary plot (`results/shap_summary_plot.png`).
5. **Step 5: Metric Reporting**
   - Export both patient-level aggregated metrics and cell-level metrics to standard CSV/JSON for manuscript updates.

---

# Requirements for Output Files
- All generated plots (Loss curves, SHAP plots, PCA variance plots) must be exported as **standalone high-resolution PNG or JPG images** (600 DPI preferred).
- **Do NOT render internal titles or captions inside the plot images themselves** (JMIR formatting guidelines require figure labels and captions to be uploaded separately as metadata).

Please inspect the existing codebase, refactor the necessary existing files, and build `Code/HR-cancer.ipynb`.

Read the manuscript at MANUSCRIPT.MD and the changes requested at CHANGES.MD
