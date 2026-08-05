# Original Paper

# Baseline Peripheral Blood Single-Cell Multimodal Profiling for Chemoimmunotherapy Response in Early-Stage Hormone Receptor-Positive Breast Cancer: Exploratory Model Development Study

Anmol Singh Josan

Eastside Preparatory School, 10613 NE 38th Place, Kirkland, WA 98033, United States

Corresponding author: Anmol Singh Josan, Eastside Preparatory School, 10613 NE 38th Place, Kirkland, WA 98033, United States

## Abstract

**Background:** Immune checkpoint inhibition is entering the treatment landscape for high-risk estrogen receptor-positive, human epidermal growth factor receptor 2-negative breast cancer, but response remains heterogeneous. Peripheral blood single-cell profiling may support minimally invasive biomarker discovery. Because treatment outcome is assigned to patients, analyses that treat thousands of cells as independent outcomes can severely overstate evidence.

**Objective:** This study evaluated an exploratory, leakage-controlled machine learning framework that combines baseline peripheral blood gene expression and T-cell receptor (TCR) features while treating the patient as the clinically relevant evaluation unit.

**Methods:** This retrospective secondary analysis used public single-cell RNA sequencing and paired TCR data from GSE300475. The primary analysis included 39,532 pretreatment cells from 4 patients (2 responders and 2 nonresponders). In each outer leave-one-patient-out fold, highly variable gene selection, scaling, principal component analysis (PCA), and TCR 3-mer vocabulary construction were fitted only with training patients; uniform manifold approximation and projection coordinates were excluded. Multilayer perceptron (MLP), 1-dimensional convolutional neural network (CNN), bidirectional long short-term memory network (BiLSTM), Transformer, and XGBoost models were tuned with inner patient-grouped cross-validation. Cell probabilities were averaged within each held-out patient. Confidence intervals (CIs) resampled whole patient clusters. Baseline transcriptome-only and TCR-only analyses, longitudinal sensitivity analyses, patient-label permutation tests, seed stability, and out-of-distribution perturbations were evaluated.

**Results:** All 4 neural architectures correctly separated the 4 held-out patients; patient-level area under the receiver operating characteristic curve (AUROC) and accuracy were 1.00. Pooled cell-level AUROC ranged from 0.996 (95% patient-clustered bootstrap CI 0.990-0.999) for the Transformer to 1.000 (95% CI 1.000-1.000) for the MLP. These values do not establish validation: only 6 balanced patient-label assignments were possible, giving a minimum attainable exact one-sided P value of .167. A retrained Transformer permutation control with 5 shuffled assignments gave P=.33. Baseline transcriptome-only Transformer AUROC was 1.00 at both cell and patient levels, whereas TCR-only AUROC was 0.00. The first 50 training-fold PCs explained 30.4%-31.3% of variance. Gene-PC ablation reversed all 4 patient classifications, while TCR-feature ablation caused no patient-label changes.

**Conclusions:** Baseline peripheral blood single-cell multimodal features contained a separable signal in this 4-patient cohort, but the evidence is hypothesis generating and cannot support clinical prediction claims. The primary contribution is a reproducible framework for leakage-controlled preprocessing, nested patient-grouped evaluation, explicit outcome aggregation, and patient-clustered uncertainty. Larger prospectively collected cohorts and external validation are required.

**Keywords:** breast cancer; immunotherapy; single-cell RNA sequencing; T-cell receptor; machine learning; leave-one-patient-out cross-validation; biomarkers

# Introduction

## Background

Hormone receptor-positive, human epidermal growth factor receptor 2-negative breast cancer is biologically heterogeneous and has historically been less responsive to immune checkpoint inhibition than triple-negative disease. Recent phase 3 studies nevertheless showed that adding programmed cell death protein 1 blockade to neoadjuvant chemotherapy can increase pathologic complete response in selected patients with high-risk estrogen receptor-positive disease [1,2]. These results intensify the need for biomarkers that identify patients most likely to benefit while limiting unnecessary toxicity.

Peripheral blood is attractive for longitudinal immune monitoring because it can be sampled repeatedly with less burden than tumor tissue. Single-cell RNA sequencing resolves transcriptional states that are obscured in bulk measurements, and paired V(D)J sequencing adds information about T-cell receptor sequence, clonal expansion, and repertoire structure [3-7]. The source study by Sun et al characterized dynamic systemic immune responses during chemoimmunotherapy in early-stage hormone receptor-positive breast cancer and released the cell-level data through the Gene Expression Omnibus [3]. The present work retains the original study motivation—linking peripheral immune state, cytotoxicity, interferon signaling, and TCR structure to therapeutic response—but narrows the predictive claim to pretreatment samples and the independent patient unit.

## Single-Cell and TCR Representation

Gene-expression profiles and TCR sequences describe related but distinct aspects of lymphocyte biology. Expression profiles reflect cell state, activation, differentiation, and technical variation. TCR complementarity-determining region 3 sequences encode antigen-recognition structure and can be represented through chain presence, sequence length, amino-acid composition, physicochemical summaries, and position-independent k-mer motifs [8-11]. DeepTCR and related neural approaches have shown that repertoire sequence concepts can be learned from high-dimensional TCR data [11]. However, missing V(D)J libraries, nonproductive chains, and sample-level repertoire properties can also reveal sample identity. These technical signals must therefore be modeled explicitly and interpreted cautiously.

Single-cell observations are nested within people. Cells from the same patient share outcome, genetics, treatment exposure, processing history, and immune environment; they are subsamples rather than independent outcome observations [12,13]. This distinction is critical in the current cohort because tens of thousands of cells arise from only 4 patients. Cell-level prediction remains useful for forming a patient score, but statistical evidence must be summarized and resampled at the patient level.

## Machine Learning and Study Objective

Machine learning provides a useful framework for combining nonlinear transcriptomic and receptor-sequence representations. MLPs learn interactions among tabular features; 1-dimensional CNNs detect local feature patterns; BiLSTMs model ordered dependencies; Transformers use self-attention; and gradient-boosted trees provide a nonneural comparator [14-18]. In very small cohorts, these architectures can overfit patient identity even when the number of cells is large. Nested grouped validation, regularization, negative controls, calibration measures, and transparent loss histories are therefore necessary, although none can replace external validation [19-22].

The objective was to revise the original exploratory analysis around 4 principles: pretreatment prediction as the primary question; all feature construction confined to outer training patients; the same inner grouped tuning framework applied to every architecture; and patient-level evaluation reported alongside descriptive cell-level performance. Secondary objectives were to quantify TCR missingness, compare transcriptome-only and TCR-only information, examine longitudinal sensitivity, test seed and label stability, and map model attributions through PCA loadings to genes and pathways.

# Methods

## Data Source and Study Cohort

This was a retrospective secondary analysis of deidentified data made public by Sun et al [3] under Gene Expression Omnibus accession GSE300475. No participants were recruited for this analysis. The source dataset contains 11 peripheral blood samples and 100,067 cells from 4 patients with high-risk hormone receptor-positive, human epidermal growth factor receptor 2-negative breast cancer. Two patients met the source study responder definition of residual cancer burden class 0 or I at surgery, and 2 were nonresponders with residual cancer burden class II or III. Residual cancer burden provides a standardized quantitative measure of residual disease after neoadjuvant therapy [23].

The primary predictive cohort was restricted to one baseline sample per patient, before treatment-associated or recurrence-associated immune states could enter the predictors. It contained 39,532 cells: 17,654 from responders and 21,878 from nonresponders. Longitudinal sensitivity analyses used all 100,067 cells or excluded the recurrence samples while retaining baseline and on-treatment samples. Sample S8, the recurrence specimen from Patient 3, contained gene expression but no paired V(D)J library. These data are described as single-cell multimodal profiling because transcript abundance and paired receptor sequencing are measured from linked cells; they are not separate biological omes.

Table 1. Baseline cohort and productive T-cell receptor coverage.

| Patient | Outcome | Baseline cells | Productive alpha chain, % | Productive beta chain, % | Any productive chain, % |
|---|---|---:|---:|---:|---:|
| Patient 1 | Responder | 8931 | 44.9 | 52.7 | 54.5 |
| Patient 2 | Responder | 8723 | 28.3 | 33.9 | 35.4 |
| Patient 3 | Nonresponder | 10398 | 51.1 | 55.0 | 56.4 |
| Patient 4 | Nonresponder | 11480 | 43.9 | 48.9 | 50.5 |

## Timepoint Stratification and TCR Missingness

The executable exposes baseline, all-timepoint, and recurrence-excluded switches. Baseline was prespecified as primary because only pretreatment observations can support a prospective response-prediction interpretation. Analyses containing on-treatment or recurrence cells were treated as sensitivity analyses of longitudinal immune-state classification, not as validation of pretreatment prediction.

For each alpha and beta chain, missingness indicators were included explicitly. Absent sequence-derived values were zero-filled after the training-fold transformation was defined. K-mer vocabularies were learned only from productive sequences in the outer training patients; unseen test-patient k-mers mapped to zero. This approach retained cells without productive receptors and allowed the V(D)J-missing recurrence sample to enter the all-timepoint sensitivity analysis without inventing receptor sequences. Productive-chain prevalence was summarized by patient, response, and timepoint. Cell-level Fisher tests were exported as descriptive diagnostics but were not interpreted as patient-level hypothesis tests because the cells are clustered.

## Leakage-Controlled Feature Engineering

All supervised preprocessing occurred separately inside each outer fold. Training patients alone determined gene filtering, highly variable gene ranking, means and variances for standardization, PCA loadings, and TCR vocabulary. The held-out patient was transformed with those fixed objects. No UMAP coordinate, Leiden cluster label, full-dataset embedding, or outcome-informed global feature selection was used as a predictor. This organization follows the principle that every data-adaptive operation must be nested within model assessment [19,20].

Gene counts were library-size normalized, log transformed, filtered to variable genes in the training partition, standardized, and projected with incremental PCA following established single-cell analysis principles [36,37]. The original 50-component representation was retained as a regularizing feature budget so its behavior could be compared with the prior analysis. Cumulative explained variance was computed in every outer training set. Because 50 PCs did not reach an 80% target, the manuscript reports the achieved variance rather than claiming comprehensive retention. The code also supports selection of the smallest component count reaching a prespecified variance threshold, subject to a documented cap [38].

TCR features included chain-presence indicators, sequence lengths, amino-acid composition and physicochemical summaries, and position-independent 3-mer counts for alpha and beta chains. Sample-level repertoire diversity values were not used as cell-intrinsic predictors in the revised primary feature matrix because they could encode sample identity. Gene and receptor matrices were standardized with training-fold statistics before concatenation.

## Outer Leave-One-Patient-Out Cross-Validation

The outer loop used leave-one-patient-out cross-validation (LOPO-CV), producing exactly one held-out prediction set for each patient. At no point were cells from the held-out patient used for fitting preprocessing, selecting hyperparameters, early stopping, or training the corresponding final model. The models generated a response probability for every held-out cell; the clinically relevant patient probability was the arithmetic mean over all cells from that held-out patient. Sample probabilities were calculated similarly for longitudinal analyses.

The MLP used dense layers, dropout, L2 regularization, and early stopping. The CNN applied 1-dimensional filters to a channelized feature sequence. The BiLSTM processed the same channelized representation bidirectionally. The Transformer used multihead self-attention followed by a regularized classification head. XGBoost supplied a tree-ensemble comparison [14-18]. Neural models minimized binary cross-entropy with Adam. Class weights were derived only from outer-training data.

## Inner Patient-Grouped Hyperparameter Tuning

Optuna optimization was applied to all 5 architectures within each outer fold [24]. Inner GroupKFold partitions were grouped by patient identifier, leaving 3 patients available for tuning in each outer fold. The objective was mean validation log loss across the grouped inner splits. Tuned quantities included learning rate, batch size, dropout, final-layer width, and early-stopping patience for all neural networks; hidden-layer width and L2 penalty for the MLP; filter number and kernel size for the CNN; recurrent width for the BiLSTM; embedding width and attention-head number for the Transformer; and estimator count, tree depth, learning rate, row and column subsampling, and L2 regularization for XGBoost.

The supplied archive was generated in completion mode with 3 Optuna trials per architecture, 8 tuning epochs, and up to 30 final epochs. This is computationally broader and fairer than tuning only the MLP, but it remains a limited search. The revised script defaults to 12 trials and supports larger user-specified budgets. Audit of the archive also identified an inappropriately low XGBoost learning-rate range and duplicated training/validation loss traces. The code now uses a tree-appropriate learning-rate range of 0.01-0.20 and records separate evaluation sets. The archived XGBoost result is reported as a failed comparator run and should be replaced by a full corrected rerun before making architecture-ranking claims.

## Outcome Aggregation and Statistical Analysis

Fold-level cell metrics included AUROC, accuracy, precision, recall, specificity, F1 score, Brier score, and log loss [25,26]. Metrics were then calculated on pooled out-of-fold cell predictions and on mean probabilities for each held-out patient. Patient and sample predictions were exported explicitly. Accuracy-related measures used a probability threshold of 0.50.

Ninety-five percent percentile CIs were estimated with 2000 bootstrap replicates. Whole patients—not individual cells—were resampled. For cell-level metrics, every cell belonging to a selected patient was retained together. For sample-level longitudinal metrics, all samples from a selected patient remained in the same bootstrap cluster. This preserves the requested metric level while respecting the hierarchy of cells within samples and patients [12,27]. With only 4 clusters, bootstrap intervals are discrete and can be degenerate, especially when every patient is classified correctly.

An exact negative control enumerated all 6 balanced assignments of 2 responder labels among 4 patients. A stronger computational control retrained the selected Transformer under 5 balanced shuffled assignments. The observed statistic was compared with the null distribution using a one-sided empirical P value. Three independent Transformer seeds evaluated optimization stability. Out-of-distribution stress tests added Gaussian noise, randomly masked features, or ablated the complete gene-PC or TCR block.

## Model Interpretability

The revised code uses SHAP Explainer for XGBoost and DeepExplainer for compatible neural models, with integrated gradients as a documented neural fallback [28,29]. The supplied archive used integrated gradients for the selected Transformer. Absolute feature attributions were summarized across held-out cells and folds. For gene-expression PCs, an attribution was back-projected by multiplying the absolute PC attribution by the absolute fold-specific loading for each gene and then averaging across folds. Signed back-projection was retained separately because a PC combines many correlated loadings and should not be interpreted as a causal gene effect.

Pathway summaries used prespecified cytotoxic-effector, interferon-response, T-cell activation, exhaustion/checkpoint, memory/naive, and mitochondrial gene sets. Empirical enrichment compared each set's mean attribution with random gene sets of equal size. These analyses evaluate concentration of model attribution, not differential expression, causal mechanism, or independent patient-level association.

## Software and Reproducibility

The archived run used Python 3.12.13, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1 [35], XGBoost 3.2.0, TensorFlow 2.20.0, SHAP 0.51.0, and Optuna 4.9.0. The fixed random seed was 93768. The executable exports cell-, sample-, patient-, fold-, and pooled metrics; clustered CIs; hyperparameters; preprocessing objects; PCA variance; loss curves; attribution summaries; sensitivity analyses; robustness controls; and software metadata. Standalone figures are saved without embedded titles or captions at 600 dots per inch. Reporting was aligned with the emphasis of TRIPOD+AI on transparent clinical prediction-model methods and limitations [30].

## Ethics Considerations

This work was a secondary computational analysis of public, deidentified data. No new participant contact, intervention, recruitment, or collection occurred. Ethical oversight and consent for the original clinical study are described by Sun et al [3].

# Results

## Cohort and TCR Coverage

The baseline cohort contained 39,532 cells, with 8931-11,480 cells per patient (Table 1). A productive alpha or beta chain was observed in 35.4%-56.4% of cells across patients. At baseline, 45.0% of responder cells and 53.3% of nonresponder cells contained at least one productive chain. These are descriptive cell fractions rather than independent group estimates. The all-timepoint cohort contained the original 100,067 cells across 11 samples; the recurrence sample S8 contributed 12,832 gene-expression profiles with both receptor chains coded as missing.

## Training-Fold PCA Variance

The first 50 training-derived PCs explained 31.30%, 30.99%, 30.36%, and 30.81% of gene-expression variance when Patients 1-4, respectively, were held out (Figure 1). The representation therefore retained approximately one-third of the training-fold variance and did not meet an 80% threshold. This makes the 50-component choice a fixed regularization budget, not a statistically complete representation.

Figure 1. Cumulative explained gene-expression variance across principal components fitted independently within each outer training fold. The dashed line marks the 80% target.

## Cell- and Patient-Level Model Performance

The neural models produced high pooled cell-level discrimination, with AUROCs from 0.996 to 1.000 and patient-clustered CIs shown in Table 2. Patient aggregation yielded the same ordering for every neural architecture: the mean probabilities of both responders exceeded 0.50 and those of both nonresponders were below 0.50 (Table 3 and Figure 2). Thus, neural patient-level AUROC and accuracy were 1.00. The MLP had the lowest pooled cell Brier score (0.00047), while the Transformer Brier score was 0.0229. The archived XGBoost run reversed the patient ordering and had AUROC and accuracy of 0.00; because its learning-rate search was defective, it cannot support a fair architecture comparison.

Table 2. Pooled out-of-fold performance with patient-clustered 95% confidence intervals.

| Model | Evaluation level | AUROC (95% CI) | Accuracy (95% CI) | Brier score (95% CI) |
|---|---|---|---|---|
| Multilayer perceptron | Cell | 1.000 (1.000-1.000) | 0.999 (0.999-1.000) | 0.0005 (0.0002-0.0008) |
| Convolutional neural network | Cell | 1.000 (0.999-1.000) | 0.992 (0.983-0.998) | 0.0096 (0.0015-0.0180) |
| Bidirectional long short-term memory network | Cell | 0.997 (0.995-1.000) | 0.979 (0.965-0.998) | 0.0169 (0.0025-0.0285) |
| Transformer | Cell | 0.996 (0.990-0.999) | 0.972 (0.951-0.991) | 0.0229 (0.0072-0.0421) |
| XGBoost, archived defective search | Cell | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.4178 (0.4058-0.4290) |
| Multilayer perceptron | Patient | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 0.000002 (0.000000-0.000004) |
| Convolutional neural network | Patient | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 0.0033 (0.0001-0.0068) |
| Bidirectional long short-term memory network | Patient | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 0.0012 (0.0001-0.0023) |
| Transformer | Patient | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) | 0.0014 (0.0001-0.0029) |
| XGBoost, archived defective search | Patient | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.4183 (0.4057-0.4292) |

Table 3. Mean response probabilities for each held-out patient.

| Model | Patient 1, responder | Patient 2, responder | Patient 3, nonresponder | Patient 4, nonresponder |
|---|---:|---:|---:|---:|
| Multilayer perceptron | 1.000 | 1.000 | 0.002 | 0.001 |
| Convolutional neural network | 0.989 | 0.905 | 0.064 | 0.005 |
| Bidirectional long short-term memory network | 0.990 | 0.986 | 0.051 | 0.044 |
| Transformer | 0.955 | 0.988 | 0.056 | 0.010 |
| XGBoost, archived defective search | 0.359 | 0.343 | 0.653 | 0.636 |

Figure 2. Mean response probability across cells for each held-out patient. Orange bars denote responders, green bars denote nonresponders, and the dashed line marks the 0.50 classification threshold.

The apparently perfect neural patient metrics represent only 4 predictions. Every valid two-class bootstrap resample preserved their ordering, so the patient-level CIs were degenerate at 1.00. Exact enumeration of the 6 balanced patient-label assignments yielded a minimum attainable one-sided P value of .167. When the Transformer was retrained under 5 shuffled balanced assignments, 1 shuffled assignment also achieved AUROC 1.00, producing an empirical P value of .33. Accordingly, the observed separation was not statistically distinguishable from the patient-label null at P<.05.

## Sensitivity and Robustness Analyses

The baseline transcriptome-only Transformer retained cell-level AUROC 1.000 (95% CI 1.000-1.000) and patient-level AUROC 1.00. In contrast, the TCR-only Transformer reversed the cell and patient ordering (AUROC 0.00). Adding receptor features did not improve discrimination over the transcriptome-only representation in this cohort. Longitudinal combined-feature analyses also separated the 4 patients, whether recurrence samples were excluded or all 11 samples were included (Table 4), but these analyses classify observations collected after treatment and cannot be interpreted as prospective baseline prediction.

Table 4. Transformer sensitivity analyses.

| Analysis | Cells | Cell AUROC (95% CI) | Cell accuracy | Patient AUROC | Patient accuracy |
|---|---:|---|---:|---:|---:|
| Baseline transcriptome only | 39532 | 1.000 (1.000-1.000) | 0.997 | 1.000 | 1.000 |
| Baseline T-cell receptor only | 39532 | 0.000 (0.000-0.000) | 0.011 | 0.000 | 0.000 |
| Baseline and on-treatment combined | 70547 | 0.999 (0.998-1.000) | 0.988 | 1.000 | 1.000 |
| All timepoints combined | 100067 | 0.999 (0.999-1.000) | 0.991 | 1.000 | 1.000 |

Across 3 seeds, the Transformer patient-level AUROC and accuracy remained 1.00. Mean probabilities ranged from 0.928 to 0.971 for Patient 1, 0.953 to 0.988 for Patient 2, 0.043 to 0.063 for Patient 3, and 0.010 to 0.031 for Patient 4. Stability across 3 optimization seeds is reassuring only within this dataset and does not address sampling uncertainty across patients.

Gaussian noise at standard deviations from 0.05 to 0.20 caused no patient-label changes. Random masking of 50% of features changed 2 labels and reduced patient AUROC to 0.75 and accuracy to 0.50. Ablating all TCR features changed mean patient probabilities by 0.012 and caused no label changes. Ablating all gene PCs changed probabilities by 0.959 on average, reversed all 4 labels, and reduced patient AUROC and accuracy to 0.00. These perturbations reinforce the conclusion that transcriptomic components, not TCR features, drove the archived Transformer.

## Loss Curves and Overfitting Assessment

Figures 3-6 display training and validation binary cross-entropy across the 8 tuning epochs for the neural architectures. Mean training and validation losses generally declined together during this short horizon; fold ranges were broad, reflecting only 3 training patients per outer fold. The curves do not show a sustained widening gap over 8 epochs, but neither the short traces nor early stopping can demonstrate absence of overfitting at an effective sample size of 4. The archived XGBoost curve is shown separately (Figure 7); its original implementation duplicated the validation series rather than recording a distinct training evaluation set. The revised code corrects this logging defect.

Figure 3. Multilayer perceptron training and validation binary cross-entropy across patient-grouped inner validation folds.

Figure 4. Convolutional neural network training and validation binary cross-entropy across patient-grouped inner validation folds.

Figure 5. Bidirectional long short-term memory network training and validation binary cross-entropy across patient-grouped inner validation folds.

Figure 6. Transformer training and validation binary cross-entropy across patient-grouped inner validation folds.

Figure 7. Archived XGBoost evaluation-loss trace. The corrected executable records distinct training and validation log-loss series in future runs.

## Model Attribution and Gene-Loading Mapping

The archived selected-Transformer attribution summary was led by gene-expression principal components 4, 3, 7, 1, and 2, with several alpha- and beta-chain physicochemical summaries also appearing among the 25 largest features (Figure 8). Fold-specific modality fractions were variable: the gene-expression block accounted for 45.9%-86.8% of absolute attribution across folds. Pairwise fold attribution correlations were modest (Spearman 0.32-0.47), and top-25 gene overlap ranged from 0.56 to 0.72, emphasizing uncertainty in the ranking.

Back-projection through PCA loadings ranked CD14, CLIC3, S100A12, MT-CO2, MATK, CCL4, MS4A6A, IL2RB, FGFBP2, KLRF1, KLRG1, and GZMH among the leading genes. Prespecified attribution enrichment was strongest for mitochondrial, cytotoxic-effector, memory/naive, and T-cell activation sets; interferon-response attribution was also above its random-set null, whereas the exhaustion/checkpoint set had no mapped attribution (Table 5). These empirical values quantify where this fitted model concentrated attribution and are not patient-level biological association P values.

Table 5. Selected gene and pathway attribution results from the Transformer.

| Result type | Gene or pathway | Attribution summary | Empirical comparison |
|---|---|---:|---:|
| Gene | CD14 | 0.1154 | Not applicable |
| Gene | CLIC3 | 0.1151 | Not applicable |
| Gene | S100A12 | 0.1146 | Not applicable |
| Gene | MATK | 0.1119 | Not applicable |
| Gene | CCL4 | 0.1105 | Not applicable |
| Gene | IL2RB | 0.1060 | Not applicable |
| Gene | FGFBP2 | 0.1035 | Not applicable |
| Gene | GZMH | 0.0966 | Not applicable |
| Pathway | Cytotoxic effector | 34.57-fold enrichment | P=.0005 |
| Pathway | Interferon response | 6.55-fold enrichment | P=.003 |
| Pathway | T-cell activation | 25.35-fold enrichment | P=.0005 |
| Pathway | Memory and naive T-cell state | 28.51-fold enrichment | P=.0005 |
| Pathway | Mitochondrial genes | 39.82-fold enrichment | P=.0005 |
| Pathway | Exhaustion and checkpoint | No mapped attribution | P=1.00 |

Figure 8. Mean absolute integrated-gradient attribution for the 25 leading selected-Transformer features. Bars show fold means and horizontal lines show between-fold ranges. The revised executable preferentially uses DeepSHAP and records integrated gradients only as a fallback.

# Discussion

## Principal Findings

This revision changes the evidentiary focus from approximately 100,000 labeled cells to 4 independent patients. In the baseline-only LOPO analysis, each neural architecture assigned higher mean probabilities to the 2 responders than to the 2 nonresponders. The same data also showed why a perfect patient AUROC must not be read as validation: 4 probabilities permit only 6 balanced label assignments, so even perfect ordering cannot achieve an exact one-sided P value below .167. The retrained permutation experiment and patient-clustered bootstrap further demonstrate that more cells do not compensate for the absence of independent patients.

The transcriptome-only sensitivity analysis performed at least as well as the combined model, whereas the TCR-only model failed. Likewise, gene-PC ablation reversed all classifications and TCR ablation changed none. These convergent analyses do not support the original suggestion that receptor encoding improved prediction or calibration. In this dataset, the dominant separable signal was transcriptomic. That signal could represent response biology, cell-composition differences, patient identity, sample processing, or a combination of these sources.

## Comparison With Prior Work

Single-cell tumor and blood studies have associated immunotherapy with clonal replacement, cytotoxic lymphocyte states, interferon programs, and dynamic systemic immunity [3-7,31,32]. TCR sequence models can recover antigen-specific motifs and repertoire structure [8-11], while paired transcriptome-receptor approaches can link clonotypes to cellular state. The current gene-loading results—CCL4, FGFBP2, IL2RB, KLRF1, and GZMH among the leading genes—are compatible with cytotoxic lymphocyte biology described in those studies. The interferon and T-cell activation attribution summaries are also biologically plausible.

Plausibility is not independent validation. Attribution methods explain a model's local sensitivity, not causality or the stability of a biomarker across populations [28,29]. The modest correlation of fold-specific gene rankings and the dominance of mitochondrial attribution reinforce the need to distinguish technical or compositional signals from response mechanisms. Future analyses should evaluate within-cell-type pseudobulk profiles, explicitly model patient and sample effects, and test whether leading pathways remain stable in an external cohort [12,13].

## Preprocessing and Dimensionality Reduction

Moving highly variable gene selection, scaling, PCA, and k-mer vocabulary construction inside each outer training fold removes an important source of test-patient leakage. Excluding UMAP from supervised features also avoids using a visualization embedding whose geometry may depend on the complete dataset [33]. These changes are consequential because a global embedding can transmit information from a held-out patient even when the classifier itself is fitted only on training rows.

The training-fold variance analysis gives a quantitative interpretation of the 50-PC choice. Approximately 30%-31% of expression variance was retained, far below the proposed 80% target. Fifty PCs can still serve as a regularizing cap in a small exploratory analysis, but the component count should be prespecified or selected within inner validation in a larger study. Variance retention alone does not determine predictive sufficiency; low-variance immune programs may be informative, whereas high-variance technical structure may not be.

## Timepoint and Missingness Interpretation

Restricting the primary analysis to baseline aligns the outcome claim with pretreatment prediction. The recurrence-excluded and all-timepoint analyses answer different questions: whether longitudinal immune states remain associated with eventual outcome after treatment has begun. Their high performance cannot be used as evidence of prospective prediction because treatment exposure and recurrence can alter both expression and repertoire state.

TCR coverage varied substantially across baseline patients, and S8 lacked receptor sequencing entirely. Explicit missingness indicators and zero filling prevent software failure and avoid discarding these cells, but they do not make missingness random. A model can learn technical provenance from missing-chain indicators. Reporting performance without sample-level repertoire summaries, comparing transcriptome-only and TCR-only models, and ablating receptor features reduced this ambiguity. The results indicate that TCR missingness was not required for the primary Transformer classification, although the cohort is too small to estimate a missingness effect.

## Architecture Tuning and Overfitting

Applying the same nested Optuna structure to MLP, CNN, BiLSTM, Transformer, and XGBoost corrects the prior unequal benchmark design. However, the inner loop contains only 3 patients, and the archived 3-trial completion budget is too small to establish optimal architectures. The low XGBoost learning-rate range demonstrates how a formally tuned model can still yield an invalid comparison when the search space is poorly chosen. The corrected code broadens that range, increases the default trial budget, and logs training and validation losses separately, but a corrected full run is still required before XGBoost is ranked against the neural models.

Deep learning is not justified here by an abundance of independent outcomes. Dropout, weight decay, early stopping, grouped validation, seed checks, and perturbation tests constrain model behavior but do not solve the 4-patient sample size. Model complexity should be reevaluated in a larger cohort against simpler penalized regression and tree baselines. Model selection and performance evaluation should ideally be separated, with an external site or study reserved for confirmation [19-22,34].

## Strengths and Limitations

The principal strengths are preservation of LOPO evaluation; pretreatment baseline as the primary question; training-only gene and receptor feature construction; removal of UMAP predictors; equal tuning logic across architectures; explicit cell, sample, and patient outputs; patient-clustered uncertainty; exact and retrained label controls; TCR missingness handling; modality ablation; seed stability; out-of-distribution perturbation; and fold-specific attribution mapping. The revised artifacts make the consequences of each modeling choice inspectable rather than relying on a single aggregate AUROC.

The overriding limitation is the cohort of 4 patients from one public study. Patient-level CIs can be degenerate because resampling 4 perfectly ordered outcomes repeatedly does not create new information. Fold metrics for a single held-out patient cannot estimate AUROC because that fold contains one outcome class. Inner grouped tuning is unstable, the archived search used only 3 trials, and the XGBoost comparator requires rerunning after correction. No independent cohort, prospective collection, prespecified clinical threshold, decision-curve analysis, or subgroup evaluation was available. The all-timepoint result may reflect treatment and recurrence states. Cell composition and processing batch could remain confounded with patient outcome. Finally, PCA back-projection distributes component attribution across correlated loadings and should be treated as a hypothesis-generation device.

## Future Directions

Future studies should recruit substantially more independent patients, retain a fully untouched external cohort, prespecify baseline collection and outcome timing, balance processing batches across response groups, and report sample and patient predictions. Within-cell-type pseudobulk or hierarchical models should be compared with cell-level learners. The number of PCs and all architecture hyperparameters should be selected only within grouped training data. TCR analyses should separate alpha- and beta-chain availability, quantify repertoire coverage, and test whether sequence features add value beyond expression and cell composition. Candidate cytotoxic and interferon programs should be validated with orthogonal assays rather than inferred from attribution alone.

## Conclusions

Leakage-controlled analysis of baseline peripheral blood single-cell multimodal data produced a separable signal in 4 held-out patients, driven primarily by gene-expression components. Patient aggregation and patient-level negative controls substantially change its interpretation: the clinically relevant evidence consists of 4 held-out probabilities, not 39,532 independent outcomes. The result is an exploratory proof of concept and is not ready for clinical use. Larger, prospectively collected and externally validated cohorts are required before peripheral single-cell features can support treatment selection.

## Acknowledgments

The author thanks Sun et al for making GSE300475 publicly available and acknowledges the patients and clinical teams who contributed to the original study. The author also thanks research mentor Dr Morteza Sarmadi for guidance.

## Data Availability

The source dataset is publicly available from the Gene Expression Omnibus under accession GSE300475. The revised analysis executable regenerates processed features from the public raw files and exports aggregate metrics, figures, model settings, and software metadata. Public repository information should be added to the journal metadata when the revision is committed and uploaded.

## Authors' Contributions

ASJ performed the computational analysis, software development, interpretation, visualization, and manuscript preparation.

## Funding

No external funding was received for this study.

## Conflicts of Interest

None declared.

## Abbreviations

AUROC: area under the receiver operating characteristic curve

BiLSTM: bidirectional long short-term memory network

CDR3: complementarity-determining region 3

CI: confidence interval

CNN: convolutional neural network

LOPO: leave-one-patient-out

MLP: multilayer perceptron

PCA: principal component analysis

SHAP: SHapley Additive exPlanations

TCR: T-cell receptor

UMAP: uniform manifold approximation and projection

V(D)J: variable, diversity, and joining gene segments

# References

1. Cardoso F, McArthur HL, Schmid P, et al. Pembrolizumab and chemotherapy in high-risk, early-stage, estrogen receptor-positive/human epidermal growth factor receptor 2-negative breast cancer: a randomized phase 3 trial. Nat Med. 2025;31:442-448. doi:10.1038/s41591-024-03415-7
2. Loi S, McArthur HL, Harbeck N, et al. Neoadjuvant nivolumab and chemotherapy in early estrogen receptor-positive breast cancer: a randomized phase 3 trial. Nat Med. 2025;31:433-441. doi:10.1038/s41591-024-03414-8
3. Sun X, Axelrod ML, Waks AG, et al. Dynamic single-cell systemic immune responses in immunotherapy-treated early-stage hormone receptor-positive breast cancer patients. NPJ Breast Cancer. 2025;11:65. doi:10.1038/s41523-025-00776-1
4. Bassez A, Vos H, Van Dyck L, et al. A single-cell map of intratumoral changes during anti-programmed death 1 treatment of patients with breast cancer. Nat Med. 2021;27:820-832. doi:10.1038/s41591-021-01323-8
5. Wu SZ, Al-Eryani G, Roden DL, et al. A single-cell and spatially resolved atlas of human breast cancers. Nat Genet. 2021;53:1334-1347. doi:10.1038/s41588-021-00911-1
6. Hao Y, Hao S, Andersen-Nissen E, et al. Integrated analysis of multimodal single-cell data. Cell. 2021;184:3573-3587.e29. doi:10.1016/j.cell.2021.04.048
7. Yost KE, Satpathy AT, Wells DK, et al. Clonal replacement of tumor-specific T cells following programmed death 1 blockade. Nat Med. 2019;25:1251-1259. doi:10.1038/s41591-019-0522-3
8. Glanville J, Huang H, Nau A, et al. Identifying specificity groups in the T cell receptor repertoire. Nature. 2017;547:94-98. doi:10.1038/nature22976
9. Dash P, Fiore-Gartland AJ, Hertz T, et al. Quantifiable predictive features define epitope-specific T cell receptor repertoires. Nature. 2017;547:89-93. doi:10.1038/nature22383
10. Jurtz VI, Jessen LE, Bentzen AK, et al. NetTCR: sequence-based prediction of T-cell receptor binding to peptide-major histocompatibility complexes using convolutional neural networks. Bioinformatics. 2018;34:i399-i407. doi:10.1093/bioinformatics/bty466
11. Sidhom JW, Larman HB, Pardoll DM, Baras AS. DeepTCR is a deep learning framework for revealing sequence concepts within T-cell repertoires. Nat Commun. 2021;12:1605. doi:10.1038/s41467-021-21879-w
12. Zimmerman KD, Espeland MA, Langefeld CD. A practical solution to pseudoreplication bias in single-cell studies. Nat Commun. 2021;12:738. doi:10.1038/s41467-021-21038-1
13. Squair JW, Gautier M, Kathe C, et al. Confronting false discoveries in single-cell differential expression. Nat Commun. 2021;12:5692. doi:10.1038/s41467-021-25960-2
14. LeCun Y, Bengio Y, Hinton G. Deep learning. Nature. 2015;521:436-444. doi:10.1038/nature14539
15. Hochreiter S, Schmidhuber J. Long short-term memory. Neural Comput. 1997;9:1735-1780. doi:10.1162/neco.1997.9.8.1735
16. Schuster M, Paliwal KK. Bidirectional recurrent neural networks. IEEE Trans Signal Process. 1997;45:2673-2681. doi:10.1109/78.650093
17. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. Adv Neural Inf Process Syst. 2017;30:5998-6008.
18. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Association for Computing Machinery; 2016:785-794. doi:10.1145/2939672.2939785
19. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics. 2006;7:91. doi:10.1186/1471-2105-7-91
20. Varoquaux G. Cross-validation failure: small sample sizes lead to large error bars. Neuroimage. 2018;180:68-77. doi:10.1016/j.neuroimage.2017.06.061
21. Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for developing a multivariable prediction model: part II—binary and time-to-event outcomes. Stat Med. 2019;38:1276-1296. doi:10.1002/sim.7992
22. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17:230. doi:10.1186/s12916-019-1466-7
23. Symmans WF, Peintinger F, Hatzis C, et al. Measurement of residual breast cancer burden to predict survival after neoadjuvant chemotherapy. J Clin Oncol. 2007;25:4414-4422. doi:10.1200/JCO.2007.10.6823
24. Akiba T, Sano S, Yanase T, Ohta T, Koyama M. Optuna: a next-generation hyperparameter optimization framework. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Association for Computing Machinery; 2019:2623-2631. doi:10.1145/3292500.3330701
25. Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic curve. Radiology. 1982;143:29-36. doi:10.1148/radiology.143.1.7063747
26. Brier GW. Verification of forecasts expressed in terms of probability. Mon Weather Rev. 1950;78:1-3. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
27. Efron B. Bootstrap methods: another look at the jackknife. Ann Stat. 1979;7:1-26. doi:10.1214/aos/1176344552
28. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Adv Neural Inf Process Syst. 2017;30:4765-4774.
29. Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. In: Proceedings of the 34th International Conference on Machine Learning. PMLR; 2017:3319-3328.
30. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378
31. Robert C. A decade of immune-checkpoint inhibitors in cancer therapy. Nat Commun. 2020;11:3801. doi:10.1038/s41467-020-17670-y
32. Borcherding N, Bormann NL, Kraus G. scRepertoire: an R-based toolkit for single-cell immune receptor analysis. F1000Res. 2020;9:47. doi:10.12688/f1000research.22139.2
33. McInnes L, Healy J, Melville J. UMAP: uniform manifold approximation and projection for dimension reduction. arXiv. Preprint posted online February 9, 2018. doi:10.48550/arXiv.1802.03426
34. Wolff RF, Moons KGM, Riley RD, et al. PROBAST: a tool to assess the risk of bias and applicability of prediction model studies. Ann Intern Med. 2019;170:51-58. doi:10.7326/M18-1376
35. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825-2830.
36. Stuart T, Butler A, Hoffman P, et al. Comprehensive integration of single-cell data. Cell. 2019;177:1888-1902.e21. doi:10.1016/j.cell.2019.05.031
37. Luecken MD, Theis FJ. Current best practices in single-cell RNA sequencing analysis: a tutorial. Mol Syst Biol. 2019;15:e8746. doi:10.15252/msb.20188746
38. Jolliffe IT, Cadima J. Principal component analysis: a review and recent developments. Philos Trans A Math Phys Eng Sci. 2016;374:20150202. doi:10.1098/rsta.2015.0202
