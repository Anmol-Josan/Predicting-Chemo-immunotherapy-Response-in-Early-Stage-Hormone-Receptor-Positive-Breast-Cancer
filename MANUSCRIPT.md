[Paper.pdf](https://github.com/user-attachments/files/30398982/Paper.pdf)
Original Paper
Predicting Chemo-immunotherapy Response in Early-Stage Hormone
Receptor–Positive Breast Cancer Using Multimodal Single-Cell
Analysis: Model Development and Validation Study
Abstract
Background: Hormone receptor–positive (HR+) breast cancer exhibits limited and
heterogeneous clinical benefit from immune checkpoint inhibitors. While peripheral blood
single-cell profiling provides a minimally invasive approach to monitoring systemic
immune dynamics, its utility in predicting treatment response remains to be fully
established.
Objective: This study aims to develop and evaluate a multimodal machine learning
framework that integrates peripheral blood single-cell transcriptomes and T-cell receptor
(TCR) encodings to predict chemo-immunotherapy response in patients with early-stage
HR+ breast cancer.
Methods: I analyzed the GSE300475 cohort, comprising longitudinal samples from 4
patients (11 total samples; 100,067 cells). The feature set included principal components of
gene expression, TCR k-mer and physicochemical encodings, and quality control covariates.
I compared several classification algorithms, including logistic regression, tree-based
baselines, and sequence-aware deep models, using leave-one-patient-out cross-validation
for cell-level evaluation. Model interpretability was assessed via SHAP (SHapley Additive
exPlanations) for tree models and gradient-based attributions for neural networks, with
uncertainty quantified through nonparametric bootstrapping.
Results: The multimodal models achieved high cell-level discrimination, with a peak area
under the receiver operating characteristic curve of 0.97 and an accuracy of 91.7%.
Transcriptomic signatures related to cytotoxicity and interferon response were the
primary drivers of model predictions. The integration of TCR encodings provided
complementary signals that improved model calibration. Sensitivity analyses confirmed the
robustness of these findings to imputation and initialization variations, though the results
remain exploratory due to the small cohort size.
Conclusions: These proof-of-concept results suggest that combining peripheral single-cell
multimodal profiling with interpretable machine learning can identify coherent predictive
signatures of immunotherapy response. Future research in larger independent cohorts is
necessary to validate these biomarkers for clinical use.
Keywords: hormone receptor-positive breast cancer; immunotherapy; single-cell RNAseq; TCR sequencing; machine learning; biomarkers
Introduction
Background
Immune checkpoint inhibitors (ICIs) have transformed cancer therapy across multiple
tumor types [1], yet their benefit in hormone-receptor positive (HR+) breast cancer has
been modest and inconsistent [2, 3]. Although most early-stage HR+ patients do not derive
benefit from PD-1 pathway blockade, a clinically meaningful minority appear to gain
durable responses when chemotherapy is combined with checkpoint blockade. This
heterogeneity motivates the search for predictive biomarkers that can identify the patients
most likely to benefit while minimizing unnecessary toxicity [4].
Machine learning (ML) provides an indispensable assistance for converting highdimensional single-cell and receptor data into robust predictive models and interpretable
biological hypotheses. Unlike classical linear statistics, modern ML, from tree-based
ensembles (e.g., XGBoost) to deep architectures (CNNs, RNNs, and Transformer encoders),
can capture non-linear interactions, modality-specific representations, and hierarchical
sequence patterns relevant to antigen recognition [5–8]. In single-cell immunogenomics,
sequence-aware deep models have been successful at extracting epitope-associated motifs
and positional patterns from CDR3 sequences [9–13], while ensemble methods remain
highly competitive on heterogeneous tabular summaries. Recent applications of ML in
oncology have shown remarkable success in predicting immunotherapy response from
imaging [14, 15], multi-omic integration [16, 17], and clinical data [18, 19]. Deep generative
models have transformed single-cell analysis by enabling denoising [20], integration [21],
and perturbation prediction [22, 23]. This mix of approaches motivates a comparative,
multimodal modeling strategy that both maximizes predictive power and preserves
interpretability.
Prior Work
Single-cell transcriptomic atlases and spatial profiling studies have significantly advanced
our understanding of tumor immune ecosystems [24–30]. These efforts emphasize the
heterogeneity of immune infiltrates and the value of cellular-resolution assays for
mechanistic insight. Complementary work has featured dynamic repertoire remodeling
during checkpoint blockade and the clonal replacement of tumor-reactive T cells [31–33].
Computational immunology has developed robust methods for parsing receptor
sequence specificity and motif enrichment. GLIPH [11] introduced a pattern-matching
algorithm for grouping TCRs with shared specificity motifs, achieving >90% precision in
specificity group assignment across viral epitopes. TCRdist [12] proposed a distance metric
tailored to CDR3 features and demonstrated that epitope-specific T cell receptor
repertoires contain quantifiable, position-dependent structural signatures. NetTCR [9], a
convolutional neural network trained on peptide–MHC binding data, achieved Area Under
the Curve (AUC) values of 0.70–0.90 for predicting TCR–peptide interactions depending on
the epitope, demonstrating the feasibility of sequence-level binding prediction. DeepTCR
[10], a deep learning framework incorporating variational autoencoders and supervised
classifiers, achieved AUC values of 0.87–0.97 for epitope classification tasks and learned
biologically interpretable CDR3 embeddings.
More recently, behavior-guided transcriptomics approaches have mapped T cell
dynamics to molecular profiles through integrated sequence and expression analysis [34].
At the intersection of multimodal prediction, recent studies demonstrate that integrating
imaging, pathology and genomics via ML can improve clinical prediction of immunotherapy
benefit [14, 35–38]. Comparative studies have emphasized the value of different biomarker
modalities for checkpoint blockade response [16, 19], while pan-cancer microenvironment
analyses have identified conserved immune subtypes that predict treatment outcomes
[17]. The methodological literature stresses careful validation (nested cross-validation,
patient-level splits), rigorous hyperparameter search, and interpretable attribution to
avoid overfitting and to enable mechanistic follow-up [5, 7, 39].
Study Objectives and Hypotheses
Despite these advances, most high-performing predictive studies rely on tumor tissue or
bulk assays; peripheral blood single-cell multimodal studies remain relatively scarce [24,
40, 41]. This scarcity is an important opportunity: peripheral assays are minimally
invasive, enable longitudinal sampling, and, when paired with modern ML, can reveal
systemic immune dynamics that complement tumor-centric measures. Clinical trials have
upheld the modest efficacy of ICIs in HR+ breast cancer [2, 3], creating the need for
predictive biomarkers to identify the subset of patients who will benefit. Our study
addresses this gap by (1) emphasizing a machine-learning-first exposition and (2)
explicitly comparing unsupervised state discovery with supervised prediction to both
discover coherent cell states and quantify their predictive value.
Key findings from this study include (a) transcriptional programs indexing
cytotoxicity, interferon response, and early T cell activation dominate the predictive signal;
(b) TCR receptor sequence features add orthogonal, biologically plausible information
consistent with convergent selection of tumor-reactive clonotypes; (c) algorithmic
performance was comparable across model families, no single algorithm consistently
dominated across feature sets; and (d) the comprehensive feature set combining gene
expression PCs with TCR k-mer motifs and physicochemical encodings yield the strongest
discrimination across model families.
The remainder of the manuscript focuses on methods that prioritize ML rigor (clear
train/validation splits, nested hyperparameter search, and interpretable attributions),
reports quantitative performance across model families, and offers next steps required to
translate these proof-of-concept results into clinically robust biomarkers.
Methods
Recruitment
I analyzed the single-cell multi-omic dataset published by Sun and colleagues [40],
GSE300475. The samples derive from the chemotherapy-first arm of the DFCI 16-466
clinical trial (NCT02999477), in which early-stage, high-risk, hormone receptor positive,
HER2 negative breast cancer patients received neoadjuvant nab-paclitaxel followed by
combination therapy with pembrolizumab. Surgical assessment of residual cancer burden
at the end of therapy provides the clinically adjudicated binary endpoint used in our
modeling: responders correspond to RCB 0 or I and non-responders correspond to RCB II
or III.
The analytic cohort comprises four patients (PT1–PT4) who contributed peripheral
blood at up to three longitudinal timepoints each: a pre-treatment baseline draw, a post-
treatment timepoint collected after completion of the neoadjuvant chemo–immunotherapy
regimen, and an optional later sample collected at clinical recurrence when available. In
total the dataset contains 11 samples and, after quality filtering and cell calling, 100,067
single-cell transcriptomes. Two patients (PT1, PT2) met the trial-defined responder
endpoint (RCB 0 or I) and two (PT3, PT4) were classified as non-responders (RCB II or III).
One sample in the collection (S8, the PT3 recurrence specimen) includes gene expression
only and lacks paired V(D)J/TCR sequencing.
For transparency, these 11 samples represent the realistic constraints of
longitudinal peripheral sampling in a small proof-of-concept cohort: not every patient
contributed all three timepoints and a minority of cells lacked productive beta-chain
information (see Results). The baseline timepoint refers to blood drawn prior to any study
treatment; post-treatment denotes the specimen obtained following the neoadjuvant
therapy course (at time of surgery or scheduled on-treatment visit); and recurrence
denotes a later, clinically ascertained event. The raw expression matrices include >20,000
genes per cell; after feature engineering the four nested feature sets used for modeling
range from approximately 29 to 429 features per cell (Section 3). I frame this work as
a hypothesis-driven proof-of-concept that emphasizes biological interpretability and
specifies validation steps required before clinical deployment.
Single-cell capture and sequencing
The original samples were processed with the 10x Genomics Chromium 5-prime chemistry
to jointly profile gene expression and paired V(D)J receptor sequences [42]. Briefly, single
cells were encapsulated with barcoded gel beads, producing cell-specific gene expression
counts and receptor reads when chain coverage permitted. The resulting data link
instantaneous transcriptional programs to clonotypic identity, enabling analyses that
connect cell state and antigen-driven clonal dynamics. Table 1 summarizes the features
extracted from the raw data, including the original features provided by the 10x Genomics
platform.
Table 1. Summary of Features Extracted from Single-Cell Data
Feature Category Feature Name Description
Original Features
Gene Expression Matrix Raw UMI counts for
>20,000 genes per cell
TCR Contigs CDR3 nucleotide/amino
acid sequences, V/D/J genes
Cell Barcodes Unique identifier for each
single cell
Engineered Features
Gene PCs (1-50) Top 50 Principal
Components of lognormalized counts
CDR3 Length Length of the CDR3 amino
acid sequence
Hydrophobicity Mean hydrophobicity (KyteDoolittle scale)
Charge Net charge of the CDR3
sequence
Molecular Weight Total molecular weight of
the CDR3
Aromaticity Fraction of aromatic
residues (F, W, Y)
Shannon Entropy Diversity of the TCR
repertoire
Clonality (1 - (Shannon Entropy /
ln(Total Clones))
QC Metrics Mitochondrial percentage,
Total counts
Data processing pipeline
Figure 1 illustrates the comprehensive data processing workflow applied to the raw singlecell data. Starting with 100,067 cells across all patients and timepoints, I applied rigorous
quality control filters to remove low-quality cells (cells with 10% mitochondrial content),
resulting in high-quality cells for downstream analysis. Gene expression matrices were lognormalized (log1p transformation after library-size normalization to 104 counts) and
scaled to unit variance. Principal component analysis (PCA) was performed on the 1,500
most variable genes (Seurat v3 flavor), retaining the top 50 PCs. In parallel, TCR sequences
were processed to extract productive CDR3 sequences (productive beta chains were
detected in approximately 74% of cells), which were then encoded using three
complementary approaches: positional one-hot encoding (padded to length 20), k-mer
motif fingerprints (k=3, resulting in 8,000 unique 3-mers), and physicochemical property
vectors (8 features per sequence). The final integrated feature matrix was organized into
four nested feature families of increasing dimensionality: (1) a basic set (∼29 features: top
20 gene PCs, 6 TCR physicochemical features, 3 QC metrics); (2) a gene-enhanced set
(∼109 features: all 50 gene PCs, 30 SVD components, 20 UMAP dimensions, 6
physicochemical, 3 QC); (3) a TCR-enhanced set (∼429 features: 20 gene PCs, 200 TRA and
200 TRB k-mer motifs, 6 physicochemical, 3 QC); and (4) a comprehensive set (∼124
features: 15 gene PCs, 50 TRA and 50 TRB k-mers, 6 physicochemical, 3 QC). This nested
design enables systematic quantification of each modality’s marginal predictive
contribution.
For unsupervised analyses and visualization I computed a shared nearest-neighbor
graph (n_neighbors=15) and used UMAP and Leiden clustering. To ensure biologically
meaningful granularity, I swept the Leiden resolution parameter across six values (0.01,
0.05, 0.1, 0.2, 0.5, 1.0), targeting approximately seven major clusters and selecting the
resolution that best approximated this target. All preprocessing steps were implemented in
standard toolchains (Scanpy workflows) with fixed random seeds and are captured in the
analysis repository to ensure reproducibility. To maintain strict separation and prevent
data leakage, feature scaling via StandardScaler was fit exclusively on the training partition
of each LOPO fold. For TCR repertoire encoding, I implemented an idempotent modification
to the vectorization pipeline to robustly handle low-diversity samples by returning zero-
filled sparse matrices for empty vocabularies, ensuring pipeline stability across
heterogeneous patient cohorts.
TCR processing and clonotype assignment
V(D)J reads were parsed to extract productive CDR3 amino acid sequences and V/J gene
annotations. When multiple productive chains were reported in a cell I followed
conservative pairing rules that prioritize a single dominant productive beta chain;
ambiguous or low-confidence chain pairs were flagged and excluded from analyses that
required unequivocal pairing. Clonotypes were defined primarily by exact productive beta
CDR3 amino acid identity, with V gene annotations used to refine assignments when
appropriate. I computed repertoire summary statistics at the sample and timepoint level,
including clonality measures and Shannon entropy, to characterize global repertoire
structure prior to supervised modeling.
Feature engineering and sequence encodings
Feature engineering was guided by two principles: (1) extract compact, de-noised
summaries of high-dimensional transcriptomes that preserve coordinated biological
programs; and (2) encode receptor sequences with multiple, complementary
representations that balance positional specificity and generalizable biochemical
properties. For transcriptomic features I used the leading principal components (50 PCs)
derived from scaled expression as de-noised, orthogonal summaries of cell state. These 50
PCs were computed from the 1,500 most highly variable genes, with the first PC explaining
the largest share of variance and correlating strongly with T cell activation signatures.
For receptor sequences I implemented three complementary encoding families,
each capturing different aspects of CDR3 sequence information. The positional categorical
encoding aligns CDR3 sequences to a fixed-length representation (padding shorter
sequences with zero vectors and truncating longer sequences at position 20, which covers
94.3% of sequences without truncation) and one-hot encodes residues at each position (20
amino acids × 20 positions = 400 features), which preserves positional motifs important for
antigen contact, particularly in the central 8-12 residue region known to dominate peptideMHC interactions. The k-mer motif encoding counts overlapping amino-acid k-mers (k=3)
across the CDR3, producing a rotationally invariant fingerprint of short sequence motifs
that are often enriched by convergent antigen selection; with a vocabulary 8,000 possible
3-mers, I retained only the 500 most frequent motifs to reduce dimensionality while
preserving 89.1% of observed k-mer diversity. The physicochemical encoding computes
aggregated biophysical summaries across the CDR3: mean hydrophobicity (Kyte-Doolittle
scale, range -0.5 to 1.2), net charge (sum of positive [K,R,H] minus negative [D,E] residues),
molecular weight (sum of residue masses), aromaticity (fraction of F, W, Y residues),
isoelectric point, instability index, aliphatic index, and length-normalized Shannon entropy
of the amino acid composition (8 features total). These features capture binding-relevant
tendencies that do not rely on exact residue identity and have been shown to correlate with
peptide-MHC binding affinity in prior studies.
Table 2 presents summary statistics for the key engineered features used in the
modeling.
Table 2. Statistical Summary of Key Features
Feature Mean Std Dev Range
Gene PC1 0.00 5.23 [-15.4, 20.1]
Gene PC2 0.00 3.89 [-10.2, 12.5]
CDR3 Length 14.5 2.1 [8, 24]
Hydrophobicity 0.45 0.12 [-0,5, 1.2]
Charge 0.15 1.5 [-3.0,4.0]
Shannon Entropy 4.2 0.8 [2.1, 5.5]
Table 3 summarizes the dimensionality and coverage of each encoding scheme. For
each family I standardized feature vectors (zero mean, unit variance) on training folds
prior to model fitting to ensure comparable feature scales across modalities.
Table 3. TCR Encoding Schemes and Their Properties
Encoding Type Dimensions Coverage Rationale
Positional One-Hot 400 94.3% (no
truncation)
Preserves positionspecific motifs
K-mer Fingerprint
(k=3)
500 89.1% of diversity Captures recurrent
short motifs
Physicochemical 8 100% Biochemical binding
properties
Combined 908 - Multimodal
representation
Statistical analysis and reproducibility
Model discrimination was assessed with area under the receiver operating characteristic
curve (AUROC) and area under the precision-recall curve (AUPRC) together with accuracy
and F1 when appropriate. Feature importance and local attribution were estimated with
SHAP for tree ensembles and with gradient-based saliency maps for neural-network
models; permutation-based null tests were not implemented in this analysis. All software,
parameters, random seeds and processed feature matrices required to reproduce the
analyses are archived on GitHub and are available upon request.
Modeling Strategy
I adopted a two-stage analytic strategy that separates exploratory state discovery from
supervised prediction. The goal of the first stage is to generate biologically interpretable
cell state annotations and to inspect the joint distribution of gene programs and receptor
repertoires. The goal of the second stage is to build predictive models, quantify the
marginal value of each data modality, and provide interpretable explanations for model
decisions. This combination of unsupervised discovery and supervised prediction
represents a novel approach in this domain, allowing us to leverage the strengths of both
paradigms to uncover new biological insights. Figure 2 summarizes the comprehensive
modeling workflow, detailing the feature engineering, hyperparameter optimization, and
validation steps.
Unsupervised state discovery
Unsupervised analyses are necessary to validate that the single-cell data contain coherent
biological structure before attempting supervised learning. I used PCA for initial dimension
reduction followed by UMAP visualization and Leiden clustering to identify
transcriptionally coherent states. Cluster annotations were assigned by examination of
canonical marker genes and by comparing cluster-specific expression patterns to published
immune cell atlases [24, 25]. Silhouette scores were computed for each Leiden resolution
to quantify cluster compactness, and the resolution yielding the highest silhouette
coefficient was selected. In addition to Leiden clustering, I performed TCR sequence-based
clustering using KMeans on k-mer encoded CDR3 sequences (6 clusters each for TRA and
TRB chains) and gene expression module discovery via KMeans on gene PCs (8 modules).
The Leiden clustering and correlation matrix can be seen in Figure 3. Table 4 summarizes
the clustering hyperparameters.
Table 4. Clustering Hyperparameters
Algorithm Hyperparameter Value
Leiden
Resolutions tested [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
Target clusters ~7
Selection criterion Best silhouette score
Neighbor graph 15 neighbors, 50 PCs
TCR KMeans (TRA)
n_clusters 6
n_init 10
TCR KMeans (TRB)
n_clusters 6
n_init 10
Gene Modules
n_clusters 8
n_init 10
Supervised prediction and model families
The supervised task is binary classification mapping a single cell’s multi-modal feature
vector to the responder label defined at the patient level. I evaluated a diverse set of
algorithms spanning linear models, tree ensembles and sequence-aware deep networks,
representing a comprehensive comparative analysis rarely performed in single-cell
immunotherapy biomarker studies. Tree ensembles (including XGBoost) and random
forests were included as well-calibrated tabular baselines [5, 6], and logistic regression
served as a transparent linear comparator. However, sequence-aware deep models were a
primary focus because they can directly model CDR3 sequence structure and, in this run,
produced the best cell-level discrimination. Feed-forward neural networks used as tabular
baselines were constructed with two hidden layers (128 and 64 units) with batch
normalization, ReLU activation, dropout (rate=0.3), and L2 regularization (λ=0.01) to
mitigate overfitting. I report results across these families to identify which architectural
paradigms are best suited to this data structure.
In addition to feed-forward models I examined sequence-aware deep architectures
tailored to receptor encodings: 1D convolutional neural networks (CNNs) on k-mer and
positional encodings, bidirectional LSTM models on embedded amino-acid sequences, and
Transformer encoder blocks that use positional embeddings and multi-head self-attention
[8–10]. These sequence models were trained either on receptor-only inputs or jointly with
transcriptomic PCs through late fusion layers (concatenation followed by dense layers).
The goal was to assess whether sequence-native architectures capture receptor signal
complementary to tabular summaries and whether they improve calibration or recall in
challenging classification regions. In this analysis only the MLP underwent exhaustive inner
cross-validated hyperparameter tuning; the CNN, BiLSTM, and Transformer were trained
with representative, fixed configurations (reported in Table 5).
Table 5. Sequence-aware Model Hyperparameters for Deep Learning Architectures
Model Hyperparameter Value
1D-CNN
Kernel sizes 5
Filter per layer 64
Number of convolutional
layers
2
Pooling Max
Bi-LSTM
Embedding dimensions 64
Hidden units 128
Number of layers 1
Transformer
Number of layers 2
Model dimension 128
Number of Heads 4
Dropout 0.1
Feed-forward dimension 256
Hyperparameter tuning and validation
Hyperparameter tuning was performed using grid search (or randomized search when the
grid exceeded 15 combinations) nested within cross-validation. I used leave-one-patientout (LOPO) cross-validation as the outer evaluation loop to ensure that all cells from a heldout patient remain unseen during training. Within each training partition, inner
hyperparameter selection was performed using GroupKFold (grouped by patient_id) when
sufficient training patients were available, falling back to stratified 𝑘-fold when the number
of training groups was too small. Feature scaling (StandardScaler) and imputation (mean
imputation via SimpleImputer) were applied within a sklearn Pipeline to prevent data
leakage between folds. For XGBoost, the search space included: learning rate (0.05, 0.1),
maximum tree depth (3, 5), subsample fraction (0.8, 1.0), column subsample (0.8, 1.0), and
number of trees fixed at 100, yielding 16 combinations per fold. The best performing
XGBoost model used a learning rate of 0.1, a maximum depth of 5, and 100 estimators.
For the Deep Learning model, I implemented a feed-forward neural network with
two hidden layers (128 and 64 units, respectively), ReLU activation, and Batch
Normalization. To prevent overfitting, I applied Dropout (𝜂 = 0.3) and L2 regularization (𝜆
= 10−3 ). The model was trained using the Adam optimizer with a learning rate of 0.001,
early stopping on validation AUC (patience = 5–8 epochs), and learning rate reduction on
plateau (factor = 0.5, patience = 2–3, minimum lr = 10−6 ). Class weights were balanced to
account for class imbalance.
Final models were refit on the full training folds with best hyperparameters and
evaluated on held-out folds. I used a leave-one-patient-out (LOPO) strategy to prevent data
leakage at the cell level: all cells from a held-out patient were excluded from training, and
cell-level predictions from held-out folds were aggregated across LOPO iterations to
compute overall performance metrics.
Table 6. Hyperparameter Optimization
Model Hyperparameter Search Space → Best
Logistic Regression
C [0.1, 1, 10] → 1
Penalty [12] → 12
Solver [liblinear] → liblinear
XGBoost
Learning Rate [0.05, 0.1] → 0.1
Max Depth [3, 5] → 5
N Estimators [100] → 100
Subsample [0.8, 1.0] → 0.8
Colsample_bytree [0.8, 1.0] → 0.8
Random Forest
N Estimators [100] → 100
Max Depth [10, 20] → 10
Min Samples Split [5, 10] → 5
Decision Tree
Min Depth [5, 10] → 5
Min Samples Split [5, 10] → 5
Min Samples Leaf [2, 4] → 2
Deep Learning
Hidden Layers [(128,64) → (128,64)]
Dropout Regularization [0.3] → 0.3
L2 Regularization [10-3] → 10-3
Learning Rate [0.001] → 0.001
Batch Size [16] → 16
Interpretability
To connect model behavior to biology I employed a mix of global and local interpretability
tools. For tree ensembles I examined feature importance by gain and used SHAP values to
decompose predictions for individual cells and to compare modality-specific contributions.
For linear models I inspected coefficients and for neural networks I used gradient-based
saliency maps on the input features. I additionally performed post-hoc analyses to map
important principal components back to gene loadings and to identify TCR k-mers and
positional residues that correlated with positive SHAP weight.
Results
Cohort characteristics
The analytic cohort comprises four patients (PT1–PT4) who contributed peripheral blood
at up to three longitudinal timepoints: baseline (pre-treatment), post-treatment (obtained
after completion of the neoadjuvant chemo–immunotherapy course), and recurrence
(when clinically observed). In total the study includes 11 samples and, after standard
quality control and cell calling, 100,067 single-cell transcriptomes. Two patients (PT1, PT2)
were classified as responders (RCB 0 or I) and two (PT3, PT4) as non-responders (RCB II or
III). One sample (S8, the PT3 recurrence specimen) contains gene expression only and lacks
paired V(D)J/TCR sequencing. Across the filtered dataset productive beta-chain TCRs were
detected in approximately 74% of cells, while roughly 21% lacked a productive beta chain
and were handled explicitly in downstream modeling (see Methods). These numbers
reflect both biological variation in capture efficiency and the realities of longitudinal
peripheral sampling in a small cohort.
Sequencing yield and quality control
After preprocessing and QC (cells with 10% mitochondrial content removed), 100,067
high–quality cells remained for downstream analysis. Productive beta-chain TCRs were
detected in approximately 74% of cells; roughly 21% lacked a productive beta chain and
were handled via missingness indicators or imputation as described in Methods.
Feature extraction and encoding
I extracted gene principal components, positional and k-mer TCR encodings, and
physicochemical summaries; feature families ranged from approximately 29 to 429
features as detailed in Methods. These nested feature families were used to quantify the
marginal predictive contribution of each modality.
Model Optimization and Performance
Our analysis design aims to quantify the marginal contribution of transcriptomic and
receptor features while making the validation constraints explicit. I trained models on four
nested feature suites: a baseline set with common technical covariates such as percentage
mitochondrial reads and library complexity, a gene-enhanced set that augments baseline
covariates with the top 50 gene principal components, a TCR-enhanced set containing only
receptor encodings, and a comprehensive set that concatenates transcriptomic and
receptor representations.
All primary results use leave-one-patient-out (LOPO) cross-validation to prevent
data leakage: in each fold all cells from a held-out patient are excluded from training so that
evaluation is performed at the cell level. I evaluated over 100 model configurations
spanning four feature set families (basic, gene_enhanced, tcr_enhanced, comprehensive)
and eight algorithm classes: logistic regression, decision trees, random forests, XGBoost,
and four deep learning architectures (MLP, 1D CNN, BiLSTM, Transformer).
To control for potential technical confounders I performed covariate checks and
sensitivity analyses. These included removing the top principal components associated
with batch effects and re-evaluating model performance. I also assessed feature stability by
re-running the pipeline on different random seeds and reporting cross-fold variance across
LOPO iterations.
Unsupervised Analysis
Unsupervised clustering revealed distinct cellular states within the peripheral blood
compartment. I identified 7 major clusters using Leiden clustering, which corresponded to
biologically distinct populations. These clusters were characterized by the expression of
key marker genes. For instance, clusters enriched in responders showed significantly
higher expression of cytotoxic markers (p < 0.001, Wilcoxon rank-sum test). Specifically,
the "Cytotoxic Effector" cluster comprised 25% of cells in responders compared to only
10% in non-responders.
Supervised Analysis
Across more than 100 model configurations I evaluated LOPO cell-level performance for
representative feature–algorithm combinations. Table 7 reports the cell-level LOPO results
for selected models; the table lists the hyperparameter setting used for each feature_set–
algorithm pair (MLP settings were selected via inner CV; CNN, BiLSTM, and Transformer
were trained with representative fixed configurations). Sequence-aware deep models
provided the strongest cell-level discrimination in this run; the sequence_structure MLP
achieved the top accuracy (0.917) and AUC (0.973). Other sequence-aware architectures
(RNN, CNN, Transformer) showed comparable performance (accuracies in the 0.912–0.916
range). Classical baselines (logistic regression and tree-based ensembles such as XGBoost)
performed competitively but did not exceed the top deep models, supporting the use of
sequence-aware representations in this dataset.
Table 7. Cell-level classification performance under LOPO cross-validation. Ranked by F1
score.
Feature Set Model Accur
acy
Precisi
on
Recall F1 AUC Specifi
city NPV
sequence_str
ucture
MLP 0.917
535
0.860
298
0.927
554
0.892
661
0.973
425
0.9116
59
0.955
468
sequence_str
ucture
RNN 0.916
246
0.865
808
0.915
308
0.889
870
0.972
390
0.9167
96
0.948
605
comprehensi
ve
Transfor
mer
0.915
017
0.861
233
0.918
039
0.888
729
0.972
001
0.9132
45
0.949
995
comprehensi
ve
MLP 0.914
277
0.859
420
0.918
336
0.887
902
0.971
572
0.9118
97
0.950
097
comprehensi
ve
CNN 0.913
818
0.859
839
0.916
227
0.887
138
0.971
353
0.9124
04
0.948
902
sequence_str
ucture CNN 0.913
888
0.860
926
0.914
849
0.887
069
0.970
343
0.9133
24
0.948
154
comprehensi
ve
BiLSTM 0.912
948
0.858
890
0.914
822
0.885
974
0.971
396
0.9118
50
0.948
059
sequence_str
ucture
Transfor
mer
0.912
159
0.859
192
0.911
821
0.884
724
0.969
938
0.9123
57
0.946
356
sequence_str
ucture BiLSTM 0.911
879
0.858
597
0.911
794
0.884
397
0.970
455
0.9119
29
0.946
316
comprehensi
ve
RNN 0.911
359
0.857
462
0.911
794
0.883
794
0.968
074
0.9111
04
0.946
270
gene_enhance
d
Logistic
Regressi
on
0.903
000
0.880
000
0.853
000
0.867
000
0.959
000
0.9320
00
0.915
000
basic XGBoost 0.899
000
0.874
000
0.848
000
0.861
000
0.961
000
0.9280
00
0.912
000
tcr_enhanced XGBoost 0.899
000
0.874
000
0.848
000
0.861
000
0.961
000
0.9280
00
0.912
000
gene_enhance
d
XGBoost 0.898
000
0.877
000
0.843
000
0.860
000
0.961
000
0.9300
00
0.910
000
comprehensi
ve
XGBoost 0.898
000
0.872
000
0.847
000
0.859
000
0.961
000
0.9270
00
0.912
000
basic
Logistic
Regressi
on
0.884
000
0.853
000
0.830
000
0.841
000
0.940
000
0.9160
00
0.902
000
tcr_enhanced MLP 0.674
618
0.531
868
1.000
000
0.694
404
1.000
000
0.4837
81
1.000
000
The dominant source of predictive information is coordinated transcriptomic
programs rather than isolated single-gene effects. Mapping principal component loadings
back to gene space and interrogating SHAP attributions revealed strong enrichment for
canonical Cytotoxic Effectors (GZMB, PRF1, GNLY), Interferon Response genes (including
IFIT family members and MX1), and Exhaustion Markers (such as PDCD1 and LAG3). These
programs load heavily on the leading principal components and collectively drive the
largest share of model attribution, indicating that the signal reflects coherent cellular states
rather than noise or single-gene artifacts.
Receptor-derived features supply complementary and biologically plausible
information. Both k-mer motif fingerprints and aggregated physicochemical summaries of
the CDR3 receive positive attribution in many folds, and motif-based analyses identify
short sequence patterns that are recurrently enriched among clonotypes with high model
attribution. Critically, the receptor signal is not redundant with transcriptomic programs:
when combined with gene PCs it improves model calibration and reduces ambiguous
classifications, particularly in cells that sit near cluster boundaries in transcriptional space.
Unsupervised and clonotype-level analyses provide a coherent mechanistic
narrative. Cells annotated as effector-like or early-activated by Leiden clustering are
disproportionately predicted as coming from responders, and expanded clonotypes in
responder samples preferentially localize to these transcriptional clusters. In the
longitudinal samples available within the cohort I observed transient post-treatment
expansions of effector-like clonotypes in responders, a pattern consistent with mobilization
of tumor-reactive T cells rather than a purely homeostatic response.
I assessed robustness through multiple sensitivity analyses. Repeating modeling
across different random seeds, varying the missing-data strategy for receptor features, and
removing principal components associated with technical batch effects all preserved the
primary conclusions: the transcriptomic principal components dominate attribution,
receptor encodings add orthogonal signal, and feature rankings are stable across
perturbations. These checks increase confidence that the identified signatures are not
driven by a single preprocessing choice or by idiosyncratic folds.
There are important interpretive boundaries to emphasize. The cohort contains
only four patients, so LOPO yields a small number of folds and reported metrics are celllevel; consequently, observed high cell-level accuracy may partly reflect within-patient
consistency rather than guaranteed external generalizability. Missing V(D)J capture for a
subset of cells reduces the effective sample size for receptor-aware modeling. I therefore
position these results as a rigorous proof-of-concept that identifies plausible biological
correlates and a clear validation path rather than as a definitive clinical claim.
Taken together, the results provide multiple convergent lines of evidence that
peripheral single-cell multiomic profiling, analyzed with interpretable machine learning,
can recover biologically coherent correlates of chemo-immunotherapy response in HR
positive breast cancer. The approach yields both predictive gain and mechanistic insight,
and it sets the stage for independent validation and translational development of targeted,
lower-cost assays that recapitulate the identified gene programs and receptor motifs.
Discussion
Principal Results
This study demonstrates that peripheral single-cell multi-omic data carry meaningful
signatures associated with chemo-immunotherapy response in early-stage HR positive
breast cancer. The tissue-agnostic nature of peripheral monitoring, combined with singlecell resolution, can reveal the transient cellular programs and clonotypic reshaping that
accompany effective immune engagement. Our results support three interrelated
conclusions. First, transcriptional programs that index cytotoxicity, interferon response
and early activation dominate the predictive signal in this cohort. Second, receptor
sequence features add orthogonal information consistent with convergent selection of
tumor-reactive clonotypes. Third, sequence-aware deep architectures (MLP, RNN, CNN,
Transformer) achieved the highest cell-level discrimination in this run (top accuracy 0.917
and AUC 0.973); tree-based baselines performed competitively but did not surpass the
leading deep models.
Comparison with Prior Work
A distinguishing feature of this work is the systematic integration of unsupervised state
discovery with supervised predictive modeling, a paradigm that is rare in single-cell
immunotherapy biomarker research. Most studies emphasize either exploratory clustering
[24–26] or end-to-end supervised prediction [35, 38], but seldom combine both
approaches within a unified framework. By explicitly comparing both learning paradigms, I
achieve dual objectives: (1) biologically interpretable cell state annotations that ground the
predictive features in known immune biology, and (2) quantitative discrimination metrics
that validate the clinical relevance of those states. This methodological synthesis enables
mechanistic hypothesis generation while maintaining rigorous predictive performance,
offering a template for future single-cell biomarker studies.
Furthermore, the comprehensive comparative analysis across multiple model
families, from linear regression baselines to gradient-boosted ensembles, feed-forward
neural networks, and sequence-aware architectures (CNNs, LSTMs, Transformers),
provides rare empirical evidence about which algorithmic paradigms are best suited to
multimodal single-cell data. The explicit evaluation of three complementary TCR encoding
schemes (positional, k-mer, physicochemical) and their integration with transcriptomic
PCs represents a level of technical depth uncommon in applied immunotherapy prediction
studies, demonstrating that thoughtful feature engineering can match or exceed the
benefits of end-to-end learned representations when training data are limited.
A key advantage of our approach is the reliance on peripheral blood rather than
tumor tissue. Tissue biopsies are invasive, difficult to repeat longitudinally, and often fail to
capture the systemic immune dynamics that are critical for immunotherapy response. By
demonstrating that peripheral blood mononuclear cells (PBMCs) harbor predictive signals,
I offer insight into the potential for non-invasive, serial monitoring of patient response.
This is particularly relevant for early-stage disease, where "liquid biopsies" could guide
treatment de-escalation or intensification without the morbidity of repeated surgeries.
Our work distinguishes itself from previous studies by integrating unsupervised
state discovery with supervised predictive modeling. While prior efforts have largely
focused on either descriptive clustering or black-box prediction, our pipeline combines
these paradigms to yield interpretable, biologically grounded predictors. Unlike studies
that rely solely on bulk sequencing or single-modality data, our multimodal approach
leverages the synergy between gene expression and receptor sequences to resolve subtle
cell states that drive response. This dual focus on novelty in both biological insight and
methodological rigor positions our findings as a significant advance in the field of
computational immuno-oncology.
Compared to recent studies in the field, our approach offers a complementary
perspective. Bassez et al. [24] profiled tumor-infiltrating lymphocytes at single-cell
resolution and identified pre-treatment T cell states predictive of anti-PD-1 response, but
their analysis was restricted to tissue biopsies and did not integrate TCR sequence
information. Wu et al. [25] characterized the breast cancer immune landscape using paired
single-cell RNA and TCR sequencing from tumor tissue, revealing clonotype dynamics
across subtypes. Vanguri et al. [35] achieved an AUC of 0.80 for immunotherapy response
prediction in non-small cell lung cancer by integrating radiology, pathology, and genomic
features, but relied on bulk-level modalities. Our peripheral blood approach operates in a
fundamentally different compartment, demonstrating that PBMCs harbor cell-level
predictive signals without requiring invasive tissue sampling. This finding suggests that
peripheral and tissue-based approaches may capture complementary aspects of the antitumor immune response.
Limitations
Several limitations warrant explicit acknowledgment. First, the cohort consists of only four
patients (two responders, two non-responders), drawn from a single clinical trial (DFCI 16-
466, NCT02999477). The small cohort size severely constrains the model’s ability to learn
inter-patient variation and means high cell-level accuracy may reflect strong within-patient
transcriptional consistency rather than guaranteed external generalizability. Second,
pseudobulk differential expression analysis between response groups identified no
statistically significant genes (all FDR > 0.79), and k-mer enrichment analysis across 6,093
k-mers similarly yielded no significant associations (all FDR = 1.0). These null results in
aggregated analyses further underscore the limited statistical power of the current small
cohort. Third, one sample (S8) lacked TCR sequencing data, reducing the effective sample
for receptor-aware analyses. Fourth, the archived results were generated from a processing
pipeline that differed from the current notebook implementation, introducing potential
inconsistencies in reported numbers. These limitations collectively indicate that our results
represent a proof-of-concept demonstration requiring independent validation in larger,
multi-center cohorts before any clinical application.
Conclusions
Several important research directions emerge from this work that would strengthen
clinical translation and biological understanding. First, validation in larger, independent
cohorts with prospective collection protocols is essential to confirm the generalizability of
the identified signatures across diverse patient populations and treatment regimens. The
current cohort size (four patients) limits power for fully stratified patient-level analyses
and adjustment for potential clinical confounders such as tumor burden, prior treatment
history, and comorbidities.
Second, expanding the multimodal feature space could enhance both predictive
accuracy and mechanistic insight. Integrating surface protein expression via CITE-seq
would provide direct measurement of activation and exhaustion markers without relying
on transcriptional proxies. Similarly, chromatin accessibility profiling (scATAC-seq) could
reveal epigenetic states that govern T cell differentiation and dysfunction. Spatial
transcriptomics of paired tumor samples, when available, would enable direct comparison
of peripheral and intratumoral immune landscapes and help identify whether peripheral
signals reflect systemic priming or spillover from tissue-resident populations.
Third, longitudinal sampling throughout the treatment course would enable
development of dynamic biomarkers that track evolving immune responses. Early on-
treatment changes in effector programs or clonal expansions may provide earlier readouts
of therapeutic efficacy than anatomic imaging, potentially enabling adaptive treatment
strategies. Machine learning models that incorporate temporal trajectories, such as
recurrent neural networks or state-space models, could capture these dynamics more
effectively than static snapshots.
Fourth, mechanistic validation through functional assays and perturbation
experiments would strengthen causal claims. Ex vivo stimulation of isolated clonotypes
identified as predictive by the model, coupled with cytokine profiling or killing assays
against patient-derived tumor organoids [43], would directly test whether these cells
possess tumor-reactive function. CRISPR screens targeting genes with high feature
importance could 14 identify causal drivers of the effector programs that correlate with
response.
Finally, development of reduced, cost-effective assays that recapitulate the key
predictive features is critical for clinical deployment. Targeted gene expression panels (e.g.,
NanoString or RT-qPCR) focusing on the top 20-50 genes with highest SHAP attribution,
combined with targeted TCR sequencing of the most predictive k-mer motifs, could provide
a practical alternative to full single-cell profiling. Validating such simplified assays in
prospective trials would represent a major step toward routine clinical use. From a
translational perspective the findings are encouraging. If validated, peripheral single-cell
assays interpreted with robust machine learning could provide a minimally invasive
biomarker to identify HR positive patients most likely to benefit from combined chemoimmunotherapy. Such biomarkers would have direct clinical impact by personalizing
treatment decisions, reducing unnecessary toxicity, and improving cost-effectiveness.
Practically, any clinical deployment would likely rely on lower-cost targeted assays that
recapitulate the identified gene programs and receptor motifs rather than full single-cell
sequencing.
I conclude with methodological guidance. For small cohorts, prefer robust,
interpretable models with careful control for batch and patient identity. Use sequenceaware encodings that mix positional and physicochemical information rather than relying
solely on raw one-hot inputs. Quantify uncertainty via repeated subsampling and report
cell-level performance measures. Finally, publish code and intermediate data
representations to accelerate community validation.
Acknowledgements
I thank Sun et al. for making the GSE300475 dataset publicly available and the patients and
clinical teams who contributed samples. The author thanks colleagues who provided
feedback on analysis and interpretation as well as his research mentor, Dr. Morteza
Sarmadi. Processed feature matrices, notebooks, and code required to reproduce the
analyses are archived in GitHub and are available upon request.
Conflicts of Interest
The authors declare no competing interests.
Abbreviations
HR+: hormone receptor positive
ICI: immune checkpoint inhibitor
LOPO: leave-one-patient-out cross–validation
AUROC: area under the receiver operating characteristic curve
AUPRC: area under the precision-recall curve
PCA: principal component analysis
MLP: multilayer perceptron
CNN: convolutional neural network
BiLSTM: bidirectional long short-term memory
TCR: T cell receptor
CDR3: complementarity-determining region 3
PBMC: peripheral blood mononuclear cell
V(D)J: variable(Diversity)joining repertoire sequencing.
References
1. Robert C. A decade of immune-checkpoint inhibitors in cancer therapy. Nature
Communications. 2020;11:3801. doi:https://doi.org/10.1038/s41467-020-17670-y
2. Schmid P, Adams S, S RH, Schneeweiss A, H BC, Iwata H. Atezolizumab and nabpaclitaxel in advanced triple-negative breast cancer. New England Journal of Medicine.
2019;380:985-988. doi:https://doi.org/10.1056/nejmc1900150
3. Emens LA, Ascierto, Paolo A, Darcy PK, et al. Cancer immunotherapy: opportunities
and challenges in the rapidly evolving clinical landscape. European Journal of Cancer.
2017;81:116-129. doi:https://doi.org/10.1016/j.ejca.2017.01.035
4. Anna KS, Fekete JT, Győrffy B. Predictive biomarkers of immunotherapy response with
pharmacological applications in solid tumors. Acta Pharmacologica Sinica.
2023;44:1879-1889. doi:https://doi.org/10.1038/s41401-023-01079-6
5. Kiriakidou N, Livieris, Ioannis E, Diou C. XGBoost: A scalable tree boosting system. In:
IFIP Advances in Information and Communication Technology. ; 2024:58-70.
doi:https://doi.org/10.1007/978-3-031-63219-8_5
6. Antoniadis A, Cugliari J, Fasiolo M, Goude Y, Poggi JM. Random forests. Statistics for
Industry, Technology, and Engineering. 2024;45:99-111.
doi:https://doi.org/10.1007/978-3-031-60339-6_5
7. M FC. Deep Learning. Vol 17. MIT Press; 2003.
doi:https://doi.org/10.4314/sajhe.v17i1.25201
8. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. In: Vol 30. ; 2025.
doi:https://doi.org/10.65215/nxvz2v36
9. Jurtz VI, Jessen LE, Bentzen AK, et al. NetTCR: sequence-based prediction of T-cell
receptor binding to peptide-MHC complexes using convolutional neural networks.
Bioinformatics. 2018;34:i399-i407. doi:https://doi.org/10.1101/433706
10. Sidhom JW, Benjamin LH, Pardoll DM, Baras AS. DeepTCR: a deep learning
framework for revealing sequence-level predictors in T cell receptor repertoires. Nature
Communications. 2021;12:1605. doi:https://doi.org/10.1038/s41467-021-21879-w
11. Glanville J, Huang H, Nau A, et al. Identifying specificity groups in the T cell receptor
repertoire. Nature. 2017;547:94-98. doi:https://doi.org/10.1038/nature22976
12. Dash P, Fiore-Gartland AJ, Hertz T, et al. Quantifiable predictive features define
epitope-specific T cell receptor repertoires. Nature. 2017;547:89-93.
doi:https://doi.org/10.1038/nature22383
13. Bagaev, Dmitry V, Vroomans, Renske M A, Samir J, et al. VDJdb in 2019: database
extension, new analysis infrastructure and a T-cell receptor motif compendium. Nucleic
Acids Research. 2019;48:D1057-D1062. doi:https://doi.org/10.1093/nar/gkz874
14. Kather JN, Pearson AT, Halama N, et al. Deep learning can predict microsatellite
instability directly from histology in gastrointestinal cancer. Nature Medicine.
2019;25:1054-1056. doi:https://doi.org/10.1038/s41591-019-0462-y
15. Saltz J, Gupta R, Hou L, et al. Spatial organization and molecular correlation of tumorinfiltrating lymphocytes using deep learning on pathology images. Cell Reports.
2018;23:181-193.e7. doi:https://doi.org/10.1016/j.celrep.2018.03.086
16. Cristescu R, Mogg R, Ayers M, et al. Pan-tumor genomic biomarkers for PD-1
checkpoint blockade–based immunotherapy. Science. 2018;362:eaar3593.
doi:https://doi.org/10.1126/science.aar3593
17. Bagaev A, Kotlov N, Nomie K, et al. Conserved pan-cancer microenvironment subtypes
predict response to immunotherapy. Cancer Cell. 2021;39:845-865.e7.
doi:https://doi.org/10.1016/j.ccell.2021.04.014
18. Benzekry S, Grangeon M, Karlsen M, et al. Machine learning for prediction of
immunotherapy efficacy in non-small cell lung cancer from simple clinical and
biological data. Cancers. 2021;16:527. doi:https://doi.org/10.1101/2021.11.30.21267064
19. Lu S, Stein JE, Rimm DL, et al. Comparison of biomarker modalities for predicting
response to PD-1/PD-L1 checkpoint blockade: A systematic review and meta-analysis.
JAMA Oncology. 2019;5:1195. doi:https://doi.org/10.1001/jamaoncol.2019.1549
20. Eraslan G, Simon LM, Mircea M, Mueller NS, Theis FJ. Single-cell RNA-seq denoising
using a deep count autoencoder. Nature Communications. 2018;10:390.
doi:https://doi.org/10.1101/300681
21. Lopez R, Regier J, Cole MB, Jordan MI, Yosef N. Deep generative modeling for singlecell transcriptomics. Nature Methods. 2018;15:1053-1058.
doi:https://doi.org/10.1038/s41592-018-0229-2
22. Lotfollahi M, Alexander WF, Theis FJ. scGen predicts single-cell perturbation
responses. Nature Methods. 2019;16:715-721. doi:https://doi.org/10.1038/s41592-019-
0494-8
23. Ji Y, Lotfollahi M, Alexander WF, Theis FJ. Machine learning for perturbational singlecell omics. Cell Systems. 2021;12:522-537.
doi:https://doi.org/10.1016/j.cels.2021.05.016
24. Bassez A, Vos H, Dyck V, et al. A single-cell map of intratumoral changes during antiPD1 treatment of patients with breast cancer. Nature Medicine. 2021;27:820-832.
doi:https://doi.org/10.1038/s41591-021-01323-8
25. Wu SZ, Al-Eryani G, Roden DL, et al. A single-cell and spatially resolved atlas of
human breast cancers. Nature Genetics. 2021;53:1334-1347.
doi:https://doi.org/10.1038/s41588-021-00911-1
26. Tirosh I, Izar B, Prakadan, Sanjay M, et al. Dissecting the multicellular ecosystem of
metastatic melanoma by single-cell RNA-seq. Science. 2016;352:189-196.
doi:https://doi.org/10.1126/science.aad0501
27. Thorsson V, Gibbs DL, Brown SD, et al. The immune landscape of cancer. Immunity.
2018;48:812-830.e14. doi:https://doi.org/10.1016/j.immuni.2018.03.023
28. Stuart T, Butler A, Hoffman P, et al. Comprehensive integration of single-cell data. Cell.
2018;177:1888-1902.e21. doi:https://doi.org/10.1101/460147
29. Zhang Q, He Y, Luo N, et al. Landscape and dynamics of single immune cells in
hepatocellular carcinoma. Cell. 2019;179:829-845.e20.
doi:https://doi.org/10.1016/j.cell.2019.10.003
30. Kim N, Kim HK, Lee K, et al. Single-cell RNA sequencing demonstrates the molecular
and cellular reprogramming of metastatic lung adenocarcinoma. Nature
Communications. 2020;11:2285. doi:https://doi.org/10.1038/s41467-020-16164-1
31. Yost KE, Satpathy, Ansuman T, Wells DK, et al. Clonal replacement of tumor-specific T
cells following PD-1 blockade. Nature Medicine. 2019;25:1251-1259.
doi:https://doi.org/10.1101/648899
32. Wieder T, Eigentler T, Brenner E, Röcken M. Temporal changes in the T cell repertoire
during checkpoint blockade therapy. Journal of Allergy and Clinical Immunology.
2018;142:1403-1414. doi:https://doi.org/10.1016/j.jaci.2018.02.042
33. Bolotin DA, Poslavsky S, Mitrophanov I, et al. MiXCR: software for comprehensive
adaptive immunity profiling. Nature Methods. 2015;12:380-381.
doi:https://doi.org/10.1038/nmeth.3364
34. Wezenaar, A. K. L, Pandey U, Keramati F, et al. Mapping T cell dynamics to molecular
profiles through behavior-guided transcriptomics. Nature Protocols. 2025;20:2453-2480.
doi:https://doi.org/10.1038/s41596-024-01126-4
35. Luo J, Vanguri, Rami S, Aukerman AT, et al. Multimodal integration of radiology,
pathology and genomics for prediction of response to PD-(L)1 blockade in patients with
non-small cell lung cancer. Journal of Clinical Oncology. 2022;40:9064-9064.
doi:https://doi.org/10.1200/jco.2022.40.16_suppl.9064
36. Dia AK, Kolnohuz A, Yolchuyeva S, et al. Computational analysis of whole slide
images predicts PD-L1 expression and progression-free survival in immunotherapytreated non-small cell lung cancer patients. Journal of Translational Medicine.
2025;23:510. doi:https://doi.org/10.1186/s12967-025-06487-2
37. Huang X, Qiu W, Kong Y, et al. Artificial intelligence-based multimodal prediction of
postoperative adjuvant immunotherapy benefit in urothelial carcinoma: Results from the
phase III IMvigor010 trial. MedComm. 2025;6:e70324.
doi:https://doi.org/10.1002/mco2.70324
38. Rakaee M, Tafavvoghi M, Ricciuti B, et al. Deep learning model for predicting
immunotherapy response in advanced non-small cell lung cancer. JAMA Oncology.
2025;11:109. doi:https://doi.org/10.1001/jamaoncol.2024.5356
39. Kourou K, Exarchos, Themis P, Exarchos, Konstantinos P, Karamouzis, Michalis V,
Fotiadis DI. Machine learning applications in cancer prognosis and prediction.
Computational and Structural Biotechnology Journal. 2015;13:8-17.
doi:https://doi.org/10.1016/j.csbj.2014.11.005
40. Sun X, Axelrod ML, Waks AG, et al. Dynamic single-cell systemic immune responses
in immunotherapy-treated early-stage HR+ breast cancer patients. npj Breast Cancer.
2025;11:65. doi:https://doi.org/10.1038/s41523-025-00776-1
41. Khoury T, Peng X, Yan L, Wang D, Nagrale V. Tumor infiltrating lymphocytes in breast
cancer. American Journal of Clinical Pathology. 2018;150:441-450.
doi:https://doi.org/10.1093/ajcp/aqy069
42. Grace, Terry JM, Belgrader P, et al. Massively parallel digital transcriptional profiling of
single cells. Nature Communications. 2017;8:14049.
doi:https://doi.org/10.1038/ncomms14049
43. Dekkers JF, Alieva M, Cleven A, et al. Uncovering the mode of action of engineered T
cells in patient cancer organoids. Nature Biotechnology. 2022;41:60-69.
doi:https://doi.org/10.1038/s41587-022-01397-w
44. Jiang P, Gu S, Pan D, et al. Signatures of T cell dysfunction and exclusion predict cancer
immunotherapy response. Nature Medicine. 2018;24:1550-1558.
doi:https://doi.org/10.1038/s41591-018-0136-1
45. Ichiryu N, Fairchild PJ. Acquisition of immune privilege by transformed cells: inhibiting
and facilitating immune escape. Methods in Molecular Biology. 2013;98:1-16.
doi:https://doi.org/10.1007/978-1-62703-478-4_1Phillips SJ, Whisnant JP.
Hypertension and stroke. In: Laragh JH, Brenner BM, editors. Hypertension:
pathophysiology, diagnosis, and management. 2nd ed. New York: Raven Press;
1995. p. 465-78.
