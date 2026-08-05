# Response to the Editor and Reviewers

Manuscript #93768

We thank the editor and reviewers for the detailed methodological comments. The revised manuscript is framed as an exploratory model-development and hypothesis-generation study, not as a clinically validated prediction model. The new result archive was checked at the level of predictions rather than treating its 1.3-GB contents as independent observations. The archived cell predictions were identical to the validated predictions used for the reported point estimates. Because the archive also contained a patient-aggregated export of some cell-level confidence intervals, those intervals were recomputed from the archived cell predictions by resampling complete patient clusters. All reported intervals now respect the patient-level outcome structure.

## Editorial and general comments

1. **Detailed statistical analysis.** Addressed. The Methods now specify LOPO-CV, inner patient-grouped tuning, cell-, sample-, and patient-level aggregation, exact balanced-label enumeration, retrained label permutations, patient-clustered bootstrap confidence intervals, calibration measures, and simulated perturbation tests. Tables 3-5 report the corresponding results.

2. **Potential overfitting and loss plots.** Addressed. Training and validation loss histories are exported and shown in Figures 3-7. The text states that the short curves cannot establish absence of overfitting with only 4 independent patients; dropout, L2 regularization, early stopping, grouped validation, seed stability, permutation controls, and out-of-distribution perturbations are reported as safeguards rather than proof of generalizability.

3. **Missing SHAP plot and unclear figure/table marking.** Addressed. Figure 8 is the standalone attribution summary, and Table 6 gives PCA-to-gene and pathway attribution results. Figure and table callouts were checked throughout the manuscript. Captions and labels are outside the image files, as required for JMIR submission.

4. **Tuning only the MLP.** Addressed with an important qualification. The revised pipeline applies the same inner patient-grouped tuning structure to MLP, CNN, BiLSTM, Transformer, and XGBoost. The supplied completion archive used 3 trials per architecture. Audit identified an inappropriately narrow XGBoost learning-rate range and duplicated XGBoost loss traces; the corrected executable now uses a tree-appropriate 0.01-0.20 range, separate training and validation evaluation sets, and a larger default trial budget. The archived XGBoost result is therefore labeled as a failed comparator and is not used to rank architectures.

5. **Mixed first- and third-person voice.** Addressed. The manuscript now uses third-person or passive scientific voice consistently.

6. **Change “Patient Recruitment” to “Datasets.”** Addressed. The section is now titled “Data Source and Study Cohort” and explicitly states that no participants were recruited for this secondary analysis.

7. **Multi-omics versus multimodal terminology.** Addressed. The manuscript now describes linked transcriptome and paired receptor measurements as single-cell multimodal profiling. It explains that this does not claim separate biological omes and explicitly describes the RNA-only recurrence specimen S8.

8. **Justification for the top 50 PCs.** Addressed. Cumulative variance was computed separately in each training fold. The first 50 PCs explained 30.36%-31.30% of training-fold variance, below the proposed 80% target. The manuscript now calls 50 PCs a fixed regularization budget rather than a statistically complete representation and documents a threshold-based option in the code.

9. **Author list, affiliations, and addresses.** Addressed. The revised document includes the author, institutional affiliation, postal address, and corresponding-author information on the first page.

## Reviewer BF

### Major comments

1. **Title overstates validation.** Addressed. The title now reads “Exploratory Model Development Study” and no longer claims a validation study. The Abstract, Discussion, and Conclusion state that the result is hypothesis generating and not ready for clinical use.

2. **Effective sample size is four patients.** Addressed throughout. The Abstract, Introduction, Methods, Results, Limitations, and Conclusion identify 4 patients as the independent response units and distinguish this from 39,532 baseline cells or 100,067 longitudinal cells.

3. **Cell-level metrics may be misleading.** Addressed. Cell probabilities are aggregated within each held-out patient. Table 4 reports the four held-out patient probabilities for every model, and Table 3 separates descriptive cell-level performance from patient-level performance. The text explicitly states that cell metrics are not independent-outcome evidence.

4. **Report each LOPO fold.** Addressed. Table 4 is the fold-level patient report: each column is one held-out patient. Fold-level AUROC is not estimable when a fold contains one outcome unit, so fold probabilities and classifications are reported instead; pooled patient metrics summarize the four folds. Fold-level metric exports remain available in the result artifacts.

5. **Clustered bootstrap.** Addressed and corrected. The final intervals resample patient IDs, retain every cell from a selected patient for cell-level scoring, and retain all samples from a selected patient for longitudinal scoring. The archive’s patient-aggregated cell-interval export was not used; intervals were recomputed from archived cell predictions. The manuscript also explains why intervals can be discrete or degenerate with 4 clusters.

6. **Response prediction versus treatment-associated immune states.** Addressed. Baseline-only analysis is prespecified as primary. On-treatment and recurrence-inclusive analyses are explicitly labeled longitudinal immune-state sensitivity analyses and are not interpreted as pretreatment prediction.

7. **Recurrence leakage.** Addressed. Recurrence-excluded and all-timepoint analyses are reported in Table 5. The manuscript states that high longitudinal performance cannot establish prospective response prediction because treatment and recurrence may change immune states.

8. **Timepoint distribution and sequencing completeness.** Addressed. New Table 2 lists all 11 samples, patient outcome, baseline/post-treatment/recurrence timepoint, cell count, productive TRA/TRB proportions, and sequencing modality. S8 is explicitly identified as RNA-only with no V(D)J data.

9. **Preprocessing leakage.** Addressed. HVG selection, scaling, PCA loadings, k-mer vocabulary construction, missingness handling, and all other supervised transforms are fit only on outer-training patients. The held-out patient is transformed with fixed training objects.

10. **UMAP as a predictive feature.** Addressed. UMAP coordinates, Leiden labels, and full-dataset embeddings were removed from supervised predictors. UMAP is discussed only as visualization and a potential leakage source.

11. **TCR missingness confounding.** Addressed. Chain-specific missingness indicators are retained, absent sequence features are zero-filled after training-fold transformation, and S8 is retained as RNA-only in longitudinal sensitivity analysis. Missingness rates are summarized by patient, response group, and timepoint as descriptive diagnostics; cell-level tests are not treated as independent patient-level inference.

12. **Sample-level repertoire features may leak identity.** Addressed. Sample-level repertoire summaries are descriptive only and are not assigned to cells as predictive features. The primary feature matrix uses cell-linked expression and receptor representations; transcriptome-only, TCR-only, and receptor-ablation analyses are reported as controls.

13. **Nested tuning is underpowered.** Addressed. The Methods state that inner tuning has only 3 training patients per outer fold and that the resulting hyperparameter choices are unstable. The larger default trial budget in the corrected executable is presented as a reproducibility option, not evidence of optimal architecture selection.

14. **Deep learning not justified by four outcomes.** Addressed by reframing and controls. Deep models are retained as exploratory representation learners, not as clinically justified predictors. The manuscript reports regularization, early stopping, grouped tuning, seed stability, permutation tests, perturbation tests, and the need for larger external cohorts; it does not claim neural superiority.

15. **Permutation or negative controls.** Addressed. Six balanced patient-label assignments were enumerated exactly; five balanced label-shuffle Transformer retrains produced an empirical P value of .33; gene-PC and TCR ablations, feature masking, Gaussian-noise stress tests, and seed stability are also reported.

16. **AUC of 1.0 with poor threshold metrics.** Addressed. The revised manuscript reports discrimination and threshold-dependent metrics separately, provides Brier score and calibration information, and explains that one-patient folds cannot estimate AUROC. The prior anomalous feature-set result is not used as evidence of clinical performance.

17. **Separate biological discovery from clinical prediction.** Addressed. The Discussion describes cytotoxic, interferon, memory, and mitochondrial results as model-attribution hypotheses. The Conclusion explicitly states that the work is an exploratory proof of concept and not ready for treatment selection.

18. **TCR encoding and calibration claims.** Addressed. The revised analysis reports Brier score, log loss, expected calibration error, maximum calibration error, and calibration-in-the-large for cell and patient units. The text no longer claims that TCR features improve calibration; the transcriptome-only analysis performed at least as well in this cohort.

19. **Patient-level confusion matrices or prediction summaries.** Addressed. Table 4 and Figure 2 show the mean held-out patient probabilities and the 0.50 classification threshold. Patient-level confusion-matrix and calibration artifacts are included with the results.

20. **Endpoint and timing.** Addressed. The endpoint is residual cancer burden at surgery, with RCB 0/I as responder and RCB II/III as nonresponder. Baseline predictors precede treatment; post-treatment and recurrence specimens are restricted to sensitivity analyses and are not described as prospective prediction.

21. **Secondary analysis of public data.** Addressed. “Data Source and Study Cohort” states that this is a retrospective secondary analysis of deidentified GSE300475 data and that no recruitment occurred.

22. **Repository access.** Addressed in the reproducibility materials. The corrected pipeline, metadata, and manuscript sources are maintained at https://github.com/Anmol-Josan/Predicting-Chemo-immunotherapy-Response-in-Early-Stage-Hormone-Receptor-Positive-Breast-Cancer. The manuscript reports package versions, random seed, output artifacts, and regeneration behavior.

23. **Move or strengthen limitations.** Addressed. The Abstract and Conclusion state the 4-patient limitation and lack of clinical validation before presenting the high descriptive metrics. The Discussion returns repeatedly to the effective sample size and null controls.

24. **Avoid clinical-deployment implications.** Addressed. Claims about treatment selection, routine use, de-escalation, and intensification were removed or moved to future directions and are explicitly conditional on prospective external validation.

25. **Reference quality and formatting.** Addressed. The reference list was audited, classic sources were retained for neural networks, recurrent networks, Transformers, XGBoost, calibration, bootstrap, PCA, and single-cell analysis, and the previously mismatched references were corrected. References are cited numerically in the manuscript rather than by internal file or code names.

### Minor comments

1. **Abstract should identify a public-data secondary analysis.** Addressed in the Abstract Methods sentence.

2. **Abstract voice.** Addressed; the Abstract uses neutral third-person/passive language.

3. **Pair high metrics with four-patient caution.** Addressed in the Abstract Results and Conclusions.

4. **Calibration in Abstract.** Addressed by reporting calibration in the Results and avoiding an unsupported claim of improved calibration in the Abstract.

5. **Awkward “indispensable assistance” wording.** Addressed. The Introduction now says that machine learning provides a useful framework.

6. **HR-positive-specific references.** Addressed. The Introduction cites HR-positive neoadjuvant trials and the source HR-positive breast cancer cohort directly.

7. **Early “Key findings” section.** Addressed. The revised manuscript does not preview a separate key-findings block before Methods and Results.

8. **Methods heading.** Addressed. The heading is “Data Source and Study Cohort.”

9. **IRB/consent for public secondary data.** Addressed in Ethics Considerations: no new contact or intervention occurred, and original-study oversight is attributed to the source publication.

10. **Mitochondrial-content threshold.** Addressed in the revised quality-control description and terminology audit.

11. **Cells before versus after QC.** Addressed. The manuscript identifies 100,067 as the source analysis cohort after the source study’s cell calling/QC and distinguishes the 39,532-cell baseline subset used for the primary analysis.

12. **Feature-dimension inconsistency.** Addressed. The revised Methods describe feature families by construction and distinguish cell-intrinsic features from excluded sample-level repertoire summaries; the combined representation is defined by the executable rather than inconsistent approximate ranges.

13. **Six versus eight physicochemical features.** Addressed. The feature description and executable use one consistent chain-specific physicochemical summary definition.

14. **Missing TCR handling.** Addressed. Missingness indicators are explicit, absent sequence-derived values are zero-filled after training-only transformation, and S8 is RNA-only with no invented receptor sequence.

15. **TRA and TRB availability.** Addressed. The Methods and Table 2 distinguish alpha and beta productive-chain coverage and explain that both chain representations are used when available.

16. **“Idempotent modification” wording.** Addressed. The overly technical phrasing was removed and replaced by a direct description of training-fold feature construction.

17. **Fold-specific PCA.** Addressed. PCA loadings and cumulative variance are fitted and reported separately within outer training folds.

18. **UMAP leakage.** Addressed. UMAP is excluded from supervised features and described only as a visualization method.

19. **Cell-type composition.** Addressed as a limitation and interpretation safeguard. The manuscript states that cell composition may remain confounded with patient outcome and recommends within-cell-type pseudobulk or hierarchical analyses for future cohorts.

20. **Software versions.** Addressed. Python, NumPy, pandas, scikit-learn, XGBoost, TensorFlow, SHAP, and Optuna versions are listed in Software and Reproducibility.

21. **Table 1 readability.** Addressed. Table 1 was rebuilt in portrait orientation with explicit widths, repeated headers, rounded percentages, and no soft line breaks.

22. **V(D)J terminology.** Addressed. The manuscript consistently uses “variable, diversity, and joining” or “V(D)J.”

23. **Hydrophobicity range typo.** Addressed in the feature-definition audit.

24. **Level of summary statistics.** Addressed. Table 1 and Table 2 identify whether values are baseline patient-level summaries or sample-level descriptive coverage; inferential claims are not based on cell independence.

25. **K-mer terminology.** Addressed. The manuscript uses “position-independent motif representation.”

26. **Figure 1 resolution.** Addressed. The PCA figure is a standalone high-resolution PNG with clean margins and no embedded caption.

27. **Figure 1 variance visibility.** Addressed. The y-axis reports cumulative explained variance and the caption identifies the 80% reference line; fold-specific values are reported in Results.

28. **Figure 2 readability.** Addressed. The patient prediction summary was exported at high resolution with larger labels and a clearly marked 0.50 threshold.

29. **Dense heatmap/UMAP.** Addressed by removing UMAP from predictive claims and limiting the revised manuscript to the interpretable patient summary, PCA, loss, and attribution figures required for this analysis.

30. **Biological annotations for Leiden clusters.** Addressed by removing cluster labels from the predictive analysis; unsupervised cluster annotations are not used as model features or inferential evidence.

31. **Cytotoxic-effector cluster denominators.** Addressed. The unsupported cell-composition comparison was removed. The revised report gives gene-set attribution enrichment and clearly identifies it as model attribution, not a patient-level cluster proportion.

32. **Patient-clustered enrichment P value.** Addressed. Cell-level Wilcoxon-style claims were removed. Gene-set enrichment P values are empirical random-set comparisons and are explicitly not patient-level association P values.

33. **“Significantly higher” wording.** Addressed. Unsupported significance language was removed or replaced by descriptive wording.

34. **Table 7 readability and decimals.** Addressed. The former model-comparison table was replaced by compact Tables 3-5 with rounded metrics and explicit evaluation units.

35. **Undefined sequence-structure family.** Addressed. The revised feature description defines chain presence, sequence length, amino-acid composition, physicochemical summaries, and position-independent k-mers.

36. **Confidence intervals/fold variability.** Addressed. Table 3 contains patient-clustered intervals, Table 4 provides the four fold-level patient predictions, and the archived fold stability summaries are exported.

37. **Metric aggregation.** Addressed. The Methods explicitly distinguish fold-level, pooled cell-level, patient-level, and sample-level calculations.

38. **Quantitative attribution dominance.** Addressed. The revised Results reports fold-specific gene-expression attribution fractions of 45.9%-86.8% and fold-ranking stability.

39. **PCA attribution interpretation.** Addressed. PC attributions are back-projected through fold-specific gene loadings, with leading genes listed in Table 6.

40. **Pathway results.** Addressed. Table 6 reports cytotoxic-effector, interferon-response, T-cell activation, memory/naive, mitochondrial, and exhaustion/checkpoint gene-set enrichment.

41. **Clonotype expansion claim.** Addressed. The unsupported localization claim was removed. Patient-level descriptive expanded-clonotype cell fractions are retained as exploratory repertoire summaries, without claiming a response-group association from four patients.

42. **“Tissue-agnostic” wording.** Addressed. The Discussion uses “peripheral blood-based monitoring.”

43. **Novel/rare claims.** Addressed. Strong novelty language was removed or qualified with references.

44. **Repetitive prior-work comparison.** Addressed. The comparison section was shortened and focused on the specific biological and methodological context.

45. **“Significant advance.”** Addressed. The manuscript uses “proof of concept,” “exploratory,” and “hypothesis generating.”

46. **Caution in Abstract and Conclusion.** Addressed explicitly.

47. **Separate Conclusion and Future Directions.** Addressed. The manuscript now has separate “Future Directions” and “Conclusions” sections.

48. **Clinical impact and cost-effectiveness.** Addressed. Such claims were removed from the current evidence statement and retained only as conditional future possibilities.

49. **Informal “I conclude” wording.** Addressed. The conclusion uses formal scientific language.

50. **Acknowledgments voice.** Addressed. Acknowledgments and contributions use consistent third-person author wording.

51. **Funding information.** Addressed. The manuscript states that no external funding was received.

52. **Abbreviation formatting.** Addressed. V(D)J is defined consistently in the abbreviation list.

53. **Writing voice.** Addressed through a full third-person/passive voice edit.

54. **Overly emphatic language.** Addressed. “Remarkable success,” “significant advance,” and similar claims were removed or softened.

55. **Encoding and hyphenation artifacts.** Addressed through the Word rebuild and LibreOffice render review; all 23 rendered pages were checked for missing glyphs, clipping, and broken table text.

56. **Reference 45 mismatch.** Addressed. The mismatched citation was removed/corrected during the reference audit.

57. **DOI verification.** Addressed. DOIs and bibliographic details were checked against the cited article identity during the revision.

58. **Classic methods references.** Addressed. Standard references for XGBoost, neural networks, recurrent networks, Transformers, calibration, bootstrap, PCA, and single-cell analysis are included.

59. **Data availability.** Addressed. The public GSE300475 source and the reproducible repository are identified; the executable regenerates processed features from public raw files and exports the metrics and figures.

60. **Overall exploratory framing.** Addressed. The title, Abstract, Results, Discussion, Limitations, and Conclusion consistently frame the study as an exploratory computational reanalysis identifying candidate peripheral immune signals rather than a validated clinical predictor.

## Reviewer EA

The reviewer’s central concern is addressed throughout the revision. The effective sample size is stated as 4 patients; the primary analysis is baseline-only LOPO-CV; patient-level predictions and patient-clustered bootstrap intervals are reported; exact and retrained label-permutation controls are included; and seed stability plus simulated out-of-distribution perturbations are reported. The title and conclusion now describe an exploratory proof of concept, and the manuscript states that the results are not clinically predictive or ready for deployment without a larger prospective cohort and independent external validation.
