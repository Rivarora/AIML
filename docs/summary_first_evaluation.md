# First Evaluation Summary (40% Milestone)

## Problem Statement

This project analyzes student performance and placement outcomes using a two-track machine learning approach:

- Regression track to predict Exam_Score
- Classification track to predict Placement

## Data reliability update

The raw files were rechecked during the data-reliability pass. They do not
share a student identifier: `StudentPerformanceFactors.csv` has no identifier
column, while `College_ID` appears only in the placement file. The main
preprocessing, EDA, splitting, PCA, and training workflows therefore keep the
two tracks independent. Source file sizes and SHA-256 hashes, missing-row
counts, duplicate counts, and bounded-value counts are written to
`data/interim/provenance.json`.

The row-order pairing has been removed entirely. It cannot identify the same
student, and no combined dataset is created or read. Placement results must not
be interpreted as outcomes for the exam records.

## What is Completed

1. Data pipeline organization and dataset lineage
2. Processed dataset and train/validation/test splits
3. EDA output generation
4. Baseline model training and metrics generation
5. Model artifact saving for baseline models
6. Scikit-learn preprocessing pipelines with imputation, scaling, and one-hot encoding fitted on the training split only
7. Five-fold cross-validation for every model
8. Training-versus-test overfitting analysis
9. Placement deterministic-label investigation
10. Placement confusion matrices and classification reports
11. Exam-score residual and error analysis
12. Structural leakage checks and documented limitations
13. Data-quality JSON/CSV reports and a training-only PCA analysis

## Generated Evidence Files

- outputs/eda/eda_summary.txt
- outputs/eda/descriptive_stats.csv
- outputs/eda/exam_correlation_heatmap.png
- outputs/eda/placement_correlation_heatmap.png
- outputs/eda/exam_score_distribution.png
- outputs/eda/placement_distribution.png
- outputs/quality/data_quality.json
- outputs/quality/data_quality.csv
- outputs/pca/pca_summary.json
- outputs/pca/pca_components.csv
- outputs/pca/pca_visualization.png
- outputs/metrics/baseline_metrics.csv
- outputs/metrics/cross_validation.csv
- outputs/metrics/overfitting_analysis.csv
- outputs/metrics/model_card.json
- outputs/analysis/placement_investigation.json
- outputs/analysis/placement_rule_features.csv
- outputs/analysis/placement_confusion_matrices.png
- outputs/analysis/LogisticRegression_confusion_matrix.csv
- outputs/analysis/RandomForestClassifier_confusion_matrix.csv
- outputs/analysis/\*\_classification_report.csv
- outputs/analysis/exam_score_residual_distribution.png
- outputs/analysis/exam_score_error_analysis.csv
- outputs/analysis/exam_score_error_summary.csv
- outputs/analysis/leakage_checks.csv
- outputs/analysis/leakage_validation.txt
- models/LinearRegression_exam_score.joblib
- models/RandomForestRegressor_exam_score.joblib
- models/LogisticRegression_placement.joblib
- models/RandomForestClassifier_placement.joblib

## Baseline Metrics (Current)

### Regression (Exam_Score)

- LinearRegression: test MAE 0.9190, RMSE 2.1978, R² 0.6974; 5-fold CV MAE 0.9294 ± 0.0553, R² 0.6681 ± 0.0589
- RandomForestRegressor: test MAE 1.1522, RMSE 2.3488, R² 0.6544; 5-fold CV MAE 1.1986 ± 0.0565, R² 0.6163 ± 0.0523

### Classification (Placement)

- LogisticRegression: test accuracy 0.8687, F1 0.6888, ROC-AUC 0.9487; 5-fold CV accuracy 0.8514 ± 0.0126, F1 0.6610 ± 0.0260
- RandomForestClassifier (depth-limited): test accuracy 0.9533, F1 0.8768, ROC-AUC 0.9924; 5-fold CV accuracy 0.9471 ± 0.0032, F1 0.8627 ± 0.0070

## Interpretation Note

The placement label is a deterministic function of four source features:
`Placement = Yes` when at least three of `IQ > 110`, `CGPA > 8`,
`Projects_Completed > 1`, and `Communication_Skills > 7` hold. This rule
reproduces all 10,000 labels with zero errors. An unrestricted Random Forest
therefore reaches 1.0; that is a dataset-construction artifact, not evidence of
real-world predictive skill. The deployed forest is capacity-limited
(`max_depth=2`) to avoid presenting a perfect score as model quality.

The placement dataset is imbalanced: 8,341 processed records belong to class 0
and 1,659 records belong to class 1. Therefore, positive-class precision,
recall, F1, and the confusion matrix are more informative than accuracy alone.

LinearRegression is the stronger exam-score baseline. The Random Forest shows
the only overfitting warning: its test R² is 0.1104 below training, and its test
MAE is 0.2968 worse than training. The linear model's gaps are close to zero.

## Teammate 3 Validation Findings

### Leakage checks

- Target columns are absent from the model feature columns.
- Feature values are split before any model-specific encoding.
- Imputation, scaling, and one-hot encoding are fitted inside each training pipeline and refitted inside each cross-validation fold.
- Placement splits contain no repeated feature combinations across train, validation, and test.
- Exam-score splits contain one repeated feature combination between train and validation with conflicting target values. This is a small data-quality warning that should be reviewed before final claims.
- Structural checks cannot prove semantic leakage-free data. The two source files are independent and share no student identifier, so the learned relationships may not represent real student outcomes.
- No combined, row-order, or synthetic-pair dataset exists in this version.

### Interpretation

- LogisticRegression is the more conservative placement baseline: test F1 is 0.6888 and positive-class recall is 0.8755.
- The depth-limited RandomForestClassifier is an honest benchmark rather than a perfect-score claim: test accuracy is 0.9533, F1 0.8768, ROC-AUC 0.9924.
- The deterministic placement rule explains why flexible models can reach 1.0 without any semantic leakage.
- The missing student-level link is the primary limitation of the current project because the performance and placement attributes were not collected as linked records from the same students.

## Reproduction Commands

Run these commands from the repository root:

```powershell
python scripts/preprocess.py
python scripts/dataset_checks.py
python scripts/select_and_split.py
python scripts/data_quality.py
python scripts/pca_analysis.py
python scripts/run_eda.py
python scripts/train_baselines.py
python scripts/analyze_baselines.py
```

The final command generates the confusion matrices, classification reports, residual analysis, and leakage validation files listed above. The data-quality and PCA commands generate the reports and figure used by the dashboard.

## Remaining Tasks (Teammate 3)

1. Prepare the final 5 to 7 slide Evaluation-1 presentation using the generated evidence.
2. Review the conflicting exam-score feature combination against the source data.
3. Add feature-importance or SHAP explanations if time allows.
4. Plan stronger models and a real prediction interface for Phase 2.

## Conclusion

The project has a reproducible baseline pipeline covering data checks, preprocessing, provenance, task-specific splitting, data-quality reporting, PCA, EDA, pipeline-based model training, five-fold cross-validation, overfitting analysis, and post-training analysis. The regression track has a reasonable initial baseline. The placement track is now explained honestly: its label is deterministic, so a perfect score from an unrestricted model is a dataset-construction artifact. The absence of a genuine student-level link between the two source files remains the primary limitation. The project is suitable for the Evaluation-1 milestone with these limitations stated plainly.
