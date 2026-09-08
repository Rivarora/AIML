# First Evaluation Summary (40% Milestone)

## Problem Statement

This project analyzes student performance and placement outcomes using a two-track machine learning approach:

- Regression track to predict Exam_Score
- Classification track to predict Placement

## What is Completed

1. Data pipeline organization and dataset lineage
2. Processed dataset and train/validation/test splits
3. EDA output generation
4. Baseline model training and metrics generation
5. Model artifact saving for baseline models
6. Training-only feature selection to reduce selection leakage
7. Placement confusion matrices and classification reports
8. Exam-score residual and error analysis
9. Structural leakage checks and documented limitations

## Generated Evidence Files

- outputs/eda/eda_summary.txt
- outputs/eda/descriptive_stats.csv
- outputs/eda/correlation_heatmap.png
- outputs/eda/exam_score_distribution.png
- outputs/eda/placement_distribution.png
- outputs/metrics/baseline_metrics.csv
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

- LinearRegression: MAE 0.9225, RMSE 2.3378, R2 0.6585
- RandomForestRegressor: MAE 1.1977, RMSE 2.5892, R2 0.5812

### Classification (Placement)

- LogisticRegression: Accuracy 0.8924, Precision 0.7258, Recall 0.5660, F1 0.6360, ROC-AUC 0.9408
- RandomForestClassifier: Accuracy 1.0000, Precision 1.0000, Recall 1.0000, F1 1.0000, ROC-AUC 1.0000

## Interpretation Note

The perfect placement performance from RandomForestClassifier remains suspicious. It survived the training-only feature-selection correction, so it was not caused only by selecting features on the full dataset. It may indicate very easy separability, source-data leakage, or an artifact of the synthetic construction. It should not be presented as proof of real-world performance.

The placement dataset is imbalanced: 5,319 records belong to class 0 and 1,059 records belong to class 1. Therefore, positive-class precision, recall, F1, and the confusion matrix are more informative than accuracy alone.

LinearRegression is the stronger exam-score baseline. Its mean residual is close to zero, but this only indicates limited overall directional bias; it does not mean that every individual prediction is accurate.

## Teammate 3 Validation Findings

### Leakage checks

- Target columns are absent from the model feature columns.
- Feature selection is fitted on training data only in `scripts/select_and_split.py`.
- Placement splits contain no repeated encoded feature combinations across train, validation, and test.
- Exam-score splits contain one repeated encoded feature combination between train and validation with conflicting target values. This is a small data-quality warning that should be reviewed before final claims.
- Structural checks cannot prove semantic leakage-free data. The two source datasets were paired row-wise synthetically, so the learned relationship may not represent real student outcomes.

### Interpretation

- LogisticRegression provides a more conservative placement baseline: positive-class recall is 0.5660 and F1 is 0.6360.
- RandomForestClassifier obtains perfect test metrics, but this result requires stronger validation before being treated as reliable.
- The synthetic pairing strategy is the primary limitation of the current project because the performance and placement attributes were not collected as linked records from the same students.

## Reproduction Commands

Run these commands from the repository root:

```powershell
python scripts/dataset_checks.py
python scripts/select_and_split.py
python scripts/run_eda.py
python scripts/train_baselines.py
python scripts/analyze_baselines.py
```

The final command generates the confusion matrices, classification reports, residual analysis, and leakage validation files listed above.

## Remaining Tasks (Teammate 3)

1. Prepare the final 5 to 7 slide Evaluation-1 presentation using the generated evidence
2. Add repeated cross-validation and permutation testing for placement
3. Review the conflicting exam-score feature combination against the source data
4. Plan stronger models, explainability, and a real prediction interface for Phase 2

## Conclusion

The project has a reproducible baseline pipeline covering data checks, preprocessing artifacts, task-specific splitting, EDA, model training, and post-training analysis. The regression track has a reasonable initial baseline, while the placement track requires additional validation because of its perfect random-forest result and the synthetic row-wise pairing of the source datasets. The project is suitable for the Evaluation-1 milestone with these limitations clearly stated.
