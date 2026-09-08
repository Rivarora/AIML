# Student Performance Analytics

Minimal project scope:

- Predict student Exam_Score (regression)
- Predict student Placement (classification)

This repository currently covers data preparation, EDA artifacts, baseline modeling, and project structure for team-based continuation.

## Quick Start (Install and Run)

### 1) Clone and open

```powershell
git clone <repo-url>
cd AIML
```

### 2) Create environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Run pipeline scripts (from repository root)

```powershell
python scripts/preprocess.py
python scripts/dataset_checks.py
python scripts/select_and_split.py
python scripts/run_eda.py
python scripts/train_baselines.py
python scripts/analyze_baselines.py
```

### 4) Where outputs are generated

- EDA text/statistics/plots: outputs/eda/
- Model metrics: outputs/metrics/baseline_metrics.csv
- Saved baseline models: models/

Generated datasets, model files, and analysis outputs are intentionally not
committed to Git. The `.gitignore` file excludes `data/interim/`,
`data/processed/`, `data/splits/`, `models/`, and `outputs/`. Run the pipeline
commands to recreate them locally from the tracked raw data and source code.

### 5) If running from scripts folder

```powershell
cd scripts
python dataset_checks.py
python select_and_split.py
python run_eda.py
python train_baselines.py
```

## Team Ownership (Evaluation-1 Workflow)

### Teammate 1 (already completed before this stage)

- Collected raw datasets
- Performed core preprocessing and dataset preparation
- Built initial processed dataset used for splitting/modeling

### Teammate 2 (completed in this stage)

- Reorganized project structure for reproducibility
- Added script-based EDA pipeline
- Added script-based baseline model training for both tasks
- Added project documentation, progress status, and handoff notes
- Generated outputs (EDA plots, metrics CSV/JSON, saved baseline models)

### Teammate 3 (expected next)

- Validate model quality and check leakage risk
- Add confusion matrix and error analysis notes
- Convert outputs into final Evaluation-1 report/slides
- Document limitations and Phase-2 plan

### Teammate 3 (completed in this stage)

- Moved feature selection after splitting so `SelectKBest` is fitted on training data only
- Added placement confusion matrices and classification reports
- Added exam-score residual and error analysis
- Added structural leakage checks for target presence, repeated feature combinations, and conflicting targets
- Re-trained baseline models and documented the findings below

## What Counts as 40% (Evaluation-1)

Evaluation-1 (target 40%) includes:

1. Problem statement and objective definition
2. Data pipeline readiness (raw -> interim -> processed -> splits)
3. EDA outputs and basic insights
4. Baseline models for both tasks
5. Metrics summary and initial interpretation
6. Presentation-ready technical report/slides

Current status relative to 40%:

- Completed: 1 to 5 (baseline level)
- Remaining to close 40% cleanly: item 6 + deeper interpretation by Teammate 3

## What Is Left After 40% (Final Phase)

1. Hyperparameter tuning and stronger model comparison
2. Robust validation and leakage checks
3. Explainability (feature importance, SHAP or equivalent)
4. Better reporting/dashboard/demo
5. Final narrative: limitations, ethics/bias note, and deployment readiness

## Project Structure and File Meanings

```text
AIML/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/
|   |   |-- StudentPerformanceFactors.csv
|   |   |-- college_student_placement_dataset.csv
|   |-- interim/
|   |   |-- StudentPerformanceFactors_clean.csv
|   |   |-- combined_synthetic_dataset.csv
|   |-- processed/
|   |   |-- combined_encoded.csv
|   |   |-- final_preprocessed_dataset.csv
|   |-- splits/
|       |-- exam_score_train.csv
|       |-- exam_score_val.csv
|       |-- exam_score_test.csv
|       |-- placement_train.csv
|       |-- placement_val.csv
|       |-- placement_test.csv
|-- scripts/
|   |-- preprocess.py
|   |-- dataset_checks.py
|   |-- select_and_split.py
|   |-- run_eda.py
|   |-- train_baselines.py
|   |-- analyze_baselines.py
|-- outputs/
|   |-- eda/
|   |   |-- eda_summary.txt
|   |   |-- descriptive_stats.csv
|   |   |-- correlation_heatmap.png
|   |   |-- exam_score_distribution.png
|   |   |-- placement_distribution.png
|   |-- metrics/
|       |-- baseline_metrics.csv
|       |-- baseline_metrics.json
|       |-- exam_score_validation_reference.csv
|       |-- placement_validation_reference.csv
|   |-- analysis/
|       |-- placement_confusion_matrices.png
|       |-- *_confusion_matrix.csv
|       |-- *_classification_report.csv
|       |-- exam_score_residual_distribution.png
|       |-- exam_score_error_analysis.csv
|       |-- exam_score_error_summary.csv
|       |-- leakage_checks.csv
|       |-- leakage_validation.txt
|-- models/
|   |-- LinearRegression_exam_score.joblib
|   |-- RandomForestRegressor_exam_score.joblib
|   |-- LogisticRegression_placement.joblib
|   |-- RandomForestClassifier_placement.joblib
|-- docs/
|   |-- Data_Preprocessing_Documentation.docx
|   |-- summary_first_evaluation.md
|-- Data_Preprocessing_Documentation.docx (temporary root duplicate if lock prevented move)
```

### File Purpose Summary

- data/raw: original source datasets
- data/interim: cleaned/combined intermediate data
- data/processed: encoded/final model-ready dataset
- data/splits: train/validation/test files for each target
- scripts: reproducible pipeline and training scripts
- outputs/eda: EDA charts and statistics
- outputs/metrics: generated model metrics
- models: serialized baseline models
- docs: documentation and evaluation summary

## Manual Rebuild Guide (From Scratch to Current Stage)

This section explains how to rebuild the project manually (without relying on undocumented "vibecoded" steps).

### 0) Prerequisites

Concepts:

- Python environment management
- Reproducibility via pinned dependencies

Commands (PowerShell from repository root):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Data Collection

Concepts:

- Source integrity
- Schema awareness

Expected files:

- data/raw/StudentPerformanceFactors.csv
- data/raw/college_student_placement_dataset.csv

### 2) Data Cleaning and Standardization

Concepts:

- Missing value handling
- Duplicate removal
- Consistent datatypes

Implementation: `python scripts/preprocess.py`

The script removes duplicate rows and drops incomplete student records. It
writes `data/interim/StudentPerformanceFactors_clean.csv`.

### 3) Data Integration / Pair Construction

Concepts:

- Dataset joining or synthetic pairing strategy
- Target leakage awareness

Output artifact:

- data/interim/combined_synthetic_dataset.csv

The current source files do not share a student identifier, so the script
pairs the cleaned student rows with placement rows by row order and records a
`synthetic_pair_id`. This is a synthetic dataset construction, not a real
student-level join.

### 4) Encoding and Final Preprocessing

Concepts:

- Categorical encoding
- Numeric consistency
- Final feature matrix preparation

Expected outputs:

- data/processed/combined_encoded.csv
- data/processed/final_preprocessed_dataset.csv

The implementation uses explicit mappings for binary and ordinal categorical
fields, one-hot encodes `Peer_Influence`, maps the `Placement` target to 0/1,
and removes identifier columns that are not model features.

### 5) Data Quality Checks

Concepts:

- Null count
- Duplicate count
- Type verification

Command:

```powershell
python scripts/dataset_checks.py
```

### 6) Feature Selection + Train/Val/Test Split

Concepts:

- Statistical feature selection
- Separate task tracks (regression vs classification)
- Reproducible random_state

Important implementation detail:

- The dataset is split before `SelectKBest` is fitted. Feature rankings are learned from the training partition only, preventing validation and test targets from influencing feature selection.

Command:

```powershell
python scripts/select_and_split.py
```

Outputs:

- exam_score_train/val/test.csv
- placement_train/val/test.csv

### 7) EDA and Visualization

Concepts:

- Distribution analysis
- Correlation analysis
- Target balance check

Command:

```powershell
python scripts/run_eda.py
```

Outputs:

- outputs/eda/\*.png
- outputs/eda/descriptive_stats.csv
- outputs/eda/eda_summary.txt

### 8) Baseline Model Training

Concepts:

- Baseline benchmarking
- Task-specific metrics
- Model artifact persistence

Command:

```powershell
python scripts/train_baselines.py
```

Outputs:

- outputs/metrics/baseline_metrics.csv
- models/\*.joblib

### 9) Teammate 3 Analysis and Validation

Concepts:

- Confusion matrix and class-specific metrics
- Residual analysis for regression
- Structural leakage checks
- Evidence-based interpretation of suspiciously strong results

Command:

```powershell
python scripts/analyze_baselines.py
```

Outputs:

- outputs/analysis/placement_confusion_matrices.png
- outputs/analysis/\*\_classification_report.csv and `.txt`
- outputs/analysis/exam_score_residual_distribution.png
- outputs/analysis/exam_score_error_analysis.csv
- outputs/analysis/exam_score_error_summary.csv
- outputs/analysis/leakage_checks.csv
- outputs/analysis/leakage_validation.txt

## One-Command Reproduction (Current Stage)

```powershell
python scripts/preprocess.py
python scripts/dataset_checks.py
python scripts/select_and_split.py
python scripts/run_eda.py
python scripts/train_baselines.py
python scripts/analyze_baselines.py
```

## Current Baseline Metrics Snapshot

From outputs/metrics/baseline_metrics.csv:

- Exam_Score (LinearRegression): MAE 0.9225, RMSE 2.3378, R2 0.6585
- Exam_Score (RandomForestRegressor): MAE 1.1984, RMSE 2.5877, R2 0.5816
- Placement (LogisticRegression): Accuracy 0.8924, F1 0.6360, ROC-AUC 0.9408
- Placement (RandomForestClassifier): Accuracy 1.0000, F1 1.0000, ROC-AUC 1.0000

Note: Perfect classification metrics should be investigated for possible leakage before final conclusions.

## Teammate 3 Findings

### Placement Classification

- LogisticRegression remains the more conservative baseline: accuracy 0.8924, positive-class precision 0.7258, recall 0.5660, and F1 0.6360.
- RandomForestClassifier still produces perfect test metrics after training-only feature selection: accuracy, precision, recall, F1, and ROC-AUC are all 1.0000.
- Therefore, the perfect result was not caused only by selecting features on the full dataset. It remains a risk signal requiring stronger validation, such as repeated cross-validation, permutation testing, and review of the source data construction.
- Because the positive class is only 1,059 of 6,378 rows, accuracy alone is not sufficient; positive-class recall and F1 should be emphasized.

### Exam Score Regression

- LinearRegression remains the stronger baseline: MAE 0.9225, RMSE 2.3378, and R2 0.6585.
- RandomForestRegressor remains weaker: MAE 1.1977, RMSE 2.5892, and R2 0.5812.
- Mean residuals are close to zero for both models, so there is no large overall directional bias in the test predictions.

### Leakage and Data Validity

- Target columns are absent from the model feature columns.
- Placement splits have no repeated encoded feature combinations across train, validation, and test.
- Exam-score splits contain one repeated encoded feature combination between train/validation with conflicting target values. This is a small data-quality warning and should be reviewed before final claims.
- Structural split checks cannot prove that the source datasets are semantically leakage-free. The two original datasets were paired row-wise synthetically, so the learned placement relationship may not represent real student outcomes.

### What Teammate 3 Should Explain in the Presentation

1. Why the split must happen before feature selection.
2. How confusion matrices reveal false positives and false negatives beyond accuracy.
3. Why residuals near zero do not mean every exam-score prediction is accurate.
4. Why perfect random-forest classification is a warning requiring validation, not automatically a success claim.
5. Why synthetic pairing limits the real-world interpretation of both tasks.

## Next Action Checklist (Teammate 3)

1. Build 5 to 7 slide deck for Evaluation-1 using the generated evidence
2. Add repeated cross-validation and permutation testing for placement
3. Review the one conflicting exam-score feature combination
4. Finalize docs/summary_first_evaluation.md with conclusions and limitations

---

This README is intentionally operational and reproducible: anyone cloning this repository should be able to understand what exists, what was done, and what remains.
