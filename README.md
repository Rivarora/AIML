# Student Performance Analytics

Student Performance Analytics is a Python/Flask project for exploring two
independent student datasets:

- **Exam score regression** using `StudentPerformanceFactors.csv`
- **Placement classification** using `college_student_placement_dataset.csv`

The project includes reproducible preprocessing, exploratory analysis, data
quality checks, PCA, baseline model training, additional supervised models,
five-fold cross-validation, overfitting analysis, leakage investigation,
feature-importance explanations, and a small web dashboard.

## Important limitation

The source datasets do not share a student identifier. They are processed and
modeled as separate tracks. The project does not claim that an exam record and a
placement record describe the same student, and no row-order pairing is used.
Placement predictions must not be interpreted as outcomes for the exam records.

The placement label is deterministic in the source dataset: `Placement` is
`Yes` when at least three of these conditions hold:

- `IQ > 110`
- `CGPA > 8`
- `Projects_Completed > 1`
- `Communication_Skills > 7`

Consequently, very high placement metrics describe the construction of this
dataset and should not be treated as evidence of real-world predictive
performance. The source data are synthetic, and the historical row-order
pairing was removed because it did not identify the same student. The
near-perfect placement Random Forest result is therefore suspicious: it is a
dataset-construction artifact, not a reliable model. All reports and UI labels
mark placement performance as **experimental**.

The project does not claim that the data are leakage-free. Structural checks
found no target column in the feature list, but the placement target is
derivable from four features, and the two source files share no genuine
student identifier.

## Requirements

- Python 3.10 or newer
- PowerShell on Windows, or an equivalent shell

Install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

The current environment pins Flask, pandas, scikit-learn, matplotlib, and
joblib. Saved scikit-learn models are version-sensitive. If the scikit-learn
version changes, retrain the generated models before starting the dashboard:

```powershell
python scripts/train_baselines.py
```

## Input data

Place the following files in `data/raw/`:

```text
data/raw/StudentPerformanceFactors.csv
data/raw/college_student_placement_dataset.csv
```

These source files are expected to contain the exam-score and placement
features used by the pipeline. The preprocessing script validates required
columns, data types, ranges, categories, duplicates, and missing values.

## Reproduce the pipeline

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
python scripts/train_models.py
python scripts/cross_validation.py
python scripts/overfitting_analysis.py
python scripts/leakage_investigation.py
python scripts/explain_models.py
python scripts/clustering.py
python scripts/fairness_analysis.py
```

The scripts run in this order:

1. `preprocess.py` cleans the two source tracks independently and writes
   provenance information.
2. `dataset_checks.py` performs structural dataset checks.
3. `select_and_split.py` creates reproducible 70/15/15 train, validation, and
   test splits.
4. `data_quality.py` writes data-quality and linkage reports.
5. `pca_analysis.py` fits task-specific PCA transformations using training
   features only.
6. `run_eda.py` creates distributions, correlations, and summary statistics.
7. `train_baselines.py` trains, validates, evaluates, and saves the original
   baseline pipelines.
8. `analyze_baselines.py` creates residual, classification, and leakage
   analysis artifacts.
9. `train_models.py` fits the expanded model set and writes held-out
   train/validation/test performance plus confusion matrices.
10. `cross_validation.py` runs five-fold `KFold` or `StratifiedKFold` on the
    training split only.
11. `overfitting_analysis.py` compares training, validation, test, and
    cross-validation results and flags suspicious cases.
12. `leakage_investigation.py` checks for target columns in features,
    duplicate/conflicting feature combinations, strongly predictive columns,
    deterministic targets, and synthetic-pairing limitations.
13. `explain_models.py` computes linear coefficients, tree importances, and
    permutation importance on the test split.
14. `clustering.py` groups exam-track inputs without using the exam score and
    writes descriptive cluster summaries.
15. `fairness_analysis.py` compares held-out exam-model error across available
    groups with a minimum group-size threshold.

`scripts/model_config.py` is a shared configuration module rather than a
script to run. `scripts/model_pipelines.py` is the reusable pipeline and model
registry module used by the expanded training, cross-validation, and
explainability scripts.

## Run the dashboard

After the pipeline has generated the required artifacts:

```powershell
python app.py
```

Open <http://127.0.0.1:5000>.

The dashboard provides:

- Dataset summaries and data-quality status
- Exam-score prediction with the deployed Linear Regression pipeline
- Placement prediction with a depth-limited Random Forest pipeline
- Model comparison across ten supervised models (see below)
- Five-fold cross-validation with mean and standard deviation
- Train/validation/test comparison and overfitting/reliability flags
- Confusion matrices and classification summaries
- Feature-importance bars with cautious, non-causal wording
- Leakage investigation, PCA visualization, and project limitations
- A clear **Experimental result — synthetic data limitation** warning for the
  placement track

This is a Flask development server and is not intended for production
deployment.

## API endpoints

### Health and reports

```text
GET  /api/health
GET  /api/summary
GET  /api/metrics
GET  /api/model-performance
GET  /api/cross-validation
GET  /api/overfitting
GET  /api/feature-importance
GET  /api/leakage
GET  /api/model-card
GET  /api/data-quality
GET  /api/pca
GET  /api/pca/visualization
GET  /api/placement-investigation
GET  /api/prediction-schema
GET  /api/clusters
GET  /api/fairness
```

`/api/model-performance`, `/api/cross-validation`, `/api/overfitting`,
`/api/feature-importance`, and `/api/leakage` read the generated evaluation
files. They do not contain hardcoded metric values.

### Predictions

```text
POST /api/predict/exam
POST /api/predict/placement
POST /api/interventions
```

Example exam request:

```json
{
  "hours_studied": 20,
  "attendance": 90,
  "previous_scores": 80,
  "tutoring_sessions": 2,
  "access_to_resources": "High",
  "parental_involvement": "Medium",
  "parental_education_level": "College",
  "family_income": "Medium"
}
```

Example placement request:

```json
{
  "iq": 120,
  "prev_sem_result": 8,
  "cgpa": 8.5,
  "academic_performance": 8,
  "internship_experience": "Yes",
  "extra_curricular_score": 7,
  "communication_skills": 8,
  "projects_completed": 3
}
```

The prediction endpoints validate numeric ranges and category values. Category
fields accept their labels; legacy zero-based category indexes are also
accepted by the API.

Prediction values are experimental estimates, not guarantees. Placement
responses explicitly identify the synthetic/experimental source data. The
interventions endpoint returns conservative advisory prompts separately from
model output.

Clustering uses K-Means with standardized, target-free exam-track inputs and
writes its assignments, summary, silhouette score, and visualization under
`outputs/clustering/`. Fairness analysis compares exam regression MAE across
available Gender, Family_Income, School_Type, and Learning_Disabilities groups
with a minimum group-size threshold; it is not a complete fairness audit.

## Modeling approach

The model feature definitions are centralized in
[`scripts/model_config.py`](scripts/model_config.py).

Each saved model is a scikit-learn `Pipeline`:

- Numeric features: median imputation, then `StandardScaler` only for models
  that need scaling (Logistic Regression, KNN, SVM)
- Categorical features: most-frequent imputation followed by
  `OneHotEncoder(handle_unknown="ignore")`
- A zero-variance `VarianceThreshold` filter is fitted inside every pipeline
- All preprocessing and feature selection are fitted on training data only
- Five-fold cross-validation uses fixed seeds (`random_state=42`)
- Classification cross-validation is stratified

Models added in the expanded comparison:

| Track | Models |
|---|---|
| Exam score regression | Linear Regression, Ridge, Random Forest Regressor |
| Placement classification | Logistic Regression, KNN, Decision Tree, Random Forest, unconstrained Random Forest (diagnostic only), SVM, Gaussian Naive Bayes |

Deployed models:

| Track | Model | Artifact |
|---|---|---|
| Exam score | Linear Regression | `models/LinearRegression_exam_score.joblib` |
| Placement | Random Forest Classifier, `max_depth=2`, `min_samples_leaf=25` | `models/RandomForestClassifier_placement.joblib` |

The unconstrained placement Random Forest is a diagnostic demonstration of the
deterministic source label. It is **not** a reliable or deployed model.

## Current model comparison

Results below come from `outputs/evaluation/model_performance.json` and
`outputs/evaluation/cross_validation_results.json`. Metrics can change if the
raw datasets or pipeline configuration changes.

| Track | Model | Test result | Five-fold CV |
|---|---|---|---|
| Exam | Linear Regression | MAE 0.9190, RMSE 2.1978, R2 0.6974 | MAE 0.9294 +/- 0.0553, R2 0.6681 +/- 0.0589 |
| Exam | Ridge | MAE 0.9190, RMSE 2.1978, R2 0.6974 | MAE 0.9293 +/- 0.0553, R2 0.6681 +/- 0.0589 |
| Exam | Random Forest Regressor | MAE 1.1516, RMSE 2.3489, R2 0.6544 | MAE 1.1983 +/- 0.0569, R2 0.6164 +/- 0.0524 |
| Placement | Logistic Regression | Accuracy 0.8687, F1 0.6888, ROC-AUC 0.9487 | Accuracy 0.8514 +/- 0.0126, F1 0.6610 +/- 0.0260 |
| Placement | KNN | Accuracy 0.9460, F1 0.8180, ROC-AUC 0.9821 | Accuracy 0.9390 +/- 0.0057, F1 0.8017 +/- 0.0200 |
| Placement | Decision Tree | Accuracy 1.0000, F1 1.0000, ROC-AUC 1.0000 | Accuracy 0.9993 +/- 0.0008, F1 0.9979 +/- 0.0023 |
| Placement | Random Forest (deployed) | Accuracy 0.9533, F1 0.8768, ROC-AUC 0.9924 | Accuracy 0.9471 +/- 0.0044, F1 0.8627 +/- 0.0097 |
| Placement | Random Forest (unconstrained, diagnostic) | Accuracy 1.0000, F1 1.0000, ROC-AUC 1.0000 | Accuracy 0.9994 +/- 0.0007, F1 0.9983 +/- 0.0021 |
| Placement | SVM | Accuracy 0.9367, F1 0.8376, ROC-AUC 0.9903 | Accuracy 0.9204 +/- 0.0079, F1 0.8022 +/- 0.0162 |
| Placement | Gaussian Naive Bayes | Accuracy 0.9167, F1 0.7191, ROC-AUC 0.9211 | Accuracy 0.9020 +/- 0.0111, F1 0.6839 +/- 0.0378 |

**Most suspicious result:** the placement Decision Tree and the unconstrained
placement Random Forest both reach 1.0000 on the test split and about 0.999 in
cross-validation. This is a deterministic-label/synthetic-data artifact. The
placement Random Forest is **not** reliable, even though its own deployed
version scores 0.9533 accuracy. The deployed exam model is the stronger exam
baseline. The overfitting report also flags the exam Random Forest for a
train-to-test R2 gap of about 0.110.

## Generated files

Generated files are excluded from Git by `.gitignore` and can be recreated by
running the pipeline.

```text
data/
|-- interim/
|   |-- StudentPerformanceFactors_clean.csv
|   |-- placement_clean.csv
|   |-- provenance.json
|-- processed/
|   |-- exam_score_preprocessed.csv
|   |-- placement_preprocessed.csv
|-- splits/
    |-- exam_score_train.csv
    |-- exam_score_val.csv
    |-- exam_score_test.csv
    |-- placement_train.csv
    |-- placement_val.csv
    |-- placement_test.csv
    |-- split_manifest.json

outputs/
|-- eda/
|-- quality/
|-- pca/
|-- metrics/
|-- analysis/
|-- evaluation/
|   |-- model_performance.csv/json
|   |-- cross_validation_results.csv/json
|   |-- overfitting_report.csv/json
|   |-- leakage_report.csv/json
|-- explanations/
|   |-- exam_score_importance.csv
|   |-- placement_importance.csv
|   |-- explanation_summary.json

models/
|-- LinearRegression_exam_score.joblib
|-- Ridge_exam_score.joblib
|-- RandomForestRegressor_exam_score.joblib
|-- LogisticRegression_placement.joblib
|-- KNeighborsClassifier_placement.joblib
|-- DecisionTreeClassifier_placement.joblib
|-- RandomForestClassifier_placement.joblib
|-- RandomForestClassifierUnconstrained_placement.joblib
|-- SVC_placement.joblib
|-- GaussianNB_placement.joblib
```

Important reports include:

- `outputs/quality/data_quality.json`: schema, missing-value, duplicate, and
  linkage checks
- `outputs/pca/pca_summary.json`: explained variance and PCA metadata
- `outputs/metrics/baseline_metrics.csv`: test, validation, and training
  metrics
- `outputs/metrics/cross_validation.csv`: five-fold validation results
- `outputs/metrics/overfitting_analysis.csv`: train-versus-test gaps
- `outputs/metrics/model_card.json`: deployed model and preprocessing metadata
- `outputs/analysis/placement_investigation.json`: deterministic-label audit
- `outputs/analysis/leakage_validation.txt`: structural leakage checks
- `outputs/evaluation/model_performance.json`: train/validation/test metrics,
  confusion matrices, and classification reports for all models
- `outputs/evaluation/cross_validation_results.json`: five-fold mean, standard
  deviation, per-fold values, and train/validation gaps
- `outputs/evaluation/overfitting_report.json`: flags for large gaps, unstable
  folds, near-perfect placement scores, and the most suspicious result
- `outputs/evaluation/leakage_report.json`: target-in-feature checks, duplicate
  feature combinations, conflicting labels, feature associations, and the
  deterministic placement rule
- `outputs/explanations/explanation_summary.json`: linear coefficients, tree
  importance, and permutation importance on the test split. These describe
  association with prediction, not causal effects.

## Repository structure

```text
AIML/
|-- app.py
|-- requirements.txt
|-- data/raw/
|-- scripts/
|   |-- model_config.py
|   |-- model_pipelines.py
|   |-- preprocess.py
|   |-- dataset_checks.py
|   |-- select_and_split.py
|   |-- data_quality.py
|   |-- pca_analysis.py
|   |-- run_eda.py
|   |-- train_baselines.py
|   |-- analyze_baselines.py
|   |-- train_models.py
|   |-- cross_validation.py
|   |-- overfitting_analysis.py
|   |-- leakage_investigation.py
|   |-- explain_models.py
|-- web/index.html
|-- docs/
|   |-- Data_Preprocessing_Documentation.docx
|   |-- summary_first_evaluation.md
|-- README.md
```

## Limitations and next steps

- The datasets are independent and cannot support student-level outcome
  analysis. The source data are synthetic and the historical row-order pairing
  was removed.
- The placement target is deterministic in the source data. Near-perfect
  placement results are **not** evidence that the placement Random Forest is
  reliable.
- The models are not production-ready or evidence of causal effects.
- Feature importance and permutation importance describe associations with
  prediction. They must not be converted into personal advice.
- The data cannot be certified leakage-free. Structural checks pass for the
  target column, but the placement label is derivable from four features and
  the exam training split contains a repeated feature combination with
  conflicting targets.
- Further work could add external validation, fairness analysis, calibrated
  probability checks, and production deployment controls.
