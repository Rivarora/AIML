# Student Performance Analytics

Student Performance Analytics is a Python/Flask project for exploring two
independent student datasets:

- **Exam score regression** using `StudentPerformanceFactors.csv`
- **Placement classification** using `college_student_placement_dataset.csv`

The project includes reproducible preprocessing, exploratory analysis, data
quality checks, PCA, baseline model training, validation, and a small web
dashboard.

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
performance.

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
7. `train_baselines.py` trains, validates, evaluates, and saves model pipelines.
8. `analyze_baselines.py` creates residual, classification, and leakage
   analysis artifacts.

`scripts/model_config.py` is the ninth Python file in this directory, but it
is a shared configuration module rather than a script to run. It defines the
model features, categorical values, model labels, and reproducibility settings
used by `train_baselines.py`, `analyze_baselines.py`, and `app.py`.

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
- Baseline test metrics and five-fold cross-validation
- Overfitting analysis
- PCA visualization
- Placement-label investigation and project limitations

This is a Flask development server and is not intended for production
deployment.

## API endpoints

### Health and reports

```text
GET  /api/health
GET  /api/summary
GET  /api/metrics
GET  /api/cross-validation
GET  /api/overfitting
GET  /api/model-card
GET  /api/data-quality
GET  /api/pca
GET  /api/pca/visualization
GET  /api/placement-investigation
GET  /api/prediction-schema
GET  /api/clusters
GET  /api/fairness
```

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

- Numeric features: median imputation followed by `StandardScaler`
- Categorical features: most-frequent imputation followed by
  `OneHotEncoder(handle_unknown="ignore")`
- All preprocessing is fitted on training data only
- Five-fold cross-validation uses `random_state=42`

Deployed models:

| Track | Model | Artifact |
|---|---|---|
| Exam score | Linear Regression | `models/LinearRegression_exam_score.joblib` |
| Placement | Random Forest Classifier, `max_depth=2`, `min_samples_leaf=25` | `models/RandomForestClassifier_placement.joblib` |

## Current baseline results

Results below come from the generated test and cross-validation artifacts.
Metrics can change if the raw datasets or pipeline configuration changes.

| Track | Model | Test result | Five-fold CV |
|---|---|---|---|
| Exam | Linear Regression | MAE 0.9190, RMSE 2.1978, R2 0.6974 | MAE 0.9294 +/- 0.0553, R2 0.6681 +/- 0.0589 |
| Exam | Random Forest Regressor | MAE 1.1522, RMSE 2.3488, R2 0.6544 | MAE 1.1986 +/- 0.0565, R2 0.6163 +/- 0.0523 |
| Placement | Logistic Regression | Accuracy 0.8687, F1 0.6888, ROC-AUC 0.9487 | Accuracy 0.8514 +/- 0.0126, F1 0.6610 +/- 0.0260 |
| Placement | Random Forest Classifier | Accuracy 0.9533, F1 0.8768, ROC-AUC 0.9924 | Accuracy 0.9471 +/- 0.0044, F1 0.8627 +/- 0.0097 |

The deployed exam model is the stronger exam baseline. The deployed placement
model is intentionally capacity-limited because an unrestricted forest
recovers the deterministic source label with a perfect score.

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

models/
|-- LinearRegression_exam_score.joblib
|-- RandomForestRegressor_exam_score.joblib
|-- LogisticRegression_placement.joblib
|-- RandomForestClassifier_placement.joblib
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

## Repository structure

```text
AIML/
|-- app.py
|-- requirements.txt
|-- data/raw/
|-- scripts/
|   |-- model_config.py
|   |-- preprocess.py
|   |-- dataset_checks.py
|   |-- select_and_split.py
|   |-- data_quality.py
|   |-- pca_analysis.py
|   |-- run_eda.py
|   |-- train_baselines.py
|   |-- analyze_baselines.py
|-- web/index.html
|-- docs/
|   |-- Data_Preprocessing_Documentation.docx
|   |-- summary_first_evaluation.md
|-- README.md
```

## Limitations and next steps

- The datasets are independent and cannot support student-level outcome
  analysis.
- The placement target is deterministic in the source data.
- The baseline models are not production-ready or evidence of causal effects.
- Further work could add explainability, stronger model comparison, external
  validation, fairness analysis, and production deployment controls.
