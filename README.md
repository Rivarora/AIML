# Student Performance Analytics

A decision-support workspace for exploring two independent student datasets:

- **Academic performance** — exam-score regression using
  `data/raw/StudentPerformanceFactors.csv`
- **Placement readiness** — placement classification using
  `data/raw/college_student_placement_dataset.csv`

The product pairs a reproducible analysis pipeline with a six-area web
workspace: Overview, Academic Performance, Placement Readiness, Student
Insights, Recommendations, and Model Insights. Every number shown in the UI is
read from generated artifacts through the Flask API.

## Important limitations

The two source datasets do **not** share a student identifier. They are
processed, split, and modeled as separate tracks. No exam record is ever linked
to a placement record, and row-order pairing has been removed.

The placement label is deterministic in the source data: `Placement` is `Yes`
when at least three of these conditions hold:

- `IQ > 110`
- `CGPA > 8`
- `Projects_Completed > 1`
- `Communication_Skills > 7`

That rule reproduces all 10,000 placement labels. Consequently, near-perfect
placement metrics describe how the synthetic dataset was constructed; they are
not evidence of real-world predictive skill. The deployed placement model is
deliberately capacity-limited, and the UI labels the placement track as
experimental.

Structural checks found no target column among the model features, but the
project does **not** claim the data are leakage-free. The placement target is
derivable from four features and the exam training split contains a repeated
feature combination with conflicting targets. All estimates are experimental,
non-causal, and not institutional decisions.

## Requirements

- Python 3.10 or newer
- The pinned packages in `requirements.txt`

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Saved scikit-learn pipelines are version-sensitive. If the scikit-learn version
changes, rerun the pipeline before starting the app.

## Reproduce the pipeline

The runner executes every step in dependency order, stops on the first failure,
and checks that each step wrote its expected artifacts.

```bash
python scripts/run_pipeline.py
python scripts/run_pipeline.py --list
python scripts/run_pipeline.py --only preprocess select_and_split train_models
```

Pipeline steps:

1. `preprocess.py` validates and cleans both source tracks, bounds impossible
   values, and writes provenance.
2. `dataset_checks.py` checks processed ranges and task-track integrity.
3. `select_and_split.py` creates reproducible 70/15/15 splits.
4. `data_quality.py` writes data-quality, split, and linkage reports.
5. `pca_analysis.py` fits training-only PCA for each track.
6. `run_eda.py` writes exploratory summaries and correlation plots.
7. `train_baselines.py` fits the original baselines, the model card, and the
   placement deterministic-rule investigation.
8. `analyze_baselines.py` writes residual, classification, and legacy analysis.
9. `train_models.py` fits all registered pipelines and saves held-out metrics
   and confusion matrices.
10. `cross_validation.py` runs five-fold cross-validation on the training split
    only.
11. `overfitting_analysis.py` compares train, validation, test, and CV results
    and flags suspicious cases.
12. `leakage_investigation.py` checks for target columns, duplicate feature
    combinations, conflicting labels, and deterministic rules.
13. `explain_models.py` computes coefficients, tree importances, and
    permutation importance on the test split.
14. `clustering.py` fits train-only K-Means on target-free exam features and
    writes descriptive cohort profiles.
15. `fairness_analysis.py` compares held-out exam error across groups with a
    minimum group size.

`scripts/model_config.py` is the single source of truth for selected model
features, API field names, validity bounds, category values, deployed artifact
names, and per-track limitations. Training, API validation, and the frontend
schema endpoint all read from it.

## Run the app

```bash
python app.py
```

Open <http://127.0.0.1:5000>. This is the Flask development server; it is not a
production deployment.

The workspace includes:

- **Overview** — dataset scale, independence warning, deployed model summary,
  and the evidence the data does and does not support.
- **Academic Performance** — schema-driven exam form with client and server
  validation, a bounded estimate, an approximate ±1 held-out-RMSE band, and
  visible limitations.
- **Placement Readiness** — schema-driven placement form with an explicit
  synthetic/deterministic-source warning. The probability is a model output at a
  0.5 threshold, not a calibrated probability.
- **Student Insights** — train-only cluster profiles, PCA views, grouped
  held-out error, and data health.
- **Recommendations** — transparent threshold-based advisory prompts. These are
  not model output or decisions.
- **Model Insights** — held-out metrics for all ten models, five-fold CV,
  train/validation/test comparison, overfitting flags, confusion matrices,
  permutation importance, and leakage evidence.

Missing artifacts return an explicit JSON `503`; the UI shows a clear empty or
error state instead of a fabricated result. The API never retrains during a
prediction request.

## Tests

The tests use the Flask test client and the standard library only.

```bash
python -m unittest discover -s tests -v
```

They cover the page shell, static assets, health and schema contracts, valid
and invalid prediction payloads, category handling, advisory prompts, JSON
`404`/`405` responses, output bounds, determinism, and the no-linkage guarantee.

## API

### Reports

```text
GET  /api/health
GET  /api/summary
GET  /api/metrics
GET  /api/model-performance
GET  /api/cross-validation
GET  /api/overfitting
GET  /api/feature-importance
GET  /api/leakage
GET  /api/placement-investigation
GET  /api/model-card
GET  /api/data-quality
GET  /api/pca
GET  /api/pca/visualization
GET  /api/clusters
GET  /api/clusters/visualization
GET  /api/fairness
GET  /api/prediction-schema
GET  /api/advisory-rules
```

Report endpoints read generated files; no metric is hardcoded in the API.

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

Validation returns `400` with the offending `field`. Invalid JSON shapes,
missing values, booleans-as-numbers, non-finite numbers, out-of-range values,
and unknown categories are rejected. Category labels are matched
case-insensitively; legacy zero-based indexes are also accepted. Model or
artifact failures return `503`. Unknown API routes and wrong methods return
JSON `404`/`405`.

`POST /api/interventions` accepts any subset of the numeric fields and returns
only the prompts whose thresholds are crossed. It never returns a prediction.

## Modeling approach

Each saved model is a scikit-learn `Pipeline`:

- Numeric features: median imputation, then `StandardScaler` only for models
  that need scaling (Logistic Regression, KNN, SVM)
- Categorical features: most-frequent imputation, then
  `OneHotEncoder(handle_unknown="ignore")`
- A `VarianceThreshold` filter is fitted inside every pipeline and every CV fold
- All preprocessing and feature selection are fitted on training data only
- Fixed seeds (`random_state=42`); classification CV is stratified
- No split file or `College_ID`-style identifier is read during inference

| Track | Models |
|---|---|
| Exam score regression | Linear Regression, Ridge, Random Forest Regressor |
| Placement classification | Logistic Regression, KNN, Decision Tree, Random Forest, unconstrained Random Forest (diagnostic only), SVM, Gaussian Naive Bayes |

Deployed models:

| Track | Model | Artifact |
|---|---|---|
| Exam score | Linear Regression | `models/LinearRegression_exam_score.joblib` |
| Placement | Random Forest, `max_depth=2`, `min_samples_leaf=25` | `models/RandomForestClassifier_placement.joblib` |

The unconstrained placement Random Forest exists only to show how easily the
deterministic source label can be recovered. It is not deployed.

## Current held-out results

Values below are regenerated from
`outputs/evaluation/model_performance.json` and
`outputs/evaluation/cross_validation_results.json`; they change if the data or
configuration changes.

| Track | Model | Test result | Five-fold CV |
|---|---|---|---|
| Exam | Linear Regression | MAE 0.9190, RMSE 2.1978, R² 0.6974 | MAE 0.9294 ± 0.0553, R² 0.6681 ± 0.0589 |
| Exam | Ridge | MAE 0.9190, RMSE 2.1978, R² 0.6974 | MAE 0.9293 ± 0.0553, R² 0.6681 ± 0.0589 |
| Exam | Random Forest Regressor | MAE 1.1516, RMSE 2.3489, R² 0.6544 | MAE 1.1983 ± 0.0569, R² 0.6164 ± 0.0524 |
| Placement | Logistic Regression | Accuracy 0.8687, F1 0.6888, ROC-AUC 0.9487 | Accuracy 0.8514 ± 0.0126, F1 0.6610 ± 0.0260 |
| Placement | KNN | Accuracy 0.9460, F1 0.8180, ROC-AUC 0.9821 | Accuracy 0.9390 ± 0.0057, F1 0.8017 ± 0.0200 |
| Placement | Decision Tree | Accuracy 1.0000, F1 1.0000, ROC-AUC 1.0000 | Accuracy 0.9993 ± 0.0008, F1 0.9979 ± 0.0023 |
| Placement | Random Forest (deployed) | Accuracy 0.9533, F1 0.8768, ROC-AUC 0.9924 | Accuracy 0.9471 ± 0.0044, F1 0.8627 ± 0.0097 |
| Placement | Random Forest (unconstrained, diagnostic) | Accuracy 1.0000, F1 1.0000, ROC-AUC 1.0000 | Accuracy 0.9994 ± 0.0007, F1 0.9983 ± 0.0021 |
| Placement | SVM | Accuracy 0.9367, F1 0.8376, ROC-AUC 0.9903 | Accuracy 0.9204 ± 0.0079, F1 0.8022 ± 0.0162 |
| Placement | Gaussian Naive Bayes | Accuracy 0.9167, F1 0.7191, ROC-AUC 0.9211 | Accuracy 0.9020 ± 0.0111, F1 0.6839 ± 0.0378 |

**Most suspicious result:** the placement Decision Tree and unconstrained
Random Forest reach 1.0000 on the test split and about 0.999 in
cross-validation. This is a deterministic-label artifact. The deployed
placement forest scores 0.9533 accuracy because it is intentionally limited; it
is still not reliable evidence about real placement. The overfitting report also
flags the exam Random Forest for a train-to-test R² gap of about 0.110.

## Generated files

Generated files are excluded from Git and can be recreated by
`scripts/run_pipeline.py`.

```text
data/
|-- interim/
|-- processed/
|-- splits/
outputs/
|-- eda/
|-- quality/
|-- pca/
|-- metrics/
|-- analysis/
|-- evaluation/
|-- explanations/
|-- clustering/
|-- fairness/
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

Key reports:

- `outputs/quality/data_quality.json` — schema, missing values, duplicates, and
  linkage facts
- `outputs/pca/pca_summary.json` — explained variance and PCA metadata
- `outputs/metrics/model_card.json` — deployed artifacts and preprocessing
- `outputs/analysis/placement_investigation.json` — deterministic-label audit
- `outputs/evaluation/model_performance.json` — train/validation/test metrics,
  confusion matrices, classification reports
- `outputs/evaluation/cross_validation_results.json` — fold means, standard
  deviations, and gaps
- `outputs/evaluation/overfitting_report.json` — gap, variance, and near-perfect
  flags
- `outputs/evaluation/leakage_report.json` — structural checks and the
  deterministic placement rule
- `outputs/explanations/explanation_summary.json` — coefficients, tree
  importance, and permutation importance
- `outputs/clustering/cluster_summary.json` — train-only clusters and limits
- `outputs/fairness/fairness_summary.json` — descriptive group error comparison

## Repository structure

```text
AIML/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/raw/                     # canonical source data
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
|   |-- clustering.py
|   |-- fairness_analysis.py
|   |-- interventions.py
|   |-- run_pipeline.py
|-- tests/test_app.py
|-- web/
|   |-- index.html
|   |-- app.js
|   |-- styles.css
|   |-- favicon.svg
|   |-- llms.txt
|-- docs/
|-- README.md
```

## Limitations and next steps

- The datasets are independent and synthetic; they cannot support student-level
  outcome analysis.
- The placement label is deterministic in the source. Near-perfect placement
  results are not evidence of reliability.
- The models are experimental, not production systems, and have no causal
  interpretation.
- Cluster profiles and group error comparisons are descriptive. They are not
  permanent student labels, a fairness certification, or evidence of
  discrimination.
- Feature and permutation importance describe association with prediction, not
  causal effects.
- Further work could add external validation, calibrated probability checks,
  confidence intervals for group metrics, and deployment controls.
