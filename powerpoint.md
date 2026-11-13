# Student Performance Analytics
## PowerPoint Content Draft

> Copy the slide titles and bullet points into PowerPoint. Project-specific
> names, mentor, date, and source URLs were not included in this repository;
> add them before submission.

---

## Slide 1: Title

### Artificial Intelligence & Machine Learning (24CSE0316)

**Student Performance Analytics**

Predicting Exam Performance and Placement Outcomes Using Machine Learning

Chitkara University Institute of Engineering and Technology  
Chitkara University, Punjab, India  
Department of Computer Science & Engineering

**Team Details**
- Team members: Not provided in repository
- Faculty mentor: Not provided in repository
- Evaluation: Evaluation-1 (date not provided)

**Speaker point:** This is a two-track machine learning project: one regression problem for Exam_Score and one classification problem for Placement.

---

## Slide 2: Project Overview

### What the project does

- Analyzes student academic, behavioral, and skill-related information.
- Predicts **Exam_Score** as a continuous value from 0 to 100.
- Predicts **Placement** as a binary outcome.
- Uses two separate prediction tracks because the source datasets do not contain a shared student identifier.
- Uses reproducible data preparation, EDA, splitting, pipeline-based modeling, five-fold cross-validation, and post-training validation.

### Datasets

- StudentPerformanceFactors.csv: 6,607 original records and 20 columns.
- college_student_placement_dataset.csv: 10,000 original records and 10 columns.
- Independent track files: 6,378 exam records with 22 encoded columns; 10,000 placement records with 9 encoded columns.
- No row-order, combined, or synthetic-pair dataset exists in this version.

### Current project stage

- Data pipeline completed.
- EDA outputs generated.
- Pipeline-based baseline models trained.
- Five-fold cross-validation and training-versus-test overfitting analysis completed.
- Placement deterministic-label investigation completed.
- Confusion matrices, residual analysis, and structural leakage checks completed.
- Advanced tuning, explainability, and deployment remain for the final phase.

**Suggested visual:** A two-branch overview: student factors -> Exam_Score regression; academic/skill factors -> Placement classification.

---

## Slide 3: Problem Statement

Educational institutions collect information about attendance, study habits, academic history, communication skills, and other student factors. However, this information is often not used early enough to identify students who may need support.

This project addresses two related questions:

1. Can student academic and behavioral factors be used to estimate Exam_Score?
2. Can academic and professional-skill indicators be used to estimate Placement likelihood?

### Intended value

- Identify important factors associated with academic performance.
- Help identify students who may benefit from early academic or career support.
- Provide a measurable machine learning workflow rather than relying only on manual observation.
- Support future development of an explainable student analytics dashboard.

### Scope limitation

The two source datasets are not linked records from the same students. They are processed as separate target tracks, and no row-order pairing is created. Therefore, results demonstrate a modeling workflow and should not yet be treated as real-world causal or deployment-ready predictions.

---

## Slide 4: Objectives

1. Collect two student-related datasets covering academic performance and placement outcomes.
2. Inspect the datasets for missing values, duplicates, invalid values, and irrelevant identifiers.
3. Clean and standardize the available data while preserving valid records.
4. Fit imputation, scaling, and one-hot encoding inside reusable preprocessing pipelines.
5. Perform exploratory data analysis and create visual summaries.
6. Use separate, explainable feature sets for the regression and classification tasks.
7. Create reproducible training, validation, and test partitions.
8. Train baseline models for both targets.
9. Evaluate the models using task-appropriate metrics and five-fold cross-validation.
10. Check for overfitting, structural leakage, repeated feature combinations, and suspicious results.
11. Document limitations and define the next phase of the project.

### Current project foundation

The completed baseline demonstrates a reproducible pipeline from raw datasets to analysis-ready model results.

---

## Slide 5: Dataset Collection and Description

### Dataset 1: StudentPerformanceFactors.csv

- **Source:** Public Kaggle educational dataset; original URL not provided in repository.
- Original size: 6,607 records, 20 columns.
- Target: **Exam_Score**.
- Task type: regression.
- Example predictors:
  - Hours_Studied
  - Attendance
  - Previous_Scores
  - Sleep_Hours
  - Tutoring_Sessions
  - Parental_Involvement
  - Teacher_Quality
  - Family_Income
  - School_Type
  - Learning_Disabilities

### Dataset 2: college_student_placement_dataset.csv

- **Source:** Public Kaggle student placement dataset; original URL not provided in repository.
- Original size: 10,000 records, 10 columns.
- Target: **Placement**.
- Task type: binary classification.
- Example predictors:
  - IQ
  - Prev_Sem_Result
  - CGPA
  - Academic_Performance
  - Internship_Experience
  - Extra_Curricular_Score
  - Communication_Skills
  - Projects_Completed

### Important data relationship

- The datasets do not share a reliable student ID.
- They are not a genuine joined dataset.
- `College_ID` is a college-level identifier and has no counterpart in the exam file.
- The project keeps the two prediction tracks separate and creates no combined dataset.

**Suggested visual:** Two dataset cards with target, task type, row count, and sample features.

### Source citation to display on the slide

```text
Data sources: Public Kaggle datasets
1. StudentPerformanceFactors.csv - URL not recorded in repository
2. college_student_placement_dataset.csv - URL not recorded in repository
Accessed: date not recorded in repository
```

---

## Slide 6: Data Preparation and Cleaning

### Checks performed

- Loaded both CSV datasets using pandas.
- Inspected shape, columns, data types, null counts, and duplicate rows.
- Checked target validity and identifier usefulness.
- Removed duplicate rows where applicable.
- Removed College_ID from the placement feature set because it is an identifier rather than a meaningful predictive factor.
- Corrected the invalid Exam_Score value above the 0-100 range using an upper bound of 100.

### Missing values found in Dataset 1

- Teacher_Quality: 78 missing values.
- Parental_Education_Level: 90 missing values.
- Distance_from_Home: 67 missing values.
- Dataset 2 had no missing values according to the preprocessing record.

### Cleaning result

- All model inputs were converted into a consistent numeric format.
- The final processed dataset contains 0 reported null values and 0 duplicate rows in the current quality check.
- The cleaning script removes duplicate rows and incomplete performance rows;
  6,378 of 6,607 performance rows remain.

**Suggested visual:** Before/after table showing issue, treatment, and result.

---

## Slide 7: Missing-Value Treatment

### Method: explicit cleaning plus training-only imputation

The cleaning script removes duplicate rows and rows with missing values from
the exam source, leaving 6,378 complete performance records. Placement has no
missing values. The invalid `Exam_Score` value of 101 is clipped to 100, and
249 `CGPA` values above 10 are clipped to 10.

Each model pipeline also contains median imputation for numeric fields and
most-frequent imputation for categorical fields. Those imputers are fitted on
the training split only, so validation and test rows cannot influence them.

### Limitation

Dropping incomplete rows can reduce the available sample and may bias the
dataset if missingness is systematic. The pipeline imputation is a safety step,
not a substitute for reviewing why values are missing.

**Suggested visual:** A small flow diagram: raw rows -> remove duplicates and
incomplete rows -> clip invalid ranges -> split -> pipeline imputation and
encoding.

---

## Slide 8: Data Transformation and Categorical Encoding

### Why encoding is required

Most machine learning algorithms require numeric input. Text categories such as Low, Medium, High, Yes, No, and school-type labels must be represented numerically.

### Transformations used in the model pipelines

- Numeric features: median imputation, then `StandardScaler`.
- Categorical features: most-frequent imputation, then `OneHotEncoder(handle_unknown="ignore")`.
- Placement target: mapped to 0/1 only inside the training/evaluation script.
- The Flask API accepts original category labels such as `Low`, `Medium`, `High`, `No`, and `Yes`.

### Outputs

- `models/*.joblib` contains the full preprocessing + model pipeline.
- `data/splits/*.csv` keeps cleaned source-form features.
- Encoded copies in `data/processed/` are used only for EDA and reporting.

### Important modeling practice

The train/validation/test split happens before any model-specific encoding. Imputation, scaling, and one-hot encoding are fitted inside each training pipeline and refitted inside each cross-validation fold.

### Limitation

One-hot encoding creates more columns than the original feature table, so the transformed feature count differs from the raw column count. This is expected and is reported in the PCA summary.

**Suggested visual:** Pipeline diagram: raw split -> imputer -> scaler/one-hot -> model.

---

## Slide 9: Exploratory Data Analysis and Visualization

### EDA performed

- Checked final dataset dimensions.
- Checked null values and duplicate rows.
- Calculated descriptive statistics.
- Examined the Exam_Score distribution.
- Examined Placement class distribution.
- Generated a correlation heatmap for numeric variables.

### Current processed tracks

- Exam track: 6,378 rows, 22 encoded columns, 0 nulls, 0 duplicates.
- Placement track: 10,000 rows, 9 encoded columns, 0 nulls, 0 duplicates.
- Exam_Score mean: 67.25; standard deviation: 3.91.
- Placement class 0: 8,341 records.
- Placement class 1: 1,659 records.
- There is no 31-column combined dataset.

### Initial observations

- Placement is imbalanced, with class 1 representing 16.6% of the data.
- Attendance and Hours_Studied are among the strongest exam-score features.
- Communication_Skills, CGPA, Prev_Sem_Result, IQ, and Projects_Completed are the core placement features.
- The placement label is a deterministic rule based on four of those features.
- Class imbalance means accuracy alone is not enough for judging placement performance.

**Suggested visuals:** Use `outputs/eda/exam_correlation_heatmap.png`, `outputs/eda/placement_correlation_heatmap.png`, `exam_score_distribution.png`, and `placement_distribution.png`.

---

## Slide 10: Feature Sets and Preprocessing Pipelines

### Method used

The project uses a small, explainable feature set per task so the web form can
collect every required value. There is no global `SelectKBest` step in the
final pipeline.

### Exam_Score features

1. Hours_Studied
2. Attendance
3. Previous_Scores
4. Tutoring_Sessions
5. Access_to_Resources
6. Parental_Involvement
7. Parental_Education_Level
8. Family_Income

### Placement features

1. Communication_Skills
2. CGPA
3. Prev_Sem_Result
4. IQ
5. Projects_Completed
6. Extra_Curricular_Score
7. Academic_Performance
8. Internship_Experience

### Why pipelines matter

- Imputation and encoding are learned only from training rows.
- The same transformation is reused for validation, test, cross-validation, and live prediction.
- No encoded columns are created before the split.
- The saved `.joblib` artifact contains both preprocessing and the model.

### Caution

A statistical relationship does not prove causation. The feature set is a
modelling choice, not a claim that these are the only relevant factors.

**Suggested visual:** Pipeline diagram plus the two feature lists.

---

## Slide 11: Dataset Splitting and Model Training

### Split strategy

Each task was split separately into 70% / 15% / 15% with random state 42:

- Exam: 4,464 / 957 / 957 rows.
- Placement: 7,000 / 1,500 / 1,500 rows.
- Stratification used for the Placement classification track.

### Regression models: Exam_Score

- Linear Regression.
- Random Forest Regressor.

### Classification models: Placement

- Logistic Regression.
- Depth-limited Random Forest Classifier (`max_depth=2`).

### Evaluation metrics

Regression:
- MAE: average absolute prediction error.
- RMSE: penalizes larger errors more strongly.
- R²: proportion of target variation explained by the model.

Classification:
- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Confusion matrix.

### Validation design

- Five-fold cross-validation on each training split.
- Training, validation, and test metrics reported separately.
- Train-versus-test gap recorded for every model.

### Reproducibility commands

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

---

## Slide 12: Results, Validation, and Conclusion

### Exam_Score regression results

| Model | Test MAE | Test RMSE | Test R² | 5-fold CV MAE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.9190 | 2.1978 | 0.6974 | 0.9294 ± 0.0553 |
| Random Forest Regressor | 1.1522 | 2.3488 | 0.6544 | 1.1986 ± 0.0565 |

**Interpretation:** Linear Regression is the stronger baseline. The Random Forest shows the only overfitting warning: test R² is 0.1104 below training.

### Placement classification results

| Model | Accuracy | F1 | ROC-AUC | 5-fold CV accuracy |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8687 | 0.6888 | 0.9487 | 0.8514 ± 0.0126 |
| Random Forest Classifier (depth 2) | 0.9533 | 0.8768 | 0.9924 | 0.9471 ± 0.0032 |

**Interpretation:** The placement label is deterministic. An unrestricted forest reaches 1.0 because it recovers the rule, not because the project found real-world signal. The deployed forest is deliberately capacity-limited.

### Validation findings

- Target columns are absent from feature columns.
- Preprocessing is fitted inside each training pipeline and each CV fold.
- Placement splits showed no repeated feature combinations.
- Exam-score splits contain one repeated feature combination with conflicting targets across train and validation.
- No row-order or synthetic-pair dataset exists in this version.
- The missing student-level link is the main validity limitation.

### Conclusion

The team has completed a reproducible baseline workflow from data preparation through cross-validation, overfitting analysis, and deterministic-label investigation. The results are meaningful as an initial modeling study when the documented limitations are stated clearly.

**Suggested visuals:** Confusion matrix image, residual distribution image, and a compact metric comparison table.

---

## Slide 13: Project Planning and Future Scope

### Completed project foundation

- Dataset collection and preparation.
- Data quality checks.
- Missing-value treatment and bounded ranges.
- Pipeline-based imputation, scaling, and encoding.
- EDA and visual outputs.
- Reproducible train/validation/test splits.
- Baseline models for both tasks.
- Five-fold cross-validation and overfitting analysis.
- Placement deterministic-label investigation.
- Metrics, confusion matrices, residual analysis, and structural leakage checks.

### Remaining for the final project

1. Review the conflicting Exam_Score feature combination.
2. Improve the data strategy using genuinely linked student records if available.
3. Tune hyperparameters and compare stronger algorithms.
4. Build explainability outputs such as feature importance or SHAP analysis.
5. Add student risk categories and intervention recommendations.
6. Document fairness, privacy, limitations, and model monitoring considerations.
7. Prepare the complete report, presentation, and deployment demonstration.

### Final vision

A responsible student analytics tool that helps educators identify performance and placement support needs while clearly communicating uncertainty and model limitations.

### Data Sources and References

- StudentPerformanceFactors.csv, public Kaggle dataset: URL not recorded in repository
- college_student_placement_dataset.csv, public Kaggle dataset: URL not recorded in repository
- pandas documentation: https://pandas.pydata.org/docs/
- scikit-learn documentation: https://scikit-learn.org/stable/

The repository currently stores the CSV files but does not record their
original URLs. Add the source URLs if the team has them before presenting.

---

# Optional Backup Slide: Repository Evidence

### Main files and outputs

- `data/raw/`: original source datasets.
- `data/interim/`: independently cleaned source frames and provenance.
- `data/processed/`: encoded copies used by EDA and reporting.
- `data/splits/`: raw-feature train, validation, and test files.
- `scripts/`: checks, splitting, EDA, pipeline training, and analysis code.
- `outputs/eda/`: statistics and visualizations.
- `outputs/metrics/`: test metrics, cross-validation, overfitting, and model card.
- `outputs/analysis/`: placement investigation, confusion matrices, residuals, reports, and leakage checks.
- `models/`: saved full preprocessing + model `.joblib` pipelines.
- `docs/`: preprocessing and evaluation documentation.

### Reproduction

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts/preprocess.py
python scripts/dataset_checks.py
python scripts/select_and_split.py
python scripts/data_quality.py
python scripts/pca_analysis.py
python scripts/run_eda.py
python scripts/train_baselines.py
python scripts/analyze_baselines.py
```

---

# Presentation Notes

- Use one consistent color for the Exam_Score regression track and another for the Placement classification track.
- Prefer charts from `outputs/eda/` and `outputs/analysis/` instead of adding decorative graphics.
- Explain that the placement label is deterministic; an unrestricted forest's perfect score is a dataset artifact, not a real-world success.
- Say “the two source files have no shared student identifier” rather than implying that both datasets belong to the same students.
- Mention that no row-order or synthetic-pair dataset exists in this version.
- Keep slide text concise; use the speaker points and tables for explanation during the presentation.
