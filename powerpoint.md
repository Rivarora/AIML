# Student Performance Analytics
## PowerPoint Content Draft

> Copy the slide titles and bullet points into PowerPoint. Replace the bracketed placeholders with university, team, and member details. Use the suggested visual on each slide where possible.

---

## Slide 1: Title

### Artificial Intelligence & Machine Learning (24CSE0316)

**Student Performance Analytics**

Predicting Exam Performance and Placement Outcomes Using Machine Learning

Chitkara University Institute of Engineering and Technology  
Chitkara University, Punjab, India  
Department of Computer Science & Engineering

**Team Details**
- Team members: [Add names and roll numbers]
- Faculty mentor: [Add name]
- Evaluation: [Add evaluation name/date]

**Speaker point:** This is a two-track machine learning project: one regression problem for Exam_Score and one classification problem for Placement.

---

## Slide 2: Project Overview

### What the project does

- Analyzes student academic, behavioral, and skill-related information.
- Predicts **Exam_Score** as a continuous value from 0 to 100.
- Predicts **Placement** as a binary outcome.
- Uses two separate prediction tracks because the source datasets do not contain a shared student identifier.
- Uses reproducible data preparation, EDA, feature selection, splitting, baseline modeling, and post-training validation.

### Datasets

- StudentPerformanceFactors.csv: 6,607 original records and 20 columns.
- college_student_placement_dataset.csv: 10,000 original records and 10 columns.
- Final model-ready paired dataset: 6,378 records and 31 columns after cleaning and preparation.

### Current project stage

- Data pipeline completed.
- EDA outputs generated.
- Baseline models trained.
- Confusion matrices, residual analysis, and structural leakage checks completed.
- Advanced tuning, repeated validation, explainability, and deployment remain for the final phase.

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

The two source datasets are not linked records from the same students. They were processed as separate target tracks and combined row-wise for the working dataset. Therefore, results demonstrate a modeling workflow and should not yet be treated as real-world causal or deployment-ready predictions.

---

## Slide 4: Objectives

1. Collect two student-related datasets covering academic performance and placement outcomes.
2. Inspect the datasets for missing values, duplicates, invalid values, and irrelevant identifiers.
3. Clean and standardize the available data while preserving valid records.
4. Encode categorical and ordinal variables into numeric form for machine learning.
5. Perform exploratory data analysis and create visual summaries.
6. Select relevant features separately for the regression and classification tasks.
7. Create reproducible training, validation, and test partitions.
8. Train baseline models for both targets.
9. Evaluate the models using task-appropriate metrics.
10. Check for structural leakage, repeated feature combinations, and suspicious results.
11. Document limitations and define the next phase of the project.

### Current project foundation

The completed baseline demonstrates a reproducible pipeline from raw datasets to analysis-ready model results.

---

## Slide 5: Dataset Collection and Description

### Dataset 1: StudentPerformanceFactors.csv

- **Source:** Public Kaggle educational dataset; add the exact dataset URL used by the team in the final slide.
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

- **Source:** Public Kaggle student placement dataset; add the exact dataset URL used by the team in the final slide.
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
- The project keeps the two prediction tracks logically separate.
- The working combined dataset uses the available rows from the performance dataset and row-wise pairing for the current experiment; this is documented as a limitation.

**Suggested visual:** Two dataset cards with target, task type, row count, and sample features.

### Source citation to display on the slide

```text
Data sources: Public Kaggle datasets
1. StudentPerformanceFactors.csv - [insert exact Kaggle URL]
2. college_student_placement_dataset.csv - [insert exact Kaggle URL]
Accessed: [insert access date]
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
- The cleaned performance dataset retains the available student records instead of deleting rows unnecessarily.

**Suggested visual:** Before/after table showing issue, treatment, and result.

---

## Slide 7: Missing-Value Treatment

### Method: Mean imputation for ordinal variables

The affected categorical/ordinal values were first mapped to numeric levels:

- Low = 0
- Medium = 1
- High = 2

For each affected column:

1. Convert known ordinal categories to numeric values.
2. Calculate the mean of the available values.
3. Round the mean to the nearest valid category level.
4. Fill missing cells with that category level.
5. Verify the result using `isnull().sum().sum()`.

### Example: Teacher_Quality

- Known-value sum: 7,819.
- Known-value count: 6,529.
- Mean: approximately 1.198.
- Rounded replacement: 1, representing Medium.

### Why this method was used

- Preserved records instead of dropping rows.
- Maintained the ordinal order of categories.
- Produced a complete numeric dataset for later encoding and modeling.

### Limitation

Mean imputation can reduce natural variation and may not represent every individual student accurately. Future work can compare it with mode imputation, model-based imputation, or a preprocessing pipeline that learns imputation values only from training data.

**Suggested visual:** A small flow diagram: ordinal category -> numeric mapping -> mean -> rounded category -> completed data.

---

## Slide 8: Data Transformation and Categorical Encoding

### Why encoding is required

Most machine learning algorithms require numeric input. Text categories such as Low, Medium, High, Yes, No, and school-type labels must be represented numerically.

### Transformations used in the prepared data

- Ordinal categories were mapped in their natural order, for example Low < Medium < High.
- Binary categories were represented as numeric indicators.
- Placement was represented as a binary target.
- The final processed file contains numeric feature columns suitable for scikit-learn models.

### Outputs

- `data/processed/combined_encoded.csv`
- `data/processed/final_preprocessed_dataset.csv`

### Important modeling practice

Feature selection is fitted after the train/validation/test split and uses the training partition only. This prevents validation and test target information from influencing the selected features.

### Limitation and future improvement

The current repository contains prepared encoded datasets, but the full preprocessing transformation should eventually be wrapped in a reusable `Pipeline` or `ColumnTransformer` so that training and future predictions use exactly the same transformations.

**Suggested visual:** Before/after sample table showing categorical values and their encoded values.

---

## Slide 9: Exploratory Data Analysis and Visualization

### EDA performed

- Checked final dataset dimensions.
- Checked null values and duplicate rows.
- Calculated descriptive statistics.
- Examined the Exam_Score distribution.
- Examined Placement class distribution.
- Generated a correlation heatmap for numeric variables.

### Current final working dataset

- Rows: 6,378.
- Columns: 31.
- Total null values: 0.
- Duplicate rows: 0.
- Exam_Score mean: 67.252.
- Exam_Score standard deviation: 3.913.
- Placement class 0: 5,319 records.
- Placement class 1: 1,059 records.

### Initial observations

- Placement is imbalanced, with class 1 representing a smaller portion of the data.
- Attendance and Hours_Studied are among the strongest selected factors for Exam_Score.
- Communication_Skills, CGPA, Prev_Sem_Result, and IQ are among the strongest selected factors for Placement.
- Class imbalance means accuracy alone is not enough for judging placement performance.

**Suggested visuals:** Use `outputs/eda/correlation_heatmap.png`, `exam_score_distribution.png`, and `placement_distribution.png`.

---

## Slide 10: Feature Selection Engineering

### Method used: SelectKBest

Feature selection was performed separately for each target:

- Regression: `f_regression`.
- Classification: `f_classif`.
- Selected features: top 8 features for each task.
- Feature rankings were learned from training data only.

### Top Exam_Score features

1. Attendance
2. Hours_Studied
3. Previous_Scores
4. Parental_Involvement
5. Access_to_Resources
6. Tutoring_Sessions
7. Parental_Education_Level
8. Family_Income

### Top Placement features

1. Communication_Skills
2. CGPA
3. Prev_Sem_Result
4. IQ
5. Projects_Completed
6. Extra_Curricular_Score
7. Peer_Positive
8. Academic_Performance

### Why feature selection matters

- Reduces the number of input variables.
- Makes the model easier to interpret.
- Focuses baseline models on the strongest statistical signals.
- Helps reduce irrelevant-feature noise.

### Caution

A statistical relationship does not prove causation. The selected features should be evaluated again using cross-validation and domain reasoning.

**Suggested visual:** Two ranked horizontal bar charts, one for each target.

---

## Slide 11: Dataset Splitting and Model Training

### Split strategy

Each task was split separately into:

- Training: 70%, 4,464 rows.
- Validation: 15%, 957 rows.
- Test: 15%, 957 rows.
- Fixed random state: 42 for reproducibility.
- Stratification used for the Placement classification track.

### Regression models: Exam_Score

- Linear Regression.
- Random Forest Regressor.

### Classification models: Placement

- Logistic Regression.
- Random Forest Classifier.

### Evaluation metrics

Regression:
- MAE: average absolute prediction error.
- RMSE: penalizes larger errors more strongly.
- R2: proportion of target variation explained by the model.

Classification:
- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- Confusion matrix.

### Reproducibility commands

```powershell
python scripts/dataset_checks.py
python scripts/select_and_split.py
python scripts/run_eda.py
python scripts/train_baselines.py
python scripts/analyze_baselines.py
```

---

## Slide 12: Results, Validation, and Conclusion

### Exam_Score regression results

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Linear Regression | 0.9225 | 2.3378 | 0.6585 |
| Random Forest Regressor | 1.1977 | 2.5892 | 0.5812 |

**Interpretation:** Linear Regression is the stronger baseline on the current test split. Its mean residual is close to zero, but individual prediction errors still require inspection.

### Placement classification results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8924 | 0.7258 | 0.5660 | 0.6360 | 0.9408 |
| Random Forest Classifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Interpretation:** The perfect Random Forest result is suspicious and should not be presented as proven real-world performance. Logistic Regression is a more conservative baseline.

### Validation findings

- Target columns are absent from feature columns.
- Feature selection is performed on training data only.
- Placement splits showed no repeated encoded feature combinations.
- Exam-score splits contain one repeated feature combination with conflicting targets across train and validation.
- Structural checks cannot prove semantic leakage-free source data.
- Row-wise synthetic pairing is the main validity limitation.

### Conclusion

The team has completed a reproducible baseline workflow from data preparation through post-training analysis. The results are meaningful as an initial modeling study when the documented limitations are stated clearly.

**Suggested visuals:** Confusion matrix image, residual distribution image, and a compact metric comparison table.

---

## Slide 13: Project Planning and Future Scope

### Completed project foundation

- Dataset collection and preparation.
- Data quality checks.
- Missing-value treatment.
- Encoding and processed datasets.
- EDA and visual outputs.
- Training-only feature selection.
- Reproducible train/validation/test splits.
- Baseline models for both tasks.
- Metrics, confusion matrices, residual analysis, and structural leakage checks.

### Remaining for the final project

1. Repeat cross-validation and permutation testing, especially for Placement.
2. Review the conflicting Exam_Score feature combination.
3. Improve the data strategy using genuinely linked student records if available.
4. Tune hyperparameters and compare stronger algorithms.
5. Build explainability outputs such as feature importance or SHAP analysis.
6. Add a prediction pipeline that saves preprocessing and model together.
7. Build a simple Streamlit or web-based demonstration interface.
8. Add student risk categories and intervention recommendations.
9. Document fairness, privacy, limitations, and model monitoring considerations.
10. Prepare the complete report, presentation, and deployment demonstration.

### Final vision

A responsible student analytics tool that helps educators identify performance and placement support needs while clearly communicating uncertainty and model limitations.

### Data Sources and References

- StudentPerformanceFactors.csv, public Kaggle dataset: [insert exact URL]
- college_student_placement_dataset.csv, public Kaggle dataset: [insert exact URL]
- pandas documentation: https://pandas.pydata.org/docs/
- scikit-learn documentation: https://scikit-learn.org/stable/

Replace the two dataset placeholders with the exact Kaggle URLs used by the team before presenting. The repository currently stores the CSV files but does not record their original URLs.

---

# Optional Backup Slide: Repository Evidence

### Main files and outputs

- `data/raw/`: original source datasets.
- `data/interim/`: cleaned and combined intermediate data.
- `data/processed/`: encoded model-ready datasets.
- `data/splits/`: train, validation, and test files.
- `scripts/`: checks, splitting, EDA, training, and analysis code.
- `outputs/eda/`: statistics and visualizations.
- `outputs/metrics/`: baseline metrics.
- `outputs/analysis/`: confusion matrices, residuals, reports, and leakage checks.
- `models/`: saved `.joblib` model artifacts.
- `docs/`: preprocessing and evaluation documentation.

### Reproduction

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts/dataset_checks.py
python scripts/select_and_split.py
python scripts/run_eda.py
python scripts/train_baselines.py
python scripts/analyze_baselines.py
```

---

# Presentation Notes

- Use one consistent color for the Exam_Score regression track and another for the Placement classification track.
- Prefer charts from `outputs/eda/` and `outputs/analysis/` instead of adding decorative graphics.
- Do not describe the Random Forest placement score of 1.0000 as a confirmed real-world success.
- Say “the current working dataset uses row-wise synthetic pairing” rather than implying that both datasets belong to the same students.
- Keep slide text concise; use the speaker points and tables for explanation during the presentation.
