"""
Feature selection and train/validation/test splitting for two targets.

What this file does:
- Reads data/processed/final_preprocessed_dataset.csv
- Runs SelectKBest per target track
- Splits each track into 70/15/15 (train/val/test)
- Saves split files into data/splits/

Why two tracks:
- Exam_Score is a regression task
- Placement is a classification task
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "final_preprocessed_dataset.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

df = pd.read_csv(INPUT_FILE)

PERFORMANCE_FEATURES = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
    "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Physical_Activity", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender",
]
PLACEMENT_FEATURES = [
    "IQ", "Prev_Sem_Result", "CGPA", "Academic_Performance", "Internship_Experience",
    "Extra_Curricular_Score", "Communication_Skills", "Projects_Completed",
    "Peer_Negative", "Peer_Neutral", "Peer_Positive",
]


def select_and_split(X, y, task, k=8):
    # --- Split: 70% train, 15% validation, 15% test ---
    stratify_arg = y if task == "classification" else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=stratify_arg
    )
    stratify_arg2 = y_temp if task == "classification" else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=stratify_arg2
    )

    # --- Feature selection: learn rankings from training data only ---
    score_func = f_regression if task == "regression" else f_classif
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X_train, y_train)

    scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
    selected_features = scores.head(k).index.tolist()
    print(f"\nTop {k} features for {task} target (training data only):")
    print(scores.head(k).round(2))

    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    print(f"Train: {X_train.shape[0]} rows | Validation: {X_val.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_val, X_test, y_train, y_val, y_test, selected_features


def main() -> None:
    # ---- Track A: Exam_Score (regression) ----
    X_perf = df[PERFORMANCE_FEATURES]
    y_perf = df["Exam_Score"]
    Xa_train, Xa_val, Xa_test, ya_train, ya_val, ya_test, _ = select_and_split(
        X_perf, y_perf, task="regression", k=8
    )

    # ---- Track B: Placement (classification) ----
    X_place = df[PLACEMENT_FEATURES]
    y_place = df["Placement"]
    Xb_train, Xb_val, Xb_test, yb_train, yb_val, yb_test, _ = select_and_split(
        X_place, y_place, task="classification", k=8
    )

    # ---- Save all splits ----
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    Xa_train.assign(Exam_Score=ya_train).to_csv(SPLITS_DIR / "exam_score_train.csv", index=False)
    Xa_val.assign(Exam_Score=ya_val).to_csv(SPLITS_DIR / "exam_score_val.csv", index=False)
    Xa_test.assign(Exam_Score=ya_test).to_csv(SPLITS_DIR / "exam_score_test.csv", index=False)

    Xb_train.assign(Placement=yb_train).to_csv(SPLITS_DIR / "placement_train.csv", index=False)
    Xb_val.assign(Placement=yb_val).to_csv(SPLITS_DIR / "placement_val.csv", index=False)
    Xb_test.assign(Placement=yb_test).to_csv(SPLITS_DIR / "placement_test.csv", index=False)

    print("\nSaved 6 files: exam_score_{train,val,test}.csv and placement_{train,val,test}.csv")


if __name__ == "__main__":
    main()
