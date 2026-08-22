"""
Step 6: Feature selection
Step 7: Train / Validation / Test split

Done as two SEPARATE tracks because Exam_Score and Placement come from
two different original datasets with no real relationship between them.
"""
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.model_selection import train_test_split

df = pd.read_csv("final_preprocessed_dataset.csv")

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
    # --- Feature selection: keep the k features most related to the target ---
    score_func = f_regression if task == "regression" else f_classif
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)

    scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
    selected_features = scores.head(k).index.tolist()
    print(f"\nTop {k} features for {task} target (by statistical score):")
    print(scores.head(k).round(2))

    X_selected = X[selected_features]

    # --- Split: 70% train, 15% validation, 15% test ---
    stratify_arg = y if task == "classification" else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_selected, y, test_size=0.30, random_state=42, stratify=stratify_arg
    )
    stratify_arg2 = y_temp if task == "classification" else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=stratify_arg2
    )

    print(f"Train: {X_train.shape[0]} rows | Validation: {X_val.shape[0]} rows | Test: {X_test.shape[0]} rows")
    return X_train, X_val, X_test, y_train, y_val, y_test, selected_features


# ---- Track A: Exam_Score (regression) ----
X_perf = df[PERFORMANCE_FEATURES]
y_perf = df["Exam_Score"]
Xa_train, Xa_val, Xa_test, ya_train, ya_val, ya_test, feats_a = select_and_split(
    X_perf, y_perf, task="regression", k=8
)

# ---- Track B: Placement (classification) ----
X_place = df[PLACEMENT_FEATURES]
y_place = df["Placement"]
Xb_train, Xb_val, Xb_test, yb_train, yb_val, yb_test, feats_b = select_and_split(
    X_place, y_place, task="classification", k=8
)

# ---- Save all splits ----
Xa_train.assign(Exam_Score=ya_train).to_csv("exam_score_train.csv", index=False)
Xa_val.assign(Exam_Score=ya_val).to_csv("exam_score_val.csv", index=False)
Xa_test.assign(Exam_Score=ya_test).to_csv("exam_score_test.csv", index=False)

Xb_train.assign(Placement=yb_train).to_csv("placement_train.csv", index=False)
Xb_val.assign(Placement=yb_val).to_csv("placement_val.csv", index=False)
Xb_test.assign(Placement=yb_test).to_csv("placement_test.csv", index=False)

print("\nSaved 6 files: exam_score_{train,val,test}.csv and placement_{train,val,test}.csv")
