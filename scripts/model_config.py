"""Shared model input definitions for training and the Flask API.

The models deliberately use small, explainable feature sets so the web forms
can collect every required value. Preprocessing for these features is fitted
inside the scikit-learn pipeline, not in the CSV preprocessing step.

This module is the single source of truth for:

* the selected model features and their categorical values,
* the API field names, labels, validity bounds, and example values,
* the deployed artifact for each track.

The Flask API, the training scripts, and the frontend schema endpoint all read
from these definitions. Changing a model feature therefore requires one edit,
not three.
"""
from __future__ import annotations

EXAM_TARGET = "Exam_Score"
PLACEMENT_TARGET = "Placement"

EXAM_MODEL_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Access_to_Resources",
    "Parental_Involvement",
    "Parental_Education_Level",
    "Family_Income",
]
EXAM_NUMERIC_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Previous_Scores",
    "Tutoring_Sessions",
]
EXAM_CATEGORICAL_FEATURES = {
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Family_Income": ["Low", "Medium", "High"],
}

PLACEMENT_MODEL_FEATURES = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Internship_Experience",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed",
]
PLACEMENT_NUMERIC_FEATURES = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed",
]
PLACEMENT_CATEGORICAL_FEATURES = {
    "Internship_Experience": ["No", "Yes"],
}

MODEL_LABELS = {
    "exam": "Linear Regression",
    "placement": "Random Forest Classifier (depth-limited)",
}

RANDOM_STATE = 42
CV_FOLDS = 5

# The bounds below match the validity ranges enforced by scripts/preprocess.py.
# They are intentionally wider than the observed data: they reject impossible
# values without rejecting plausible out-of-sample records.
PREDICTION_FIELDS: dict[str, list[dict]] = {
    "exam": [
        {
            "name": "hours_studied",
            "column": "Hours_Studied",
            "label": "Hours studied per week",
            "type": "number",
            "min": 0,
            "max": 168,
            "step": 1,
            "example": 20,
            "unit": "hours",
            "help": "Total study time in a typical week.",
        },
        {
            "name": "attendance",
            "column": "Attendance",
            "label": "Attendance",
            "type": "number",
            "min": 0,
            "max": 100,
            "step": 1,
            "example": 85,
            "unit": "%",
            "help": "Share of scheduled classes attended.",
        },
        {
            "name": "previous_scores",
            "column": "Previous_Scores",
            "label": "Previous scores",
            "type": "number",
            "min": 0,
            "max": 100,
            "step": 1,
            "example": 75,
            "unit": "points",
            "help": "Most recent prior assessment score.",
        },
        {
            "name": "tutoring_sessions",
            "column": "Tutoring_Sessions",
            "label": "Tutoring sessions",
            "type": "number",
            "min": 0,
            "max": 100,
            "step": 1,
            "example": 2,
            "unit": "sessions",
            "help": "Tutoring sessions attended in the period.",
        },
        {
            "name": "access_to_resources",
            "column": "Access_to_Resources",
            "label": "Access to resources",
            "type": "category",
            "allowed": EXAM_CATEGORICAL_FEATURES["Access_to_Resources"],
            "example": "Medium",
            "help": "Availability of learning materials and support.",
        },
        {
            "name": "parental_involvement",
            "column": "Parental_Involvement",
            "label": "Parental involvement",
            "type": "category",
            "allowed": EXAM_CATEGORICAL_FEATURES["Parental_Involvement"],
            "example": "Medium",
            "help": "Reported level of family engagement.",
        },
        {
            "name": "parental_education_level",
            "column": "Parental_Education_Level",
            "label": "Parental education",
            "type": "category",
            "allowed": EXAM_CATEGORICAL_FEATURES["Parental_Education_Level"],
            "example": "College",
            "help": "Highest reported parental education level.",
        },
        {
            "name": "family_income",
            "column": "Family_Income",
            "label": "Family income",
            "type": "category",
            "allowed": EXAM_CATEGORICAL_FEATURES["Family_Income"],
            "example": "Medium",
            "help": "Reported household income band.",
        },
    ],
    "placement": [
        {
            "name": "iq",
            "column": "IQ",
            "label": "IQ",
            "type": "number",
            "min": 0,
            "max": 250,
            "step": 1,
            "example": 115,
            "unit": "score",
            "help": "Recorded IQ score in the placement dataset.",
        },
        {
            "name": "prev_sem_result",
            "column": "Prev_Sem_Result",
            "label": "Previous semester result",
            "type": "number",
            "min": 0,
            "max": 10,
            "step": 0.01,
            "example": 8.0,
            "unit": "/ 10",
            "help": "Previous semester result on a 10-point scale.",
        },
        {
            "name": "cgpa",
            "column": "CGPA",
            "label": "CGPA",
            "type": "number",
            "min": 0,
            "max": 10,
            "step": 0.01,
            "example": 8.2,
            "unit": "/ 10",
            "help": "Cumulative grade point average.",
        },
        {
            "name": "academic_performance",
            "column": "Academic_Performance",
            "label": "Academic performance",
            "type": "number",
            "min": 0,
            "max": 10,
            "step": 1,
            "example": 8,
            "unit": "/ 10",
            "help": "Recorded academic performance score.",
        },
        {
            "name": "internship_experience",
            "column": "Internship_Experience",
            "label": "Internship experience",
            "type": "category",
            "allowed": PLACEMENT_CATEGORICAL_FEATURES["Internship_Experience"],
            "example": "No",
            "help": "Whether the record includes internship experience.",
        },
        {
            "name": "extra_curricular_score",
            "column": "Extra_Curricular_Score",
            "label": "Extra-curricular score",
            "type": "number",
            "min": 0,
            "max": 10,
            "step": 1,
            "example": 6,
            "unit": "/ 10",
            "help": "Recorded extra-curricular participation score.",
        },
        {
            "name": "communication_skills",
            "column": "Communication_Skills",
            "label": "Communication skills",
            "type": "number",
            "min": 0,
            "max": 10,
            "step": 1,
            "example": 8,
            "unit": "/ 10",
            "help": "Recorded communication skills score.",
        },
        {
            "name": "projects_completed",
            "column": "Projects_Completed",
            "label": "Projects completed",
            "type": "number",
            "min": 0,
            "max": 100,
            "step": 1,
            "example": 3,
            "unit": "projects",
            "help": "Number of completed practical projects.",
        },
    ],
}

TASK_META: dict[str, dict] = {
    "exam": {
        "label": "Academic performance",
        "short_label": "Exam score",
        "task": "exam_score",
        "target": EXAM_TARGET,
        "target_label": "Exam score",
        "task_type": "regression",
        "artifact": "LinearRegression_exam_score.joblib",
        "model_label": MODEL_LABELS["exam"],
        "description": (
            "Estimate an exam score from study habits, attendance, prior "
            "attainment, and reported support factors."
        ),
        "limitations": [
            "This is an experimental estimate from a baseline model, not a grade.",
            "The estimate is not guaranteed and has no causal interpretation.",
            "The exam dataset has no student identifier and is never linked to the placement dataset.",
        ],
    },
    "placement": {
        "label": "Placement readiness",
        "short_label": "Placement",
        "task": "placement",
        "target": PLACEMENT_TARGET,
        "target_label": "Placement outcome",
        "task_type": "classification",
        "artifact": "RandomForestClassifier_placement.joblib",
        "model_label": MODEL_LABELS["placement"],
        "description": (
            "Estimate a placement outcome from academic, communication, and "
            "experience signals in the independent placement dataset."
        ),
        "limitations": [
            "This probability is not a guarantee of placement or a hiring decision.",
            "The placement source is synthetic and its label is deterministic; high scores reflect dataset construction.",
            "The placement dataset is never linked to the exam dataset.",
        ],
    },
}


def fields_for(track: str) -> list[dict]:
    """Return a copy of the ordered API fields for one track."""
    try:
        return [dict(field) for field in PREDICTION_FIELDS[track]]
    except KeyError as exc:
        raise KeyError(f"Unknown prediction track: {track!r}") from exc


def field_names(track: str) -> list[str]:
    return [field["name"] for field in fields_for(track)]


def model_columns(track: str) -> list[str]:
    """Return the model DataFrame columns in registry order."""
    return [field["column"] for field in fields_for(track)]
