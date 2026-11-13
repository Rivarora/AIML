"""Shared model input definitions for training and the Flask API.

The model deliberately uses a small, explainable feature set so the web form
can collect every required value. Preprocessing for these features is fitted
inside the scikit-learn pipeline, not in the CSV preprocessing step.
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
