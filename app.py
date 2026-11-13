"""Flask API and web server for the Student Performance Analytics dashboard."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parent
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
PROCESSED_FILE = ROOT / "data" / "processed" / "final_preprocessed_dataset.csv"
METRICS_FILE = ROOT / "outputs" / "metrics" / "baseline_metrics.csv"

app = Flask(__name__, template_folder="web")


def _feature_names(prefix: str, target: str) -> list[str]:
    columns = pd.read_csv(SPLITS_DIR / f"{prefix}_train.csv", nrows=0).columns
    return [column for column in columns if column != target]


EXAM_FEATURES = _feature_names("exam_score", "Exam_Score")
PLACEMENT_FEATURES = _feature_names("placement", "Placement")
EXAM_MODEL = joblib.load(MODELS_DIR / "LinearRegression_exam_score.joblib")
PLACEMENT_MODEL = joblib.load(MODELS_DIR / "RandomForestClassifier_placement.joblib")


def _number(payload: dict, key: str, minimum: float, maximum: float) -> float:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{key} is required.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number.") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}.")
    return number


def _prediction_frame(values: dict[str, float], features: list[str]) -> pd.DataFrame:
    missing = [feature for feature in features if feature not in values]
    if missing:
        raise ValueError(f"Missing model features: {', '.join(missing)}.")
    return pd.DataFrame([[values[feature] for feature in features]], columns=features)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "models_loaded": True})


@app.get("/api/summary")
def summary():
    data = pd.read_csv(PROCESSED_FILE)
    placement_counts = data["Placement"].value_counts().to_dict()
    return jsonify(
        {
            "rows": int(len(data)),
            "columns": int(len(data.columns)),
            "nulls": int(data.isna().sum().sum()),
            "duplicates": int(data.duplicated().sum()),
            "exam_mean": round(float(data["Exam_Score"].mean()), 2),
            "exam_std": round(float(data["Exam_Score"].std()), 2),
            "exam_min": int(data["Exam_Score"].min()),
            "exam_max": int(data["Exam_Score"].max()),
            "placement_unplaced": int(placement_counts.get(0, 0)),
            "placement_placed": int(placement_counts.get(1, 0)),
            "placement_rate": round(float(data["Placement"].mean()), 4),
            "exam_features": EXAM_FEATURES,
            "placement_features": PLACEMENT_FEATURES,
        }
    )


@app.get("/api/metrics")
def metrics():
    frame = pd.read_csv(METRICS_FILE).astype(object).where(lambda values: pd.notna(values), None)
    return jsonify(frame.to_dict("records"))


@app.post("/api/predict/exam")
def predict_exam():
    payload = request.get_json(silent=True) or {}
    try:
        values = {
            "Attendance": _number(payload, "attendance", 0, 100),
            "Hours_Studied": _number(payload, "hours_studied", 0, 50),
            "Previous_Scores": _number(payload, "previous_scores", 0, 100),
            "Access_to_Resources": _number(payload, "access_to_resources", 0, 2),
            "Parental_Involvement": _number(payload, "parental_involvement", 0, 2),
            "Tutoring_Sessions": _number(payload, "tutoring_sessions", 0, 10),
            "Parental_Education_Level": _number(payload, "parental_education_level", 0, 2),
            "Family_Income": _number(payload, "family_income", 0, 2),
        }
        prediction = float(EXAM_MODEL.predict(_prediction_frame(values, EXAM_FEATURES))[0])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    score = max(0.0, min(100.0, prediction))
    return jsonify(
        {
            "score": round(score, 2),
            "rounded_score": round(score),
            "interval": [round(max(0, score - 2.1981), 2), round(min(100, score + 2.1981), 2)],
            "tier": "Strong" if score >= 75 else "Average" if score >= 50 else "At risk",
            "model": "Linear Regression",
        }
    )


@app.post("/api/predict/placement")
def predict_placement():
    payload = request.get_json(silent=True) or {}
    try:
        values = {
            "CGPA": _number(payload, "cgpa", 0, 10),
            "Communication_Skills": _number(payload, "communication_skills", 1, 10),
            "Prev_Sem_Result": _number(payload, "prev_sem_result", 0, 10),
            "IQ": _number(payload, "iq", 70, 160),
            "Projects_Completed": _number(payload, "projects_completed", 0, 10),
            "Peer_Negative": _number(payload, "peer_negative", 0, 1),
            "Peer_Positive": _number(payload, "peer_positive", 0, 1),
            "Academic_Performance": _number(payload, "academic_performance", 0, 5),
        }
        frame = _prediction_frame(values, PLACEMENT_FEATURES)
        prediction = int(PLACEMENT_MODEL.predict(frame)[0])
        probability = float(PLACEMENT_MODEL.predict_proba(frame)[0, 1])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "placed": bool(prediction),
            "probability": round(probability, 4),
            "percentage": round(probability * 100, 1),
            "model": "Random Forest Classifier",
            "warning": "Experimental result from synthetically paired source datasets.",
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
