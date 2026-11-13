"""Flask API and web server for the Student Performance Analytics dashboard."""

from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from scripts.model_config import (
    EXAM_CATEGORICAL_FEATURES,
    EXAM_MODEL_FEATURES,
    MODEL_LABELS,
    PLACEMENT_CATEGORICAL_FEATURES,
    PLACEMENT_MODEL_FEATURES,
)


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
METRICS_FILE = METRICS_DIR / "baseline_metrics.csv"
CV_FILE = METRICS_DIR / "cross_validation.json"
OVERFITTING_FILE = METRICS_DIR / "overfitting_analysis.json"
MODEL_CARD_FILE = METRICS_DIR / "model_card.json"
EXAM_FILE = ROOT / "data" / "processed" / "exam_score_preprocessed.csv"
PLACEMENT_FILE = ROOT / "data" / "processed" / "placement_preprocessed.csv"
QUALITY_FILE = ROOT / "outputs" / "quality" / "data_quality.json"
PCA_FILE = ROOT / "outputs" / "pca" / "pca_summary.json"
PCA_IMAGE = ROOT / "outputs" / "pca" / "pca_visualization.png"
PROVENANCE_FILE = ROOT / "data" / "interim" / "provenance.json"
PLACEMENT_INVESTIGATION_FILE = (
    ROOT / "outputs" / "analysis" / "placement_investigation.json"
)

app = Flask(__name__, template_folder="web")


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


EXAM_FEATURES = EXAM_MODEL_FEATURES
PLACEMENT_FEATURES = PLACEMENT_MODEL_FEATURES
EXAM_MODEL = joblib.load(MODELS_DIR / "LinearRegression_exam_score.joblib")
PLACEMENT_MODEL = joblib.load(MODELS_DIR / "RandomForestClassifier_placement.joblib")
MODEL_CARD = _read_json(MODEL_CARD_FILE) or {}


def _metrics_records() -> list[dict]:
    if not METRICS_FILE.exists():
        return []
    frame = pd.read_csv(METRICS_FILE)
    frame = frame.astype(object).where(lambda values: pd.notna(values), None)
    return frame.to_dict("records")


METRICS = _metrics_records()


def _metric(task: str, model: str, key: str):
    for row in METRICS:
        if row.get("task") == task and row.get("model") == model:
            value = row.get(key)
            return float(value) if value is not None else None
    return None


EXAM_RMSE = _metric("exam_score_regression", "LinearRegression", "RMSE")


def _model_label(track: str) -> str:
    deployed = MODEL_CARD.get("deployed", {}).get(track, {})
    return deployed.get("label") or MODEL_LABELS[track]


def _linkage_status() -> dict:
    """Return the generated linkage facts, never a hand-written claim."""
    quality = _read_json(QUALITY_FILE)
    if quality and quality.get("linkage"):
        link = quality["linkage"]
        return {
            "source_records_genuinely_linked": bool(link.get("genuinely_linked", False)),
            "shared_student_identifier": link.get("shared_student_identifier"),
            "synthetic_pairing_used_in_modeling": bool(
                link.get("synthetic_pairing_used_in_modeling", False)
            ),
            "synthetic_artifact_available": bool(
                link.get("synthetic_artifact", {}).get("exists", False)
            ),
            "row_order_pairing_status": link.get(
                "row_order_pairing_status", "removed"
            ),
            "investigation": link.get("investigation", ""),
        }

    provenance = _read_json(PROVENANCE_FILE) or {}
    link = provenance.get("linkage", {})
    pairing = provenance.get("row_order_pairing", {})
    return {
        "source_records_genuinely_linked": bool(link.get("genuinely_linked", False)),
        "shared_student_identifier": link.get("shared_student_identifier"),
        "synthetic_pairing_used_in_modeling": False,
        "synthetic_artifact_available": False,
        "row_order_pairing_status": pairing.get("status", "removed"),
        "investigation": link.get("investigation", ""),
    }


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


def _category(payload: dict, key: str, allowed: list[str]) -> str:
    """Accept the category label, or a legacy 0-based index for convenience."""
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{key} is required.")
    text = str(value).strip()
    for candidate in allowed:
        if text.lower() == candidate.lower():
            return candidate
    try:
        index = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be one of: {', '.join(allowed)}.") from exc
    if 0 <= index < len(allowed):
        return allowed[index]
    raise ValueError(f"{key} must be one of: {', '.join(allowed)}.")


def _prediction_frame(values: dict[str, object], features: list[str]) -> pd.DataFrame:
    missing = [feature for feature in features if feature not in values]
    if missing:
        raise ValueError(f"Missing model features: {', '.join(missing)}.")
    return pd.DataFrame([[values[feature] for feature in features]], columns=features)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "models_loaded": True,
            "artifacts": {
                "metrics": METRICS_FILE.exists(),
                "cross_validation": CV_FILE.exists(),
                "overfitting": OVERFITTING_FILE.exists(),
                "placement_investigation": PLACEMENT_INVESTIGATION_FILE.exists(),
                "data_quality": QUALITY_FILE.exists(),
                "pca_summary": PCA_FILE.exists(),
                "pca_visualization": PCA_IMAGE.exists(),
                "provenance": PROVENANCE_FILE.exists(),
            },
        }
    )


@app.get("/api/summary")
def summary():
    exam = pd.read_csv(EXAM_FILE)
    placement = pd.read_csv(PLACEMENT_FILE)
    placement_counts = placement["Placement"].value_counts().to_dict()
    linkage = _linkage_status()
    return jsonify(
        {
            "exam_rows": int(len(exam)),
            "exam_columns": int(len(exam.columns)),
            "placement_rows": int(len(placement)),
            "placement_columns": int(len(placement.columns)),
            "exam_nulls": int(exam.isna().sum().sum()),
            "placement_nulls": int(placement.isna().sum().sum()),
            "exam_duplicates": int(exam.duplicated().sum()),
            "placement_duplicates": int(placement.duplicated().sum()),
            "exam_mean": round(float(exam["Exam_Score"].mean()), 2),
            "exam_std": round(float(exam["Exam_Score"].std()), 2),
            "exam_min": int(exam["Exam_Score"].min()),
            "exam_max": int(exam["Exam_Score"].max()),
            "placement_unplaced": int(placement_counts.get(0, 0)),
            "placement_placed": int(placement_counts.get(1, 0)),
            "placement_rate": round(float(placement["Placement"].mean()), 4),
            "exam_features": EXAM_FEATURES,
            "placement_features": PLACEMENT_FEATURES,
            "deployed_models": {
                "exam": _model_label("exam"),
                "placement": _model_label("placement"),
            },
            **linkage,
            "datasets_are_independent": not linkage["source_records_genuinely_linked"],
        }
    )


@app.get("/api/data-quality")
def data_quality():
    report = _read_json(QUALITY_FILE)
    if report is None:
        return jsonify({"error": "Run python scripts/data_quality.py first."}), 503
    return jsonify(report)


@app.get("/api/pca")
def pca():
    report = _read_json(PCA_FILE)
    if report is None:
        return jsonify({"error": "Run python scripts/pca_analysis.py first."}), 503
    return jsonify(report)


@app.get("/api/pca/visualization")
def pca_visualization():
    if not PCA_IMAGE.exists():
        return jsonify({"error": "Run python scripts/pca_analysis.py first."}), 503
    return send_file(PCA_IMAGE, mimetype="image/png")


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.get("/api/cross-validation")
def cross_validation():
    report = _read_json(CV_FILE)
    if report is None:
        return jsonify({"error": "Run python scripts/train_baselines.py first."}), 503
    return jsonify(report)


@app.get("/api/overfitting")
def overfitting():
    report = _read_json(OVERFITTING_FILE)
    if report is None:
        return jsonify({"error": "Run python scripts/train_baselines.py first."}), 503
    return jsonify(report)


@app.get("/api/placement-investigation")
def placement_investigation():
    report = _read_json(PLACEMENT_INVESTIGATION_FILE)
    if report is None:
        return jsonify({"error": "Run python scripts/train_baselines.py first."}), 503
    return jsonify(report)


@app.get("/api/model-card")
def model_card():
    if not MODEL_CARD:
        return jsonify({"error": "Run python scripts/train_baselines.py first."}), 503
    return jsonify(MODEL_CARD)


@app.post("/api/predict/exam")
def predict_exam():
    payload = request.get_json(silent=True) or {}
    try:
        values = {
            "Hours_Studied": _number(payload, "hours_studied", 0, 168),
            "Attendance": _number(payload, "attendance", 0, 100),
            "Previous_Scores": _number(payload, "previous_scores", 0, 100),
            "Tutoring_Sessions": _number(payload, "tutoring_sessions", 0, 100),
            "Access_to_Resources": _category(
                payload, "access_to_resources", EXAM_CATEGORICAL_FEATURES["Access_to_Resources"]
            ),
            "Parental_Involvement": _category(
                payload, "parental_involvement", EXAM_CATEGORICAL_FEATURES["Parental_Involvement"]
            ),
            "Parental_Education_Level": _category(
                payload,
                "parental_education_level",
                EXAM_CATEGORICAL_FEATURES["Parental_Education_Level"],
            ),
            "Family_Income": _category(
                payload, "family_income", EXAM_CATEGORICAL_FEATURES["Family_Income"]
            ),
        }
        prediction = float(EXAM_MODEL.predict(_prediction_frame(values, EXAM_FEATURES))[0])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    score = max(0.0, min(100.0, prediction))
    interval = None
    interval_basis = None
    if EXAM_RMSE is not None:
        interval = [round(max(0, score - EXAM_RMSE), 2), round(min(100, score + EXAM_RMSE), 2)]
        interval_basis = f"+/- 1 RMSE ({EXAM_RMSE:.4f})"
    return jsonify(
        {
            "score": round(score, 2),
            "rounded_score": round(score),
            "interval": interval,
            "interval_basis": interval_basis,
            "tier": "Strong" if score >= 75 else "Average" if score >= 50 else "At risk",
            "model": _model_label("exam"),
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
            "Academic_Performance": _number(payload, "academic_performance", 0, 10),
            "Extra_Curricular_Score": _number(payload, "extra_curricular_score", 0, 10),
            "Internship_Experience": _category(
                payload,
                "internship_experience",
                PLACEMENT_CATEGORICAL_FEATURES["Internship_Experience"],
            ),
        }
        frame = _prediction_frame(values, PLACEMENT_FEATURES)
        prediction = int(PLACEMENT_MODEL.predict(frame)[0])
        probability = float(PLACEMENT_MODEL.predict_proba(frame)[0, 1])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    linkage = _linkage_status()
    return jsonify(
        {
            "placed": bool(prediction),
            "probability": round(probability, 4),
            "percentage": round(probability * 100, 1),
            "model": _model_label("placement"),
            "source_track": "placement",
            "exam_records_linked": linkage["source_records_genuinely_linked"],
            "warning": (
                "This result comes from the independent placement dataset. "
                "The placement label is deterministic in this source, so high "
                "scores reflect dataset construction, not real-world accuracy."
            ),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
