"""Flask API and web server for the Student Performance Analytics product.

The API reads only generated artifacts and saved pipelines. It never retrains a
model during a request. Prediction inputs are validated against the shared
field definitions in ``scripts/model_config.py``; model and artifact failures
return an explicit 503 instead of a misleading prediction.
"""
from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

from scripts.interventions import advisory_rules, build_interventions
from scripts.model_config import (
    EXAM_MODEL_FEATURES,
    MODEL_LABELS,
    PLACEMENT_MODEL_FEATURES,
    PREDICTION_FIELDS,
    TASK_META,
    fields_for,
)


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MODELS_DIR = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
EVALUATION_DIR = ROOT / "outputs" / "evaluation"
EXPLANATION_DIR = ROOT / "outputs" / "explanations"
QUALITY_DIR = ROOT / "outputs" / "quality"
PCA_DIR = ROOT / "outputs" / "pca"
CLUSTERING_DIR = ROOT / "outputs" / "clustering"
FAIRNESS_DIR = ROOT / "outputs" / "fairness"
ANALYSIS_DIR = ROOT / "outputs" / "analysis"

METRICS_FILE = METRICS_DIR / "baseline_metrics.csv"
CV_FILE = METRICS_DIR / "cross_validation.json"
OVERFITTING_FILE = METRICS_DIR / "overfitting_analysis.json"
MODEL_CARD_FILE = METRICS_DIR / "model_card.json"
MODEL_PERFORMANCE_FILE = EVALUATION_DIR / "model_performance.json"
EVALUATION_CV_FILE = EVALUATION_DIR / "cross_validation_results.json"
EVALUATION_OVERFITTING_FILE = EVALUATION_DIR / "overfitting_report.json"
LEAKAGE_REPORT_FILE = EVALUATION_DIR / "leakage_report.json"
EXPLANATION_SUMMARY_FILE = EXPLANATION_DIR / "explanation_summary.json"
EXAM_FILE = ROOT / "data" / "processed" / "exam_score_preprocessed.csv"
PLACEMENT_FILE = ROOT / "data" / "processed" / "placement_preprocessed.csv"
QUALITY_FILE = QUALITY_DIR / "data_quality.json"
PCA_FILE = PCA_DIR / "pca_summary.json"
PCA_IMAGE = PCA_DIR / "pca_visualization.png"
CLUSTER_FILE = CLUSTERING_DIR / "cluster_summary.json"
CLUSTER_IMAGE = CLUSTERING_DIR / "cluster_visualization.png"
FAIRNESS_FILE = FAIRNESS_DIR / "fairness_summary.json"
PROVENANCE_FILE = ROOT / "data" / "interim" / "provenance.json"
PLACEMENT_INVESTIGATION_FILE = ANALYSIS_DIR / "placement_investigation.json"

ADVISORY_NOTE = (
    "Suggestions are transparent advisory prompts, not model output, causal "
    "claims, or institutional decisions."
)

app = Flask(
    __name__,
    template_folder="web",
    static_folder="web",
    static_url_path="/assets",
)
app.json.sort_keys = False


class InputError(ValueError):
    """A client input failed validation."""

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class ArtifactUnavailable(RuntimeError):
    """A required generated artifact is missing or unreadable."""


_JSON_CACHE: dict[Path, tuple[int, int, Any]] = {}
_MODEL_CACHE: dict[str, tuple[int, int, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _read_json(path: Path):
    """Read a generated JSON artifact with a small mtime-aware cache."""
    if not path.exists():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    with _CACHE_LOCK:
        cached = _JSON_CACHE.get(path)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    with _CACHE_LOCK:
        _JSON_CACHE[path] = (stat.st_mtime_ns, stat.st_size, value)
    return value


def _load_model(track: str):
    """Load a saved pipeline once per artifact version."""
    meta = TASK_META[track]
    path = MODELS_DIR / meta["artifact"]
    if not path.exists():
        raise ArtifactUnavailable(
            f"Missing {meta['artifact']}. Run python scripts/train_models.py first."
        )
    try:
        stat = path.stat()
    except OSError as exc:
        raise ArtifactUnavailable(f"Could not read {meta['artifact']}: {exc}") from exc
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(track)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]
    try:
        model = joblib.load(path)
    except Exception as exc:  # joblib raises several backend-specific errors
        raise ArtifactUnavailable(
            f"Could not load {meta['artifact']}: {exc}"
        ) from exc
    with _CACHE_LOCK:
        _MODEL_CACHE[track] = (stat.st_mtime_ns, stat.st_size, model)
    return model


def _prediction_payload() -> dict:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise InputError("Request body must be a JSON object.")
    return payload


def _number_field(field: dict, payload: dict) -> float:
    key = field["name"]
    label = field["label"]
    value = payload.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise InputError(f"{label} is required.", key)
    if isinstance(value, bool):
        raise InputError(f"{label} must be a number.", key)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{label} must be a number.", key) from exc
    if not math.isfinite(number):
        raise InputError(f"{label} must be a finite number.", key)
    minimum = float(field["min"])
    maximum = float(field["max"])
    if not minimum <= number <= maximum:
        raise InputError(
            f"{label} must be between {minimum:g} and {maximum:g}.", key
        )
    return number


def _category_field(field: dict, payload: dict) -> str:
    key = field["name"]
    label = field["label"]
    allowed = list(field["allowed"])
    value = payload.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise InputError(f"{label} is required.", key)
    text = str(value).strip()
    for candidate in allowed:
        if text.lower() == candidate.lower():
            return candidate
    try:
        index = int(text)
    except (TypeError, ValueError) as exc:
        raise InputError(
            f"{label} must be one of: {', '.join(allowed)}.", key
        ) from exc
    if 0 <= index < len(allowed):
        return allowed[index]
    raise InputError(f"{label} must be one of: {', '.join(allowed)}.", key)


def _validated_inputs(track: str, payload: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a track payload and return model values plus advisory values."""
    model_values: dict[str, Any] = {}
    advisory_values: dict[str, Any] = {}
    for field in fields_for(track):
        if field["type"] == "number":
            number = _number_field(field, payload)
            model_values[field["column"]] = number
            advisory_values[field["name"]] = number
        else:
            model_values[field["column"]] = _category_field(field, payload)
    return model_values, advisory_values


def _metrics_records() -> list[dict]:
    if not METRICS_FILE.exists():
        return []
    try:
        frame = pd.read_csv(METRICS_FILE)
    except (OSError, ValueError):
        return []
    frame = frame.astype(object).where(lambda values: pd.notna(values), None)
    return frame.to_dict("records")


METRICS = _metrics_records()


def _legacy_metric(task: str, model: str, key: str):
    for row in METRICS:
        if row.get("task") == task and row.get("model") == model:
            value = row.get(key)
            return float(value) if value is not None else None
    return None


def _deployed_test_metric(track: str, metric: str):
    """Read the deployed model's held-out metric from the current artifacts."""
    meta = TASK_META[track]
    report = _read_json(MODEL_PERFORMANCE_FILE)
    if report:
        for record in report.get("models", []):
            if record.get("task") == meta["task"] and record.get("deployed"):
                value = record.get("metrics", {}).get("test", {}).get(metric)
                if value is not None:
                    return float(value)
    legacy_task = "exam_score_regression" if track == "exam" else "placement_classification"
    legacy_model = "LinearRegression" if track == "exam" else "RandomForestClassifier"
    return _legacy_metric(legacy_task, legacy_model, metric)


def _model_label(track: str) -> str:
    card = _read_json(MODEL_CARD_FILE) or {}
    deployed = card.get("deployed", {}).get(track, {})
    return deployed.get("label") or TASK_META[track]["model_label"] or MODEL_LABELS[track]


def _linkage_status() -> dict:
    """Return generated linkage facts, never a hand-written claim."""
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


def _artifact_status() -> dict[str, bool]:
    return {
        "exam_data": EXAM_FILE.exists(),
        "placement_data": PLACEMENT_FILE.exists(),
        "exam_model": (MODELS_DIR / TASK_META["exam"]["artifact"]).exists(),
        "placement_model": (MODELS_DIR / TASK_META["placement"]["artifact"]).exists(),
        "metrics": METRICS_FILE.exists(),
        "model_performance": MODEL_PERFORMANCE_FILE.exists(),
        "cross_validation": EVALUATION_CV_FILE.exists() or CV_FILE.exists(),
        "overfitting": EVALUATION_OVERFITTING_FILE.exists() or OVERFITTING_FILE.exists(),
        "feature_importance": EXPLANATION_SUMMARY_FILE.exists(),
        "leakage_report": LEAKAGE_REPORT_FILE.exists(),
        "placement_investigation": PLACEMENT_INVESTIGATION_FILE.exists(),
        "data_quality": QUALITY_FILE.exists(),
        "pca_summary": PCA_FILE.exists(),
        "pca_visualization": PCA_IMAGE.exists(),
        "clusters": CLUSTER_FILE.exists(),
        "cluster_visualization": CLUSTER_IMAGE.exists(),
        "fairness": FAIRNESS_FILE.exists(),
        "provenance": PROVENANCE_FILE.exists(),
        "model_card": MODEL_CARD_FILE.exists(),
    }


def _generated_json(path: Path, instruction: str):
    report = _read_json(path)
    if report is None:
        return jsonify({"error": instruction}), 503
    return jsonify(report)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/llms.txt")
def llms():
    path = WEB_DIR / "llms.txt"
    if not path.exists():
        return jsonify({"error": "llms.txt is not available."}), 404
    return send_file(path, mimetype="text/plain")


@app.get("/api/health")
def health():
    artifacts = _artifact_status()
    core_ready = all(
        artifacts[key]
        for key in ("exam_data", "placement_data", "exam_model", "placement_model")
    )
    return jsonify(
        {
            "status": "ok" if core_ready else "degraded",
            "core_ready": core_ready,
            "models": {
                track: {
                    "ready": artifacts[f"{track}_model"],
                    "artifact": TASK_META[track]["artifact"],
                    "label": _model_label(track),
                }
                for track in TASK_META
            },
            "artifacts": artifacts,
            "prediction_tracks": list(TASK_META),
        }
    )


@app.get("/api/summary")
def summary():
    try:
        exam = pd.read_csv(EXAM_FILE)
        placement = pd.read_csv(PLACEMENT_FILE)
        placement_counts = placement["Placement"].value_counts().to_dict()
        exam_mean = float(exam["Exam_Score"].mean())
        exam_std = float(exam["Exam_Score"].std())
        exam_min = int(exam["Exam_Score"].min())
        exam_max = int(exam["Exam_Score"].max())
        placement_rate = float(placement["Placement"].mean())
    except (OSError, ValueError, KeyError) as exc:
        return (
            jsonify(
                {
                    "error": (
                        "Processed datasets are unavailable. Run python "
                        f"scripts/preprocess.py first. ({exc})"
                    )
                }
            ),
            503,
        )

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
            "exam_mean": round(exam_mean, 2),
            "exam_std": round(exam_std, 2),
            "exam_min": exam_min,
            "exam_max": exam_max,
            "placement_unplaced": int(placement_counts.get(0, 0)),
            "placement_placed": int(placement_counts.get(1, 0)),
            "placement_rate": round(placement_rate, 4),
            "exam_features": EXAM_MODEL_FEATURES,
            "placement_features": PLACEMENT_MODEL_FEATURES,
            "deployed_models": {
                "exam": _model_label("exam"),
                "placement": _model_label("placement"),
            },
            "exam_model_test_rmse": _deployed_test_metric("exam", "RMSE"),
            **linkage,
            "datasets_are_independent": not linkage["source_records_genuinely_linked"],
        }
    )


@app.get("/api/metrics")
def metrics():
    if not METRICS:
        return jsonify({"error": "Run python scripts/train_baselines.py first."}), 503
    return jsonify(METRICS)


@app.get("/api/model-performance")
def model_performance():
    return _generated_json(
        MODEL_PERFORMANCE_FILE,
        "Run python scripts/train_models.py first.",
    )


@app.get("/api/cross-validation")
def cross_validation():
    report = _read_json(EVALUATION_CV_FILE)
    if report is not None:
        return jsonify(report)
    legacy = _read_json(CV_FILE)
    if legacy is None:
        return jsonify(
            {"error": "Run python scripts/cross_validation.py first."}
        ), 503
    return jsonify({"legacy_format": True, "results": legacy})


@app.get("/api/overfitting")
def overfitting():
    report = _read_json(EVALUATION_OVERFITTING_FILE)
    if report is not None:
        return jsonify(report)
    legacy = _read_json(OVERFITTING_FILE)
    if legacy is None:
        return jsonify(
            {"error": "Run python scripts/overfitting_analysis.py first."}
        ), 503
    return jsonify({"legacy_format": True, **legacy})


@app.get("/api/feature-importance")
def feature_importance():
    return _generated_json(
        EXPLANATION_SUMMARY_FILE,
        "Run python scripts/explain_models.py first.",
    )


@app.get("/api/leakage")
def leakage():
    return _generated_json(
        LEAKAGE_REPORT_FILE,
        "Run python scripts/leakage_investigation.py first.",
    )


@app.get("/api/placement-investigation")
def placement_investigation():
    return _generated_json(
        PLACEMENT_INVESTIGATION_FILE,
        "Run python scripts/train_baselines.py first.",
    )


@app.get("/api/model-card")
def model_card():
    report = _read_json(MODEL_CARD_FILE)
    if report is None:
        return jsonify({"error": "Run python scripts/train_baselines.py first."}), 503
    return jsonify(report)


@app.get("/api/data-quality")
def data_quality():
    return _generated_json(
        QUALITY_FILE, "Run python scripts/data_quality.py first."
    )


@app.get("/api/pca")
def pca():
    return _generated_json(PCA_FILE, "Run python scripts/pca_analysis.py first.")


@app.get("/api/pca/visualization")
def pca_visualization():
    if not PCA_IMAGE.exists():
        return jsonify({"error": "Run python scripts/pca_analysis.py first."}), 503
    return send_file(PCA_IMAGE, mimetype="image/png")


@app.get("/api/clusters")
def clusters():
    return _generated_json(
        CLUSTER_FILE, "Run python scripts/clustering.py first."
    )


@app.get("/api/clusters/visualization")
def cluster_visualization():
    if not CLUSTER_IMAGE.exists():
        return jsonify({"error": "Run python scripts/clustering.py first."}), 503
    return send_file(CLUSTER_IMAGE, mimetype="image/png")


@app.get("/api/fairness")
def fairness():
    return _generated_json(
        FAIRNESS_FILE, "Run python scripts/fairness_analysis.py first."
    )


@app.get("/api/prediction-schema")
def prediction_schema():
    tracks = {}
    for track, meta in TASK_META.items():
        fields = fields_for(track)
        tracks[track] = {
            **meta,
            "fields": fields,
            "feature_order": [field["column"] for field in fields],
        }
    return jsonify(
        {
            "tracks": tracks,
            "exam": fields_for("exam"),
            "placement": fields_for("placement"),
        }
    )


@app.get("/api/advisory-rules")
def advisory_rule_list():
    return jsonify({"rules": advisory_rules(), "advisory_only": True, "note": ADVISORY_NOTE})


@app.post("/api/interventions")
def interventions():
    try:
        payload = _prediction_payload()
        fields: dict[str, dict] = {}
        for track in PREDICTION_FIELDS:
            for field in fields_for(track):
                if field["type"] == "number":
                    fields.setdefault(field["name"], field)
        values: dict[str, float] = {}
        for name, field in fields.items():
            if name in payload and payload[name] is not None:
                if isinstance(payload[name], str) and not payload[name].strip():
                    continue
                values[name] = _number_field(field, payload)
    except InputError as exc:
        return jsonify({"error": str(exc), "field": exc.field}), 400
    return jsonify(
        {
            "suggestions": build_interventions(values),
            "provided_fields": sorted(values),
            "advisory_only": True,
            "rules": advisory_rules(),
            "note": ADVISORY_NOTE,
        }
    )


@app.post("/api/predict/exam")
def predict_exam():
    try:
        payload = _prediction_payload()
        values, advisory_values = _validated_inputs("exam", payload)
        model = _load_model("exam")
        frame = pd.DataFrame(
            [values], columns=[field["column"] for field in fields_for("exam")]
        )
        prediction = float(model.predict(frame)[0])
    except InputError as exc:
        return jsonify({"error": str(exc), "field": exc.field}), 400
    except ArtifactUnavailable as exc:
        return jsonify({"error": f"Prediction unavailable: {exc}"}), 503

    score = max(0.0, min(100.0, prediction))
    rmse = _deployed_test_metric("exam", "RMSE")
    interval = None
    interval_basis = None
    if rmse is not None:
        interval = [
            round(max(0.0, score - rmse), 2),
            round(min(100.0, score + rmse), 2),
        ]
        interval_basis = f"Approximate +/- 1 held-out RMSE ({rmse:.3f})"
    tier = "Strong" if score >= 75 else "On track" if score >= 50 else "Needs support"
    return jsonify(
        {
            "track": "exam",
            "task": "regression",
            "score": round(score, 2),
            "rounded_score": round(score),
            "interval": interval,
            "interval_basis": interval_basis,
            "interval_note": (
                "This is an approximate error band from the held-out test set, "
                "not a confidence interval or a guarantee."
            ),
            "tier": tier,
            "model": _model_label("exam"),
            "limitations": TASK_META["exam"]["limitations"],
            "advisory_suggestions": build_interventions(advisory_values),
            "advisory_note": ADVISORY_NOTE,
        }
    )


@app.post("/api/predict/placement")
def predict_placement():
    try:
        payload = _prediction_payload()
        values, advisory_values = _validated_inputs("placement", payload)
        model = _load_model("placement")
        frame = pd.DataFrame(
            [values],
            columns=[field["column"] for field in fields_for("placement")],
        )
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0, 1])
    except InputError as exc:
        return jsonify({"error": str(exc), "field": exc.field}), 400
    except ArtifactUnavailable as exc:
        return jsonify({"error": f"Prediction unavailable: {exc}"}), 503

    linkage = _linkage_status()
    return jsonify(
        {
            "track": "placement",
            "task": "classification",
            "placed": bool(prediction),
            "probability": round(probability, 4),
            "percentage": round(probability * 100, 1),
            "decision_threshold": 0.5,
            "model": _model_label("placement"),
            "source_track": "placement",
            "exam_records_linked": linkage["source_records_genuinely_linked"],
            "warning": (
                "This result comes from the independent placement dataset. "
                "The placement label is deterministic in this source, so high "
                "scores reflect dataset construction, not real-world accuracy. "
                "The data are synthetic and experimental."
            ),
            "probability_note": (
                "This is the model's output at a 0.5 threshold. It is not a "
                "calibrated probability and not a guarantee of placement."
            ),
            "limitations": TASK_META["placement"]["limitations"],
            "advisory_suggestions": build_interventions(advisory_values),
            "advisory_note": ADVISORY_NOTE,
        }
    )


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found."}), 404
    return ("Not found", 404)


@app.errorhandler(405)
def method_not_allowed(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Method not allowed for this endpoint."}), 405
    return ("Method not allowed", 405)


@app.errorhandler(500)
def internal_error(error):
    app.logger.error("Unhandled server error: %s", error)
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error."}), 500
    return ("Internal server error", 500)


@app.after_request
def security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    return response


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
