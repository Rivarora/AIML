"""Fit every registered supervised model and save held-out performance.

The goal is an honest comparison, not model selection by test score.  Each
pipeline is fitted once on the training split, then evaluated on the training,
validation, and test splits.  Cross-validation is handled separately by
``scripts/cross_validation.py`` so that the test split is never touched during
model fitting or model selection.

Outputs:
- models/<model>_<track>.joblib
- outputs/evaluation/model_performance.csv
- outputs/evaluation/model_performance.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from model_pipelines import (
    EVALUATION_DIR,
    MODELS_DIR,
    MODEL_SPECS,
    ModelSpec,
    build_pipeline,
    json_safe,
    load_task_frames,
    score_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERFORMANCE_CSV = EVALUATION_DIR / "model_performance.csv"
PERFORMANCE_JSON = EVALUATION_DIR / "model_performance.json"


def _flatten_model_row(
    spec: ModelSpec,
    rows: dict[str, dict[str, float]],
    split_sizes: dict[str, int],
) -> dict:
    row = {
        "task": spec.task,
        "model": spec.name,
        "preprocessing": spec.preprocessing,
        "deployed": bool(spec.deployed),
        "target": spec.target,
        "note": spec.note,
        "train_rows": split_sizes["train"],
        "val_rows": split_sizes["val"],
        "test_rows": split_sizes["test"],
    }
    for split in ("train", "val", "test"):
        for metric in spec.metrics:
            row[f"{split}_{metric}"] = rows[split][metric]
    for metric in spec.metrics:
        # Positive values mean the training score is better than the held-out
        # score for score-like metrics.  For MAE/RMSE, a negative value means
        # held-out error is larger; the JSON/UI labels this explicitly.
        row[f"train_minus_test_{metric}"] = rows["train"][metric] - rows["test"][metric]
        row[f"train_minus_val_{metric}"] = rows["train"][metric] - rows["val"][metric]
    return row


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models: list[dict] = []
    rows: list[dict] = []

    for spec in MODEL_SPECS:
        print(f"Fitting {spec.name} ({spec.task})...")
        frames = load_task_frames(spec)
        X_train, y_train = frames["train"]
        pipeline = build_pipeline(spec)
        pipeline.fit(X_train, y_train)

        artifact = MODELS_DIR / f"{spec.name}_{spec.prefix}.joblib"
        joblib.dump(pipeline, artifact)

        metrics = {
            split: score_predictions(spec.task, pipeline, X, y)
            for split, (X, y) in frames.items()
        }
        split_sizes = {split: int(len(X)) for split, (X, _) in frames.items()}

        model_record = {
            "task": spec.task,
            "model": spec.name,
            "preprocessing": spec.preprocessing,
            "deployed": bool(spec.deployed),
            "target": spec.target,
            "features": spec.features,
            "artifact": str(artifact.relative_to(PROJECT_ROOT)),
            "note": spec.note,
            "split_rows": split_sizes,
            "metrics": metrics,
        }

        if spec.task == "placement":
            classification_frame = frames["test"]
            X_test, y_test = classification_frame
            predictions = pipeline.predict(X_test)
            matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
            report = classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                output_dict=True,
                zero_division=0,
            )
            model_record["confusion_matrix"] = {
                "labels": [0, 1],
                "matrix": matrix.tolist(),
                "rows": ["actual_0", "actual_1"],
                "columns": ["predicted_0", "predicted_1"],
            }
            model_record["classification_report"] = json_safe(report)

        models.append(json_safe(model_record))
        rows.append(json_safe(_flatten_model_row(spec, metrics, split_sizes)))

    performance_frame = pd.DataFrame(rows)
    performance_frame.to_csv(PERFORMANCE_CSV, index=False)

    report = {
        "generated_by": "scripts/train_models.py",
        "random_state": 42,
        "split_policy": "70% train / 15% validation / 15% test",
        "metric_direction": {
            "MAE": "lower is better",
            "RMSE": "lower is better",
            "R2": "higher is better",
            "Accuracy": "higher is better",
            "Precision": "higher is better",
            "Recall": "higher is better",
            "F1": "higher is better",
            "ROC_AUC": "higher is better",
        },
        "gap_note": (
            "train_minus_test_<metric> is training minus test. For MAE/RMSE a "
            "negative gap means the held-out error is larger; for score metrics "
            "a positive gap means the training score is higher."
        ),
        "models": models,
        "warnings": [
            (
                "Experimental result — synthetic data limitation. The placement "
                "label is deterministic and the raw datasets are synthetic; "
                "near-perfect placement scores are dataset-construction "
                "artifacts, not evidence of real-world reliability."
            ),
            (
                "The exam-score and placement source files share no genuine "
                "student identifier. Cross-track conclusions are not supported."
            ),
        ],
    }
    PERFORMANCE_JSON.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )

    print(f"Saved model performance: {PERFORMANCE_CSV}")
    print(f"Saved model performance JSON: {PERFORMANCE_JSON}")
    print(f"Saved {len(models)} model artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
