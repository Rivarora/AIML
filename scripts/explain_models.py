"""Explain model predictions with coefficients, tree importances, and permutations.

Feature importance is reported as *association with prediction*, never as a
causal effect.  Permutation importance is computed on the held-out test split
using saved artifacts, so it does not refit preprocessing on test data.

Outputs:
- outputs/explanations/exam_score_importance.csv
- outputs/explanations/placement_importance.csv
- outputs/explanations/explanation_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from model_pipelines import (
    EXPLANATION_DIR,
    MODELS_DIR,
    ModelSpec,
    json_safe,
    load_task_frames,
    spec_for,
    transformed_feature_names,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAM_CSV = EXPLANATION_DIR / "exam_score_importance.csv"
PLACEMENT_CSV = EXPLANATION_DIR / "placement_importance.csv"
SUMMARY_JSON = EXPLANATION_DIR / "explanation_summary.json"

EXAM_MODELS = ("LinearRegression", "Ridge", "RandomForestRegressor")
PLACEMENT_MODELS = (
    "LogisticRegression",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "RandomForestClassifierUnconstrained",
    "SVC",
    "GaussianNB",
)

COMMON_NOTE = (
    "Associated with prediction, not causal. Do not turn this ranking into "
    "personal advice."
)


def _rows_for_linear_model(spec: ModelSpec, pipeline, model_name: str) -> list[dict]:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return []
    coefficients = np.ravel(np.asarray(model.coef_, dtype=float))
    names = transformed_feature_names(pipeline)
    if len(names) != len(coefficients):
        return []
    rows = []
    for name, coefficient in zip(names, coefficients):
        rows.append(
            {
                "task": spec.task,
                "model": model_name,
                "method": "linear_coefficient",
                "feature": name,
                "importance": float(abs(coefficient)),
                "importance_std": None,
                "coefficient": float(coefficient),
                "direction": "positive" if coefficient >= 0 else "negative",
                "scoring": "standardized coefficient magnitude",
                "note": COMMON_NOTE,
            }
        )
    return rows


def _rows_for_tree_model(spec: ModelSpec, pipeline, model_name: str) -> list[dict]:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return []
    importances = np.asarray(model.feature_importances_, dtype=float)
    names = transformed_feature_names(pipeline)
    if len(names) != len(importances):
        return []
    return [
        {
            "task": spec.task,
            "model": model_name,
            "method": "tree_importance",
            "feature": name,
            "importance": float(importance),
            "importance_std": None,
            "coefficient": None,
            "direction": "n/a",
            "scoring": "impurity-based importance",
            "note": COMMON_NOTE,
        }
        for name, importance in zip(names, importances)
    ]


def _rows_for_permutation(
    spec: ModelSpec, pipeline, model_name: str, X_test, y_test
) -> list[dict]:
    scoring = "r2" if spec.task == "exam_score" else "roc_auc"
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )
    return [
        {
            "task": spec.task,
            "model": model_name,
            "method": "permutation_importance",
            "feature": feature,
            "importance": float(result.importances_mean[index]),
            "importance_std": float(result.importances_std[index]),
            "coefficient": None,
            "direction": "n/a",
            "scoring": f"decrease in test {scoring} when shuffled",
            "note": COMMON_NOTE,
        }
        for index, feature in enumerate(spec.features)
    ]


def _rank_rows(rows: list[dict]) -> list[dict]:
    ranked = sorted(rows, key=lambda row: row["importance"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _top(rows: list[dict], method: str, limit: int = 8) -> list[dict]:
    selected = [row for row in rows if row["method"] == method]
    return selected[:limit]


def _explain_task(task: str, model_names: tuple[str, ...]) -> tuple[list[dict], dict]:
    all_rows: list[dict] = []
    model_summaries: list[dict] = []

    for model_name in model_names:
        spec = spec_for(task, model_name)
        artifact = MODELS_DIR / f"{model_name}_{spec.prefix}.joblib"
        if not artifact.exists():
            raise FileNotFoundError(
                f"Missing {artifact}. Run scripts/train_models.py first."
            )
        pipeline = joblib.load(artifact)
        frames = load_task_frames(spec)
        X_test, y_test = frames["test"]

        rows: list[dict] = []
        rows.extend(_rows_for_linear_model(spec, pipeline, model_name))
        rows.extend(_rows_for_tree_model(spec, pipeline, model_name))
        rows.extend(_rows_for_permutation(spec, pipeline, model_name, X_test, y_test))
        ranked = _rank_rows(rows)
        all_rows.extend(ranked)

        methods = sorted({row["method"] for row in ranked})
        model_summaries.append(
            {
                "model": model_name,
                "methods": methods,
                "top_permutation_features": _top(ranked, "permutation_importance"),
                "top_linear_coefficients": _top(ranked, "linear_coefficient"),
                "top_tree_features": _top(ranked, "tree_importance"),
            }
        )

    return all_rows, {
        "task": task,
        "target": "Exam_Score" if task == "exam_score" else "Placement",
        "test_metric": "R2" if task == "exam_score" else "ROC_AUC",
        "models": model_summaries,
    }


def main() -> None:
    EXPLANATION_DIR.mkdir(parents=True, exist_ok=True)

    exam_rows, exam_summary = _explain_task("exam_score", EXAM_MODELS)
    placement_rows, placement_summary = _explain_task("placement", PLACEMENT_MODELS)

    columns = [
        "task",
        "model",
        "method",
        "feature",
        "importance",
        "importance_std",
        "coefficient",
        "direction",
        "rank",
        "scoring",
        "note",
    ]
    pd.DataFrame(exam_rows, columns=columns).to_csv(EXAM_CSV, index=False)
    pd.DataFrame(placement_rows, columns=columns).to_csv(PLACEMENT_CSV, index=False)

    summary = {
        "generated_by": "scripts/explain_models.py",
        "caveats": [
            "All importance values describe association with model prediction, not causal effects.",
            "Permutation importance is measured on the held-out test split using the saved pipeline.",
            "Coefficients are on the standardized/encoded scale and are not directly comparable across different feature types.",
            "Feature rankings are not personal advice and should not be used to make individual decisions.",
            "Placement results are experimental: the label is deterministic and the source data are synthetic.",
        ],
        "data_limitation": (
            "The exam and placement source files share no genuine student "
            "identifier; the historical row-order/synthetic pairing was removed. "
            "No cross-track student-level conclusion is supported."
        ),
        "tasks": {
            "exam_score": exam_summary,
            "placement": placement_summary,
        },
    }
    SUMMARY_JSON.write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )

    print(f"Saved exam explanations: {EXAM_CSV}")
    print(f"Saved placement explanations: {PLACEMENT_CSV}")
    print(f"Saved explanation summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
