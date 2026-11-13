"""Five-fold cross-validation for every registered model.

Regression uses shuffled ``KFold`` and classification uses shuffled
``StratifiedKFold``.  Cross-validation is run on the training split only.  The
validation and test splits are not used here, and every fold refits the full
preprocessing graph (imputation, scaling, encoding, and the zero-variance
filter) from scratch.

Outputs:
- outputs/evaluation/cross_validation_results.csv
- outputs/evaluation/cross_validation_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from model_pipelines import (
    EVALUATION_DIR,
    MODEL_SPECS,
    build_pipeline,
    cv_result_to_metric_dict,
    cv_scoring,
    json_safe,
    load_task_frames,
)
from model_config import CV_FOLDS, RANDOM_STATE


CV_CSV = EVALUATION_DIR / "cross_validation_results.csv"
CV_JSON = EVALUATION_DIR / "cross_validation_results.json"


def _cv_splitter(task: str):
    if task == "exam_score":
        return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )


def _flatten_metric_row(
    task: str,
    model: str,
    preprocessing: str,
    metric: str,
    values: dict,
) -> dict:
    row = {
        "task": task,
        "model": model,
        "preprocessing": preprocessing,
        "n_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "metric": metric,
        "mean": values["mean"],
        "std": values["std"],
        "min": values["min"],
        "max": values["max"],
        "train_mean": values["train_mean"],
        "train_std": values["train_std"],
        "train_minus_test_gap": values["train_mean"] - values["mean"],
        "test_minus_train_gap": values["mean"] - values["train_mean"],
        "higher_is_better": values["higher_is_better"],
    }
    for index, fold_value in enumerate(values["folds"], start=1):
        row[f"fold_{index}"] = fold_value
    for index, fold_value in enumerate(values["train_folds"], start=1):
        row[f"train_fold_{index}"] = fold_value
    return json_safe(row)


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict] = []
    model_groups: list[dict] = []

    for spec in MODEL_SPECS:
        print(f"Cross-validating {spec.name} ({spec.task}) with {CV_FOLDS} folds...")
        frames = load_task_frames(spec)
        X_train, y_train = frames["train"]
        pipeline = build_pipeline(spec)
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=_cv_splitter(spec.task),
            scoring=cv_scoring(spec.task),
            return_train_score=True,
            n_jobs=1,
            error_score="raise",
        )
        metric_values = cv_result_to_metric_dict(cv_result, spec.task)

        group_metrics = {}
        for metric, values in metric_values.items():
            result_rows.append(
                _flatten_metric_row(
                    spec.task, spec.name, spec.preprocessing, metric, values
                )
            )
            group_metrics[metric] = json_safe(values)

        model_groups.append(
            {
                "task": spec.task,
                "model": spec.name,
                "preprocessing": spec.preprocessing,
                "deployed": bool(spec.deployed),
                "n_folds": CV_FOLDS,
                "metrics": group_metrics,
            }
        )

    frame = pd.DataFrame(result_rows)
    frame.to_csv(CV_CSV, index=False)

    report = {
        "generated_by": "scripts/cross_validation.py",
        "folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "cv_scheme": {
            "exam_score": "KFold(shuffle=True)",
            "placement": "StratifiedKFold(shuffle=True)",
        },
        "fit_policy": (
            "Cross-validation uses the training split only. Every fold refits "
            "imputation, scaling, one-hot encoding, and the zero-variance filter."
        ),
        "gap_note": (
            "train_minus_test_gap is the fold-averaged training score minus the "
            "fold-averaged validation score. For MAE/RMSE a negative value means "
            "held-out error is larger; for score metrics a positive value means "
            "training performance is higher."
        ),
        "models": model_groups,
        "results": result_rows,
    }
    CV_JSON.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )

    print(f"Saved cross-validation CSV: {CV_CSV}")
    print(f"Saved cross-validation JSON: {CV_JSON}")


if __name__ == "__main__":
    main()
