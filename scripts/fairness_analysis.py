"""Compare held-out exam-model error across available cohort groups.

This is a group error comparison, not a fairness certification. It uses only
groups that exist in the exam test split and enforces a minimum group size.
Differences are descriptive: the data are synthetic, the groups are not causal
attributes, and no confidence interval or discriminatory-intent claim is made.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

from model_config import EXAM_MODEL_FEATURES, MODEL_LABELS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CARD_FILE = PROJECT_ROOT / "outputs" / "metrics" / "model_card.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fairness"
MIN_GROUP_SIZE = 30
GROUP_COLUMNS = ["Gender", "Family_Income", "School_Type", "Learning_Disabilities"]


def _model_label() -> str:
    if MODEL_CARD_FILE.exists():
        try:
            card = json.loads(MODEL_CARD_FILE.read_text(encoding="utf-8"))
            label = card.get("deployed", {}).get("exam", {}).get("label")
            if label:
                return str(label)
        except (OSError, ValueError):
            pass
    return MODEL_LABELS["exam"]


def run() -> dict:
    test = pd.read_csv(SPLITS_DIR / "exam_score_test.csv")
    artifact = MODELS_DIR / "LinearRegression_exam_score.joblib"
    if not artifact.exists():
        raise FileNotFoundError(
            f"Missing {artifact.name}. Run python scripts/train_models.py first."
        )
    model = joblib.load(artifact)
    predictions = model.predict(test[EXAM_MODEL_FEATURES])
    test = test.assign(prediction=predictions)

    overall_mae = float(mean_absolute_error(test["Exam_Score"], test["prediction"]))
    rows: list[dict] = []
    excluded: dict[str, str] = {}

    for group_column in GROUP_COLUMNS:
        if group_column not in test:
            excluded[group_column] = "Column is not available in the test split."
            continue
        for value, group in test.groupby(group_column, dropna=False):
            size = len(group)
            if size < MIN_GROUP_SIZE:
                excluded[f"{group_column}={value}"] = (
                    f"Insufficient data: {size} rows; minimum is {MIN_GROUP_SIZE}."
                )
                continue
            group_mae = float(
                mean_absolute_error(group["Exam_Score"], group["prediction"])
            )
            rows.append(
                {
                    "task": "exam_score_regression",
                    "group_column": group_column,
                    "group": str(value),
                    "group_size": int(size),
                    "mean_absolute_error": group_mae,
                    "difference_from_overall": group_mae - overall_mae,
                }
            )

    gaps: list[dict] = []
    for group_column in sorted({row["group_column"] for row in rows}):
        group_rows = [row for row in rows if row["group_column"] == group_column]
        if len(group_rows) < 2:
            continue
        best = min(group_rows, key=lambda row: row["mean_absolute_error"])
        worst = max(group_rows, key=lambda row: row["mean_absolute_error"])
        gaps.append(
            {
                "group_column": group_column,
                "lowest_error_group": best["group"],
                "highest_error_group": worst["group"],
                "absolute_gap": float(
                    worst["mean_absolute_error"] - best["mean_absolute_error"]
                ),
            }
        )
    largest_gap = max(gaps, key=lambda row: row["absolute_gap"]) if gaps else None

    result = {
        "task": "exam_score_regression",
        "model": _model_label(),
        "model_artifact": "models/LinearRegression_exam_score.joblib",
        "evaluation_split": "test",
        "metric": "mean_absolute_error",
        "overall_mae": overall_mae,
        "minimum_group_size": MIN_GROUP_SIZE,
        "groups_analyzed": sorted({row["group_column"] for row in rows}),
        "groups_excluded": excluded,
        "largest_group_gap": largest_gap,
        "results": rows,
        "interpretation": (
            "These are descriptive error differences across available groups. "
            "They do not establish discrimination, fairness, or a causal effect, "
            "and they are not a substitute for a full fairness audit."
        ),
        "limitations": [
            "This is not a complete fairness audit.",
            "The source dataset is synthetic and small groups make comparisons unstable.",
            "Only mean absolute error is compared; no confidence intervals are estimated.",
            "Group membership is descriptive and is not used as a model feature.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "fairness_results.csv", index=False)
    (OUTPUT_DIR / "fairness_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
