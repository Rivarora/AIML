"""Compare deployed-model behavior across available exam-track groups."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

from model_config import EXAM_MODEL_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fairness"
MIN_GROUP_SIZE = 30


def run() -> dict:
    test = pd.read_csv(SPLITS_DIR / "exam_score_test.csv")
    model = joblib.load(MODELS_DIR / "LinearRegression_exam_score.joblib")
    predictions = model.predict(test[EXAM_MODEL_FEATURES])
    test = test.assign(prediction=predictions)
    available = ["Gender", "Family_Income", "School_Type", "Learning_Disabilities"]
    rows = []
    excluded = {}
    for group_column in available:
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
            rows.append(
                {
                    "task": "exam_score_regression",
                    "group_column": group_column,
                    "group": str(value),
                    "group_size": int(size),
                    "mean_absolute_error": float(mean_absolute_error(group["Exam_Score"], group["prediction"])),
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "positive_prediction_rate": None,
                }
            )
    result = {
        "task": "exam_score_regression",
        "model": "Linear Regression",
        "minimum_group_size": MIN_GROUP_SIZE,
        "groups_analyzed": sorted({row["group_column"] for row in rows}),
        "groups_excluded": excluded,
        "results": rows,
        "limitations": [
            "This is not a complete fairness audit.",
            "Dataset construction is synthetic.",
            "Small group sizes make comparisons unstable.",
            "Classification metrics and positive prediction rates are not applicable to this regression track.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "fairness_results.csv", index=False)
    (OUTPUT_DIR / "fairness_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
