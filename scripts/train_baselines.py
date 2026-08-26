"""Train baseline ML models for both project tasks.

What this file does:
- Loads split datasets from data/splits/
- Trains two regression baselines for Exam_Score
- Trains two classification baselines for Placement
- Saves trained models and evaluation metrics

Outputs:
- models/*.joblib
- outputs/metrics/baseline_metrics.csv
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def load_split(prefix: str):
    """Load train/validation/test CSV files for a given prefix."""
    train = pd.read_csv(SPLITS_DIR / f"{prefix}_train.csv")
    val = pd.read_csv(SPLITS_DIR / f"{prefix}_val.csv")
    test = pd.read_csv(SPLITS_DIR / f"{prefix}_test.csv")
    return train, val, test


def train_exam_score_models() -> list[dict]:
    """Train regression baselines and return metrics rows."""
    train, val, test = load_split("exam_score")

    target = "Exam_Score"
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    rows: list[dict] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, pred) ** 0.5
        row = {
            "task": "exam_score_regression",
            "model": name,
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": float(rmse),
            "R2": float(r2_score(y_test, pred)),
        }
        rows.append(row)
        joblib.dump(model, MODELS_DIR / f"{name}_exam_score.joblib")

    val.to_csv(METRICS_DIR / "exam_score_validation_reference.csv", index=False)
    return rows


def train_placement_models() -> list[dict]:
    """Train classification baselines and return metrics rows."""
    train, val, test = load_split("placement")

    target = "Placement"
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }

    rows: list[dict] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, proba))
        else:
            roc_auc = float("nan")

        row = {
            "task": "placement_classification",
            "model": name,
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1": float(f1_score(y_test, pred, zero_division=0)),
            "ROC_AUC": roc_auc,
        }
        rows.append(row)
        joblib.dump(model, MODELS_DIR / f"{name}_placement.joblib")

    val.to_csv(METRICS_DIR / "placement_validation_reference.csv", index=False)
    return rows


def main() -> None:
    # Ensure output directories exist before writing files.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_rows.extend(train_exam_score_models())
    all_rows.extend(train_placement_models())

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(METRICS_DIR / "baseline_metrics.csv", index=False)
    (METRICS_DIR / "baseline_metrics.json").write_text(
        json.dumps(all_rows, indent=2),
        encoding="utf-8",
    )

    print("Baseline models trained.")
    print("Metrics file:", METRICS_DIR / "baseline_metrics.csv")
    print("Saved models in:", MODELS_DIR)


if __name__ == "__main__":
    main()
