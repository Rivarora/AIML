"""Train pipeline-based baselines with 5-fold cross-validation.

Every model is a scikit-learn ``Pipeline`` containing its own preprocessing:
median imputation plus scaling for numeric columns, and most-frequent
imputation plus one-hot encoding for categorical columns. The pipeline is
fitted on the training split only, so validation, test, and cross-validation
folds never influence the learned transformations.

Outputs:
- models/*.joblib                         (full preprocessing + model pipelines)
- outputs/metrics/baseline_metrics.csv/json
- outputs/metrics/cross_validation.csv/json
- outputs/metrics/overfitting_analysis.csv/json
- outputs/metrics/model_card.json
- outputs/analysis/placement_investigation.json/csv
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_config import (
    CV_FOLDS,
    EXAM_CATEGORICAL_FEATURES,
    EXAM_MODEL_FEATURES,
    EXAM_NUMERIC_FEATURES,
    MODEL_LABELS,
    PLACEMENT_CATEGORICAL_FEATURES,
    PLACEMENT_MODEL_FEATURES,
    PLACEMENT_NUMERIC_FEATURES,
    RANDOM_STATE,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
CLEAN_DIR = PROJECT_ROOT / "data" / "interim"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "analysis"


def load_split(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(SPLITS_DIR / f"{prefix}_train.csv")
    val = pd.read_csv(SPLITS_DIR / f"{prefix}_val.csv")
    test = pd.read_csv(SPLITS_DIR / f"{prefix}_test.csv")
    return train, val, test


def json_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a frame to records without NaN, which is invalid JSON."""
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict("records")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: dict[str, list[str]],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    estimator,
    numeric_features: list[str],
    categorical_features: dict[str, list[str]],
) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ]
    )


def regression_scores(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_scores(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_proba)),
    }


def _cv_summary(cv_result: dict, task: str) -> dict[str, float]:
    summary: dict[str, float] = {}
    metric_names = (
        ("MAE", "neg_mean_absolute_error"),
        ("RMSE", "neg_root_mean_squared_error"),
        ("R2", "r2"),
    ) if task == "exam_score" else (
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("ROC_AUC", "roc_auc"),
    )
    for label, key in metric_names:
        test_values = np.asarray(cv_result[f"test_{key}"], dtype=float)
        train_values = np.asarray(cv_result[f"train_{key}"], dtype=float)
        if key.startswith("neg_"):
            test_values = -test_values
            train_values = -train_values
        summary[f"cv_{label}_mean"] = float(test_values.mean())
        summary[f"cv_{label}_std"] = float(test_values.std())
        summary[f"cv_train_{label}_mean"] = float(train_values.mean())
    return summary


def train_track(task: str) -> tuple[list[dict], dict[str, Pipeline]]:
    if task == "exam_score":
        prefix = "exam_score"
        target = "Exam_Score"
        features = EXAM_MODEL_FEATURES
        numeric = EXAM_NUMERIC_FEATURES
        categorical = EXAM_CATEGORICAL_FEATURES
        estimators = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "neg_mean_absolute_error": "neg_mean_absolute_error",
            "neg_root_mean_squared_error": "neg_root_mean_squared_error",
            "r2": "r2",
        }
    else:
        prefix = "placement"
        target = "Placement"
        features = PLACEMENT_MODEL_FEATURES
        numeric = PLACEMENT_NUMERIC_FEATURES
        categorical = PLACEMENT_CATEGORICAL_FEATURES
        # The source label is deterministic: 3 of 4 thresholds reproduce it
        # exactly. An unconstrained forest therefore scores 1.0. This
        # deliberately capacity-limited forest is an honest benchmark rather
        # than a claim that the data supports near-perfect prediction.
        estimators = {
            "LogisticRegression": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=300,
                max_depth=2,
                min_samples_leaf=25,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }
        cv = StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }

    train, val, test = load_split(prefix)
    X_train, y_train_raw = train[features], train[target]
    X_val, y_val_raw = val[features], val[target]
    X_test, y_test_raw = test[features], test[target]

    if task == "placement":
        y_train = (y_train_raw == "Yes").astype(int)
        y_val = (y_val_raw == "Yes").astype(int)
        y_test = (y_test_raw == "Yes").astype(int)
    else:
        y_train = y_train_raw.astype(float)
        y_val = y_val_raw.astype(float)
        y_test = y_test_raw.astype(float)

    rows: list[dict] = []
    pipelines: dict[str, Pipeline] = {}

    for name, estimator in estimators.items():
        pipeline = build_pipeline(estimator, numeric, categorical)
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=1,
            error_score="raise",
        )
        cv_summary = _cv_summary(cv_result, task)

        pipeline.fit(X_train, y_train)
        train_pred = pipeline.predict(X_train)
        val_pred = pipeline.predict(X_val)
        test_pred = pipeline.predict(X_test)

        if task == "exam_score":
            train_scores = regression_scores(y_train, train_pred)
            val_scores = regression_scores(y_val, val_pred)
            test_scores = regression_scores(y_test, test_pred)
        else:
            train_scores = classification_scores(
                y_train, train_pred, pipeline.predict_proba(X_train)[:, 1]
            )
            val_scores = classification_scores(
                y_val, val_pred, pipeline.predict_proba(X_val)[:, 1]
            )
            test_scores = classification_scores(
                y_test, test_pred, pipeline.predict_proba(X_test)[:, 1]
            )

        row = {
            "task": "exam_score_regression" if task == "exam_score" else "placement_classification",
            "model": name,
            "features": len(features),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            **test_scores,
            **{f"train_{key}": value for key, value in train_scores.items()},
            **{f"val_{key}": value for key, value in val_scores.items()},
            **cv_summary,
        }
        rows.append(row)
        pipelines[name] = pipeline
        joblib.dump(pipeline, MODELS_DIR / f"{name}_{prefix}.joblib")

    return rows, pipelines


def build_overfitting_analysis(metrics_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    details: list[dict] = []
    summaries: list[dict] = []

    for row in metrics_rows:
        task = row["task"]
        model = row["model"]
        if task == "exam_score_regression":
            gaps = {
                "MAE": row["MAE"] - row["train_MAE"],
                "RMSE": row["RMSE"] - row["train_RMSE"],
                "R2": row["train_R2"] - row["R2"],
            }
            cv_gaps = {
                "MAE": row["cv_MAE_mean"] - row["cv_train_MAE_mean"],
                "R2": row["cv_train_R2_mean"] - row["cv_R2_mean"],
            }
            threshold = 0.10
            worst_metric = "R2"
            worst_gap = gaps["R2"]
        else:
            gaps = {
                "Accuracy": row["train_Accuracy"] - row["Accuracy"],
                "F1": row["train_F1"] - row["F1"],
                "ROC_AUC": row["train_ROC_AUC"] - row["ROC_AUC"],
            }
            cv_gaps = {
                "Accuracy": row["cv_train_Accuracy_mean"] - row["cv_Accuracy_mean"],
                "F1": row["cv_train_F1_mean"] - row["cv_F1_mean"],
            }
            threshold = 0.05
            worst_metric = "Accuracy"
            worst_gap = gaps["Accuracy"]

        for metric, gap in gaps.items():
            details.append(
                {
                    "task": task,
                    "model": model,
                    "metric": metric,
                    "train": row.get(f"train_{metric}"),
                    "validation": row.get(f"val_{metric}"),
                    "test": row.get(metric),
                    "train_minus_test_gap": float(gap),
                }
            )
        for metric, gap in cv_gaps.items():
            details.append(
                {
                    "task": task,
                    "model": model,
                    "metric": f"cv_{metric}",
                    "train": row.get(f"cv_train_{metric}_mean"),
                    "validation": None,
                    "test": row.get(f"cv_{metric}_mean"),
                    "train_minus_test_gap": float(gap),
                }
            )

        status = "review" if worst_gap > threshold else "ok"
        summaries.append(
            {
                "task": task,
                "model": model,
                "headline_metric": worst_metric,
                "train_minus_test_gap": float(worst_gap),
                "threshold": threshold,
                "status": status,
                "interpretation": (
                    "Training score is materially higher than the held-out score; "
                    "inspect variance before claiming generalisation."
                    if status == "review"
                    else "Training and held-out scores are close enough for this dataset."
                ),
            }
        )

    return details, summaries


def build_placement_investigation() -> tuple[dict, list[dict]]:
    """Test whether the placement label is a deterministic synthetic rule."""
    placement = pd.read_csv(CLEAN_DIR / "placement_clean.csv")
    y = (placement["Placement"] == "Yes").astype(int)

    conditions = {
        "IQ": placement["IQ"] > 110,
        "CGPA": placement["CGPA"] > 8,
        "Projects_Completed": placement["Projects_Completed"] > 1,
        "Communication_Skills": placement["Communication_Skills"] > 7,
    }
    condition_frame = pd.DataFrame(conditions)
    votes = condition_frame.sum(axis=1)
    rule_prediction = (votes >= 3).astype(int)
    rule_accuracy = float(accuracy_score(y, rule_prediction))
    rule_errors = int((rule_prediction != y).sum())

    train, _, test = load_split("placement")
    X_train, y_train = train[PLACEMENT_MODEL_FEATURES], (train["Placement"] == "Yes").astype(int)
    X_test, y_test = test[PLACEMENT_MODEL_FEATURES], (test["Placement"] == "Yes").astype(int)

    unconstrained = build_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        PLACEMENT_NUMERIC_FEATURES,
        PLACEMENT_CATEGORICAL_FEATURES,
    )
    unconstrained.fit(X_train, y_train)
    unconstrained_pred = unconstrained.predict(X_test)
    unconstrained_proba = unconstrained.predict_proba(X_test)[:, 1]
    unconstrained_scores = classification_scores(
        y_test, unconstrained_pred, unconstrained_proba
    )

    constrained = build_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=2,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        PLACEMENT_NUMERIC_FEATURES,
        PLACEMENT_CATEGORICAL_FEATURES,
    )
    constrained.fit(X_train, y_train)
    constrained_scores = classification_scores(
        y_test,
        constrained.predict(X_test),
        constrained.predict_proba(X_test)[:, 1],
    )

    feature_rows = [
        {
            "feature": feature,
            "threshold": "> 110" if feature == "IQ" else "> 8" if feature == "CGPA" else "> 1" if feature == "Projects_Completed" else "> 7",
            "share_above_threshold": float(condition.mean()),
            "share_placed_above": float(y[condition].mean()),
            "share_placed_below": float(y[~condition].mean()),
        }
        for feature, condition in conditions.items()
    ]

    majority_accuracy = float(max(y.mean(), 1 - y.mean()))
    report = {
        "dataset": "college_student_placement_dataset.csv",
        "target": "Placement",
        "deterministic_label_rule": {
            "description": (
                "Placement = Yes when at least 3 of these 4 conditions hold: "
                "IQ > 110, CGPA > 8, Projects_Completed > 1, "
                "Communication_Skills > 7."
            ),
            "accuracy": rule_accuracy,
            "errors": rule_errors,
            "rows": int(len(placement)),
        },
        "majority_class_baseline_accuracy": majority_accuracy,
        "unconstrained_random_forest": {
            **unconstrained_scores,
            "note": "Fitted only as diagnostic evidence; it is not the deployed model.",
        },
        "constrained_random_forest": {
            **constrained_scores,
            "note": "Depth-limited benchmark saved and deployed by this project.",
        },
        "conclusion": (
            "The placement label is a deterministic function of four features. "
            "A flexible random forest reaches 1.0 because it recovers that rule, "
            "not because it discovered a transferable real-world signal. The "
            "deployed model is deliberately capacity-limited and the perfect "
            "score is reported as a dataset-construction artifact."
        ),
        "leakage_suspected": False,
    }
    return report, feature_rows


def main() -> None:
    for directory in (MODELS_DIR, METRICS_DIR, ANALYSIS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    exam_rows, exam_pipelines = train_track("exam_score")
    placement_rows, placement_pipelines = train_track("placement")
    all_rows = exam_rows + placement_rows

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(METRICS_DIR / "baseline_metrics.csv", index=False)
    (METRICS_DIR / "baseline_metrics.json").write_text(
        json.dumps(json_records(metrics_df), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    cv_columns = ["task", "model"] + [
        column for column in metrics_df.columns if column.startswith("cv_")
    ]
    metrics_df[cv_columns].to_csv(METRICS_DIR / "cross_validation.csv", index=False)
    (METRICS_DIR / "cross_validation.json").write_text(
        json.dumps(
            json_records(metrics_df[cv_columns]), indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )

    gap_details, gap_summaries = build_overfitting_analysis(all_rows)
    pd.DataFrame(gap_details).to_csv(
        METRICS_DIR / "overfitting_analysis.csv", index=False
    )
    (METRICS_DIR / "overfitting_analysis.json").write_text(
        json.dumps(
            {"thresholds": {"regression_R2": 0.10, "classification_accuracy": 0.05},
             "details": gap_details, "summary": gap_summaries},
            indent=2,
        ),
        encoding="utf-8",
    )

    investigation, feature_rows = build_placement_investigation()
    (ANALYSIS_DIR / "placement_investigation.json").write_text(
        json.dumps(investigation, indent=2), encoding="utf-8"
    )
    pd.DataFrame(feature_rows).to_csv(
        ANALYSIS_DIR / "placement_rule_features.csv", index=False
    )

    model_card = {
        "deployed": {
            "exam": {
                "model": "LinearRegression",
                "label": MODEL_LABELS["exam"],
                "artifact": "models/LinearRegression_exam_score.joblib",
                "features": EXAM_MODEL_FEATURES,
            },
            "placement": {
                "model": "RandomForestClassifier",
                "label": MODEL_LABELS["placement"],
                "artifact": "models/RandomForestClassifier_placement.joblib",
                "features": PLACEMENT_MODEL_FEATURES,
                "capacity_limit": "max_depth=2, min_samples_leaf=25",
            },
        },
        "preprocessing": {
            "numeric": "median imputation + StandardScaler",
            "categorical": "most-frequent imputation + OneHotEncoder(handle_unknown='ignore')",
            "fit_policy": "inside each Pipeline, on the training split only",
        },
        "cross_validation": {"folds": CV_FOLDS, "random_state": RANDOM_STATE},
        "linked_records": False,
        "synthetic_pairing_used": False,
    }
    (METRICS_DIR / "model_card.json").write_text(
        json.dumps(model_card, indent=2), encoding="utf-8"
    )

    print("Pipeline baselines trained.")
    print("Metrics:", METRICS_DIR / "baseline_metrics.csv")
    print("Cross-validation:", METRICS_DIR / "cross_validation.csv")
    print("Overfitting:", METRICS_DIR / "overfitting_analysis.csv")
    print("Placement investigation:", ANALYSIS_DIR / "placement_investigation.json")


if __name__ == "__main__":
    main()
