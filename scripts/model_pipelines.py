"""Reusable, leakage-safe scikit-learn pipelines and model registry.

Every model is represented by a :class:`ModelSpec`.  The spec decides whether
numeric columns are scaled.  Logistic Regression, KNN, and SVM use scaling;
Decision Tree, Random Forest, and Naive Bayes use an unscaled preprocessing
branch.

The full preprocessing graph lives *inside* the pipeline:

* numeric: median imputation, optional ``StandardScaler``
* categorical: most-frequent imputation, one-hot encoding
* a zero-variance filter (``VarianceThreshold``) fitted on training data only

Because the graph is part of the pipeline, refitting it inside a
cross-validation fold cannot leak validation/test information into imputation,
scaling, encoding, or feature selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:  # scripts/ is on sys.path when a script is executed directly.
    from model_config import (
        EXAM_CATEGORICAL_FEATURES,
        EXAM_MODEL_FEATURES,
        EXAM_NUMERIC_FEATURES,
        PLACEMENT_CATEGORICAL_FEATURES,
        PLACEMENT_MODEL_FEATURES,
        PLACEMENT_NUMERIC_FEATURES,
        RANDOM_STATE,
    )
except ImportError:  # pragma: no cover - used when imported as scripts.model_pipelines
    from .model_config import (
        EXAM_CATEGORICAL_FEATURES,
        EXAM_MODEL_FEATURES,
        EXAM_NUMERIC_FEATURES,
        PLACEMENT_CATEGORICAL_FEATURES,
        PLACEMENT_MODEL_FEATURES,
        PLACEMENT_NUMERIC_FEATURES,
        RANDOM_STATE,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
EXPLANATION_DIR = PROJECT_ROOT / "outputs" / "explanations"

REGRESSION_METRICS = ("MAE", "RMSE", "R2")
CLASSIFICATION_METRICS = ("Accuracy", "Precision", "Recall", "F1", "ROC_AUC")
HIGHER_IS_BETTER = {
    "MAE": False,
    "RMSE": False,
    "R2": True,
    "Accuracy": True,
    "Precision": True,
    "Recall": True,
    "F1": True,
    "ROC_AUC": True,
}


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one supervised model track."""

    task: str
    name: str
    estimator_factory: Callable[[], Any]
    scale_numeric: bool
    deployed: bool = False
    note: str = ""

    @property
    def prefix(self) -> str:
        return "exam_score" if self.task == "exam_score" else "placement"

    @property
    def metrics(self) -> tuple[str, ...]:
        return REGRESSION_METRICS if self.task == "exam_score" else CLASSIFICATION_METRICS

    @property
    def target(self) -> str:
        return "Exam_Score" if self.task == "exam_score" else "Placement"

    @property
    def features(self) -> list[str]:
        return EXAM_MODEL_FEATURES if self.task == "exam_score" else PLACEMENT_MODEL_FEATURES

    @property
    def numeric_features(self) -> list[str]:
        return (
            EXAM_NUMERIC_FEATURES
            if self.task == "exam_score"
            else PLACEMENT_NUMERIC_FEATURES
        )

    @property
    def categorical_features(self) -> dict[str, list[str]]:
        return (
            EXAM_CATEGORICAL_FEATURES
            if self.task == "exam_score"
            else PLACEMENT_CATEGORICAL_FEATURES
        )

    @property
    def preprocessing(self) -> str:
        return "scaled" if self.scale_numeric else "unscaled"


EXAM_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        task="exam_score",
        name="LinearRegression",
        estimator_factory=lambda: LinearRegression(),
        scale_numeric=True,
        deployed=True,
        note="Deployed exam baseline; coefficients are interpretable.",
    ),
    ModelSpec(
        task="exam_score",
        name="Ridge",
        estimator_factory=lambda: Ridge(alpha=1.0),
        scale_numeric=True,
        note="Regularized linear benchmark.",
    ),
    ModelSpec(
        task="exam_score",
        name="RandomForestRegressor",
        estimator_factory=lambda: RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scale_numeric=False,
        note="Non-linear tree benchmark; does not require scaling.",
    ),
)

PLACEMENT_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        task="placement",
        name="LogisticRegression",
        estimator_factory=lambda: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        scale_numeric=True,
        note="Linear probability benchmark; scaling required.",
    ),
    ModelSpec(
        task="placement",
        name="KNeighborsClassifier",
        estimator_factory=lambda: KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        ),
        scale_numeric=True,
        note="Distance-based model; scaling required.",
    ),
    ModelSpec(
        task="placement",
        name="DecisionTreeClassifier",
        estimator_factory=lambda: DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        scale_numeric=False,
        note="Capacity-limited tree benchmark.",
    ),
    ModelSpec(
        task="placement",
        name="RandomForestClassifier",
        estimator_factory=lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=2,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scale_numeric=False,
        deployed=True,
        note=(
            "Deployed placement benchmark. Deliberately capacity-limited because "
            "the source label is deterministic; its score is not real-world accuracy."
        ),
    ),
    ModelSpec(
        task="placement",
        name="RandomForestClassifierUnconstrained",
        estimator_factory=lambda: RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scale_numeric=False,
        note=(
            "Diagnostic only. Recovers the deterministic source label and is expected "
            "to look near-perfect; do not treat it as a reliable model."
        ),
    ),
    ModelSpec(
        task="placement",
        name="SVC",
        estimator_factory=lambda: SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        scale_numeric=True,
        note="Margin-based classifier; scaling required and probability estimates enabled.",
    ),
    ModelSpec(
        task="placement",
        name="GaussianNB",
        estimator_factory=lambda: GaussianNB(),
        scale_numeric=False,
        note="Probabilistic benchmark; no scaling required.",
    ),
)

MODEL_SPECS: tuple[ModelSpec, ...] = EXAM_MODEL_SPECS + PLACEMENT_MODEL_SPECS


def spec_for(task: str, model: str) -> ModelSpec:
    """Return the registry entry for a task/model pair."""
    for spec in MODEL_SPECS:
        if spec.task == task and spec.name == model:
            return spec
    raise KeyError(f"Unknown model specification: task={task!r}, model={model!r}")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: dict[str, list[str]],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build the training-fitted preprocessing graph."""
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
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


def build_pipeline(spec: ModelSpec) -> Pipeline:
    """Build a complete pipeline for one model specification.

    ``VarianceThreshold`` is a zero-variance feature filter.  It is the only
    explicit feature-selection step and, because it lives inside the pipeline,
    it is fitted separately on every training split/fold.
    """
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    spec.numeric_features,
                    spec.categorical_features,
                    spec.scale_numeric,
                ),
            ),
            ("selector", VarianceThreshold(threshold=0.0)),
            ("model", spec.estimator_factory()),
        ]
    )


def load_split(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the reproducible train/validation/test split for a track."""
    train = pd.read_csv(SPLITS_DIR / f"{prefix}_train.csv")
    val = pd.read_csv(SPLITS_DIR / f"{prefix}_val.csv")
    test = pd.read_csv(SPLITS_DIR / f"{prefix}_test.csv")
    return train, val, test


def load_task_frames(spec: ModelSpec) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Return ``{split: (X, y)}`` with the target encoded for scikit-learn."""
    train, val, test = load_split(spec.prefix)
    frames: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        X = frame[spec.features]
        if spec.task == "exam_score":
            y = frame[spec.target].astype(float)
        else:
            y = (frame[spec.target].astype(str).str.lower() == "yes").astype(int)
        frames[name] = (X, y)
    return frames


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


def score_predictions(
    task: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    """Evaluate a fitted pipeline on one split."""
    predictions = pipeline.predict(X)
    if task == "exam_score":
        return regression_scores(y, predictions)
    probabilities = pipeline.predict_proba(X)[:, 1]
    return classification_scores(y, predictions, probabilities)


def cv_scoring(task: str) -> dict[str, Any]:
    """Scoring dictionary for ``cross_validate`` with explicit zero division."""
    if task == "exam_score":
        return {
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
        }
    return {
        "Accuracy": "accuracy",
        "Precision": make_scorer(precision_score, zero_division=0),
        "Recall": make_scorer(recall_score, zero_division=0),
        "F1": make_scorer(f1_score, zero_division=0),
        "ROC_AUC": "roc_auc",
    }


def cv_result_to_metric_dict(cv_result: dict[str, Any], task: str) -> dict[str, dict[str, Any]]:
    """Convert a ``cross_validate`` result into per-metric fold statistics."""
    metrics = REGRESSION_METRICS if task == "exam_score" else CLASSIFICATION_METRICS
    output: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        test_values = np.asarray(cv_result[f"test_{metric}"], dtype=float)
        train_values = np.asarray(cv_result[f"train_{metric}"], dtype=float)
        if metric in ("MAE", "RMSE"):
            test_values = -test_values
            train_values = -train_values
        output[metric] = {
            "folds": [float(value) for value in test_values],
            "train_folds": [float(value) for value in train_values],
            "mean": float(test_values.mean()),
            "std": float(test_values.std()),
            "min": float(test_values.min()),
            "max": float(test_values.max()),
            "train_mean": float(train_values.mean()),
            "train_std": float(train_values.std()),
            "higher_is_better": HIGHER_IS_BETTER[metric],
        }
    return output


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return post-preprocessing feature names, respecting the selector."""
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    selector = pipeline.named_steps.get("selector")
    if selector is not None and hasattr(selector, "get_support"):
        support = np.asarray(selector.get_support(), dtype=bool)
        if support.shape[0] == len(names):
            names = [name for name, keep in zip(names, support) if keep]
    return names


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars and NaN to plain JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
