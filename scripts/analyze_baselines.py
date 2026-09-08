"""Analyze baseline predictions for classification, regression, and leakage risk.

What this file does:
- Loads the saved train/validation/test splits and baseline models
- Produces placement confusion matrices and classification reports
- Produces exam-score residual tables and error distributions
- Runs simple leakage checks on the final split artifacts

Outputs:
- outputs/analysis/*.csv
- outputs/analysis/*.txt
- outputs/analysis/*.png
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis"


def load_split(prefix: str):
    """Load train/validation/test CSV files for a given prefix."""
    train = pd.read_csv(SPLITS_DIR / f"{prefix}_train.csv")
    val = pd.read_csv(SPLITS_DIR / f"{prefix}_val.csv")
    test = pd.read_csv(SPLITS_DIR / f"{prefix}_test.csv")
    return train, val, test


def analyze_placement() -> None:
    """Save confusion matrices, classification reports, and a comparison table."""
    _, _, test = load_split("placement")
    target = "Placement"
    X_test = test.drop(columns=[target])
    y_test = test[target]
    rows = []

    plt.figure(figsize=(10, 4))
    for plot_index, model_name in enumerate(("LogisticRegression", "RandomForestClassifier"), start=1):
        model = joblib.load(MODELS_DIR / f"{model_name}_placement.joblib")
        predictions = model.predict(X_test)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        matrix_df = pd.DataFrame(
            matrix,
            index=["actual_0", "actual_1"],
            columns=["predicted_0", "predicted_1"],
        )
        matrix_df.to_csv(OUTPUT_DIR / f"{model_name}_confusion_matrix.csv")

        report = classification_report(y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(
            OUTPUT_DIR / f"{model_name}_classification_report.csv"
        )
        (OUTPUT_DIR / f"{model_name}_classification_report.txt").write_text(
            classification_report(y_test, predictions, labels=[0, 1], zero_division=0),
            encoding="utf-8",
        )

        rows.append(
            {
                "model": model_name,
                "accuracy": report["accuracy"],
                "positive_precision": report["1"]["precision"],
                "positive_recall": report["1"]["recall"],
                "positive_f1": report["1"]["f1-score"],
            }
        )

        axis = plt.subplot(1, 2, plot_index)
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(model_name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        axis.set_xticks([0, 1], ["0", "1"])
        axis.set_yticks([0, 1], ["0", "1"])
        for row_index in range(2):
            for column_index in range(2):
                axis.text(column_index, row_index, matrix[row_index, column_index], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "placement_confusion_matrices.png", dpi=200)
    plt.close()
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "placement_model_comparison.csv", index=False)


def analyze_exam_score() -> None:
    """Save prediction errors and a residual distribution plot."""
    _, _, test = load_split("exam_score")
    target = "Exam_Score"
    X_test = test.drop(columns=[target])
    y_test = test[target]
    error_frames = []

    plt.figure(figsize=(8, 5))
    for model_name in ("LinearRegression", "RandomForestRegressor"):
        model = joblib.load(MODELS_DIR / f"{model_name}_exam_score.joblib")
        predictions = model.predict(X_test)
        model_errors = pd.DataFrame(
            {
                "model": model_name,
                "actual_exam_score": y_test,
                "predicted_exam_score": predictions,
                "residual": y_test - predictions,
                "absolute_error": (y_test - predictions).abs(),
            }
        )
        error_frames.append(model_errors)
        plt.hist(model_errors["residual"], bins=25, alpha=0.55, label=model_name)

    errors = pd.concat(error_frames, ignore_index=True)
    errors.to_csv(OUTPUT_DIR / "exam_score_error_analysis.csv", index=False)
    summary = (
        errors.groupby("model")
        .apply(
            lambda group: pd.Series(
                {
                    "MAE": mean_absolute_error(group["actual_exam_score"], group["predicted_exam_score"]),
                    "RMSE": mean_squared_error(group["actual_exam_score"], group["predicted_exam_score"]) ** 0.5,
                    "R2": r2_score(group["actual_exam_score"], group["predicted_exam_score"]),
                    "mean_residual": group["residual"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "exam_score_error_summary.csv", index=False)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Exam Score Residual Distribution")
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exam_score_residual_distribution.png", dpi=200)
    plt.close()


def _overlap_counts(left: pd.DataFrame, right: pd.DataFrame, target: str) -> tuple[int, int]:
    """Count shared feature rows and shared rows with conflicting targets."""
    feature_columns = [column for column in left.columns if column != target]
    left_groups = left.groupby(feature_columns, dropna=False)[target].apply(set).to_dict()
    right_groups = right.groupby(feature_columns, dropna=False)[target].apply(set).to_dict()
    shared_features = set(left_groups) & set(right_groups)
    conflicting_targets = sum(
        bool(left_groups[features] ^ right_groups[features]) for features in shared_features
    )
    return len(shared_features), conflicting_targets


def validate_leakage() -> None:
    """Run structural leakage checks and write a plain-language report."""
    checks = []
    for prefix, target in (("exam_score", "Exam_Score"), ("placement", "Placement")):
        train, val, test = load_split(prefix)
        feature_columns = [column for column in train.columns if column != target]
        target_present = target in feature_columns
        split_pairs = (("train/validation", train, val), ("train/test", train, test), ("validation/test", val, test))
        checks.append(
            {"task": prefix, "check": "target absent from features", "result": not target_present}
        )
        for pair_name, left, right in split_pairs:
            shared_count, conflicting_count = _overlap_counts(left, right, target)
            checks.extend(
                [
                    {
                        "task": prefix,
                        "check": f"{pair_name} exact feature overlap",
                        "result": shared_count == 0,
                        "overlap_count": shared_count,
                    },
                    {
                        "task": prefix,
                        "check": f"{pair_name} conflicting targets for shared features",
                        "result": conflicting_count == 0,
                        "overlap_count": conflicting_count,
                    },
                ]
            )

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(OUTPUT_DIR / "leakage_checks.csv", index=False)
    report_lines = [
        "Leakage Validation",
        "==================",
        "",
        "These checks validate the generated split artifacts; they cannot prove that the source datasets are free from semantic leakage.",
        "Feature selection is performed on training data only in scripts/select_and_split.py.",
        "Repeated encoded feature combinations are reported separately from repeated combinations with conflicting targets.",
        "",
    ]
    for check in checks:
        status = "PASS" if check["result"] else "REVIEW"
        detail = ""
        if "overlap_count" in check:
            detail = f" (count: {check['overlap_count']})"
        report_lines.append(f"{status}: {check['task']} - {check['check']}{detail}")
    report_lines.extend(
        [
            "",
            "Interpretation:",
            "- Exact feature overlap can occur because encoded features have limited discrete values.",
            "- Conflicting targets for the same feature combination are a stronger warning sign than overlap alone.",
            "- The synthetic row-wise pairing of the two source datasets remains a validity limitation.",
            "- Perfect random-forest placement results should not be treated as proof of real-world performance.",
        ]
    )
    (OUTPUT_DIR / "leakage_validation.txt").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analyze_placement()
    analyze_exam_score()
    validate_leakage()
    print("Baseline analysis outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
