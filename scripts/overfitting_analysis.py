"""Compare training, validation, test, and cross-validation performance.

The report flags three different problems instead of collapsing them into one
number:

* a large train-versus-held-out gap (overfitting or distribution shift),
* huge fold-to-fold variance (unstable model), and
* near-perfect placement scores (a dataset-construction artifact).

Inputs are the artifacts written by ``scripts/train_models.py`` and
``scripts/cross_validation.py``.  This script never refits a model on test data.

Outputs:
- outputs/evaluation/overfitting_report.csv
- outputs/evaluation/overfitting_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from model_pipelines import EVALUATION_DIR, HIGHER_IS_BETTER, json_safe


PERFORMANCE_FILE = EVALUATION_DIR / "model_performance.json"
CV_FILE = EVALUATION_DIR / "cross_validation_results.json"
REPORT_CSV = EVALUATION_DIR / "overfitting_report.csv"
REPORT_JSON = EVALUATION_DIR / "overfitting_report.json"

# Thresholds are intentionally explicit and conservative.  A "review" flag is
# a prompt to inspect evidence, not proof of leakage.
THRESHOLDS: dict[str, dict[str, float | None]] = {
    "R2": {"gap": 0.10, "std": 0.10, "relative_std": None, "near_perfect": 0.99},
    "MAE": {"gap": 0.20, "std": None, "relative_std": 0.15, "near_perfect": None},
    "RMSE": {"gap": 0.25, "std": None, "relative_std": 0.15, "near_perfect": None},
    "Accuracy": {"gap": 0.05, "std": 0.03, "relative_std": None, "near_perfect": 0.99},
    "Precision": {"gap": 0.10, "std": 0.05, "relative_std": None, "near_perfect": 0.99},
    "Recall": {"gap": 0.10, "std": 0.05, "relative_std": None, "near_perfect": 0.99},
    "F1": {"gap": 0.05, "std": 0.03, "relative_std": None, "near_perfect": 0.99},
    "ROC_AUC": {"gap": 0.05, "std": 0.03, "relative_std": None, "near_perfect": 0.99},
}

HEADLINE_METRIC = {"exam_score": "R2", "placement": "F1"}


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/train_models.py and "
            "scripts/cross_validation.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _cv_lookup(cv_report: dict) -> dict[tuple[str, str, str], dict]:
    lookup: dict[tuple[str, str, str], dict] = {}
    for row in cv_report.get("results", []):
        lookup[(row["task"], row["model"], row["metric"])] = row
    return lookup


def _flag_text(flags: list[str]) -> str:
    return "; ".join(flags) if flags else "none"


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    performance = _load_json(PERFORMANCE_FILE)
    cv_report = _load_json(CV_FILE)
    cv_lookup = _cv_lookup(cv_report)

    details: list[dict] = []
    summaries: list[dict] = []
    suspicious_cases: list[dict] = []

    for model_record in performance.get("models", []):
        task = model_record["task"]
        model = model_record["model"]
        deployed = bool(model_record.get("deployed", False))
        metrics = model_record["metrics"]
        primary = HEADLINE_METRIC[task]
        model_flags: list[str] = []
        model_notes: list[str] = []

        for metric, values in metrics["train"].items():
            train = float(values)
            validation = float(metrics["val"][metric])
            test = float(metrics["test"][metric])
            cv_row = cv_lookup.get((task, model, metric), {})
            cv_mean = float(cv_row.get("mean", float("nan")))
            cv_std = float(cv_row.get("std", float("nan")))
            cv_min = float(cv_row.get("min", float("nan")))
            cv_max = float(cv_row.get("max", float("nan")))

            thresholds = THRESHOLDS[metric]
            higher_is_better = HIGHER_IS_BETTER[metric]
            # Orient the gap so a positive value always means "held-out worse
            # than training", which makes the overfitting flag unambiguous.
            generalization_gap = (
                train - test if higher_is_better else test - train
            )
            if higher_is_better:
                gap_definition = "train - test (positive = held-out worse)"
            else:
                gap_definition = "test - train (positive = held-out error larger)"

            overfitting_gap_flag = bool(
                generalization_gap > float(thresholds["gap"])
            )
            absolute_std_threshold = thresholds.get("std")
            absolute_variance_exceeded = bool(
                absolute_std_threshold is not None
                and cv_std > float(absolute_std_threshold)
            )
            relative_threshold = thresholds.get("relative_std")
            relative_variance_exceeded = bool(
                relative_threshold is not None
                and cv_mean == cv_mean  # NaN-safe
                and cv_mean != 0
                and (cv_std / abs(cv_mean)) > float(relative_threshold)
            )
            high_variance_flag = bool(
                cv_min == cv_min  # NaN-safe
                and (absolute_variance_exceeded or relative_variance_exceeded)
            )
            near_perfect_threshold = thresholds["near_perfect"]
            near_perfect_test_flag = bool(
                near_perfect_threshold is not None and test >= float(near_perfect_threshold)
            )
            near_perfect_cv_flag = bool(
                near_perfect_threshold is not None
                and cv_mean == cv_mean
                and cv_mean >= float(near_perfect_threshold)
            )

            if overfitting_gap_flag:
                model_flags.append(f"train_test_gap:{metric}")
            if high_variance_flag:
                model_flags.append(f"high_cv_variance:{metric}")
            if near_perfect_test_flag:
                model_flags.append(f"near_perfect_test:{metric}")
            if near_perfect_cv_flag:
                model_flags.append(f"near_perfect_cv:{metric}")

            details.append(
                json_safe(
                    {
                        "task": task,
                        "model": model,
                        "preprocessing": model_record.get("preprocessing"),
                        "deployed": deployed,
                        "metric": metric,
                        "higher_is_better": higher_is_better,
                        "train": train,
                        "validation": validation,
                        "test": test,
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                        "cv_min": cv_min,
                        "cv_max": cv_max,
                        "train_minus_test_gap": train - test,
                        "train_minus_validation_gap": train - validation,
                        "generalization_gap": generalization_gap,
                        "gap_definition": gap_definition,
                        "gap_threshold": thresholds["gap"],
                        "variance_threshold": thresholds["std"],
                        "relative_variance_threshold": thresholds.get("relative_std"),
                        "overfitting_gap_flag": overfitting_gap_flag,
                        "high_cv_variance_flag": high_variance_flag,
                        "near_perfect_test_flag": near_perfect_test_flag,
                        "near_perfect_cv_flag": near_perfect_cv_flag,
                    }
                )
            )

        # Model-level summary, using the primary metric for the headline.
        headline = metrics["train"][primary]
        headline_test = metrics["test"][primary]
        headline_cv = cv_lookup.get((task, model, primary), {})
        headline_gap = (
            float(headline) - float(headline_test)
            if HIGHER_IS_BETTER[primary]
            else float(headline_test) - float(headline)
        )

        if task == "placement" and model == "RandomForestClassifierUnconstrained":
            model_flags.append("diagnostic_deterministic_label_model")
            model_notes.append(
                "Unconstrained diagnostic forest; it recovers the deterministic "
                "placement rule and must not be presented as a deployable model."
            )
        if task == "placement" and any(
            flag.startswith(("near_perfect_test", "near_perfect_cv"))
            for flag in model_flags
        ):
            model_flags.append("placement_synthetic_data_warning")
            model_notes.append(
                "Near-perfect placement performance is suspicious: the source "
                "datasets are synthetic and were historically paired by row order. "
                "Treat this as an experimental dataset-construction result."
            )

        model_flags = sorted(set(model_flags))
        critical = any(
            flag.startswith(("near_perfect", "placement_synthetic"))
            for flag in model_flags
        )
        review = any(
            flag.startswith(("train_test_gap", "high_cv_variance"))
            for flag in model_flags
        )
        status = "critical" if critical else "review" if review else "ok"

        summary = json_safe(
            {
                "task": task,
                "model": model,
                "preprocessing": model_record.get("preprocessing"),
                "deployed": deployed,
                "headline_metric": primary,
                "headline_train": float(headline),
                "headline_validation": float(metrics["val"][primary]),
                "headline_test": float(headline_test),
                "headline_cv_mean": float(headline_cv.get("mean", float("nan"))),
                "headline_cv_std": float(headline_cv.get("std", float("nan"))),
                "headline_gap": float(headline_gap),
                "status": status,
                "flags": model_flags,
                "flag_text": _flag_text(model_flags),
                "note": " ".join(model_notes)
                if model_notes
                else "No automatic overfitting or stability flag was raised.",
            }
        )
        summaries.append(summary)

        if critical or review:
            suspicious_cases.append(
                {
                    "task": task,
                    "model": model,
                    "status": status,
                    "flags": model_flags,
                    "reason": summary["note"],
                }
            )

    details_frame = pd.DataFrame(details)
    details_frame.to_csv(REPORT_CSV, index=False)

    # Prefer the explicitly diagnostic unconstrained forest when it is present;
    # otherwise use the first critical case, then the largest headline gap.
    most_suspicious = next(
        (
            case
            for case in suspicious_cases
            if case["status"] == "critical"
            and case["model"] == "RandomForestClassifierUnconstrained"
        ),
        None,
    )
    if most_suspicious is None:
        most_suspicious = next(
            (case for case in suspicious_cases if case["status"] == "critical"),
            None,
        )
    if most_suspicious is None:
        ranked = sorted(
            summaries, key=lambda item: item["headline_gap"], reverse=True
        )
        if ranked:
            most_suspicious = {
                "task": ranked[0]["task"],
                "model": ranked[0]["model"],
                "status": ranked[0]["status"],
                "flags": ranked[0]["flags"],
                "reason": ranked[0]["note"],
            }

    report = {
        "generated_by": "scripts/overfitting_analysis.py",
        "inputs": [
            "outputs/evaluation/model_performance.json",
            "outputs/evaluation/cross_validation_results.json",
        ],
        "thresholds": THRESHOLDS,
        "headline_metric": HEADLINE_METRIC,
        "global_warnings": [
            (
                "Experimental result — synthetic data limitation. The placement "
                "source is synthetic and its label is deterministic; near-perfect "
                "placement scores are dataset-construction artifacts."
            ),
            (
                "The two source datasets share no genuine student identifier. "
                "The historical row-order/synthetic pairing was removed; no "
                "student-level cross-track conclusion is supported."
            ),
            (
                "A flag means the evidence should be inspected before making a "
                "generalisation claim. It does not by itself prove leakage."
            ),
        ],
        "most_suspicious": most_suspicious,
        "suspicious_cases": suspicious_cases,
        "summary": summaries,
        "details": details,
    }
    REPORT_JSON.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8"
    )

    print(f"Saved overfitting CSV: {REPORT_CSV}")
    print(f"Saved overfitting JSON: {REPORT_JSON}")
    print(f"Flagged {len(suspicious_cases)} model(s) for review.")


if __name__ == "__main__":
    main()
