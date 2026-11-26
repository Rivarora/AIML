"""Run the full reproducible analytics pipeline in dependency order.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --only preprocess select_and_split train_models
    python scripts/run_pipeline.py --list

The runner streams each script's output, stops on the first failure, and
verifies that the artifacts expected from each step exist. It never downloads
data and never calls a remote service.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

STEPS: list[dict] = [
    {
        "name": "preprocess",
        "script": "scripts/preprocess.py",
        "description": "Validate raw files, clean both tracks, write provenance",
        "outputs": [
            "data/interim/StudentPerformanceFactors_clean.csv",
            "data/interim/placement_clean.csv",
            "data/processed/exam_score_preprocessed.csv",
            "data/processed/placement_preprocessed.csv",
            "data/interim/provenance.json",
        ],
    },
    {
        "name": "dataset_checks",
        "script": "scripts/dataset_checks.py",
        "description": "Check processed ranges and task-track integrity",
        "outputs": [],
    },
    {
        "name": "select_and_split",
        "script": "scripts/select_and_split.py",
        "description": "Create reproducible 70/15/15 train/validation/test splits",
        "outputs": [
            "data/splits/exam_score_train.csv",
            "data/splits/placement_train.csv",
            "data/splits/split_manifest.json",
        ],
    },
    {
        "name": "data_quality",
        "script": "scripts/data_quality.py",
        "description": "Write data-quality, split, and linkage reports",
        "outputs": ["outputs/quality/data_quality.json"],
    },
    {
        "name": "pca_analysis",
        "script": "scripts/pca_analysis.py",
        "description": "Fit leakage-safe PCA and write the visualization",
        "outputs": [
            "outputs/pca/pca_summary.json",
            "outputs/pca/pca_visualization.png",
        ],
    },
    {
        "name": "run_eda",
        "script": "scripts/run_eda.py",
        "description": "Write exploratory summaries and correlation plots",
        "outputs": ["outputs/eda/descriptive_stats.csv"],
    },
    {
        "name": "train_baselines",
        "script": "scripts/train_baselines.py",
        "description": "Fit legacy baselines, model card, placement investigation",
        "outputs": [
            "outputs/metrics/baseline_metrics.csv",
            "outputs/metrics/model_card.json",
            "outputs/analysis/placement_investigation.json",
        ],
    },
    {
        "name": "analyze_baselines",
        "script": "scripts/analyze_baselines.py",
        "description": "Write residual, classification, and legacy leakage analysis",
        "outputs": ["outputs/analysis/exam_score_error_summary.csv"],
    },
    {
        "name": "train_models",
        "script": "scripts/train_models.py",
        "description": "Fit all registered pipelines and save held-out performance",
        "outputs": [
            "outputs/evaluation/model_performance.json",
            "models/LinearRegression_exam_score.joblib",
            "models/RandomForestClassifier_placement.joblib",
        ],
    },
    {
        "name": "cross_validation",
        "script": "scripts/cross_validation.py",
        "description": "Run five-fold cross-validation on the training splits",
        "outputs": ["outputs/evaluation/cross_validation_results.json"],
    },
    {
        "name": "overfitting_analysis",
        "script": "scripts/overfitting_analysis.py",
        "description": "Compare train/validation/test/CV results and flag reviews",
        "outputs": ["outputs/evaluation/overfitting_report.json"],
    },
    {
        "name": "leakage_investigation",
        "script": "scripts/leakage_investigation.py",
        "description": "Check structural leakage, duplicates, and deterministic rules",
        "outputs": ["outputs/evaluation/leakage_report.json"],
    },
    {
        "name": "explain_models",
        "script": "scripts/explain_models.py",
        "description": "Write coefficients, tree importances, and permutation importance",
        "outputs": ["outputs/explanations/explanation_summary.json"],
    },
    {
        "name": "clustering",
        "script": "scripts/clustering.py",
        "description": "Fit train-only K-Means and write cohort descriptions",
        "outputs": [
            "outputs/clustering/cluster_summary.json",
            "outputs/clustering/cluster_visualization.png",
        ],
    },
    {
        "name": "fairness_analysis",
        "script": "scripts/fairness_analysis.py",
        "description": "Compare held-out exam error across available groups",
        "outputs": ["outputs/fairness/fairness_summary.json"],
    },
]


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEP",
        help="Run only the named steps (dependencies are not resolved automatically).",
    )
    parser.add_argument("--list", action="store_true", help="List steps and exit.")
    args = parser.parse_args()

    if args.list:
        for index, step in enumerate(STEPS, start=1):
            print(f"{index:2d}. {step['name']:24} {step['description']}")
        return

    known = {step["name"] for step in STEPS}
    if args.only:
        unknown = sorted(set(args.only) - known)
        if unknown:
            fail(f"Unknown step(s): {', '.join(unknown)}. Use --list.")
        selected = [step for step in STEPS if step["name"] in set(args.only)]
    else:
        selected = STEPS

    for index, step in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] {step['name']}: {step['description']}")
        command = [sys.executable, step["script"]]
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            fail(
                f"{step['name']} failed with exit code {result.returncode}. "
                f"Command: {' '.join(command)}"
            )
        missing = [
            output for output in step["outputs"] if not (ROOT / output).exists()
        ]
        if missing:
            fail(
                f"{step['name']} finished but did not write: {', '.join(missing)}"
            )
        print(f"OK: {step['name']}")

    print("\nPipeline complete. Start the app with: python app.py")


if __name__ == "__main__":
    main()
