"""Generate EDA outputs for the two independent task tracks.

The old version read the row-order combined file and produced one correlation
matrix as if exam and placement belonged to the same students. This version
reads the independent processed files and keeps the tracks separate.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS = {
    "exam": (PROJECT_ROOT / "data" / "processed" / "exam_score_preprocessed.csv", "Exam_Score"),
    "placement": (PROJECT_ROOT / "data" / "processed" / "placement_preprocessed.csv", "Placement"),
}
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"


def _heatmap(frame: pd.DataFrame, target: str, track: str, label: str) -> None:
    numeric = frame.select_dtypes(include=["number"])
    correlation = numeric.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title(f"{label} Correlation Heatmap ({track} track only)")
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(correlation.columns)), correlation.columns, fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{track}_correlation_heatmap.png", dpi=200)
    plt.close()


def _target_plot(frame: pd.DataFrame, target: str, track: str, label: str) -> None:
    plt.figure(figsize=(7, 4))
    if pd.api.types.is_numeric_dtype(frame[target]) and frame[target].nunique() > 10:
        frame[target].hist(bins=30)
        plt.xlabel(target)
        plt.ylabel("Count")
        plt.title(f"{label} Distribution")
    else:
        frame[target].value_counts().sort_index().plot(kind="bar")
        plt.xlabel(target)
        plt.ylabel("Count")
        plt.title(f"{label} Class Distribution")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / ("exam_score_distribution.png" if track == "exam" else "placement_distribution.png"),
        dpi=200,
    )
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove the old combined heatmap so it cannot be mistaken for linked data.
    legacy = OUTPUT_DIR / "correlation_heatmap.png"
    if legacy.exists():
        legacy.unlink()

    summary_lines = [
        "EDA Summary",
        "===========",
        "The exam and placement files are independent source tracks.",
        "No shared student identifier exists, so there is no combined analysis.",
        "",
    ]
    stats_frames = []

    for track, (path, target) in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing processed file: {path}. Run scripts/preprocess.py first.")
        frame = pd.read_csv(path)
        label = "Exam score" if track == "exam" else "Placement"

        null_count = int(frame.isna().sum().sum())
        duplicate_count = int(frame.duplicated().sum())
        summary_lines.extend(
            [
                f"{label} track ({path.name})",
                "-" * (len(label) + len(path.name) + 9),
                f"Rows: {len(frame)}",
                f"Columns: {len(frame.columns)}",
                f"Null values (total): {null_count}",
                f"Duplicate rows: {duplicate_count}",
            ]
        )
        if target == "Exam_Score":
            summary_lines.append(
                f"Exam_Score mean: {frame[target].mean():.3f}, "
                f"std: {frame[target].std():.3f}, "
                f"range: {frame[target].min()}-{frame[target].max()}"
            )
        else:
            counts = frame[target].value_counts().sort_index().to_dict()
            summary_lines.append(f"Placement counts: {counts}")
        summary_lines.append("")

        _heatmap(frame, target, track, label)
        _target_plot(frame, target, track, label)

        stats = frame.describe(include="all").transpose()
        stats.insert(0, "track", track)
        stats_frames.append(stats)

    combined_stats = pd.concat(stats_frames)
    combined_stats.to_csv(OUTPUT_DIR / "descriptive_stats.csv")
    (OUTPUT_DIR / "eda_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("EDA outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
