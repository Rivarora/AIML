"""Generate baseline EDA outputs from the processed dataset.

What this file does:
- Reads data/processed/final_preprocessed_dataset.csv
- Produces summary text and descriptive statistics
- Saves correlation and target distribution plots

Output folder:
- outputs/eda/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "final_preprocessed_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    # 1) Basic shape and quality summary
    null_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    summary_lines = [
        "EDA Summary",
        "===========",
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Null values (total): {null_count}",
        f"Duplicate rows: {duplicate_count}",
        "",
        "Numeric columns:",
        ", ".join(numeric_cols),
        "",
    ]

    # 2) Target distributions
    if "Exam_Score" in df.columns:
        summary_lines.append(
            f"Exam_Score mean: {df['Exam_Score'].mean():.3f}, std: {df['Exam_Score'].std():.3f}"
        )

    if "Placement" in df.columns:
        placement_counts = df["Placement"].value_counts(dropna=False).to_dict()
        summary_lines.append(f"Placement counts: {placement_counts}")

    # 3) Correlation heatmap for numeric features
    corr = df[numeric_cols].corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    # 4) Exam score histogram
    if "Exam_Score" in df.columns:
        plt.figure(figsize=(8, 5))
        df["Exam_Score"].hist(bins=30)
        plt.title("Exam Score Distribution")
        plt.xlabel("Exam_Score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "exam_score_distribution.png", dpi=200)
        plt.close()

    # 5) Placement class distribution
    if "Placement" in df.columns:
        plt.figure(figsize=(6, 4))
        df["Placement"].value_counts().sort_index().plot(kind="bar")
        plt.title("Placement Class Distribution")
        plt.xlabel("Placement")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "placement_distribution.png", dpi=200)
        plt.close()

    # 6) Save descriptive stats
    df.describe(include="all").transpose().to_csv(OUTPUT_DIR / "descriptive_stats.csv")

    (OUTPUT_DIR / "eda_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("EDA outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
