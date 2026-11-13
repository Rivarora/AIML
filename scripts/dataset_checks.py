"""Quick quality checks for the processed dataset.

What this file does:
- Loads data/processed/combined_encoded.csv
- Prints total null values, duplicate-row count, and dtype distribution

Why this is useful:
- Confirms the processed dataset is in a stable state before splitting/training.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "combined_encoded.csv"


def main() -> None:
    combined_df = pd.read_csv(DATA_FILE)

    # Total number of missing values across all columns.
    null_count = int(combined_df.isnull().sum().sum())
    print(null_count)

    # Number of duplicate rows in the full dataset.
    duplicate_count = int(combined_df.duplicated().sum())
    print(duplicate_count)

    # Count of column datatypes (for quick schema sanity-check).
    print(combined_df.dtypes.apply(lambda x: x.name).value_counts())

    if null_count or duplicate_count:
        raise ValueError("Processed data contains null values or duplicate rows.")
    if not combined_df["Exam_Score"].between(0, 100).all():
        raise ValueError("Exam_Score contains values outside the 0-100 range.")
    if not set(combined_df["Placement"].unique()).issubset({0, 1}):
        raise ValueError("Placement contains values other than 0 and 1.")


if __name__ == "__main__":
	main()

