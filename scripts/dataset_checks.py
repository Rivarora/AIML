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
	print(combined_df.isnull().sum().sum())

	# Number of duplicate rows in the full dataset.
	print(combined_df.duplicated().sum())

	# Count of column datatypes (for quick schema sanity-check).
	print(combined_df.dtypes.apply(lambda x: x.name).value_counts())


if __name__ == "__main__":
	main()


