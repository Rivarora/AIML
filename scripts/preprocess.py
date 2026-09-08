"""Build the interim and processed datasets from the raw CSV files.

Pipeline steps:
1. Remove duplicate and incomplete rows from the student-performance data.
2. Pair the cleaned student rows with placement rows by row order.
3. Convert categorical values to documented numeric codes.
4. One-hot encode ``Peer_Influence``.
5. Save the intermediate and model-ready datasets.

The placement dataset does not contain a matching student identifier, so the
pairing is explicitly synthetic and row-wise. This limitation should be
considered when interpreting model results.
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STUDENT_FILE = RAW_DIR / "StudentPerformanceFactors.csv"
PLACEMENT_FILE = RAW_DIR / "college_student_placement_dataset.csv"


STUDENT_CATEGORICAL_MAPS = {
    "Parental_Involvement": {"Low": 0, "Medium": 1, "High": 2},
    "Access_to_Resources": {"Low": 0, "Medium": 1, "High": 2},
    "Extracurricular_Activities": {"No": 0, "Yes": 1},
    "Motivation_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Internet_Access": {"No": 0, "Yes": 1},
    "Family_Income": {"Low": 0, "Medium": 1, "High": 2},
    "Teacher_Quality": {"Low": 0, "Medium": 1, "High": 2},
    "School_Type": {"Public": 0, "Private": 1},
    "Learning_Disabilities": {"No": 0, "Yes": 1},
    "Parental_Education_Level": {
        "High School": 0,
        "College": 1,
        "Postgraduate": 2,
    },
    "Distance_from_Home": {"Near": 0, "Moderate": 1, "Far": 2},
    "Gender": {"Female": 0, "Male": 1},
    "Internship_Experience": {"No": 0, "Yes": 1},
}


def clean_student_data() -> pd.DataFrame:
    """Remove duplicate and incomplete student-performance records."""
    student_df = pd.read_csv(STUDENT_FILE)
    student_df = student_df.drop_duplicates()
    student_df = student_df.dropna().reset_index(drop=True)
    return student_df


def build_synthetic_dataset(student_df: pd.DataFrame) -> pd.DataFrame:
    """Pair cleaned student rows with placement rows by row order."""
    placement_df = pd.read_csv(PLACEMENT_FILE)
    if len(placement_df) < len(student_df):
        raise ValueError("The placement dataset has fewer rows than the cleaned student dataset.")

    placement_df = placement_df.iloc[: len(student_df)].reset_index(drop=True)
    student_df = student_df.reset_index(drop=True)

    return pd.concat(
        [
            pd.Series(range(1, len(student_df) + 1), name="synthetic_pair_id"),
            student_df,
            placement_df,
        ],
        axis=1,
    )


def encode_dataset(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns and return the model-ready dataset."""
    encoded_df = combined_df.drop(columns=["synthetic_pair_id", "College_ID"]).copy()

    for column, mapping in STUDENT_CATEGORICAL_MAPS.items():
        encoded_df[column] = encoded_df[column].map(mapping)

    encoded_df["Peer_Influence"] = encoded_df["Peer_Influence"].astype("category")
    peer_dummies = pd.get_dummies(
        encoded_df["Peer_Influence"],
        prefix="Peer",
        dtype=int,
    )
    encoded_df = pd.concat(
        [encoded_df.drop(columns=["Peer_Influence"]), peer_dummies],
        axis=1,
    )

    encoded_df["Placement"] = encoded_df["Placement"].map({"No": 0, "Yes": 1})

    if encoded_df.isnull().any().any():
        invalid_columns = encoded_df.columns[encoded_df.isnull().any()].tolist()
        raise ValueError(f"Unmapped or missing values remain in: {invalid_columns}")

    return encoded_df


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    clean_student_df = clean_student_data()
    clean_student_df.to_csv(
        INTERIM_DIR / "StudentPerformanceFactors_clean.csv",
        index=False,
    )

    combined_df = build_synthetic_dataset(clean_student_df)
    combined_df.to_csv(INTERIM_DIR / "combined_synthetic_dataset.csv", index=False)

    encoded_df = encode_dataset(combined_df)
    encoded_df.to_csv(PROCESSED_DIR / "combined_encoded.csv", index=False)

    final_columns = [column for column in encoded_df.columns if column not in {"Exam_Score", "Placement"}]
    final_columns += ["Exam_Score", "Placement"]
    encoded_df[final_columns].to_csv(
        PROCESSED_DIR / "final_preprocessed_dataset.csv",
        index=False,
    )

    print(f"Clean student rows: {len(clean_student_df)}")
    print(f"Synthetic combined rows: {len(combined_df)}")
    print(f"Processed columns: {len(encoded_df.columns)}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()