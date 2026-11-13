"""Validate and prepare the two source datasets as separate task tracks.

The raw files do not contain a shared student identifier.  Consequently,
exam-score and placement records are never joined for the main modeling
workflow.  A row-order pairing is retained only as an explicitly labelled
experimental artifact for backwards-compatible exploratory work.
"""
from __future__ import annotations

import json
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
    "Parental_Education_Level": {"High School": 0, "College": 1, "Postgraduate": 2},
    "Distance_from_Home": {"Near": 0, "Moderate": 1, "Far": 2},
    "Gender": {"Female": 0, "Male": 1},
    "Internship_Experience": {"No": 0, "Yes": 1},
}
PLACEMENT_CATEGORICAL_MAPS = {"Placement": {"No": 0, "Yes": 1}}

STUDENT_COLUMNS = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores",
    "Motivation_Level", "Internet_Access", "Tutoring_Sessions", "Family_Income",
    "Teacher_Quality", "School_Type", "Peer_Influence", "Physical_Activity",
    "Learning_Disabilities", "Parental_Education_Level", "Distance_from_Home",
    "Gender", "Exam_Score",
]
PLACEMENT_COLUMNS = [
    "College_ID", "IQ", "Prev_Sem_Result", "CGPA", "Academic_Performance",
    "Internship_Experience", "Extra_Curricular_Score", "Communication_Skills",
    "Projects_Completed", "Placement",
]


def _validate_schema(frame: pd.DataFrame, expected: list[str], name: str) -> None:
    if list(frame.columns) != expected:
        raise ValueError(f"{name} columns do not match the expected schema: {list(frame.columns)}")
    if frame.empty:
        raise ValueError(f"{name} contains no records.")


def _validate_numeric(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any():
            raise ValueError(f"{name}.{column} contains non-numeric values.")
        frame[column] = converted


def _validate_categories(frame: pd.DataFrame, mappings: dict[str, dict[str, int]], name: str) -> None:
    for column, mapping in mappings.items():
        if column not in frame:
            continue
        unexpected = sorted(set(frame[column].dropna().unique()) - set(mapping))
        if unexpected:
            raise ValueError(f"{name}.{column} contains unexpected categories: {unexpected}")


def _validate_ranges(frame: pd.DataFrame, ranges: dict[str, tuple[float, float]], name: str) -> None:
    for column, (minimum, maximum) in ranges.items():
        invalid = ~frame[column].between(minimum, maximum)
        if invalid.any():
            raise ValueError(
                f"{name}.{column} contains {int(invalid.sum())} values outside "
                f"{minimum:g}-{maximum:g}."
            )


def clean_student_data() -> tuple[pd.DataFrame, dict]:
    """Validate, explicitly remove incomplete duplicates, and bound exam scores."""
    frame = pd.read_csv(STUDENT_FILE)
    _validate_schema(frame, STUDENT_COLUMNS, "StudentPerformanceFactors")
    _validate_numeric(
        frame,
        ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
         "Tutoring_Sessions", "Physical_Activity", "Exam_Score"],
        "StudentPerformanceFactors",
    )
    _validate_categories(frame, STUDENT_CATEGORICAL_MAPS, "StudentPerformanceFactors")
    missing = {column: int(count) for column, count in frame.isna().sum().items() if count}
    duplicate_count = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    incomplete_count = int(frame.isna().any(axis=1).sum())
    frame = frame.dropna().reset_index(drop=True)
    invalid_score_count = int((~frame["Exam_Score"].between(0, 100)).sum())
    frame["Exam_Score"] = frame["Exam_Score"].clip(0, 100)
    _validate_ranges(
        frame,
        {"Hours_Studied": (0, 100), "Attendance": (0, 100),
         "Sleep_Hours": (0, 24), "Previous_Scores": (0, 100),
         "Tutoring_Sessions": (0, 100), "Physical_Activity": (0, 24),
         "Exam_Score": (0, 100)},
        "StudentPerformanceFactors",
    )
    return frame, {
        "source_rows": int(len(pd.read_csv(STUDENT_FILE))),
        "missing_values": missing,
        "duplicates_removed": duplicate_count,
        "incomplete_rows_removed": incomplete_count,
        "exam_scores_bounded": invalid_score_count,
        "output_rows": int(len(frame)),
    }


def clean_placement_data() -> tuple[pd.DataFrame, dict]:
    """Validate and encode the independent placement dataset."""
    frame = pd.read_csv(PLACEMENT_FILE)
    _validate_schema(frame, PLACEMENT_COLUMNS, "college_student_placement_dataset")
    numeric = ["IQ", "Prev_Sem_Result", "CGPA", "Academic_Performance",
               "Extra_Curricular_Score", "Communication_Skills", "Projects_Completed"]
    _validate_numeric(frame, numeric, "college_student_placement_dataset")
    _validate_categories(frame, PLACEMENT_CATEGORICAL_MAPS, "college_student_placement_dataset")
    missing = {column: int(count) for column, count in frame.isna().sum().items() if count}
    if missing:
        raise ValueError(f"Placement data contains missing values: {missing}")
    invalid_cgpa_count = int((~frame["CGPA"].between(0, 10)).sum())
    frame["CGPA"] = frame["CGPA"].clip(0, 10)
    _validate_ranges(
        frame,
        {"IQ": (0, 250), "Prev_Sem_Result": (0, 10), "CGPA": (0, 10),
         "Academic_Performance": (0, 10), "Extra_Curricular_Score": (0, 10),
         "Communication_Skills": (0, 10), "Projects_Completed": (0, 100)},
        "college_student_placement_dataset",
    )
    duplicate_count = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    encoded = frame.drop(columns=["College_ID"]).copy()
    encoded["Internship_Experience"] = encoded["Internship_Experience"].map(
        STUDENT_CATEGORICAL_MAPS["Internship_Experience"]
    )
    encoded["Placement"] = encoded["Placement"].map(PLACEMENT_CATEGORICAL_MAPS["Placement"])
    return encoded, {
        "source_rows": int(len(frame)),
        "missing_values": missing,
        "duplicates_removed": duplicate_count,
        "cgpa_values_bounded": invalid_cgpa_count,
        "output_rows": int(len(encoded)),
    }


def encode_student_data(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = frame.copy()
    for column, mapping in STUDENT_CATEGORICAL_MAPS.items():
        if column in encoded:
            encoded[column] = encoded[column].map(mapping)
    peer_dummies = pd.get_dummies(encoded.pop("Peer_Influence"), prefix="Peer", dtype=int)
    encoded = pd.concat([encoded, peer_dummies], axis=1)
    if encoded.isna().any().any():
        raise ValueError(f"Unmapped or missing student values remain in: {encoded.columns[encoded.isna().any()].tolist()}")
    return encoded


def build_synthetic_dataset(student_df: pd.DataFrame, placement_df: pd.DataFrame) -> pd.DataFrame:
    """Create an isolated, clearly labelled row-order artifact (never main input)."""
    count = min(len(student_df), len(placement_df))
    return pd.concat(
        [
            pd.Series(range(1, count + 1), name="synthetic_pair_id"),
            student_df.iloc[:count].reset_index(drop=True),
            placement_df.iloc[:count].reset_index(drop=True),
        ],
        axis=1,
    )


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    student_df, student_quality = clean_student_data()
    placement_df, placement_quality = clean_placement_data()
    encoded_student = encode_student_data(student_df)

    student_df.to_csv(INTERIM_DIR / "StudentPerformanceFactors_clean.csv", index=False)
    placement_df.to_csv(INTERIM_DIR / "placement_clean.csv", index=False)
    encoded_student.to_csv(PROCESSED_DIR / "exam_score_preprocessed.csv", index=False)
    placement_df.to_csv(PROCESSED_DIR / "placement_preprocessed.csv", index=False)

    synthetic = build_synthetic_dataset(student_df, placement_df)
    synthetic.to_csv(INTERIM_DIR / "combined_synthetic_dataset.csv", index=False)
    encoded_synthetic = pd.concat(
        [encode_student_data(student_df.iloc[:len(synthetic)]).reset_index(drop=True),
         placement_df.iloc[:len(synthetic)].reset_index(drop=True)],
        axis=1,
    )
    encoded_synthetic.to_csv(PROCESSED_DIR / "combined_encoded.csv", index=False)
    encoded_synthetic.to_csv(PROCESSED_DIR / "final_preprocessed_dataset.csv", index=False)

    provenance = {
        "sources": [STUDENT_FILE.name, PLACEMENT_FILE.name],
        "source_records_linked": False,
        "link_key": None,
        "synthetic_pairing": {
            "used": True,
            "purpose": "backwards-compatible experimental artifact only",
            "main_modeling_workflow": False,
            "method": "truncate each independently cleaned file and concatenate by row order",
        },
        "student_quality": student_quality,
        "placement_quality": placement_quality,
    }
    (INTERIM_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Exam rows: {len(encoded_student)} | Placement rows: {len(placement_df)}")
    print(f"Experimental synthetic rows (not used for modeling): {len(synthetic)}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
