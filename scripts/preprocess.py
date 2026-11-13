"""Validate and prepare the two source datasets as separate task tracks.

The raw files do not contain a shared student identifier, so exam-score and
placement records cannot be joined at student level. No row-order pairing is
created. Each track is cleaned independently and modelled from its own
features and target. All preprocessing used by the models (imputation,
scaling, and one-hot encoding) is fitted inside the training pipeline.

The script is deterministic: it does not sample rows and it uses no random
state. Downstream splitting uses a fixed seed (42).
"""
from __future__ import annotations

import hashlib
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
PEER_INFLUENCE_CATEGORIES = ("Negative", "Neutral", "Positive")
PLACEMENT_CATEGORICAL_MAPS = {"Placement": {"No": 0, "Yes": 1}}
PLACEMENT_CATEGORICAL_EXPECTED = {
    "Internship_Experience": {"No", "Yes"},
    "Placement": {"No", "Yes"},
}

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

STUDENT_NUMERIC_COLUMNS = [
    "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
    "Tutoring_Sessions", "Physical_Activity", "Exam_Score",
]
PLACEMENT_NUMERIC_COLUMNS = [
    "IQ", "Prev_Sem_Result", "CGPA", "Academic_Performance",
    "Extra_Curricular_Score", "Communication_Skills", "Projects_Completed",
]

# Deliberately wider than the observed data: these are validity bounds, not
# descriptive min/max values. Values outside them indicate a real data error.
STUDENT_RANGES = {
    "Hours_Studied": (0, 168),
    "Attendance": (0, 100),
    "Sleep_Hours": (0, 24),
    "Previous_Scores": (0, 100),
    "Tutoring_Sessions": (0, 100),
    "Physical_Activity": (0, 24),
    "Exam_Score": (0, 100),
}
PLACEMENT_RANGES = {
    "IQ": (0, 250),
    "Prev_Sem_Result": (0, 10),
    "CGPA": (0, 10),
    "Academic_Performance": (0, 10),
    "Extra_Curricular_Score": (0, 10),
    "Communication_Skills": (0, 10),
    "Projects_Completed": (0, 100),
}


def _validate_schema(frame: pd.DataFrame, expected: list[str], name: str) -> None:
    if list(frame.columns) != expected:
        raise ValueError(f"{name} columns do not match the expected schema: {list(frame.columns)}")
    if frame.empty:
        raise ValueError(f"{name} contains no records.")
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{name} contains duplicate column names: {duplicated}")


def _validate_numeric(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        converted = pd.to_numeric(series, errors="coerce")
        if converted.isna().any():
            raise ValueError(f"{name}.{column} contains non-numeric values.")
        frame[column] = converted


def _validate_category_column(
    frame: pd.DataFrame, column: str, allowed: set[str], name: str
) -> None:
    series = frame[column]
    if not pd.api.types.is_string_dtype(series):
        raise ValueError(f"{name}.{column} must contain text categories, got {series.dtype}.")
    unexpected = sorted(set(series.dropna().unique()) - allowed)
    if unexpected:
        raise ValueError(f"{name}.{column} contains unexpected categories: {unexpected}")


def _validate_categories(
    frame: pd.DataFrame, mappings: dict[str, dict[str, int]], name: str
) -> None:
    for column, mapping in mappings.items():
        if column in frame:
            _validate_category_column(frame, column, set(mapping), name)


def _validate_ranges(
    frame: pd.DataFrame, ranges: dict[str, tuple[float, float]], name: str
) -> None:
    for column, (minimum, maximum) in ranges.items():
        if column not in frame:
            continue
        invalid = ~pd.to_numeric(frame[column], errors="coerce").between(minimum, maximum)
        if invalid.any():
            raise ValueError(
                f"{name}.{column} contains {int(invalid.sum())} values outside "
                f"{minimum:g}-{maximum:g}."
            )


def _count_outside(value: pd.Series, minimum: float, maximum: float) -> int:
    return int((~pd.to_numeric(value, errors="coerce").between(minimum, maximum)).sum())


def _missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int(count) for column, count in frame.isna().sum().items() if count}


def _file_fingerprint(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def clean_student_data() -> tuple[pd.DataFrame, dict]:
    """Validate the exam source, remove unusable rows, and bound exam scores."""
    raw = pd.read_csv(STUDENT_FILE)
    _validate_schema(raw, STUDENT_COLUMNS, "StudentPerformanceFactors")
    _validate_numeric(raw, STUDENT_NUMERIC_COLUMNS, "StudentPerformanceFactors")
    _validate_categories(raw, STUDENT_CATEGORICAL_MAPS, "StudentPerformanceFactors")
    _validate_category_column(
        raw, "Peer_Influence", set(PEER_INFLUENCE_CATEGORIES), "StudentPerformanceFactors"
    )

    missing = _missing_counts(raw)
    duplicate_count = int(raw.duplicated().sum())
    incomplete_rows = int(raw.isna().any(axis=1).sum())

    frame = raw.drop_duplicates().dropna().reset_index(drop=True)
    exam_scores_bounded = _count_outside(frame["Exam_Score"], 0, 100)
    frame["Exam_Score"] = frame["Exam_Score"].clip(0, 100)
    _validate_ranges(frame, STUDENT_RANGES, "StudentPerformanceFactors")

    quality = {
        "source_file": STUDENT_FILE.name,
        "source_rows": int(len(raw)),
        "source_columns": int(len(raw.columns)),
        "identifier_columns": [],
        "missing_values": missing,
        "incomplete_rows_removed": incomplete_rows,
        "duplicates_removed": duplicate_count,
        "exam_scores_bounded": exam_scores_bounded,
        "output_rows": int(len(frame)),
        "output_columns": int(len(frame.columns)),
        "missing_value_policy": "drop rows with any missing value after duplicate removal",
    }
    return frame, quality


def clean_placement_data() -> tuple[pd.DataFrame, dict]:
    """Validate the independent placement dataset and keep raw categories."""
    raw = pd.read_csv(PLACEMENT_FILE)
    _validate_schema(raw, PLACEMENT_COLUMNS, "college_student_placement_dataset")
    _validate_numeric(raw, PLACEMENT_NUMERIC_COLUMNS, "college_student_placement_dataset")
    for column, allowed in PLACEMENT_CATEGORICAL_EXPECTED.items():
        _validate_category_column(raw, column, allowed, "college_student_placement_dataset")
    if not pd.api.types.is_string_dtype(raw["College_ID"]):
        raise ValueError("college_student_placement_dataset.College_ID must contain text identifiers.")

    missing = _missing_counts(raw)
    if missing:
        raise ValueError(f"Placement data contains missing values: {missing}")

    duplicate_count = int(raw.duplicated().sum())
    frame = raw.drop_duplicates().reset_index(drop=True)
    cgpa_values_bounded = _count_outside(frame["CGPA"], 0, 10)
    frame["CGPA"] = frame["CGPA"].clip(0, 10)
    _validate_ranges(frame, PLACEMENT_RANGES, "college_student_placement_dataset")

    quality = {
        "source_file": PLACEMENT_FILE.name,
        "source_rows": int(len(raw)),
        "source_columns": int(len(raw.columns)),
        "missing_values": missing,
        "duplicates_removed": duplicate_count,
        "cgpa_values_bounded": cgpa_values_bounded,
        "output_rows": int(len(frame)),
        "output_columns": int(len(frame.columns)),
        "identifier_columns": ["College_ID"],
        "identifier_removed_for_modeling": "College_ID",
        "college_id_unique_values": int(raw["College_ID"].nunique()),
        "college_id_is_unique": bool(raw["College_ID"].is_unique),
        "missing_value_policy": "raise an error; this source must be complete",
    }
    return frame, quality


def encode_placement_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the backwards-compatible encoded copy used by EDA and reports."""
    encoded = frame.drop(columns=["College_ID"]).copy()
    encoded["Internship_Experience"] = encoded["Internship_Experience"].map(
        STUDENT_CATEGORICAL_MAPS["Internship_Experience"]
    )
    encoded["Placement"] = encoded["Placement"].map(PLACEMENT_CATEGORICAL_MAPS["Placement"])
    if encoded.isna().any().any():
        raise ValueError("Placement encoding produced missing values.")
    return encoded


def encode_student_data(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = frame.copy()
    for column, mapping in STUDENT_CATEGORICAL_MAPS.items():
        if column in encoded:
            encoded[column] = encoded[column].map(mapping)
    peer_dummies = pd.get_dummies(
        encoded.pop("Peer_Influence"), prefix="Peer", dtype=int
    ).reindex(columns=[f"Peer_{value}" for value in PEER_INFLUENCE_CATEGORIES], fill_value=0)
    encoded = pd.concat([encoded, peer_dummies], axis=1)
    if encoded.isna().any().any():
        remaining = encoded.columns[encoded.isna().any()].tolist()
        raise ValueError(f"Unmapped or missing student values remain in: {remaining}")
    return encoded


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    student_df, student_quality = clean_student_data()
    placement_df, placement_quality = clean_placement_data()
    encoded_student = encode_student_data(student_df)
    encoded_placement = encode_placement_data(placement_df)
    student_quality["encoded_columns"] = int(len(encoded_student.columns))
    placement_quality["encoded_columns"] = int(len(encoded_placement.columns))

    student_df.to_csv(INTERIM_DIR / "StudentPerformanceFactors_clean.csv", index=False)
    placement_df.to_csv(INTERIM_DIR / "placement_clean.csv", index=False)
    encoded_student.to_csv(PROCESSED_DIR / "exam_score_preprocessed.csv", index=False)
    encoded_placement.to_csv(PROCESSED_DIR / "placement_preprocessed.csv", index=False)

    # Remove old artifacts from earlier versions. They paired rows by position,
    # which does not identify the same student and must not be used downstream.
    for legacy_path in (
        INTERIM_DIR / "combined_synthetic_dataset.csv",
        PROCESSED_DIR / "combined_encoded.csv",
        PROCESSED_DIR / "final_preprocessed_dataset.csv",
    ):
        legacy_path.unlink(missing_ok=True)

    provenance = {
        "source_files": [
            _file_fingerprint(STUDENT_FILE),
            _file_fingerprint(PLACEMENT_FILE),
        ],
        "linkage": {
            "genuinely_linked": False,
            "shared_student_identifier": None,
            "shared_columns": [],
            "exam_identifier_columns": [],
            "placement_identifier_columns": ["College_ID"],
            "investigation": (
                "StudentPerformanceFactors.csv has no identifier column, and "
                "College_ID exists only in the placement source. It has "
                f"{placement_quality['college_id_unique_values']} unique values "
                f"across {placement_quality['source_rows']} rows, so it is a "
                "college-level identifier, not a student identifier. The files "
                "share no column that could join individual students, so no "
                "genuine student-level link can be constructed."
            ),
            "linkage_status": "unavailable_no_shared_student_identifier",
        },
        "row_order_pairing": {
            "status": "removed",
            "used_in_main_modeling_workflow": False,
            "removed_artifacts": [
                "data/interim/combined_synthetic_dataset.csv",
                "data/processed/combined_encoded.csv",
                "data/processed/final_preprocessed_dataset.csv",
            ],
            "reason": (
                "Row order does not identify the same student. Keeping the "
                "artifact risked implying a false student-level link."
            ),
        },
        "main_modeling_inputs": [
            "data/splits/exam_score_train.csv",
            "data/splits/placement_train.csv",
        ],
        "student_quality": student_quality,
        "placement_quality": placement_quality,
    }
    (INTERIM_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    print(f"Exam rows: {len(encoded_student)} | Placement rows: {len(placement_df)}")
    print("Row-order synthetic artifacts removed; no combined dataset is written.")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
