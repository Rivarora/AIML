"""Validate the two independent processed task tracks.

This replaces the old check that read the row-order combined file. The combined
file is an experimental artifact and must not be treated as one student table.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAM_FILE = PROJECT_ROOT / "data" / "processed" / "exam_score_preprocessed.csv"
PLACEMENT_FILE = PROJECT_ROOT / "data" / "processed" / "placement_preprocessed.csv"

EXAM_RANGES = {
    "Exam_Score": (0, 100),
    "Attendance": (0, 100),
    "Hours_Studied": (0, 168),
    "Previous_Scores": (0, 100),
    "Sleep_Hours": (0, 24),
    "Tutoring_Sessions": (0, 100),
    "Physical_Activity": (0, 24),
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


def check_track(
    path: Path,
    track: str,
    target: str,
    ranges: dict[str, tuple[float, float]],
    other_target: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{track} file is missing: {path}")

    frame = pd.read_csv(path)
    problems: list[str] = []

    if frame.empty:
        problems.append("no rows")
    if target not in frame.columns:
        problems.append(f"missing target column {target}")
    if other_target in frame.columns:
        problems.append(f"contains the other task target {other_target}")
    if "synthetic_pair_id" in frame.columns:
        problems.append("contains synthetic pairing identifier")

    null_count = int(frame.isna().sum().sum())
    duplicate_count = int(frame.duplicated().sum())
    if null_count:
        problems.append(f"{null_count} missing values")
    if duplicate_count:
        problems.append(f"{duplicate_count} duplicate rows")

    for column, (minimum, maximum) in ranges.items():
        if column not in frame:
            continue
        invalid = ~pd.to_numeric(frame[column], errors="coerce").between(minimum, maximum)
        if invalid.any():
            problems.append(f"{column} has {int(invalid.sum())} values outside {minimum:g}-{maximum:g}")

    if problems:
        raise ValueError(f"{track} failed validation: {', '.join(problems)}")

    print(f"{track}: {len(frame)} rows, {len(frame.columns)} columns, 0 nulls, 0 duplicates")
    if target == "Exam_Score":
        print(f"  Exam_Score range: {frame[target].min()}-{frame[target].max()}")
    else:
        counts = frame[target].value_counts().sort_index().to_dict()
        print(f"  Placement counts: {counts}")
        if not set(frame[target].unique()).issubset({0, 1}):
            raise ValueError("Placement contains values other than 0 and 1.")


def main() -> None:
    check_track(EXAM_FILE, "exam_score", "Exam_Score", EXAM_RANGES, "Placement")
    check_track(PLACEMENT_FILE, "placement", "Placement", PLACEMENT_RANGES, "Exam_Score")
    print("Independent processed tracks are valid.")


if __name__ == "__main__":
    main()
