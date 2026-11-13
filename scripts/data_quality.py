"""Generate machine-readable data-quality and provenance reports."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from preprocess import (
    PLACEMENT_CATEGORICAL_MAPS,
    PLACEMENT_FILE,
    STUDENT_CATEGORICAL_MAPS,
    STUDENT_FILE,
)


ROOT = Path(__file__).resolve().parent.parent
QUALITY_DIR = ROOT / "outputs" / "quality"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"


def _dataset_report(path: Path, name: str, target: str | None = None) -> dict:
    frame = pd.read_csv(path)
    report = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_values_by_column": {k: int(v) for k, v in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "dtypes": {k: str(v) for k, v in frame.dtypes.items()},
    }
    if target and target in frame:
        report["target_distribution"] = {
            str(k): int(v) for k, v in frame[target].value_counts(dropna=False).items()
        }
    return report


def _unexpected(frame: pd.DataFrame, mappings: dict[str, dict[str, int]]) -> dict:
    return {
        column: sorted(set(frame[column].dropna().unique()) - set(mapping))
        for column, mapping in mappings.items()
        if column in frame and set(frame[column].dropna().unique()) - set(mapping)
    }


def build_report() -> dict:
    student_raw = pd.read_csv(STUDENT_FILE)
    placement_raw = pd.read_csv(PLACEMENT_FILE)
    exam = pd.read_csv(PROCESSED_DIR / "exam_score_preprocessed.csv")
    placement = pd.read_csv(PROCESSED_DIR / "placement_preprocessed.csv")

    report = {
        "source_records_genuinely_linked": False,
        "shared_student_identifier": None,
        "synthetic_pairing_used": True,
        "synthetic_pairing_scope": "experimental artifact only; excluded from main modeling workflow",
        "provenance": {
            "exam_source": STUDENT_FILE.name,
            "placement_source": PLACEMENT_FILE.name,
            "link_investigation": (
                "No common student identifier exists. StudentPerformanceFactors has no ID "
                "column and placement College_ID has no counterpart in the exam source."
            ),
        },
        "datasets": {
            "exam_raw": _dataset_report(STUDENT_FILE, "exam_raw", "Exam_Score"),
            "placement_raw": _dataset_report(PLACEMENT_FILE, "placement_raw", "Placement"),
            "exam_processed": _dataset_report(
                PROCESSED_DIR / "exam_score_preprocessed.csv", "exam_processed", "Exam_Score"
            ),
            "placement_processed": _dataset_report(
                PROCESSED_DIR / "placement_preprocessed.csv", "placement_processed", "Placement"
            ),
        },
        "invalid_ranges": {
            "exam_raw_exam_score_outside_0_100": int((~pd.to_numeric(student_raw["Exam_Score"], errors="coerce").between(0, 100)).sum()),
            "exam_processed_exam_score_outside_0_100": int((~exam["Exam_Score"].between(0, 100)).sum()),
            "placement_raw_cgpa_outside_0_10": int((~placement_raw["CGPA"].between(0, 10)).sum()),
        },
        "unexpected_categories": {
            "exam_raw": _unexpected(student_raw, STUDENT_CATEGORICAL_MAPS),
            "placement_raw": _unexpected(placement_raw, PLACEMENT_CATEGORICAL_MAPS),
        },
        "target_distributions": {
            "exam_score": {str(k): int(v) for k, v in exam["Exam_Score"].value_counts().sort_index().items()},
            "placement": {str(k): int(v) for k, v in placement["Placement"].value_counts().sort_index().items()},
        },
        "splits": {},
    }
    for prefix, target in (("exam_score", "Exam_Score"), ("placement", "Placement")):
        report["splits"][prefix] = {}
        for split in ("train", "val", "test"):
            path = SPLITS_DIR / f"{prefix}_{split}.csv"
            if path.exists():
                frame = pd.read_csv(path)
                report["splits"][prefix][split] = {
                    "rows": int(len(frame)),
                    "target": target,
                    "target_values": {str(k): int(v) for k, v in frame[target].value_counts().items()},
                }
    return report


def main() -> None:
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    (QUALITY_DIR / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = []
    for dataset, values in report["datasets"].items():
        for metric, value in values.items():
            if isinstance(value, dict):
                continue
            rows.append({"section": "dataset", "name": dataset, "metric": metric, "value": value})
    for metric, value in report["invalid_ranges"].items():
        rows.append({"section": "invalid_ranges", "name": "all", "metric": metric, "value": value})
    rows.extend([
        {"section": "linkage", "name": "all", "metric": "source_records_genuinely_linked", "value": report["source_records_genuinely_linked"]},
        {"section": "linkage", "name": "all", "metric": "synthetic_pairing_used", "value": report["synthetic_pairing_used"]},
    ])
    for track, splits in report["splits"].items():
        for split, values in splits.items():
            rows.append({"section": "split", "name": f"{track}_{split}", "metric": "rows", "value": values["rows"]})
    pd.DataFrame(rows).to_csv(QUALITY_DIR / "data_quality.csv", index=False)
    print(f"Wrote {QUALITY_DIR / 'data_quality.json'}")
    print(f"Wrote {QUALITY_DIR / 'data_quality.csv'}")


if __name__ == "__main__":
    main()
