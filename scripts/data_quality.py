"""Generate machine-readable data-quality and provenance reports.

Outputs:
- outputs/quality/data_quality.json
- outputs/quality/data_quality.csv

The report is built from the actual raw, processed, split, and provenance
files. No quality values are written by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from preprocess import (  # noqa: E402
    PEER_INFLUENCE_CATEGORIES,
    PLACEMENT_CATEGORICAL_EXPECTED,
    PLACEMENT_FILE,
    PLACEMENT_RANGES,
    STUDENT_CATEGORICAL_MAPS,
    STUDENT_FILE,
    STUDENT_RANGES,
)


ROOT = SCRIPT_DIR.parent
QUALITY_DIR = ROOT / "outputs" / "quality"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"
PROVENANCE_FILE = ROOT / "data" / "interim" / "provenance.json"

EXAM_PROCESSED = PROCESSED_DIR / "exam_score_preprocessed.csv"
PLACEMENT_PROCESSED = PROCESSED_DIR / "placement_preprocessed.csv"


def _missing_by_column(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int(count) for column, count in frame.isna().sum().items() if count}


def _invalid_ranges(
    frame: pd.DataFrame, ranges: dict[str, tuple[float, float]]
) -> dict[str, int]:
    invalid: dict[str, int] = {}
    for column, (minimum, maximum) in ranges.items():
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        count = int((~values.between(minimum, maximum)).sum())
        if count:
            invalid[column] = count
    return invalid


def _unexpected_categories(
    frame: pd.DataFrame, expected: dict[str, set[str]]
) -> dict[str, list]:
    unexpected: dict[str, list] = {}
    for column, allowed in expected.items():
        if column not in frame:
            continue
        values = set(frame[column].dropna().unique())
        extra = sorted(values - allowed)
        if extra:
            unexpected[column] = extra
    return unexpected


def _target_summary(frame: pd.DataFrame, target: str, kind: str) -> dict:
    values = frame[target]
    if kind == "numeric":
        return {
            "type": "numeric",
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
        }
    counts = {str(key): int(count) for key, count in values.value_counts().sort_index().items()}
    total = sum(counts.values())
    return {
        "type": "categorical",
        "counts": counts,
        "positive_share": round(counts.get("1", 0) / total, 4) if total else None,
    }


def _dataset_report(
    path: Path,
    target: str,
    target_kind: str,
    ranges: dict[str, tuple[float, float]],
    expected_categories: dict[str, set[str]] | None = None,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    frame = pd.read_csv(path)
    report = {
        "path": path.relative_to(ROOT).as_posix(),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_values_total": int(frame.isna().sum().sum()),
        "missing_values_by_column": _missing_by_column(frame),
        "duplicate_rows": int(frame.duplicated().sum()),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "invalid_ranges": _invalid_ranges(frame, ranges),
        "target": target,
        "target_summary": _target_summary(frame, target, target_kind),
    }
    if expected_categories is not None:
        report["unexpected_categories"] = _unexpected_categories(frame, expected_categories)
    return report


def _flatten_report(report: dict) -> list[dict]:
    rows: list[dict] = []

    def add(section: str, name: str, metric: str, value) -> None:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        rows.append({"section": section, "name": name, "metric": metric, "value": value})

    add("linkage", "all", "source_records_genuinely_linked", report["source_records_genuinely_linked"])
    add("linkage", "all", "synthetic_pairing_used_in_modeling", report["synthetic_pairing_used_in_modeling"])
    add("linkage", "all", "synthetic_artifact_available", report["synthetic_artifact_available"])
    add("linkage", "all", "row_order_pairing_status", report["row_order_pairing_status"])
    add("linkage", "all", "shared_student_identifier", report["shared_student_identifier"])
    add("linkage", "all", "investigation", report["linkage"]["investigation"])

    for track, values in report["processing"].items():
        for metric, value in values.items():
            if not isinstance(value, (dict, list)):
                add("processing", track, metric, value)

    for dataset, values in report["datasets"].items():
        for metric in ("rows", "columns", "missing_values_total", "duplicate_rows"):
            add("dataset", dataset, metric, values[metric])
        for column, count in values["missing_values_by_column"].items():
            add("missing", f"{dataset}.{column}", "count", count)
        for column, count in values["invalid_ranges"].items():
            add("invalid_range", f"{dataset}.{column}", "count", count)
        for column, extra in values.get("unexpected_categories", {}).items():
            add("unexpected_category", f"{dataset}.{column}", "values", extra)
        summary = values["target_summary"]
        if summary["type"] == "numeric":
            for metric in ("min", "max", "mean", "std"):
                add("target", dataset, metric, summary[metric])
        else:
            for label, count in summary["counts"].items():
                add("target_distribution", dataset, label, count)

    for track, splits in report["splits"].items():
        for split, values in splits.items():
            add("split", f"{track}_{split}", "rows", values["rows"])
            for label, count in values["target_values"].items():
                add("split_target", f"{track}_{split}", label, count)
        add("split_total", track, "rows", report["split_row_totals"][track]["sum"])
        add("split_total", track, "matches_processed", report["split_row_totals"][track]["matches_processed"])

    return rows


def build_report() -> dict:
    provenance = {}
    if PROVENANCE_FILE.exists():
        provenance = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))

    student_raw = pd.read_csv(STUDENT_FILE)
    placement_raw = pd.read_csv(PLACEMENT_FILE)

    student_identifiers = [column for column in student_raw.columns if "id" in column.lower()]
    placement_identifiers = [column for column in placement_raw.columns if "id" in column.lower()]
    shared_identifiers = sorted(set(student_identifiers) & set(placement_identifiers))

    unexpected_student = dict(STUDENT_CATEGORICAL_MAPS)
    unexpected_student["Peer_Influence"] = {
        value: index for index, value in enumerate(PEER_INFLUENCE_CATEGORIES)
    }
    student_expected = {column: set(mapping) for column, mapping in unexpected_student.items()}

    datasets = {
        "exam_raw": _dataset_report(
            STUDENT_FILE, "Exam_Score", "numeric", STUDENT_RANGES, student_expected
        ),
        "exam_processed": _dataset_report(
            EXAM_PROCESSED, "Exam_Score", "numeric", STUDENT_RANGES
        ),
        "placement_raw": _dataset_report(
            PLACEMENT_FILE, "Placement", "categorical", PLACEMENT_RANGES,
            PLACEMENT_CATEGORICAL_EXPECTED,
        ),
        "placement_processed": _dataset_report(
            PLACEMENT_PROCESSED, "Placement", "categorical", PLACEMENT_RANGES
        ),
    }

    splits: dict[str, dict] = {}
    split_totals: dict[str, dict] = {}
    expected_rows = {
        "exam_score": datasets["exam_processed"]["rows"],
        "placement": datasets["placement_processed"]["rows"],
    }
    for prefix, target in (("exam_score", "Exam_Score"), ("placement", "Placement")):
        splits[prefix] = {}
        row_sum = 0
        for split in ("train", "val", "test"):
            path = SPLITS_DIR / f"{prefix}_{split}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing split: {path}")
            frame = pd.read_csv(path)
            row_sum += len(frame)
            splits[prefix][split] = {
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "target": target,
                "target_values": {
                    str(key): int(count)
                    for key, count in frame[target].value_counts().sort_index().items()
                },
            }
        split_totals[prefix] = {
            "sum": int(row_sum),
            "processed_rows": int(expected_rows[prefix]),
            "matches_processed": bool(row_sum == expected_rows[prefix]),
        }

    synthetic_used_in_modeling = False
    row_order_status = provenance.get("row_order_pairing", {}).get(
        "status", "removed"
    )
    shared_columns = sorted(set(student_raw.columns) & set(placement_raw.columns))
    linkage = {
        "genuinely_linked": bool(shared_identifiers),
        "shared_student_identifier": shared_identifiers[0] if shared_identifiers else None,
        "shared_columns": shared_columns,
        "candidate_identifier_columns": {
            "exam": student_identifiers,
            "placement": placement_identifiers,
        },
        "investigation": (
            "StudentPerformanceFactors.csv has no identifier column. "
            "College_ID exists only in the placement file, where it has "
            f"{int(placement_raw['College_ID'].nunique())} unique values across "
            f"{len(placement_raw)} rows; it identifies colleges, not students, "
            "and has no counterpart in the exam file. The source files share no "
            "column that could join individual students. No genuine "
            "student-level link can be constructed."
            if not shared_identifiers and "College_ID" in placement_raw.columns
            else (
                "StudentPerformanceFactors.csv has no identifier column. "
                "No shared student identifier was found."
                if not shared_identifiers
                else f"Shared identifier columns: {shared_identifiers}"
            )
        ),
        "synthetic_pairing_used_in_modeling": synthetic_used_in_modeling,
        "row_order_pairing_status": row_order_status,
        "synthetic_artifact": {
            "exists": False,
            "path": None,
            "status": "removed",
            "used_in_main_modeling_workflow": False,
            "note": (
                "Row-order pairing was removed because it does not identify the "
                "same student. No combined dataset is created or consumed."
            ),
        },
    }

    return {
        "source_records_genuinely_linked": linkage["genuinely_linked"],
        "shared_student_identifier": linkage["shared_student_identifier"],
        "synthetic_pairing_used_in_modeling": linkage["synthetic_pairing_used_in_modeling"],
        "synthetic_artifact_available": False,
        "row_order_pairing_status": row_order_status,
        "linkage": linkage,
        "provenance": {
            "source_files": provenance.get("source_files", []),
            "exam_source": STUDENT_FILE.name,
            "placement_source": PLACEMENT_FILE.name,
            "main_modeling_inputs": provenance.get(
                "main_modeling_inputs",
                [
                    "data/splits/exam_score_train.csv",
                    "data/splits/placement_train.csv",
                ],
            ),
        },
        "processing": {
            "exam": provenance.get("student_quality", {}),
            "placement": provenance.get("placement_quality", {}),
        },
        "datasets": datasets,
        "invalid_ranges": {
            name: values["invalid_ranges"] for name, values in datasets.items()
        },
        "unexpected_categories": {
            name: values.get("unexpected_categories", {})
            for name, values in datasets.items()
        },
        "target_distributions": {
            "exam_score": datasets["exam_processed"]["target_summary"],
            "placement": datasets["placement_processed"]["target_summary"],
        },
        "splits": splits,
        "split_row_totals": split_totals,
    }


def main() -> None:
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    (QUALITY_DIR / "data_quality.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(_flatten_report(report)).to_csv(
        QUALITY_DIR / "data_quality.csv", index=False
    )
    print(f"Wrote {QUALITY_DIR / 'data_quality.json'}")
    print(f"Wrote {QUALITY_DIR / 'data_quality.csv'}")


if __name__ == "__main__":
    main()
