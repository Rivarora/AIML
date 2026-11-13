"""Investigate structural and semantic leakage risks in both task tracks.

This script deliberately avoids the phrase "leakage-free".  It reports what can
be checked from the data and what cannot:

* target columns in the feature list,
* duplicate feature combinations and conflicting labels,
* strongly predictive individual columns,
* whether the placement target is derivable from its features, and
* whether the independent/synthetic source construction limits conclusions.

Outputs:
- outputs/evaluation/leakage_report.csv
- outputs/evaluation/leakage_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, roc_auc_score

from model_pipelines import (
    EVALUATION_DIR,
    EXAM_MODEL_SPECS,
    PLACEMENT_MODEL_SPECS,
    ModelSpec,
    load_split,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUALITY_FILE = PROJECT_ROOT / "outputs" / "quality" / "data_quality.json"
PROVENANCE_FILE = PROJECT_ROOT / "data" / "interim" / "provenance.json"
REPORT_CSV = EVALUATION_DIR / "leakage_report.csv"
REPORT_JSON = EVALUATION_DIR / "leakage_report.json"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _duplicate_summary(
    frame: pd.DataFrame, features: list[str], target: str
) -> dict[str, int]:
    grouped = frame.groupby(features, dropna=False)[target]
    sizes = grouped.size()
    repeated = sizes[sizes > 1]
    unique_targets = grouped.nunique()
    conflicting = unique_targets[unique_targets > 1]
    return {
        "repeated_feature_combinations": int(len(repeated)),
        "duplicate_rows_beyond_first": int((repeated - 1).sum()),
        "rows_in_repeated_combinations": int(repeated.sum()),
        "conflicting_target_combinations": int(len(conflicting)),
        "rows_in_conflicting_combinations": int(sizes.loc[conflicting.index].sum())
        if len(conflicting)
        else 0,
    }


def _overlap_summary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    features: list[str],
    target: str,
) -> dict[str, int]:
    left_groups = left.groupby(features, dropna=False)[target].apply(set)
    right_groups = right.groupby(features, dropna=False)[target].apply(set)
    shared = set(left_groups.index) & set(right_groups.index)
    conflicts = sum(
        bool(left_groups[index] ^ right_groups[index]) for index in shared
    )
    return {
        "shared_feature_combinations": int(len(shared)),
        "conflicting_target_combinations": int(conflicts),
    }


def _feature_association_rows(spec: ModelSpec, frame: pd.DataFrame) -> list[dict]:
    features = spec.features
    target = spec.target
    if spec.task == "exam_score":
        y = frame[target].astype(float)
    else:
        y = (frame[target].astype(str).str.lower() == "yes").astype(int)

    encoded = frame[features].copy()
    categorical = list(spec.categorical_features)
    for column in categorical:
        encoded[column] = pd.factorize(encoded[column], sort=True)[0]

    if spec.task == "exam_score":
        mi_values = mutual_info_regression(encoded, y, random_state=42)
    else:
        mi_values = mutual_info_classif(encoded, y, random_state=42)

    rows: list[dict] = []
    for index, feature in enumerate(features):
        series = encoded[feature].astype(float)
        correlation = float(series.corr(y)) if series.std(ddof=0) > 0 else 0.0
        single_feature_auc = None
        if spec.task == "placement" and feature in spec.numeric_features:
            auc = float(roc_auc_score(y, series))
            single_feature_auc = max(auc, 1.0 - auc)

        flags: list[str] = []
        if abs(correlation) >= 0.70:
            flags.append("strong_linear_association")
        if spec.task == "placement" and single_feature_auc is not None and single_feature_auc >= 0.90:
            flags.append("single_feature_auc_at_least_0.90")
        mi_threshold = 1.0 if spec.task == "exam_score" else 0.30
        if float(mi_values[index]) >= mi_threshold:
            flags.append("high_mutual_information")

        rows.append(
            {
                "task": spec.task,
                "feature": feature,
                "feature_type": "categorical" if feature in categorical else "numeric",
                "pearson_correlation": correlation,
                "absolute_correlation": abs(correlation),
                "single_feature_roc_auc": single_feature_auc,
                "mutual_information": float(mi_values[index]),
                "mutual_information_threshold": mi_threshold,
                "suspicion_flags": "; ".join(flags) if flags else "none",
                "suspicious": bool(flags),
                "note": (
                    "Correlation/association is descriptive and is not evidence "
                    "of a causal relationship."
                ),
            }
        )
    return rows


def _placement_rule_check() -> dict:
    placement = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "placement_clean.csv")
    y = (placement["Placement"].astype(str).str.lower() == "yes").astype(int)
    conditions = {
        "IQ": placement["IQ"] > 110,
        "CGPA": placement["CGPA"] > 8,
        "Projects_Completed": placement["Projects_Completed"] > 1,
        "Communication_Skills": placement["Communication_Skills"] > 7,
    }
    votes = pd.DataFrame(conditions).sum(axis=1)
    prediction = (votes >= 3).astype(int)
    return {
        "description": (
            "Placement = Yes when at least three of IQ > 110, CGPA > 8, "
            "Projects_Completed > 1, and Communication_Skills > 7 hold."
        ),
        "rows": int(len(placement)),
        "accuracy": float(accuracy_score(y, prediction)),
        "errors": int((prediction != y).sum()),
        "target_is_derivable_from_features": bool((prediction == y).all()),
        "interpretation": (
            "The placement target is a deterministic function of four features. "
            "A flexible model can therefore reproduce the label without learning "
            "a transferable real-world relationship."
        ),
    }


def main() -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    quality = _read_json(QUALITY_FILE)
    provenance = _read_json(PROVENANCE_FILE)

    checks: list[dict] = []
    association_rows: list[dict] = []

    # One representative feature definition per task is enough for the
    # structural checks; both tasks use the same split files.
    for spec in (EXAM_MODEL_SPECS[0], PLACEMENT_MODEL_SPECS[0]):
        train, val, test = load_split(spec.prefix)
        features = spec.features
        target = spec.target

        checks.append(
            {
                "task": spec.task,
                "check": "target column absent from model features",
                "result": target not in features,
                "value": target in features,
                "severity": "critical" if target in features else "info",
                "detail": f"Model features: {', '.join(features)}",
            }
        )

        for split_name, frame in (("train", train), ("validation", val), ("test", test)):
            summary = _duplicate_summary(frame, features, target)
            for key, value in summary.items():
                severity = "review" if value else "info"
                checks.append(
                    {
                        "task": spec.task,
                        "check": f"{split_name} {key}",
                        "result": value == 0,
                        "value": value,
                        "severity": severity,
                        "detail": (
                            "No repeated feature combinations found."
                            if value == 0
                            else "Repeated feature combinations or conflicting labels were found; inspect before claiming leakage-free data."
                        ),
                    }
                )

        for left_name, right_name, left, right in (
            ("train", "validation", train, val),
            ("train", "test", train, test),
            ("validation", "test", val, test),
        ):
            overlap = _overlap_summary(left, right, features, target)
            for key, value in overlap.items():
                severity = "review" if value else "info"
                checks.append(
                    {
                        "task": spec.task,
                        "check": f"{left_name}/{right_name} {key}",
                        "result": value == 0,
                        "value": value,
                        "severity": severity,
                        "detail": (
                            "No shared feature combinations found."
                            if value == 0
                            else "Shared feature combinations exist; compare their targets before interpreting model performance."
                        ),
                    }
                )

        if spec.name in ("LinearRegression", "LogisticRegression"):
            association_rows.extend(_feature_association_rows(spec, train))

    rule = _placement_rule_check()
    checks.append(
        {
            "task": "placement",
            "check": "target is derivable from four features",
            "result": not rule["target_is_derivable_from_features"],
            "value": rule["target_is_derivable_from_features"],
            "severity": "critical",
            "detail": rule["interpretation"],
        }
    )

    # Exact duplicate of the target under another feature name.
    for spec in (EXAM_MODEL_SPECS[0], PLACEMENT_MODEL_SPECS[0]):
        train, _, _ = load_split(spec.prefix)
        if spec.task == "exam_score":
            y = train[spec.target].astype(float).to_numpy()
        else:
            y = (train[spec.target].astype(str).str.lower() == "yes").astype(int).to_numpy()
        exact_matches = []
        for feature in spec.features:
            values = pd.to_numeric(train[feature], errors="coerce").to_numpy(dtype=float)
            if np.allclose(values, y, equal_nan=False):
                exact_matches.append(feature)
        checks.append(
            {
                "task": spec.task,
                "check": "feature exactly reproduces the target",
                "result": not exact_matches,
                "value": exact_matches,
                "severity": "critical" if exact_matches else "info",
                "detail": "No exact duplicate target column found."
                if not exact_matches
                else f"Exact target copies: {', '.join(exact_matches)}",
            }
        )

    linkage = quality.get("linkage") or provenance.get("linkage") or {}
    shared_columns = linkage.get("shared_columns", [])
    genuinely_linked = bool(linkage.get("genuinely_linked", False))
    synthetic_used = bool(linkage.get("synthetic_pairing_used_in_modeling", False))
    row_order_status = linkage.get("row_order_pairing_status", "unknown")

    synthetic_pairing = {
        "genuinely_linked": genuinely_linked,
        "shared_student_identifier": linkage.get("shared_student_identifier"),
        "shared_columns": shared_columns,
        "synthetic_pairing_used_in_modeling": synthetic_used,
        "row_order_pairing_status": row_order_status,
        "investigation": linkage.get("investigation", ""),
        "limitation": (
            "The source datasets are synthetic and share no genuine student "
            "identifier. Historical row-order/synthetic pairing was removed. "
            "No cross-track student-level conclusion is supported."
        ),
    }
    checks.append(
        {
            "task": "both",
            "check": "genuine student-level link exists",
            "result": genuinely_linked,
            "value": genuinely_linked,
            "severity": "critical" if not genuinely_linked else "info",
            "detail": synthetic_pairing["limitation"],
        }
    )

    suspicious_features = [row for row in association_rows if row["suspicious"]]
    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(REPORT_CSV, index=False)

    report = {
        "generated_by": "scripts/leakage_investigation.py",
        "leakage_free_claim": False,
        "verdict": (
            "No direct target column is present in the model features, but the "
            "placement target is deterministic in the source and the datasets are "
            "synthetic/independent. The data therefore cannot be certified "
            "leakage-free, and high placement scores are not evidence of "
            "real-world predictive performance."
        ),
        "checks": checks,
        "feature_associations": association_rows,
        "suspicious_features": suspicious_features,
        "placement_rule": rule,
        "synthetic_pairing": synthetic_pairing,
        "method_notes": [
            "Duplicate and overlap checks use the selected model feature subset, not every raw column.",
            "Correlation and mutual information are descriptive associations, not causal effects.",
            "Mutual information for categorical columns uses factorized category codes and is approximate.",
            "Structural checks cannot prove semantic leakage-free data.",
        ],
    }
    REPORT_JSON.write_text(
        json.dumps(report, indent=2, allow_nan=False, default=float), encoding="utf-8"
    )

    print(f"Saved leakage CSV: {REPORT_CSV}")
    print(f"Saved leakage JSON: {REPORT_JSON}")
    print(f"Placement rule accuracy: {rule['accuracy']:.4f}")
    print(f"Suspicious feature associations: {len(suspicious_features)}")


if __name__ == "__main__":
    main()
