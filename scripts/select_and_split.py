"""Create leakage-safe train/validation/test splits for both task tracks.

The exam-score and placement source files share no student identifier, so the
two tracks are split and modelled independently. Splits are created from the
cleaned raw feature values. Imputation, scaling, and one-hot encoding are
learned later inside each model pipeline, using the training split only.

Outputs:
- data/splits/exam_score_{train,val,test}.csv
- data/splits/placement_{train,val,test}.csv
- data/splits/split_manifest.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EXAM_INPUT_FILE = INTERIM_DIR / "StudentPerformanceFactors_clean.csv"
PLACEMENT_INPUT_FILE = INTERIM_DIR / "placement_clean.csv"
RANDOM_STATE = 42


def _feature_columns(frame: pd.DataFrame, target: str, drop: tuple[str, ...] = ()) -> list[str]:
    return [column for column in frame.columns if column != target and column not in drop]


def split_track(
    frame: pd.DataFrame,
    target: str,
    task: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return train/validation/test frames and a small manifest entry."""
    X = frame[feature_columns]
    y = frame[target]

    stratify_arg = y if task == "classification" else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=stratify_arg,
    )
    stratify_arg2 = y_temp if task == "classification" else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=stratify_arg2,
    )

    train = X_train.assign(**{target: y_train})
    val = X_val.assign(**{target: y_val})
    test = X_test.assign(**{target: y_test})

    manifest = {
        "target": target,
        "task": task,
        "feature_columns": feature_columns,
        "rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
    }
    return train, val, test, manifest


def main() -> None:
    if not EXAM_INPUT_FILE.exists() or not PLACEMENT_INPUT_FILE.exists():
        raise FileNotFoundError(
            "Cleaned source files are missing. Run python scripts/preprocess.py first."
        )

    exam = pd.read_csv(EXAM_INPUT_FILE)
    placement = pd.read_csv(PLACEMENT_INPUT_FILE)

    exam_features = _feature_columns(exam, "Exam_Score")
    placement_features = _feature_columns(
        placement, "Placement", drop=("College_ID",)
    )

    exam_train, exam_val, exam_test, exam_manifest = split_track(
        exam, "Exam_Score", "regression", exam_features
    )
    placement_train, placement_val, placement_test, placement_manifest = split_track(
        placement, "Placement", "classification", placement_features
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    exam_train.to_csv(SPLITS_DIR / "exam_score_train.csv", index=False)
    exam_val.to_csv(SPLITS_DIR / "exam_score_val.csv", index=False)
    exam_test.to_csv(SPLITS_DIR / "exam_score_test.csv", index=False)
    placement_train.to_csv(SPLITS_DIR / "placement_train.csv", index=False)
    placement_val.to_csv(SPLITS_DIR / "placement_val.csv", index=False)
    placement_test.to_csv(SPLITS_DIR / "placement_test.csv", index=False)

    manifest = {
        "random_state": RANDOM_STATE,
        "split_policy": "70% train / 15% validation / 15% test",
        "exam_score": exam_manifest,
        "placement": placement_manifest,
        "note": (
            "Feature values are kept in their cleaned source form. Encoding and "
            "scaling are fitted inside the training pipelines."
        ),
    }
    (SPLITS_DIR / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Exam rows: {len(exam_train)} / {len(exam_val)} / {len(exam_test)}")
    print(
        f"Placement rows: {len(placement_train)} / {len(placement_val)} / {len(placement_test)}"
    )
    print(f"Saved splits and manifest to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
