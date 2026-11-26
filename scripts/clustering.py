"""Cluster exam-track records without using the exam-score target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_config import EXAM_MODEL_FEATURES, RANDOM_STATE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clustering"


def _load_records() -> pd.DataFrame:
    frames = []
    for split_name in ("train", "val", "test"):
        frame = pd.read_csv(SPLITS_DIR / f"exam_score_{split_name}.csv")
        frame["split"] = split_name
        frames.append(frame)
    records = pd.concat(frames, ignore_index=True)
    missing = [column for column in EXAM_MODEL_FEATURES if column not in records]
    if missing:
        raise ValueError(f"Exam split is missing clustering features: {missing}")
    return records


def _describe_cluster(center: dict[str, float], overall: dict[str, float]) -> str:
    higher = []
    lower = []
    for field, label in (
        ("Attendance", "attendance"),
        ("Previous_Scores", "previous scores"),
        ("Hours_Studied", "study hours"),
    ):
        if center[field] >= overall[field] * 1.05:
            higher.append(label)
        elif center[field] <= overall[field] * 0.95:
            lower.append(label)
    if higher and lower:
        return f"Higher {', '.join(higher)} and lower {', '.join(lower)} than the overall cluster average."
    if higher:
        return f"Higher {', '.join(higher)} than the overall cluster average."
    if lower:
        return f"Lower {', '.join(lower)} than the overall cluster average."
    return "Near the overall average on the measured clustering indicators."


def run(n_clusters: int = 3, compare_hierarchical: bool = False) -> dict:
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    records = _load_records()
    numeric = [
        column for column in EXAM_MODEL_FEATURES
        if pd.api.types.is_numeric_dtype(records[column])
    ]
    categorical = [column for column in EXAM_MODEL_FEATURES if column not in numeric]
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )
    transformed = preprocessor.fit_transform(records[EXAM_MODEL_FEATURES])
    if n_clusters >= len(records):
        raise ValueError("n_clusters must be smaller than the number of records.")
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(transformed)
    score = float(silhouette_score(transformed, labels))
    records["cluster"] = labels

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assignments = records[["split", *EXAM_MODEL_FEATURES, "cluster"]]
    assignments.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)
    overall = records[numeric].mean().to_dict()
    centers = []
    for cluster in range(n_clusters):
        subset = records[records["cluster"] == cluster]
        center = {field: float(subset[field].mean()) for field in numeric}
        centers.append(
            {
                "cluster": cluster,
                "count": int(len(subset)),
                "numeric_center": center,
                "description": _describe_cluster(center, overall),
            }
        )

    summary = {
        "method": "K-Means",
        "n_clusters": n_clusters,
        "features": EXAM_MODEL_FEATURES,
        "target_columns_excluded": ["Exam_Score"],
        "rows": int(len(records)),
        "cluster_counts": {str(k): int((labels == k).sum()) for k in range(n_clusters)},
        "cluster_centers": centers,
        "silhouette_score": score,
        "hierarchical_comparison": None,
        "limitations": [
            "Clusters describe associations in this dataset and are not permanent student labels.",
            "Cluster selection and interpretation can change with the source data.",
        ],
    }
    if compare_hierarchical:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_labels = hierarchical.fit_predict(transformed)
        summary["hierarchical_comparison"] = {
            "method": "AgglomerativeClustering",
            "silhouette_score": float(silhouette_score(transformed, hierarchical_labels)),
        }
    (OUTPUT_DIR / "cluster_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 4))
    plt.scatter(
        transformed[:, 0],
        transformed[:, 1] if transformed.shape[1] > 1 else [0] * len(transformed),
        c=labels,
        cmap="viridis",
        alpha=0.55,
        s=12,
    )
    plt.xlabel("Scaled feature component 1")
    plt.ylabel("Scaled feature component 2")
    plt.title("Exam-track K-Means clusters")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_visualization.png", dpi=150)
    plt.close()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--compare-hierarchical", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.clusters, args.compare_hierarchical), indent=2))
