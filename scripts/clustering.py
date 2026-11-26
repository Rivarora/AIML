"""Describe cohort groups in the exam track without using the exam score.

The transformer and K-Means model are fitted on the training split only.
Validation and test records are assigned with the fitted model, so the
descriptive view never uses held-out records to learn the cluster structure.
The target is excluded from the clustering matrix and is not used to choose
the number of clusters.

Outputs:
- outputs/clustering/cluster_summary.json
- outputs/clustering/cluster_assignments.csv
- outputs/clustering/cluster_visualization.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_config import EXAM_MODEL_FEATURES, RANDOM_STATE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clustering"


def _load_records() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for split_name in ("train", "val", "test"):
        frame = pd.read_csv(SPLITS_DIR / f"exam_score_{split_name}.csv")
        frame["split"] = split_name
        frames.append(frame)
    missing = [column for column in EXAM_MODEL_FEATURES if column not in frames[0]]
    if missing:
        raise ValueError(f"Exam split is missing clustering features: {missing}")
    return frames[0], frames[1], frames[2]


def _describe_cluster(center: dict[str, float], overall: dict[str, float]) -> str:
    higher: list[str] = []
    lower: list[str] = []
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
        return (
            f"Higher {', '.join(higher)} and lower {', '.join(lower)} "
            "than the overall cohort average."
        )
    if higher:
        return f"Higher {', '.join(higher)} than the overall cohort average."
    if lower:
        return f"Lower {', '.join(lower)} than the overall cohort average."
    return "Near the overall average on the measured clustering indicators."


def run(n_clusters: int = 3, compare_hierarchical: bool = False) -> dict:
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    train, val, test = _load_records()
    records = pd.concat([train, val, test], ignore_index=True)

    numeric = [
        column
        for column in EXAM_MODEL_FEATURES
        if pd.api.types.is_numeric_dtype(records[column])
    ]
    categorical = [column for column in EXAM_MODEL_FEATURES if column not in numeric]

    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    train_matrix = preprocessor.fit_transform(train[EXAM_MODEL_FEATURES])
    val_matrix = preprocessor.transform(val[EXAM_MODEL_FEATURES])
    test_matrix = preprocessor.transform(test[EXAM_MODEL_FEATURES])

    if n_clusters >= len(train_matrix):
        raise ValueError("n_clusters must be smaller than the training row count.")

    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    train_labels = model.fit_predict(train_matrix)
    val_labels = model.predict(val_matrix)
    test_labels = model.predict(test_matrix)
    labels = pd.concat(
        [
            pd.Series(train_labels),
            pd.Series(val_labels),
            pd.Series(test_labels),
        ],
        ignore_index=True,
    ).to_numpy()

    silhouette = float(silhouette_score(train_matrix, train_labels))
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
                "share": round(float(len(subset) / len(records)), 4),
                "numeric_center": center,
                "description": _describe_cluster(center, overall),
            }
        )

    # A two-component PCA fitted on the training matrix is used only for the
    # descriptive scatter plot; it is not part of the clustering distance.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    train_components = pca.fit_transform(train_matrix)
    components = pca.transform(
        np.vstack([train_matrix, val_matrix, test_matrix])
    )

    summary = {
        "method": "K-Means",
        "n_clusters": n_clusters,
        "fit_split": "train",
        "assignment_policy": (
            "K-Means is fitted on the training split; validation and test "
            "records are assigned with the fitted centroids."
        ),
        "features": EXAM_MODEL_FEATURES,
        "target_columns_excluded": ["Exam_Score"],
        "rows": int(len(records)),
        "train_rows": int(len(train)),
        "silhouette_score": silhouette,
        "inertia": float(model.inertia_),
        "pca_explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
        "cluster_counts": {
            str(cluster): int((labels == cluster).sum())
            for cluster in range(n_clusters)
        },
        "train_cluster_counts": {
            str(cluster): int((train_labels == cluster).sum())
            for cluster in range(n_clusters)
        },
        "cluster_centers": centers,
        "hierarchical_comparison": None,
        "limitations": [
            "Clusters describe associations in this dataset and are not permanent student labels.",
            "Cluster selection and interpretation can change with the source data.",
            "The exam score target was excluded, so clusters are not a prediction of performance.",
            "The source data are synthetic; cluster descriptions are descriptive only.",
        ],
    }
    if compare_hierarchical:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_labels = hierarchical.fit_predict(train_matrix)
        summary["hierarchical_comparison"] = {
            "method": "AgglomerativeClustering",
            "fit_split": "train",
            "silhouette_score": float(
                silhouette_score(train_matrix, hierarchical_labels)
            ),
        }
    (OUTPUT_DIR / "cluster_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    scatter = axis.scatter(
        components[:, 0],
        components[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.5,
        s=12,
    )
    train_centers = [
        train_components[train_labels == cluster].mean(axis=0)
        for cluster in range(n_clusters)
    ]
    for cluster, center in enumerate(train_centers):
        axis.scatter(
            center[0],
            center[1],
            marker="X",
            s=130,
            c=[scatter.cmap(scatter.norm(cluster))],
            edgecolors="white",
            linewidths=1.2,
            label=f"Cluster {cluster} centroid",
        )
    axis.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% of variance)"
    )
    axis.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% of variance)"
    )
    axis.set_title(f"Exam-track cohort clusters (train silhouette {silhouette:.3f})")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "cluster_visualization.png", dpi=150)
    plt.close(figure)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--compare-hierarchical", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.clusters, args.compare_hierarchical), indent=2))
