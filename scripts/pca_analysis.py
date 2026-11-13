"""Fit leakage-safe two-component PCA models for each independent task track."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs" / "pca"
TRACKS = {
    "exam_score": ("Exam_Score", "Exam score"),
    "placement": ("Placement", "Placement"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    transformed = []
    summaries = {}
    plot_data = []
    for prefix, (target, label) in TRACKS.items():
        train = pd.read_csv(SPLITS / f"{prefix}_train.csv")
        val = pd.read_csv(SPLITS / f"{prefix}_val.csv")
        test = pd.read_csv(SPLITS / f"{prefix}_test.csv")
        features = [column for column in train.columns if column != target]
        scaler = StandardScaler().fit(train[features])
        pca = PCA(n_components=2, random_state=42).fit(scaler.transform(train[features]))
        points = []
        for split, frame in (("train", train), ("validation", val), ("test", test)):
            components = pca.transform(scaler.transform(frame[features]))
            part = pd.DataFrame({
                "track": prefix,
                "split": split,
                "PC1": components[:, 0],
                "PC2": components[:, 1],
                "target": frame[target],
            })
            transformed.append(part)
            if split == "test":
                plot_data.append((label, part, target))
        summaries[prefix] = {
            "target": target,
            "features": features,
            "fit_rows": int(len(train)),
            "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
            "total_explained_variance": float(pca.explained_variance_ratio_.sum()),
        }

    points = pd.concat(transformed, ignore_index=True)
    points.to_csv(OUT / "pca_components.csv", index=False)
    (OUT / "pca_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, (label, frame, target) in zip(axes, plot_data):
        scatter = axis.scatter(frame["PC1"], frame["PC2"], c=frame["target"], cmap="viridis", s=10, alpha=0.65)
        axis.set_title(f"{label} test split")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        figure.colorbar(scatter, ax=axis, label=target)
    figure.savefig(OUT / "pca_visualization.png", dpi=150)
    plt.close(figure)
    print(f"Wrote PCA outputs to {OUT}")


if __name__ == "__main__":
    main()
