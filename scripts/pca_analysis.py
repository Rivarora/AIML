"""Fit leakage-safe two-component PCA models for each independent task track.

The preprocessing transformer (median imputation plus scaling for numeric
columns, most-frequent imputation plus one-hot encoding for categorical
columns) and PCA are fitted on the training split only. Validation and test
rows are transformed with those fitted objects. The target is used only for
coloring the test-split plot and is never part of the fitted matrix.

Outputs:
- outputs/pca/pca_summary.json
- outputs/pca/pca_components.csv
- outputs/pca/pca_visualization.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
OUT = ROOT / "outputs" / "pca"
TRACKS = {
    "exam_score": ("Exam_Score", "Exam score"),
    "placement": ("Placement", "Placement"),
}
RANDOM_STATE = 42


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    transformed: list[pd.DataFrame] = []
    summaries: dict[str, dict] = {}
    plot_data: list[tuple[str, pd.DataFrame, str, list[float]]] = []

    for prefix, (target, label) in TRACKS.items():
        train = pd.read_csv(SPLITS / f"{prefix}_train.csv")
        val = pd.read_csv(SPLITS / f"{prefix}_val.csv")
        test = pd.read_csv(SPLITS / f"{prefix}_test.csv")

        if target not in train.columns:
            raise ValueError(f"{prefix} training split has no {target} column.")

        features = [column for column in train.columns if column != target]
        numeric = [
            column for column in features if pd.api.types.is_numeric_dtype(train[column])
        ]
        categorical = [column for column in features if column not in numeric]

        preprocessor = build_preprocessor(numeric, categorical)
        train_matrix = preprocessor.fit_transform(train[features])
        pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(train_matrix)

        for split, frame in (("train", train), ("validation", val), ("test", test)):
            matrix = preprocessor.transform(frame[features])
            components = pca.transform(matrix)
            if target == "Placement":
                target_values = (frame[target] == "Yes").astype(int).to_numpy()
            else:
                target_values = frame[target].to_numpy()
            transformed.append(
                pd.DataFrame(
                    {
                        "track": prefix,
                        "split": split,
                        "PC1": components[:, 0],
                        "PC2": components[:, 1],
                        "target": target_values,
                    }
                )
            )
            if split == "test":
                plot_data.append(
                    (
                        label,
                        transformed[-1],
                        target,
                        [float(value) for value in pca.explained_variance_ratio_],
                    )
                )

        summaries[prefix] = {
            "target": target,
            "features": features,
            "transformed_feature_count": int(train_matrix.shape[1]),
            "fit_split": "train",
            "fit_rows": int(len(train)),
            "preprocessing": (
                "median imputation + StandardScaler for numeric features; "
                "most-frequent imputation + OneHotEncoder for categorical features"
            ),
            "n_components": 2,
            "random_state": RANDOM_STATE,
            "explained_variance_ratio": [
                float(value) for value in pca.explained_variance_ratio_
            ],
            "explained_variance": [float(value) for value in pca.explained_variance_],
            "total_explained_variance": float(pca.explained_variance_ratio_.sum()),
        }

    pd.concat(transformed, ignore_index=True).to_csv(
        OUT / "pca_components.csv", index=False
    )
    (OUT / "pca_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for axis, (label, frame, target, ratios) in zip(axes, plot_data):
        scatter = axis.scatter(
            frame["PC1"],
            frame["PC2"],
            c=frame["target"],
            cmap="viridis",
            s=10,
            alpha=0.6,
        )
        axis.set_title(
            f"{label}: test split\nPC1 {ratios[0] * 100:.1f}% + PC2 {ratios[1] * 100:.1f}%"
        )
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        figure.colorbar(scatter, ax=axis, label=target)
    figure.savefig(OUT / "pca_visualization.png", dpi=150)
    plt.close(figure)

    print(f"Wrote PCA outputs to {OUT}")


if __name__ == "__main__":
    main()
