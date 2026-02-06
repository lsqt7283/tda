import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_returns(data_path: Path) -> pd.DataFrame:
    df = pd.read_excel(data_path, parse_dates=["Dates"])
    df = df.sort_values("Dates").set_index("Dates")
    df = df.dropna(axis=1, how="all")
    df = df.ffill().dropna(axis=0, how="any")
    return df


def build_asset_embedding(
    df: pd.DataFrame,
    variance_threshold: float = 0.9,
    max_components: Optional[int] = None,
) -> Tuple[np.ndarray, PCA, np.ndarray]:
    asset_matrix = df.T
    scaler = StandardScaler()
    scaled = scaler.fit_transform(asset_matrix)
    threshold = float(min(max(variance_threshold, 0.0), 1.0))
    min_dim = int(min(scaled.shape[0], scaled.shape[1]))
    if threshold > 0.0 and threshold < 1.0 and min_dim > 1:
        n_components: Union[int, float] = threshold
    else:
        cap = max_components if max_components is not None else min_dim
        n_components = int(max(1, min(cap, min_dim)))
    pca = PCA(n_components=n_components, random_state=42)
    embedding = pca.fit_transform(scaled)
    return embedding, pca, scaled


def analyze_pca(
    pca: PCA,
    embedding: np.ndarray,
    asset_names: List[str],
    base_dir: Path,
    variance_threshold: float = 0.9,
    top_n: int = 10,
) -> Dict[str, object]:
    components_full = [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))]
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    threshold = float(min(max(variance_threshold, 0.0), 1.0))
    if len(cumulative) == 0:
        selected_count = 0
    else:
        selected_count = int(np.searchsorted(cumulative, threshold) + 1)
        selected_count = max(1, min(selected_count, len(components_full)))

    selected_components = components_full[:selected_count]
    selected_explained = explained[:selected_count]
    selected_cumulative = cumulative[:selected_count]

    explained_df = pd.DataFrame(
        {
            "component": selected_components,
            "explained_variance_ratio": selected_explained,
            "cumulative_explained_variance": selected_cumulative,
        }
    )
    explained_path = base_dir / "pca_explained_variance.csv"
    explained_df.to_csv(explained_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(selected_components, selected_explained, color="#4C72B0")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA Explained Variance")
    if len(selected_explained) > 0:
        ax.set_ylim(0, max(selected_explained) * 1.2)
    ax2 = ax.twinx()
    ax2.plot(selected_components, selected_cumulative, marker="o", color="#C44E52")
    ax2.axhline(threshold, color="#55A868", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Cumulative variance")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(base_dir / "pca_explained_variance.png", dpi=300)
    plt.close(fig)

    asset_scores = pd.DataFrame(
        embedding[:, :selected_count],
        index=asset_names,
        columns=selected_components,
    )
    scores_path = base_dir / "pca_asset_loadings.csv"
    asset_scores.to_csv(scores_path)

    top_rows: List[Dict[str, object]] = []
    top_assets_summary: Dict[str, List[str]] = {}
    for comp in selected_components:
        sorted_assets = asset_scores[comp].abs().sort_values(ascending=False).head(top_n).index
        display_assets: List[str] = []
        for rank, asset in enumerate(sorted_assets, start=1):
            loading_value = float(asset_scores.at[asset, comp])
            top_rows.append(
                {
                    "component": comp,
                    "rank": rank,
                    "asset": asset,
                    "loading": loading_value,
                    "direction": "positive" if loading_value >= 0 else "negative",
                }
            )
            if rank <= 3:
                display_assets.append(f"{asset} ({loading_value:+.3f})")
        top_assets_summary[comp] = display_assets

    top_loadings_path = base_dir / "pca_top_loadings.csv"
    pd.DataFrame(top_rows).to_csv(top_loadings_path, index=False)

    variance_ratios = {
        comp: float(val) for comp, val in zip(selected_components, selected_explained)
    }
    cumulative_variance = float(selected_cumulative[-1]) if len(selected_cumulative) > 0 else float("nan")

    return {
        "components": selected_components,
        "variance_ratios": variance_ratios,
        "cumulative_variance": cumulative_variance,
        "variance_threshold": threshold,
        "top_assets": top_assets_summary,
        "output_files": [
            "pca_explained_variance.csv",
            "pca_explained_variance.png",
            "pca_asset_loadings.csv",
            "pca_top_loadings.csv",
        ],
    }


def write_summary(
    output_path: Path,
    observations: int,
    assets: int,
    variance_threshold: float,
    pca_components: List[str],
    pca_variance: Dict[str, float],
    pca_cumulative: float,
    pca_top_assets: Dict[str, List[str]],
) -> None:
    lines = [
        "# PCA Analysis Summary",
        f"- Observations processed: {observations}",
        f"- Assets included: {assets}",
        f"- Variance threshold: {variance_threshold:.0%}",
    ]
    if pca_components:
        lines.append(f"- PCs analyzed: {', '.join(pca_components)}")
        if not np.isnan(pca_cumulative):
            lines.append(f"- Cumulative variance (selected PCs): {pca_cumulative:.2%}")
        for comp in pca_components:
            variance_ratio = pca_variance.get(comp, float("nan"))
            if not np.isnan(variance_ratio):
                lines.append(f"  - {comp} variance explained: {variance_ratio:.2%}")
            assets_list = pca_top_assets.get(comp, [])
            if assets_list:
                lines.append(f"    Key contributors: {', '.join(assets_list)}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "returns.xlsx"
    df = load_returns(data_path)

    variance_threshold = 0.9
    embedding_full, pca_full, _ = build_asset_embedding(
        df,
        variance_threshold=variance_threshold,
    )
    pca_outputs = analyze_pca(
        pca_full,
        embedding_full,
        list(df.columns),
        base_dir,
        variance_threshold=variance_threshold,
    )

    summary_path = base_dir / "pca_summary.md"
    write_summary(
        summary_path,
        len(df),
        df.shape[1],
        pca_outputs["variance_threshold"],
        pca_outputs["components"],
        pca_outputs["variance_ratios"],
        pca_outputs["cumulative_variance"],
        pca_outputs["top_assets"],
    )

    info_path = base_dir / "pca_run_metadata.json"
    info = {
        "observations": len(df),
        "assets": int(df.shape[1]),
        "variance_threshold": pca_outputs["variance_threshold"],
        "components": pca_outputs["components"],
        "variance_ratios": pca_outputs["variance_ratios"],
        "cumulative_variance": pca_outputs["cumulative_variance"],
        "output_files": pca_outputs["output_files"] + ["pca_summary.md"],
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
