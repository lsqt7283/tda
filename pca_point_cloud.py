from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from point_clouds import PointCloudResult, PointCloudWindow


def _build_embedding(
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


def _analyze_pca(
    pca: PCA,
    embedding: np.ndarray,
    asset_names: List[str],
    variance_threshold: float,
    top_n: int,
) -> Tuple[List[str], Dict[str, float], float, Dict[str, List[str]]]:
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

    asset_scores = pd.DataFrame(
        embedding[:, :selected_count],
        index=asset_names,
        columns=selected_components,
    )

    variance_ratios = {
        comp: float(val) for comp, val in zip(selected_components, selected_explained)
    }
    cumulative_variance = float(selected_cumulative[-1]) if len(selected_cumulative) > 0 else float("nan")

    top_assets_summary: Dict[str, List[str]] = {}
    for comp in selected_components:
        sorted_assets = asset_scores[comp].abs().sort_values(ascending=False).head(top_n).index
        display_assets: List[str] = []
        for rank, asset in enumerate(sorted_assets, start=1):
            loading_value = float(asset_scores.at[asset, comp])
            if rank <= 3:
                display_assets.append(f"{asset} ({loading_value:+.3f})")
        top_assets_summary[comp] = display_assets

    return selected_components, variance_ratios, cumulative_variance, top_assets_summary


def create_pca_point_cloud(
    *,
    base_dir: Path,
    returns: pd.DataFrame,
    pca_cfg: Dict[str, object],
) -> PointCloudResult:
    variance_threshold = float(pca_cfg.get("variance_threshold", 0.9))
    top_n = int(pca_cfg.get("top_assets", 10))
    max_components = pca_cfg.get("max_components")
    if max_components is not None:
        max_components = int(max_components)

    embedding, pca_model, scaled = _build_embedding(
        returns,
        variance_threshold=variance_threshold,
        max_components=max_components,
    )
    components, variance_ratios, cumulative_variance, top_assets = _analyze_pca(
        pca_model,
        embedding,
        list(returns.columns),
        variance_threshold,
        top_n,
    )

    summary_lines: List[str] = [
        f"Variance threshold: {variance_threshold:.0%}",
    ]
    if components:
        summary_lines.append(f"PCs analysed: {', '.join(components)}")
        if not np.isnan(cumulative_variance):
            summary_lines.append(f"Cumulative variance: {cumulative_variance:.2%}")
        for comp in components:
            ratio = variance_ratios.get(comp, float("nan"))
            if not np.isnan(ratio):
                summary_lines.append(f"  - {comp} variance explained: {ratio:.2%}")
            assets_list = top_assets.get(comp, [])
            if assets_list:
                summary_lines.append(f"    Key contributors: {', '.join(assets_list)}")

    metadata = {
        "variance_threshold": variance_threshold,
        "components": components,
        "variance_ratios": variance_ratios,
        "cumulative_variance": cumulative_variance,
        "top_assets": top_assets,
    }

    windows = [
        PointCloudWindow(
            index=0,
            label="PCA_Full",
            points=embedding,
        )
    ]

    point_cloud = PointCloudResult(
        method="pca",
        full_points=embedding,
        assets=list(returns.columns),
        windows=windows,
        summary_lines=summary_lines,
        metadata=metadata,
    )

    return point_cloud
