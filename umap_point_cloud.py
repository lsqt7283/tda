from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from point_clouds import PointCloudResult, PointCloudWindow

try:
    import umap
except ImportError:  # pragma: no cover - optional dependency
    umap = None

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties


def _build_umap_embedding(
    data: pd.DataFrame,
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: Optional[int],
    spread: float,
) -> np.ndarray:
    if umap is None:
        raise ImportError("umap-learn is required. Install it via 'pip install umap-learn'.")

    asset_matrix = data.T
    scaler = StandardScaler()
    scaled = scaler.fit_transform(asset_matrix)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        spread=spread,
    )
    embedding = reducer.fit_transform(scaled)
    return embedding


def _rolling_slices(length: int, window_size: int, step: int) -> List[Tuple[int, int]]:
    if length <= 0 or window_size <= 0 or step <= 0:
        return []
    if window_size > length:
        return [(0, length)]
    slices: List[Tuple[int, int]] = []
    for start in range(0, length - window_size + 1, step):
        end = start + window_size
        slices.append((start, end))
    if not slices and length > 0:
        slices.append((max(0, length - window_size), length))
    return slices


def _resolve_font_properties() -> FontProperties:
    """Select a font that can display extended Unicode labels (e.g., Chinese)."""
    candidates = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
        except (ValueError, OSError):
            continue
        return FontProperties(family=name)
    return FontProperties()


def _save_embedding_outputs(
    *,
    base_dir: Path,
    assets: List[str],
    embedding: np.ndarray,
    metric: str,
    n_neighbors: int,
) -> List[str]:
    column_names = [f"component_{idx + 1}" for idx in range(embedding.shape[1])]
    embedding_df = pd.DataFrame(embedding, index=assets, columns=column_names)
    embedding_csv = base_dir / "umap_embedding.csv"
    embedding_df.to_csv(embedding_csv)

    font_props = _resolve_font_properties()

    scatter_primary = base_dir / "umap_embedding.png"
    scatter_13 = base_dir / "umap_embedding_1_3.png"
    scatter_23 = base_dir / "umap_embedding_2_3.png"

    scatter_files: List[Tuple[Path, Tuple[int, int], str]] = []
    if embedding.shape[1] >= 2:
        scatter_files.append((scatter_primary, (0, 1), "UMAP Embedding (components 1 & 2)"))
    if embedding.shape[1] >= 3:
        scatter_files.append((scatter_13, (0, 2), "UMAP Embedding (components 1 & 3)"))
        scatter_files.append((scatter_23, (1, 2), "UMAP Embedding (components 2 & 3)"))

    generated_charts: List[str] = []
    for path, (idx_x, idx_y), title in scatter_files:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            embedding[:, idx_x],
            embedding[:, idx_y],
            c="tab:blue",
            edgecolor="white",
            s=60,
        )
        for asset, x_coord, y_coord in zip(assets, embedding[:, idx_x], embedding[:, idx_y]):
            ax.text(
                x_coord,
                y_coord,
                asset,
                fontsize=8,
                ha="center",
                va="center",
                fontproperties=font_props,
            )
        ax.set_title(title, fontproperties=font_props)
        ax.set_xlabel(f"Component {idx_x + 1}", fontproperties=font_props)
        ax.set_ylabel(f"Component {idx_y + 1}", fontproperties=font_props)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        generated_charts.append(path.name)

    distance_matrix = np.linalg.norm(embedding[:, None, :] - embedding[None, :, :], axis=2)
    dist_df = pd.DataFrame(distance_matrix, index=assets, columns=assets)
    distance_csv = base_dir / "umap_embedding_distances.csv"
    dist_df.to_csv(distance_csv)

    neighbor_summary: List[str] = []
    for idx, asset in enumerate(assets):
        sorted_indices = np.argsort(distance_matrix[idx])
        neighbors = [assets[i] for i in sorted_indices if i != idx][:min(3, len(assets) - 1)]
        if neighbors:
            neighbor_summary.append(f"  - {asset}: {', '.join(neighbors)}")

    neighbors_path = base_dir / "umap_nearest_neighbors.txt"
    header = f"Nearest neighbors (metric={metric}, n_neighbors={n_neighbors})\n"
    neighbors_path.write_text(header + "\n".join(neighbor_summary), encoding="utf-8")

    outputs = [embedding_csv.name, distance_csv.name, neighbors_path.name]
    outputs.extend(generated_charts)
    return outputs


def create_umap_point_cloud(
    *,
    base_dir: Path,
    returns: pd.DataFrame,
    umap_cfg: Dict[str, object],
    tda_cfg: Dict[str, object],
) -> PointCloudResult:
    n_components = int(umap_cfg.get("n_components", 3))
    n_neighbors = int(umap_cfg.get("n_neighbors", 15))
    min_dist = float(umap_cfg.get("min_dist", 0.1))
    metric = str(umap_cfg.get("metric", "euclidean"))
    random_state_raw = umap_cfg.get("random_state", 42)
    random_state = int(random_state_raw) if random_state_raw is not None else None
    spread = float(umap_cfg.get("spread", 1.0))

    total_rows = len(returns)
    if total_rows == 0:
        raise ValueError("Returns dataframe is empty; cannot build UMAP embeddings.")

    window_size = int(tda_cfg.get("window_size", total_rows))
    step = int(tda_cfg.get("step", 1))
    window_size = max(1, min(window_size, total_rows))
    step = max(1, step)

    slices = _rolling_slices(total_rows, window_size, step)
    windows: List[PointCloudWindow] = []
    assets = list(returns.columns)
    for idx, (start, end) in enumerate(slices):
        window_df = returns.iloc[start:end]
        embedding_window = _build_umap_embedding(
            window_df,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            spread=spread,
        )
        start_ts = window_df.index[0] if not window_df.empty else None
        end_ts = window_df.index[-1] if not window_df.empty else None
        label = (
            f"{start_ts.date()}_{end_ts.date()}"
            if start_ts is not None and end_ts is not None
            else f"Window_{idx+1:03d}"
        )
        windows.append(
            PointCloudWindow(
                index=idx,
                label=label,
                points=embedding_window,
                start=start_ts,
                end=end_ts,
            )
        )

    latest_embedding = windows[-1].points if windows else _build_umap_embedding(
        returns,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        spread=spread,
    )

    summary_lines = [
        f"Components: {n_components}",
        f"Neighbors: {n_neighbors}",
        f"Min dist: {min_dist}",
        f"Metric: {metric}",
        f"TDA window size: {window_size}",
        f"TDA step: {step}",
    ]
    if random_state is not None:
        summary_lines.append(f"Random state: {random_state}")

    outputs = _save_embedding_outputs(
        base_dir=base_dir,
        assets=assets,
        embedding=latest_embedding,
        metric=metric,
        n_neighbors=n_neighbors,
    )

    metadata = {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "spread": spread,
        "random_state": random_state,
        "window_size": window_size,
        "step": step,
        "windows": len(windows),
    }

    return PointCloudResult(
        method="umap",
        full_points=latest_embedding,
        assets=assets,
        windows=windows,
        summary_lines=summary_lines,
        metadata=metadata,
        output_files=outputs,
    )
