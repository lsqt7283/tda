from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from point_clouds import PointCloudResult, PointCloudWindow


def load_te_snapshots(csv_path: Path, assets: List[str]) -> List[Dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Transfer entropy snapshot CSV not found at {csv_path}. "
            "Run transfer_entropy_pipeline.py first."
        )
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Transfer entropy snapshot CSV is empty; nothing to analyse.")

    asset_index = {asset: idx for idx, asset in enumerate(assets)}
    snapshots: List[Dict[str, object]] = []
    for idx, group in df.groupby("snapshot_index"):
        matrix = np.zeros((len(assets), len(assets)), dtype=float)
        for row in group.itertuples(index=False):
            try:
                i = asset_index[row.asset_i]
                j = asset_index[row.asset_j]
            except KeyError as exc:
                raise KeyError(
                    f"Asset '{row.asset_i}' or '{row.asset_j}' in TE CSV not found in returns dataset."
                ) from exc
            value = float(row.transfer_entropy)
            if value <= 0.0:
                continue
            matrix[i, j] = matrix[j, i] = value
        start_date = pd.to_datetime(group["start_date"].iloc[0])
        end_date = pd.to_datetime(group["end_date"].iloc[0])
        snapshots.append(
            {
                "index": int(idx),
                "start": start_date,
                "end": end_date,
                "matrix": matrix,
            }
        )
    snapshots.sort(key=lambda snap: snap["index"])
    return snapshots


def build_te_asset_embedding(snapshots: List[Dict[str, object]]) -> np.ndarray:
    if not snapshots:
        raise ValueError("At least one transfer entropy snapshot is required for embedding.")

    matrices = [snap["matrix"] for snap in snapshots]
    n_assets = matrices[0].shape[0]
    for matrix in matrices:
        if matrix.shape[0] != n_assets:
            raise ValueError("All transfer entropy matrices must share the same asset dimension.")

    feature_columns: List[np.ndarray] = []
    for matrix in matrices:
        if matrix.shape[1] <= 1:
            averages = np.zeros(n_assets, dtype=float)
        else:
            positive_mask = matrix > 0.0
            row_sums = (matrix * positive_mask).sum(axis=1)
            counts = positive_mask.sum(axis=1)
            denom = np.where(counts > 0, counts, 1)
            averages = row_sums / denom
        feature_columns.append(averages)

    embedding = np.column_stack(feature_columns)
    embedding = embedding - embedding.mean(axis=0, keepdims=True)
    std = embedding.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return embedding / std


def summarize_transfer_entropy_snapshots(
    snapshots: List[Dict[str, object]],
    assets: List[str],
    top_n: int = 5,
) -> Tuple[Dict[str, float], List[str]]:
    if not snapshots:
        return {
            "mean": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
        }, []

    paired_values: Dict[Tuple[str, str], List[float]] = {}
    all_values: List[float] = []
    for snap in snapshots:
        matrix = snap["matrix"]
        upper_i, upper_j = np.triu_indices(len(assets), k=1)
        for i_idx, j_idx, value in zip(upper_i, upper_j, matrix[upper_i, upper_j]):
            if value <= 0.0:
                continue
            value = float(value)
            all_values.append(value)
            key = (assets[i_idx], assets[j_idx])
            paired_values.setdefault(key, []).append(value)

    values_array = np.asarray(all_values, dtype=float)
    summary = {
        "mean": float(np.nanmean(values_array)) if values_array.size else float("nan"),
        "max": float(np.nanmax(values_array)) if values_array.size else float("nan"),
        "min": float(np.nanmin(values_array)) if values_array.size else float("nan"),
    }

    averaged_pairs = [
        (pair, float(np.nanmean(vals)))
        for pair, vals in paired_values.items()
        if vals
    ]
    averaged_pairs.sort(key=lambda item: item[1], reverse=True)
    formatted_pairs = [f"{a} <-> {b}: {value:.4f}" for (a, b), value in averaged_pairs[:top_n]]
    return summary, formatted_pairs


def _rolling_indices(length: int, window_size: int, step: int) -> List[Tuple[int, int]]:
    if length < window_size:
        return []
    windows: List[Tuple[int, int]] = []
    for start in range(0, length - window_size + 1, step):
        end = start + window_size
        windows.append((start, end))
    return windows


def create_transfer_entropy_point_cloud(
    *,
    base_dir: Path,
    returns: pd.DataFrame,
    data_cfg: Dict[str, object],
    te_cfg: Dict[str, object],
    tda_cfg: Dict[str, object],
    te_csv_override: Optional[Path] = None,
) -> PointCloudResult:
    assets = list(returns.columns)
    if te_csv_override is not None:
        te_csv_path = Path(te_csv_override)
    else:
        te_csv_path = base_dir / data_cfg.get("te_snapshot_csv", "transfer_entropy_snapshots.csv")
    te_snapshots = load_te_snapshots(te_csv_path, assets)
    te_stats, te_top_pairs = summarize_transfer_entropy_snapshots(te_snapshots, assets)

    window_size = int(tda_cfg.get("window_size", 25))
    window_step = int(tda_cfg.get("step", 5))

    if len(te_snapshots) >= window_size:
        reference_snapshots = te_snapshots[-window_size:]
    else:
        reference_snapshots = te_snapshots
    full_points = build_te_asset_embedding(reference_snapshots)

    indices = _rolling_indices(len(te_snapshots), window_size, window_step)
    windows: List[PointCloudWindow] = []
    if not indices:
        raise ValueError("Insufficient transfer entropy snapshots for the requested TDA window.")

    for idx, (start, end) in enumerate(indices):
        window_snapshots = te_snapshots[start:end]
        embedding = build_te_asset_embedding(window_snapshots)
        start_date = window_snapshots[0]["start"]
        end_date = window_snapshots[-1]["end"]
        label = f"{start_date.date()}_{end_date.date()}"
        windows.append(
            PointCloudWindow(
                index=idx,
                label=label,
                points=embedding,
                start=start_date,
                end=end_date,
                metadata={
                    "snapshot_start_index": int(window_snapshots[0]["index"]),
                    "snapshot_end_index": int(window_snapshots[-1]["index"]),
                },
            )
        )

    summary_lines = []
    if not np.isnan(te_stats.get("mean", float("nan"))):
        summary_lines.append(
            "Symmetric transfer entropy (global): "
            f"mean {te_stats['mean']:.4f}, max {te_stats['max']:.4f}, min {te_stats['min']:.4f}"
        )
    if te_top_pairs:
        summary_lines.append("Strongest average TE pairs:")
        summary_lines.extend([f"  - {entry}" for entry in te_top_pairs])

    metadata = {
        "snapshots": len(te_snapshots),
        "statistics": te_stats,
        "top_pairs": te_top_pairs,
        "window_size": window_size,
        "window_step": window_step,
        "te_csv": te_csv_path.name,
    }

    return PointCloudResult(
        method="transfer_entropy",
        full_points=full_points,
        assets=assets,
        windows=windows,
        summary_lines=summary_lines,
        metadata=metadata,
        output_files=[te_csv_path.name] if te_csv_path.exists() else [],
    )
