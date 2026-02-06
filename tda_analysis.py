import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persim import bottleneck, plot_diagrams
from ripser import ripser
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

try:
    import yaml
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("PyYAML is required. Install it via 'pip install pyyaml'.") from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

try:
    from tdamapper.cover import CubicalCover
    from tdamapper.learn import MapperAlgorithm
    from tdamapper.plot import MapperPlot
except ImportError:  # pragma: no cover - optional dependency
    CubicalCover = None
    MapperAlgorithm = None
    MapperPlot = None


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_returns(data_path: Path) -> pd.DataFrame:
    df = pd.read_excel(data_path, parse_dates=["Dates"])
    df = df.sort_values("Dates").set_index("Dates")
    df = df.dropna(axis=1, how="all")
    df = df.ffill().dropna(axis=0, how="any")
    return df


def progress_iter(iterable: Iterable, total: Optional[int] = None, desc: str = "") -> Iterable:
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=False)

    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)  # type: ignore[arg-type]

    def generator():
        count = 0
        checkpoints = max(1, (total or 100) // 10)
        for item in iterable:
            yield item
            count += 1
            if total:
                if count == 1 or count == total or count % checkpoints == 0:
                    prefix = f"{desc}: " if desc else ""
                    print(f"{prefix}{count}/{total}")
            elif count % checkpoints == 0:
                prefix = f"{desc}: " if desc else ""
                print(f"{prefix}{count}")

    return generator()


def load_te_snapshots(csv_path: Path, assets: List[str]) -> List[Dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Transfer entropy snapshot CSV not found at {csv_path}. Run transfer_entropy_pipeline.py first."
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


def build_asset_embedding(snapshots: List[Dict[str, object]]) -> np.ndarray:
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


def compute_diagrams(point_cloud: np.ndarray, maxdim: int = 1) -> List[np.ndarray]:
    diagrams = ripser(point_cloud, maxdim=maxdim)["dgms"]
    return diagrams


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    finite_points = [pt for pt in diagram if np.isfinite(pt[1])]
    if not finite_points:
        return np.empty((0, 2))
    return np.array(finite_points)


def diagram_summary(diagram: np.ndarray) -> Dict[str, float]:
    diag = finite_diagram(diagram)
    if diag.size == 0:
        return {
            "feature_count": 0,
            "avg_lifetime": float("nan"),
            "total_persistence": 0.0,
        }
    births = diag[:, 0]
    deaths = diag[:, 1]
    lifetimes = deaths - births
    total_persistence = float(np.sum(lifetimes))
    avg_lifetime = float(np.mean(lifetimes))
    return {
        "feature_count": int(len(diag)),
        "avg_lifetime": avg_lifetime,
        "total_persistence": total_persistence,
    }


def save_diagram_plot(diagrams: List[np.ndarray], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_diagrams(diagrams, show=False, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_barcode_plot(diagrams: List[np.ndarray], title: str, output_path: Path) -> None:
    dims = len(diagrams)
    if dims == 0:
        raise ValueError("At least one diagram is required for barcode plotting.")

    fig_height = max(3, 2 * dims)
    fig, axes = plt.subplots(dims, 1, figsize=(8, fig_height), sharex=True)
    if dims == 1:
        axes = [axes]

    max_end = 0.0
    for diagram in diagrams:
        finite = finite_diagram(diagram)
        if finite.size:
            max_end = max(max_end, float(np.max(finite[:, 1])))

    for dim, (diagram, ax) in enumerate(zip(diagrams, axes)):
        finite = finite_diagram(diagram)
        if finite.size == 0:
            ax.text(0.5, 0.5, "No finite features", ha="center", va="center", fontsize=10)
            ax.set_ylim(-0.5, 0.5)
        else:
            for idx, (birth, death) in enumerate(sorted(finite, key=lambda pair: pair[0])):
                ax.hlines(idx, birth, death, colors="tab:blue", linewidth=2)
            ax.set_ylim(-0.5, len(finite) - 0.5)
        ax.set_ylabel(f"H{dim}")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Filtration value")
    axes[0].set_title(title)
    if max_end > 0:
        axes[-1].set_xlim(left=0, right=max_end * 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def rolling_indices(length: int, window_size: int, step: int) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    for start in range(0, length - window_size + 1, step):
        end = start + window_size
        windows.append((start, end))
    return windows


def save_line_chart(
    x_values: List[pd.Timestamp],
    series_map: Dict[str, List[float]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in series_map.items():
        ax.plot(x_values, values, marker="o", label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if len(series_map) > 1:
        ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def create_mapper_visualization(
    point_cloud: np.ndarray,
    output_path: Path,
    mapper_cfg: Dict[str, object],
) -> Dict[str, object]:
    if MapperAlgorithm is None or CubicalCover is None or MapperPlot is None:
        raise ImportError("tdamapper is required for Mapper visualisation. Install it via 'pip install tdamapper'.")
    if point_cloud.shape[0] < 2:
        raise ValueError("Mapper visualisation requires at least two data points.")

    intervals = mapper_cfg.get("intervals", 12)
    overlap = float(mapper_cfg.get("overlap", 0.35))
    cover_algorithm = mapper_cfg.get("cover_algorithm", "proximity")
    lens_components = int(mapper_cfg.get("lens_components", min(2, point_cloud.shape[1])))
    lens_components = max(1, min(lens_components, point_cloud.shape[1]))
    pca_seed = mapper_cfg.get("pca_seed", 42)

    eps = float(mapper_cfg.get("dbscan_eps", 0.5))
    min_samples = int(mapper_cfg.get("dbscan_min_samples", 3))
    eps_growth = mapper_cfg.get("dbscan_eps_growth", [1.0, 2.0, 4.0])

    layout_dim = int(mapper_cfg.get("layout_dim", 2))
    layout_iterations = int(mapper_cfg.get("layout_iterations", 75))
    layout_seed = int(mapper_cfg.get("layout_seed", 42))
    node_size = float(mapper_cfg.get("node_size", 6.0))
    title = mapper_cfg.get("title", "Mapper Graph")
    cmap = mapper_cfg.get("cmap", "Jet")
    plot_width = int(mapper_cfg.get("plot_width", 960))
    plot_height = int(mapper_cfg.get("plot_height", 720))

    pca = PCA(n_components=lens_components, random_state=pca_seed)
    lens = pca.fit_transform(point_cloud)

    n_intervals = intervals
    if isinstance(intervals, int):
        n_intervals = [intervals] * lens_components
    elif isinstance(intervals, (list, tuple)):
        n_intervals = list(intervals)
        if len(n_intervals) < lens_components:
            n_intervals.extend([n_intervals[-1]] * (lens_components - len(n_intervals)))
        else:
            n_intervals = n_intervals[:lens_components]

    def _serialise_intervals(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return int(value)

    cover = CubicalCover(
        n_intervals=n_intervals,
        overlap_frac=overlap,
        algorithm=cover_algorithm,
    )
    intervals_effective: object = n_intervals
    cover_fallback: Optional[str] = None
    try:
        cover.fit(lens)
    except ValueError as exc:
        msg = str(exc)
        if "boolean array indexing assignment" in msg:
            scalar_intervals = (
                int(n_intervals[0]) if isinstance(n_intervals, list) else int(n_intervals)
            )
            scalar_intervals = max(1, scalar_intervals)
            cover = CubicalCover(
                n_intervals=scalar_intervals,
                overlap_frac=overlap,
                algorithm=cover_algorithm,
            )
            cover.fit(lens)
            intervals_effective = scalar_intervals
            cover_fallback = "scalar_intervals"
        else:
            raise

    cluster_meta: Dict[str, object] = {}
    graph = None
    last_error: Optional[Exception] = None

    scales: List[float]
    if isinstance(eps_growth, (list, tuple)) and eps_growth:
        scales = [float(scale) for scale in eps_growth]
    else:
        scales = [1.0, 2.0, 4.0]

    for scale in scales:
        clusterer = DBSCAN(eps=eps * scale, min_samples=min_samples)
        mapper = MapperAlgorithm(cover=cover, clustering=clusterer, failsafe=True, n_jobs=-1)
        try:
            candidate = mapper.fit_transform(point_cloud, lens)
        except Exception as exc:  # pragma: no cover - propagate to fallback handling
            last_error = exc
            continue
        if candidate.number_of_nodes() > 0:
            graph = candidate
            cluster_meta = {
                "algorithm": "DBSCAN",
                "eps": eps * scale,
                "min_samples": min_samples,
            }
            break

    if graph is None or graph.number_of_nodes() == 0:
        fallback_clusters = int(mapper_cfg.get("fallback_clusters", 8))
        fallback_clusters = max(2, min(fallback_clusters, point_cloud.shape[0]))
        clusterer = AgglomerativeClustering(n_clusters=fallback_clusters)
        mapper = MapperAlgorithm(cover=cover, clustering=clusterer, failsafe=True, n_jobs=-1)
        graph = mapper.fit_transform(point_cloud, lens)
        if graph.number_of_nodes() == 0:
            if last_error is not None:
                raise RuntimeError(
                    "Mapper graph is empty even after fallback cluster strategies"
                ) from last_error
            raise RuntimeError("Mapper graph is empty even after fallback cluster strategies")
        cluster_meta = {
            "algorithm": "AgglomerativeClustering",
            "n_clusters": fallback_clusters,
        }

    plotter = MapperPlot(graph, dim=layout_dim, iterations=layout_iterations, seed=layout_seed)
    color_values = lens[:, 0] if lens.shape[1] else np.zeros(point_cloud.shape[0], dtype=float)

    plot_backend = "plotly"
    backend_error: Optional[str] = None
    try:
        fig = plotter.plot_plotly(
            colors=color_values,
            node_size=node_size,
            title=title,
            cmap=cmap,
            width=plot_width,
            height=plot_height,
        )
        fig.write_html(output_path)
    except Exception as exc:
        plot_backend = "pyvis"
        backend_error = str(exc)
        try:
            plotter.plot_pyvis(
                output_file=str(output_path),
                colors=color_values,
                node_size=node_size,
                title=title,
                cmap=cmap,
                width=plot_width,
                height=plot_height,
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                "Plotly backend failed and PyVis fallback also failed."
            ) from fallback_exc
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "lens_components": lens_components,
        "intervals": _serialise_intervals(intervals_effective),
        "intervals_requested": _serialise_intervals(n_intervals),
        "overlap": overlap,
        "dbscan_eps": eps,
        "dbscan_min_samples": min_samples,
        "layout_dim": layout_dim,
        "layout_iterations": layout_iterations,
        "layout_seed": layout_seed,
        "output": output_path.name,
        "plot_backend": plot_backend,
        "plot_backend_error": backend_error,
        "cover_fallback": cover_fallback,
        "cluster_strategy": cluster_meta,
    }


def write_summary(
    output_path: Path,
    observations: int,
    assets: int,
    te_window_size: int,
    te_step: int,
    te_snapshots: int,
    tda_window_size: int,
    tda_window_step: int,
    tda_windows: int,
    te_history: int,
    te_bins: int,
    te_stats: Dict[str, float],
    te_top_pairs: List[str],
    bn_stats: Dict[str, float],
    betti_stats: Dict[int, Dict[str, float]],
    mapper_note: Optional[str] = None,
) -> None:
    lines = [
        "# Topological Data Analysis Summary",
        f"- Observations processed: {observations}",
        f"- Assets included: {assets}",
        f"- Transfer entropy window (trading days): {te_window_size}",
        f"- Transfer entropy sampling step (days): {te_step}",
        f"- Transfer entropy snapshots ingested: {te_snapshots}",
        f"- TDA window size (snapshots): {tda_window_size}",
        f"- TDA window step (snapshots): {tda_window_step}",
        f"- TDA windows evaluated: {tda_windows}",
        f"- Transfer entropy history length (k): {te_history}",
        f"- Transfer entropy discretization bins: {te_bins}",
    ]
    if not np.isnan(te_stats.get("mean", float("nan"))):
        lines.append(
            "- Symmetric transfer entropy (global): "
            f"mean {te_stats['mean']:.4f}, max {te_stats['max']:.4f}, min {te_stats['min']:.4f}"
        )
    if te_top_pairs:
        lines.append("- Strongest average TE pairs:")
        for entry in te_top_pairs:
            lines.append(f"  - {entry}")
    if np.isnan(bn_stats.get("mean", float("nan"))):
        lines.append("- Rolling bottleneck distance (H1): not available")
    else:
        lines.append(
            "- Rolling bottleneck distance (H1): "
            f"mean {bn_stats['mean']:.4f}, max {bn_stats['max']:.4f}, min {bn_stats['min']:.4f}"
        )
    for dim, stats in sorted(betti_stats.items()):
        if any(np.isnan(val) for val in stats.values()):
            lines.append(f"- Betti numbers (H{dim}): not available")
        else:
            lines.append(
                f"- Betti numbers (H{dim}): mean {stats['mean']:.2f}, "
                f"max {int(stats['max'])}, min {int(stats['min'])}"
            )
    if mapper_note:
        lines.append(f"- Mapper visualisation: {mapper_note}")
    lines.append(
        "\nBetti numbers represent the count of topological features (components, loops, etc.) "
        "detected within each rolling window's persistent homology diagrams."
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topological analysis using transfer entropy snapshots")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "tda_config.yaml",
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--te-csv",
        type=Path,
        default=None,
        help="Optional override path for transfer entropy snapshot CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    config = load_config(args.config)

    data_cfg = config.get("data", {})
    te_cfg = config.get("transfer_entropy", {})
    tda_cfg = config.get("tda", {})
    mapper_cfg = config.get("mapper", {})

    returns_path = base_dir / data_cfg.get("returns_file", "returns.xlsx")
    te_csv_path = (
        Path(args.te_csv)
        if args.te_csv
        else base_dir / data_cfg.get("te_snapshot_csv", "transfer_entropy_snapshots.csv")
    )

    df = load_returns(returns_path)
    assets = list(df.columns)
    te_snapshots = load_te_snapshots(te_csv_path, assets)
    te_stats, te_top_pairs = summarize_transfer_entropy_snapshots(te_snapshots, assets)

    te_window_size = int(te_cfg.get("window_size", 40))
    te_step = int(te_cfg.get("step", 1))
    te_history = int(te_cfg.get("history", 1))
    te_bins = int(te_cfg.get("bins", 12))
    tda_window_size = int(tda_cfg.get("window_size", 25))
    tda_window_step = int(tda_cfg.get("step", 5))

    if len(te_snapshots) >= tda_window_size:
        full_reference = te_snapshots[-tda_window_size:]
    else:
        full_reference = te_snapshots
    embedding_full = build_asset_embedding(full_reference)
    diagrams_full = compute_diagrams(embedding_full)
    diagram_full_path = base_dir / "persistent_diagrams_full.png"
    save_diagram_plot(
        diagrams_full,
        "Persistent Diagrams (Full Sample)",
        diagram_full_path,
    )
    barcode_full_path = base_dir / "persistent_barcodes_full.png"
    save_barcode_plot(
        diagrams_full,
        "Persistence Barcodes (Full Sample)",
        barcode_full_path,
    )

    mapper_note: Optional[str] = None
    mapper_result: Optional[Dict[str, object]] = None
    mapper_enabled = bool(mapper_cfg.get("enabled", True))
    mapper_output_raw = mapper_cfg.get("output_html", "mapper_graph.html")
    mapper_output_path = Path(mapper_output_raw)
    if not mapper_output_path.is_absolute():
        mapper_output_path = base_dir / mapper_output_path

    if mapper_enabled:
        try:
            mapper_result = create_mapper_visualization(embedding_full, mapper_output_path, mapper_cfg)
            backend = str(mapper_result.get("plot_backend", "plotly"))
            note = f"exported to {mapper_output_path.name}"
            if backend != "plotly":
                note += f" via {backend}"
            note += f" ({mapper_result['nodes']} nodes)"
            if backend != "plotly":
                reason = mapper_result.get("plot_backend_error")
                if reason:
                    note += f"; plotly failed: {reason}"
            mapper_note = note
        except ImportError as exc:
            mapper_note = str(exc)
        except Exception as exc:  # pragma: no cover - runtime mishaps
            mapper_note = f"failed: {exc}"
            mapper_result = None
    else:
        mapper_note = "disabled via configuration"

    indices = rolling_indices(len(te_snapshots), tda_window_size, tda_window_step)
    if not indices:
        raise ValueError("Insufficient transfer entropy snapshots for the requested TDA window.")

    metrics: List[Dict[str, float]] = []
    bn_records: List[Dict[str, float]] = []
    rolling_diagrams: List[Dict[str, object]] = []

    prev_diagrams: Optional[List[np.ndarray]] = None
    window_iterator = progress_iter(indices, total=len(indices), desc="TDA rolling windows")
    for idx, (start, end) in enumerate(window_iterator):
        window_snapshots = te_snapshots[start:end]
        embedding = build_asset_embedding(window_snapshots)
        diagrams = compute_diagrams(embedding)

        start_date = window_snapshots[0]["start"]
        end_date = window_snapshots[-1]["end"]
        label = f"{start_date.date()}_{end_date.date()}"
        rolling_diagrams.append(
            {
                "index": idx,
                "start": start_date,
                "end": end_date,
                "label": label,
                "diagrams": diagrams,
            }
        )

        for dim, diagram in enumerate(diagrams):
            summary = diagram_summary(diagram)
            summary["label"] = label
            summary["dimension"] = dim
            summary["start_date"] = start_date
            summary["end_date"] = end_date
            metrics.append(summary)

        if prev_diagrams is not None:
            diag_prev = finite_diagram(prev_diagrams[1])
            diag_curr = finite_diagram(diagrams[1])
            if diag_prev.size == 0 or diag_curr.size == 0:
                bn_value = float("nan")
            else:
                bn_value = float(bottleneck(diag_prev, diag_curr))
            bn_records.append(
                {
                    "window_index": idx,
                    "start_date": start_date,
                    "end_date": end_date,
                    "bottleneck_distance_h1": bn_value,
                }
            )
        prev_diagrams = diagrams

    metrics_df = pd.DataFrame(metrics).sort_values("end_date").reset_index(drop=True)
    metrics_path = base_dir / "tda_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    bn_df = pd.DataFrame(bn_records)
    bn_chart_path = base_dir / "bottleneck_distance.png"
    if not bn_df.empty:
        bn_df = bn_df.sort_values("end_date").reset_index(drop=True)
        save_line_chart(
            bn_df["end_date"].to_list(),
            {"H1": bn_df["bottleneck_distance_h1"].to_list()},
            "Rolling Bottleneck Distance (H1)",
            "Bottleneck distance",
            bn_chart_path,
        )

    betti_chart_path = base_dir / "betti_numbers.png"
    if not metrics_df.empty:
        x_values = [pd.Timestamp(x) for x in sorted(metrics_df["end_date"].unique())]
        series_map: Dict[str, List[float]] = {}
        for dim, dim_df in metrics_df.groupby("dimension"):
            lookup = {
                pd.Timestamp(end): float(count)
                for end, count in zip(dim_df["end_date"], dim_df["feature_count"])
            }
            series_map[f"H{dim}"] = [lookup.get(dt, float("nan")) for dt in x_values]
        save_line_chart(
            x_values,
            series_map,
            "Rolling Betti Numbers",
            "Feature count",
            betti_chart_path,
        )

    sample_indices = sorted({0, len(rolling_diagrams) // 2, len(rolling_diagrams) - 1})
    sample_paths: List[str] = []
    sample_barcode_paths: List[str] = []
    for idx in sample_indices:
        window_info = rolling_diagrams[idx]
        title = (
            "Persistent Diagrams ("
            f"{window_info['start'].date()} to {window_info['end'].date()})"
        )
        sample_path = base_dir / f"persistent_diagrams_window_{idx + 1:03d}.png"
        save_diagram_plot(window_info["diagrams"], title, sample_path)
        sample_paths.append(sample_path.name)
        barcode_title = (
            "Persistence Barcodes ("
            f"{window_info['start'].date()} to {window_info['end'].date()})"
        )
        barcode_path = base_dir / f"persistent_barcodes_window_{idx + 1:03d}.png"
        save_barcode_plot(window_info["diagrams"], barcode_title, barcode_path)
        sample_barcode_paths.append(barcode_path.name)

    bn_stats = {
        "mean": float(bn_df["bottleneck_distance_h1"].mean())
        if not bn_df.empty
        else float("nan"),
        "max": float(bn_df["bottleneck_distance_h1"].max())
        if not bn_df.empty
        else float("nan"),
        "min": float(bn_df["bottleneck_distance_h1"].min())
        if not bn_df.empty
        else float("nan"),
    }

    betti_summary: Dict[int, Dict[str, float]] = {}
    for dim, dim_df in metrics_df.groupby("dimension"):
        if dim_df.empty:
            betti_summary[dim] = {"mean": float("nan"), "max": float("nan"), "min": float("nan")}
        else:
            betti_summary[dim] = {
                "mean": float(dim_df["feature_count"].mean()),
                "max": float(dim_df["feature_count"].max()),
                "min": float(dim_df["feature_count"].min()),
            }

    summary_path = base_dir / "tda_summary.md"
    write_summary(
        summary_path,
        len(df),
        df.shape[1],
        te_window_size,
        te_step,
        len(te_snapshots),
        tda_window_size,
        tda_window_step,
        len(indices),
        te_history,
        te_bins,
        te_stats,
        te_top_pairs,
        bn_stats,
        betti_summary,
        mapper_note,
    )

    output_files = [
        diagram_full_path.name,
        barcode_full_path.name,
        *sample_paths,
        *sample_barcode_paths,
    ]
    if te_csv_path.exists():
        output_files.append(te_csv_path.name)
    if bn_chart_path.exists():
        output_files.append(bn_chart_path.name)
    if betti_chart_path.exists():
        output_files.append(betti_chart_path.name)
    output_files.extend(["tda_metrics.csv", "tda_summary.md"])
    if mapper_result:
        output_files.append(mapper_output_path.name)

    info_path = base_dir / "tda_run_metadata.json"
    info = {
        "observations": len(df),
        "assets": int(df.shape[1]),
        "config": {
            "config_path": str(args.config),
            "transfer_entropy": {
                "window_size": te_window_size,
                "step": te_step,
                "history": te_history,
                "bins": te_bins,
            },
            "tda": {
                "window_size": tda_window_size,
                "step": tda_window_step,
            },
        },
        "transfer_entropy": {
            "snapshots": len(te_snapshots),
            "statistics": te_stats,
            "top_pairs": te_top_pairs,
        },
        "tda_window": {
            "count": len(indices),
        },
        "output_files": output_files,
    }
    info["mapper"] = {
        "enabled": mapper_enabled,
        "note": mapper_note,
        "result": mapper_result,
        "output_html": mapper_output_path.name if mapper_result else None,
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
