import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from persim import bottleneck, plot_diagrams, heat
from ripser import ripser
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

from point_clouds import PointCloudResult
from pca_point_cloud import create_pca_point_cloud
from transfer_entropy_point_cloud import create_transfer_entropy_point_cloud
from umap_point_cloud import create_umap_point_cloud

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


def rolling_indices(length: int, window_size: int, step: int) -> List[Tuple[int, int]]:
    if length < window_size or window_size <= 0 or step <= 0:
        return []
    windows: List[Tuple[int, int]] = []
    for start in range(0, length - window_size + 1, step):
        end = start + window_size
        windows.append((start, end))
    return windows


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
    else:
        axes[-1].set_xlim(left=0, right=1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_diagram_and_barcode(
    diagrams: List[np.ndarray],
    title: str,
    output_path: Path,
) -> None:
    dims = len(diagrams)
    if dims == 0:
        raise ValueError("At least one diagram is required for plotting.")

    fig, axes = plt.subplots(dims, 2, figsize=(12, max(3, 3.5 * dims)))
    if dims == 1:
        axes = np.array([axes])  # type: ignore[assignment]

    max_limit = 0.0
    for diagram in diagrams:
        finite = finite_diagram(diagram)
        if finite.size:
            max_limit = max(max_limit, float(np.max(finite)))

    if max_limit <= 0:
        max_limit = 1.0

    for dim, (diagram, row_axes) in enumerate(zip(diagrams, axes)):
        diag_ax, bar_ax = row_axes
        finite = finite_diagram(diagram)
        if finite.size == 0:
            diag_ax.text(0.5, 0.5, "No finite features", ha="center", va="center", fontsize=10)
        else:
            plot_diagrams([diagram], show=False, ax=diag_ax)
        diag_ax.set_xlim(0, max_limit * 1.05)
        diag_ax.set_ylim(0, max_limit * 1.05)
        diag_ax.set_title(f"H{dim} Persistence Diagram")
        diag_ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

        if finite.size == 0:
            bar_ax.text(0.5, 0.5, "No finite features", ha="center", va="center", fontsize=10)
            bar_ax.set_ylim(-0.5, 0.5)
        else:
            for idx, (birth, death) in enumerate(sorted(finite, key=lambda pair: pair[0])):
                bar_ax.hlines(idx, birth, death, colors="tab:blue", linewidth=2)
            bar_ax.set_ylim(-0.5, len(finite) - 0.5)
        bar_ax.set_xlim(0, max_limit * 1.05)
        bar_ax.set_title(f"H{dim} Barcode")
        bar_ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
        if dim == dims - 1:
            bar_ax.set_xlabel("Filtration value")
        bar_ax.set_ylabel("Feature")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def persistent_entropy(diagram: np.ndarray) -> float:
    finite = finite_diagram(diagram)
    if finite.size == 0:
        return 0.0
    lifetimes = finite[:, 1] - finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if lifetimes.size == 0:
        return 0.0
    total = float(np.sum(lifetimes))
    if total <= 0:
        return 0.0
    probs = lifetimes / total
    entropy = float(-np.sum(probs * np.log(probs)))
    return entropy


def euler_characteristic(betti_numbers: Dict[int, int]) -> int:
    return int(sum(((-1) ** dim) * count for dim, count in betti_numbers.items()))


def save_line_chart(
    x_values: Sequence[object],
    series_map: Dict[str, List[float]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if not x_values:
        raise ValueError("At least one x value is required for line chart plotting.")

    first_val = x_values[0]
    is_datetime = isinstance(first_val, pd.Timestamp)
    if is_datetime:
        x_axis = list(x_values)
    else:
        x_axis = list(range(len(x_values)))

    for label, values in series_map.items():
        ax.plot(x_axis, values, marker="o", label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if len(series_map) > 1:
        ax.legend()
    if is_datetime:
        ax.set_xticks(x_axis)
        ax.set_xticklabels([pd.Timestamp(x).date() for x in x_values], rotation=45, ha="right")
    else:
        ax.set_xticks(x_axis)
        ax.set_xticklabels([str(x) for x in x_values], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def prepare_axis(values: List[object]) -> List[object]:
    if not values:
        return []
    if all(isinstance(val, pd.Timestamp) for val in values):
        return list(values)
    return [str(val) for val in values]


def build_point_cloud(
    base_dir: Path,
    config: Dict[str, object],
    returns: pd.DataFrame,
    te_csv_override: Optional[Path] = None,
) -> PointCloudResult:
    point_cfg = config.get("point_cloud", {}) or {}
    method = str(point_cfg.get("method", "transfer_entropy")).lower()
    tda_cfg = config.get("tda", {}) or {}

    if method == "transfer_entropy":
        data_cfg = config.get("data", {}) or {}
        te_cfg = config.get("transfer_entropy", {}) or {}
        return create_transfer_entropy_point_cloud(
            base_dir=base_dir,
            returns=returns,
            data_cfg=data_cfg,
            te_cfg=te_cfg,
            tda_cfg=tda_cfg,
            te_csv_override=te_csv_override,
        )
    if method == "pca":
        pca_cfg = config.get("pca", {}) or {}
        return create_pca_point_cloud(
            base_dir=base_dir,
            returns=returns,
            pca_cfg=pca_cfg,
        )
    if method == "umap":
        umap_cfg = config.get("umap", {}) or {}
        return create_umap_point_cloud(
            base_dir=base_dir,
            returns=returns,
            umap_cfg=umap_cfg,
            tda_cfg=tda_cfg,
        )

    available = ["transfer_entropy", "pca", "umap"]
    raise ValueError(
        "Unknown point cloud method '"
        f"{method}' specified. Supported methods: {', '.join(available)}."
    )


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
    *,
    observations: int,
    assets: int,
    point_cloud_method: str,
    point_cloud_summary: List[str],
    tda_config: Dict[str, int],
    tda_window_count: int,
    bn_stats: Dict[str, float],
    diffusion_stats: Dict[str, float],
    betti_stats: Dict[int, Dict[str, float]],
    euler_stats: Optional[Dict[str, float]],
    entropy_stats: Dict[int, Dict[str, float]],
    mapper_note: Optional[str] = None,
) -> None:
    lines = [
        "# Topological Data Analysis Summary",
        f"- Observations processed: {observations}",
        f"- Assets included: {assets}",
        f"- Point cloud method: {point_cloud_method}",
        f"- TDA window size: {tda_config.get('window_size', 'n/a')}",
        f"- TDA window step: {tda_config.get('step', 'n/a')}",
        f"- TDA windows evaluated: {tda_window_count}",
    ]

    if point_cloud_summary:
        lines.append("- Point cloud details:")
        for entry in point_cloud_summary:
            if entry.startswith("  "):
                lines.append(entry)
            else:
                lines.append(f"  {entry}")

    if np.isnan(bn_stats.get("mean", float("nan"))):
        lines.append("- Rolling bottleneck distance (H1): not available")
    else:
        lines.append(
            "- Rolling bottleneck distance (H1): "
            f"mean {bn_stats['mean']:.4f}, max {bn_stats['max']:.4f}, min {bn_stats['min']:.4f}"
        )

    if np.isnan(diffusion_stats.get("mean", float("nan"))):
        lines.append("- Rolling diffusion distance (H1): not available")
    else:
        lines.append(
            "- Rolling diffusion distance (H1): "
            f"mean {diffusion_stats['mean']:.4f}, max {diffusion_stats['max']:.4f}, min {diffusion_stats['min']:.4f}"
        )

    for dim, stats in sorted(betti_stats.items()):
        if any(np.isnan(val) for val in stats.values()):
            lines.append(f"- Betti numbers (H{dim}): not available")
        else:
            lines.append(
                f"- Betti numbers (H{dim}): mean {stats['mean']:.2f}, "
                f"max {int(stats['max'])}, min {int(stats['min'])}"
            )

    if euler_stats:
        lines.append(
            "- Euler characteristic: "
            f"mean {euler_stats['mean']:.2f}, max {euler_stats['max']:.2f}, min {euler_stats['min']:.2f}"
        )

    if entropy_stats:
        lines.append("- Persistent entropy:")
        for dim, stats in sorted(entropy_stats.items()):
            if any(np.isnan(val) for val in stats.values()):
                lines.append(f"  H{dim}: not available")
            else:
                lines.append(
                    f"  H{dim}: mean {stats['mean']:.4f}, max {stats['max']:.4f}, min {stats['min']:.4f}"
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

    data_cfg = config.get("data", {}) or {}
    tda_cfg = config.get("tda", {}) or {}
    mapper_cfg = config.get("mapper", {}) or {}

    returns_path = base_dir / data_cfg.get("returns_file", "returns.xlsx")
    df = load_returns(returns_path)

    te_csv_override = Path(args.te_csv) if args.te_csv else None
    point_cloud = build_point_cloud(base_dir, config, df, te_csv_override)

    tda_max_dim = int(tda_cfg.get("max_dimension", 2))
    tda_max_dim = max(1, tda_max_dim)

    diagrams_full = compute_diagrams(point_cloud.full_points, maxdim=tda_max_dim)
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
    overview_full_path = base_dir / "persistent_overview_full.png"
    save_diagram_and_barcode(
        diagrams_full,
        "Persistence Diagram & Barcode (Full Sample)",
        overview_full_path,
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
            mapper_result = create_mapper_visualization(point_cloud.full_points, mapper_output_path, mapper_cfg)
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

    windows = sorted(point_cloud.ensure_windows(), key=lambda item: item.index)
    metrics: List[Dict[str, Any]] = []
    bn_records: List[Dict[str, Any]] = []
    window_results: List[Dict[str, Any]] = []
    euler_records: List[Dict[str, Any]] = []
    entropy_series: Dict[int, List[float]] = defaultdict(list)

    prev_diagrams: Optional[List[np.ndarray]] = None
    window_iterator = progress_iter(windows, total=len(windows), desc="TDA windows")
    for window in window_iterator:
        diagrams = compute_diagrams(window.points, maxdim=tda_max_dim)
        window_results.append({"window": window, "diagrams": diagrams})
        axis_value: object = window.end if window.end is not None else window.label

        betti_counts: Dict[int, int] = {}
        for dim, diagram in enumerate(diagrams):
            summary = diagram_summary(diagram)
            summary["label"] = window.label
            summary["dimension"] = dim
            summary["window_index"] = window.index
            summary["start_date"] = window.start
            summary["end_date"] = window.end
            summary["axis_value"] = axis_value
            metrics.append(summary)
            betti_counts[dim] = summary["feature_count"]
            entropy_series[dim].append(persistent_entropy(diagram))

        euler_value = euler_characteristic(betti_counts)
        euler_records.append(
            {
                "window_index": window.index,
                "label": window.label,
                "axis_value": axis_value,
                "value": euler_value,
            }
        )

        if prev_diagrams is not None:
            bn_value = float("nan")
            diffusion_value = float("nan")
            if len(prev_diagrams) > 1 and len(diagrams) > 1:
                diag_prev = finite_diagram(prev_diagrams[1])
                diag_curr = finite_diagram(diagrams[1])
                if diag_prev.size != 0 and diag_curr.size != 0:
                    bn_value = float(bottleneck(diag_prev, diag_curr))
                    try:
                        diffusion_value = float(heat(diag_prev, diag_curr))
                    except Exception:  # pragma: no cover - robustness for numerical issues
                        diffusion_value = float("nan")
            bn_records.append(
                {
                    "window_index": window.index,
                    "label": window.label,
                    "axis_value": axis_value,
                    "bottleneck_distance_h1": bn_value,
                    "diffusion_distance_h1": diffusion_value,
                }
            )
        prev_diagrams = diagrams

    metrics_df = pd.DataFrame(metrics)
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["dimension", "window_index"]).reset_index(drop=True)
    metrics_path = base_dir / "tda_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    bn_df = pd.DataFrame(bn_records)
    bn_chart_path = base_dir / "bottleneck_distance.png"
    if not bn_df.empty:
        bn_df = bn_df.sort_values("window_index").reset_index(drop=True)
        bn_axis = prepare_axis(bn_df["axis_value"].to_list())
        save_line_chart(
            bn_axis,
            {"H1": bn_df["bottleneck_distance_h1"].to_list()},
            "Rolling Bottleneck Distance (H1)",
            "Bottleneck distance",
            bn_chart_path,
        )
    else:
        if window_results:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(
                0.5,
                0.5,
                "Bottleneck distance requires at least two windows",
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_axis_off()
            fig.suptitle("Rolling Bottleneck Distance (H1)")
            fig.tight_layout()
            fig.savefig(bn_chart_path, dpi=300)
            plt.close(fig)

    diffusion_chart_path = base_dir / "diffusion_distance.png"
    if not bn_df.empty and "diffusion_distance_h1" in bn_df:
        diffusion_axis = prepare_axis(bn_df["axis_value"].to_list())
        save_line_chart(
            diffusion_axis,
            {"H1": bn_df["diffusion_distance_h1"].to_list()},
            "Rolling Diffusion Distance (H1)",
            "Diffusion distance",
            diffusion_chart_path,
        )
    else:
        if window_results:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(
                0.5,
                0.5,
                "Diffusion distance requires at least two windows",
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_axis_off()
            fig.suptitle("Rolling Diffusion Distance (H1)")
            fig.tight_layout()
            fig.savefig(diffusion_chart_path, dpi=300)
            plt.close(fig)

    betti_chart_path = base_dir / "betti_numbers.png"
    ordered_windows = sorted(window_results, key=lambda item: item["window"].index)
    ordered_indices = [item["window"].index for item in ordered_windows]
    all_have_dates = all(item["window"].end is not None for item in ordered_windows)
    axis_values = prepare_axis(
        [item["window"].end for item in ordered_windows]
        if all_have_dates
        else [item["window"].label for item in ordered_windows]
    )

    if not metrics_df.empty:
        pivot = (
            metrics_df.pivot_table(
                index="window_index",
                columns="dimension",
                values="feature_count",
                aggfunc="first",
            )
            .reindex(ordered_indices)
        )
        series_map = {
            f"H{int(dim)}": pivot[dim].tolist()
            for dim in sorted(pivot.columns)
        }
        save_line_chart(
            axis_values,
            series_map,
            "Rolling Betti Numbers",
            "Feature count",
            betti_chart_path,
        )

    euler_chart_path = base_dir / "euler_characteristic.png"
    if euler_records:
        euler_df = pd.DataFrame(euler_records).sort_values("window_index").reset_index(drop=True)
        euler_axis = prepare_axis(euler_df["axis_value"].to_list())
        save_line_chart(
            euler_axis,
            {"Euler characteristic": euler_df["value"].to_list()},
            "Rolling Euler Characteristic",
            "Value",
            euler_chart_path,
        )

    entropy_chart_path = base_dir / "persistent_entropy.png"
    if entropy_series:
        entropy_series_map = {
            f"H{dim}": values
            for dim, values in sorted(entropy_series.items())
        }
        save_line_chart(
            axis_values,
            entropy_series_map,
            "Persistent Entropy",
            "Entropy",
            entropy_chart_path,
        )

    sample_paths: List[str] = []
    sample_barcode_paths: List[str] = []
    sample_overview_paths: List[str] = []
    if window_results:
        latest_index = window_results[-1]["window"].index
        window_data = window_results[-1]
        window = window_data["window"]
        diagrams = window_data["diagrams"]
        suffix = f"{latest_index + 1:03d}"
        if window.start is not None and window.end is not None:
            diag_title = (
                "Persistent Diagrams ("
                f"{window.start.date()} to {window.end.date()})"
            )
            barcode_title = (
                "Persistence Barcodes ("
                f"{window.start.date()} to {window.end.date()})"
            )
            overview_title = (
                "Diagram & Barcode ("
                f"{window.start.date()} to {window.end.date()})"
            )
        else:
            diag_title = f"Persistent Diagrams ({window.label})"
            barcode_title = f"Persistence Barcodes ({window.label})"
            overview_title = f"Diagram & Barcode ({window.label})"

        sample_path = base_dir / f"persistent_diagrams_window_{suffix}.png"
        save_diagram_plot(diagrams, diag_title, sample_path)
        sample_paths.append(sample_path.name)

        barcode_path = base_dir / f"persistent_barcodes_window_{suffix}.png"
        save_barcode_plot(diagrams, barcode_title, barcode_path)
        sample_barcode_paths.append(barcode_path.name)

        overview_path = base_dir / f"persistent_overview_window_{suffix}.png"
        save_diagram_and_barcode(diagrams, overview_title, overview_path)
        sample_overview_paths.append(overview_path.name)

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

    if not bn_df.empty and "diffusion_distance_h1" in bn_df:
        diffusion_series = bn_df["diffusion_distance_h1"]
        diffusion_stats = {
            "mean": float(diffusion_series.mean()),
            "max": float(diffusion_series.max()),
            "min": float(diffusion_series.min()),
        }
    else:
        diffusion_stats = {
            "mean": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
        }

    betti_summary: Dict[int, Dict[str, float]] = {}
    if not metrics_df.empty and "dimension" in metrics_df.columns:
        for dim, dim_df in metrics_df.groupby("dimension"):
            if dim_df.empty:
                betti_summary[dim] = {"mean": float("nan"), "max": float("nan"), "min": float("nan")}
            else:
                betti_summary[dim] = {
                    "mean": float(dim_df["feature_count"].mean()),
                    "max": float(dim_df["feature_count"].max()),
                    "min": float(dim_df["feature_count"].min()),
                }

    if euler_records:
        euler_array = np.asarray([rec["value"] for rec in euler_records], dtype=float)
        euler_stats = {
            "mean": float(np.nanmean(euler_array)),
            "max": float(np.nanmax(euler_array)),
            "min": float(np.nanmin(euler_array)),
        }
    else:
        euler_stats = None

    entropy_summary: Dict[int, Dict[str, float]] = {}
    for dim, values in entropy_series.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            entropy_summary[dim] = {"mean": float("nan"), "max": float("nan"), "min": float("nan")}
        else:
            entropy_summary[dim] = {
                "mean": float(np.nanmean(arr)),
                "max": float(np.nanmax(arr)),
                "min": float(np.nanmin(arr)),
            }

    summary_path = base_dir / "tda_summary.md"
    write_summary(
        summary_path,
        observations=len(df),
        assets=df.shape[1],
        point_cloud_method=point_cloud.method,
        point_cloud_summary=point_cloud.summary_lines,
        tda_config={
            "window_size": int(tda_cfg.get("window_size", len(windows))),
            "step": int(tda_cfg.get("step", 1)),
        },
        tda_window_count=len(windows),
        bn_stats=bn_stats,
        diffusion_stats=diffusion_stats,
        betti_stats=betti_summary,
        euler_stats=euler_stats,
        entropy_stats=entropy_summary,
        mapper_note=mapper_note,
    )

    output_files = [
        diagram_full_path.name,
        barcode_full_path.name,
        overview_full_path.name,
        *sample_paths,
        *sample_barcode_paths,
        *sample_overview_paths,
    ]
    if bn_chart_path.exists():
        output_files.append(bn_chart_path.name)
    if diffusion_chart_path.exists():
        output_files.append(diffusion_chart_path.name)
    if betti_chart_path.exists():
        output_files.append(betti_chart_path.name)
    if euler_chart_path.exists():
        output_files.append(euler_chart_path.name)
    if entropy_chart_path.exists():
        output_files.append(entropy_chart_path.name)
    output_files.extend(["tda_metrics.csv", "tda_summary.md"])
    output_files.extend(point_cloud.output_files)
    if mapper_result:
        output_files.append(mapper_output_path.name)

    info_path = base_dir / "tda_run_metadata.json"
    info = {
        "observations": len(df),
        "assets": int(df.shape[1]),
        "config": {
            "config_path": str(args.config),
            "tda": {
                "window_size": int(tda_cfg.get("window_size", len(windows))),
                "step": int(tda_cfg.get("step", 1)),
                "max_dimension": tda_max_dim,
            },
        },
        "point_cloud": {
            "method": point_cloud.method,
            "summary": point_cloud.summary_lines,
            "metadata": point_cloud.metadata,
            "output_files": point_cloud.output_files,
        },
        "tda": {
            "windows": len(windows),
            "bottleneck": bn_stats,
            "diffusion": diffusion_stats,
            "betti": betti_summary,
            "euler": euler_stats,
            "entropy": entropy_summary,
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
