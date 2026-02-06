import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from dit import Distribution
from dit.shannon import mutual_information

try:
    import yaml
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("PyYAML is required. Install it via 'pip install pyyaml'.") from exc


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
    try:
        from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel
    except ImportError:
        tqdm = None  # type: ignore

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


def discretize_series(values: np.ndarray, bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=int)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(len(arr), dtype=int)

    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    edges = np.quantile(finite, quantiles) if quantiles.size else np.array([])
    edges = np.unique(edges)
    if edges.size == 0:
        unique_vals = np.unique(finite)
        if unique_vals.size <= 1:
            return np.zeros(len(arr), dtype=int)
        edges = unique_vals[:-1]
    return np.digitize(arr, edges, right=False)


def transfer_entropy_direction(target_disc: np.ndarray, source_disc: np.ndarray, history: int) -> float:
    if target_disc.size != source_disc.size:
        raise ValueError("Series must have identical lengths for transfer entropy estimation.")
    if history <= 0:
        raise ValueError("History length must be positive.")
    if target_disc.size <= history:
        return 0.0

    samples: List[Tuple[int, ...]] = []
    for t in range(history, target_disc.size):
        x_future = int(target_disc[t])
        x_past = tuple(int(val) for val in target_disc[t - history : t])
        y_past = tuple(int(val) for val in source_disc[t - history : t])
        samples.append((x_future, *x_past, *y_past))

    if not samples:
        return 0.0

    counts = Counter(samples)
    total = float(sum(counts.values()))
    outcomes = list(counts.keys())
    pmf = [count / total for count in counts.values()]
    dist = Distribution(outcomes, pmf)

    rvs_x_future = [0]
    rvs_x_past = list(range(1, 1 + history))
    rvs_y_past = list(range(1 + history, 1 + 2 * history))

    mi_all = mutual_information(dist, rvs_y_past, rvs_x_future + rvs_x_past)
    mi_cond = mutual_information(dist, rvs_y_past, rvs_x_past)
    te_value = float(mi_all - mi_cond)
    if not np.isfinite(te_value):
        return 0.0
    return max(te_value, 0.0)


def symmetric_transfer_entropy(series_i: np.ndarray, series_j: np.ndarray, history: int, bins: int) -> float:
    if len(series_i) <= history or len(series_j) <= history:
        return 0.0

    disc_i = discretize_series(series_i, bins)
    disc_j = discretize_series(series_j, bins)
    if disc_i.size <= history or disc_j.size <= history:
        return 0.0

    te_j_to_i = transfer_entropy_direction(disc_i, disc_j, history)
    te_i_to_j = transfer_entropy_direction(disc_j, disc_i, history)

    values = [val for val in (te_j_to_i, te_i_to_j) if np.isfinite(val)]
    if not values:
        return 0.0
    return float(np.mean(values))


def compute_transfer_entropy_matrix(window_df: pd.DataFrame, history: int, bins: int) -> np.ndarray:
    assets = list(window_df.columns)
    n_assets = len(assets)
    matrix = np.zeros((n_assets, n_assets), dtype=float)
    for i in range(n_assets):
        series_i = window_df.iloc[:, i].to_numpy()
        for j in range(i + 1, n_assets):
            series_j = window_df.iloc[:, j].to_numpy()
            value = symmetric_transfer_entropy(series_i, series_j, history=history, bins=bins)
            matrix[i, j] = matrix[j, i] = value
    np.fill_diagonal(matrix, 0.0)
    return matrix


def compute_transfer_entropy_snapshots(
    df: pd.DataFrame,
    window_size: int,
    step: int,
    history: int,
    bins: int,
    start_index: int = 0,
) -> List[Dict[str, object]]:
    if window_size <= 1:
        raise ValueError("Transfer entropy window size must be at least 2 observations.")
    snapshots: List[Dict[str, object]] = []
    all_starts = list(range(0, len(df) - window_size + 1, step))
    starts = all_starts[start_index:]
    if not starts:
        return snapshots

    iterator = progress_iter(starts, total=len(starts), desc="Transfer entropy snapshots")
    for start in iterator:
        end = start + window_size
        window_df = df.iloc[start:end]
        matrix = compute_transfer_entropy_matrix(window_df, history=history, bins=bins)
        snapshots.append(
            {
                "index": len(snapshots) + start_index,
                "start": window_df.index[0],
                "end": window_df.index[-1],
                "matrix": matrix,
            }
        )
    return snapshots


def summarize_snapshots(snapshots: List[Dict[str, object]], assets: List[str], top_n: int = 5) -> Tuple[Dict[str, float], List[str]]:
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
            all_values.append(float(value))
            key = (assets[i_idx], assets[j_idx])
            paired_values.setdefault(key, []).append(float(value))

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


def save_snapshots_to_csv(
    snapshots: List[Dict[str, object]],
    assets: List[str],
    output_path: Path,
    append: bool = False,
) -> None:
    rows: List[Dict[str, object]] = []
    for snap in snapshots:
        matrix = snap["matrix"]
        start = snap["start"]
        end = snap["end"]
        idx = snap["index"]
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                value = float(matrix[i, j])
                if value <= 0.0:
                    continue
                rows.append(
                    {
                        "snapshot_index": idx,
                        "start_date": start,
                        "end_date": end,
                        "asset_i": assets[i],
                        "asset_j": assets[j],
                        "transfer_entropy": value,
                    }
                )
    columns = [
        "snapshot_index",
        "start_date",
        "end_date",
        "asset_i",
        "asset_j",
        "transfer_entropy",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if append and output_path.exists() and not df.empty:
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, index=False)


def save_metadata(
    output_path: Path,
    observations: int,
    assets: int,
    config: Dict[str, object],
    te_stats: Dict[str, float],
    te_top_pairs: List[str],
    total_snapshots: int,
) -> None:
    metadata = {
        "observations": observations,
        "assets": assets,
        "transfer_entropy": {
            "window_size": config["transfer_entropy"]["window_size"],
            "step": config["transfer_entropy"]["step"],
            "history": config["transfer_entropy"]["history"],
            "bins": config["transfer_entropy"]["bins"],
            "snapshots": total_snapshots,
            "statistics": te_stats,
            "top_pairs": te_top_pairs,
        },
    }
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def determine_start_index(csv_path: Path) -> Tuple[int, int]:
    if not csv_path.exists():
        return 0, 0
    df = pd.read_csv(csv_path, usecols=["snapshot_index"])
    if df.empty:
        return 0, 0
    max_index = int(df["snapshot_index"].max())
    return max_index + 1, len(df["snapshot_index"].unique())


def load_existing_snapshot_count(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    df = pd.read_csv(csv_path, usecols=["snapshot_index"])
    if df.empty:
        return 0
    return df["snapshot_index"].nunique()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer entropy snapshot generator")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "tda_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        choices={"full", "update"},
        default="full",
        help="Execution mode: full recomputation or incremental update.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / config["data"]["returns_file"]
    output_csv = Path(args.output) if args.output else base_dir / config["data"]["te_snapshot_csv"]
    metadata_path = output_csv.with_name("transfer_entropy_metadata.json")

    df = load_returns(data_path)
    assets = list(df.columns)
    te_cfg = config["transfer_entropy"]
    window_size = int(te_cfg["window_size"])
    step = int(te_cfg["step"])
    history = int(te_cfg["history"])
    bins = int(te_cfg["bins"])

    if args.mode == "full":
        start_index = 0
        existing_count = 0
    else:
        start_index, existing_count = determine_start_index(output_csv)

    snapshots = compute_transfer_entropy_snapshots(
        df,
        window_size=window_size,
        step=step,
        history=history,
        bins=bins,
        start_index=start_index,
    )

    if args.mode == "update" and not snapshots:
        print("No new snapshots to append; existing TE outputs already up to date.")
        return

    append = args.mode == "update" and existing_count > 0
    save_snapshots_to_csv(snapshots, assets, output_csv, append=append)

    # Build in-memory snapshots for stats only when needed (avoid recomputing old ones in update mode).
    if args.mode == "full":
        summary_snapshots = snapshots
    else:
        # Load full set for summary after append.
        summary_snapshots = []
        df_edges = pd.read_csv(output_csv)
        grouped = df_edges.groupby("snapshot_index")
        for idx, group in grouped:
            matrix = np.zeros((len(assets), len(assets)), dtype=float)
            for _, row in group.iterrows():
                i = assets.index(row["asset_i"])
                j = assets.index(row["asset_j"])
                value = float(row["transfer_entropy"])
                matrix[i, j] = matrix[j, i] = value
            summary_snapshots.append(
                {
                    "index": int(idx),
                    "start": pd.to_datetime(group["start_date"].iloc[0]),
                    "end": pd.to_datetime(group["end_date"].iloc[0]),
                    "matrix": matrix,
                }
            )

    te_stats, te_top_pairs = summarize_snapshots(summary_snapshots, assets)
    total_snapshots = start_index + len(snapshots) if args.mode == "update" else len(snapshots)
    save_metadata(metadata_path, len(df), len(assets), config, te_stats, te_top_pairs, total_snapshots)

    print(f"Transfer entropy snapshots written to {output_csv}")
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
