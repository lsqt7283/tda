import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def plot_heatmap(csv_file: Path, snapshot_index: int, output_dir: Path) -> None:
    df = pd.read_csv(csv_file)
    snapshot = df[df["snapshot_index"] == snapshot_index]
    if snapshot.empty:
        raise ValueError(f"Snapshot {snapshot_index} not found in {csv_file}")

    assets = sorted(set(snapshot["asset_i"]) | set(snapshot["asset_j"]))
    asset_to_idx = {asset: idx for idx, asset in enumerate(assets)}
    matrix = np.zeros((len(assets), len(assets)))
    for _, row in snapshot.iterrows():
        i = asset_to_idx[row["asset_i"]]
        j = asset_to_idx[row["asset_j"]]
        matrix[i, j] = row["transfer_entropy"]
        matrix[j, i] = row["transfer_entropy"]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        xticklabels=assets,
        yticklabels=assets,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Symmetric Transfer Entropy"},
    )
    plt.title(f"Transfer Entropy Heatmap (Snapshot {snapshot_index})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"transfer_entropy_snapshot_{snapshot_index:03d}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "transfer_entropy_snapshots.csv"
    output_directory = base_dir / "TE_Heatmaps"
    plot_heatmap(csv_path, snapshot_index=0, output_dir=output_directory)
