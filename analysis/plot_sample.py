"""
Visualise a single sample from PrecachedMANOGraspDataset.

Usage (as a script):
    python plot_sample.py --root <processed_dir> --idx <sample_index>

Usage (as a module):
    from plot_sample import plot_sample
    plot_sample(dataset, idx=0)
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

from graspcnn.data import PrecachedMANOGraspDataset


def plot_sample(dataset: PrecachedMANOGraspDataset, idx: int = 0, normal_scale: float = 0.02):
    data = dataset[idx]

    pos     = data.x[:, 0:3].numpy()   # (N, 3)
    normals = data.x[:, 3:6].numpy()   # (N, 3)
    weights = data.x[:, 6].numpy()     # (N,)

    # Normalise weights to [0, 1] for the colormap
    w_min, w_max = weights.min(), weights.max()
    w_norm = (weights - w_min) / (w_max - w_min + 1e-9)

    cmap   = cm.get_cmap("plasma")
    colors = cmap(w_norm)               # (N, 4) RGBA

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        c=colors, s=3, depthshade=True, zorder=2,
    )

    # Quiver arrows for normals — scaled by normal_scale
    ax.quiver(
        pos[:, 0], pos[:, 1], pos[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        length=normal_scale,
        normalize=True,
        colors=colors,
        linewidth=0.4,
        alpha=0.6,
        zorder=1,
    )

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w_min, vmax=w_max))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, label="Gaussian weight")

    ax.set_title(f"Sample idx={idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def plot_sample_from_cache(processed_dir, idx: int = 0, normal_scale: float = 0.02):
    """Load directly from .pt cache files without reconstructing the dataset object."""
    from pathlib import Path

    processed_dir = Path(processed_dir)
    sample_files  = sorted(processed_dir.glob("sample_*.pt"))

    if not sample_files:
        raise FileNotFoundError(f"No sample_*.pt files found in {processed_dir}")
    if idx >= len(sample_files):
        raise IndexError(f"idx={idx} out of range — {len(sample_files)} samples available")

    sample = torch.load(sample_files[idx], weights_only=False)
    mesh   = torch.load(processed_dir / f"mesh_{sample['uid']}.pt", weights_only=False)
    pos     = mesh["pos"].numpy()               # (N, 3)
    normals = mesh["normals"].numpy()           # (N, 3)
    weights = sample["weights"].squeeze().numpy()  # (N,)
    print(pos.shape)

    w_min, w_max = weights.min(), weights.max()
    w_norm = (weights - w_min) / (w_max - w_min + 1e-9)

    cmap   = cm.get_cmap("plasma")
    colors = cmap(w_norm)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xlim(pos.min() - np.abs(pos).max()*0.1, pos.max() + np.abs(pos).max()*0.1)
    ax.set_ylim(pos.min() - np.abs(pos).max()*0.1, pos.max() + np.abs(pos).max()*0.1)
    ax.set_zlim(pos.min() - np.abs(pos).max()*0.1, pos.max() + np.abs(pos).max()*0.1)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=3, depthshade=True, zorder=2)
    ax.quiver(
        pos[:, 0], pos[:, 1], pos[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        length=normal_scale, normalize=True,
        colors=colors, linewidth=0.4, alpha=0.6, zorder=1,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=w_min, vmax=w_max))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, label="Gaussian weight")

    ax.set_title(f"Sample idx={idx}  |  mesh uid: {sample['uid']}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root (contains processed/ subdir)")
    parser.add_argument("--idx",  type=int, default=0, help="Sample index to plot")
    parser.add_argument("--normal_scale", type=float, default=0.02, help="Arrow length for normals")
    args = parser.parse_args()

    from pathlib import Path
    plot_sample_from_cache(Path(args.root) / "processed", idx=args.idx, normal_scale=args.normal_scale)
