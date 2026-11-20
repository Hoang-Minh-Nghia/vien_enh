"""Render saved depth/absorption results without retraining any model."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulate import ROOT


def plot(report_path, output):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    cells = report["target_domain"]
    levels = sorted({c["mua_mm_inv"] for c in cells})
    bins = sorted({(c["depth_low_mm"], c["depth_high_mm"]) for c in cells})
    vmax = max(c[f"mean_MAE_{b}_mm"] for c in cells for b in ("F", "K"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, branch in zip(axes, ("F", "K")):
        grid = np.full((len(levels), len(bins)), np.nan)
        for cell in cells:
            i = levels.index(cell["mua_mm_inv"])
            j = bins.index((cell["depth_low_mm"], cell["depth_high_mm"]))
            grid[i, j] = cell[f"mean_MAE_{branch}_mm"]
        im = ax.imshow(grid, aspect="auto", vmin=0, vmax=vmax, cmap="YlOrRd")
        for (i, j), value in np.ndenumerate(grid):
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}" + ("*" if value <= report["target_MAE_mm"] else ""),
                        ha="center", va="center", fontsize=8, color="white" if value > vmax * 0.6 else "black")
        ax.set_yticks(range(len(levels)), [str(a) for a in levels])
        ax.set_xticks(range(len(bins)), [f"{lo:g}-{hi:g}" for lo, hi in bins], rotation=45)
        ax.set(xlabel="True depth bin (mm)", ylabel="Channel absorption (1/mm)", title=f"Model {branch}: MAE (mm)")
    fig.colorbar(im, ax=axes, label="Mean MAE across three seeds (mm)", shrink=0.8)
    fig.suptitle("Simulation only | * marks bin MAE <= 0.5 mm")
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    plot(L.at("04_training/results/FK/comparison.json"), L.at("04_training/results/FK/mae_depth_absorption.png"))
