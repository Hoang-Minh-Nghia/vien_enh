"""Visualize completed depth-LUT and development-CNN outputs only."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulate import ROOT


def main():
    stage2 = L.at("02_forward_model/results")
    with np.load(stage2 / "lut_K.npz", allow_pickle=False) as lut:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(lut["depths_mm"])))
        for depth, curve, color in zip(lut["depths_mm"], lut["profiles"], colors):
            ax.plot(lut["x_mm"], curve, color=color, label=f"{depth:g} mm")
        ax.set(xlabel="x (mm)", ylabel="Reflectance / N0", title="Standard absorption 0.508 /mm: 12 simulated depths")
        ax.legend(ncol=2, fontsize=8)
        fig.savefig(stage2 / "depth_profiles.png", dpi=160)
        plt.close(fig)
    stage3 = L.at("04_training/results/pilot_K")
    training = json.loads((stage3 / "training_K_seed42.json").read_text(encoding="utf-8"))
    report = json.loads((stage3 / "trial_report.json").read_text(encoding="utf-8"))
    with np.load(stage3 / "trial_predictions.npz", allow_pickle=False) as p:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        history = training["history"]
        for key, label in (("train_mae_mm", "Train"), ("validation_mae_mm", "Validation")):
            axes[0].plot([h["epoch"] for h in history], [h[key] for h in history], label=label)
        axes[0].axvline(training["best_epoch"], color="gray", linestyle="--", label="Selected epoch")
        axes[0].set(xlabel="Epoch", ylabel="MAE (mm)", title="Checkpoint selected by validation")
        axes[0].legend()
        axes[1].scatter(p["true_depth_mm"], p["prediction_mm"], s=15, alpha=0.7)
        axes[1].plot([0.5, 5], [0.5, 5], color="gray", linestyle="--")
        axes[1].set(xlabel="True simulated depth (mm)", ylabel="CNN depth (mm)",
                    title=f"Development test MAE: {report['cnn']['mae_mm']:.3f} mm")
        fig.suptitle("Development CNN | standard absorption only | interpolated MC data")
        fig.savefig(stage3 / "trial_overview.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
