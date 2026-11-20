"""Train three paired seeds and evaluate both models on identical F test inputs."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulate import ROOT
from train import train, evaluate, metrics
from plot_mae_map import plot as plot_target_domain


def main():
    out = L.at("04_training/results/FK")
    data_path = L.at("03_dataset/data/dataset_FK/dataset.npz")
    config = json.loads((L.at("configs/learning.json")).read_text(encoding="utf-8"))
    reports = []
    for seed in config["training"]["seeds"]:
        model_f = train(data_path, out / "models", "F", seed)
        model_k = train(data_path, out / "models", "K", seed)
        pred_f, data = evaluate(model_f, data_path, "F")
        pred_k, kdata = evaluate(model_k, data_path, "F")
        if not np.array_equal(data["scene_id"], kdata["scene_id"]):
            raise RuntimeError("Unpaired test scene IDs")
        absorption = data["mua_F"]
        masks = {"all": np.ones(len(pred_f), bool), "weak": absorption < 0.4,
                 "standard": np.isclose(absorption, 0.508), "high": absorption > 0.9}
        masks.update({f"mua_{a:.3f}": np.isclose(absorption, a) for a in (0.15, 0.30, 0.508, 1.0)})
        depth_edges = np.arange(0.5, 5.01, 0.5)
        masks.update({f"depth_{lo:.1f}_{hi:.1f}": (data["y_mm"] >= lo) & (data["y_mm"] < hi)
                      for lo, hi in zip(depth_edges[:-1], depth_edges[1:])})
        masks.update({f"mua_{a:.3f}_depth_{lo:.1f}_{hi:.1f}": np.isclose(absorption, a) & (data["y_mm"] >= lo) & (data["y_mm"] < hi)
                      for a in (0.15, 0.30, 0.508, 1.0) for lo, hi in zip(depth_edges[:-1], depth_edges[1:])})
        groups = {}
        for name, mask in masks.items():
            mf, mk = metrics(pred_f[mask], data["y_mm"][mask]), metrics(pred_k[mask], data["y_mm"][mask])
            groups[name] = {"F": mf, "K": mk, "delta_mae_K_minus_F_mm": mk["mae_mm"] - mf["mae_mm"] if mf else None}
        standard_k, standard_data = evaluate(model_k, data_path, "K")
        report = {"seed": seed, "groups": groups,
                  "supplementary_K_on_K": metrics(standard_k, standard_data["y_mm"])}
        reports.append(report)
        with (out / f"predictions_seed{seed}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["scene_id", "mua_mm_inv", "depth_mm", "pred_F_mm", "pred_K_mm", "error_F_mm", "error_K_mm"])
            for i in range(len(pred_f)):
                truth = float(data["y_mm"][i])
                writer.writerow([data["scene_id"][i], float(absorption[i]), truth, float(pred_f[i]), float(pred_k[i]),
                                 float(pred_f[i])-truth, float(pred_k[i])-truth])
        (out / "comparison_progress.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    weak_f = np.array([r["groups"]["weak"]["F"]["mae_mm"] for r in reports])
    weak_k = np.array([r["groups"]["weak"]["K"]["mae_mm"] for r in reports])
    final = {"scope": "Ideal simulation, one straight 3 mm channel, shared interpolated MC LUT; no measured camera noise or phantom validation.",
             "primary_group": "mua 0.15 and 0.30 mm^-1 on shared F test (300 scenes)",
             "mean_weak_mae_F_mm": float(weak_f.mean()), "mean_weak_mae_K_mm": float(weak_k.mean()),
             "mean_delta_mae_K_minus_F_mm": float((weak_k-weak_f).mean()),
             "std_weak_mae_F_across_seeds_mm": float(weak_f.std(ddof=1)),
             "std_weak_mae_K_across_seeds_mm": float(weak_k.std(ddof=1)), "seeds": reports}
    target_domain = []
    for a in (0.15, 0.30, 0.508, 1.0):
        for lo, hi in zip(depth_edges[:-1], depth_edges[1:]):
            key = f"mua_{a:.3f}_depth_{lo:.1f}_{hi:.1f}"
            first = reports[0]["groups"][key]
            if first["F"] is None:
                continue
            cell = {"mua_mm_inv": a, "depth_low_mm": float(lo), "depth_high_mm": float(hi),
                    "n_scenes": first["F"]["n"]}
            for branch in ("F", "K"):
                mean_mae = float(np.mean([r["groups"][key][branch]["mae_mm"] for r in reports]))
                cell[f"mean_MAE_{branch}_mm"] = mean_mae
                cell[f"MAE_{branch}_le_0p5mm"] = mean_mae <= 0.5
            target_domain.append(cell)
    final["target_MAE_mm"] = 0.5
    final["target_domain"] = target_domain
    final["target_interpretation"] = "Mean MAE across seeds in each depth/absorption bin; not a per-scene error bound or a clinical standard."
    (out / "comparison.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "group", "n", "MAE_F_mm", "MAE_K_mm", "RMSE_F_mm", "RMSE_K_mm", "bias_F_mm", "bias_K_mm", "delta_MAE_K_minus_F_mm"])
        for report in reports:
            for group, values in report["groups"].items():
                a, b = values["F"], values["K"]
                if a:
                    writer.writerow([report["seed"], group, a["n"], a["mae_mm"], b["mae_mm"], a["rmse_mm"], b["rmse_mm"], a["bias_mm"], b["bias_mm"], values["delta_mae_K_minus_F_mm"]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    levels = (0.15, 0.30, 0.508, 1.0)
    xx = np.arange(4)
    for branch, shift, color in (("F", -0.18, "#167d9a"), ("K", 0.18, "#d27a29")):
        maes = np.array([[r["groups"][f"mua_{a:.3f}"][branch]["mae_mm"] for a in levels] for r in reports])
        biases = np.array([[r["groups"][f"mua_{a:.3f}"][branch]["bias_mm"] for a in levels] for r in reports])
        axes[0].bar(xx+shift, maes.mean(axis=0), width=0.36, yerr=maes.std(axis=0, ddof=1), label=branch, color=color, capsize=3)
        axes[1].plot(xx, biases.mean(axis=0), marker="o", label=branch, color=color)
    for ax in axes:
        ax.set_xticks(xx, [str(a) for a in levels])
        ax.set_xlabel("Channel absorption (1/mm)")
        ax.legend()
    axes[0].set_ylabel("MAE depth (mm)")
    axes[0].set_title("Same F test inputs; mean and SD of three seeds")
    axes[1].set_ylabel("Prediction minus true depth (mm)")
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set_title("Signed bias by absorption")
    fig.suptitle("F/K comparison in ideal simulation")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    plot_target_domain(out / "comparison.json", out / "mae_depth_absorption.png")
    print(json.dumps({k:v for k,v in final.items() if k not in ("seeds", "target_domain")}, indent=2))


if __name__ == "__main__":
    main()
