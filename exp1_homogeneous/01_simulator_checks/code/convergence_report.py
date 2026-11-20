"""Create a numerical gate report and plots from actual MCX output."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulate import ROOT, load_config
from readout import READOUT_VERSION, load_run, profile, diagnostics, compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    folder = L.at("01_simulator_checks/results")
    cfg = load_config()
    pairs = [("reference", "channel_2mm", "N0"), ("repeat", "channel_2mm_repeat", "N0"),
             ("double_photons", "channel_2mm_double", "N0"),
             ("fine_grid", "channel_2mm_fine", "N0_fine"),
             ("wide_domain", "channel_2mm_wide", "N0_wide"),
             ("longer_time", "channel_2mm_longtime", "N0_longtime")]
    report = {"scope": "Simulated ideal hemispherical surface reflectance at depth 2 mm; no camera/phantom validation.",
              "readout_version": READOUT_VERSION,
              "diagnostic_smoothing_sigma_mm": cfg["profile"]["diagnostic_smoothing_sigma_mm"],
              "relative_tolerance": cfg["profile"]["relative_tolerance"],
              "absolute_contrast_tolerance": cfg["profile"]["weak_contrast_absolute_tolerance"],
              "profiles": {}, "checks": {}, "missing": [], "integrated_top_reflectance": {}}
    curves = {}
    for label, c, b in pairs:
        if not (folder / f"{c}.npz").exists() or not (folder / f"{b}.npz").exists():
            report["missing"].append(label)
            continue
        ch, bg = load_run(folder / f"{c}.npz"), load_run(folder / f"{b}.npz")
        x, y = profile(ch, bg)
        curves[label] = (x, y)
        report["profiles"][label] = diagnostics(x, y)
        for run_name, data in ((c, ch), (b, bg)):
            meta = json.loads(str(data["metadata_json"]))
            voxel = meta["config"]["geometry"]["voxel_mm"]
            report["integrated_top_reflectance"][run_name] = float(data["R_xy"].sum(dtype=np.float64) * voxel**2)
        np.savetxt(folder / f"profile_{label}.csv", np.column_stack([x, y]), delimiter=",",
                   header="x_mm,reflectance_ratio", comments="")
        if label == "reference":
            np.savez_compressed(folder / "input_128.npz", X=y[None, :], y_mm=np.array([2.0], np.float32), x_mm=x)
    if "reference" not in curves:
        raise RuntimeError("Primary simulations are not complete")
    ref = report["profiles"]["reference"]
    for label, values in report["profiles"].items():
        if label != "reference":
            report["checks"][label] = compare(ref, values)
    report["energy_bounds_pass"] = all(0 < value <= 1.001 for value in report["integrated_top_reflectance"].values())
    report["gate_pass"] = bool(not report["missing"] and report["energy_bounds_pass"]
                               and all(c["pass"] for c in report["checks"].values()))
    report["next_action"] = "Proceed to multiple depths" if report["gate_pass"] else "Complete or investigate numerical checks before generating main datasets"
    report_path = folder / ("report_partial.json" if args.partial else "report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    ch = load_run(folder / "channel_2mm.npz")
    bg = load_run(folder / "N0.npz")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes[0, 0].imshow(ch["labels_xz"].T, origin="upper", extent=[-30, 30, 20, -0.1], aspect="auto", interpolation="nearest")
    axes[0, 0].set(xlim=(-10, 10), ylim=(10, -0.1), xlabel="x (mm)", ylabel="Depth (mm)", title="Geometry: 3 mm channel, upper edge at 2 mm")
    ratio_map = ch["R_xy"] / bg["R_xy"]
    im = axes[0, 1].imshow(ratio_map.T, origin="lower", extent=[-30, 30, 0, 8], aspect="auto", vmin=0.5, vmax=1.1, cmap="viridis")
    axes[0, 1].set(xlim=(-16, 16), xlabel="x (mm)", ylabel="y (mm)", title="Top diffuse reflectance / N0 (raw MC)")
    fig.colorbar(im, ax=axes[0, 1], label="Reflectance ratio")
    for label, (x, y) in curves.items():
        axes[1, 0].plot(x, y, label=label, linewidth=1.3)
    axes[1, 0].set(xlabel="x (mm)", ylabel="Reflectance ratio", title="128-point inputs; no per-profile min-max scaling")
    axes[1, 0].legend(fontsize=8)
    x, y = curves["reference"]
    axes[1, 1].plot(x, 1-y)
    axes[1, 1].axhline(ref["central_contrast"]/2, color="gray", linestyle="--", label="Half diagnostic peak")
    axes[1, 1].set(xlabel="x (mm)", ylabel="1 - reflectance ratio", title=f"Reference contrast {ref['central_contrast']:.4f}; FWHM {ref['fwhm_mm']:.2f} mm" if ref["fwhm_mm"] else "Reference diagnostic contrast")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Stage 1: MCX-CL reflection test | ideal observation | 850 nm", fontsize=14)
    fig.savefig(folder / "stage1_overview.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
