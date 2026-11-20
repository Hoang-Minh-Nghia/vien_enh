"""Compare pilot MC from the new GPU with the archived AMD RX 6600 pilot runs.

Each GPU's channel map is normalized by that GPU's own N0. Uses the locked
convergence rule and opens no CNN predictions. AMD files are read only.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import json
import sys

import numpy as np

from readout import compare, diagnostics, load_run, profile
from simulate import ROOT
from run_independent_mc import load_locked_manifest

NEW = L.at("06_final_test/results")
OLD = L.at("maintenance/archive_AMD_RX6600")


def main():
    manifest = load_locked_manifest()
    tolerance = manifest["quality_plan"]["convergence"]
    n0_rows, channel_rows = [], []
    for job in manifest["n0_runs"]:
        name = job["output"].rsplit("/", 1)[1]
        new, old = load_run(NEW / "mc" / name)["R_xy"], load_run(OLD / "mc" / name)["R_xy"]
        n0_rows.append({"run_id": job["run_id"], "mean_R_new": float(new.mean()), "mean_R_amd": float(old.mean()),
                        "mean_relative_difference": float(new.mean() / old.mean() - 1),
                        "pixel_rms_relative_difference": float(np.sqrt(np.mean((new / old - 1) ** 2)))})
    for job in manifest["channel_runs"]:
        item = job["photon_tiers"][0]
        name = item["output"].rsplit("/", 1)[1]
        if not (OLD / "mc/channel" / name).exists():
            continue
        n0_name = f"N0_rep{job['repeat']}.npz"
        x, new = profile(load_run(NEW / "mc/channel" / name), load_run(NEW / "mc" / n0_name))
        _, old = profile(load_run(OLD / "mc/channel" / name), load_run(OLD / "mc" / n0_name))
        result = compare(diagnostics(x, old), diagnostics(x, new),
                         tolerance["relative_tolerance"], tolerance["weak_contrast_absolute_tolerance"])
        channel_rows.append({"run_id": job["run_id"], "profile_mae": float(np.mean(np.abs(new - old))), **result})
    if len(channel_rows) != 9:
        sys.exit(f"Expected 9 archived AMD channel runs, found {len(channel_rows)}")
    report = {"pass": all(r["pass"] for r in channel_rows), "n_channel_runs_compared": len(channel_rows),
              "same_seed_note": "Both GPUs used the same manifest seeds; agreement is a cross-hardware consistency check, not an independent-repeat estimate.",
              "cnn_predictions_opened": False, "n0": n0_rows, "channel": channel_rows}
    path = NEW / "hardware_comparison.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "channel"}, indent=2, ensure_ascii=False))
    print(f"HARDWARE_COMPARISON pass={report['pass']} {path}")


if __name__ == "__main__":
    main()
