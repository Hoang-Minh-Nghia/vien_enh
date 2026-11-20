"""Sequential smoke, reproducibility, photon, grid, boundary and time checks."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import copy
import json
from pathlib import Path
import argparse

from simulate import ROOT, load_config, run


def jobs():
    base = load_config()
    channel_n = base["mc"]["channel_photons"]
    background_n = base["mc"]["background_photons"]
    output = [
        ("N0", base, None, background_n, 1101),
        ("channel_2mm", base, 2.0, channel_n, 2101),
        ("channel_2mm_repeat", base, 2.0, channel_n, 2102),
        ("channel_2mm_double", base, 2.0, 2 * channel_n, 2103),
    ]
    fine = copy.deepcopy(base)
    fine["geometry"]["voxel_mm"] = 0.05
    # Large fine-grid kernels returned NaN on this AMD/Windows runtime.
    # Keep the total photon budget, but shorten each kernel invocation.
    fine["mc"]["batch_photons"] = 1000000
    wide = copy.deepcopy(base)
    wide["geometry"]["width_mm"] = 80.0
    late = copy.deepcopy(base)
    late["mc"]["tend_s"] = 1e-8
    for label, config, seed in (("fine", fine, 3101), ("wide", wide, 4101), ("longtime", late, 5101)):
        output.extend([(f"N0_{label}", config, None, background_n, seed),
                       (f"channel_2mm_{label}", config, 2.0, channel_n, seed + 1)])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()
    target = L.at("01_simulator_checks/results")
    target.mkdir(parents=True, exist_ok=True)
    queue = jobs()[:2] if args.primary_only else jobs()
    status = {"stage": 1, "state": "running", "completed": [], "total_jobs": len(queue)}
    for name, config, depth, count, seed in queue:
        status["current"] = name
        (target / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"START {name} ({count} photons)", flush=True)
        try:
            run(config, target / f"{name}.npz", depth, 0.508, count, seed)
        except Exception as exc:
            status.update(state="failed", error=str(exc))
            (target / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            raise
        status["completed"].append(name)
    status.update(state="simulations_complete", current=None)
    (target / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
