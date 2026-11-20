"""V1: run the manifest-locked direct MC and assess its numerical quality.

CNN predictions stay closed: this module never imports torch or reads a
checkpoint. Every file is checked against the V0 lock before anything runs, and
quality decisions use MC diagnostics only.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import argparse
from collections import defaultdict
import hashlib
import json
import time

import numpy as np

from readout import READOUT_VERSION, compare, diagnostics, load_run, profile
from simulate import ROOT, config_hash, load_config, run

MANIFEST = L.at("configs/validation_independent.json")
OUTPUT = L.at("06_final_test/results")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_locked_manifest():
    expected = MANIFEST.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if sha256(MANIFEST) != expected:
        raise RuntimeError("Manifest does not match its locked SHA-256")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for path, value in manifest["locked_files"].items():
        if sha256(L.at(path)) != value:
            raise RuntimeError(f"Locked file changed: {path}")
    if manifest["readout_version"] != READOUT_VERSION:
        raise RuntimeError("Readout version differs from the lock")
    return manifest


def check_run(path, manifest, seed, photons, depth=None, mua=None):
    """Validity from the plan: seed, photons, energy, finite output, layer, geometry."""
    data = load_run(path)
    meta = json.loads(str(data["metadata_json"]))
    r_xy = data["R_xy"]
    energies = [b["stat"]["energytot"] / b["requested_photons"] for b in meta["batches"]]
    geometry = meta["geometry_check"]
    voxel = geometry["voxel_mm"]
    checks = {
        "config_hash": config_hash(meta["config"]) == manifest["mc"]["config_hash"],
        "engine_version": meta["pmcxcl_version"] == manifest["tool_versions"]["mc_environment"]["pmcxcl"],
        "seed": meta["seed"] == seed,
        "photons": sum(b["requested_photons"] for b in meta["batches"]) == photons == meta["photons"],
        "initial_energy": all(abs(e - 1) <= 1e-5 for e in energies),
        "finite_reflectance": bool(np.isfinite(r_xy).all()),
        "positive_reflectance_map": bool(r_xy.min() >= 0 and r_xy.max() > 0),
        "collection_layer": geometry["top_air_layer"] and meta["readout"].startswith("MCX dref at top air layer"),
    }
    if depth is None:
        checks["no_channel"] = geometry["channel_voxels_cross_section"] == 0
        checks["strictly_positive_N0"] = bool(r_xy.min() > 0)
    else:
        checks["depth_mua"] = meta["depth_mm"] == depth and meta["mua_mm_inv"] == mua
        checks["voxelized_top"] = abs(geometry["voxelized_top_mm"] - depth) <= voxel / 2 + 1e-6
    return {"pass": all(checks.values()), "checks": checks, "runtime_s": meta["runtime_s"]}


def assess_condition(jobs, tier, background, manifest):
    """Repeats 2 and 3 must each agree with the pre-declared reference repeat."""
    quality = manifest["quality_plan"]["convergence"]
    reference_repeat = manifest["quality_plan"]["reference_repeat"]
    repeats, diagnostics_by_repeat = [], {}
    for job in sorted(jobs, key=lambda j: j["repeat"]):
        item = job["photon_tiers"][tier]
        path = L.at(item["output"])
        validity = check_run(path, manifest, item["seed"], item["photons"], job["depth_mm"], job["mua_mm_inv"])
        x, curve = profile(load_run(path), background[job["n0_run_id"]])
        diag = diagnostics(x, curve)
        diagnostics_by_repeat[job["repeat"]] = diag
        repeats.append({"repeat": job["repeat"], "run_id": job["run_id"], "seed": item["seed"],
                        "validity": validity, "diagnostics": diag})
    comparisons = [{"candidate_repeat": r,
                    **compare(diagnostics_by_repeat[reference_repeat], diagnostics_by_repeat[r],
                              quality["relative_tolerance"], quality["weak_contrast_absolute_tolerance"])}
                   for r in sorted(diagnostics_by_repeat) if r != reference_repeat]
    return {"tier": tier, "photons": jobs[0]["photon_tiers"][tier]["photons"],
            "valid": all(r["validity"]["pass"] for r in repeats),
            "converged": all(c["pass"] for c in comparisons),
            "repeats": repeats, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Run only the pilot entries of the manifest")
    parser.add_argument("--max-tier", type=int, choices=(0, 1, 2), default=0,
                        help="Highest locked photon tier a failing condition may escalate to")
    parser.add_argument("--dry-run", action="store_true", help="Verify the lock and list the work; run no MC")
    parser.add_argument("--analyze-only", action="store_true", help="Assess completed outputs; run no MC")
    args = parser.parse_args()

    manifest = load_locked_manifest()
    config = load_config()
    if config_hash(config) != manifest["mc"]["config_hash"]:
        raise RuntimeError("Simulation config differs from the lock")
    conditions = defaultdict(list)
    for job in manifest["channel_runs"]:
        if job["pilot"] or not args.pilot:
            conditions[job["condition_id"]].append(job)
    scope = "pilot" if args.pilot else "matrix"
    print(f"LOCK_OK scope={scope} conditions={len(conditions)} "
          f"channel_runs={sum(map(len, conditions.values()))} N0={len(manifest['n0_runs'])} max_tier={args.max_tier}", flush=True)
    if args.dry_run:
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if not args.analyze_only:
        for job in manifest["n0_runs"]:
            run(config, L.at(job["output"]), None, 0.508, job["photons"], job["seed"])
    background, n0_records = {}, []
    for job in manifest["n0_runs"]:
        path = L.at(job["output"])
        validity = check_run(path, manifest, job["seed"], job["photons"])
        background[job["run_id"]] = load_run(path)
        n0_records.append({"run_id": job["run_id"], "seed": job["seed"], "validity": validity})

    reports = []
    for condition_id, jobs in conditions.items():
        attempts = []
        for tier in range(args.max_tier + 1):
            present = all((L.at(j["photon_tiers"][tier]["output"])).exists() for j in jobs)
            if args.analyze_only and not present:
                break
            if not args.analyze_only:
                for job in sorted(jobs, key=lambda j: j["repeat"]):
                    item = job["photon_tiers"][tier]
                    run(config, L.at(item["output"]), job["depth_mm"], job["mua_mm_inv"], item["photons"], item["seed"])
            attempts.append(assess_condition(jobs, tier, background, manifest))
            print(f"ASSESSED {condition_id} tier={tier} valid={attempts[-1]['valid']} converged={attempts[-1]['converged']}", flush=True)
            if attempts[-1]["valid"] and attempts[-1]["converged"]:
                break
        accepted = bool(attempts) and attempts[-1]["valid"] and attempts[-1]["converged"]
        reports.append({"condition_id": condition_id, "depth_mm": jobs[0]["depth_mm"],
                        "mua_mm_inv": jobs[0]["mua_mm_inv"], "mua_group": jobs[0]["mua_group"],
                        "accepted": accepted, "accepted_tier": attempts[-1]["tier"] if accepted else None,
                        "attempts": attempts})

    report = {"scope": scope, "manifest_sha256": sha256(MANIFEST), "max_tier_allowed": args.max_tier,
              "gate_pass": all(r["accepted"] for r in reports) and all(n["validity"]["pass"] for n in n0_records),
              "n_conditions": len(reports), "n_accepted": sum(r["accepted"] for r in reports),
              "cnn_predictions_opened": False, "elapsed_s": time.perf_counter() - start,
              "interpretation": "Numerical MC quality only; ideal surface reflectance; not CNN accuracy or physical validation.",
              "n0_runs": n0_records, "conditions": reports}
    path = OUTPUT / f"quality_{scope}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"QUALITY_REPORT {path} gate_pass={report['gate_pass']} accepted={report['n_accepted']}/{report['n_conditions']}", flush=True)


if __name__ == "__main__":
    main()
