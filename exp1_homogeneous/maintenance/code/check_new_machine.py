"""Preflight for a new MC machine: lock integrity, engine, GPU choice and a short MC smoke run.

Run with the MC environment. Outputs stay in results/migration_check and never
enter the validation set. The smoke comparison is a coarse check against the
original 2 mm reference, not a validation result.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import argparse
import importlib.metadata
import json
import platform
import sys

from readout import compare, diagnostics, load_run, profile
from simulate import ROOT, load_config, run
from run_independent_mc import check_run, load_locked_manifest, sha256

SMOKE_PHOTONS = 6_000_000
N0_SEED, CHANNEL_SEED = 39_000_001, 39_000_002
OUT = L.at("maintenance/migration_check")


def finish(report):
    report["pass"] = all(report["checks"].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "machine_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("MACHINE_CHECK_PASS" if report["pass"] else "MACHINE_CHECK_FAIL")
    sys.exit(0 if report["pass"] else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--any-gpu", action="store_true", help="Testing only: do not require an NVIDIA device")
    args = parser.parse_args()

    manifest = load_locked_manifest()
    config = load_config()
    checks = {"manifest_and_locked_files": True}
    checks["checkpoints"] = all(sha256(L.at(c["path"])) == c["sha256"] for c in manifest["checkpoints"])
    checks["training_data"] = all(sha256(L.at(v["path"])) == v["sha256"] for v in manifest["training_data"].values())
    checks["python_3_12"] = sys.version_info[:2] == (3, 12)
    locked_engine = manifest["tool_versions"]["mc_environment"]["pmcxcl"]
    installed_engine = importlib.metadata.version("pmcxcl")
    checks["pmcxcl_version"] = installed_engine == locked_engine
    report = {"python": platform.python_version(), "pmcxcl": installed_engine, "locked_pmcxcl": locked_engine,
              "manifest_amendments": len(manifest["amendments"]), "checks": checks}

    import pmcxcl
    devices = [d.get("name") for d in pmcxcl.gpuinfo()]
    index = config["mc"]["gpuid"]
    selected = devices[index - 1] if 1 <= index <= len(devices) else None
    report.update(opencl_devices=devices, gpuid=index, selected_device=selected)
    name = (selected or "").lower()
    checks["selected_gpu"] = selected is not None and (args.any_gpu or "nvidia" in name or "rtx" in name)
    if not all(checks.values()):
        finish(report)

    n0_path, channel_path = OUT / "smoke_N0.npz", OUT / "smoke_channel_2mm.npz"
    run(config, n0_path, None, 0.508, SMOKE_PHOTONS, N0_SEED)
    run(config, channel_path, 2.0, 0.508, SMOKE_PHOTONS, CHANNEL_SEED)
    n0_validity = check_run(n0_path, manifest, N0_SEED, SMOKE_PHOTONS)
    channel_validity = check_run(channel_path, manifest, CHANNEL_SEED, SMOKE_PHOTONS, 2.0, 0.508)
    checks["smoke_N0_valid"] = n0_validity["pass"]
    checks["smoke_channel_valid"] = channel_validity["pass"]

    x, smoke = profile(load_run(channel_path), load_run(n0_path))
    _, reference = profile(load_run(L.at("01_simulator_checks/results/channel_2mm.npz")), load_run(L.at("01_simulator_checks/results/N0.npz")))
    reference_diag, smoke_diag = diagnostics(x, reference), diagnostics(x, smoke)
    tolerance = manifest["quality_plan"]["convergence"]
    agreement = compare(reference_diag, smoke_diag, tolerance["relative_tolerance"], tolerance["weak_contrast_absolute_tolerance"])
    checks["smoke_matches_original_reference"] = agreement["pass"]

    speed = SMOKE_PHOTONS / channel_validity["runtime_s"]
    channel_photons = sum(j["photon_tiers"][0]["photons"] for j in manifest["channel_runs"])
    n0_photons = sum(j["photons"] for j in manifest["n0_runs"])
    report.update(smoke_photons=SMOKE_PHOTONS, n0_validity=n0_validity, channel_validity=channel_validity,
                  reference_diagnostics=reference_diag, smoke_diagnostics=smoke_diag, agreement=agreement,
                  photons_per_second=speed, estimated_matrix_hours_tier0=(channel_photons + n0_photons) / speed / 3600,
                  scope="Short smoke run on this machine; not part of the validation set.")
    finish(report)


if __name__ == "__main__":
    main()
