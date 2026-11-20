"""V0: lock the independent-validation manifest before any new MC or CNN output exists.

Writes configs/validation_independent.json and a SHA-256 sidecar. It runs no MC
and evaluates no CNN on research data; the only forward pass is a zero-input
shape probe. It refuses to run once a manifest or validation output exists, so
a lock cannot be silently replaced after results have been seen.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import glob
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from readout import READOUT_VERSION
from simulate import ROOT, config_hash, load_config, make_volume, geometry_summary
from train import DepthCNN

MANIFEST = L.at("configs/validation_independent.json")
SIDECAR = MANIFEST.with_suffix(".sha256")
OUTPUT = L.at("06_final_test/results")
VENV_PYTHON = ROOT.parent / ".venv-fk/Scripts/python.exe"

DEPTHS = (0.6, 0.9, 1.4, 1.8, 2.3, 2.8, 3.3, 3.8, 4.3, 4.8)
MUA_GROUPS = {0.15: "seen_weak", 0.22: "unseen", 0.30: "seen_weak", 0.40: "unseen",
              0.508: "standard", 1.0: "high"}
REPEATS = (1, 2, 3)
PILOT_DEPTHS, PILOT_MUAS = (0.6, 3.3, 4.8), (0.15, 0.508)
PHOTON_TIERS = (60_000_000, 120_000_000, 240_000_000)
N0_PHOTONS = 120_000_000
OLD_HELDOUT_DEPTHS = (0.625, 2.25, 4.75)
SEEDS, BRANCHES = (42, 43, 44), ("F", "K")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def n0_seed(repeat):
    return 30_000_000 + repeat * 10_000


def channel_seed(tier, repeat, mua_index, depth_index):
    return 31_000_000 + tier * 100_000 + repeat * 10_000 + mua_index * 100 + depth_index


def batch_seeds(seed, photons, batch_photons):
    """Replays the batch-seed stream used by simulate.run."""
    rng = np.random.default_rng(seed)
    return [int(rng.integers(1, 2**31 - 1)) for _ in range(math.ceil(photons / batch_photons))]


def build_jobs():
    n0_jobs = [{"run_id": f"N0_r{r}", "kind": "N0", "repeat": r, "pilot": True, "photons": N0_PHOTONS,
                "seed": n0_seed(r), "output": f"results/validation_independent/mc/N0_rep{r}.npz"}
               for r in REPEATS]
    channel_jobs = []
    for m, mua in enumerate(MUA_GROUPS):
        for d, depth in enumerate(DEPTHS):
            for r in REPEATS:
                name = f"mua_{mua:.3f}_depth_{depth:.3f}_rep{r}"
                channel_jobs.append({
                    "run_id": f"C_z{depth:.3f}_m{mua:.3f}_r{r}", "kind": "channel",
                    "condition_id": f"z{depth:.3f}_m{mua:.3f}", "depth_mm": depth, "mua_mm_inv": mua,
                    "mua_group": MUA_GROUPS[mua], "repeat": r, "n0_run_id": f"N0_r{r}",
                    "pilot": depth in PILOT_DEPTHS and mua in PILOT_MUAS,
                    "photon_tiers": [{"tier": t, "photons": n, "seed": channel_seed(t, r, m, d),
                                      "output": f"results/validation_independent/mc/channel/{name}_ph{n // 1_000_000:03d}M.npz"}
                                     for t, n in enumerate(PHOTON_TIERS)]})
    return n0_jobs, channel_jobs


def check_seed_independence(n0_jobs, channel_jobs, batch_photons):
    """New run and batch seeds must not reuse each other or any earlier MC run."""
    old_runs, old_batches, files = set(), set(), 0
    for name in glob.glob(str(L.EXP1 / "**" / "*.json"), recursive=True):
        try:
            record = json.loads(open(name, encoding="utf-8").read())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(record, dict) and "batches" in record:
            files += 1
            old_runs.add(record["seed"])
            old_batches.update(b["seed"] for b in record["batches"])
    pairs = [(j["seed"], j["photons"]) for j in n0_jobs]
    pairs += [(t["seed"], t["photons"]) for j in channel_jobs for t in j["photon_tiers"]]
    run_seeds = [s for s, _ in pairs]
    if len(set(run_seeds)) != len(run_seeds):
        raise RuntimeError("Duplicate run seed inside the manifest")
    if set(run_seeds) & old_runs:
        raise RuntimeError(f"Run seeds reuse earlier MC: {sorted(set(run_seeds) & old_runs)[:5]}")
    new_batches = [b for s, n in pairs for b in batch_seeds(s, n, batch_photons)]
    if len(set(new_batches)) != len(new_batches):
        raise RuntimeError("Duplicate batch seed inside the manifest")
    if set(new_batches) & old_batches:
        raise RuntimeError(f"Batch seeds reuse earlier MC: {sorted(set(new_batches) & old_batches)[:5]}")
    return {"earlier_mc_records_scanned": files, "earlier_run_seeds": len(old_runs),
            "earlier_batch_seeds": len(old_batches), "new_run_seeds": len(run_seeds),
            "new_batch_seeds_all_tiers": len(new_batches), "collisions": 0}


def lock_checkpoints(dataset_sha):
    records = []
    for seed in SEEDS:
        for branch in BRANCHES:
            path = L.at(f"04_training/results/FK/models/cnn_{branch}_seed{seed}.pt")
            training = path.with_name(f"training_{branch}_seed{seed}.json")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if (state["branch"], state["seed"]) != (branch, seed):
                raise RuntimeError(f"Checkpoint identity mismatch: {path.name}")
            if state["dataset_sha256"] != dataset_sha:
                raise RuntimeError(f"Checkpoint was not trained on the audited dataset: {path.name}")
            model = DepthCNN()
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            with torch.no_grad():
                probe = model(torch.zeros(2, 1, 128))
            if probe.shape != (2,) or not torch.isfinite(probe).all():
                raise RuntimeError(f"Checkpoint failed the shape probe: {path.name}")
            records.append({"branch": branch, "seed": seed, "path": rel(path), "sha256": sha256(path),
                            "training_log": rel(training), "training_log_sha256": sha256(training),
                            "best_epoch": state["best_epoch"], "validation_mae_mm": state["validation_mae_mm"]})
    return records


def check_geometry(config):
    voxel = config["geometry"]["voxel_mm"]
    rows = []
    for depth in DEPTHS:
        vol, _, z = make_volume(config, depth)
        summary = geometry_summary(config, vol, z)
        offset = summary["voxelized_top_mm"] - depth
        if abs(offset) > voxel / 2 + 1e-6 or not summary["top_air_layer"]:
            raise RuntimeError(f"Voxelized channel top is wrong at depth {depth}: {summary}")
        rows.append({"depth_mm": depth, "voxelized_top_mm": round(summary["voxelized_top_mm"], 6),
                     "voxelized_bottom_mm": round(summary["voxelized_bottom_mm"], 6),
                     "voxelized_width_mm": round(summary["voxelized_width_mm"], 6)})
    return rows


def venv_versions():
    code = "import importlib.metadata as m,json,platform;" \
           "print(json.dumps({'python':platform.python_version(),'pmcxcl':m.version('pmcxcl'),'numpy':m.version('numpy')}))"
    return json.loads(subprocess.run([str(VENV_PYTHON), "-c", code], capture_output=True, text=True, check=True).stdout)


def analysis_plan(train_mean_depth):
    return {
        "error_definition": "e = predicted_depth_mm - true_depth_mm",
        "model_input": "readout.profile(channel, N0_same_repeat) as float32 batch x 1 x 128; identical input for F and K; channel absorption is never given to the network; model.eval() with torch.no_grad()",
        "metrics": {"MAE": "mean(|e|)", "RMSE": "sqrt(mean(e^2))", "bias": "mean(e)"},
        "aggregation": "For each condition and model seed, compute the metric across the three MC repeats; then average conditions with equal weight. Report each model seed, the mean and the SD (ddof=1) across the three seeds.",
        "primary_endpoint": {
            "name": "delta_MAE_unseen",
            "definition": "MAE_K - MAE_F over the 20 conditions with mua in {0.22, 0.40}; positive means F has lower MAE",
            "mua_mm_inv": [0.22, 0.40], "n_conditions": 20},
        "secondary_groups": {
            "seen_weak": "mua in {0.15, 0.30}, each level and pooled; new MC, not a new absorption level for F",
            "standard_tradeoff": "delta_standard = MAE_F - MAE_K at mua 0.508, with absolute values",
            "high": "mua 1.0"},
        "absorption_stability": "At each depth and model seed, average predictions across the three MC repeats at each mua; R = max_over_mua - min_over_mua. Report R_F and R_K with MAE at that depth. Small R counts only if predictions are also close to the true depth.",
        "depth_discrimination": "For each mua, report predicted-vs-true line, MAE, slope and intercept. R^2 alone is not used to conclude accuracy.",
        "constant_baseline": {"definition": "always predict the mean depth of the original training split",
                              "value_mm": train_mean_depth},
        "target": {"mae_mm": 0.5, "interpretation": "Report the conditions meeting and missing MAE <= 0.5 mm. Project target; not a clinical standard or a per-prediction bound."},
        "paired_comparison": "Absolute-error difference K - F on the same condition, MC repeat and model-seed pair.",
        "independence": "60 conditions are a fixed design grid. Three MC repeats measure numerical variation and three CNN seeds measure training variation; the three N0 maps are shared across conditions by repeat index. Do not multiply these into a physical sample size.",
        "statistical_inference": "Descriptive paired effect sizes and consistency across seeds and conditions only; no population significance claim.",
        "conclusion_rules": "Section 6 of Ke_hoach_tiep_theo_kiem_chung_doc_lap_FK.md. Completion does not depend on a result favourable to F."}


def quality_plan():
    return {
        "validity": "Check seed, actual photons, initial energy, finite reflectance, positive N0, collection layer and geometry. Do not repair NaN or drop bad points to let the CNN run.",
        "pairing": "Channel repeat r is normalized only by N0 repeat r.",
        "reference_repeat": 1,
        "convergence": {"method": "readout.compare(diagnostics(repeat 1), diagnostics(candidate)) for repeats 2 and 3; a condition passes when both pass",
                        "relative_tolerance": 0.10, "weak_contrast_reference_below": 0.05,
                        "weak_contrast_absolute_tolerance": 0.005, "fwhm": "must be defined and within the relative tolerance, including for weak signals"},
        "escalation": "If a condition fails at tier t, run all three repeats of that condition at tier t+1 using the pre-assigned seeds. Keep every tier. Selection uses MC quality only and is fixed before CNN predictions are opened.",
        "if_still_failing": "Mark the condition as insufficiently reliable after tier 2; do not claim full validation of that region.",
        "n0_escalation": "N0 stays at 1.2e8 photons in this lock; changing it requires a dated amendment.",
        "pilot_reuse": "Pilot runs are the same manifest entries as the main matrix and are reused only if the configuration is unchanged and they pass.",
        "forbidden_selection": "Never drop a condition because the CNN is wrong, and never try extra seeds to keep favourable runs."}


def main():
    if MANIFEST.exists() or SIDECAR.exists() or OUTPUT.exists():
        sys.exit("Refusing to lock: a manifest or validation output already exists. A lock must precede all results.")
    config = load_config()
    batch_photons = config["mc"]["batch_photons"]
    collides = set(DEPTHS) & (set(config["depths_mm"]) | set(OLD_HELDOUT_DEPTHS))
    if collides:
        raise RuntimeError(f"Validation depths reuse LUT or held-out depths: {sorted(collides)}")

    n0_jobs, channel_jobs = build_jobs()
    seeds = check_seed_independence(n0_jobs, channel_jobs, batch_photons)
    data_path, lut_path = L.at("03_dataset/data/dataset_FK/dataset.npz"), L.at("03_dataset/data/lut_FK.npz")
    dataset_sha = sha256(data_path)
    with np.load(data_path, allow_pickle=False) as data:
        train_mean_depth = float(data["y_mm"][data["split"] == "train"].mean())

    tracked = ["configs/simulation.json", "configs/learning.json", "code/simulate.py", "code/readout.py", "code/train.py"]
    pilot_channel = sum(j["pilot"] for j in channel_jobs)
    manifest = {
        "schema": "reflection-fk-validation/1.0",
        "status": "locked_before_new_MC_and_before_CNN_predictions",
        "locked_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "design_source": "Ke_hoach_tiep_theo_kiem_chung_doc_lap_FK.md",
        "amendment_policy": "Append dated entries to 'amendments' with a reason, before CNN predictions are opened. After a model is adjusted using these data they become development data and a new validation set is required.",
        "amendments": [],
        "design": {"depths_mm": list(DEPTHS), "mua_mm_inv": list(MUA_GROUPS),
                   "mua_groups": {str(k): v for k, v in MUA_GROUPS.items()}, "repeats": list(REPEATS),
                   "channel_x_mm": 0.0, "observation": config["observation"]["mode"],
                   "lateral_shift_checks": "Not part of this lock; any such check must be reported separately and adds no independent MC conditions.",
                   "n_conditions": 60, "n_channel_runs": len(channel_jobs), "n_N0_runs": len(n0_jobs),
                   "n_predictions_planned": 6 * len(channel_jobs),
                   "predictions_note": "Not independent samples: repeated MC of the same condition, and the same inputs scored by several models."},
        "pilot": {"depths_mm": list(PILOT_DEPTHS), "mua_mm_inv": list(PILOT_MUAS), "repeats": list(REPEATS),
                  "n_channel_runs": pilot_channel, "n_N0_runs": len(n0_jobs),
                  "purpose": "Check numerical quality before the full matrix; CNN predictions stay closed."},
        "mc": {"config_hash": config_hash(config), "photon_tiers": list(PHOTON_TIERS), "n0_photons": N0_PHOTONS,
               "batch_photons": batch_photons, "gpuid": config["mc"]["gpuid"],
               "seed_rule": "N0: 30000000 + repeat*10000. Channel: 31000000 + tier*100000 + repeat*10000 + mua_index*100 + depth_index.",
               "runner": "simulate.run(config, output, depth, mua, photons, seed) with configs/simulation.json unchanged"},
        "seed_independence": seeds,
        "readout_version": READOUT_VERSION,
        "tool_versions": {"mc_environment": venv_versions(),
                          "analysis_environment": {"python": platform.python_version(),
                                                   **{p: importlib.metadata.version(p) for p in ("numpy", "scipy", "torch")}}},
        "locked_files": {p: sha256(L.at(p)) for p in tracked},
        "training_data": {"dataset": {"path": rel(data_path), "sha256": dataset_sha},
                          "lut": {"path": rel(lut_path), "sha256": sha256(lut_path)}},
        "checkpoints": lock_checkpoints(dataset_sha),
        "geometry_check": check_geometry(config),
        "quality_plan": quality_plan(),
        "analysis_plan": analysis_plan(train_mean_depth),
        "n0_runs": n0_jobs,
        "channel_runs": channel_jobs,
    }
    text = json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    MANIFEST.write_text(text, encoding="utf-8", newline="\n")
    SIDECAR.write_text(f"{sha256(MANIFEST)}  {MANIFEST.name}\n", encoding="utf-8", newline="\n")
    print(f"LOCKED {rel(MANIFEST)} sha256={sha256(MANIFEST)}")
    print(json.dumps({"conditions": 60, "channel_runs": len(channel_jobs), "N0_runs": len(n0_jobs),
                      "pilot_channel_runs": pilot_channel, "seed_independence": seeds,
                      "train_mean_depth_mm": train_mean_depth}, indent=2))


if __name__ == "__main__":
    main()
