"""step 05 GPU job: 14-parameter finite-difference Jacobians at 40 validation_ext vessels (see report_crlb_14_params.md).

Per point: base + central +/- steps of 4 vessel and 9 layer parameters; every vessel profile is run as two
independent halves (24 M photons each, half-seeds a/b) so an unbiased Fisher matrix J_a^T J_b can be formed.
Within a half, all +/- runs share the same seed (common random numbers).  Each layer perturbation gets its
own N0 (12 M photons, same seed as base N0).  Raw y-mean profiles and N0 scalars are saved; binning/division
happen in the analysis.  Usage: crlb_runs.py manifest | run | check
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generate as G  # noqa: E402

OUT = exp2_paths.EXP2 / "05_error_analysis/results/crlb_14_params"
T4 = G.OUT
SEED, HALF_PH, N0_PH, N_PER_BIN = 20261120, 24_000_000, 12_000_000, 10
BINS = ((0, 2), (2, 3), (3, 4), (4, 9))
# name: (kind, log?, step)   log -> multiply by exp(+-step); else add +-step
STEPS = {"depth_mm": ("vessel", False, 0.2), "diameter_mm": ("vessel", False, 0.4),
         "mua": ("vessel", True, 0.1), "mus": ("vessel", True, 0.1),
         "mel_fraction": ("layer", True, 0.2), "dermis_mm": ("layer", False, 0.2),
         "derm_musp": ("layer", True, 0.2), "derm_mua": ("layer", True, 0.2),
         "fat_mm": ("layer", False, 0.5), "fat_musp": ("layer", True, 0.2),
         "fat_mua": ("layer", True, 0.2), "musc_musp": ("layer", True, 0.2), "musc_mua": ("layer", True, 0.2)}


def admissible(L, v):
    skin = G.EPI_MM + L["dermis_mm"]
    return v["depth_mm"] >= skin + 0.3 and v["depth_mm"] + v["diameter_mm"] <= skin + L["fat_mm"] - 0.9


def all_ok(L, v):
    """Base and every +/- perturbation keep the vessel inside the fat layer with the margins."""
    q = {"id": "?", "layers": L, "vessel": v}
    try:
        return admissible(L, v) and all(perturbed(q, n, sg) for n in STEPS for sg in (1, -1))
    except AssertionError:
        return False


def manifest():
    cf = json.loads((T4 / "manifest_ext.json").read_text(encoding="utf-8"))["splits"]["validation_ext"]["configs"]
    picked, used = {b: [] for b in BINS}, set()
    for c in cf:
        for j, v in enumerate(c["vessels"]):
            b = next(b for b in BINS if b[0] <= v["depth_mm"] < b[1])
            if c["id"] in used or len(picked[b]) >= N_PER_BIN or not all_ok(c["layers"], v):
                continue
            picked[b].append({"source_id": c["id"], "vessel_index": j, "layers": c["layers"], "vessel": v})
            used.add(c["id"])
    r = np.random.default_rng(SEED)
    pts = [dict(p, id=f"pt_{i:02d}", seed_a=int(r.integers(1, 2 ** 31 - 1)), seed_b=int(r.integers(1, 2 ** 31 - 1)))
           for i, p in enumerate(q for b in BINS for q in picked[b])]
    return {"version": "t6a/1", "steps": STEPS, "half_photons": HALF_PH, "n0_photons": N0_PH, "seed": SEED,
            "counts_per_bin": {f"{a}-{b}": len(picked[(a, b)]) for a, b in BINS}, "points": pts}


def perturbed(p, name, sign):
    L, v = copy.deepcopy(p["layers"]), copy.deepcopy(p["vessel"])
    kind, lg, st = STEPS[name]
    d = v if kind == "vessel" else L
    d[name] = d[name] * float(np.exp(sign * st)) if lg else d[name] + sign * st
    assert admissible(L, v), (p["id"], name, sign)
    return L, v


def run():
    m = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    t0 = time.time()
    for k, p in enumerate(m["points"], 1):
        f = OUT / f"{p['id']}.npz"
        if f.exists():
            continue
        prof, n0 = {}, {}
        try:
            settings = [("base", p["layers"], p["vessel"])]
            for name in STEPS:
                for s, sg in (("+", 1), ("-", -1)):
                    settings.append((f"{name}{s}",) + perturbed(p, name, sg))
            for tag, L, v in settings:
                if tag == "base" or STEPS[tag[:-1]][0] == "layer":
                    R, x = G.simulate(L, None, N0_PH, p["seed_a"])
                    n0[tag] = float(R.mean())
                for h in ("a", "b"):
                    prof[f"{tag}|{h}"] = G.simulate(L, v, HALF_PH, p[f"seed_{h}"])[0].mean(1).astype(np.float32)
        except RuntimeError as e:
            (OUT / f"{p['id']}.invalid.json").write_text(json.dumps({"error": str(e)}), encoding="utf-8")
            raise
        tmp = f.with_suffix(".partial.npz")
        np.savez_compressed(tmp, x_mm=x, keys=np.array(list(prof)), profiles=np.stack(list(prof.values())),
                            n0_keys=np.array(list(n0)), n0=np.array(list(n0.values())), point_json=json.dumps(p))
        tmp.replace(f)
        print(f"PT {p['id']} done ({k}/{len(m['points'])}); elapsed {time.time()-t0:.0f}s", flush=True)
    print("T6A_DONE", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "manifest":
        p = OUT / "manifest.json"
        if p.exists():
            raise SystemExit("manifest exists; refusing to overwrite")
        mm = manifest()
        for q in mm["points"]:
            for name in STEPS:
                perturbed(q, name, 1), perturbed(q, name, -1)
        p.write_text(json.dumps(mm, indent=1), encoding="utf-8")
        print("written", p, mm["counts_per_bin"])
    else:
        run()
