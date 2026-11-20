"""step 02 + step 01 GPU job (single runner).

Part A (step 01 check): homogeneous medium, pencil beam, radial reflectance vs the Farrell 1992
diffusion dipole (Med Phys 19:879, doi:10.1118/1.596777).

Part B (step 02): finite-difference Jacobian of the 128-point profile with respect to six parameters
(depth, channel mua, background mus', channel diameter, background mua, lateral shift is done
analytically later) at six points, with the same MC seed for every variant at a point so the
differences are correlated. Every channel run is normalised by an N0 run with the matching
background optics. The locked simulate.run is reused unchanged with modified config copies.
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

ROOT = exp2_paths.EXP1
sys.path.insert(0, str(exp2_paths.exp1_at("common")))
import simulate  # noqa: E402

OUT = exp2_paths.EXP2 / "02_forward_model/results/nuisance"
PHOTONS, N0_PHOTONS, SEED = 60_000_000, 120_000_000, 20261110

POINTS = [(1.4, 0.15), (2.8, 0.15), (3.8, 0.15), (1.4, 0.508), (2.8, 0.508), (3.8, 0.508)]
STEPS = {"depth": 0.2, "mua": 0.1, "musp": 0.1, "diameter": 0.4, "mua_bg": 0.005}  # mua, musp relative (+-10%)


def variant(base, name, sign):
    cfg = copy.deepcopy(base)
    if name == "musp":
        cfg["optics"]["background_musp_mm_inv"] *= 1 + sign * STEPS["musp"]
    elif name == "diameter":
        cfg["geometry"]["diameter_mm"] += sign * STEPS["diameter"]
    elif name == "mua_bg":
        cfg["optics"]["background_mua_mm_inv"] += sign * STEPS["mua_bg"]
    return cfg


def part_a():
    cfg = simulate.load_config()
    import pmcxcl
    v = 0.2
    n = (300, 300, 100)
    vol = np.ones(n, dtype=np.uint8, order="F")
    vol[:, :, 0] = 0
    prop = np.array([[0, 0, 1, 1], [0.01, 1.0 / (1 - 0.511), 0.511, 1.35]], dtype=np.float32)
    total = np.zeros(n[:2])
    for b in range(10):
        res = pmcxcl.run(dict(nphoton=6_000_000, vol=vol, prop=prop, tstart=0, tend=5e-9, tstep=5e-9,
                              srctype="pencil", srcpos=[150, 150, 0], srcdir=[0, 0, 1], unitinmm=v, issaveref=1,
                              isreflect=1, bc="ccrcca", issrcfrom0=1, isnormalized=1, outputtype="fluence",
                              autopilot=1, gpuid=1, seed=SEED + b))
        total += np.asarray(res["dref"])[:, :, 0, 0]
    total /= 10
    x = (np.arange(n[0]) + 0.5 - 150) * v
    rr = np.hypot(*np.meshgrid(x, x, indexing="ij"))
    edges = np.arange(0.0, 15.01, 0.4)
    idx = np.digitize(rr.ravel(), edges) - 1
    mc = np.array([total.ravel()[idx == k].mean() for k in range(len(edges) - 1)])
    rho = 0.5 * (edges[:-1] + edges[1:])
    np.savez_compressed(OUT / "t1_farrell_mc.npz", rho_mm=rho, R_mc=mc, voxel_mm=v)
    print("PART_A_DONE", flush=True)


def part_b():
    base = simulate.load_config()
    jobs = []
    for name in ("base", "musp+", "musp-", "mua_bg+", "mua_bg-"):
        cfg = base if name == "base" else variant(base, name[:-1], 1 if name.endswith("+") else -1)
        jobs.append((cfg, OUT / f"n0_{name}.npz", None, 0.508, N0_PHOTONS))
    for z, a in POINTS:
        tag = f"z{z:.1f}_m{a:.3f}"
        jobs.append((base, OUT / f"{tag}_base.npz", z, a, PHOTONS))
        for s, sign in (("+", 1), ("-", -1)):
            jobs.append((base, OUT / f"{tag}_depth{s}.npz", z + sign * STEPS["depth"], a, PHOTONS))
            jobs.append((base, OUT / f"{tag}_mua{s}.npz", z, a * (1 + sign * STEPS["mua"]), PHOTONS))
            for name in ("musp", "diameter", "mua_bg"):
                jobs.append((variant(base, name, sign), OUT / f"{tag}_{name}{s}.npz", z, a, PHOTONS))
    start = time.time()
    for i, (cfg, out, z, a, ph) in enumerate(jobs, 1):
        simulate.run(cfg, out, z, a, ph, SEED)
        print(f"JOB {i}/{len(jobs)} {out.name} elapsed {time.time()-start:.0f}s", flush=True)
    print("PART_B_DONE", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design.json").write_text(json.dumps({"points": POINTS, "steps": STEPS, "photons": PHOTONS,
                                                 "n0_photons": N0_PHOTONS, "seed": SEED}, indent=2), encoding="utf-8")
    part_a()
    part_b()
