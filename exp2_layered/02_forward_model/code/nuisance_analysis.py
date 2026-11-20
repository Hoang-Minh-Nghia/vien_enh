"""Analyse the step 01/step 02 GPU job.

A) Farrell 1992 diffusion dipole vs MC radial reflectance (homogeneous, pencil beam).
B) Jacobian of the 128-point profile in six parameters and depth CRLB as nuisances are added:
   S1 depth only; S2 + channel mua; S3 + lateral shift; S4 + background mus';
   S5 + channel diameter; S6 + background mua. White noise sigma per bin (ratio units).
   Penalty = sqrt(CRLB_S / CRLB_S1) does not depend on sigma.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import json
import sys
from pathlib import Path

import numpy as np

ROOT = exp2_paths.EXP1
sys.path.insert(0, str(exp2_paths.exp1_at("common")))
from readout import load_run, profile  # noqa: E402

RES = exp2_paths.EXP2 / "02_forward_model/results/nuisance"
SIGMAS = (0.001, 0.003, 0.01)


def farrell(rho, mua, musp, n):
    """Steady-state diffuse reflectance R(rho) [1/mm^2], Farrell, Patterson & Wilson 1992."""
    mutp = mua + musp
    ap = musp / mutp
    mueff = np.sqrt(3 * mua * mutp)
    ri = -1.440 / n ** 2 + 0.710 / n + 0.668 + 0.0636 * n
    A = (1 + ri) / (1 - ri)
    z0 = 1 / mutp
    zb = 2 * A / (3 * mutp)
    r1 = np.sqrt(z0 ** 2 + rho ** 2)
    r2 = np.sqrt((z0 + 2 * zb) ** 2 + rho ** 2)
    return ap / (4 * np.pi) * (z0 * (mueff + 1 / r1) * np.exp(-mueff * r1) / r1 ** 2
                               + (z0 + 2 * zb) * (mueff + 1 / r2) * np.exp(-mueff * r2) / r2 ** 2)


def part_a():
    d = np.load(RES / "t1_farrell_mc.npz")
    rho, mc = d["rho_mm"], d["R_mc"]
    th = farrell(rho, 0.01, 1.0, 1.35)
    sel = (rho >= 1.0) & (rho <= 10.0)
    rel = mc[sel] / th[sel] - 1
    scale = np.exp(np.mean(np.log(mc[sel] / th[sel])))
    shape = mc[sel] / (scale * th[sel]) - 1
    return {"rho_range_mm": [1.0, 10.0], "absolute_rel_diff_median": float(np.median(rel)),
            "absolute_rel_diff_max_abs": float(np.abs(rel).max()), "best_scale": float(scale),
            "shape_rel_diff_max_abs": float(np.abs(shape).max()),
            "table": [{"rho_mm": float(r), "R_mc": float(m), "R_farrell": float(t)} for r, m, t in zip(rho[sel], mc[sel], th[sel])]}


def prof(path, n0):
    ch = load_run(path)
    cm = json.loads(str(ch["metadata_json"]))
    bm = json.loads(str(n0["metadata_json"]))
    if cm["config"]["geometry"] != bm["config"]["geometry"]:
        # simulate.py only uses diameter_mm when a channel exists (depth is not None), so the
        # N0 field does not depend on it. Allow exactly that one difference, nothing else.
        g_ch, g_bg = dict(cm["config"]["geometry"]), dict(bm["config"]["geometry"])
        g_ch.pop("diameter_mm"); g_bg.pop("diameter_mm")
        assert g_ch == g_bg, "N0 geometry differs beyond diameter_mm"
        bm["config"]["geometry"]["diameter_mm"] = cm["config"]["geometry"]["diameter_mm"]
        n0 = dict(n0, metadata_json=np.array(json.dumps(bm)))
    return profile(ch, n0)[1].astype(float)


def part_b():
    design = json.loads((RES / "design.json").read_text(encoding="utf-8"))
    steps = design["steps"]
    n0 = {k: load_run(RES / f"n0_{k}.npz") for k in ("base", "musp+", "musp-", "mua_bg+", "mua_bg-")}
    out = []
    for z, a in design["points"]:
        tag = f"z{z:.1f}_m{a:.3f}"
        base = prof(RES / f"{tag}_base.npz", n0["base"])
        def pair(name, n0p="base", n0m="base"):
            return prof(RES / f"{tag}_{name}+.npz", n0[n0p]), prof(RES / f"{tag}_{name}-.npz", n0[n0m])
        J = {}
        p, m = pair("depth"); J["depth"] = (p - m) / (2 * steps["depth"])
        p, m = pair("mua"); J["ln_mua"] = (p - m) / (np.log(1 + steps["mua"]) - np.log(1 - steps["mua"]))
        J["shift"] = -np.gradient(base, 0.25)
        p, m = pair("musp", "musp+", "musp-"); J["ln_musp"] = (p - m) / (np.log(1 + steps["musp"]) - np.log(1 - steps["musp"]))
        p, m = pair("diameter"); J["diameter"] = (p - m) / (2 * steps["diameter"])
        p, m = pair("mua_bg", "mua_bg+", "mua_bg-"); J["mua_bg"] = (p - m) / (2 * steps["mua_bg"])
        order = ["depth", "ln_mua", "shift", "ln_musp", "diameter", "mua_bg"]
        M = np.stack([J[k] for k in order], axis=1)
        F = M.T @ M
        crlb = {}
        for k in range(1, len(order) + 1):
            Fi = np.linalg.inv(F[:k, :k])
            crlb[f"S{k}"] = float(np.sqrt(Fi[0, 0]))          # times sigma gives sqrt CRLB in mm
        base_c = crlb["S1"]
        # Correlation of each nuisance derivative with the depth derivative.
        corr = {k: float(J[k] @ J["depth"] / np.sqrt((J[k] @ J[k]) * (J["depth"] @ J["depth"]))) for k in order[1:]}
        # Single-nuisance penalties (depth + one nuisance).
        single = {}
        for j, k in enumerate(order[1:], 1):
            Fi = np.linalg.inv(F[np.ix_([0, j], [0, j])])
            single[k] = float(np.sqrt(Fi[0, 0]) / base_c)
        out.append({"depth_mm": z, "mua": a, "contrast": float(1 - base.min()),
                    "penalty_vs_S1": {s: c / base_c for s, c in crlb.items()},
                    "single_nuisance_penalty": single, "corr_with_depth_derivative": corr,
                    "sqrt_crlb_mm": {f"sigma_{s}": {k: c * s for k, c in crlb.items()} for s in SIGMAS}})
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    result = {"farrell_check": part_a(), "nuisance_crlb": part_b()}
    (RES / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    a = result["farrell_check"]
    print("FARRELL abs median %.3f max %.3f | scale %.3f shape max %.3f" % (a["absolute_rel_diff_median"], a["absolute_rel_diff_max_abs"], a["best_scale"], a["shape_rel_diff_max_abs"]))
    print("depth mua | penalty S2..S6 | sqrtCRLB S6 @0.003 | single-nuisance penalties")
    for r in result["nuisance_crlb"]:
        pen = " ".join(f"{r['penalty_vs_S1'][f'S{k}']:.1f}" for k in range(2, 7))
        sing = " ".join(f"{k}={v:.1f}" for k, v in r["single_nuisance_penalty"].items())
        print(f"{r['depth_mm']} {r['mua']} | {pen} | {r['sqrt_crlb_mm']['sigma_0.003']['S6']:.3f} mm | {sing}")


if __name__ == "__main__":
    main()
