"""Analyse the step 02 layered-skin job.

A) Contrast of the vessel profile in the layered model (light and darker skin).
B) Depth CRLB with nine parameters: depth, vessel ln mua, lateral shift, vessel diameter and five
   layer nuisances (ln epidermal mua = melanin, ln dermal mus', ln fat mus', ln fat mua, dermal
   thickness). Nested sets and single-nuisance penalties; white noise sigma per bin.
C) Model mismatch: the homogeneous physics fit B3 (stage-4 LUT, non-scattering vessel in a
   homogeneous background) applied to the layered profiles -> depth bias if the inverse model
   ignores skin layers.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from readout import load_run, profile  # noqa: E402
from b3_physics_fit import Forward, fit  # noqa: E402

RES = exp2_paths.EXP2 / "02_forward_model/results/layered"
SIGMAS = (0.001, 0.003, 0.01)
LN = lambda s: np.log(1 + s) - np.log(1 - s)  # noqa: E731


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = json.loads((RES / "design.json").read_text(encoding="utf-8"))
    n0 = {p.stem[3:]: load_run(p) for p in RES.glob("n0_*.npz") if not p.name.endswith(".invalid.npz")}
    prof = lambda name, bg="base": profile(load_run(RES / f"{name}.npz"), n0[bg])[1].astype(float)  # noqa: E731
    fwd = Forward()
    order = ["depth", "ln_vmua", "shift", "diameter", "ln_epi_mua", "ln_derm_musp", "ln_fat_musp", "ln_fat_mua", "dermis_mm"]
    out = {"points": []}
    for z in d["points"]:
        t = f"z{z:.1f}"
        base = prof(f"{t}_base"); darkp = prof(f"{t}_dark", "dark")
        J = {"depth": (prof(f"{t}_depth+") - prof(f"{t}_depth-")) / (2 * d["depth_step"]),
             "ln_vmua": (prof(f"{t}_vmua+") - prof(f"{t}_vmua-")) / LN(d["mua_step"]),
             "shift": -np.gradient(base, 0.25),
             "diameter": (prof(f"{t}_diam+") - prof(f"{t}_diam-")) / (2 * d["d_step"])}
        for k, (_, st) in d["bg_steps"].items():
            p, m = prof(f"{t}_{k}+", f"{k}+"), prof(f"{t}_{k}-", f"{k}-")
            J[("" if k == "dermis_mm" else "ln_") + k] = (p - m) / (2 * st if k == "dermis_mm" else LN(st))
        M = np.stack([J[k] for k in order], 1)
        F = M.T @ M
        nested = {}
        for k in range(1, len(order) + 1):
            nested[order[k - 1]] = float(np.sqrt(np.linalg.inv(F[:k, :k])[0, 0]))
        s1 = nested["depth"]
        single = {k: float(np.sqrt(np.linalg.inv(F[np.ix_([0, j], [0, j])])[0, 0]) / s1) for j, k in enumerate(order[1:], 1)}
        # Which nuisances matter most once all others are in: drop-one penalties.
        full = nested[order[-1]]
        drop = {}
        for j, k in enumerate(order[1:], 1):
            idx = [i for i in range(len(order)) if i != j]
            drop[k] = float(full / np.sqrt(np.linalg.inv(F[np.ix_(idx, idx)])[0, 0]))
        rec = {"depth_mm": z, "contrast_light": float(1 - base.min()), "contrast_dark": float(1 - darkp.min()),
               "penalty_nested_cumulative": {k: v / s1 for k, v in nested.items()},
               "single_nuisance_penalty": single, "drop_one_gain": drop,
               "sqrt_crlb_full_mm": {f"sigma_{s}": full * s for s in SIGMAS},
               "sqrt_crlb_depth_only_mm": {f"sigma_{s}": s1 * s for s in SIGMAS}}
        # C) homogeneous-model fit applied to layered data
        for lab, y in (("light", base), ("dark", darkp)):
            zj, aj, sj, cj = fit(fwd, y, True)
            rec[f"homogeneous_B3_fit_{lab}"] = {"pred_depth_mm": zj, "bias_mm": zj - z, "mua_fit": aj, "shift_mm": sj,
                                                "rms_residual": float(np.sqrt(2 * cj / len(y)))}
        out["points"].append(rec)
    (RES / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for r in out["points"]:
        print(f"z={r['depth_mm']} contrast light {r['contrast_light']:.3f} dark {r['contrast_dark']:.3f}")
        print("  cumulative penalty:", " ".join(f"{k}={v:.1f}" for k, v in r["penalty_nested_cumulative"].items()))
        print("  single:", " ".join(f"{k}={v:.1f}" for k, v in r["single_nuisance_penalty"].items()))
        print("  drop-one gain:", " ".join(f"{k}={v:.2f}" for k, v in r["drop_one_gain"].items()))
        print("  sqrtCRLB full:", " ".join(f"{k}={v:.3f}" for k, v in r["sqrt_crlb_full_mm"].items()))
        for lab in ("light", "dark"):
            h = r[f"homogeneous_B3_fit_{lab}"]
            print(f"  B3-homogeneous on {lab}: pred {h['pred_depth_mm']:.2f} bias {h['bias_mm']:+.2f} mua {h['mua_fit']:.3f} rms {h['rms_residual']:.4f}")


if __name__ == "__main__":
    main()
