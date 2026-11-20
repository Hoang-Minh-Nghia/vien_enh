"""step 02 add-on: separate the two model mismatches seen in step 02.

Same runner as layered_runs.py, but every tissue layer gets the homogeneous project optics
(mua 0.010, mus' 1.0, g 0.511, n 1.35) while the vessel stays scattering whole blood. Fitting these
profiles with the homogeneous B3 isolates the bias caused by vessel scattering alone; the rest of
the step 02 bias is then attributable to the skin layers.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp2_paths  # noqa: E402,F401  (all step code folders on sys.path)
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layered_runs as T  # noqa: E402

T.G_TISSUE = 0.511
T.OUT = exp2_paths.EXP2 / "02_forward_model/results/split"
cfg = copy.deepcopy(T.BASE)
for k in ("epi", "derm", "fat", "musc"):
    cfg["optics"][f"{k}_mua"] = 0.010
    cfg["optics"][f"{k}_musp"] = 1.0
cfg["optics"]["n"] = 1.35
cfg["geometry"]["epidermis_mm"] = cfg["geometry"]["voxel_mm"]   # scale factor 1 for the top voxel
T.OUT.mkdir(parents=True, exist_ok=True)
(T.OUT / "design.json").write_text(json.dumps({"cfg": cfg, "g": 0.511, "vessel": T.VESSEL, "points": T.POINTS}, indent=2), encoding="utf-8")
T.run(cfg, T.OUT / "n0_hom.npz", None, photons=T.N0PH)
for z in T.POINTS:
    T.run(cfg, T.OUT / f"z{z:.1f}_hom_scatvessel.npz", z)
print("SPLIT_DONE", flush=True)
