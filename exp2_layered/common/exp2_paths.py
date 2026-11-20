"""Locations used by every experiment-2 script. Importing it puts all step code folders on sys.path."""
import sys
from pathlib import Path

EXP2 = Path(__file__).resolve().parents[1]          # exp2_layered/
REPO = EXP2.parent
EXP1 = REPO / "exp1_homogeneous"
STEPS = sorted(p for p in EXP2.iterdir() if p.is_dir() and p.name[:2].isdigit())
for _p in [EXP2 / "common", *[s / "code" for s in STEPS], EXP1 / "common"]:   # EXP1/common: frozen core
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
import exp1_paths  # noqa: E402  (experiment-1 locations)


def exp1_at(rel):
    """Absolute path of an experiment-1 file (accepts the original names, e.g. results/stage4/lut_FK.npz)."""
    return exp1_paths.at(rel)
