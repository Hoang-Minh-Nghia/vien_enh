"""Paths for the real-hand scripts. Real images are NOT part of the repository.

Set VEIN_REAL_DATA to a folder containing images/*.png and masks/*.png (same file names).
Outputs go to real_hand/results and real_hand/figs (both git-ignored).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path.insert(0, str(REPO / "exp2_layered" / "common"))
import exp2_paths  # noqa: E402,F401  (experiment-2 code on sys.path)

EXP2 = exp2_paths.EXP2
OUT = HERE / "results"
FIG = HERE / "figs"
DATA = Path(os.environ.get("VEIN_REAL_DATA", str(HERE / "data")))
FIG.mkdir(parents=True, exist_ok=True)
