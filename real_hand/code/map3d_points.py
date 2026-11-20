"""3D vein depth map for one real image (exploratory, no ground truth).
Depth is predicted at every centre-line pixel (ridge of the distance transform of the vessel mask), not every 14 px.
Image = the slide-4 example (same selection as figures_summary.py); named only by an anonymous label.
Usage: map3d_points.py [n_images]   -> real_hand/figs/map3d_points_<k>.png
"""
import csv
import sys
from pathlib import Path

import _paths as RP  # noqa: E402  (adds exp2_layered/code to sys.path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

N_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 1
sys.argv = sys.argv[:1]          # measure_depth parses argv at import (argv[1] = mm/px)
import measure_depth as T  # noqa: E402

R = RP.OUT / "depth_points"
FIG = RP.FIG
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def dense_depth(name, L):
    im = np.asarray(Image.open(T.DS / "images" / f"{name}.png"), float)
    mask = np.asarray(Image.open(T.DS / "masks" / f"{name}.png")) > 0
    hand = T.hand_roi(im); mk = mask & hand
    dt = ndi.distance_transform_edt(mk)
    ridge = (dt >= ndi.maximum_filter(dt, 3)) & (dt >= 2)
    cx, cy = T.directions(mk)
    ys, xs = np.nonzero(ridge)
    keep, prof = [], []
    for y, x in zip(ys, xs):
        o = T.profile(im, mk, hand, y, x, cx[y, x], cy[y, x], dt)
        if o is not None:
            keep.append((y, x)); prof.append(o[0])
    X = torch.from_numpy(np.array(prof)[:, None, :])
    with torch.no_grad():
        mu = np.mean([m(X)[:, 0].numpy() for m in L], 0)
    yx = np.array(keep)
    return im, hand, mk, yx, mu


def main():
    n_img = N_IMG
    with open(R / "points_cal.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cnt = {}
    for q in rows:
        cnt[q["image"]] = cnt.get(q["image"], 0) + 1
    names = sorted(cnt)
    picks = [max(names[::25], key=lambda n: cnt[n])] + names[::max(1, len(names) // 4)][1:n_img]
    L = T.models()
    for k, name in enumerate(picks[:n_img]):
        im, hand, mk, yx, mu = dense_depth(name, L)
        s = T.MM_PER_PX
        Xmm, Ymm = yx[:, 1] * s, yx[:, 0] * s
        fig = plt.figure(figsize=(15, 6.2))
        a = fig.add_subplot(1, 2, 1, projection="3d")
        sc = a.scatter(Xmm, Ymm, -mu, c=-mu, cmap="jet", vmin=-4, vmax=-1, s=3, depthshade=False)
        a.set_xlabel("X (mm)"); a.set_ylabel("Y (mm)"); a.set_zlabel("Độ sâu (mm)")
        a.set_zlim(-4.5, 0); a.invert_yaxis(); a.view_init(elev=32, azim=-60)
        a.set_title("Độ sâu tĩnh mạch dự đoán (3D)")
        b = fig.add_subplot(1, 2, 2)
        ys_, xs_ = np.nonzero(hand)
        b.imshow(np.where(hand, im, im * 0.3), cmap="gray", vmin=0, vmax=255)
        b.scatter(yx[:, 1], yx[:, 0], c=-mu, cmap="jet", vmin=-4, vmax=-1, s=2)
        b.set_xlim(xs_.min() - 5, xs_.max() + 5); b.set_ylim(ys_.max() + 5, ys_.min() - 5); b.axis("off")
        b.set_title(f"Cùng kết quả nhìn từ trên xuống ({len(mu):,} điểm)".replace(",", "."))
        cb = fig.colorbar(sc, ax=[a, b], shrink=0.8, pad=0.02); cb.set_label("Độ sâu (mm, âm = dưới da)")
        fig.savefig(FIG / f"map3d_points_{k}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
        print(k, "mm/px", T.MM_PER_PX, len(mu), "median", np.median(mu), "p5-p95", np.percentile(mu, [5, 95]))


if __name__ == "__main__":
    main()
