"""Figure for the slide: where the background (nền) is taken for the slide-4 example point.
Same example selection as figures_summary.py. Exploratory; no ground truth. Writes real_hand/figs/background_region.png.
"""
import csv
from pathlib import Path

import _paths as RP  # noqa: E402  (adds exp2_layered/code to sys.path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image
from scipy import ndimage as ndi

import measure_depth as T

R = RP.OUT / "depth_points"
FIG = RP.FIG
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
GREEN, ORANGE, PURPLE, TEAL, GREY = "#2FB86A", "#E07B2E", "#9B59B6", "#0F6E78", "#5B6770"


def internals(img, mask, hand, y, x, nx, ny):
    """Same steps as T.profile, but returns every intermediate."""
    t_mm = np.arange(-16, 16.001, 0.125)
    s_mm = np.linspace(-T.AVG_MM, T.AVG_MM, 7)
    tx, ty = -ny, nx
    Tg, Sg = np.meshgrid(t_mm / T.MM_PER_PX, s_mm / T.MM_PER_PX)
    X = x + Tg * nx + Sg * tx; Y = y + Tg * ny + Sg * ty
    p = ndi.map_coordinates(img, [Y, X], order=1).mean(0) - T.DARK
    mv = ndi.map_coordinates(mask.astype(float), [Y[3], X[3]], order=0) > 0.5
    c = len(t_mm) // 2; lo = hi = c
    while lo > 0 and mv[lo - 1]: lo -= 1
    while hi < len(mv) - 1 and mv[hi + 1]: hi += 1
    width = (hi - lo + 1) * 0.125
    others = mv.copy(); others[lo:hi + 1] = False
    oth_d = ndi.binary_dilation(others, iterations=int(round(1.0 / 0.125)))
    fit = ~oth_d & (np.abs(t_mm) > T.EXCL_MM)
    base = np.polyval(np.polyfit(t_mm[fit], p[fit], 2), t_mm)
    r = p / base
    repl = oth_d & (np.abs(t_mm) > width)
    r[repl] = 1.0
    return t_mm, X, Y, p, base, r, fit, oth_d, tx, ty


def main():
    with open(R / "points_cal.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    img_n = {}
    for q in rows:
        img_n[q["image"]] = img_n.get(q["image"], 0) + 1
    con = np.array([float(q["contrast"]) for q in rows])
    names = sorted(img_n)
    ex = max(names[::25], key=lambda n: img_n[n])
    im = np.asarray(Image.open(T.DS / "images" / f"{ex}.png"), float)
    mask = np.asarray(Image.open(T.DS / "masks" / f"{ex}.png")) > 0
    hand = T.hand_roi(im); mk = mask & hand
    pts, dt = T.centre_points(mk); cx, cy = T.directions(mk)
    cand = []
    for (y, x) in pts:
        o = T.profile(im, mk, hand, y, x, cx[y, x], cy[y, x], dt)
        if o is not None:
            cand.append((y, x, o))
    cmed = float(np.median(con))
    ok = [c_ for c_ in cand if np.abs(c_[2][0][:12] - 1).max() < 0.015 and np.abs(c_[2][0][-12:] - 1).max() < 0.015] or cand
    y, x, _ = min(ok, key=lambda t: abs(t[2][2] - 1.3 * cmed))
    nx, ny = cx[y, x], cy[y, x]
    t, X, Y, p, base, r, fit, oth, tx, ty = internals(im, mk, hand, y, x, nx, ny)
    centre = np.abs(t) <= T.EXCL_MM

    fig = plt.figure(figsize=(16, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[1, 1], hspace=0.45, wspace=0.12)
    a0 = fig.add_subplot(gs[:, 0]); a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1], sharex=a1)

    # image, zoomed around the strip
    pad = 25
    xs_, ys_ = X.ravel(), Y.ravel()
    x0, x1 = int(xs_.min()) - pad, int(xs_.max()) + pad; y0, y1 = int(ys_.min()) - pad - 30, int(ys_.max()) + pad + 30
    a0.imshow(np.where(hand, im, im * 0.35), cmap="gray", vmin=0, vmax=255)
    a0.contour(mk, [0.5], colors="#5EC8D8", linewidths=0.8)
    # strip outline (7 parallel lines, +-1.5 mm)
    corners = [(X[0, 0], Y[0, 0]), (X[0, -1], Y[0, -1]), (X[-1, -1], Y[-1, -1]), (X[-1, 0], Y[-1, 0])]
    a0.add_patch(Polygon(corners, closed=True, fill=False, ec="white", lw=1, ls="--"))
    # colour each sample along the centre line by its role
    def seg(sel, col, lw=7):
        idx = np.nonzero(sel)[0]
        if len(idx) == 0:
            return
        breaks = np.nonzero(np.diff(idx) > 1)[0]
        for part in np.split(idx, breaks + 1):
            a0.plot(X[3, part], Y[3, part], color=col, lw=lw, solid_capstyle="butt", alpha=0.9)
    seg(fit, GREEN); seg(centre, ORANGE); seg(oth & ~centre, PURPLE)
    a0.plot(x, y, "o", color="yellow", ms=8, mec="black")
    a0.set_xlim(x0, x1); a0.set_ylim(y1, y0); a0.axis("off")
    a0.set_title("Dải đo 32 × 3 mm trên ảnh (phóng to)")
    a0.plot([], [], color=GREEN, lw=6, label="dùng để tìm nền (cách tâm > 6 mm)")
    a0.plot([], [], color=ORANGE, lw=6, label="vùng bóng mạch (±6 mm): không dùng")
    a0.plot([], [], color=PURPLE, lw=6, label="mạch khác: không dùng")
    a0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=11)

    # brightness + background
    for sel, col in ((centre, ORANGE), (oth & ~centre, PURPLE)):
        a1.plot(np.where(sel, t, np.nan), np.where(sel, p, np.nan), ".", color=col, ms=4)
    a1.plot(np.where(fit, t, np.nan), np.where(fit, p, np.nan), ".", color=GREEN, ms=5)
    a1.plot(t, base, color="black", lw=2, ls="--", label="nền ước lượng (đường cong bậc hai qua các điểm xanh)")
    a1.axvspan(-T.EXCL_MM, T.EXCL_MM, color=ORANGE, alpha=0.08)
    i0 = np.argmin(np.abs(t))
    a1.annotate(f"thật {p[i0]:.0f}", (0, p[i0]), (2.2, p[i0] - 2), fontsize=11, color=TEAL, arrowprops=dict(arrowstyle="->", color=TEAL))
    a1.annotate(f"nền {base[i0]:.0f}", (0, base[i0]), (2.2, base[i0] + 3), fontsize=11, color="black", arrowprops=dict(arrowstyle="->"))
    a1.set_ylabel("Độ sáng (0–255)"); a1.set_title("1) Độ sáng thật dọc dải đo và nền"); a1.legend(frameon=False, fontsize=10, loc="lower left")

    a2.plot(t, r, color=TEAL, lw=2); a2.axhline(1, color="#C9D1D6", lw=1)
    a2.axvspan(-T.EXCL_MM, T.EXCL_MM, color=ORANGE, alpha=0.08)
    a2.annotate(f"{p[i0]:.0f} / {base[i0]:.0f} = {r[i0]:.2f}".replace(".", ","), (0, r[i0]), (2.2, r[i0] + 0.01), fontsize=11, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL))
    a2.set_xlabel("Vị trí ngang qua mạch (mm)"); a2.set_ylabel("Thật ÷ nền"); a2.set_title("2) Chia từng vị trí cho nền của chính nó → nền = 1")
    fig.savefig(FIG / "background_region.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("centre raw", p[i0], "base", base[i0], "ratio", r[i0], "fit n", int(fit.sum()))


if __name__ == "__main__":
    main()
