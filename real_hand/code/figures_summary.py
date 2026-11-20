"""real-hand figures for the real-hand slide deck (hand-ROI run, tag roi). Exploratory, no ground truth.
Images are referred to by an anonymous index (Ảnh A, B, ...); file names (real names) never appear.
Writes PNGs to real_hand/figs and real_hand/results/depth_points/summary_cal.json.
Main run = tag cal (measured scale 156,6 mm / 752 px = 0,2082 mm/px, 16/11/2026).
"""
import csv
import json
from pathlib import Path

import _paths as RP  # noqa: E402  (adds exp2_layered/code to sys.path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi

import cnn_variants as V
import measure_depth as T

R = RP.OUT / "depth_points"
DS = T.DS
FIG = RP.FIG
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
TEAL, RUST, GREY = "#0F6E78", "#C2562B", "#5B6770"


def load(tag):
    with open(R / f"points_{tag}.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("depth_mm", "sigma_mm", "contrast", "mask_width_mm"):
            r[k] = float(r[k])
        r["x"], r["y"] = int(r["x"]), int(r["y"])
    return rows


def old_hand(img):
    return ndi.binary_opening(ndi.gaussian_filter(img, 4) > T.HAND_GREY, iterations=5)


def stats(rows):
    d = np.array([r["depth_mm"] for r in rows])
    img = {}
    for r in rows:
        img.setdefault(r["image"], []).append(r["depth_mm"])
    per_img = {k: float(np.median(v)) for k, v in img.items()}
    hand = {}
    for k, v in per_img.items():
        hand.setdefault(k.rsplit("_", 1)[0], []).append(v)
    rep = np.array([np.std(v, ddof=1) for v in hand.values() if len(v) >= 3])
    hm = {k: float(np.median(v)) for k, v in hand.items()}
    pers = {}
    for k, v in hm.items():
        p, side = k.rsplit("_", 1); pers.setdefault(p, {})[side] = v
    lr = np.array([[v["L"], v["R"]] for v in pers.values() if "L" in v and "R" in v])
    return d, img, per_img, hand, rep, hm, lr


def main():
    rows = load("cal"); d, img, per_img, hand, rep, hm, lr = stats(rows)
    s = np.array([r["sigma_mm"] for r in rows]); con = np.array([r["contrast"] for r in rows])
    names = sorted(img)
    anon = {n: i for i, n in enumerate(names)}           # anonymous index only

    # ── leak of the old threshold + hand width (scale cross-check) over all images
    leak, width_px = [], []
    for f in sorted((DS / "images").glob("*.png")):
        im = np.asarray(Image.open(f), float)
        h_new = T.hand_roi(im); h_old = old_hand(im)
        leak.append((h_old & ~ndi.binary_dilation(h_new, iterations=10)).sum() / max(h_new.sum(), 1))
        rows_w = h_new[: im.shape[0] // 2].sum(1)        # widest row in the upper (knuckle) half
        width_px.append(np.percentile(rows_w, 95))
    leak = np.array(leak); width_px = np.array(width_px)

    sens = {}
    for tag in ("cal", "cal_s202", "cal_s214", "cal_dark10", "cal_excl9", "roi", "main"):
        if (R / f"points_{tag}.csv").exists():
            q = load(tag); sens[tag] = {"depth_median": float(np.median([r["depth_mm"] for r in q])), "n": len(q),
                                        "n_images": len({r["image"] for r in q})}
    out = {"n_points": len(rows), "n_images": len(img), "n_hands": len(hand), "n_LR_pairs": int(len(lr)),
           "depth_median": float(np.median(d)), "depth_p5_p95": np.percentile(d, [5, 95]).tolist(),
           "depth_iqr": np.percentile(d, [25, 75]).tolist(),
           "sigma_median": float(np.median(s)), "contrast_real_median": float(np.median(con)),
           "repeat_sd_per_hand_median_mm": float(np.median(rep)), "between_hand_sd_mm": float(np.std(list(hm.values()), ddof=1)),
           "LR_corr": float(np.corrcoef(lr[:, 0], lr[:, 1])[0, 1]),
           "old_threshold_leak_frac_median": float(np.median(leak)), "old_threshold_leak_images_gt5pct": int((leak > 0.05).sum()),
           "hand_width_px_median": float(np.median(width_px)), "hand_width_px_p5_p95": np.percentile(width_px, [5, 95]).tolist(),
           "n_images_all": len(leak), "sensitivity": sens,
           "mm_per_px": T.MM_PER_PX, "hand_width_mm_median": float(np.median(width_px)) * T.MM_PER_PX,
           "sens_range_mm": float(max(abs(v["depth_median"] - sens["cal"]["depth_median"]) for k, v in sens.items() if k.startswith("cal")))}
    (R / "summary_cal.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))

    # example image: the one with most points among a fixed stride (deterministic)
    ex = max(names[::25], key=lambda n: len(img[n]))
    im = np.asarray(Image.open(DS / "images" / f"{ex}.png"), float)
    mask = np.asarray(Image.open(DS / "masks" / f"{ex}.png")) > 0
    h_new = T.hand_roi(im); h_old = old_hand(im)

    # ── fig A: cropping the hand
    ys, xs = np.nonzero(h_new); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].imshow(im, cmap="gray", vmin=0, vmax=255); ax[0].set_title("Ảnh gốc 752 × 560 pixel")
    ax[1].imshow(im, cmap="gray", vmin=0, vmax=255)
    ax[1].contour(h_new, [0.5], colors="#3FD17A", linewidths=2)
    ax[1].set_title("Tìm viền bàn tay (xanh)")
    crop = np.where(h_new, im, np.nan)[y0:y1 + 1, x0:x1 + 1]
    ax[2].imshow(crop, cmap="gray", vmin=0, vmax=255); ax[2].set_title("Vùng dùng để đo: chỉ bàn tay")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "crop.png", dpi=130); plt.close(fig)

    # ── fig B: how one point is measured (image with the line + its profile vs sim)
    L = T.models()
    mk = mask & h_new
    pts, dt = T.centre_points(mk); cx, cy = T.directions(mk)
    cand = []
    for (y, x) in pts:
        o = T.profile(im, mk, h_new, y, x, cx[y, x], cy[y, x], dt)
        if o is not None:
            cand.append((y, x, o))
    cmed = float(np.median(con))
    ok = [c_ for c_ in cand if np.abs(c_[2][0][:12] - 1).max() < 0.015 and np.abs(c_[2][0][-12:] - 1).max() < 0.015] or cand
    y, x, (B, wmm, c) = min(ok, key=lambda t: abs(t[2][2] - 1.3 * cmed))   # typical, clean-tailed point
    with torch.no_grad():
        o = [m(torch.from_numpy(B[None, None])) for m in L]
    mu = float(np.mean([q[0, 0] for q in o]))
    nx, ny = cx[y, x], cy[y, x]
    tt = np.array([-16, 16]) / T.MM_PER_PX
    xv, Pv, mv = V.load("validation_ext", "manifest_ext.json"); Bs = V.Binner(xv)(Pv, np.zeros(len(Pv)))
    ys_ = np.array([m["depth_mm"] for m in mv]); near = np.argsort(np.abs(ys_ - mu))[:40]
    t = (V.EDGES[:-1] + V.EDGES[1:]) / 2
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax[0].imshow(np.where(h_new, im, im * 0.35), cmap="gray", vmin=0, vmax=255)
    ax[0].contour(mk, [0.5], colors="#5EC8D8", linewidths=0.8)
    ax[0].plot(x + tt * nx, y + tt * ny, color=RUST, lw=2.5); ax[0].plot(x, y, "o", color="yellow", ms=7)
    ax[0].set_title("Đường cắt ngang mạch: ±16 mm"); ax[0].axis("off")
    ax[1].plot(t, np.median(Bs[near], 0), color=GREY, lw=2, ls="--", label=f"mô phỏng, mạch sâu ≈ {mu:.1f} mm".replace(".", ","))
    ax[1].plot(t, B, color=TEAL, lw=2.2, label="ảnh thật (đã chia cho nền)")
    ax[1].axhline(1, color="#C9D1D6", lw=1)
    ax[1].set_xlabel("Vị trí ngang qua mạch (mm)"); ax[1].set_ylabel("Cường độ / nền")
    ax[1].set_title(f"Profile đưa vào mạng → độ sâu dự đoán {mu:.2f} mm".replace(".", ",")); ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "profile.png", dpi=130); plt.close(fig)

    # ── fig C: depth maps on 4 anonymous images (hand only)
    pick = names[::max(1, len(names) // 4)][:4]
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.8))
    for a, n, lab in zip(axs, pick, "ABCD"):
        im2 = np.asarray(Image.open(DS / "images" / f"{n}.png"), float); h2 = T.hand_roi(im2)
        yy, xx = np.nonzero(h2)
        a.imshow(np.where(h2, im2, im2 * 0.3), cmap="gray", vmin=0, vmax=255)
        q = [r for r in rows if r["image"] == n]
        sc = a.scatter([r["x"] for r in q], [r["y"] for r in q], c=[r["depth_mm"] for r in q], s=22, cmap="viridis", vmin=1.5, vmax=3.5,
                       edgecolors="white", linewidths=0.3)
        a.set_xlim(xx.min() - 5, xx.max() + 5); a.set_ylim(yy.max() + 5, yy.min() - 5)
        a.set_title(f"Ảnh {lab} · trung vị {per_img[n]:.2f} mm".replace(".", ","), fontsize=13); a.axis("off")
    fig.colorbar(sc, ax=axs, shrink=0.85, label="Độ sâu dự đoán (mm)")
    fig.savefig(FIG / "maps.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # ── fig D: distribution + real vs sim shadow
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].hist(d, bins=60, range=(0.5, 5.0), color=TEAL)
    ax[0].axvline(np.median(d), color=RUST, lw=2); ax[0].text(np.median(d) + 0.08, ax[0].get_ylim()[1] * 0.9, f"trung vị {np.median(d):.2f} mm".replace(".", ","), color=RUST)
    ax[0].set_xlabel("Độ sâu dự đoán (mm)"); ax[0].set_ylabel("Số điểm đo"); ax[0].set_title(f"{len(rows):,} điểm trên {len(img)} ảnh".replace(",", "."))
    ax[1].hist(con * 100, bins=25, range=(0, 25), weights=np.full(len(con), 100 / len(con)), color=TEAL, alpha=0.75, label=f"ảnh thật ({len(con):,} điểm)".replace(",", "."))
    cs = (1 - Bs.min(1)) * 100; ax[1].hist(cs, bins=25, range=(0, 25), weights=np.full(len(cs), 100 / len(cs)), histtype="step", lw=2.4, color=RUST, label=f"mô phỏng ({len(cs)} mạch)")
    ax[1].set_xlabel("Độ tương phản bóng mạch (%)"); ax[1].set_ylabel("Tỉ lệ số mạch (%)"); ax[1].legend(frameon=False); ax[1].set_title("Bóng mạch đậm nhạt: thật và mô phỏng")
    for lo, hi, col in ((1.5, 2.5, TEAL), (2.5, 3.5, RUST)):
        sel = (d >= lo) & (d < hi); ss = (ys_ >= lo) & (ys_ < hi)
        P = np.load(R / "profiles_cal.npy")
        ax[2].plot(t, np.median(P[sel], 0), color=col, lw=2.2, label=f"thật, đoán {lo}–{hi} mm".replace(".", ","))
        ax[2].plot(t, np.median(Bs[ss], 0), color=col, lw=1.5, ls="--", label=f"mô phỏng, sâu {lo}–{hi} mm".replace(".", ","))
    ax[2].set_xlabel("Vị trí ngang (mm)"); ax[2].set_ylabel("Cường độ / nền"); ax[2].legend(frameon=False, fontsize=10)
    ax[2].set_title("Hình dạng bóng mạch trung vị")
    fig.tight_layout(); fig.savefig(FIG / "dist.png", dpi=130); plt.close(fig)

    # ── fig E: repeatability
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ks = sorted(hand, key=lambda k: np.median(hand[k]))
    for i, k in enumerate(ks):
        ax[0].plot([i] * len(hand[k]), hand[k], ".", color=TEAL, ms=4)
    ax[0].set_xlabel("Bàn tay (xếp theo trung vị)"); ax[0].set_ylabel("Trung vị độ sâu mỗi ảnh (mm)")
    ax[0].set_title(f"Chụp lại cùng bàn tay: lệch {np.median(rep):.2f} mm\nkhác nhau giữa các bàn tay: {out['between_hand_sd_mm']:.2f} mm".replace(".", ","))
    ax[1].plot(lr[:, 0], lr[:, 1], "o", color=TEAL); m_ = [lr.min() - 0.1, lr.max() + 0.1]; ax[1].plot(m_, m_, color=GREY, lw=1)
    ax[1].set_xlabel("Tay trái (mm)"); ax[1].set_ylabel("Tay phải (mm)"); ax[1].set_title(f"Trái và phải cùng người: r = {out['LR_corr']:.2f} ({len(lr)} người)".replace(".", ","))
    ax[1].set_aspect("equal")
    fig.tight_layout(); fig.savefig(FIG / "repeat.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
