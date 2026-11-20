# real_hand

Applies the layered-skin model (`exp2_layered/05_training`, hetero ensemble) to real 850 nm reflectance photos of the back of the hand.
There is **no ground-truth depth**: outputs are exploratory.

**No images or derived data are part of this repository.** Provide your own data folder:

```
$VEIN_REAL_DATA/images/<name>.png   # grayscale reflectance image
$VEIN_REAL_DATA/masks/<name>.png    # vessel mask (non-zero = vessel), same file name
```

| Script | Purpose |
|---|---|
| `measure_depth.py [mm_per_px] [dark_offset] [tag] [excl_mm] [hand_mode]` | Otsu hand ROI, centre-line points, perpendicular profile ±16 mm averaged over ±1.5 mm along the vessel, quadratic background fitted at \|t\| > 6 mm, ratio → network → depth. Writes `results/depth_points/points_<tag>.csv`, `profiles_<tag>.npy`, `meta_<tag>.json` |
| `figures_overview.py`, `figures_summary.py`, `figure_background.py` | Summary figures (ROI, example profile, depth maps, distributions, repeatability, background region) |
| `map3d.py [n_images]` (`map3d_points.py`: earlier sparse version) | Skeleton-based 3D depth map (harmonic gap filling along the vessel graph) as PNG and interactive HTML |

The default scale, 0.2082 mm/px, is specific to the authors' camera setup (field of view 156.6 × 116.6 mm over 752 × 560 px).
Pass your own value. Outputs go to `real_hand/results` and `real_hand/figs` (git-ignored).
Figures label images anonymously; keep it that way for any human data.
