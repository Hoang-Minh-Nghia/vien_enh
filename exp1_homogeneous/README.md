# exp1_homogeneous — homogeneous skin

Skin is one homogeneous medium (μs′ = 1.0 mm⁻¹, μa = 0.010 mm⁻¹, g = 0.511, n = 1.35) containing one straight,
non-scattering 3 mm channel ("vein") at depth 0.5–5 mm. Depth = **surface to the upper edge** of the channel.
Wavelength 850 nm, planar illumination, diffuse reflectance (pmcxcl, 0.1 mm voxels). All fixed inputs are in
`configs/` (frozen).

Plan and results: `00_plan_and_results.md` (Vietnamese).

## Steps

| Folder | What it does | Main outputs |
|---|---|---|
| `01_simulator_checks/` | Is the MC converged? Reference channel at 2 mm against a finer grid, double photons, wider domain, longer time window, an independent repeat | `results/report.json`, `stage1_overview.png`, `profile_*.csv` |
| `02_forward_model/` | Depth sweep at standard absorption μa = 0.508 mm⁻¹ (12 depths), interpolation check at 3 held-out depths | `results/lut_K.npz`, `depth_profiles.png` |
| `03_dataset/` | MC at 3 more absorption levels × 12 depths, lookup table, the F/K training set | `data/` (below) |
| `04_training/` | Pilot CNN K (1,200 scenes), then CNN **F** and **K**, 3 seeds each, compared on the same F test inputs | `results/FK/comparison.json`, `mae_depth_absorption.png`, `models/` |
| `06_final_test/` | Independent validation: direct MC (not interpolated) at new seeds and at untrained absorptions, locked by hash before running | `results/report/KET_LUAN_V2_V6.md`, `v3…v6_*.csv/png` |
| `maintenance/` | New-machine check, AMD vs RTX hardware comparison, artifact audit, archived AMD RX 6600 run | `audit.json`, `migration_check/`, `archive_AMD_RX6600/` |
| `common/` | Frozen core `simulate.py` (MC), `readout.py` (profile), `train.py` (DepthCNN); `exp1_paths.py` | — |

Step 05 (error analysis) exists only in experiment 2.

## The data (`03_dataset/data/`)

| Path | Content |
|---|---|
| `mc/mua_<μa>_depth_<d>.npz` + `.json` | One MC run each: 3 absorption levels (0.15, 0.30, 1.0 mm⁻¹) × 12 depths. The `.json` holds the configuration, seeds and identity hash. |
| `mc/heldout_*`, `mc/repeat_*` | Checks: depths that are not lookup-table knots, and repeats with a new seed. Not used for training. |
| `lut_FK.npz` | Lookup table: profile (128 points) for 4 absorption levels × 12 depths. The standard level 0.508 comes from `02_forward_model/results/lut_K.npz`. |
| `dataset_FK/dataset.npz` | **6,000 scenes, interpolated from the lookup table, not simulated one by one.** `X_F`, `X_K` (6000 × 128 profiles), `y_mm` (depth), `mua_F` (absorption of the F scene), `shift_mm`, `split` (4,800 train / 600 validation / 600 test). |
| `dataset_FK/manifest.json` | Seed, sizes, provenance (LUT path and SHA-256), per-scene parameters. |
| `simulation_report.json`, `progress.json` | Gate results and progress of the MC runs. |

From MC to network input:
1. MC run → reflectance map.
2. Divide by N0, the reflectance without a vessel, pixel by pixel.
3. Average along the vessel, then bin to 128 points over ±16 mm (`readout.py`).

Each scene uses the same depth and shift in both branches. **F** sees the profile at its own absorption (`X_F`);
**K** always sees the standard absorption (`X_K`).

## Models

| File | Meaning |
|---|---|
| `04_training/results/FK/models/cnn_F_seed{42,43,44}.pt` | **F_locked**: trained on mixed absorption. This is the homogeneous model that experiment 2 compares against. |
| `04_training/results/FK/models/cnn_K_seed{42,43,44}.pt` | K: trained at standard absorption only. |
| `04_training/results/pilot_K/cnn_K_seed42.pt` | Pilot model on the small dataset, not used later. |

## Abbreviations

| Term | Meaning |
|---|---|
| N0 | Reflectance of the same medium **without** a vessel; profile = reflectance / N0 |
| F / K | Two training branches: **F** = absorption varies, **K** = absorption fixed at 0.508 mm⁻¹ |
| LUT | Lookup table of MC profiles; training scenes are interpolated from it |
| weak absorption | μa = 0.15 and 0.30 mm⁻¹, the main comparison group |
| V0–V6 | Steps of the independent validation: V0 lock manifest, V1 MC, V2 predict, V3–V6 report |
| gate | Pass/fail check that must succeed before the next stage runs |

## Conventions

- Every script imports `common/exp1_paths.py`. It puts all `*/code` folders on `sys.path`, and its `at()` resolves file paths
  stored inside configs and manifests.
- `run_plan.ps1` runs steps 01–04 in order (MC steps need the GPU environment).
- Unit tests: `python -m unittest discover -p "test_*.py"` inside `01_simulator_checks/code`, `04_training/code`
  and `06_final_test/code`.

To regenerate this experiment's data and models step by step, see [`../REPRODUCE.md`](../REPRODUCE.md).
