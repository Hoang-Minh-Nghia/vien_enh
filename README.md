# nir-vein-depth

Estimating the depth of superficial veins from a single **850 nm diffuse-reflectance** image, using
GPU Monte Carlo light transport (pmcxcl) and convolutional networks trained on simulated vessel profiles.

> Tiếng Việt: [README_vi.md](README_vi.md)

## Main result

On a locked test set of 40 unseen layered-skin configurations (120 veins), opened once under a pre-registered plan
(`exp2_layered/06_final_test/report_locked_test.md`), the final model is the heteroscedastic WideCNN, an ensemble of 3 seeds. It reaches
**MAE 0.206 mm (95 % CI 0.179–0.235 mm, bootstrap over skins)**. This is the mean over camera-noise levels
σ = 0.005 / 0.007 / 0.010 per bin. Its ±1.96σ uncertainty covers 92–96 % of the errors.

| Model (same locked test) | MAE (mm) |
|---|---|
| **hetero WideCNN, final** | **0.206** |
| WideCNN | 0.207 |
| DepthCNN trained on layered skin | 0.224 |
| DepthCNN trained on homogeneous skin (experiment 1) | 1.232 |
| Homogeneous physics fit (B3) | 1.270 |
| Constant = mean training depth | 0.859 |

These numbers come from simulation only. The phantom validation is planned (`phantom/`) but has not been carried
out yet. No ground-truth depth exists for the real-hand images.

## Repository layout

Both experiments use **the same step folders**. The same number always means the same role:

| Step folder | Role | `exp1_homogeneous/` (homogeneous skin) | `exp2_layered/` (layered skin) |
|---|---|---|---|
| `common/` | shared code + paths | frozen MC core `simulate.py`, `readout.py`, `train.py`; `exp1_paths.py` | `cnn_variants.py`, `b3_physics_fit.py`; `exp2_paths.py` |
| `configs/` / `00_*.md` | fixed inputs | `configs/*.json` (frozen) | `00_parameters_850nm.md` |
| `01_simulator_checks/` | is the Monte Carlo converged and correct? | photon count, grid, domain, time window, repeat | pmcxcl capabilities, CPU vs GPU MC |
| `02_forward_model/` | how does the profile depend on depth (and on everything else)? | depth sweep at standard absorption, LUT_K | information limit (CRLB), nuisance parameters, layered-skin model |
| `03_dataset/` | the simulated data (`data/`) | 48 MC runs → lookup table → 6,000 interpolated scenes per branch (F, K) | MC of 300 train / 30 validation / 90 validation_ext / 40 locked_test skins |
| `04_training/` | train and compare networks | pilot CNN K; CNN F vs K, 3 seeds | layered CNN, variants, **final model** (hetero) |
| `05_error_analysis/` | why the remaining error? | — | 14-parameter bound, fixed bias per skin, retraining |
| `06_final_test/` | one final check on data never used before | independent direct-MC validation (V0–V6) | locked test, opened once |
| `maintenance/` | not science: machine migration, hardware comparison, audit | archive of the AMD RX 6600 run | — |

Each step folder has `code/`, its outputs (`results/` or, for the dataset, `data/`) and its report(s).
Every experiment folder has a `README.md` that explains its files and abbreviations.

| Other folder | Content |
|---|---|
| `real_hand/` | The same pipeline applied to real reflectance photos of the hand (hand ROI, perpendicular profiles, background division, depth maps, 3D map). **Code only. No images are included.** |
| `phantom/` | Plan and design numbers for a physical layered agar–Intralipid–ink phantom (error budget, number of blocks). Not yet performed. |
| `colab/` | Notebook that runs the checks and evaluations, one task per cell. |
| `tools/` | `compare_regenerated.py`: compares regenerated MC data with the stored data (see `REPRODUCE.md`). |
| `docs/` | SHA-256 list of frozen files. |
| `requirements/` | Pinned environments: `mc.txt` (GPU Monte Carlo, needs OpenCL) and `analysis.txt` (NumPy / SciPy / PyTorch CPU). |


### What is included

- All code.
- Trained checkpoints (`*.pt`, about 13 MB in total).
- Simulated MC datasets and evaluation outputs (`results/`), so that every evaluation can be rerun on CPU without a GPU.
- Not included: run logs, and any real-hand image or anything derived from one.

### Frozen files and locked test

- `exp1_homogeneous/configs/*.json` and `exp1_homogeneous/common/{simulate,readout,train}.py` are frozen. They are
  byte-identical to the versions used in the experiments; see `docs/FROZEN_FILES.sha256`. File paths stored inside
  configs and manifests are resolved by `exp1_homogeneous/common/exp1_paths.py` (`at()`).
- `exp2_layered/06_final_test/results/locked_test/OPENED.json` records that the locked test has been opened.
  `score_locked_test.py` refuses to run again. Any model change must be scored on a new locked set.

## Setup

Python 3.12.

```bash
python -m venv .venv-analysis && .venv-analysis/Scripts/pip install -r requirements/analysis.txt   # evaluation, training (CPU)
python -m venv .venv-fk       && .venv-fk/Scripts/pip install -r requirements/mc.txt               # Monte Carlo (GPU, OpenCL)
```

Run only one GPU (MC) process at a time.

## Reproducing

```bash
# experiment 1: pipeline self-check (F/K checkpoints reproduce saved predictions) and unit tests
(cd exp1_homogeneous/06_final_test/code && python check_pipeline.py)
(cd exp1_homogeneous/01_simulator_checks/code && python -m unittest discover -p "test_*.py")

# experiment 2: model comparison on validation_ext (CPU, ~20 s)
(cd exp2_layered/04_training/code && python compare_validation_ext.py)

# phantom design numbers (error budget, number of blocks)
(cd phantom/code && python design_budget.py)
```

To regenerate the Monte-Carlo data from scratch (GPU, hours) and check it against the stored data, see
**[REPRODUCE.md](REPRODUCE.md)**.

### Real-hand images

The scripts in `real_hand/code` expect a folder with `images/*.png` and `masks/*.png` (same file names), given by
the environment variable `VEIN_REAL_DATA`. Outputs go to `real_hand/results` and `real_hand/figs`, which are
git-ignored. Images of people may only be used with their consent and the appropriate ethics approval.

## License

Not chosen yet.
