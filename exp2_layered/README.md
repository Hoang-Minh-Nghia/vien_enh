# exp2_layered — layered skin

Skin has four layers: epidermis with melanin, dermis, subcutaneous fat containing the vein, and muscle.
**14 parameters vary per skin:**
- melanin;
- thickness, μs′ and μa of each layer;
- vein depth, diameter, blood μa and μs;
- lateral shift.

Depth = **surface to the upper edge** of the vein. Wavelength 850 nm, diffuse reflectance (pmcxcl).
Parameter values and sources: `00_parameters_850nm.md`. Sampling ranges: `03_dataset/report_dataset_design.md`.

Reports are in Vietnamese.

## Steps (same numbers and roles as `../exp1_homogeneous`)

| Folder | What it does | Report |
|---|---|---|
| `01_simulator_checks/` | pmcxcl capabilities (sources, detectors, polarization), CPU MC reference vs GPU MC | — |
| `02_forward_model/` | Information limit (preliminary CRLB), nuisance parameters; layered-skin forward model and the bias of a homogeneous fit | `report_crlb_nuisance.md`, `report_layered_model.md` |
| `03_dataset/` | MC datasets in `data/` (below) | `report_dataset_design.md` |
| `04_training/` | Layered CNN, variants, comparison on validation_ext, choice of the **final model** | `report_validation.md`, `plan_improvements.md` |
| `05_error_analysis/` | Why about 0.2 mm remains: 14-parameter bound, fixed bias per skin, retraining | `report_crlb_14_params.md`, `report_fixed_bias.md`, `report_retraining.md` |
| `06_final_test/` | Locked test, scored once: **MAE 0.206 mm** | `report_locked_test.md` |
| `common/` | `cnn_variants.py` (networks, binning, augmentation, training CLI), `b3_physics_fit.py` (homogeneous physics fit B3), `exp2_paths.py` | — |

## The data (`03_dataset/data/`)

| Folder | Skins | Veins per skin | Photons | Defined in | Use |
|---|---|---|---|---|---|
| `train/` | 5 | 12 M | `manifest.json` | training |
| `validation/` | 5 | 12 M | `manifest.json` | early stopping, model selection |
| `validation_ext/` | 5 | 24 M | `manifest_ext.json` | comparing models and variants (steps 04, 05) |
| `locked_test/` | 3 | 60 M | `manifest.json` | final test, **opened once**, never used for any choice |
| `train_hisnr/` | 5 | 96 M | `manifest_hisnr.json` | the same skins as `train[:120]`, re-simulated with 8× the photons (noise-floor test) |
| `train_hisnr2/` | 5 | 96 M | `manifest_hisnr2.json` | **incomplete run, not used anywhere** |

`manifest_ext.json` also defines `train_ext` (900 skins); it is not simulated, so there is no data for it.

**One file = one skin, not one sample.** Each `<split>_NNN.npz` contains:

| Key | Shape | Meaning |
|---|---|---|
| `profiles_y_mean` | (veins, 600) | Reflectance across each vein, averaged along it; **not yet normalised** |
| `x_mm` | (600,) | Position across the vein, −29.95…+29.95 mm, step 0.1 mm |
| `n0_scalar` | () | Mean reflectance of the same skin without vessels (N0) |
| `n0_cv`, `n0_y_mean` | | Spread and profile of N0, for checks |
| `config_json` | str | All 14 parameters of the skin and its veins (identical to the manifest entry) |

From file to network input (`common/cnn_variants.py`):
1. `profile = profiles_y_mean / n0_scalar`.
2. Random lateral shift U(−2, 2) mm.
3. Bin to 128 × 0.25 mm over ±16 mm.
4. Add white noise σ ∈ [0.001, 0.01], log-uniform (camera noise).

Steps 2–4 happen at training and evaluation time, so the data on disk never contains them.

## Models (`04_training/results/`)

| Path | Meaning |
|---|---|
| `variants/hetero_seed{42,43,44}.pt` | **Final model**: heteroscedastic WideCNN, predicts depth and its uncertainty. Ensemble of 3 seeds. Locked by SHA-256 in `06_final_test/code/score_locked_test.py`. |
| `variants/wide_*.pt` | Runner-up (same accuracy, no uncertainty) |
| `variants/base_*`, `sched_*`, `multi_*` | Other variants: base = experiment-1 DepthCNN; sched = same with LR schedule; multi = WideCNN with auxiliary targets (diameter, blood μa, melanin) |
| `variants/*_hisnr_*`, `*_n50…n200_*`, `*_n120_*` | Noise-floor and learning-curve experiments |
| `layered_cnn/models/F_layered_seed*.pt` | First layered CNN, same architecture as experiment 1 |
| `variants/compare_validation_ext.json` | All variants scored on validation_ext |
| `*.err` files | Training logs (stderr) |

## Abbreviations

| Term | Meaning |
|---|---|
| N0 | Reflectance of the skin without vessels; profile = reflectance / N0 |
| hetero | Network with two outputs (depth, log σ), trained with Gaussian negative log-likelihood |
| wide | WideCNN: dilated convolutions, receptive field about 30 mm, BatchNorm |
| F_locked | Homogeneous-skin CNN F from experiment 1, used as comparator |
| F_layered | Experiment-1 architecture retrained on layered skin |
| B3 | Physics fit with a homogeneous forward model (lookup table of experiment 1) |
| CRLB | Cramér–Rao lower bound: best possible precision from the information in the profile |
| ext | "extended": a second, larger set (validation_ext; train_ext postponed) |
| hisnr | High signal-to-noise: the same skins simulated with 8× more photons |
| n50…n200, n120 | Model trained on only the first N training skins (learning curve) |
| σ_mc | Monte-Carlo noise per bin already present in the simulated profile |

## Conventions

- Every script imports `common/exp2_paths.py`. It puts all `*/code` folders, `common/` and the frozen experiment-1
  core on `sys.path`. Its `exp1_at()` locates experiment-1 files. Scripts can be run from any folder.
- Module names are unique across folders.
- Shell helpers (`04_training/code/run_paired.sh`, `05_error_analysis/code/run_retrain.sh`) take the interpreter
  from `$PYTHON`.
- Scripts in `01_simulator_checks`, `02_forward_model`, `03_dataset` and `05_error_analysis/code/crlb_runs.py` need
  the GPU environment (`requirements/mc.txt`). Everything else runs on CPU.

Real-hand application: `../real_hand`. Phantom plan: `../phantom`.

To regenerate this experiment's data and models step by step, see [`../REPRODUCE.md`](../REPRODUCE.md).
