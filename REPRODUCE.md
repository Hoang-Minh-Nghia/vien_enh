# Regenerating the simulated data

The repository already contains all simulated data and trained models, so evaluations run on CPU without
regenerating anything. This guide shows how to **regenerate the Monte-Carlo (MC) data from scratch** on
your own machine, and how to check the result against the stored data.

> Tiếng Việt: [REPRODUCE_vi.md](REPRODUCE_vi.md)

## How it works

- Every generation script **skips output files that already exist**. You can interrupt a run and start it
  again, and it continues where it stopped. Running it in this repository as cloned does nothing, because
  all data are present.
- To regenerate, use a **separate clone**, delete the data folders you want to rebuild, run the steps
  below, then compare with the original using `tools/compare_regenerated.py`.
- Each configuration file is written from fixed seeds before any simulation, so a new run simulates exactly
  the same skins, veins and photon counts.
- Run **one MC process at a time**: all scripts use the first GPU (`gpuid = 1`).

## 1. Requirements

| | Used for this repository |
|---|---|
| OS | Windows 11 (the commands below are PowerShell) |
| GPU | any OpenCL GPU. Measured times below: experiment 1 on AMD Radeon RX 6600, experiment 2 on NVIDIA RTX 3060 |
| Python | 3.12 |
| Disk | about 1 GB free |

```powershell
git clone https://github.com/Hoang-Minh-Nghia/vien_enh.git vien_enh_repro
cd vien_enh_repro
python -m venv .venv-fk;       .\.venv-fk\Scripts\pip install -r requirements\mc.txt              # MC (pmcxcl, GPU)
python -m venv .venv-analysis; .\.venv-analysis\Scripts\pip install -r requirements\analysis.txt  # analysis, training (CPU)
.\.venv-fk\Scripts\python -c "import pmcxcl; print(pmcxcl.gpuinfo())"                             # the GPU must be listed
```

All commands below start from the root of `vien_enh_repro`.

## 2. Experiment 1 — homogeneous skin

**Delete the outputs to rebuild.** `common/` and `configs/` are frozen inputs and must stay.

```powershell
Remove-Item -Recurse -Force exp1_homogeneous\01_simulator_checks\results, exp1_homogeneous\02_forward_model\results, `
  exp1_homogeneous\03_dataset\data, exp1_homogeneous\04_training\results
```

**Run all four steps** (stops as soon as a numerical check fails):

```powershell
cd exp1_homogeneous
.\run_plan.ps1 -AnalysisPython ..\.venv-analysis\Scripts\python.exe     # MC uses ..\.venv-fk by default
cd ..
```

What `run_plan.ps1` does, step by step. To run one part only, use `-FromStage`/`-ToStage`, e.g.
`.\run_plan.ps1 -FromStage 2 -ToStage 2`.

| Stage | Command (from `exp1_homogeneous/`) | Environment | Time measured | Output |
|---|---|---|---|---|
| 1 | `01_simulator_checks/code/mc_convergence.py`, then `convergence_report.py` | GPU, then CPU | 26 min | `01_simulator_checks/results/` (10 MC runs, `report.json` with `gate_pass`) |
| 2 | `02_forward_model/code/depth_sweep.py` | GPU | 12 min | `02_forward_model/results/` (12 depths, `lut_K.npz`, `report.json`) |
| 3 | `03_dataset/code/lut_dataset.py`; `common/train.py --dataset 04_training/results/pilot_K/data/dataset.npz --output 04_training/results/pilot_K --branch K --seed 42 --test-input K`; `02_forward_model/code/plot_depth_profiles.py` | CPU | a few min | pilot dataset and pilot CNN K in `04_training/results/pilot_K/` |
| 4 | `03_dataset/code/mc_runs_FK.py`, then `04_training/code/train_FK.py` | GPU, then CPU | 34 min MC + ~3 min training | `03_dataset/data/` (41 MC runs: 36 table points + 5 checks, `lut_FK.npz`, `dataset_FK/`); `04_training/results/FK/` (6 models, `comparison.json`) |

`run_plan.ps1` writes its progress to `maintenance/plan_status.json`, and each command's output to a
`stage*.log` file next to the script.

**Independent validation (step 06, optional).** Direct MC at new seeds and untrained absorptions (162 runs,
about 3 h measured on an RTX 3060):

```powershell
cd exp1_homogeneous
..\.venv-fk\Scripts\python 06_final_test\code\run_independent_mc.py --max-tier 2
..\.venv-analysis\Scripts\python 06_final_test\code\predict_independent.py
..\.venv-analysis\Scripts\python 06_final_test\code\report_independent.py
cd ..
```

The validation is locked to the **stored** F/K checkpoints by SHA-256 (see
`06_final_test/plan_independent_validation.md`). Run it before regenerating `04_training`, or in a clone where
`04_training/results/FK/models` is unchanged; otherwise the scripts refuse to run.

## 3. Experiment 2 — layered skin

**Delete the outputs to rebuild.** This removes the manifests too, so they are rebuilt from their seeds.

```powershell
Remove-Item -Recurse -Force exp2_layered\03_dataset\data
```

**Generate the datasets** (from `exp2_layered/03_dataset/code/`):

```powershell
cd exp2_layered\03_dataset\code
$MC = "..\..\..\.venv-fk\Scripts\python.exe"; $PY = "..\..\..\.venv-analysis\Scripts\python.exe"

& $MC dataset_generate.py manifest              # manifest.json: 300 train, 30 validation, 40 locked_test skins
& $MC dataset_generate.py run                   # simulates train, validation, then locked_test   (~9 h)
& $MC dataset_extend_ext.py manifest            # manifest_ext.json
& $MC dataset_extend_ext.py run validation_ext  # 90 skins                                        (~3.7 h)
& $PY dataset_check.py                          # sanity checks on train/validation (never opens locked_test)
& $PY dataset_summary.py                        # writes data/check_train_validation.json
cd ..\..\..
```

| Command | Output (in `exp2_layered/03_dataset/data/`) | Time measured (RTX 3060) |
|---|---|---|
| `dataset_generate.py run` | `train/` 300, `validation/` 30, `locked_test/` 40 files | 9.0 h |
| `dataset_extend_ext.py run validation_ext` | `validation_ext/` 90 files | 3.7 h |
| optional: `dataset_high_snr_a.py manifest`, then `dataset_high_snr_a.py run` | `train_hisnr/` 120 files (the first 120 train skins at 8× photons; used only by the noise-floor test) | ~18 h (estimate) |

- **Interrupted run:** start the same command again. Finished skins are skipped, and a half-written file
  (`*.partial.npz`) is overwritten.
- **MC error:** the script writes `<id>.invalid.json` and stops. Keep the file and find the cause before you
  continue; do not delete it to force a rerun.
- **locked_test:** a regenerated `locked_test/` is still the final test set. Do not use it to choose or tune
  models.
- `dataset_extend_ext.py run` without a split name would continue into `train_ext` (900 skins). That set is
  not used for any result.

**Optional GPU steps.** None of these are needed to train the networks.

| Step | Command (from `exp2_layered/`) | Time |
|---|---|---|
| 01 simulator checks | `01_simulator_checks/code/sim_capabilities.py`, `sim_capabilities_2.py` | minutes |
| 02 forward model | `02_forward_model/code/nuisance_runs.py`, `layered_runs.py` (then the matching `*_analysis.py` on CPU) | hours |
| 05 14-parameter information bound | `05_error_analysis/code/crlb_runs.py` (then `crlb_analysis.py` on CPU) | ~18 h |

## 4. From data to models (CPU)

From `exp2_layered/`:

```powershell
$PY = "..\.venv-analysis\Scripts\python.exe"
foreach ($s in 42, 43, 44) { & $PY common\cnn_variants.py hetero $s }    # final model, ~13 min per seed
foreach ($s in 42, 43, 44) { & $PY common\cnn_variants.py wide $s }      # runner-up
& $PY 04_training\code\compare_validation_ext.py                          # scores every model on validation_ext
```

The models are written to `04_training/results/variants/`. Other variants are trained the same way:
`base`, `sched`, `multi`, `--nskin=N` for the learning curve, `--hisnr` (needs `train_hisnr`).

The locked test was scored once with the stored models
(`06_final_test/results/locked_test/OPENED.json`), and `score_locked_test.py` refuses to score it again.

## 5. Check the regenerated data

```powershell
.\.venv-analysis\Scripts\python tools\compare_regenerated.py <path to the original repository> .
```

Every MC file that exists in both copies is compared:
- run configuration and seed, or skin configuration: must be identical;
- mean reflectance or N0: must agree within 0.5 %;
- profile (1 mm bins over ±16 mm): the difference must stay below 3× the MC noise expected for two independent runs;
- manifests: must be identical;
- lookup tables and the F/K dataset: must agree within 2 %.

On the same GPU the files agree almost exactly. On a different GPU the photon streams differ, so they agree
only within MC noise.

Example (a short test: 2 runs of experiment 1 regenerated on an RTX 3060, original made on an AMD RX 6600;
1 skin of experiment 2):

```
OK   exp1_homogeneous/01_simulator_checks/results/N0.npz: same config/seed True, mean diff 9.28e-06, profile RMS diff 1.03e-03 (limit 3.38e-03)
OK   exp1_homogeneous/01_simulator_checks/results/channel_2mm.npz: same config/seed True, mean diff 6.57e-05, profile RMS diff 1.51e-03 (limit 4.89e-03)
OK   exp2_layered/03_dataset/data/train/train_000.npz: same config True, N0 diff 1.32e-05, worst vein RMS diff 1.33e-04 (limit 1.10e-02)
```

The exit code is 0 only when every compared file passes.
