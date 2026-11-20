"""Locations used by every experiment-1 script. Importing it puts all step code folders on sys.path.

Configs and manifests store file paths such as "results/stage4/models/cnn_F_seed42.pt" or "code/simulate.py";
`at()` maps such a stored path to where the file is in this repository.
"""
import sys
from pathlib import Path

EXP1 = Path(__file__).resolve().parents[1]          # exp1_homogeneous/
REPO = EXP1.parent
STEPS = sorted(p for p in EXP1.iterdir() if p.is_dir() and p.name[:2].isdigit())
for _p in [EXP1 / "common", *[s / "code" for s in STEPS], EXP1 / "maintenance" / "code"]:
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

LEGACY = [
    ("results/validation_independent_archive_AMD_RX6600", "maintenance/archive_AMD_RX6600"),
    ("results/stage4/simulation_report.json", "03_dataset/data/simulation_report.json"),
    ("code/check_validation_pipeline.py", "06_final_test/code/check_pipeline.py"),
    ("code/test_validation_manifest.py", "06_final_test/code/test_validation_manifest.py"),
    ("results/validation_independent", "06_final_test/results"),
    ("results/stage4/progress.json", "03_dataset/data/progress.json"),
    ("code/plot_target_domain.py", "04_training/code/plot_mae_map.py"),
    ("results/stage4/lut_FK.npz", "03_dataset/data/lut_FK.npz"),
    ("code/check_new_machine.py", "maintenance/code/check_new_machine.py"),
    ("results/environment.json", "maintenance/environment.json"),
    ("results/plan_status.json", "maintenance/plan_status.json"),
    ("code/plot_development.py", "02_forward_model/code/plot_depth_profiles.py"),
    ("code/v0_lock_manifest.py", "06_final_test/code/lock_manifest.py"),
    ("code/compare_hardware.py", "maintenance/code/compare_hardware.py"),
    ("results/migration_check", "maintenance/migration_check"),
    ("code/stage4_simulate.py", "03_dataset/code/mc_runs_FK.py"),
    ("code/report_stage1.py", "01_simulator_checks/code/convergence_report.py"),
    ("code/test_learning.py", "04_training/code/test_learning.py"),
    ("code/audit_results.py", "maintenance/code/audit_results.py"),
    ("code/test_physics.py", "01_simulator_checks/code/test_physics.py"),
    ("code/stage4_train.py", "04_training/code/train_FK.py"),
    ("code/v3_v6_report.py", "06_final_test/code/report_independent.py"),
    ("code/run_after_v1.py", "06_final_test/code/run_after_mc.py"),
    ("results/stage4/data", "03_dataset/data/dataset_FK"),
    ("results/diagnostics", "maintenance/diagnostics"),
    ("results/audit.json", "maintenance/audit.json"),
    ("code/v2_predict.py", "06_final_test/code/predict_independent.py"),
    ("results/stage4/mc", "03_dataset/data/mc"),
    ("code/v1_run_mc.py", "06_final_test/code/run_independent_mc.py"),
    ("code/simulate.py", "common/simulate.py"),
    ("code/readout.py", "common/readout.py"),
    ("code/dataset.py", "03_dataset/code/lut_dataset.py"),
    ("results/stage4", "04_training/results/FK"),
    ("results/stage3", "04_training/results/pilot_K"),
    ("results/stage2", "02_forward_model/results"),
    ("results/stage1", "01_simulator_checks/results"),
    ("code/stage1.py", "01_simulator_checks/code/mc_convergence.py"),
    ("code/stage2.py", "02_forward_model/code/depth_sweep.py"),
    ("code/train.py", "common/train.py"),
]


def at(rel):
    """Absolute path for a path relative to exp1_homogeneous, translating the original layout."""
    rel = str(rel).replace("\\", "/")
    for old, new in LEGACY:
        if rel == old or rel.startswith(old + "/"):
            return EXP1 / (new + rel[len(old):])
    return EXP1 / rel
