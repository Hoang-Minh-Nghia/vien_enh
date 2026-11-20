"""Wait for the V1 matrix, resume it if it stopped early, then run V2 and V3-V6.

Runs apart from any interactive session and holds a system-awake request while
it lives. An MC run that produced an .invalid.npz is never retried, and no CNN
prediction starts without a complete quality_matrix.json. Usage:
    python 06_final_test/code/run_after_mc.py <PID of the running v1_run_mc.py>
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
import exp1_paths as L  # noqa: E402  (step folders on sys.path; L.at() resolves stored paths)
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = L.EXP1
OUTPUT = L.at("06_final_test/results")
STATUS = OUTPUT / "pipeline_status.json"
QUALITY = OUTPUT / "quality_matrix.json"
MC_PYTHON = ROOT.parent / ".venv-fk/Scripts/python.exe"
MAX_RESUMES = 3
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def status(state, **details):
    record = {"state": state, "updated_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **details}
    STATUS.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def alive(pid):
    listed = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in listed


def step(command, log):
    with open(L.at(log), "w", encoding="utf-8") as handle:
        return subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def finished():
    if not QUALITY.exists():
        return False
    try:
        return json.loads(QUALITY.read_text(encoding="utf-8"))["scope"] == "matrix"
    except (ValueError, KeyError):
        return False


def main():
    pid = int(sys.argv[1])
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    status("waiting_for_v1_matrix", pid=pid)
    while alive(pid):
        time.sleep(30)

    resumes = 0
    while not finished():
        if list(OUTPUT.rglob("*.invalid.npz")):
            status("stopped_invalid_mc", detail="An .invalid.npz exists; inspect it. No CNN prediction was opened.")
            return 1
        if resumes == MAX_RESUMES:
            status("stopped_v1_not_finished", resumes=resumes, detail="No CNN prediction was opened.")
            return 1
        resumes += 1
        status("resuming_v1_matrix", resume=resumes)
        step([str(MC_PYTHON), "-u", "06_final_test/code/run_independent_mc.py", "--max-tier", "2"], f"v1_matrix_run_RTX3060_resume{resumes}.log")

    status("v1_complete_running_v2", resumes=resumes)
    if step([sys.executable, "-u", "06_final_test/code/predict_independent.py"], "v2_predict.log") != 0:
        status("stopped_v2_failed", log="v2_predict.log")
        return 1
    status("v2_complete_running_v3_v6")
    if step([sys.executable, "-u", "06_final_test/code/report_independent.py"], "v3_v6_report.log") != 0:
        status("stopped_report_failed", log="v3_v6_report.log")
        return 1
    status("complete", report="results/validation_independent/report/KET_LUAN_V2_V6.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
