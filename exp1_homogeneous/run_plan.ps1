param(
    [ValidateRange(1, 4)][int]$FromStage = 1,
    [ValidateRange(1, 4)][int]$ToStage = 4,
    [string]$SimulationPython = (Join-Path $PSScriptRoot '..\.venv-fk\Scripts\python.exe'),   # GPU / pmcxcl environment
    [string]$AnalysisPython = 'python'
)

$ErrorActionPreference = 'Stop'
if ($FromStage -gt $ToStage) { throw 'FromStage must be <= ToStage' }
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Path 'maintenance' -Force | Out-Null

function Set-PlanStatus([int]$Stage, [string]$State, [string]$Detail) {
    @{stage=$Stage; state=$State; detail=$Detail; updated=(Get-Date -Format o)} |
        ConvertTo-Json | Set-Content -LiteralPath 'maintenance/plan_status.json' -Encoding utf8
}

function Invoke-Checked([string]$Runtime, [string[]]$Arguments, [string]$LogFile) {
    & $Runtime @Arguments *> $LogFile
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE). Inspect $LogFile" }
}

$stage = $FromStage
try {
    for ($stage = $FromStage; $stage -le $ToStage; $stage++) {
        Set-PlanStatus $stage 'running' 'Numerical gates must pass before the next stage'
        Write-Output "Starting stage $stage"
        switch ($stage) {
            1 {
                Invoke-Checked $SimulationPython @('-u', '01_simulator_checks/code/mc_convergence.py') 'stage1_run.log'
                Invoke-Checked $AnalysisPython @('01_simulator_checks/code/convergence_report.py') 'stage1_analysis.log'
                $report = Get-Content -LiteralPath '01_simulator_checks/results/report.json' -Raw | ConvertFrom-Json
                if (-not $report.gate_pass) { throw 'Stage 1 numerical checks did not pass' }
            }
            2 {
                Invoke-Checked $SimulationPython @('-u', '02_forward_model/code/depth_sweep.py') 'stage2_run.log'
            }
            3 {
                Invoke-Checked $AnalysisPython @('03_dataset/code/lut_dataset.py') 'stage3_data.log'
                Invoke-Checked $AnalysisPython @('-u', 'common/train.py', '--dataset', '04_training/results/pilot_K/data/dataset.npz', '--output', '04_training/results/pilot_K', '--branch', 'K', '--seed', '42', '--test-input', 'K') 'stage3_train.log'
                Invoke-Checked $AnalysisPython @('02_forward_model/code/plot_depth_profiles.py') 'stage3_plots.log'
            }
            4 {
                Invoke-Checked $SimulationPython @('-u', '03_dataset/code/mc_runs_FK.py') 'stage4_mc.log'
                Invoke-Checked $AnalysisPython @('-u', '04_training/code/train_FK.py') 'stage4_train.log'
            }
        }
        Set-PlanStatus $stage 'stage_complete' 'Inspect the corresponding report for scope and limitations'
    }
    Set-PlanStatus $ToStage 'requested_stages_complete' 'No measured camera or phantom validation'
} catch {
    Set-PlanStatus $stage 'failed' $_.Exception.Message
    throw
}
