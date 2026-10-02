# Windows NPU PC: same actions as .workshop/npu.yaml, run with uv directly (no workshop).
# Keep this file ASCII-only: Windows PowerShell 5.1 reads BOM-less UTF-8 as ANSI.
# When you change the actions here, change .workshop/npu.yaml too (and vice versa).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\npu.ps1 <action> [args...]
#   actions: setup | check | test | decide | bench
#   e.g.     powershell -ExecutionPolicy Bypass -File scripts\npu.ps1 bench --device NPU

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("setup", "check", "test", "decide", "bench")]
    [string]$Action,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
# Same as the workshop environment: keep the Hugging Face cache inside the project
$env:HF_HOME = Join-Path (Get-Location) ".cache\huggingface"
$env:PYTHONUTF8 = "1"

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Install it with: winget install astral-sh.uv"
}

$cli = @("run", "python", "-m", "local_decision_model")

switch ($Action) {
    "setup" {
        Invoke-Checked uv @("sync", "--extra", "openvino")
    }
    "check" {
        Invoke-Checked uv ($cli + @("devices"))
        # Replacement for the /dev/accel check on Linux: ask OpenVINO for its device list
        $devices = & uv run python -c "import openvino as ov; print(' '.join(ov.Core().available_devices))"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        if ($devices -notmatch "\bNPU\b") {
            Write-Host "NPU is not in the OpenVINO device list ($devices)."
            Write-Host "Check Device Manager for 'Intel(R) AI Boost' and update the Intel NPU driver."
            exit 1
        }
        Write-Host "NPU OK ($devices)"
    }
    "test" {
        Invoke-Checked uv (@("run", "pytest") + $Rest)
    }
    "decide" {
        Invoke-Checked uv ($cli + @("decide", "--backend", "openvino", "--device", "NPU") + $Rest)
    }
    "bench" {
        Invoke-Checked uv ($cli + @("bench", "--backend", "openvino", "--device", "NPU") + $Rest)
    }
}
