[CmdletBinding()]
param(
    [string]$InputDir = "D:\download\TempFakeImages",
    [string]$OutputDir = "D:\download\TempFakeResults_v1_5fields",
    [string]$Checkpoint = "checkpoints\receipt_lrcnn_v1\best.pt",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [ValidateRange(1, 10000)]
    [int]$ShardCount = 60,
    [ValidateRange(0, 9999)]
    [int]$StartShard = 0,
    [int]$EndShard = -1,
    [ValidateRange(0, 1000000)]
    [int]$Limit = 0,
    [ValidateRange(0.0, 1.0)]
    [double]$ScoreThreshold = 0.50,
    [switch]$OcrOrientation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$InferScript = Join-Path $PSScriptRoot "infer.py"

if ($EndShard -lt 0) {
    $EndShard = $ShardCount - 1
}
if ($StartShard -ge $ShardCount) {
    throw "StartShard must be smaller than ShardCount."
}
if ($EndShard -lt $StartShard -or $EndShard -ge $ShardCount) {
    throw "EndShard must be between StartShard and ShardCount - 1."
}

Push-Location $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python executable not found: $Python"
    }
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Checkpoint not found: $Checkpoint"
    }
    if (-not (Test-Path -LiteralPath $InputDir -PathType Container)) {
        throw "Input directory not found: $InputDir"
    }

    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $LogDir = Join-Path $OutputDir "_logs"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    Write-Host "Input:      $InputDir"
    Write-Host "Output:     $OutputDir"
    Write-Host "Checkpoint: $Checkpoint"
    Write-Host "Shards:     $StartShard through $EndShard of $ShardCount"
    if ($Limit -gt 0) {
        Write-Host "Limit:      fixed first $Limit images in each selected shard"
    }
    Write-Host "Rule:       exactly one box for each of the five fields"

    for ($ShardIndex = $StartShard; $ShardIndex -le $EndShard; $ShardIndex++) {
        $ShardDisplay = $ShardIndex + 1
        $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $LogPath = Join-Path $LogDir ("shard_{0:D3}_of_{1:D3}_{2}.log" -f $ShardIndex, $ShardCount, $Timestamp)
        $Arguments = @(
            $InferScript,
            "--checkpoint", $Checkpoint,
            "--input", $InputDir,
            "--output", $OutputDir,
            "--device", "cuda",
            "--ocr", "paddle",
            "--score-threshold", $ScoreThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--require-complete",
            "--continue-on-error",
            "--skip-existing",
            "--shard-count", $ShardCount,
            "--shard-index", $ShardIndex
        )
        if ($OcrOrientation) {
            $Arguments += "--ocr-orientation"
        }
        if ($Limit -gt 0) {
            $Arguments += @("--limit", $Limit)
        }

        Write-Host ""
        Write-Host ("[{0}] Starting shard {1}/{2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $ShardDisplay, $ShardCount)
        $Timer = [System.Diagnostics.Stopwatch]::StartNew()
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        $ExitCode = $LASTEXITCODE
        $Timer.Stop()

        if ($ExitCode -ne 0) {
            throw "Shard $ShardIndex stopped with exit code $ExitCode. Fix the cause and rerun the same shard; completed images will be skipped. Log: $LogPath"
        }

        $Suffix = ".shard-{0:D3}-of-{1:D3}" -f $ShardIndex, $ShardCount
        $ManifestPath = Join-Path $OutputDir ("inference_manifest{0}.jsonl" -f $Suffix)
        $ErrorPath = Join-Path $OutputDir ("inference_errors{0}.jsonl" -f $Suffix)
        $SuccessCount = 0
        $ErrorCount = 0
        if (Test-Path -LiteralPath $ManifestPath) {
            $SuccessCount = (Get-Content -LiteralPath $ManifestPath | Measure-Object -Line).Lines
        }
        if (Test-Path -LiteralPath $ErrorPath) {
            $ErrorCount = (Get-Content -LiteralPath $ErrorPath | Measure-Object -Line).Lines
        }
        Write-Host ("Completed shard {0}/{1}: normal={2}, errors_or_incomplete={3}, elapsed={4}" -f $ShardDisplay, $ShardCount, $SuccessCount, $ErrorCount, $Timer.Elapsed)
        Write-Host "Log: $LogPath"
    }

    $ManifestPattern = "inference_manifest.shard-*-of-{0:D3}.jsonl" -f $ShardCount
    $ErrorPattern = "inference_errors.shard-*-of-{0:D3}.jsonl" -f $ShardCount
    $TotalNormal = 0
    $TotalErrors = 0
    Get-ChildItem -LiteralPath $OutputDir -Filter $ManifestPattern -File | ForEach-Object {
        $TotalNormal += (Get-Content -LiteralPath $_.FullName | Measure-Object -Line).Lines
    }
    Get-ChildItem -LiteralPath $OutputDir -Filter $ErrorPattern -File | ForEach-Object {
        $TotalErrors += (Get-Content -LiteralPath $_.FullName | Measure-Object -Line).Lines
    }
    $OutputSize = Get-ChildItem -LiteralPath $OutputDir -Recurse -File | Measure-Object -Property Length -Sum
    $OutputGiB = if ($null -eq $OutputSize.Sum) { 0.0 } else { [double]$OutputSize.Sum / 1GB }

    Write-Host ""
    Write-Host "Current summary for completed shard files: normal=$TotalNormal, errors_or_incomplete=$TotalErrors"
    Write-Host ("Current output size: {0:N2} GiB" -f $OutputGiB)
    Write-Host "Rerunning this script with the same output directory is safe because --skip-existing is always enabled."
}
finally {
    Pop-Location
}
