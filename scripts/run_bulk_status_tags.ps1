[CmdletBinding()]
param(
    [string]$SourceDir = "D:\download\TempFakeImages",
    [string]$InputDir = "D:\download\TempFakeResults_v1_pilot100_timefix",
    [string]$OutputDir = "D:\download\TempFakeStatusTags_v1",
    [string]$Checkpoint = "checkpoints\status_style_v1\best.pt",
    [string]$Python = ".\.venv\Scripts\python.exe",
    [ValidateRange(1, 10000)]
    [int]$ShardCount = 60,
    [ValidateRange(1, 10000)]
    [int]$V1ShardCount = 60,
    [ValidateRange(0, 9999)]
    [int]$StartShard = 0,
    [int]$EndShard = -1,
    [ValidateRange(0, 1000000)]
    [int]$Limit = 0,
    [ValidateRange(0.0, 1.0)]
    [double]$ConfidenceThreshold = 0.80,
    [ValidateRange(0.0, 1.0)]
    [double]$AbsentConfidenceThreshold = 0.95,
    [switch]$AllowIncompleteV1
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnrichScript = Join-Path $PSScriptRoot "enrich_status_tags.py"

function Get-NormalizedFullPath([string]$Path) {
    $Separators = [char[]]@(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $ProviderPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    return [System.IO.Path]::GetFullPath($ProviderPath).TrimEnd($Separators)
}

function Test-IsSameOrChildPath([string]$Candidate, [string]$Root) {
    return $Candidate.Equals($Root, [System.StringComparison]::OrdinalIgnoreCase) -or
        $Candidate.StartsWith(
            $Root + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        )
}

function Test-PathsOverlap([string]$First, [string]$Second) {
    return (Test-IsSameOrChildPath $First $Second) -or (Test-IsSameOrChildPath $Second $First)
}

if ($EndShard -lt 0) {
    $EndShard = $ShardCount - 1
}
if ($StartShard -ge $ShardCount) {
    throw "StartShard must be smaller than ShardCount."
}
if ($EndShard -lt $StartShard -or $EndShard -ge $ShardCount) {
    throw "EndShard must be between StartShard and ShardCount - 1."
}
if ($AbsentConfidenceThreshold -lt $ConfidenceThreshold) {
    throw "AbsentConfidenceThreshold must be at least ConfidenceThreshold."
}

Push-Location $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python executable not found: $Python"
    }
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Status-style checkpoint not found: $Checkpoint"
    }
    if (-not (Test-Path -LiteralPath $InputDir -PathType Container)) {
        throw "v1 result directory not found: $InputDir"
    }
    if (-not (Test-Path -LiteralPath $SourceDir -PathType Container)) {
        throw "Source image directory not found: $SourceDir"
    }
    $SourceFullPath = Get-NormalizedFullPath $SourceDir
    $InputFullPath = Get-NormalizedFullPath $InputDir
    $OutputFullPath = Get-NormalizedFullPath $OutputDir
    if (Test-PathsOverlap $OutputFullPath $InputFullPath) {
        throw "OutputDir and InputDir must be separate, non-overlapping directory trees: $OutputDir"
    }
    if (Test-PathsOverlap $OutputFullPath $SourceFullPath) {
        throw "OutputDir and SourceDir must be separate, non-overlapping directory trees: $OutputDir"
    }

    if (-not $AllowIncompleteV1) {
        $MissingV1Files = @()
        $AccountedV1Images = 0
        for ($V1Shard = 0; $V1Shard -lt $V1ShardCount; $V1Shard++) {
            $V1ManifestName = "inference_manifest.shard-{0:D3}-of-{1:D3}.jsonl" -f $V1Shard, $V1ShardCount
            $V1ErrorName = "inference_errors.shard-{0:D3}-of-{1:D3}.jsonl" -f $V1Shard, $V1ShardCount
            $V1ManifestPath = Join-Path $InputDir $V1ManifestName
            $V1ErrorPath = Join-Path $InputDir $V1ErrorName
            if (-not (Test-Path -LiteralPath $V1ManifestPath -PathType Leaf)) {
                $MissingV1Files += $V1ManifestName
            }
            else {
                $AccountedV1Images += (Get-Content -LiteralPath $V1ManifestPath | Measure-Object -Line).Lines
            }
            if (-not (Test-Path -LiteralPath $V1ErrorPath -PathType Leaf)) {
                $MissingV1Files += $V1ErrorName
            }
            else {
                $AccountedV1Images += (Get-Content -LiteralPath $V1ErrorPath | Measure-Object -Line).Lines
            }
        }
        if ($MissingV1Files.Count -gt 0) {
            $Preview = ($MissingV1Files | Select-Object -First 3) -join ", "
            throw "v1 result directory is still incomplete ($($MissingV1Files.Count) shard accounting file(s) missing; e.g. $Preview). Finish v1 first so later results are not missed. Use -AllowIncompleteV1 only for an intentional pilot, then rerun every tag shard after v1 completes."
        }
        $SupportedExtensions = @(".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
        $SourceImageCount = (
            Get-ChildItem -LiteralPath $SourceDir -Recurse -File |
                Where-Object { $SupportedExtensions -contains $_.Extension.ToLowerInvariant() } |
                Measure-Object
        ).Count
        if ($SourceImageCount -le 0) {
            throw "SourceDir contains no supported images: $SourceDir"
        }
        if ($AccountedV1Images -ne $SourceImageCount) {
            throw "v1 is not a complete snapshot: source_images=$SourceImageCount but success_plus_current_errors=$AccountedV1Images. Finish or rerun all v1 shards without -Limit before bulk status tagging."
        }
    }

    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $LogDir = Join-Path $OutputDir "_logs"
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    Write-Host "Sources:    $SourceDir"
    Write-Host "v1 results: $InputDir"
    Write-Host "Tag output: $OutputDir"
    Write-Host "Checkpoint: $Checkpoint"
    Write-Host "Shards:     $StartShard through $EndShard of $ShardCount"
    Write-Host "Thresholds: normal=$ConfidenceThreshold, check_absent=$AbsentConfidenceThreshold"
    if ($Limit -gt 0) {
        Write-Host "Limit:      fixed first $Limit results in each selected shard"
    }

    for ($ShardIndex = $StartShard; $ShardIndex -le $EndShard; $ShardIndex++) {
        $ShardDisplay = $ShardIndex + 1
        $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
        $LogPath = Join-Path $LogDir ("status_{0:D3}_of_{1:D3}_{2}.log" -f $ShardIndex, $ShardCount, $Timestamp)
        $Arguments = @(
            $EnrichScript,
            "--checkpoint", $Checkpoint,
            "--input", $InputDir,
            "--output", $OutputDir,
            "--device", "cuda",
            "--confidence-threshold", $ConfidenceThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--absent-confidence-threshold", $AbsentConfidenceThreshold.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--skip-existing",
            "--continue-on-error",
            "--shard-count", $ShardCount,
            "--shard-index", $ShardIndex
        )
        if ($Limit -gt 0) {
            $Arguments += @("--limit", $Limit)
        }

        Write-Host ""
        Write-Host ("[{0}] Starting status-tag shard {1}/{2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $ShardDisplay, $ShardCount)
        $Timer = [System.Diagnostics.Stopwatch]::StartNew()
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $ExitCode = $null
        try {
            & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
            $ExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $PreviousErrorActionPreference
        }
        $Timer.Stop()
        if ($null -eq $ExitCode -or $ExitCode -ne 0) {
            throw "Status-tag shard $ShardIndex stopped with exit code $ExitCode. Rerun the same shard after fixing the cause. Log: $LogPath"
        }

        $Suffix = ".shard-{0:D3}-of-{1:D3}" -f $ShardIndex, $ShardCount
        $ManifestPath = Join-Path $OutputDir ("status_style_manifest{0}.jsonl" -f $Suffix)
        $ErrorPath = Join-Path $OutputDir ("status_style_errors{0}.jsonl" -f $Suffix)
        $NormalCount = if (Test-Path -LiteralPath $ManifestPath) {
            (Get-Content -LiteralPath $ManifestPath | Measure-Object -Line).Lines
        } else { 0 }
        $ErrorCount = if (Test-Path -LiteralPath $ErrorPath) {
            (Get-Content -LiteralPath $ErrorPath | Measure-Object -Line).Lines
        } else { 0 }
        Write-Host ("Completed status shard {0}/{1}: tagged_or_skipped={2}, errors={3}, elapsed={4}" -f $ShardDisplay, $ShardCount, $NormalCount, $ErrorCount, $Timer.Elapsed)
        Write-Host "Log: $LogPath"
    }

    $ManifestPattern = "status_style_manifest.shard-*-of-{0:D3}.jsonl" -f $ShardCount
    $ErrorPattern = "status_style_errors.shard-*-of-{0:D3}.jsonl" -f $ShardCount
    $TotalNormal = 0
    $TotalErrors = 0
    Get-ChildItem -LiteralPath $OutputDir -Filter $ManifestPattern -File | ForEach-Object {
        $TotalNormal += (Get-Content -LiteralPath $_.FullName | Measure-Object -Line).Lines
    }
    Get-ChildItem -LiteralPath $OutputDir -Filter $ErrorPattern -File | ForEach-Object {
        $TotalErrors += (Get-Content -LiteralPath $_.FullName | Measure-Object -Line).Lines
    }
    Write-Host ""
    Write-Host "Current status-tag summary: tagged_or_skipped=$TotalNormal, errors=$TotalErrors"
    Write-Host "Rerunning the same command is safe; matching sidecars are skipped."
}
finally {
    Pop-Location
}
