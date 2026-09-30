<#
.SYNOPSIS
  Runs the headless render test sequences and reports the failures found in the output.

.DESCRIPTION
  Each sequence file in this directory is run as one headless invocation of the sample.
  Every SEQUENCE in a file renders for -SequenceFrames frames and then saves a screenshot
  named "screenshot_<index>_<sequence name>.jpg" into its own subdirectory of -OutDir.

  There is no golden image comparison: the point is that every permutation builds its
  pipelines and renders without falling over. The screenshots are there to be looked at
  when something does go wrong.

  The sample keeps running (and keeps presenting black or stale frames) when a pipeline
  fails to build, so a zero exit code is not enough. The log is scanned for the messages
  that accompany a shader compile failure, a device loss, or a validation error.

.EXAMPLE
  ./run_tests.ps1
  ./run_tests.ps1 -Config Debug -SequenceFrames 32
  ./run_tests.ps1 -Sequences seq_visualize.txt
#>
[CmdletBinding()]
param(
    # build configuration subdirectory of _bin to test
    [string]   $Config         = 'Release',
    # explicit executable, overrides -Config
    [string]   $Exe            = '',
    # where the screenshots and logs are written
    [string]   $OutDir         = '',
    # scene configuration applied before the sequences run
    [string]   $Scene          = '',
    # sequence files to run, defaults to all seq_*.txt next to this script
    [string[]] $Sequences      = @(),
    # frames rendered per sequence: enough for streaming to settle, no more. These are not
    # performance runs, so there is nothing to gain from rendering longer.
    [int]      $SequenceFrames = 32,
    # 0 none, 1 full window, 2 rendered viewport
    [int]      $Screenshot     = 2,
    # Vulkan validation layers. On by default - without them the VUID / Validation Error
    # scan below never has anything to find, which is half of what these runs are for.
    # Pass 0 for a quick pass.
    [int]      $Validation     = 1,
    # 0 default, 1 standard, 2 reduced overhead, 3 best practices, 4 synchronization, 5 gpu assisted
    [int]      $ValidationPreset = 1,
    # extra arguments appended to every invocation, e.g. -Extra '--device','1'
    [string[]] $Extra          = @()
)

$ErrorActionPreference = 'Stop'
$testsDir = $PSScriptRoot
$rootDir  = Split-Path -Parent $testsDir

if (-not $Exe) {
    $Exe = Join-Path $rootDir "_bin/$Config/vk_lod_clusters_internal.exe"
}
if (-not (Test-Path $Exe)) {
    throw "executable not found: $Exe (build it first, or pass -Exe)"
}
$Exe = (Resolve-Path $Exe).Path

if (-not $Scene) {
    $Scene = Join-Path $testsDir 'bunny_grid.cfg'
}
$Scene = (Resolve-Path $Scene).Path

if (-not $OutDir) {
    $OutDir = Join-Path $testsDir ('_results/' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

if ($Sequences.Count -eq 0) {
    $Sequences = Get-ChildItem -Path $testsDir -Filter 'seq_*.txt' | Select-Object -ExpandProperty Name
}

# Messages that mean the frame that was rendered is not the frame that was asked for.
# The sample survives all of these, so they have to be found in the log.
$failurePatterns = @(
    'shaders failed'
    'error:'
    'ERROR:'
    'VUID-'
    'Validation Error'
    'DEVICE_LOST'
    'failed to allocate'
)

Write-Host "executable : $Exe"
Write-Host "scene      : $Scene"
Write-Host "results    : $OutDir"
Write-Host ""

$anyFailed = $false
$summary   = @()

foreach ($sequenceName in $Sequences) {
    $sequenceFile = Join-Path $testsDir $sequenceName
    if (-not (Test-Path $sequenceFile)) {
        throw "sequence file not found: $sequenceFile"
    }
    $sequenceFile = (Resolve-Path $sequenceFile).Path

    # a sequence file may bring its own scene: options that only take effect at startup
    # (--force16bitdispatch) or a scene shape a sequence cannot switch to on its own
    $sequenceScene = $Scene
    $ownScene = [System.IO.Path]::ChangeExtension($sequenceFile, '.cfg')
    if (-not $PSBoundParameters.ContainsKey('Scene') -and (Test-Path $ownScene)) {
        $sequenceScene = $ownScene
    }

    $sequenceCount = @(Select-String -Path $sequenceFile -Pattern '^\s*SEQUENCE\s').Count
    if ($sequenceCount -eq 0) {
        Write-Warning "$sequenceName contains no SEQUENCE, skipped"
        continue
    }

    # the sequencer advances every SequenceFrames frames and closes the app when done,
    # the headless budget only has to be large enough to reach that point
    $headlessFrames = $sequenceCount * $SequenceFrames + $SequenceFrames

    $caseDir = Join-Path $OutDir ([System.IO.Path]::GetFileNameWithoutExtension($sequenceName))
    New-Item -ItemType Directory -Force -Path $caseDir | Out-Null
    $logFile = Join-Path $caseDir 'output.log'

    # --sequenceframes is only registered once the sequencer has initialized, so it cannot be
    # passed on the command line: the sequence file carries it, and -SequenceFrames overrides
    # it in a copy that doubles as the record of what this run actually did
    $runFile = Join-Path $caseDir 'sequence.txt'
    (Get-Content -Raw $sequenceFile) -replace '--sequenceframes\s+\d+', "--sequenceframes $SequenceFrames" |
        Set-Content -NoNewline -Path $runFile

    $arguments = @(
        '--scene',              $sequenceScene
        '--sequencefile',       $runFile
        '--sequencescreenshot', $Screenshot
        '--validation',         $Validation
        '--validationpreset',   $ValidationPreset
        '--headless'
        '--headlessframes',     $headlessFrames
    ) + $Extra

    Write-Host ("running {0} ({1} sequences, {2} frames, scene {3})" -f `
                $sequenceName, $sequenceCount, $headlessFrames, (Split-Path -Leaf $sequenceScene))

    # screenshots are written to the working directory
    $elapsed = [System.Diagnostics.Stopwatch]::StartNew()
    Push-Location $caseDir
    try {
        & $Exe @arguments 2>&1 | Tee-Object -FilePath $logFile
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
        $elapsed.Stop()
    }

    $hits  = @(Select-String -Path $logFile -Pattern $failurePatterns -SimpleMatch)
    $shots = @(Get-ChildItem -Path $caseDir -Filter 'screenshot_*.jpg' -ErrorAction SilentlyContinue).Count

    $missingShots = ($Screenshot -ne 0) -and ($shots -lt $sequenceCount)
    $failed       = ($exitCode -ne 0) -or ($hits.Count -gt 0) -or $missingShots
    if ($failed) { $anyFailed = $true }

    $summary += [pscustomobject]@{
        File        = $sequenceName
        Result      = if ($failed) { 'FAIL' } else { 'ok' }
        Sequences   = $sequenceCount
        Screenshots = $shots
        ExitCode    = $exitCode
        Suspicious  = $hits.Count
        Seconds     = [math]::Round($elapsed.Elapsed.TotalSeconds, 1)
    }

    foreach ($hit in $hits | Select-Object -First 20) {
        Write-Host ("  {0}:{1}: {2}" -f $sequenceName, $hit.LineNumber, $hit.Line.Trim()) -ForegroundColor Yellow
    }
}

Write-Host ""
$summary | Format-Table -AutoSize
Write-Host ("total {0:n1} s" -f ($summary | Measure-Object -Property Seconds -Sum).Sum)

if ($anyFailed) {
    Write-Host "FAILED - see the logs and screenshots under $OutDir" -ForegroundColor Red
    exit 1
}

Write-Host "all sequences ran, screenshots under $OutDir" -ForegroundColor Green
exit 0
