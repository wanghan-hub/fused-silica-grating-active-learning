#requires -Version 5.1
<#!
.SYNOPSIS
Recompute this package's 90 initial-validation conditions using COMSOL 6.1.
.DESCRIPTION
Uses the installed COMSOL Java compiler and native batch solver. No Python,
PyTorch, MATLAB, MPh package, or modification of COMSOL security preferences.
Results replace initial_results.csv only after strict completeness and power QC.
#>
[CmdletBinding()]
param(
    [string]$ComsolRoot = 'E:\COMSOL61\Multiphysics',
    [ValidateRange(1, 64)][int]$Cores = 2
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$compiler = Join-Path $ComsolRoot 'java\win64\jre\bin\javac.exe'
$batch = Join-Path $ComsolRoot 'bin\win64\comsolbatch.exe'
foreach ($file in @($compiler, $batch)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "COMSOL executable missing: $file" }
}
$request = Join-Path $root 'comsol_request.csv'
$expectedHash = '36ed217941bc25d22ae6c3d4537e8c5edb5725964b2d7eacc3779ac43051c81a'
if ((Get-FileHash -LiteralPath $request -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
    throw 'This model is tied to the supplied 90-condition request. Request CSV changed; regenerate and verify the model before solving.'
}
$runDir = Join-Path $root ('build\run_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$null = New-Item -ItemType Directory -Path $runDir -Force
$stdoutLog = Join-Path $runDir 'stdout.log'
$progressLog = Join-Path $runDir 'comsol.log'
$requestedOutput = Join-Path $runDir 'computed.mph'
$actualOutput = Join-Path $runDir 'computed_Model.mph'
$sources = @('BuildSilicaGrating.java', 'SweepData.java', 'RunInitial.java') | ForEach-Object { Join-Path $root $_ }
& $compiler -encoding UTF-8 -cp (Join-Path $ComsolRoot 'plugins\*') @sources
if ($LASTEXITCODE -ne 0) { throw 'COMSOL Java compilation failed.' }
& $batch -inputfile (Join-Path $root 'RunInitial.class') -outputfile $requestedOutput -classpathadd $root -batchlog $progressLog -np $Cores |
    Tee-Object -FilePath $stdoutLog
# COMSOL can report failure in its log while returning process exit code zero.
if (-not (Test-Path -LiteralPath $actualOutput -PathType Leaf) -or
    -not (Select-String -LiteralPath $stdoutLog -Pattern '^ALL_CONDITIONS_SOLVED=90$' -Quiet)) {
    throw "COMSOL did not finish all conditions. See $progressLog and $stdoutLog"
}
& (Join-Path $root 'Merge-ComsolResults.ps1') -BatchLog $stdoutLog
Copy-Item -LiteralPath $actualOutput -Destination (Join-Path $root 'silica_grating_initial_solved.mph') -Force
Write-Host 'Finished. Open silica_grating_initial_solved.mph; import initial_results.csv in the Python project.'
