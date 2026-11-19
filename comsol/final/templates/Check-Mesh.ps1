#requires -Version 5.1
[CmdletBinding()]
param([string]$ComsolRoot='E:\COMSOL61\Multiphysics', [ValidateRange(1,64)][int]$Cores=2)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$runDir=Join-Path $root ('build\mesh_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
$null=New-Item -ItemType Directory -Path $runDir -Force
$sources=@('BuildSilicaGrating.java','SweepData.java','CheckFinalMesh.java') | ForEach-Object {Join-Path $root $_}
& (Join-Path $ComsolRoot 'java\win64\jre\bin\javac.exe') -encoding UTF-8 -cp (Join-Path $ComsolRoot 'plugins\*') @sources
if ($LASTEXITCODE -ne 0) {throw 'Compilation failed'}
$stdout=Join-Path $runDir 'stdout.log'
& (Join-Path $ComsolRoot 'bin\win64\comsolbatch.exe') -inputfile (Join-Path $root 'CheckFinalMesh.class') -classpathadd $root -outputfile (Join-Path $runDir 'mesh_check.mph') -batchlog (Join-Path $runDir 'comsol.log') -np $Cores | Tee-Object -FilePath $stdout
if (-not (Select-String -LiteralPath $stdout -Pattern '^FINAL_MESH_CHECK_PASSED=28$' -Quiet)) {throw "Mesh check failed or incomplete: $stdout"}
Write-Host "Mesh refinement passed. Evidence: $stdout"
