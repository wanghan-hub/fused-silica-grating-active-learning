#requires -Version 5.1
<#!
.SYNOPSIS
Merge independently computed COMSOL powers into the original 90-row request CSV.
.DESCRIPTION
Accepts a COMSOL numeric table (comments and headers are allowed) or batch-log
lines beginning COMSOL_RESULT,. The rightmost 13 columns must be:
case_id, period_nm, height_nm, top_duty, bottom_duty, wavelength_nm, angle_deg,
R_TE, T_TE, R_TM, T_TM, energy_TE, energy_TM.
An explicitly supplied BatchLog takes precedence over RawCsv. All 45 cases must
occur exactly once. Only R_comsol and T_comsol cells are replaced; identifiers,
coordinate strings, other columns, and original request row order are retained.
No result CSV is written if validation fails. A sibling *_qc.json is written for
both success and failure; an existing result CSV is left unchanged on failure.
.EXAMPLE
.\Merge-ComsolResults.ps1 -RawCsv .\comsol_raw.csv
.EXAMPLE
.\Merge-ComsolResults.ps1 -BatchLog .\solve.log
#>
[CmdletBinding()]
param(
    [string]$RequestCsv,
    [string]$RawCsv,
    [string]$OutputCsv,
    [string]$BatchLog
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Resolve script-relative defaults after parameter binding (Windows PowerShell
# 5.1 can expose an empty PSScriptRoot in default-argument expressions).
if ([string]::IsNullOrWhiteSpace($RequestCsv)) { $RequestCsv = Join-Path $PSScriptRoot 'comsol_request.csv' }
if ([string]::IsNullOrWhiteSpace($RawCsv)) { $RawCsv = Join-Path $PSScriptRoot 'comsol_raw.csv' }
if ([string]::IsNullOrWhiteSpace($OutputCsv)) { $OutputCsv = Join-Path $PSScriptRoot 'initial_results.csv' }
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$numberStyle = [System.Globalization.NumberStyles]::Float
$coordinateTolerance = 1.0e-8
$powerTolerance = 1.0e-8
$energyTolerance = 1.0e-5
$coordinateColumns = @('period_nm', 'height_nm', 'top_duty', 'bottom_duty',
                       'wavelength_nm', 'angle_deg')
$outputPath = [System.IO.Path]::GetFullPath($OutputCsv)
$qcPath = Join-Path ([System.IO.Path]::GetDirectoryName($outputPath)) ([System.IO.Path]::GetFileNameWithoutExtension($outputPath) + '_qc.json')
$qc = [ordered]@{
    passed = $false
    generated_utc = [DateTime]::UtcNow.ToString('o', $culture)
    request_file = [System.IO.Path]::GetFullPath($RequestCsv)
    source_file = $null
    source_kind = $null
    output_file = $outputPath
    csv_written = $false
    existing_output_preserved_on_failure = $false
    expected_request_rows = 90
    expected_physical_cases = 45
    request_rows = 0
    physical_cases = 0
    source_rows = 0
    ignored_source_lines = 0
    coordinate_tolerance = $coordinateTolerance
    power_tolerance = $powerTolerance
    energy_tolerance = $energyTolerance
    max_energy_error = $null
    max_reported_energy_error = $null
    min_power = $null
    max_power = $null
    errors = @()
}

function Convert-FiniteNumber {
    param([string]$Text, [string]$Context)
    [double]$value = 0
    if (-not [double]::TryParse($Text, $numberStyle, $culture, [ref]$value)) {
        throw "Invalid numeric value in ${Context}: '$Text'."
    }
    if ([double]::IsNaN($value) -or [double]::IsInfinity($value)) {
        throw "Nonfinite numeric value in ${Context}: '$Text'."
    }
    return $value
}

function Read-NumericRows {
    param([string]$Path, [bool]$IsBatchLog)
    $rows = [System.Collections.Generic.List[object]]::new()
    $lineNumber = 0
    foreach ($line in [System.IO.File]::ReadLines($Path)) {
        $lineNumber++
        $text = $line.Trim().TrimStart([char]0xfeff)
        if ($IsBatchLog) {
            if (-not $text.StartsWith('COMSOL_RESULT,', [StringComparison]::Ordinal)) {
                $qc.ignored_source_lines++
                continue
            }
            $text = $text.Substring('COMSOL_RESULT,'.Length)
        } elseif (-not $text -or $text.StartsWith('%') -or $text.StartsWith('#')) {
            $qc.ignored_source_lines++
            continue
        }
        # Numeric COMSOL tables use commas, tabs, or whitespace. Quoted numeric
        # cells are accepted; commas inside explanatory headers are immaterial.
        if ($text.Contains(',')) {
            $parts = $text.Split(',')
        } elseif ($text.Contains(';')) {
            $parts = $text.Split(';')
        } else {
            $parts = [regex]::Split($text, '\s+')
        }
        $parts = @($parts | ForEach-Object { $_.Trim().Trim('"') })
        if ($parts.Count -lt 13) {
            if ($IsBatchLog -or $text -match '^["\s]*[+\-]?(?:[0-9.]|NaN|Inf)') {
                throw "Source line $lineNumber has fewer than the required 13 columns."
            }
            $qc.ignored_source_lines++
            continue
        }
        $tail = @($parts[($parts.Count - 13)..($parts.Count - 1)])
        [double]$first = 0
        $numericCase = [double]::TryParse($tail[0], $numberStyle, $culture, [ref]$first)
        if (-not $numericCase) {
            if ($IsBatchLog -or $text -match '^["\s]*[+\-]?(?:[0-9.]|NaN|Inf)') {
                throw "Source line $lineNumber has an invalid case_id or malformed numeric columns."
            }
            $qc.ignored_source_lines++
            continue
        }
        $values = [double[]]::new(13)
        for ($index = 0; $index -lt 13; $index++) {
            $values[$index] = Convert-FiniteNumber $tail[$index] "source line $lineNumber, column $($index + 1)"
        }
        $rows.Add([pscustomobject]@{ line = $lineNumber; values = $values })
    }
    return $rows.ToArray()
}

function Write-QcReport {
    foreach ($protectedPath in @($qc.request_file, $qc.source_file)) {
        if (-not [string]::IsNullOrWhiteSpace($protectedPath) -and
            $qcPath.Equals($protectedPath, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Cannot write QC report over an input file.'
        }
    }
    $parent = [System.IO.Path]::GetDirectoryName($qcPath)
    [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    $temporary = $qcPath + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    try {
        $json = $qc | ConvertTo-Json -Depth 8
        [System.IO.File]::WriteAllText($temporary, $json, [System.Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $temporary -Destination $qcPath -Force
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
    }
}

try {
    $requestPath = [System.IO.Path]::GetFullPath($RequestCsv)
    $isLog = -not [string]::IsNullOrWhiteSpace($BatchLog)
    $sourcePath = if ($isLog) { [System.IO.Path]::GetFullPath($BatchLog) } else {
        [System.IO.Path]::GetFullPath($RawCsv)
    }
    $qc.source_file = $sourcePath
    $qc.source_kind = if ($isLog) { 'batch_log' } else { 'numeric_table' }
    foreach ($inputPath in @($requestPath, $sourcePath)) {
        if ($outputPath.Equals($inputPath, [StringComparison]::OrdinalIgnoreCase) -or
            $qcPath.Equals($inputPath, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Output/QC paths must not overwrite either input file.'
        }
        if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
            throw "Required input file does not exist: $inputPath"
        }
    }
    $request = @(Import-Csv -LiteralPath $requestPath -Encoding UTF8)
    $qc.request_rows = $request.Count
    if ($request.Count -ne 90) { throw "Expected 90 request conditions; found $($request.Count)." }
    $columns = @($request[0].PSObject.Properties.Name)
    foreach ($name in @('condition_id', 'polarization', 'R_comsol', 'T_comsol') + $coordinateColumns) {
        if ($columns -cnotcontains $name) { throw "Missing request column: $name" }
    }
    $ids = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $byTuple = [System.Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    $cases = [System.Collections.Generic.List[object]]::new()
    foreach ($row in $request) {
        if ([string]::IsNullOrWhiteSpace($row.condition_id) -or -not $ids.Add($row.condition_id)) {
            throw "Missing or duplicate request condition_id: '$($row.condition_id)'."
        }
        if (@('TE', 'TM') -cnotcontains $row.polarization) {
            throw "Request polarization must be exactly TE or TM: '$($row.polarization)'."
        }
        $coordinates = [double[]]::new(6)
        for ($i = 0; $i -lt 6; $i++) {
            $coordinates[$i] = Convert-FiniteNumber $row.($coordinateColumns[$i]) "request $($row.condition_id), $($coordinateColumns[$i])"
        }
        $tupleKey = ($coordinates | ForEach-Object { $_.ToString('R', $culture) }) -join '|'
        if (-not $byTuple.ContainsKey($tupleKey)) {
            $case = [pscustomobject]@{
                id = $cases.Count + 1
                coordinates = $coordinates
                rows = [System.Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
            }
            $byTuple.Add($tupleKey, $case)
            $cases.Add($case)
        }
        $case = $byTuple[$tupleKey]
        if ($case.rows.ContainsKey($row.polarization)) {
            throw "Duplicate polarization $($row.polarization) for physical case $($case.id)."
        }
        $case.rows.Add($row.polarization, $row)
    }
    $qc.physical_cases = $cases.Count
    if ($cases.Count -ne 45) { throw "Expected 45 unique physical cases; found $($cases.Count)." }
    foreach ($case in $cases) {
        if ($case.rows.Count -ne 2 -or -not $case.rows.ContainsKey('TE') -or -not $case.rows.ContainsKey('TM')) {
            throw "Physical case $($case.id) must contain exactly one TE and one TM condition."
        }
    }
    $source = @(Read-NumericRows $sourcePath $isLog)
    $qc.source_rows = $source.Count
    if ($source.Count -ne 45) { throw "Expected 45 numeric COMSOL rows; found $($source.Count)." }
    $seen = [System.Collections.Generic.HashSet[int]]::new()
    $failures = [System.Collections.Generic.List[string]]::new()
    $maxEnergy = 0.0
    $maxReportedEnergy = 0.0
    $minPower = [double]::PositiveInfinity
    $maxPower = [double]::NegativeInfinity
    foreach ($raw in $source) {
        $v = $raw.values
        if ($v[0] -ne [Math]::Truncate($v[0]) -or $v[0] -lt 1 -or $v[0] -gt 45) {
            throw "Source line $($raw.line) has out-of-range/noninteger case_id: $($v[0])."
        }
        $caseId = [int]$v[0]
        if (-not $seen.Add($caseId)) { throw "Duplicate COMSOL case_id: $caseId." }
        $case = $cases[$caseId - 1]
        for ($i = 0; $i -lt 6; $i++) {
            if ([Math]::Abs($v[$i + 1] - $case.coordinates[$i]) -gt $coordinateTolerance) {
                $failures.Add("Case ${caseId}: coordinate mismatch in $($coordinateColumns[$i]).")
            }
        }
        for ($i = 7; $i -le 10; $i++) {
            $minPower = [Math]::Min($minPower, $v[$i])
            $maxPower = [Math]::Max($maxPower, $v[$i])
            if ($v[$i] -lt -$powerTolerance -or $v[$i] -gt 1 + $powerTolerance) {
                $failures.Add("Case ${caseId}: power outside allowed range in numeric column $($i + 1).")
            }
        }
        foreach ($polIndex in 0..1) {
            $pol = @('TE', 'TM')[$polIndex]
            $r = $v[7 + 2 * $polIndex]
            $t = $v[8 + 2 * $polIndex]
            $energy = [Math]::Abs($r + $t - 1)
            $reported = [Math]::Abs($v[11 + $polIndex])
            $maxEnergy = [Math]::Max($maxEnergy, $energy)
            $maxReportedEnergy = [Math]::Max($maxReportedEnergy, $reported)
            if ($energy -gt $energyTolerance -or $reported -gt $energyTolerance) {
                $failures.Add("Case ${caseId} ${pol}: energy closure exceeds $energyTolerance.")
            }
            if ([Math]::Abs($reported - $energy) -gt 1.0e-8) {
                $failures.Add("Case ${caseId} ${pol}: exported energy error disagrees with independent R+T check.")
            }
            # These are raw, independently computed powers: no clipping or
            # renormalization, and transmission is never constructed from R.
            $case.rows[$pol].R_comsol = $r.ToString('G17', $culture)
            $case.rows[$pol].T_comsol = $t.ToString('G17', $culture)
        }
    }
    $qc.max_energy_error = $maxEnergy
    $qc.max_reported_energy_error = $maxReportedEnergy
    $qc.min_power = $minPower
    $qc.max_power = $maxPower
    if ($seen.Count -ne 45) { $failures.Add('Missing COMSOL case IDs.') }
    if ($failures.Count -gt 0) {
        $qc.errors = $failures.ToArray()
        throw "COMSOL result QC failed ($($failures.Count) violations); inspect $qcPath."
    }
    [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($outputPath)) | Out-Null
    $temporaryCsv = $outputPath + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    try {
        $request | Export-Csv -LiteralPath $temporaryCsv -NoTypeInformation -Encoding UTF8
        Move-Item -LiteralPath $temporaryCsv -Destination $outputPath -Force
    } finally {
        if (Test-Path -LiteralPath $temporaryCsv) { Remove-Item -LiteralPath $temporaryCsv }
    }
    $qc.csv_written = $true
    $qc.passed = $true
    Write-QcReport
    Write-Host "COMSOL merge passed: 45 physical cases / 90 conditions."
    Write-Host "Results: $outputPath"
    Write-Host "QC:      $qcPath"
} catch {
    $failureMessage = $_.Exception.Message
    $qc.passed = $false
    if (@($qc.errors).Count -eq 0) { $qc.errors = @($failureMessage) }
    $qc.existing_output_preserved_on_failure = (-not $qc.csv_written -and (Test-Path -LiteralPath $outputPath -PathType Leaf))
    try { Write-QcReport } catch { Write-Warning "Could not write QC report: $($_.Exception.Message)" }
    throw $failureMessage
}
