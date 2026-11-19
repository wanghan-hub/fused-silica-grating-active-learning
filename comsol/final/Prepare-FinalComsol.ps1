#requires -Version 5.1
<#! Generate a separate final COMSOL package from a new v2 request; never reuse v1 powers. #>
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$RequestCsv, [string]$OutputDirectory)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
if (-not $OutputDirectory) { $OutputDirectory=Join-Path $PSScriptRoot 'v2_repeat_000' }
$target=[IO.Path]::GetFullPath($OutputDirectory)
if ((Test-Path -LiteralPath $target) -and @(Get-ChildItem -LiteralPath $target -Force).Count -gt 0) {
    throw 'Output directory is not empty. Run its existing Run-Comsol.ps1, or choose a new output directory.'
}
$request=@(Import-Csv -LiteralPath $RequestCsv -Encoding UTF8)
if ($request.Count -eq 0) { throw 'Empty request' }
$names=@('period_nm','height_nm','top_duty','bottom_duty','wavelength_nm','angle_deg')
$culture=[Globalization.CultureInfo]::InvariantCulture
$pairs=[Collections.Generic.List[object]]::new()
$map=[Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
$ids=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach($row in $request) {
    if (-not $ids.Add($row.condition_id) -or -not $row.condition_id) {throw 'Duplicate/missing condition ID'}
    if (@('TE','TM') -cnotcontains $row.polarization) {throw 'Unexpected polarization'}
    $values=@($names | ForEach-Object {
        $v=[double]::Parse($row.$_,[Globalization.NumberStyles]::Float,$culture)
        if ([double]::IsNaN($v) -or [double]::IsInfinity($v)) {throw 'Nonfinite coordinate'}
        $v
    })
    $p,$h,$ft,$fb,$wave,$angle=$values
    if ($p -le 0 -or $h -lt 0 -or $ft -le 0 -or $fb -ge 1 -or $ft -gt $fb -or $angle -lt 0 -or $angle -ge 90 -or $wave -lt 210 -or $wave -gt 3710) {throw 'Unsupported geometry/state'}
    $u=$wave/1000.0
    $n=[Math]::Sqrt(1+0.6961663*$u*$u/($u*$u-0.0684043*0.0684043)+0.4079426*$u*$u/($u*$u-0.1162414*0.1162414)+0.8974794*$u*$u/($u*$u-9.896161*9.896161))
    if ($wave/$p-[Math]::Sin($angle*[Math]::PI/180)-$n -le 0) {throw 'Higher diffraction orders propagate; this port model is not applicable'}
    $text=@($values | ForEach-Object {$_.ToString('R',$culture)})
    $key=$text -join '|'
    if (-not $map.ContainsKey($key)) {
        $item=[pscustomobject]@{Values=$text;Pols=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)}
        $map.Add($key,$item);$pairs.Add($item)
    }
    if (-not $map[$key].Pols.Add($row.polarization)) {throw 'Duplicate polarization for physical case'}
}
foreach($pair in $pairs) {if ($pair.Pols.Count -ne 2) {throw 'Missing TE/TM pair'}}
$caseCount=$pairs.Count
$stateCount=2*$caseCount
$null=New-Item -ItemType Directory -Path $target -Force
[IO.File]::WriteAllBytes((Join-Path $target 'comsol_request.csv'),[IO.File]::ReadAllBytes([IO.Path]::GetFullPath($RequestCsv)))
$hash=(Get-FileHash -LiteralPath $RequestCsv -Algorithm SHA256).Hash.ToLowerInvariant()
$utf8=[Text.UTF8Encoding]::new($false)
$templates=Join-Path $PSScriptRoot 'templates'
foreach($name in @('BuildSilicaGrating.java','RunFinal.java','Run-Comsol.ps1','Merge-ComsolResults.ps1')) {
    $text=[IO.File]::ReadAllText((Join-Path $templates $name))
    $text=[regex]::Replace($text,'\b105\b','__CASE_COUNT__')
    $text=[regex]::Replace($text,'\b210\b','__STATE_COUNT__')
    $text=$text.Replace('__CASE_COUNT__',[string]$caseCount).Replace('__STATE_COUNT__',[string]$stateCount)
    $text=$text.Replace('dc0de922ecceda3a1695e0dea837409b2eeceda3e42dc83d680f7a6c607c4e9b',$hash)
    if ($name -eq 'BuildSilicaGrating.java') {
        for($i=0;$i -lt 6;$i++) {
            $pattern='m\.param\(\)\.set\("'+$names[$i]+'", "[^"]*"\);'
            $replacement='m.param().set("'+$names[$i]+'", "'+$pairs[0].Values[$i]+'");'
            $text=[regex]::Replace($text,$pattern,$replacement)
        }
    }
    [IO.File]::WriteAllText((Join-Path $target $name),$text,$utf8)
}
$lists=[Collections.Generic.List[string]]::new()
$lists.Add(((1..$caseCount) -join ' '))
for($i=0;$i -lt 6;$i++) { $lists.Add((@($pairs | ForEach-Object {$_.Values[$i]}) -join ' ')) }
$java="public class SweepData {`n  static final String[] LISTS = {`n"+((@($lists | ForEach-Object {'    "'+$_+'"'}) -join ",`n"))+"`n  };`n}`n"
[IO.File]::WriteAllText((Join-Path $target 'SweepData.java'),$java,$utf8)
# Check both wavelength endpoints at the largest requested angle for each geometry.
$waves=@($pairs | ForEach-Object {[double]::Parse($_.Values[4],$culture)})
$angles=@($pairs | ForEach-Object {[double]::Parse($_.Values[5],$culture)})
$minWave=($waves | Measure-Object -Minimum).Minimum
$maxWave=($waves | Measure-Object -Maximum).Maximum
$maxAngle=($angles | Measure-Object -Maximum).Maximum
$meshCases=@(for($i=0;$i -lt $caseCount;$i++) {if (($waves[$i] -eq $minWave -or $waves[$i] -eq $maxWave) -and $angles[$i] -eq $maxAngle) {$i+1}})
$meshStates=2*$meshCases.Count
$mesh=[IO.File]::ReadAllText((Join-Path $templates 'CheckFinalMesh.java'))
$mesh=[regex]::Replace($mesh,'int\[\] cases=\{[^}]+\};',('int[] cases={'+($meshCases -join ',')+'};'))
$mesh=$mesh.Replace('FINAL_MESH_CHECK_PASSED=28',"FINAL_MESH_CHECK_PASSED=$meshStates")
[IO.File]::WriteAllText((Join-Path $target 'CheckFinalMesh.java'),$mesh,$utf8)
$meshScript=[IO.File]::ReadAllText((Join-Path $templates 'Check-Mesh.ps1')).Replace('FINAL_MESH_CHECK_PASSED=28',"FINAL_MESH_CHECK_PASSED=$meshStates")
[IO.File]::WriteAllText((Join-Path $target 'Check-Mesh.ps1'),$meshScript,$utf8)
@{request_sha256=$hash;paired_cases=$caseCount;conditions=$stateCount;mesh_conditions=$meshStates;results_computed=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $target 'package.json') -Encoding UTF8
Write-Host "Prepared $caseCount paired cases / $stateCount states in $target. No result powers have been generated."
Write-Host 'Run Check-Mesh.ps1, then Run-Comsol.ps1, then import this new final_results.csv into v2.'
