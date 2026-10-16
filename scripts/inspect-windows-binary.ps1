param(
    [Parameter(Mandatory)][string]$Binary,
    [Parameter(Mandatory)][ValidateSet('x86_64-pc-windows-msvc', 'aarch64-pc-windows-msvc')][string]$Target,
    [Parameter(Mandatory)][string]$Report
)
$ErrorActionPreference = 'Stop'
$expected = if ($Target -eq 'aarch64-pc-windows-msvc') { 'Arm64' } else { 'X64' }
if ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() -ne $expected) {
    throw 'Windows distribution requires a native build and inspection host'
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$installation = & $vswhere -latest -products '*' -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $installation) { throw 'Visual Studio inspection tools missing' }
$hostDirectory = if ($expected -eq 'Arm64') { 'Hostarm64/arm64' } else { 'Hostx64/x64' }
$dumpbin = Get-ChildItem "$installation/VC/Tools/MSVC/*/bin/$hostDirectory/dumpbin.exe" |
    Sort-Object FullName -Descending | Select-Object -First 1
if (-not $dumpbin) { throw 'Native dumpbin missing' }
$headers = & $dumpbin.FullName /nologo /headers $Binary
if ($LASTEXITCODE -ne 0) { throw 'PE header inspection failed' }
$machine = if ($expected -eq 'Arm64') { 'AA64' } else { '8664' }
if (($headers -join "`n") -notmatch "(?im)^\s*$machine machine ") { throw 'PE machine differs from target' }
$imports = & $dumpbin.FullName /nologo /dependents $Binary
if ($LASTEXITCODE -ne 0) { throw 'PE import inspection failed' }
$dependencies = @($imports | ForEach-Object {
    if ($_ -match '^\s+([A-Za-z0-9_.-]+\.dll)\s*$') { $Matches[1].ToLowerInvariant() }
} | Sort-Object -Unique)
if ($dependencies.Count -eq 0) { throw 'PE dependency inventory missing' }
# Static CRT avoids a Visual C++ redistributable or build-tool DLL requirement.
# Every non-API-set import must be an OS component present in native System32.
$systemDlls = @('advapi32.dll', 'bcrypt.dll', 'bcryptprimitives.dll', 'crypt32.dll',
    'dnsapi.dll', 'iphlpapi.dll', 'kernel32.dll', 'mswsock.dll', 'ncrypt.dll',
    'normaliz.dll', 'ntdll.dll', 'ole32.dll', 'oleaut32.dll', 'powrprof.dll',
    'psapi.dll', 'rpcrt4.dll', 'secur32.dll', 'shell32.dll', 'shlwapi.dll',
    'user32.dll', 'userenv.dll', 'version.dll', 'winhttp.dll', 'winmm.dll', 'ws2_32.dll')
foreach ($dependency in $dependencies) {
    if ($dependency -match '^(api-ms-win-|ext-ms-win-)[a-z0-9-]+\.dll$') { continue }
    if ($dependency -notin $systemDlls -or
        -not (Test-Path (Join-Path $env:SystemRoot "System32/$dependency") -PathType Leaf)) {
        throw "Unsupported or unresolved runtime DLL: $dependency"
    }
}
@("PE target: $Target", "Native OS architecture: $expected", '', $headers, '', $imports) |
    Set-Content -Encoding utf8 $Report
