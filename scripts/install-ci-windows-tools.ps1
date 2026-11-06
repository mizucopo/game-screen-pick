param([Parameter(Mandatory = $true)][string]$Prefix)
$ErrorActionPreference = 'Stop'

# Native tools are CI inputs, never bundled in the product archive.
# Pin a month-end build: BtbN retains these for two years, but only the latest
# 14 daily builds. Renew this pin before 2028-10-31; do not use floating latest.
# https://github.com/BtbN/FFmpeg-Builds#release-retention-policy
$release = 'autobuild-2026-10-31-12-59'
$architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
$sources = @{
    X64 = @{
        Name = 'ffmpeg-n9.0.2-39-g7adfbe40dc-win64-lgpl-9.0.zip'
        SHA256 = '9d37c564e0d01c18cde184a079fa6ef04d7d2783ada41da81d04165cc8f1a650'
        Machine = 0x8664
    }
    Arm64 = @{
        Name = 'ffmpeg-n9.0.2-39-g7adfbe40dc-winarm64-lgpl-9.0.zip'
        SHA256 = 'd1981518b99a1c05d6b864200b1ac9459b09e2ee174feffe84b6e6bb0bfa79e2'
        Machine = 0xAA64
    }
}
if (-not $sources.ContainsKey($architecture)) { throw "Unsupported Windows architecture: $architecture" }
$source = $sources[$architecture]
$temporary = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
New-Item -ItemType Directory $temporary | Out-Null
try {
    $archive = Join-Path $temporary 'ffmpeg.zip'
    $url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/$release/$($source.Name)"
    Invoke-WebRequest $url -OutFile $archive
    if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $source.SHA256) {
        throw 'FFmpeg archive checksum differs from the pinned CI input'
    }
    Expand-Archive $archive (Join-Path $temporary 'unpacked')
    $bin = Join-Path $Prefix 'bin'
    New-Item -ItemType Directory -Force $bin | Out-Null
    foreach ($name in @('ffmpeg.exe', 'ffprobe.exe')) {
        $matches = @(Get-ChildItem (Join-Path $temporary 'unpacked') -Recurse -File -Filter $name)
        if ($matches.Count -ne 1) { throw "Expected one $name" }
        $bytes = [System.IO.File]::ReadAllBytes($matches[0].FullName)
        $header = [BitConverter]::ToInt32($bytes, 0x3C)
        if ([BitConverter]::ToUInt32($bytes, $header) -ne 0x00004550 -or
            [BitConverter]::ToUInt16($bytes, $header + 4) -ne $source.Machine) {
            throw "$name is not a native $architecture PE executable"
        }
        Copy-Item $matches[0].FullName (Join-Path $bin $name)
        & (Join-Path $bin $name) -version
        if ($LASTEXITCODE -ne 0) { throw "$name failed to start" }
    }
    $bin | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
} finally {
    Remove-Item -Recurse -Force $temporary
}
