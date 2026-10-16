param([Parameter(Mandatory = $true)][string]$Prefix)
$ErrorActionPreference = 'Stop'

# Native tools are CI inputs, never bundled in the product archive.
$architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
$sources = @{
    X64 = @{
        Name = 'ffmpeg-n9.0.2-28-g669f5ab083-win64-lgpl-9.0.zip'
        SHA256 = 'd05ca4eeab509d631928ad26363597d3fb0dfc3ef4163980cfbfef784ca64e41'
        Machine = 0x8664
    }
    Arm64 = @{
        Name = 'ffmpeg-n9.0.2-28-g669f5ab083-winarm64-lgpl-9.0.zip'
        SHA256 = 'f64f8a72a4b7994ff10bc9b8fcf9b75a14a166889156b12487b3ca78da499f6e'
        Machine = 0xAA64
    }
}
if (-not $sources.ContainsKey($architecture)) { throw "Unsupported Windows architecture: $architecture" }
$source = $sources[$architecture]
$temporary = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
New-Item -ItemType Directory $temporary | Out-Null
try {
    $archive = Join-Path $temporary 'ffmpeg.zip'
    $url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-15-13-03/$($source.Name)"
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
