param(
    [Parameter(Mandatory)][string]$Destination,
    [string]$ArchivePath
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "verify-pyapp-source.ps1")
$SourceMetadata = Get-Content -LiteralPath (Join-Path $PSScriptRoot "pyapp-source.json") -Raw | ConvertFrom-Json
if (-not $ArchivePath) {
    $ArchivePath = Join-Path ([System.IO.Path]::GetTempPath()) "pyapp-$([guid]::NewGuid()).tar.gz"
    Invoke-WebRequest $SourceMetadata.url -OutFile $ArchivePath
}
$SourceMetadata = Assert-PyAppSourceArchive -ArchivePath $ArchivePath
if (Test-Path -LiteralPath $Destination) {
    if (Get-ChildItem -LiteralPath $Destination -Force | Select-Object -First 1) {
        throw "PyApp source destination must be empty: $Destination"
    }
} else {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
}
tar -xzf $ArchivePath -C $Destination --strip-components=1
if ($LASTEXITCODE -ne 0) { throw "Could not extract verified PyApp source" }
$SourceMetadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Destination "WHISPER-KEY-SOURCE.json") -Encoding utf8
Write-Host "Verified and extracted PyApp $($SourceMetadata.version)"
