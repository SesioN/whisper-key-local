function Assert-PyAppSourceArchive {
    param(
        [Parameter(Mandatory)][string]$ArchivePath,
        [string]$MetadataPath = (Join-Path $PSScriptRoot "pyapp-source.json")
    )
    $SourceMetadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
    if ($SourceMetadata.sha256 -notmatch '^[0-9a-f]{64}$') {
        throw "Invalid pinned PyApp source checksum"
    }
    $ActualHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLower()
    if ($ActualHash -ne $SourceMetadata.sha256) {
        throw "PyApp source checksum mismatch; refusing to extract $ArchivePath"
    }
    return $SourceMetadata
}
