import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

BUILD_DIR = Path(__file__).resolve().parents[1] / "pyapp-build"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
POWERSHELL_ENV = {key: value for key, value in os.environ.items() if key.lower() != "psmodulepath"}


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is needed to check source archive verification")
@pytest.mark.parametrize("valid", [True, False])
def test_source_archive_checksum_before_extraction(tmp_path, valid):
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"test archive input")
    metadata = tmp_path / "source.json"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest() if valid else "0" * 64
    metadata.write_text(json.dumps({"sha256": digest}), encoding="utf-8")
    helper = str(BUILD_DIR / "verify-pyapp-source.ps1").replace("'", "''")
    archive_path = str(archive).replace("'", "''")
    metadata_path = str(metadata).replace("'", "''")
    command = f"$ErrorActionPreference = 'Stop'; . '{helper}'; Assert-PyAppSourceArchive -ArchivePath '{archive_path}' -MetadataPath '{metadata_path}'"
    result = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, text=True, env=POWERSHELL_ENV)
    assert (result.returncode == 0) is valid, result.stderr
    if not valid:
        assert "checksum mismatch" in result.stderr


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is needed to check source archive verification")
def test_invalid_archive_does_not_create_extraction_destination(tmp_path):
    archive = tmp_path / "invalid.tar.gz"
    archive.write_bytes(b"corrupt source archive")
    destination = tmp_path / "unpacked"
    result = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-File",
                             str(BUILD_DIR / "fetch-pyapp-source.ps1"), "-Destination", str(destination),
                             "-ArchivePath", str(archive)], capture_output=True, text=True, env=POWERSHELL_ENV)
    assert result.returncode != 0
    assert "checksum mismatch" in result.stderr
    assert not destination.exists()
