import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import urllib.request

import pytest

from whisper_key import update_checker as updates

WHEEL_NAME = "whisper_key_local-1.10.0-py3-none-any.whl"
WHEEL_BYTES = b"release wheel bytes"
WHEEL_HASH = hashlib.sha256(WHEEL_BYTES).hexdigest()


def release_data():
    return {
        "tag_name": "v1.10.0",
        "assets": [
            {"name": name, "browser_download_url": f"{updates.RELEASE_DOWNLOAD_URL}/v1.10.0/{name}"}
            for name in (WHEEL_NAME, "SHA256SUMS.txt")
        ],
    }


@pytest.fixture
def release():
    return updates.select_release_update(release_data())


def serve_downloads(monkeypatch, release, manifest=None, wheel=WHEEL_BYTES):
    if manifest is None:
        manifest = f"{WHEEL_HASH}  {WHEEL_NAME}\n".encode()
    def open_asset(url):
        return io.BytesIO(manifest if url == release.manifest_url else wheel)
    monkeypatch.setattr(updates, "open_release_asset", open_asset)


def test_selects_matching_release_wheel(release):
    assert release.version == "1.10.0"
    assert release.wheel_name == WHEEL_NAME


@pytest.mark.parametrize("name", ["other_project-1.10.0-py3-none-any.whl",
                                  "whisper_key_local-1.9.0-py3-none-any.whl", "bad.whl"])
def test_rejects_wrong_project_version_and_invalid_wheels(name):
    data = release_data()
    data["assets"][0]["name"] = name
    with pytest.raises(ValueError):
        updates.select_release_update(data)


@pytest.mark.parametrize("index", [0, 1])
def test_rejects_duplicate_or_missing_release_assets(index):
    data = release_data()
    data["assets"].append(data["assets"][index])
    with pytest.raises(ValueError):
        updates.select_release_update(data)
    data = release_data()
    data["assets"].pop(index)
    with pytest.raises(ValueError):
        updates.select_release_update(data)


@pytest.mark.parametrize("url", ["http://github.com/SesioN/whisper-key-local/releases/download/v1.10.0/a.whl",
                                 "https://evil.example/a.whl",
                                 "https://github.com/other/repository/releases/download/v1.10.0/a.whl"])
def test_rejects_untrusted_initial_asset_url(url):
    data = release_data()
    data["assets"][0]["browser_download_url"] = url
    with pytest.raises(ValueError):
        updates.select_release_update(data)


def test_fetch_returns_none_for_malformed_release(monkeypatch):
    monkeypatch.setattr(updates.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b'{}'))
    assert updates.fetch_latest_release() is None


def test_fetch_uses_release_selection(monkeypatch, release):
    monkeypatch.setattr(updates.urllib.request, "urlopen",
                        lambda *args, **kwargs: io.BytesIO(json.dumps(release_data()).encode()))
    assert updates.fetch_latest_release() == release


@pytest.mark.parametrize("target", ["https://evil.example/file", "http://release-assets.githubusercontent.com/file",
                                    "https://user:secret@release-assets.githubusercontent.com/file",
                                    "https://release-assets.githubusercontent.com:8443/file",
                                    "https://release-assets.githubusercontent.com.evil.example/file"])
def test_redirect_rejected_before_request(release, target):
    handler = updates.ReleaseRedirectHandler(release.wheel_url)
    with pytest.raises(ValueError):
        handler.redirect_request(urllib.request.Request(release.wheel_url), None, 302, "Found", {}, target)


def test_github_asset_redirect_allowed(release):
    target = "https://release-assets.githubusercontent.com/github-production-release-asset/123/file?sig=abc"
    handler = updates.ReleaseRedirectHandler(release.wheel_url)
    request = handler.redirect_request(urllib.request.Request(release.wheel_url), None, 302, "Found", {}, target)
    assert request.full_url == target


@pytest.mark.parametrize("manifest", ["", f"{'0' * 64}  other.whl\n", "invalid checksum  wheel.whl\n",
                                      f"{WHEEL_HASH}  {WHEEL_NAME}\n{WHEEL_HASH}  {WHEEL_NAME}\n"])
def test_invalid_manifest_never_runs_pip(monkeypatch, release, manifest):
    serve_downloads(monkeypatch, release, manifest=manifest.encode())
    pip = Mock()
    restart = Mock()
    monkeypatch.setattr(updates.subprocess, "run", pip)
    monkeypatch.setattr(updates, "restart_or_exit", restart)
    assert updates.run_update(release) is False
    pip.assert_not_called()
    restart.assert_not_called()


@pytest.mark.parametrize("wheel", [b"tampered", WHEEL_BYTES[:-1], b""])
def test_corrupt_or_truncated_wheel_never_runs_pip(monkeypatch, release, wheel):
    serve_downloads(monkeypatch, release, wheel=wheel)
    pip = Mock()
    monkeypatch.setattr(updates.subprocess, "run", pip)
    assert updates.run_update(release) is False
    pip.assert_not_called()


def test_interrupted_download_never_runs_pip(monkeypatch, release):
    monkeypatch.setattr(updates, "open_release_asset", Mock(side_effect=OSError("download interrupted")))
    pip = Mock()
    monkeypatch.setattr(updates.subprocess, "run", pip)
    assert updates.run_update(release) is False
    pip.assert_not_called()


def test_verified_wheel_installed_locally_and_cleaned_up(monkeypatch, release):
    serve_downloads(monkeypatch, release)
    paths = []
    def install(command):
        path = Path(command[-1])
        assert path.name == WHEEL_NAME
        assert path.read_bytes() == WHEEL_BYTES
        paths.append(path)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(updates.subprocess, "run", install)
    restart = Mock()
    monkeypatch.setattr(updates, "restart_or_exit", restart)
    assert updates.run_update(release) is True
    restart.assert_called_once()
    assert not paths[0].exists()


def test_pip_failure_does_not_restart(monkeypatch, release):
    serve_downloads(monkeypatch, release)
    monkeypatch.setattr(updates.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=1)))
    restart = Mock()
    monkeypatch.setattr(updates, "restart_or_exit", restart)
    assert updates.run_update(release) is False
    restart.assert_not_called()


@pytest.mark.parametrize("test_mode,version", [(True, "1.9.0"), (False, "1.9.0-dev")])
def test_source_and_test_mode_never_install(monkeypatch, release, test_mode, version):
    monkeypatch.setattr(updates, "fetch_latest_release", lambda: release)
    monkeypatch.setattr(updates, "get_version", lambda: version)
    installer = Mock()
    monkeypatch.setattr(updates, "run_update", installer)
    updates.check_for_updates(Mock(), test_mode=test_mode)
    installer.assert_not_called()


def test_auto_update_uses_verified_path(monkeypatch, release):
    monkeypatch.setattr(updates, "fetch_latest_release", lambda: release)
    monkeypatch.setattr(updates, "get_version", lambda: "1.9.0")
    installer = Mock()
    monkeypatch.setattr(updates, "run_update", installer)
    config = Mock()
    config.get_update_config.return_value = {"mode": "auto"}
    updates.check_for_updates(config)
    installer.assert_called_once_with(release)


@pytest.mark.parametrize("name", ["whisper_key_local-1.10.0-py3-none-any/../../escape.whl",
                                  "whisper_key_local-1.10.0-py3-none-any\\..\\escape.whl"])
def test_wheel_name_cannot_include_path_components(name):
    data = release_data()
    data["assets"][0]["name"] = name
    data["assets"][0]["browser_download_url"] = f"{updates.RELEASE_DOWNLOAD_URL}/v1.10.0/{name}"
    with pytest.raises(ValueError):
        updates.select_release_update(data)


def test_oversized_manifest_never_runs_pip(monkeypatch, release):
    monkeypatch.setattr(updates, "MANIFEST_MAX_BYTES", 16)
    serve_downloads(monkeypatch, release)
    pip = Mock()
    monkeypatch.setattr(updates.subprocess, "run", pip)
    assert updates.run_update(release) is False
    pip.assert_not_called()


def test_partial_wheel_read_failure_never_runs_pip(monkeypatch, release):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.side_effect = [WHEEL_BYTES[:3], OSError("connection lost")]
    manifest = io.BytesIO(f"{WHEEL_HASH}  {WHEEL_NAME}\n".encode())
    monkeypatch.setattr(updates, "open_release_asset", Mock(side_effect=[manifest, response]))
    pip = Mock()
    monkeypatch.setattr(updates.subprocess, "run", pip)
    assert updates.run_update(release) is False
    pip.assert_not_called()


def test_final_download_url_rejection_closes_response(monkeypatch, release):
    response = Mock()
    response.geturl.return_value = "https://evil.example/file"
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(updates.urllib.request, "build_opener", Mock(return_value=opener))
    with pytest.raises(ValueError):
        updates.open_release_asset(release.wheel_url)
    response.close.assert_called_once()
