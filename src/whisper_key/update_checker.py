import hashlib
import json
import logging
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from .terminal_ui import BOLD_GREEN, BOLD_RED, RESET, prompt_choice
from .utils import get_version, restart_or_exit

RELEASES_URL = "https://api.github.com/repos/SesioN/whisper-key-local/releases/latest"
RELEASE_DOWNLOAD_URL = "https://github.com/SesioN/whisper-key-local/releases/download"
RELEASES_TIMEOUT = 3
DOWNLOAD_TIMEOUT = 30
MANIFEST_MAX_BYTES = 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

UPDATE_NOW = 1
ALWAYS_UPDATE = 2
NOT_NOW = 3


@dataclass(frozen=True)
class ReleaseUpdate:
    version: str
    wheel_name: str
    wheel_url: str
    manifest_url: str


def select_release_update(data):
    tag = data["tag_name"]
    if not re.fullmatch(r"v[0-9][A-Za-z0-9.+-]*", tag):
        raise ValueError("Invalid release tag")
    version = Version(tag[1:])
    wheels = []
    manifests = []
    for asset in data["assets"]:
        name = asset["name"]
        if name == "SHA256SUMS.txt":
            manifests.append(asset)
        elif re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", name):
            try:
                project, wheel_version, _, _ = parse_wheel_filename(name)
            except ValueError:
                continue
            if canonicalize_name(project) == "whisper-key-local" and wheel_version == version:
                wheels.append(asset)
    if len(wheels) != 1 or len(manifests) != 1:
        raise ValueError("Release needs exactly one matching wheel and checksum manifest")
    for asset in (wheels[0], manifests[0]):
        expected_url = f"{RELEASE_DOWNLOAD_URL}/{tag}/{asset['name']}"
        if asset["browser_download_url"] != expected_url:
            raise ValueError("Release asset URL is outside this repository")
    return ReleaseUpdate(str(version), wheels[0]["name"], wheels[0]["browser_download_url"],
                         manifests[0]["browser_download_url"])


def fetch_latest_release():
    try:
        request = urllib.request.Request(RELEASES_URL, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=RELEASES_TIMEOUT) as response:
            data = json.loads(response.read())
        return select_release_update(data)
    except Exception as error:
        logging.getLogger(__name__).warning("Could not check release updates: %s", error)
        return None


def is_newer(latest, current):
    try:
        return Version(latest) > Version(current)
    except ValueError:
        return False


def check_for_updates(config_manager, test_mode=False):
    version = get_version()
    is_dev = version.endswith("-dev") or test_mode
    release = fetch_latest_release()
    compare_version = version.replace("-dev", "") if version.endswith("-dev") else version
    if release is None or not is_newer(release.version, compare_version):
        return
    if is_dev:
        print(f"   ** Update available: {version} -> {release.version} (git pull to update)")
        return
    update_config = config_manager.get_update_config()
    if update_config.get('mode') == 'auto':
        run_update(release)
        return
    choice = prompt_choice("Update available: {} -> {}".format(version, release.version), [
        ("Update now", "Downloads, verifies and installs the update"),
        ("Always keep up-to-date", "Update now and auto-install future updates"),
        ("Not now", "Skip for this session"),
    ])
    if choice == UPDATE_NOW:
        run_update(release)
    elif choice == ALWAYS_UPDATE:
        config_manager.update_user_setting('update', 'mode', 'auto')
        run_update(release)
    else:
        print()


def validate_download_url(url, initial_url):
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Release downloads require HTTPS without URL credentials")
    if url != initial_url and parsed.hostname != "release-assets.githubusercontent.com":
        raise ValueError("Release download redirected outside GitHub release assets")


class ReleaseRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, initial_url):
        self.initial_url = initial_url

    def redirect_request(self, request, response, code, message, headers, new_url):
        validate_download_url(new_url, self.initial_url)
        return super().redirect_request(request, response, code, message, headers, new_url)


def open_release_asset(url):
    if not url.startswith(f"{RELEASE_DOWNLOAD_URL}/"):
        raise ValueError("Release asset URL is outside this repository")
    validate_download_url(url, url)
    opener = urllib.request.build_opener(ReleaseRedirectHandler(url))
    response = opener.open(url, timeout=DOWNLOAD_TIMEOUT)
    try:
        validate_download_url(response.geturl(), url)
    except Exception:
        response.close()
        raise
    return response


def checksum_for_wheel(manifest, wheel_name):
    entries = {}
    for line in manifest.splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *]([^\r\n]+)", line)
        if match is None:
            raise ValueError("Malformed release checksum manifest")
        checksum, name = match.groups()
        if name in entries:
            raise ValueError("Duplicate release checksum entry")
        entries[name] = checksum.lower()
    if wheel_name not in entries:
        raise ValueError("Release checksum manifest does not contain the update wheel")
    return entries[wheel_name]


def download_verified_wheel(release, directory):
    with open_release_asset(release.manifest_url) as response:
        manifest = response.read(MANIFEST_MAX_BYTES + 1)
    if len(manifest) > MANIFEST_MAX_BYTES:
        raise ValueError("Release checksum manifest is too large")
    expected_checksum = checksum_for_wheel(manifest.decode("utf-8-sig"), release.wheel_name)
    wheel_path = Path(directory) / release.wheel_name
    digest = hashlib.sha256()
    with open_release_asset(release.wheel_url) as response, wheel_path.open("wb") as output:
        while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected_checksum:
        raise ValueError("Update wheel checksum mismatch; the download was discarded")
    return wheel_path


def run_update(release):
    print(f"\n{BOLD_GREEN}Whisper Key {release.version} available. Downloading and verifying update...{RESET}\n")
    try:
        with tempfile.TemporaryDirectory(prefix="whisper-key-update-") as directory:
            wheel_path = download_verified_wheel(release, directory)
            result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", str(wheel_path)])
        if result.returncode != 0:
            raise RuntimeError("pip could not install the verified update")
    except Exception as error:
        logging.getLogger(__name__).warning("Update failed: %s", error)
        print(f"\n{BOLD_RED}Update failed: {error}. Please try again later.{RESET}\n")
        return False
    restart_or_exit(
        f"\n{BOLD_GREEN}Whisper Key {release.version} installed. Restarting...{RESET}\n",
        f"\n{BOLD_GREEN}Whisper Key {release.version} installed. Please relaunch the app.{RESET}\n",
    )
    return True
