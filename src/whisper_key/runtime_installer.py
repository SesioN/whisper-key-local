import glob
import hashlib
import http.client
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable, Optional

from .onboarding import CT2_WHEEL_URLS, NVIDIA_PACKAGES, ROCM_72_PACKAGES
from .runtime_loader import (CUDA, MARKER_FILE, ONNX_CUDA, ONNX_DIRECTML, ORT_RUNTIMES, PYTHON_TAG, ROCM, VULKAN,
                             active_ct2_runtime, active_ort_runtime, get_runtime_dir)

ORT_PACKAGES = {
    ONNX_DIRECTML: ["onnxruntime-directml==1.24.4"],
    ONNX_CUDA: ["onnxruntime-gpu==1.24.4"],
}
ORT_CUDA_LIBRARIES = ["nvidia-cuda-nvrtc-cu12~=12.0", "nvidia-cuda-runtime-cu12~=12.0", "nvidia-cufft-cu12~=11.0",
                      "nvidia-curand-cu12~=10.0", "nvidia-cudnn-cu12~=9.0", "nvidia-cublas-cu12~=12.0"]
ORT_PROVIDER_NAMES = {ONNX_DIRECTML: "DmlExecutionProvider", ONNX_CUDA: "CUDAExecutionProvider"}
HF_TREE_URL = "https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
HF_FILE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
CT2_MODEL_FILES = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json")
CT2_REQUIRED_FILES = ("config.json", "model.bin")

ROCM_SYSTEM_SDK_VERSION = "7.2"
ROCM_SYSTEM_DLLS = ("amdhip64_7.dll", "hipblas.dll", "rocblas.dll")
CUDA_SYSTEM_DLLS = ("cublas64_12.dll", "cudnn64_9.dll")
CT2_FALLBACK_VERSION = "4.7.1"

WHISPER_CPP_VERSION = "v1.9.4"
WHISPER_CPP_SOURCE_SHA256 = "873e67727d51213d3a14a6700c7415900a6645b78c4e6edad9328eea90e53572"
WHISPER_CPP_SOURCE_URL = "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/{tag}.zip"
GGML_MODEL_REVISION = "5359861c739e955e79d9a303bcbc70fb988958b1"
GGML_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/{revision}/{file_name}"
GGML_MODEL_NAMES = (
    "large-v3-turbo", "large-v3", "large-v2", "large-v1", "medium.en", "medium",
    "small.en", "small", "base.en", "base", "tiny.en", "tiny",
)
GGML_MODEL_ALIASES = {
    "large": "large-v3", "distil-small.en": "small.en", "distil-medium.en": "medium.en",
    "distil-large-v2": "large-v2", "distil-large-v3": "large-v3", "distil-large-v3.5": "large-v3",
}
GGML_MODEL_SHA256 = {
    "ggml-tiny.bin": "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
    "ggml-tiny.en.bin": "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f",
    "ggml-base.bin": "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
    "ggml-base.en.bin": "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
    "ggml-small.bin": "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
    "ggml-small.en.bin": "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d",
    "ggml-medium.bin": "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
    "ggml-medium.en.bin": "cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356",
    "ggml-large-v1.bin": "7d99f41a10525d0206bddadd86760181fa920438b6b33237e3118ff6c83bb53d",
    "ggml-large-v2.bin": "9a423fe4d40c82774b6af34115b8b935f34152246eb19e80e376071d3f999487",
    "ggml-large-v3.bin": "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
    "ggml-large-v3-turbo.bin": "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
}

WINGET_PACKAGES = {
    "cmake": ["Kitware.CMake"],
    "vulkan_sdk": ["KhronosGroup.VulkanSDK"],
    "build_tools": ["Microsoft.VisualStudio.2022.BuildTools", "--override",
                    "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"],
}

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_ATTEMPTS = 3
INSTALL_FREE_SPACE_BYTES = 5 * 1024 ** 3
BUILD_FREE_SPACE_BYTES = 2 * 1024 ** 3
PIP_TIMEOUT_SECONDS = 3600
BUILD_TIMEOUT_SECONDS = 3600
WINGET_REBOOT_REQUIRED = 0x8A150109
WINGET_SUCCESS_CODES = (0, 3010, WINGET_REBOOT_REQUIRED, WINGET_REBOOT_REQUIRED - 2**32)
OUTPUT_TAIL_LINES = 2000
RENAME_ATTEMPTS = 10
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


@dataclass(frozen=True)
class InstallOption:
    use_system_sdk: bool
    description: str
    download_size: str


class RuntimeInstallError(Exception):
    pass


def find_system_rocm_bin() -> Optional[str]:
    candidates = [os.environ.get(f"HIP_PATH_{ROCM_SYSTEM_SDK_VERSION.replace('.', '')}"), os.environ.get("HIP_PATH")]
    for hip_path in filter(None, candidates):
        bin_dir = os.path.join(hip_path, "bin")
        version_matches = any(part == ROCM_SYSTEM_SDK_VERSION or part.startswith(ROCM_SYSTEM_SDK_VERSION + ".")
                              for part in Path(hip_path).parts)
        if version_matches and all(os.path.isfile(os.path.join(bin_dir, dll)) for dll in ROCM_SYSTEM_DLLS):
            return bin_dir
    return None


def find_system_cuda_dirs() -> Optional[list]:
    search_dirs = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        search_dirs.append(os.path.join(cuda_path, "bin"))
    search_dirs.extend(os.environ.get("PATH", "").split(os.pathsep))

    found_dirs = []
    for dll in CUDA_SYSTEM_DLLS:
        directory = next((d for d in search_dirs if d and os.path.isfile(os.path.join(d, dll))), None)
        if directory is None:
            return None
        if directory not in found_dirs:
            found_dirs.append(directory)
    return found_dirs


def install_options(runtime_key: str) -> list:
    if runtime_key == ROCM:
        options = [InstallOption(False, "Download the complete ROCm runtime", "1.1 GB")]
        if find_system_rocm_bin():
            options.insert(0, InstallOption(True, f"Use the installed AMD HIP SDK {ROCM_SYSTEM_SDK_VERSION}", "21 MB"))
        return options
    if runtime_key == CUDA:
        options = [InstallOption(False, "Download the complete CUDA runtime", "1.2 GB")]
        if find_system_cuda_dirs():
            options.insert(0, InstallOption(True, "Use the installed CUDA 12 and cuDNN 9", "60 MB"))
        return options
    if runtime_key == VULKAN:
        return [InstallOption(False, "Build whisper.cpp with Vulkan (installs CMake, Vulkan SDK and C++ build tools if missing)", "2-4 GB")]
    if runtime_key == ONNX_DIRECTML:
        return [InstallOption(False, "Download ONNX Runtime with DirectML (works with any DirectX 12 GPU)", "25 MB")]
    if runtime_key == ONNX_CUDA:
        return [InstallOption(False, "Download ONNX Runtime with CUDA 12 and cuDNN 9", "1.5 GB")]
    return []


def ensure_free_space(directory: Path, required_bytes: int):
    free_bytes = shutil.disk_usage(directory).free
    if free_bytes < required_bytes:
        raise RuntimeInstallError(f"Not enough free disk space in {directory}: "
                                  f"{required_bytes / 1e9:.1f} GB needed, {free_bytes / 1e9:.1f} GB free")


def ggml_model_name(model_key: str) -> Optional[str]:
    key = model_key.rsplit("/", 1)[-1].lower()
    key = re.sub(r"^(?:faster-)?whisper-", "", key)
    key = re.sub(r"-ct2$", "", key)
    key = GGML_MODEL_ALIASES.get(key, key)
    return key if key in GGML_MODEL_NAMES else None


def ggml_model_file_name(model_key: str) -> str:
    key = ggml_model_name(model_key)
    if key:
        return f"ggml-{key}.bin"
    raise RuntimeInstallError(f"No whisper.cpp (ggml) model matches '{model_key}'")


STALE_CT2_RUNTIME_PATTERN = re.compile(r"^ct2-(?:cuda|rocm)-(cp\d+)(?:\.old|\.partial)?$")


class RuntimeInstaller:
    def __init__(self, on_progress: Callable[[str], None]):
        self.on_progress = on_progress
        self.logger = logging.getLogger(__name__)
        self._process = None
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self):
        self._cancelled = True
        process = self._process
        if process and process.poll() is None:
            self._kill_process_tree(process)

    def install(self, runtime_key: str, option: InstallOption, model_key: Optional[str] = None) -> Path:
        if self._cancelled:
            raise RuntimeInstallError("Installation cancelled")
        if runtime_key in (active_ct2_runtime(), active_ort_runtime()):
            raise RuntimeInstallError("This runtime is in use; switch to another runtime and restart first")
        final_dir = get_runtime_dir(runtime_key)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        ensure_free_space(final_dir.parent, INSTALL_FREE_SPACE_BYTES)
        staging_dir = final_dir.with_name(final_dir.name + ".partial")
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True)

        try:
            if runtime_key == VULKAN:
                marker = self._install_whisper_cpp(staging_dir, model_key)
            elif runtime_key in ORT_RUNTIMES:
                marker = self._install_ort_runtime(runtime_key, staging_dir)
            else:
                marker = self._install_ct2_runtime(runtime_key, option, staging_dir)
            marker["installed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            (staging_dir / MARKER_FILE).write_text(json.dumps(marker, indent=2), encoding="utf-8")
            self._swap_into_place(staging_dir, final_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        self._report(f"{runtime_key} runtime installed")
        if runtime_key in (CUDA, ROCM):
            self._remove_stale_ct2_runtimes(final_dir.parent)
        return final_dir

    def _remove_stale_ct2_runtimes(self, runtimes_dir: Path):
        for runtime_dir in runtimes_dir.iterdir():
            match = STALE_CT2_RUNTIME_PATTERN.match(runtime_dir.name)
            if not match or match.group(1) == PYTHON_TAG or not runtime_dir.is_dir():
                continue
            try:
                shutil.rmtree(runtime_dir)
                self.logger.info(f"Removed runtime for an unused Python version: {runtime_dir}")
            except OSError as e:
                self.logger.warning(f"Could not remove old runtime {runtime_dir}: {e}")

    def _swap_into_place(self, staging_dir: Path, final_dir: Path):
        previous_dir = final_dir.with_name(final_dir.name + ".old")
        shutil.rmtree(previous_dir, ignore_errors=True)
        if final_dir.exists():
            try:
                self._rename_with_retry(final_dir, previous_dir)
            except OSError as e:
                raise RuntimeInstallError(f"The existing runtime is in use and cannot be replaced: {e}")
            self._keep_downloaded_models(previous_dir, staging_dir)
        try:
            self._rename_with_retry(staging_dir, final_dir)
        except OSError:
            if previous_dir.exists():
                previous_dir.rename(final_dir)
            raise
        shutil.rmtree(previous_dir, ignore_errors=True)

    def _rename_with_retry(self, source: Path, destination: Path):
        for attempt in range(RENAME_ATTEMPTS):
            try:
                source.rename(destination)
                return
            except PermissionError:
                if attempt == RENAME_ATTEMPTS - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))

    def _keep_downloaded_models(self, previous_dir: Path, staging_dir: Path):
        previous_models = previous_dir / "models"
        if not previous_models.is_dir():
            return
        staging_models = staging_dir / "models"
        staging_models.mkdir(exist_ok=True)
        for model_file in previous_models.glob("*.bin"):
            if not (staging_models / model_file.name).exists():
                shutil.move(str(model_file), staging_models / model_file.name)

    def _report(self, message: str):
        self.logger.info(f"Runtime install: {message}")
        self.on_progress(message)

    def _run(self, command: list, timeout: int, env: Optional[dict] = None, cwd: Optional[str] = None,
             success_codes: tuple = (0,)) -> str:
        if self._cancelled:
            raise RuntimeInstallError("Installation cancelled")
        self.logger.info(f"Running: {' '.join(command)}")
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env, cwd=cwd, **NO_WINDOW)
        self._process = process
        output_lines = deque(maxlen=OUTPUT_TAIL_LINES)

        def collect_output():
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                self.logger.debug(line)
                if line.startswith(("Collecting", "Downloading", "Installing", "Building")):
                    self._report(line[:120])

        reader = threading.Thread(target=collect_output, daemon=True)
        reader.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_process_tree(process)
            raise RuntimeInstallError(f"Timed out after {timeout}s: {command[0]}")
        finally:
            self._process = None
        reader.join()
        if self._cancelled:
            raise RuntimeInstallError("Installation cancelled")
        if process.returncode not in success_codes:
            error_lines = [line for line in output_lines if " error " in line.lower() or "error:" in line.lower()]
            tail = "\n".join((error_lines or output_lines)[-15:])
            raise RuntimeInstallError(f"Command failed ({process.returncode}): {' '.join(command[:4])}...\n{tail}")
        return "\n".join(output_lines)

    def _kill_process_tree(self, process: subprocess.Popen):
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, **NO_WINDOW)
        else:
            process.kill()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def _pip_install(self, target_dir: Path, packages: list, no_deps: bool = False):
        if importlib.util.find_spec("pip") is None:
            raise RuntimeInstallError("pip is not available in this Python environment; run 'python -m ensurepip' and try again")
        command = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check",
                   "--progress-bar", "off", "--target", str(target_dir)]
        if no_deps:
            command.append("--no-deps")
        self._run(command + packages, PIP_TIMEOUT_SECONDS)

    def _install_ct2_runtime(self, runtime_key: str, option: InstallOption, staging_dir: Path) -> dict:
        system_dll_dirs = []
        if runtime_key == ROCM:
            ct2_package = CT2_WHEEL_URLS.get("amd_rdna2+")
            if not ct2_package:
                raise RuntimeInstallError(f"No CTranslate2 ROCm wheel for Python {sys.version_info.major}.{sys.version_info.minor}")
            if option.use_system_sdk:
                system_dll_dirs = [find_system_rocm_bin()]
            else:
                self._report("Downloading ROCm runtime libraries (1.1 GB)...")
                self._pip_install(staging_dir, ROCM_72_PACKAGES)
        else:
            try:
                ct2_version = metadata.version("ctranslate2").split("+")[0]
            except metadata.PackageNotFoundError:
                ct2_version = CT2_FALLBACK_VERSION
            ct2_package = f"ctranslate2=={ct2_version}"
            if option.use_system_sdk:
                system_dll_dirs = find_system_cuda_dirs()
            else:
                self._report("Downloading CUDA runtime libraries (1.2 GB)...")
                self._pip_install(staging_dir, NVIDIA_PACKAGES)

        self._report("Installing CTranslate2...")
        self._pip_install(staging_dir, [ct2_package], no_deps=True)

        marker = {"runtime": runtime_key, "system_dll_dirs": system_dll_dirs, "uses_system_sdk": option.use_system_sdk}
        self._report("Checking the GPU...")
        self._verify_ct2_runtime(staging_dir, marker)
        return marker

    def _verify_ct2_runtime(self, runtime_dir: Path, marker: dict):
        from .runtime_loader import runtime_dll_directories
        dll_dirs = runtime_dll_directories(runtime_dir, marker)
        script = (
            "import os, sys\n"
            f"sys.path.insert(0, {str(runtime_dir)!r})\n"
            f"for d in {dll_dirs!r}:\n"
            "    os.add_dll_directory(d)\n"
            "    os.environ['PATH'] = d + os.pathsep + os.environ['PATH']\n"
            "import ctranslate2\n"
            "print('CT2_FILE', ctranslate2.__file__)\n"
            "print('CT2_DEVICES', ctranslate2.get_cuda_device_count())\n"
            "print('CT2_TYPES', len(ctranslate2.get_supported_compute_types('cuda')))\n"
        )
        env = {name: value for name, value in os.environ.items() if name not in ("PYTHONPATH", "PYTHONHOME")}
        output = self._run([sys.executable, "-s", "-c", script], timeout=300, env=env)
        values = dict(line.split(" ", 1) for line in output.splitlines() if line.startswith("CT2_") and " " in line)
        if not Path(values.get("CT2_FILE", "")).resolve().is_relative_to(runtime_dir.resolve()):
            raise RuntimeInstallError(f"The check loaded CTranslate2 from outside the runtime: {values.get('CT2_FILE')}")
        if int(values.get("CT2_DEVICES", 0)) < 1 or int(values.get("CT2_TYPES", 0)) < 1:
            raise RuntimeInstallError("The runtime was installed but found no usable GPU")

    def _install_ort_runtime(self, runtime_key: str, staging_dir: Path) -> dict:
        if runtime_key == ONNX_CUDA:
            self._report("Downloading CUDA 12 and cuDNN libraries (1.3 GB)...")
            self._pip_install(staging_dir, ORT_CUDA_LIBRARIES)
        self._report("Installing ONNX Runtime...")
        self._pip_install(staging_dir, ORT_PACKAGES[runtime_key], no_deps=True)
        marker = {"runtime": runtime_key, "system_dll_dirs": []}
        self._report("Checking ONNX Runtime...")
        self._verify_ort_runtime(runtime_key, staging_dir, marker)
        return marker

    def _verify_ort_runtime(self, runtime_key: str, runtime_dir: Path, marker: dict):
        from .runtime_loader import runtime_dll_directories
        dll_dirs = runtime_dll_directories(runtime_dir, marker)
        script = (
            "import os, sys\n"
            f"sys.path.insert(0, {str(runtime_dir)!r})\n"
            f"for d in {dll_dirs!r}:\n"
            "    os.add_dll_directory(d)\n"
            "    os.environ['PATH'] = d + os.pathsep + os.environ['PATH']\n"
            "import onnxruntime\n"
            "print('ORT_FILE', onnxruntime.__file__)\n"
            "print('ORT_PROVIDERS', ','.join(onnxruntime.get_available_providers()))\n"
        )
        env = {name: value for name, value in os.environ.items() if name not in ("PYTHONPATH", "PYTHONHOME")}
        output = self._run([sys.executable, "-s", "-c", script], timeout=300, env=env)
        values = dict(line.split(" ", 1) for line in output.splitlines() if line.startswith("ORT_") and " " in line)
        if not Path(values.get("ORT_FILE", "")).resolve().is_relative_to(runtime_dir.resolve()):
            raise RuntimeInstallError(f"The check loaded ONNX Runtime from outside the runtime: {values.get('ORT_FILE')}")
        if ORT_PROVIDER_NAMES[runtime_key] not in values.get("ORT_PROVIDERS", ""):
            raise RuntimeInstallError(f"ONNX Runtime was installed but has no {ORT_PROVIDER_NAMES[runtime_key]}")

    def download_onnx_model(self, model_registry, model_key: str) -> Path:
        from .onnx_asr_engine import matches_any_pattern, onnx_model_file_patterns, onnx_required_file_patterns
        model = model_registry.get_model(model_key)
        patterns = onnx_model_file_patterns(model.onnx_model_type, model.quantization)
        required_groups = [required_group[:2] for required_group in onnx_required_file_patterns(model.onnx_model_type, model.quantization)]
        return self._download_hf_model(model_key, model.source, model.revision,
                                       Path(model_registry.get_onnx_model_dir(model_key)),
                                       Path(model_registry.get_onnx_complete_marker(model_key)),
                                       lambda path: matches_any_pattern(path, patterns),
                                       [lambda path, group=group: matches_any_pattern(path, group) for group in required_groups])

    def download_whisper_model(self, model_registry, model_key: str) -> Path:
        repo = model_registry.get_hf_repo(model_key)
        if not repo:
            raise RuntimeInstallError(f"The [{model_key}] model has no Hugging Face source to download from")
        return self._download_hf_model(model_key, repo, model_registry.get_revision(model_key),
                                       Path(model_registry.get_whisper_model_dir(model_key)),
                                       Path(model_registry.get_whisper_complete_marker(model_key)),
                                       lambda path: path in CT2_MODEL_FILES or path.startswith("vocabulary."),
                                       [lambda path, name=name: path == name for name in CT2_REQUIRED_FILES])

    def _download_hf_model(self, model_key: str, repo: str, revision: str, model_dir: Path, marker: Path,
                           wanted, required_checks: list) -> Path:
        self._report("Fetching the model file list...")
        tree = self._fetch_json(HF_TREE_URL.format(repo=repo, revision=revision))
        files = [entry for entry in tree if entry.get("type") == "file" and wanted(entry["path"])]
        for required in required_checks:
            if not any(required(entry["path"]) for entry in files):
                raise RuntimeInstallError(f"{repo} is missing a required model file for [{model_key}]")
        total_bytes = sum(entry.get("size", 0) for entry in files)
        model_dir.mkdir(parents=True, exist_ok=True)
        ensure_free_space(model_dir, total_bytes)
        for index, entry in enumerate(files, start=1):
            destination = model_dir / entry["path"]
            expected_sha256 = (entry.get("lfs") or {}).get("oid")
            if destination.exists() and destination.stat().st_size == entry.get("size"):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._report(f"Downloading {entry['path']} ({index}/{len(files)}, {entry.get('size', 0) / 1e9:.2f} GB)...")
            self._download_file(HF_FILE_URL.format(repo=repo, revision=revision, path=entry["path"]),
                                destination, expected_sha256, entry.get("size"))
        marker.write_text(json.dumps({"source": repo, "revision": revision, "files": [entry["path"] for entry in files]}, indent=2),
                          encoding="utf-8")
        self._report(f"[{model_key}] downloaded")
        return model_dir

    def _fetch_json(self, url: str):
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            if self._cancelled:
                raise RuntimeInstallError("Installation cancelled")
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (OSError, http.client.HTTPException, ValueError) as e:
                client_error = isinstance(e, urllib.error.HTTPError) and e.code < 500
                if self._cancelled or client_error or attempt == DOWNLOAD_ATTEMPTS:
                    raise
                self._report(f"Request failed, retrying ({attempt}/{DOWNLOAD_ATTEMPTS - 1})...")
                time.sleep(attempt * 3)

    def _install_whisper_cpp(self, staging_dir: Path, model_key: Optional[str]) -> dict:
        if not model_key:
            raise RuntimeInstallError("A model is needed to set up whisper.cpp")
        model_file_name = ggml_model_file_name(model_key)

        build_env = self._prepare_build_tools()
        tag = WHISPER_CPP_VERSION

        work_dir = Path(tempfile.gettempdir()) / "wkb"
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)
        ensure_free_space(work_dir, BUILD_FREE_SPACE_BYTES)
        try:
            source_dir = self._download_whisper_cpp_source(tag, work_dir)
            build_dir = work_dir / "b"
            self._report(f"Configuring whisper.cpp {tag} with Vulkan...")
            self._run([build_env["CMAKE"], "-S", str(source_dir), "-B", str(build_dir), "-DGGML_VULKAN=ON",
                       "-DWHISPER_BUILD_TESTS=OFF", "-DWHISPER_BUILD_SERVER=ON"], BUILD_TIMEOUT_SECONDS, env=build_env)
            self._report("Compiling whisper.cpp (this takes a few minutes)...")
            self._run([build_env["CMAKE"], "--build", str(build_dir), "--config", "Release", "--parallel"],
                      BUILD_TIMEOUT_SECONDS, env=build_env)

            bin_dir = staging_dir / "bin"
            bin_dir.mkdir()
            built_files = glob.glob(str(build_dir / "bin" / "**" / "*.dll"), recursive=True)
            for executable in ("whisper-cli.exe", "whisper-server.exe"):
                built_files += glob.glob(str(build_dir / "bin" / "**" / executable), recursive=True)
            for executable in ("whisper-cli.exe", "whisper-server.exe"):
                if not any(name.endswith(executable) for name in built_files):
                    raise RuntimeInstallError(f"The whisper.cpp build did not produce {executable}")
            for built_file in built_files:
                shutil.copy2(built_file, bin_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        model_dir = staging_dir / "models"
        model_dir.mkdir()
        self._download_file(GGML_MODEL_URL.format(revision=GGML_MODEL_REVISION, file_name=model_file_name), model_dir / model_file_name,
                            GGML_MODEL_SHA256[model_file_name])

        binary = bin_dir / "whisper-cli.exe"
        self._report("Checking the GPU...")
        self._verify_whisper_cpp(binary, model_dir / model_file_name)
        return {"runtime": VULKAN, "whisper_cpp_version": tag, "model": model_file_name}

    def _prepare_build_tools(self) -> dict:
        env = dict(os.environ)

        cmake = shutil.which("cmake") or self._first_existing([r"C:\Program Files\CMake\bin\cmake.exe"])
        if not cmake:
            self._winget_install("cmake")
            cmake = self._first_existing([r"C:\Program Files\CMake\bin\cmake.exe"])
        if not cmake:
            raise RuntimeInstallError("CMake could not be installed")

        vulkan_sdk = self._find_vulkan_sdk()
        if not vulkan_sdk:
            self._winget_install("vulkan_sdk")
            vulkan_sdk = self._find_vulkan_sdk()
        if not vulkan_sdk:
            raise RuntimeInstallError("The Vulkan SDK could not be installed")

        if not self._has_cpp_build_tools():
            self._winget_install("build_tools")
            if not self._has_cpp_build_tools():
                raise RuntimeInstallError("The C++ build tools could not be installed")

        env["CMAKE"] = cmake
        env["VULKAN_SDK"] = vulkan_sdk
        env["PATH"] = os.pathsep.join([os.path.join(vulkan_sdk, "Bin"), os.path.dirname(cmake), env.get("PATH", "")])
        return env

    def _first_existing(self, paths: list) -> Optional[str]:
        return next((path for path in paths if os.path.isfile(path)), None)

    def _find_vulkan_sdk(self) -> Optional[str]:
        candidates = [os.environ.get("VULKAN_SDK"), self._machine_environment_variable("VULKAN_SDK")]
        candidates += sorted(glob.glob(r"C:\VulkanSDK\*"), key=self._version_key, reverse=True)
        for candidate in filter(None, candidates):
            if os.path.isfile(os.path.join(candidate, "Bin", "glslc.exe")):
                return candidate
        return None

    @staticmethod
    def _version_key(path: str) -> tuple:
        return tuple(int(part) for part in re.findall(r"\d+", os.path.basename(path)))

    def _machine_environment_variable(self, name: str) -> Optional[str]:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                value, value_type = winreg.QueryValueEx(key, name)
        except OSError:
            return None
        return winreg.ExpandEnvironmentStrings(value) if value_type == winreg.REG_EXPAND_SZ else value

    def _has_cpp_build_tools(self) -> bool:
        vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                               "Microsoft Visual Studio", "Installer", "vswhere.exe")
        if not os.path.isfile(vswhere):
            return False
        result = subprocess.run([vswhere, "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                 "-property", "installationPath"], capture_output=True, text=True, **NO_WINDOW)
        return bool(result.stdout.strip())

    def _winget_install(self, tool: str):
        package_id, *extra_arguments = WINGET_PACKAGES[tool]
        winget = shutil.which("winget")
        if not winget:
            raise RuntimeInstallError(f"winget is not available; install {package_id} manually and try again")
        self._report(f"Installing {package_id} (approve the Windows prompt if one appears)...")
        self._run([winget, "install", "--id", package_id, "-e", "--source", "winget", "--silent",
                   "--accept-package-agreements", "--accept-source-agreements", *extra_arguments],
                  BUILD_TIMEOUT_SECONDS, success_codes=WINGET_SUCCESS_CODES)

    def _download_whisper_cpp_source(self, tag: str, work_dir: Path) -> Path:
        archive = work_dir / "whisper.cpp.zip"
        self._download_file(WHISPER_CPP_SOURCE_URL.format(tag=tag), archive, WHISPER_CPP_SOURCE_SHA256)
        with zipfile.ZipFile(archive) as source_zip:
            source_zip.extractall(work_dir / "s")
        top_level_dirs = [path for path in (work_dir / "s").iterdir() if path.is_dir()]
        if len(top_level_dirs) != 1:
            raise RuntimeInstallError("Unexpected whisper.cpp source archive layout")
        return top_level_dirs[0]

    def download_ggml_model(self, model_key: str, model_dir: Path) -> Path:
        file_name = ggml_model_file_name(model_key)
        model_dir.mkdir(parents=True, exist_ok=True)
        destination = model_dir / file_name
        self._download_file(GGML_MODEL_URL.format(revision=GGML_MODEL_REVISION, file_name=file_name), destination, GGML_MODEL_SHA256[file_name])
        return destination

    def _download_file(self, url: str, destination: Path, expected_sha256: Optional[str],
                       expected_size: Optional[int] = None):
        partial = destination.with_name(destination.name + ".download")
        failures_without_progress = 0
        largest_partial_bytes = partial.stat().st_size if partial.exists() else 0
        while True:
            try:
                return self._download_file_once(url, destination, expected_sha256, expected_size)
            except (OSError, http.client.HTTPException) as e:
                client_error = isinstance(e, urllib.error.HTTPError) and e.code < 500
                bytes_after_attempt = partial.stat().st_size if partial.exists() else 0
                if bytes_after_attempt > largest_partial_bytes:
                    largest_partial_bytes = bytes_after_attempt
                    failures_without_progress = 0
                else:
                    failures_without_progress += 1
                if self._cancelled or client_error or failures_without_progress >= DOWNLOAD_ATTEMPTS:
                    partial.unlink(missing_ok=True)
                    if isinstance(e, http.client.IncompleteRead):
                        raise RuntimeInstallError(f"The download of {destination.name} keeps stopping early at "
                                                  f"{bytes_after_attempt} bytes; a proxy or firewall may be cutting it off") from e
                    raise
                self.logger.warning(f"Download of {destination.name} failed at {bytes_after_attempt} bytes ({e}), retrying")
                self._report(f"Download interrupted at {bytes_after_attempt / 1e9:.2f} GB, resuming...")
                time.sleep(max(failures_without_progress, 1) * 5)
                if self._cancelled:
                    partial.unlink(missing_ok=True)
                    raise RuntimeInstallError("Installation cancelled")

    def _download_file_once(self, url: str, destination: Path, expected_sha256: Optional[str],
                            expected_size: Optional[int]):
        partial = destination.with_name(destination.name + ".download")
        digest = hashlib.sha256()
        resume_from = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={resume_from}-"} if resume_from else {})
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code != 416:
                raise
            partial.unlink()
            return self._download_file_once(url, destination, expected_sha256, expected_size)
        with response:
            if resume_from:
                self.logger.info(f"Resuming {destination.name} at {resume_from} bytes, server answered HTTP {response.status}")
            if response.status != 206:
                resume_from = 0
            if resume_from:
                with open(partial, "rb") as existing:
                    while chunk := existing.read(DOWNLOAD_CHUNK_BYTES):
                        digest.update(chunk)
            content_length = int(response.headers.get("Content-Length") or 0)
            ensure_free_space(destination.parent, content_length)
            total_bytes = expected_size or (content_length + resume_from if content_length else 0)
            received_bytes = resume_from
            with open(partial, "ab" if resume_from else "wb") as output:
                last_reported_percent = -1
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    if self._cancelled:
                        raise RuntimeInstallError("Installation cancelled")
                    output.write(chunk)
                    digest.update(chunk)
                    received_bytes += len(chunk)
                    if total_bytes:
                        percent = received_bytes * 100 // total_bytes
                        if percent >= last_reported_percent + 10:
                            last_reported_percent = percent
                            self._report(f"Downloading {destination.name}... {percent}%")
        if total_bytes and received_bytes < total_bytes:
            raise http.client.IncompleteRead(b"", total_bytes - received_bytes)
        if expected_sha256 and digest.hexdigest() != expected_sha256:
            partial.unlink(missing_ok=True)
            self.logger.error(f"Checksum mismatch for {destination.name}: received {received_bytes} bytes, "
                              f"expected {total_bytes or 'unknown'} bytes")
            raise RuntimeInstallError(f"Checksum mismatch for {destination.name}; the download was discarded. "
                                      f"A proxy or virus scanner may be altering the download")
        partial.replace(destination)

    def _verify_whisper_cpp(self, binary: Path, model_path: Path):
        with tempfile.TemporaryDirectory() as temp_dir:
            silent_wav = Path(temp_dir) / "silence.wav"
            with wave.open(str(silent_wav), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 16000)
            output = self._run([str(binary), "-m", str(model_path), "-f", str(silent_wav), "-nt"], timeout=300)
        if not re.search(r"ggml_vulkan: Found [1-9]\d* Vulkan devices?", output):
            raise RuntimeInstallError("whisper.cpp was built but did not find a Vulkan GPU")
