import glob
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
from .runtime_loader import CUDA, MARKER_FILE, ROCM, VULKAN, active_ct2_runtime, get_runtime_dir

ROCM_SYSTEM_SDK_VERSION = "7.2"
ROCM_SYSTEM_DLLS = ("amdhip64_7.dll", "hipblas.dll", "rocblas.dll")
CUDA_SYSTEM_DLLS = ("cublas64_12.dll", "cudnn64_9.dll")
CT2_FALLBACK_VERSION = "4.7.1"

WHISPER_CPP_RELEASE_API = "https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest"
WHISPER_CPP_SOURCE_URL = "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/{tag}.zip"
GGML_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{file_name}"
GGML_MODEL_NAMES = (
    "large-v3-turbo", "large-v3", "large-v2", "large-v1", "medium.en", "medium",
    "small.en", "small", "base.en", "base", "tiny.en", "tiny",
)
GGML_MODEL_ALIASES = {"large": "large-v3"}

WINGET_PACKAGES = {
    "cmake": ["Kitware.CMake"],
    "vulkan_sdk": ["KhronosGroup.VulkanSDK"],
    "build_tools": ["Microsoft.VisualStudio.2022.BuildTools", "--override",
                    "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"],
}

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
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
    return []


def ggml_model_file_name(model_key: str) -> str:
    key = GGML_MODEL_ALIASES.get(model_key, model_key).lower()
    if key in GGML_MODEL_NAMES:
        return f"ggml-{key}.bin"
    raise RuntimeInstallError(f"No whisper.cpp (ggml) model matches '{model_key}'")


class RuntimeInstaller:
    def __init__(self, on_progress: Callable[[str], None]):
        self.on_progress = on_progress
        self.logger = logging.getLogger(__name__)
        self._process = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        process = self._process
        if process and process.poll() is None:
            self._kill_process_tree(process)

    def install(self, runtime_key: str, option: InstallOption, model_key: Optional[str] = None) -> Path:
        if runtime_key == active_ct2_runtime():
            raise RuntimeInstallError("This runtime is in use; switch to another runtime and restart first")
        final_dir = get_runtime_dir(runtime_key)
        staging_dir = final_dir.with_name(final_dir.name + ".partial")
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True)

        try:
            if runtime_key == VULKAN:
                marker = self._install_whisper_cpp(staging_dir, model_key)
            else:
                marker = self._install_ct2_runtime(runtime_key, option, staging_dir)
            marker["installed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            (staging_dir / MARKER_FILE).write_text(json.dumps(marker, indent=2), encoding="utf-8")
            self._swap_into_place(staging_dir, final_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        self._report(f"{runtime_key} runtime installed")
        return final_dir

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
        values = dict(line.split(" ", 1) for line in output.splitlines() if line.startswith("CT2_"))
        if not Path(values.get("CT2_FILE", "")).resolve().is_relative_to(runtime_dir.resolve()):
            raise RuntimeInstallError(f"The check loaded CTranslate2 from outside the runtime: {values.get('CT2_FILE')}")
        if int(values.get("CT2_DEVICES", 0)) < 1 or int(values.get("CT2_TYPES", 0)) < 1:
            raise RuntimeInstallError("The runtime was installed but found no usable GPU")

    def _install_whisper_cpp(self, staging_dir: Path, model_key: Optional[str]) -> dict:
        if not model_key:
            raise RuntimeInstallError("A model is needed to set up whisper.cpp")
        model_file_name = ggml_model_file_name(model_key)

        build_env = self._prepare_build_tools()
        tag = self._latest_whisper_cpp_tag()

        work_dir = Path(tempfile.gettempdir()) / "wkb"
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True)
        try:
            source_dir = self._download_whisper_cpp_source(tag, work_dir)
            build_dir = work_dir / "b"
            self._report(f"Configuring whisper.cpp {tag} with Vulkan...")
            self._run([build_env["CMAKE"], "-S", str(source_dir), "-B", str(build_dir), "-DGGML_VULKAN=ON",
                       "-DWHISPER_BUILD_TESTS=OFF", "-DWHISPER_BUILD_SERVER=OFF"], BUILD_TIMEOUT_SECONDS, env=build_env)
            self._report("Compiling whisper.cpp (this takes a few minutes)...")
            self._run([build_env["CMAKE"], "--build", str(build_dir), "--config", "Release", "--parallel"],
                      BUILD_TIMEOUT_SECONDS, env=build_env)

            bin_dir = staging_dir / "bin"
            bin_dir.mkdir()
            built_files = glob.glob(str(build_dir / "bin" / "**" / "*.dll"), recursive=True)
            built_files += glob.glob(str(build_dir / "bin" / "**" / "whisper-cli.exe"), recursive=True)
            if not any(name.endswith("whisper-cli.exe") for name in built_files):
                raise RuntimeInstallError("The whisper.cpp build did not produce whisper-cli.exe")
            for built_file in built_files:
                shutil.copy2(built_file, bin_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        model_dir = staging_dir / "models"
        model_dir.mkdir()
        self._download_file(GGML_MODEL_URL.format(file_name=model_file_name), model_dir / model_file_name)

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

    def _latest_whisper_cpp_tag(self) -> str:
        with urllib.request.urlopen(WHISPER_CPP_RELEASE_API, timeout=30) as response:
            return json.load(response)["tag_name"]

    def _download_whisper_cpp_source(self, tag: str, work_dir: Path) -> Path:
        archive = work_dir / "whisper.cpp.zip"
        self._download_file(WHISPER_CPP_SOURCE_URL.format(tag=tag), archive)
        with zipfile.ZipFile(archive) as source_zip:
            source_zip.extractall(work_dir / "s")
        top_level_dirs = [path for path in (work_dir / "s").iterdir() if path.is_dir()]
        if len(top_level_dirs) != 1:
            raise RuntimeInstallError("Unexpected whisper.cpp source archive layout")
        return top_level_dirs[0]

    def _download_file(self, url: str, destination: Path):
        partial = destination.with_name(destination.name + ".download")
        with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as output:
            total_bytes = int(response.headers.get("Content-Length") or 0)
            received_bytes = 0
            last_reported_percent = -1
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                output.write(chunk)
                received_bytes += len(chunk)
                if total_bytes:
                    percent = received_bytes * 100 // total_bytes
                    if percent >= last_reported_percent + 10:
                        last_reported_percent = percent
                        self._report(f"Downloading {destination.name}... {percent}%")
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
