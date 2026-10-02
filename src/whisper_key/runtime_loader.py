import json
import os
import sys
from pathlib import Path

CPU = "cpu"
CUDA = "cuda"
ROCM = "rocm"
VULKAN = "vulkan"
RUNTIME_KEYS = (CPU, CUDA, ROCM, VULKAN)
CT2_RUNTIMES = (CUDA, ROCM)

PYTHON_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
RUNTIME_FOLDERS = {CUDA: f"ct2-cuda-{PYTHON_TAG}", ROCM: f"ct2-rocm-{PYTHON_TAG}", VULKAN: "whisper.cpp-vulkan"}
MARKER_FILE = "runtime.json"
BUNDLED_DLL_GLOBS = ("_rocm_sdk_*/bin", "nvidia/*/bin")

_active_ct2_runtime = CPU


def get_runtimes_dir() -> Path:
    override = os.environ.get("WHISPERKEY_RUNTIMES_DIR")
    if override:
        return Path(override)
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "whisperkey" / "runtimes"


def get_runtime_dir(runtime_key: str) -> Path:
    return get_runtimes_dir() / RUNTIME_FOLDERS[runtime_key]


def read_runtime_marker(runtime_key: str):
    if runtime_key not in RUNTIME_FOLDERS:
        return None
    marker_path = get_runtime_dir(runtime_key) / MARKER_FILE
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_runtime_installed(runtime_key: str) -> bool:
    return read_runtime_marker(runtime_key) is not None


def whisper_cpp_runtime_paths():
    if not is_runtime_installed(VULKAN):
        return None, None
    runtime_dir = get_runtime_dir(VULKAN)
    return str(runtime_dir / "bin" / "whisper-cli.exe"), str(runtime_dir / "models")


def derive_runtime(whisper_settings: dict, onboarding_settings: dict) -> str:
    if whisper_settings.get("runtime") in RUNTIME_KEYS:
        return whisper_settings["runtime"]
    if whisper_settings.get("engine_type") == "whisper_cpp":
        return VULKAN
    if whisper_settings.get("device") == "cuda":
        gpu_class = onboarding_settings.get("gpu_class") or ""
        return ROCM if gpu_class.startswith("amd") else CUDA
    return CPU


def runtime_dll_directories(runtime_dir: Path, marker: dict) -> list:
    directories = [str(path) for pattern in BUNDLED_DLL_GLOBS for path in sorted(runtime_dir.glob(pattern)) if path.is_dir()]
    directories.extend(directory for directory in marker.get("system_dll_dirs", []) if os.path.isdir(directory))
    return directories


def add_dll_directories(directories: list):
    for directory in directories:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(directory)
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")


def read_selected_runtime() -> str:
    from ruamel.yaml import YAML
    from .utils import get_user_app_data_path

    settings_path = Path(get_user_app_data_path()) / "user_settings.yaml"
    try:
        with open(settings_path, encoding="utf-8") as settings_file:
            settings = YAML(typ="safe").load(settings_file) or {}
    except Exception:
        return CPU
    return derive_runtime(settings.get("whisper") or {}, settings.get("onboarding") or {})


def activate_selected_runtime():
    global _active_ct2_runtime
    runtime = read_selected_runtime()
    if runtime not in CT2_RUNTIMES:
        return

    marker = read_runtime_marker(runtime)
    if marker is None:
        print(f"⚠ Runtime [{runtime}] is selected but not installed, using CPU")
        return

    missing_system_dirs = [directory for directory in marker.get("system_dll_dirs", []) if not os.path.isdir(directory)]
    if missing_system_dirs:
        print(f"⚠ Runtime [{runtime}] needs {missing_system_dirs[0]}, which no longer exists. Using CPU")
        return

    runtime_dir = get_runtime_dir(runtime)
    add_dll_directories(runtime_dll_directories(runtime_dir, marker))
    sys.path.insert(0, str(runtime_dir))
    _active_ct2_runtime = runtime


def active_ct2_runtime() -> str:
    return _active_ct2_runtime
