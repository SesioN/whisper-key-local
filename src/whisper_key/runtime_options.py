import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

from .platform import gpu
from .runtime_loader import CPU, CUDA, ROCM, VULKAN, active_ct2_runtime, is_runtime_installed, whisper_cpp_runtime_paths

COMPUTE_TYPES = ("int8", "int8_float32", "int8_float16", "int8_bfloat16", "int16", "float16", "bfloat16", "float32")
PREFERRED_COMPUTE_TYPES = {
    "cpu": ("int8", "int8_float32", "float32"),
    "cuda": ("float16", "int8_float16", "int8", "float32"),
}

FASTER_WHISPER = "faster_whisper"
WHISPER_CPP = "whisper_cpp"

INSTALLED = "installed"
INSTALLABLE = "installable"
UNSUPPORTED = "unsupported"

GPU_RUNTIME_LABELS = {CUDA: "NVIDIA CUDA", ROCM: "AMD ROCm"}
GPU_RUNTIME_HARDWARE = {CUDA: ("nvidia",), ROCM: ("amd_rdna2+",)}


@dataclass(frozen=True)
class Runtime:
    key: str
    label: str
    engine_type: str
    device: str
    state: str
    reason: str = ""
    compute_types: frozenset = field(default_factory=frozenset)
    needs_restart: bool = False
    install_size: str = ""

    @property
    def available(self) -> bool:
        return self.state == INSTALLED and not self.needs_restart

    @property
    def selectable(self) -> bool:
        return self.state != UNSUPPORTED

    @property
    def menu_label(self) -> str:
        if self.state == INSTALLABLE:
            return f"{self.label} (install, {self.install_size})"
        if self.state == UNSUPPORTED:
            return f"{self.label} ({self.reason})"
        if self.needs_restart:
            return f"{self.label} (restarts the app)"
        return self.label


def detect_runtimes(whisper_cpp_binary: Optional[str]) -> list:
    logger = logging.getLogger(__name__)
    cpu_types = frozenset()
    gpu_types = frozenset()
    gpu_device_count = 0

    try:
        import ctranslate2
        cpu_types = frozenset(ctranslate2.get_supported_compute_types("cpu"))
        gpu_device_count = ctranslate2.get_cuda_device_count()
        if gpu_device_count > 0:
            gpu_types = frozenset(ctranslate2.get_supported_compute_types("cuda"))
    except Exception as e:
        logger.warning(f"CTranslate2 capability check failed: {e}")

    loaded_ct2_variant = gpu.detect_ct2_variant()
    gpu_class, gpu_name = gpu.detect_gpu_class()
    cpu_state = INSTALLED if cpu_types else UNSUPPORTED

    runtimes = [Runtime(CPU, "CPU", FASTER_WHISPER, "cpu", cpu_state, "CTranslate2 not available", cpu_types)]
    if sys.platform != "darwin":
        runtimes += [
            _gpu_runtime(CUDA, loaded_ct2_variant, gpu_device_count, gpu_types, gpu_class),
            _gpu_runtime(ROCM, loaded_ct2_variant, gpu_device_count, gpu_types, gpu_class),
        ]
    runtimes.append(_whisper_cpp_runtime(whisper_cpp_binary, gpu_class or gpu_name))
    return runtimes


def _install_size(runtime_key: str) -> str:
    from .runtime_installer import install_options
    options = install_options(runtime_key)
    return options[0].download_size if options else ""


def _gpu_runtime(key: str, loaded_ct2_variant: str, gpu_device_count: int, gpu_types: frozenset, gpu_class: Optional[str]) -> Runtime:
    label = GPU_RUNTIME_LABELS[key]
    if loaded_ct2_variant == key and gpu_device_count > 0 and gpu_types:
        return Runtime(key, label, FASTER_WHISPER, "cuda", INSTALLED, compute_types=gpu_types)
    if is_runtime_installed(key) and active_ct2_runtime() != key:
        return Runtime(key, label, FASTER_WHISPER, "cuda", INSTALLED, needs_restart=True)
    if gpu_class in GPU_RUNTIME_HARDWARE[key] and sys.platform == "win32":
        return Runtime(key, label, FASTER_WHISPER, "cuda", INSTALLABLE, install_size=_install_size(key))
    if key == ROCM and gpu_class == "amd_rdna1":
        return Runtime(key, label, FASTER_WHISPER, "cuda", UNSUPPORTED, "RDNA1 needs manual setup")
    reason = "no NVIDIA GPU" if key == CUDA else "no supported AMD GPU"
    return Runtime(key, label, FASTER_WHISPER, "cuda", UNSUPPORTED, reason)


def _whisper_cpp_runtime(whisper_cpp_binary: Optional[str], has_gpu) -> Runtime:
    label = "Vulkan (whisper.cpp)"
    if whisper_cpp_binary:
        return Runtime(VULKAN, label, WHISPER_CPP, "vulkan", INSTALLED)
    if has_gpu and gpu.has_vulkan_driver() and sys.platform == "win32":
        return Runtime(VULKAN, label, WHISPER_CPP, "vulkan", INSTALLABLE, install_size=_install_size(VULKAN))
    return Runtime(VULKAN, label, WHISPER_CPP, "vulkan", UNSUPPORTED, "no Vulkan GPU driver")


def current_runtime_key(engine_type: str, device: str, runtimes: list) -> str:
    if engine_type == WHISPER_CPP:
        return VULKAN
    if device == "cuda":
        return next((runtime.key for runtime in runtimes if runtime.key in (CUDA, ROCM) and runtime.available), CUDA)
    return CPU


def with_whisper_cpp_paths(whisper_config: dict) -> dict:
    binary, model_dir = whisper_cpp_runtime_paths()
    whisper_config = dict(whisper_config)
    if binary and not whisper_config.get('cpp_binary'):
        whisper_config['cpp_binary'] = binary
    if model_dir and not whisper_config.get('cpp_model_dir'):
        whisper_config['cpp_model_dir'] = model_dir
    return whisper_config


def apply_runtime_selection(whisper_config: dict) -> dict:
    runtime = whisper_config.get('runtime', CPU)
    whisper_config = dict(whisper_config)
    if runtime == VULKAN:
        whisper_config['engine_type'] = WHISPER_CPP
        return with_whisper_cpp_paths(whisper_config)

    whisper_config['engine_type'] = FASTER_WHISPER
    runtime_is_loaded = active_ct2_runtime() == runtime or gpu.detect_ct2_variant() == runtime
    if runtime in (CUDA, ROCM) and runtime_is_loaded:
        whisper_config['device'] = 'cuda'
    else:
        whisper_config['device'] = 'cpu'
        if whisper_config.get('compute_type') not in PREFERRED_COMPUTE_TYPES['cpu']:
            whisper_config['compute_type'] = 'int8'
    return whisper_config


def choose_compute_type(runtime: Runtime, requested: Optional[str]) -> Optional[str]:
    if not runtime.compute_types:
        if runtime.engine_type == FASTER_WHISPER and runtime.device == "cuda" and runtime.needs_restart:
            return PREFERRED_COMPUTE_TYPES["cuda"][0]
        return requested
    if requested in runtime.compute_types:
        return requested
    for compute_type in PREFERRED_COMPUTE_TYPES.get(runtime.device, ()):
        if compute_type in runtime.compute_types:
            return compute_type
    return next(compute_type for compute_type in COMPUTE_TYPES + tuple(sorted(runtime.compute_types)) if compute_type in runtime.compute_types)
