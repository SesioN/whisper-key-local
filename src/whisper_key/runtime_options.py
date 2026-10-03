import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .platform import gpu

COMPUTE_TYPES = ("int8", "int8_float32", "int8_float16", "int8_bfloat16", "int16", "float16", "bfloat16", "float32")
PREFERRED_COMPUTE_TYPES = {
    "cpu": ("int8", "int8_float32", "float32"),
    "cuda": ("float16", "int8_float16", "int8", "float32"),
}

CPU = "cpu"
CUDA = "cuda"
ROCM = "rocm"
VULKAN = "vulkan"

FASTER_WHISPER = "faster_whisper"
WHISPER_CPP = "whisper_cpp"


@dataclass(frozen=True)
class Runtime:
    key: str
    label: str
    engine_type: str
    device: str
    available: bool
    unavailable_reason: str = ""
    compute_types: frozenset = field(default_factory=frozenset)

    @property
    def menu_label(self) -> str:
        if self.available:
            return self.label
        return f"{self.label} ({self.unavailable_reason})"


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

    ct2_variant = gpu.detect_ct2_variant()

    runtimes = [Runtime(CPU, "CPU", FASTER_WHISPER, "cpu", bool(cpu_types),
                        "CTranslate2 not available", cpu_types)]
    if sys.platform != "darwin":
        runtimes += [
            _gpu_runtime(CUDA, "NVIDIA CUDA", ct2_variant, gpu_device_count, gpu_types),
            _gpu_runtime(ROCM, "AMD ROCm", ct2_variant, gpu_device_count, gpu_types),
        ]
    runtimes.append(_whisper_cpp_runtime(whisper_cpp_binary))
    return runtimes


def _gpu_runtime(key: str, label: str, ct2_variant: str, gpu_device_count: int, gpu_types: frozenset) -> Runtime:
    if ct2_variant != key:
        other_build = {CUDA: "CUDA", ROCM: "ROCm"}.get(ct2_variant)
        reason = f"CTranslate2 {other_build} build installed" if other_build else "CTranslate2 GPU build not installed"
        return Runtime(key, label, FASTER_WHISPER, "cuda", False, reason)
    if gpu_device_count == 0 or not gpu_types:
        return Runtime(key, label, FASTER_WHISPER, "cuda", False, "no usable GPU found")
    return Runtime(key, label, FASTER_WHISPER, "cuda", True, compute_types=gpu_types)


def _whisper_cpp_runtime(whisper_cpp_binary: Optional[str]) -> Runtime:
    label = "Vulkan (whisper.cpp)" if _has_vulkan_backend(whisper_cpp_binary) else "whisper.cpp"
    if not whisper_cpp_binary:
        return Runtime(VULKAN, label, WHISPER_CPP, "vulkan", False, "whisper.cpp not installed")
    return Runtime(VULKAN, label, WHISPER_CPP, "vulkan", True)


def _has_vulkan_backend(whisper_cpp_binary: Optional[str]) -> bool:
    if not whisper_cpp_binary:
        return True
    return any(Path(whisper_cpp_binary).resolve().parent.glob("*ggml-vulkan*"))


def current_runtime_key(engine_type: str, device: str, runtimes: list) -> str:
    if engine_type == WHISPER_CPP:
        return VULKAN
    if device == "cuda":
        return next((runtime.key for runtime in runtimes if runtime.key in (CUDA, ROCM) and runtime.available), CUDA)
    return CPU


def choose_compute_type(runtime: Runtime, requested: Optional[str]) -> Optional[str]:
    if not runtime.compute_types:
        return requested
    if requested in runtime.compute_types:
        return requested
    for compute_type in PREFERRED_COMPUTE_TYPES.get(runtime.device, ()):
        if compute_type in runtime.compute_types:
            return compute_type
    return next(compute_type for compute_type in COMPUTE_TYPES + tuple(sorted(runtime.compute_types)) if compute_type in runtime.compute_types)
