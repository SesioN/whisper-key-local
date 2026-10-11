import fnmatch
import gc
import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

from .audio_chunking import split_at_quiet_points
from .languages import QWEN_LANGUAGE_NAMES
from .runtime_loader import ONNX_CPU, ONNX_CUDA, ONNX_DIRECTML

SPEECH_LLM = "speech-llm"
SAMPLE_RATE = 16000
SPEECH_LLM_MAX_CHUNK_SECONDS = 28
DEFAULT_MAX_CHUNK_SECONDS = 60

PROVIDERS = {
    ONNX_CPU: ["CPUExecutionProvider"],
    ONNX_DIRECTML: ["DmlExecutionProvider", "CPUExecutionProvider"],
    ONNX_CUDA: ["CUDAExecutionProvider", "CPUExecutionProvider"],
}

def onnx_model_class(onnx_model_type: str):
    if onnx_model_type == SPEECH_LLM:
        from .onnx_speech_llm import SpeechLlm
        return SpeechLlm
    from onnx_asr.models.nemo import NemoConformerAED, NemoConformerCtc, NemoConformerRnnt, NemoConformerTdt
    return {
        "nemo-conformer-ctc": NemoConformerCtc,
        "nemo-conformer-rnnt": NemoConformerRnnt,
        "nemo-conformer-tdt": NemoConformerTdt,
        "nemo-conformer-aed": NemoConformerAED,
    }[onnx_model_type]


def onnx_model_file_patterns(onnx_model_type: str, quantization: Optional[str]) -> list:
    return ["config.json", *[pattern for group in onnx_required_file_patterns(onnx_model_type, quantization) for pattern in group]]


def onnx_required_file_patterns(onnx_model_type: str, quantization: Optional[str]) -> list:
    groups = []
    for pattern in onnx_model_class(onnx_model_type)._get_model_files(quantization).values():
        group = [pattern, pattern.removeprefix("**/")]
        if pattern.endswith(".onnx"):
            group += [variant[:-len(".onnx")] + ".onnx?data" for variant in list(group)]
        groups.append(group)
    return groups


def matches_any_pattern(file_path: str, patterns: list) -> bool:
    return any(fnmatch.fnmatchcase(file_path, pattern) for pattern in patterns)


def available_onnx_providers() -> list:
    try:
        import onnxruntime
        return onnxruntime.get_available_providers()
    except Exception:
        return []


class OnnxAsrEngine:
    ENGINE_TYPE = "onnx_asr"

    def __init__(self,
                 model_key: str,
                 onnx_runtime: str = ONNX_CPU,
                 vad_manager=None,
                 model_registry=None):
        self.model_key = model_key
        self.onnx_runtime = onnx_runtime
        self.device = "cpu"
        self.vad_manager = vad_manager
        self.registry = model_registry
        self.model = None
        self.logger = logging.getLogger(__name__)
        self._loading_thread = None
        self._model_lock = threading.Lock()

        self._load_model()

    def _is_model_cached(self, model_key: str = None) -> bool:
        return self.registry.is_onnx_model_downloaded(model_key or self.model_key)

    def _create_model(self, model_key: str):
        from onnx_asr.loader import Manager
        from onnx_asr.onnx import update_onnx_providers
        from onnx_asr.resolver import Resolver

        definition = self.registry.get_model(model_key)
        if not definition or definition.engine != self.ENGINE_TYPE:
            raise ValueError(f"[{model_key}] is not an ONNX model")
        if not self._is_model_cached(model_key):
            raise FileNotFoundError(f"The [{model_key}] model is not downloaded yet")

        if self.onnx_runtime == ONNX_CUDA:
            import onnxruntime
            if hasattr(onnxruntime, "preload_dlls"):
                onnxruntime.preload_dlls()

        providers = PROVIDERS[self.onnx_runtime]
        if self.onnx_runtime == ONNX_DIRECTML and not definition.directml:
            print(f"   ℹ️ [{model_key}] runs faster on the CPU than with DirectML, using the CPU")
            providers = PROVIDERS[ONNX_CPU]
        missing = [provider for provider in providers if provider not in available_onnx_providers()]
        if missing:
            raise RuntimeError(f"ONNX Runtime has no {missing[0]}; reinstall the runtime from the Runtime menu")

        model_class = onnx_model_class(definition.onnx_model_type)
        resolver = Resolver(model_class, None, self.registry.get_onnx_model_dir(model_key), offline=True)
        manager = Manager(providers=providers)
        onnx_options = update_onnx_providers(manager.default_onnx_config, excluded_providers=model_class._get_excluded_providers())
        asr = model_class(resolver.resolve_model(quantization=definition.quantization), manager._create_preprocessor, onnx_options)
        device = "cpu" if providers[0] == "CPUExecutionProvider" else "gpu"
        return manager._create_asr_adapter(asr), definition.onnx_model_type, device

    def _chunk_seconds(self, model_key: str) -> float:
        definition = self.registry.get_model(model_key)
        if definition.max_chunk_seconds:
            return definition.max_chunk_seconds
        return SPEECH_LLM_MAX_CHUNK_SECONDS if definition.onnx_model_type == SPEECH_LLM else DEFAULT_MAX_CHUNK_SECONDS

    def _load_model(self):
        print(f"🧠 Loading ONNX model [{self.model_key}] on [{self.onnx_runtime}]...")
        model, model_type, device = self._create_model(self.model_key)
        with self._model_lock:
            self.model = model
            self._model_type = model_type
            self._max_chunk_seconds = self._chunk_seconds(self.model_key)
            self.device = device
        print(f"   ✓ ONNX model [{self.model_key}] ready")

    def unload(self):
        with self._model_lock:
            self.model = None
        gc.collect()

    def close(self):
        self.unload()

    def reload(self):
        self._load_model()

    def warm_up(self):
        self._recognize(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def _recognize_kwargs(self, requested_language: Optional[str]) -> dict:
        language_code = self.registry.transcription_language(self.model_key, requested_language)
        if self._model_type != SPEECH_LLM or not language_code:
            return {}
        language_name = QWEN_LANGUAGE_NAMES.get(language_code)
        if language_name is None:
            self.logger.warning(f"Language [{language_code}] is not supported by [{self.model_key}], using auto-detect")
            return {}
        return {"language": language_name}

    def _recognize(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        with self._model_lock:
            model = self.model
            if model is None:
                raise RuntimeError("No ONNX model loaded")
            kwargs = self._recognize_kwargs(language)
            texts = []
            for chunk in split_at_quiet_points(audio, self._max_chunk_seconds):
                text = model.recognize(chunk, sample_rate=SAMPLE_RATE, **kwargs).strip()
                if not text:
                    self._warn_if_speech_lost(chunk)
                texts.append(text)
            return " ".join(text for text in texts if text)

    def _warn_if_speech_lost(self, chunk: np.ndarray):
        if self.vad_manager and self.vad_manager.is_available() and self.vad_manager.check_audio_for_speech(chunk):
            self.logger.warning(f"[{self.model_key}] returned no text for a {len(chunk) / SAMPLE_RATE:.1f} s chunk that contains speech")

    def _load_model_async(self, new_model_key: str, progress_callback: Optional[Callable[[str], None]] = None):
        def _background_loader():
            try:
                if progress_callback:
                    progress_callback("Loading model...")
                if self.device != "cpu":
                    self.unload()
                model, model_type, device = self._create_model(new_model_key)
                with self._model_lock:
                    self.model = None
                gc.collect()
                with self._model_lock:
                    self.model = model
                    self._model_type = model_type
                    self._max_chunk_seconds = self._chunk_seconds(new_model_key)
                    self.model_key = new_model_key
                    self.device = device
                self.logger.info(f"ONNX model [{new_model_key}] loaded on [{self.onnx_runtime}]")
                if progress_callback:
                    progress_callback("Model ready!")
            except Exception as e:
                self.logger.error(f"Failed to load ONNX model [{new_model_key}]: {e}")
                if self.model is None:
                    try:
                        self._load_model()
                    except Exception as restore_error:
                        self.logger.error(f"Failed to restore ONNX model [{self.model_key}]: {restore_error}")
                if progress_callback:
                    progress_callback(f"Failed to load model: {e}")
            finally:
                self._loading_thread = None

        if self.is_loading():
            self.logger.warning("Model loading already in progress, ignoring new request")
            return
        self._loading_thread = threading.Thread(target=_background_loader, daemon=True)
        self._loading_thread.start()

    def is_loading(self) -> bool:
        return self._loading_thread is not None and self._loading_thread.is_alive()

    def transcribe_audio(self, audio_data: np.ndarray, language: Optional[str] = None) -> Optional[str]:
        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("No audio data to transcribe")
            return None

        start_time = time.time()
        text = self._recognize(np.ascontiguousarray(audio_data.flatten(), dtype=np.float32), language).strip()
        elapsed = time.time() - start_time
        print(f"   ✓ Transcription completed in {elapsed:.1f} seconds")
        self.logger.info(f"Transcription complete ({self.model_key}, {self.onnx_runtime}) - Time: {elapsed:.2f}s")
        return text or None

    def change_model(self, new_model_key: str, progress_callback: Optional[Callable[[str], None]] = None):
        if new_model_key == self.model_key:
            if progress_callback:
                progress_callback("Model already loaded")
            return
        self._load_model_async(new_model_key, progress_callback)
