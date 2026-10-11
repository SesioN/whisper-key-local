import gc
import logging
import os
import time
import threading
from typing import Optional, Callable

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model

from .audio_chunking import SAMPLE_RATE, split_at_quiet_points
from .runtime_options import ONNX_ASR, WHISPER_CPP, resolve_cpu_threads

WHISPER_WINDOW_SECONDS = 30
PROMPTED_CHUNK_SECONDS = 28
MIN_LANGUAGE_PROBABILITY_TO_REUSE = 0.5


class WhisperEngine:
    ENGINE_TYPE = "faster_whisper"

    def __init__(self,
                 model_key: str = "tiny",
                 device: str = "cpu",
                 compute_type: str = "int8",
                 beam_size: int = 5,
                 initial_prompt: str = "",
                 hotwords: list = None,
                 model_registry = None,
                 cpu_threads: int = 0):

        self.model_key = model_key
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt or None
        self.hotwords = ", ".join(hotwords) if hotwords else None
        self.cpu_threads = resolve_cpu_threads(cpu_threads)
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.registry = model_registry
        self._prompt_skip_logged_models = set()

        self._loading_thread = None
        self._progress_callback = None
        self._warmed_model = None

        self._load_model()
    
    def _model_supports_prompt(self) -> bool:
        if not self.registry or self.registry.supports_prompt(self.model_key):
            return True
        if (self.initial_prompt or self.hotwords) and self.model_key not in self._prompt_skip_logged_models:
            self._prompt_skip_logged_models.add(self.model_key)
            self.logger.info(f"Model {self.model_key} has supports_prompt: false, skipping initial_prompt and hotwords")
        return False

    def _get_model_source(self, model_key: str) -> str:
        if self.registry:
            return self.registry.get_load_source(model_key)
        return model_key

    def _is_model_cached(self, model_key: str = None) -> bool:
        if model_key is None:
            model_key = self.model_key
        if self.registry:
            return self.registry.is_model_cached(model_key)
        return False

    def _load_model(self):
        try:
            print(f"🧠 Loading Whisper AI model [{self.model_key}]...")

            was_cached = self._is_model_cached()
            if not was_cached:
                print("Downloading model, this may take a few minutes....")

            model = self._create_model(self.model_key)
            self._warm_up_model(model, self.model_key)
            self.model = model

            if not was_cached:
                print("\n")  # Workaround for download status bar misplacement

            print(f"   ✓ Whisper model [{self.model_key}] ready!")
            device_label = "GPU" if self.device == "cuda" else f"CPU ({self.cpu_threads} threads)"
            print(f"   ✓ Running on {device_label} with {self.compute_type} precision")

        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _create_model(self, model_key: str, model_source: Optional[str] = None) -> WhisperModel:
        return WhisperModel(
            model_source or self._get_model_source(model_key),
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads
        )

    def unload(self):
        self.model = None
        self._warmed_model = None

    def close(self):
        pass

    def reload(self):
        self._load_model()

    def warm_up(self):
        if self.model is not None and self._warmed_model is not self.model:
            self._warm_up_model(self.model, self.model_key)

    def _warm_up_model(self, model, model_key: str):
        segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1, language=self._warm_up_language(model_key))
        list(segments)
        self._warmed_model = model

    def _warm_up_language(self, model_key: str) -> str:
        return self._transcription_language(model_key, None) or "en"

    def _transcription_language(self, model_key: str, requested_language: Optional[str]) -> Optional[str]:
        if self.registry:
            return self.registry.transcription_language(model_key, requested_language)
        return "en" if model_key.endswith(".en") else requested_language

    def _load_model_async(self,
                          new_model_key: str,
                          progress_callback: Optional[Callable[[str], None]] = None):
        def _background_loader():
            old_model_key = self.model_key
            try:
                if progress_callback:
                    progress_callback("Checking model cache...")

                was_cached = self._is_model_cached(new_model_key)

                if progress_callback:
                    if was_cached:
                        progress_callback("Loading cached model...")
                    else:
                        progress_callback("Downloading model...")

                self.logger.info(f"Loading Whisper model: {new_model_key} (async)")

                model_source = self._get_model_source(new_model_key)
                if self.device != "cpu":
                    model_source = self._download_before_release(model_source)
                    self.unload()
                    gc.collect()

                new_model = self._create_model(new_model_key, model_source)
                if progress_callback:
                    progress_callback("Warming up model...")
                self._warm_up_model(new_model, new_model_key)
                self.model_key = new_model_key
                self.model = new_model
                self.logger.info(f"Whisper model [{new_model_key}] loaded successfully (async)")

                if progress_callback:
                    progress_callback("Model ready!")

            except Exception as e:
                self.logger.error(f"Failed to load Whisper model async: {e}")
                restored = self.model is not None or self._restore_model(old_model_key)
                if progress_callback:
                    if restored:
                        progress_callback(f"Failed to load model: {e}")
                    else:
                        progress_callback(f"Failed to load model, and the previous model could not be reloaded either (no model loaded): {e}")
            finally:
                self._loading_thread = None
                self._progress_callback = None

        if self._loading_thread and self._loading_thread.is_alive():
            self.logger.warning("Model loading already in progress, ignoring new request")
            return
        
        self._progress_callback = progress_callback
        self._loading_thread = threading.Thread(target=_background_loader, daemon=True)
        self._loading_thread.start()
    
    def _download_before_release(self, model_source: str) -> str:
        if os.path.isdir(model_source):
            return model_source
        return download_model(model_source)

    def _restore_model(self, model_key: str) -> bool:
        try:
            gc.collect()
            model = self._create_model(model_key)
            self._warm_up_model(model, model_key)
            self.model_key = model_key
            self.model = model
            self.logger.info(f"Restored Whisper model [{model_key}]")
            return True
        except Exception as restore_error:
            self.logger.error(f"Failed to restore Whisper model [{model_key}]: {restore_error}")
            print(f"❌ No transcription model is loaded: {restore_error}")
            return False

    def is_loading(self) -> bool:
        return self._loading_thread is not None and self._loading_thread.is_alive()
    

    def transcribe_audio(self,
                         audio_data: np.ndarray,
                         language: Optional[str] = None) -> Optional[str]:
        model = self.model
        if model is None:
            return None

        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("No audio data to transcribe")
            return None

        start_time = time.time()
        audio_data = np.ascontiguousarray(audio_data.reshape(-1), dtype=np.float32)

        transcribe_kwargs = dict(
            beam_size=self.beam_size,
            language=self._transcription_language(self.model_key, language),
            condition_on_previous_text=False,
        )
        if self._model_supports_prompt():
            if self.initial_prompt:
                transcribe_kwargs["initial_prompt"] = self.initial_prompt
            if self.hotwords:
                transcribe_kwargs["hotwords"] = self.hotwords

        transcribed_text, info = self._transcribe_chunks(model, self._split_for_prompt_carryover(audio_data, transcribe_kwargs), transcribe_kwargs)

        transcription_time = time.time() - start_time
        print(f"   ✓ Transcription completed in {transcription_time:.1f} seconds")
        self.logger.info(f"Transcription complete. Language: {info.language} (confidence: {info.language_probability:.2f}) - Time: {transcription_time:.2f}s")

        if transcribed_text:
            return transcribed_text
        self.logger.info("Transcription was empty")
        return None

    @staticmethod
    def _fits_single_window(audio_data: np.ndarray) -> bool:
        return len(audio_data) <= WHISPER_WINDOW_SECONDS * SAMPLE_RATE

    def _split_for_prompt_carryover(self, audio_data: np.ndarray, transcribe_kwargs: dict) -> list:
        if transcribe_kwargs.get("initial_prompt") and not self._fits_single_window(audio_data):
            return split_at_quiet_points(audio_data, PROMPTED_CHUNK_SECONDS)
        return [audio_data]

    def _transcribe_chunks(self, model, chunks: list, transcribe_kwargs: dict) -> tuple:
        texts = []
        first_info = None
        for chunk in chunks:
            segments, info = model.transcribe(chunk, without_timestamps=self._fits_single_window(chunk), **transcribe_kwargs)
            texts.append("".join(segment.text for segment in segments).strip())
            if first_info is None:
                first_info = info
                transcribe_kwargs = {**transcribe_kwargs, "language": self._language_for_remaining_chunks(transcribe_kwargs["language"], info)}
        return " ".join(text for text in texts if text), first_info

    def _language_for_remaining_chunks(self, configured_language: Optional[str], first_chunk_info) -> Optional[str]:
        if configured_language:
            return configured_language
        if first_chunk_info.language_probability >= MIN_LANGUAGE_PROBABILITY_TO_REUSE:
            self.logger.info(f"Using detected language [{first_chunk_info.language}] for the remaining chunks")
            return first_chunk_info.language
        self.logger.info(f"Detected language [{first_chunk_info.language}] is uncertain ({first_chunk_info.language_probability:.2f}), detecting per chunk")
        return None

    def change_model(self,
                     new_model_key: str,
                     progress_callback: Optional[Callable[[str], None]] = None):
        
        if new_model_key == self.model_key:
            if progress_callback:
                progress_callback("Model already loaded")
            return
        
        self._load_model_async(new_model_key, progress_callback)


def create_whisper_engine(engine_type: str, whisper_config: dict, vad_manager, model_registry):

    if engine_type == ONNX_ASR:
        from .onnx_asr_engine import OnnxAsrEngine
        return OnnxAsrEngine(
            model_key=whisper_config['model'],
            onnx_runtime=whisper_config['onnx_runtime'],
            vad_manager=vad_manager,
            model_registry=model_registry,
        )

    if engine_type == WHISPER_CPP:
        from .whisper_cpp_engine import WhisperCppEngine
        return WhisperCppEngine(
            model_key=whisper_config['model'],
            beam_size=whisper_config['beam_size'],
            initial_prompt=whisper_config.get('initial_prompt', ''),
            hotwords=whisper_config.get('hotwords', []),
            model_registry=model_registry,
            binary_path=whisper_config.get('cpp_binary'),
            model_dir=whisper_config.get('cpp_model_dir'),
            cpu_threads=whisper_config.get('cpu_threads', 0),
        )

    return WhisperEngine(
        model_key=whisper_config['model'],
        device=whisper_config['device'],
        compute_type=whisper_config['compute_type'],
        beam_size=whisper_config['beam_size'],
        initial_prompt=whisper_config.get('initial_prompt', ''),
        hotwords=whisper_config.get('hotwords', []),
        model_registry=model_registry,
        cpu_threads=whisper_config.get('cpu_threads', 0),
    )
