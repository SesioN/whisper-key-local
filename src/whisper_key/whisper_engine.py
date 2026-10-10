import gc
import logging
import time
import threading
from typing import Optional, Callable

import numpy as np
from faster_whisper import WhisperModel

from .runtime_options import ONNX_ASR, WHISPER_CPP


class WhisperEngine:
    ENGINE_TYPE = "faster_whisper"

    def __init__(self,
                 model_key: str = "tiny",
                 device: str = "cpu",
                 compute_type: str = "int8",
                 language: str = None,
                 beam_size: int = 5,
                 initial_prompt: str = "",
                 hotwords: list = None,
                 vad_manager = None,
                 model_registry = None):

        self.model_key = model_key
        self.device = device
        self.compute_type = compute_type
        self.language = None if language == 'auto' else language
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt or None
        self.hotwords = ", ".join(hotwords) if hotwords else None
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.registry = model_registry
        self._prompt_skip_logged_models = set()

        self._loading_thread = None
        self._progress_callback = None
        self._warmed_model = None

        self.vad_manager = vad_manager

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
            device_label = "GPU" if self.device == "cuda" else "CPU"
            print(f"   ✓ Running on {device_label} with {self.compute_type} precision")

        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _create_model(self, model_key: str) -> WhisperModel:
        return WhisperModel(
            self._get_model_source(model_key),
            device=self.device,
            compute_type=self.compute_type
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
        if self._is_english_only(model_key):
            return "en"
        return self.language or "en"

    def _is_english_only(self, model_key: str) -> bool:
        if self.registry:
            return self.registry.is_english_only(model_key)
        return model_key.endswith(".en")

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

                if self.device != "cpu":
                    self.unload()
                    gc.collect()

                new_model = self._create_model(new_model_key)
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
                if self.model is None:
                    self._restore_model(old_model_key)
                if progress_callback:
                    progress_callback(f"Failed to load model: {e}")
            finally:
                self._loading_thread = None
                self._progress_callback = None

        if self._loading_thread and self._loading_thread.is_alive():
            self.logger.warning("Model loading already in progress, ignoring new request")
            return
        
        self._progress_callback = progress_callback
        self._loading_thread = threading.Thread(target=_background_loader, daemon=True)
        self._loading_thread.start()
    
    def _restore_model(self, model_key: str):
        try:
            gc.collect()
            model = self._create_model(model_key)
            self._warm_up_model(model, model_key)
            self.model_key = model_key
            self.model = model
            self.logger.info(f"Restored Whisper model [{model_key}]")
        except Exception as restore_error:
            self.logger.error(f"Failed to restore Whisper model [{model_key}]: {restore_error}")

    def is_loading(self) -> bool:
        return self._loading_thread is not None and self._loading_thread.is_alive()
    

    def transcribe_audio(self,
                         audio_data: np.ndarray) -> Optional[str]:
        if self.model is None:
            return None

        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("No audio data to transcribe")
            return None

        if self.vad_manager and self.vad_manager.is_available():
            if not self.vad_manager.check_audio_for_speech(audio_data):
                print("   ✗ No speech detected, skipping transcription")
                return None

        start_time = time.time()
        audio_data = np.ascontiguousarray(audio_data.reshape(-1), dtype=np.float32)

        transcribe_kwargs = dict(
            beam_size=self.beam_size,
            language="en" if self._is_english_only(self.model_key) else self.language,
            condition_on_previous_text=False,
        )
        if self._model_supports_prompt():
            if self.initial_prompt:
                transcribe_kwargs["initial_prompt"] = self.initial_prompt
            if self.hotwords:
                transcribe_kwargs["hotwords"] = self.hotwords

        segments, info = self.model.transcribe(audio_data, **transcribe_kwargs)
        transcribed_text = "".join(segment.text for segment in segments).strip()

        transcription_time = time.time() - start_time
        print(f"   ✓ Transcription completed in {transcription_time:.1f} seconds")
        self.logger.info(f"Transcription complete. Language: {info.language} (confidence: {info.language_probability:.2f}) - Time: {transcription_time:.2f}s")

        if transcribed_text:
            return transcribed_text
        self.logger.info("Transcription was empty")
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
            language=whisper_config['language'],
            vad_manager=vad_manager,
            model_registry=model_registry,
        )

    if engine_type == WHISPER_CPP:
        from .whisper_cpp_engine import WhisperCppEngine
        return WhisperCppEngine(
            model_key=whisper_config['model'],
            language=whisper_config['language'],
            beam_size=whisper_config['beam_size'],
            initial_prompt=whisper_config.get('initial_prompt', ''),
            hotwords=whisper_config.get('hotwords', []),
            vad_manager=vad_manager,
            model_registry=model_registry,
            binary_path=whisper_config.get('cpp_binary'),
            model_dir=whisper_config.get('cpp_model_dir'),
        )

    return WhisperEngine(
        model_key=whisper_config['model'],
        device=whisper_config['device'],
        compute_type=whisper_config['compute_type'],
        language=whisper_config['language'],
        beam_size=whisper_config['beam_size'],
        initial_prompt=whisper_config.get('initial_prompt', ''),
        hotwords=whisper_config.get('hotwords', []),
        vad_manager=vad_manager,
        model_registry=model_registry,
    )
