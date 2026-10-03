import logging
import os
import shutil
import re
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from .audio_recorder import AudioRecorder


def _find_whisper_cli():
    env = os.environ.get("WHISPER_CPP_BINARY", "")
    if env and os.path.isfile(env):
        return env
    candidates = [
        Path(__file__).parents[2] / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli.exe",
        Path.home() / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli.exe",
    ]
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    which = shutil.which("whisper-cli")
    if which:
        return which
    which_exe = shutil.which("whisper-cli.exe")
    if which_exe:
        return which_exe
    return None


def _find_model_dir():
    env = os.environ.get("WHISPER_CPP_MODEL_DIR", "")
    if env:
        p = Path(env)
        if p.is_dir():
            return str(p)
    candidates = [
        Path.home() / "tools" / "whisper.cpp",
        Path.home() / "tools" / "whisper.cpp" / "models",
        Path(__file__).parents[2] / "tools" / "whisper.cpp",
    ]
    for p in candidates:
        if p.is_dir():
            return str(p)
    return None


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NON_SPEECH_TAG = re.compile(r"\[[A-Z_ ]+\]|\[(?:[Mm]usic|[Ll]aughter|[Aa]pplause|[Ss]ilence)\]|\([^()]*\)")


class WhisperCppEngine:

    def __init__(self,
                 model_key: str = "base",
                 device: str = "cpu",
                 compute_type: str = "int8",
                 language: str = None,
                 beam_size: int = 5,
                 initial_prompt: str = "",
                 hotwords: list = None,
                 vad_manager=None,
                 model_registry=None,
                 binary_path: str = None,
                 model_dir: str = None):

        self.model_key = model_key
        self.language = None if language == 'auto' else language
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt or None
        self.hotwords = ", ".join(hotwords) if hotwords else None
        self.logger = logging.getLogger(__name__)
        if self.hotwords:
            self.logger.warning("hotwords are not supported by whisper.cpp and will be ignored")
        self.vad_manager = vad_manager
        self.registry = model_registry

        self._binary = binary_path or _find_whisper_cli()
        self._model_dir = model_dir or _find_model_dir()

        self._loading_thread = None
        self._progress_callback = None

        if not self._binary:
            raise RuntimeError(
                "whisper-cli not found. Set WHISPER_CPP_BINARY env var "
                "or install whisper.cpp (https://github.com/ggerganov/whisper.cpp)"
            )
        if not os.path.isfile(self._binary):
            raise RuntimeError(f"whisper-cli not found at: {self._binary}")

        self.logger.info("WhisperCppEngine: binary=%s model_dir=%s", self._binary, self._model_dir)
        self._load_model()

    def _get_model_path(self, model_key: str = None):
        key = model_key or self.model_key
        filename = f"ggml-{key}.bin"

        if self._model_dir:
            candidate = os.path.join(self._model_dir, filename)
            if os.path.isfile(candidate):
                return candidate

        if self.registry:
            source = self.registry.get_source(key)
            if source and os.path.isfile(source):
                return source

        return None

    def _is_model_cached(self, model_key: str = None):
        return self._get_model_path(model_key) is not None

    def _load_model(self):
        model_path = self._get_model_path()
        if not model_path:
            raise FileNotFoundError(
                f"Model file not found: ggml-{self.model_key}.bin in {self._model_dir}\n"
                f"Download models with: cd whisper.cpp && ./models/download-ggml-model.cmd {self.model_key}"
            )
        self.logger.info("whisper.cpp model [%s] at %s", self.model_key, model_path)
        print(f"   ✓ whisper.cpp model [{self.model_key}] ready")

    def _load_model_async(self,
                          new_model_key: str,
                          progress_callback: Optional[Callable[[str], None]] = None):

        def _background_loader():
            try:
                if progress_callback:
                    progress_callback("Checking model cache...")

                if not self._is_model_cached(new_model_key):
                    if progress_callback:
                        progress_callback(f"Model '{new_model_key}' not found. Download it first.")
                    raise FileNotFoundError(f"Model '{new_model_key}' not available")

                if progress_callback:
                    progress_callback("Loading model...")

                old_model_key = self.model_key
                self.model_key = new_model_key
                self.logger.info("WhisperCpp model changed: %s -> %s", old_model_key, new_model_key)

                if progress_callback:
                    progress_callback("Model ready!")
            except Exception as e:
                self.logger.error("Failed to change model: %s", e)
                if progress_callback:
                    progress_callback(f"Failed: {e}")

        if self._loading_thread and self._loading_thread.is_alive():
            self.logger.warning("Model loading already in progress, ignoring new request")
            return

        def _run_and_clear():
            try:
                _background_loader()
            finally:
                self._loading_thread = None

        self._progress_callback = progress_callback
        self._loading_thread = threading.Thread(target=_run_and_clear, daemon=True)
        self._loading_thread.start()

    def is_loading(self) -> bool:
        return self._loading_thread is not None and self._loading_thread.is_alive()

    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("No audio data to transcribe")
            return None

        if self.vad_manager and self.vad_manager.is_available():
            speech_detected = self.vad_manager.check_audio_for_speech(audio_data)
            if not speech_detected:
                print("   X No speech detected, skipping transcription")
                return None

        start_time = time.time()
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="whisperkey_")
            os.close(tmp_fd)
            self._write_wav(tmp_path, audio_data)

            model_path = self._get_model_path()
            if not model_path:
                self.logger.error("whisper.cpp model [%s] not found", self.model_key)
                return None

            cmd = self._build_command(model_path, tmp_path)
            self.logger.debug("Running: %s", " ".join(cmd))
            audio_seconds = len(audio_data) / AudioRecorder.WHISPER_SAMPLE_RATE
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(120, audio_seconds * 4),
                creationflags=_NO_WINDOW,
            )

            if result.returncode != 0:
                stderr_tail = "\n".join(result.stderr.strip().splitlines()[-20:])
                self.logger.error("whisper-cli exited with code %s:\n%s", result.returncode, stderr_tail)
                print(f"   ✗ whisper.cpp failed (exit code {result.returncode}), see log for details")
                return None

            transcribed = self._clean_output(result.stdout)

            elapsed = time.time() - start_time
            self.logger.info(f"Transcription completed in {elapsed:.2f}s")

            if transcribed:
                return transcribed

            self.logger.info("Transcription was empty")
            return None

        except subprocess.TimeoutExpired as e:
            self.logger.error("Transcription timed out after %.0fs", e.timeout)
            print(f"   ✗ whisper.cpp timed out after {e.timeout:.0f}s")
            return None
        except Exception as e:
            self.logger.error("Transcription failed: %s", e)
            return None
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def change_model(self, new_model_key: str, progress_callback=None):
        if new_model_key == self.model_key:
            if progress_callback:
                progress_callback("Model already loaded")
            return
        self._load_model_async(new_model_key, progress_callback)

    def _build_command(self, model_path: str, audio_path: str) -> list:
        cmd = [
            self._binary,
            "-m", model_path,
            "-f", audio_path,
            "-nt",
            "-l", self.language or "auto",
        ]
        if self.beam_size:
            cmd.extend(["-bs", str(self.beam_size)])
        if self.initial_prompt:
            cmd.extend(["--prompt", self.initial_prompt])
        return cmd

    @staticmethod
    def _clean_output(stdout: str) -> str:
        text = " ".join(line.strip() for line in stdout.splitlines() if line.strip())
        text = _NON_SPEECH_TAG.sub(" ", text)
        return " ".join(text.split())

    @staticmethod
    def _write_wav(path: str, audio_data: np.ndarray):
        if len(audio_data.shape) > 1:
            audio_data = audio_data.ravel()
        if np.issubdtype(audio_data.dtype, np.floating):
            audio_data = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
            audio_data = audio_data.clip(-32768, 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(AudioRecorder.WHISPER_SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
