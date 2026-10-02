import io
import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from .platform import child_processes

SERVER_HOST = "127.0.0.1"
SERVER_STARTUP_TIMEOUT_SECONDS = 120
SERVER_REQUEST_TIMEOUT_SECONDS = 120
SERVER_POLL_INTERVAL_SECONDS = 0.2
SERVER_STOP_TIMEOUT_SECONDS = 5
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


def find_whisper_server(cli_binary: str) -> Optional[str]:
    for name in ("whisper-server.exe", "whisper-server"):
        candidate = Path(cli_binary).with_name(name)
        if candidate.is_file():
            return str(candidate)
    return None


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((SERVER_HOST, 0))
        return probe.getsockname()[1]


def _multipart_body(fields: dict, file_field: str, file_name: str, file_bytes: bytes):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8"))
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{file_name}"\r\n'
        f'Content-Type: audio/wav\r\n\r\n'.encode("utf-8") + file_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def find_whisper_cli():
    env = os.environ.get("WHISPER_CPP_BINARY", "")
    if env and os.path.isfile(env):
        return env
    candidates = [
        Path(__file__).parent.parent.parent.parent / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli.exe",
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
        Path(__file__).parent.parent.parent.parent / "tools" / "whisper.cpp",
    ]
    for p in candidates:
        if p.is_dir():
            return str(p)
    return None


_MODEL_FILE_MAP = {
    "tiny": "ggml-tiny.bin",
    "tiny.en": "ggml-tiny.en.bin",
    "base": "ggml-base.bin",
    "base.en": "ggml-base.en.bin",
    "small": "ggml-small.bin",
    "small.en": "ggml-small.en.bin",
    "medium": "ggml-medium.bin",
    "medium.en": "ggml-medium.en.bin",
    "large": "ggml-large.bin",
    "large-v3": "ggml-large-v3.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
}


class WhisperCppEngine:
    ENGINE_TYPE = "whisper_cpp"

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
        self.vad_manager = vad_manager
        self.registry = model_registry
        self.logger = logging.getLogger(__name__)

        self._binary = binary_path or find_whisper_cli()
        self._model_dir = model_dir or _find_model_dir()

        self._loading_thread = None
        self._progress_callback = None

        if not self._binary:
            raise RuntimeError(
                "whisper-cli not found. Set WHISPER_CPP_BINARY env var "
                "or install whisper.cpp (https://github.com/ggerganov/whisper.cpp)"
            )

        self.logger.info("WhisperCppEngine: binary=%s model_dir=%s", self._binary, self._model_dir)
        self._load_model()

        self._server_process = None
        self._server_url = None
        self._server_lock = threading.Lock()
        self._closed = False
        self._start_server()

    def _prompt(self) -> Optional[str]:
        return " ".join(filter(None, [self.initial_prompt, self.hotwords])) or None

    def _decoding_arguments(self, include_prompt: bool = True) -> list:
        arguments = ["-l", self.language or "auto"]
        if self.beam_size:
            arguments.extend(["-bs", str(self.beam_size)])
        if include_prompt and self._prompt():
            arguments.extend(["--prompt", self._prompt()])
        return arguments

    def _start_server(self):
        server_binary = find_whisper_server(self._binary)
        if not server_binary:
            self.logger.info("whisper-server not found, using whisper-cli per transcription")
            return

        port = _free_local_port()
        request_path = f"/{secrets.token_urlsafe(16)}"
        command = [server_binary, "-m", self._get_model_path(), "--host", SERVER_HOST, "--port", str(port),
                   "--request-path", request_path, "-nt", *self._decoding_arguments(include_prompt=False)]
        self.logger.info("Starting whisper-server: %s", " ".join(command))
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **NO_WINDOW)
        child_processes.tie_to_current_process(process)

        deadline = time.time() + SERVER_STARTUP_TIMEOUT_SECONDS
        while time.time() < deadline:
            if process.poll() is not None:
                self.logger.warning(f"whisper-server exited with code {process.returncode}, using whisper-cli instead")
                return
            try:
                with socket.create_connection((SERVER_HOST, port), timeout=1):
                    pass
                if process.poll() is None:
                    break
            except OSError:
                time.sleep(SERVER_POLL_INTERVAL_SECONDS)
        else:
            self.logger.warning("whisper-server did not start in time, using whisper-cli instead")
            process.kill()
            return

        with self._server_lock:
            if self._closed:
                process.kill()
                return
            self._server_process = process
            self._server_url = f"http://{SERVER_HOST}:{port}{request_path}/inference"
        print(f"   ✓ whisper.cpp model kept loaded (whisper-server on port {port})")

    def close(self):
        with self._server_lock:
            self._closed = True
            process = self._server_process
            self._server_process = None
            self._server_url = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=SERVER_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()

    def _stop_server(self):
        with self._server_lock:
            process = self._server_process
            self._server_process = None
            self._server_url = None
        if process and process.poll() is None:
            process.kill()

    def _restart_server(self):
        self._stop_server()
        self._start_server()

    def _transcribe_with_server(self, wav_bytes: bytes) -> Optional[str]:
        with self._server_lock:
            server_url = self._server_url
        if not server_url:
            return None
        fields = {"temperature": "0.0", "response_format": "json"}
        if self._prompt():
            fields["prompt"] = self._prompt()
        body, content_type = _multipart_body(fields, "file", "audio.wav", wav_bytes)
        request = urllib.request.Request(server_url, data=body, headers={"Content-Type": content_type}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=SERVER_REQUEST_TIMEOUT_SECONDS) as response:
                result = json.load(response)
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.logger.warning(f"whisper-server stopped responding, using whisper-cli from now on: {e}")
            self._stop_server()
            return None
        if "error" in result or "text" not in result:
            self.logger.warning(f"whisper-server returned an error, falling back to whisper-cli: {result.get('error', result)}")
            return None
        return result["text"].strip()

    def _get_model_path(self, model_key: str = None):
        key = model_key or self.model_key
        filename = _MODEL_FILE_MAP.get(key, f"ggml-{key}.bin")

        if self._model_dir:
            candidate = os.path.join(self._model_dir, filename)
            if os.path.isfile(candidate):
                return candidate

        if self.registry:
            source = self.registry.get_source(key)
            if source and os.path.isfile(source):
                return source

        return filename

    def _is_model_cached(self, model_key: str = None):
        path = self._get_model_path(model_key)
        return os.path.isfile(path)

    def _load_model(self):
        model_path = self._get_model_path()
        print(f"[WhisperCpp] Loading model [{self.model_key}]...")
        if not os.path.isfile(model_path):
            print(f"[!] Model file not found: {model_path}")
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Download models with: cd whisper.cpp && ./models/download-ggml-model.cmd {self.model_key}"
            )
        print(f"   OK WhisperCpp model [{self.model_key}] ready at {model_path}")
        print(f"   OK Binary: {self._binary}")

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
                if self._server_url:
                    self._restart_server()
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

        self._progress_callback = progress_callback
        self._loading_thread = threading.Thread(target=_background_loader, daemon=True)
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
        wav_bytes = self._wav_bytes(audio_data)
        server_text = self._transcribe_with_server(wav_bytes)
        if server_text is not None:
            self.logger.info(f"Transcription completed in {time.time() - start_time:.2f}s (whisper-server)")
            return server_text or None

        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="whisperkey_")
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                tmp_file.write(wav_bytes)

            model_path = self._get_model_path()
            cmd = [
                self._binary,
                "-m", model_path,
                "-f", tmp_path,
                "-nt",
                *self._decoding_arguments(),
            ]

            self.logger.info("Running: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )

            stderr = result.stderr.strip()
            if stderr:
                for line in stderr.splitlines():
                    if "ggml_vulkan" in line.lower() or "vulkan" in line.lower() or "amd" in line.lower() or "radeon" in line.lower():
                        self.logger.debug("Vulkan: %s", line)

            transcribed = result.stdout.strip()

            if transcribed and transcribed.startswith("["):
                lines = transcribed.splitlines()
                text_lines = []
                for line in lines:
                    if "]   " in line:
                        text_lines.append(line.split("]   ", 1)[1])
                    elif line.startswith("[") and "-->" in line:
                        continue
                    elif line.strip() and not line.startswith("whisper_"):
                        text_lines.append(line)
                transcribed = " ".join(text_lines).strip()

            elapsed = time.time() - start_time
            self.logger.info(f"Transcription completed in {elapsed:.2f}s")

            if transcribed:
                return transcribed

            self.logger.info("Transcription was empty")
            return None

        except subprocess.TimeoutExpired:
            self.logger.error("Transcription timed out")
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

    @staticmethod
    def _wav_bytes(audio_data: np.ndarray) -> bytes:
        if len(audio_data.shape) > 1:
            audio_data = audio_data.ravel()
        if audio_data.dtype == np.float32:
            audio_data = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
            audio_data = audio_data.astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data.tobytes())
        return buffer.getvalue()
