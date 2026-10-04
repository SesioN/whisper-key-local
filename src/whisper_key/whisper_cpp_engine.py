import io
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections import deque
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from .audio_recorder import AudioRecorder
from .platform import child_processes
from .utils import get_user_app_data_path

SERVER_HOST = "127.0.0.1"
SERVER_STARTUP_TIMEOUT_SECONDS = 60
WARM_UP_TIMEOUT_SECONDS = 120
SERVER_POLL_INTERVAL_SECONDS = 0.2
SERVER_STOP_TIMEOUT_SECONDS = 5
SERVER_MAX_RESTARTS = 2
SERVER_LOG_FILE = "whisper-server.log"
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


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


def _log_tail(path: str, lines: int = 20) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as log_file:
            return "".join(deque(log_file, maxlen=lines)).strip()
    except OSError:
        return ""


def ggml_file_name(model_key: str) -> str:
    from .runtime_installer import RuntimeInstallError, ggml_model_file_name
    try:
        return ggml_model_file_name(model_key)
    except RuntimeInstallError:
        return f"ggml-{model_key}.bin"


def find_whisper_cli():
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
_NON_SPEECH_TAG = re.compile(r"\[[A-Z]+(?:_[A-Z]+)+\]|[\[(](?:music|laughs|laughter|applause|silence|inaudible|no speech)[\])]", re.IGNORECASE)


class WhisperCppEngine:
    ENGINE_TYPE = "whisper_cpp"
    device = "vulkan"

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

        self._binary = binary_path or find_whisper_cli()
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

        self._server_process = None
        self._server_url = None
        self._server_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._server_enabled = True
        self._closed = False
        self._restarts = 0
        self._server_log = os.path.join(get_user_app_data_path(), SERVER_LOG_FILE)
        self._restart_server(self.model_key)

    def _start_server(self, model_key: str) -> bool:
        server_binary = find_whisper_server(self._binary)
        if not server_binary:
            self.logger.warning("whisper-server not found next to %s, using whisper-cli per transcription "
                                "(reinstall the Vulkan runtime to keep the model loaded)", self._binary)
            return False

        model_path = self._get_model_path(model_key)
        if not model_path:
            self.logger.warning("whisper.cpp model [%s] not found, not starting whisper-server", model_key)
            return False
        port = _free_local_port()
        request_path = f"/{secrets.token_urlsafe(16)}"
        command = [server_binary, "-m", model_path, "--host", SERVER_HOST, "--port", str(port),
                   "--request-path", request_path, "-nt", "-l", self.language or "auto"]
        if self.beam_size:
            command.extend(["-bs", str(self.beam_size)])
        print("   Starting whisper.cpp server...", flush=True)
        self.logger.info("Starting whisper-server: %s", " ".join(command).replace(request_path, "/***"))
        with self._server_lock:
            if self._closed or not self._server_enabled:
                return False
            with open(self._server_log, "wb") as log_file:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log_file,
                                           stderr=subprocess.STDOUT, creationflags=_NO_WINDOW)
            self._server_process = process
        child_processes.tie_to_current_process(process)

        health_url = f"http://{SERVER_HOST}:{port}{request_path}/health"
        deadline = time.time() + SERVER_STARTUP_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self._server_process is not process:
                return False
            if process.poll() is not None:
                self.logger.warning("whisper-server exited with code %s, using whisper-cli instead:\n%s",
                                    process.returncode, _log_tail(self._server_log))
                self._stop_server(process)
                return False
            try:
                with _LOCAL_OPENER.open(health_url, timeout=1) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(SERVER_POLL_INTERVAL_SECONDS)
        else:
            self.logger.warning("whisper-server did not start in time, using whisper-cli instead:\n%s",
                                _log_tail(self._server_log))
            print(f"   ⚠ whisper-server did not start within {SERVER_STARTUP_TIMEOUT_SECONDS}s, using whisper-cli (slower), see {self._server_log}")
            self._stop_server(process)
            return False

        with self._server_lock:
            if self._server_process is not process:
                return False
            self._server_url = f"http://{SERVER_HOST}:{port}{request_path}/inference"
        print(f"   ✓ whisper.cpp model kept loaded (whisper-server on port {port})")
        return True

    def _stop_server(self, only_process=None):
        with self._server_lock:
            process = self._server_process
            if only_process is not None and process is not only_process:
                return
            self._server_process = None
            self._server_url = None
        if not process:
            return
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=SERVER_STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _restart_server(self, model_key: str) -> bool:
        with self._lifecycle_lock:
            self._stop_server()
            return self._start_server(model_key)

    def _recover_server(self, process):
        with self._lifecycle_lock:
            with self._server_lock:
                if self._server_process is not process or self._restarts >= SERVER_MAX_RESTARTS:
                    return
                self._restarts += 1
            self._stop_server(process)
            if self._start_server(self.model_key) or not self._server_enabled:
                return
        print("   ⚠ whisper-server could not be restarted, using whisper-cli (slower)")

    def _transcribe_with_server(self, wav_bytes: bytes, timeout: float) -> Optional[str]:
        with self._server_lock:
            server_url = self._server_url
            process = self._server_process
        if not server_url:
            return None
        fields = {"response_format": "json", "language": self.language or "auto"}
        if self.initial_prompt:
            fields["prompt"] = self.initial_prompt
        body, content_type = _multipart_body(fields, "file", "audio.wav", wav_bytes)
        request = urllib.request.Request(server_url, data=body, headers={"Content-Type": content_type}, method="POST")
        try:
            with _LOCAL_OPENER.open(request, timeout=timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as e:
            self.logger.warning("whisper-server returned HTTP %s, using whisper-cli for this recording", e.code)
            return None
        except (urllib.error.URLError, OSError, ValueError) as e:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.logger.warning("whisper-server request failed, using whisper-cli for this recording: %s", e)
                return None
            if self._server_process is not process:
                return None
            self.logger.warning("whisper-server exited with code %s:\n%s", process.returncode,
                                _log_tail(self._server_log))
            print("   ⚠ whisper-server stopped, restarting it; using whisper-cli for this recording")
            threading.Thread(target=self._recover_server, args=(process,), daemon=True).start()
            return None
        if not isinstance(result, dict) or "text" not in result:
            self.logger.warning("whisper-server returned an error, using whisper-cli for this recording: %s", result)
            return None
        with self._server_lock:
            self._restarts = 0
        return self._clean_output(result["text"])

    def _get_model_path(self, model_key: str = None):
        key = model_key or self.model_key
        filename = ggml_file_name(key)

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
                f"Model file not found: ggml-{self.model_key}.bin in {self._model_dir or '(no model dir found)'}\n"
                f"Download models with: cd whisper.cpp && ./models/download-ggml-model.cmd {self.model_key}"
            )
        self.logger.info("whisper.cpp model [%s] at %s", self.model_key, model_path)
        print(f"   ✓ whisper.cpp model [{self.model_key}] ready")

    def unload(self):
        with self._server_lock:
            self._server_enabled = False
        self._stop_server()

    def reload(self):
        self._load_model()
        with self._server_lock:
            self._server_enabled = True
            self._restarts = 0
        self._restart_server(self.model_key)

    def close(self):
        with self._server_lock:
            self._closed = True
        self.unload()

    def warm_up(self):
        if self._server_url:
            silence = np.zeros(AudioRecorder.WHISPER_SAMPLE_RATE, dtype=np.float32)
            self._transcribe_with_server(self._wav_bytes(silence), WARM_UP_TIMEOUT_SECONDS)

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
                with self._server_lock:
                    restart = self._server_enabled and not self._closed
                    self._restarts = 0
                server_started = self._restart_server(new_model_key) if restart else True
                self.model_key = new_model_key
                self.logger.info("WhisperCpp model changed: %s -> %s", old_model_key, new_model_key)

                if progress_callback:
                    progress_callback("Model ready!" if server_started else "Model ready! (whisper-server unavailable, using whisper-cli)")
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
        audio_seconds = len(audio_data) / AudioRecorder.WHISPER_SAMPLE_RATE
        timeout = max(120, audio_seconds * 4)
        wav_bytes = self._wav_bytes(audio_data)
        server_text = self._transcribe_with_server(wav_bytes, timeout)
        if server_text is not None:
            self.logger.info(f"Transcription completed in {time.time() - start_time:.2f}s (whisper-server)")
            return server_text or None

        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="whisperkey_")
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                tmp_file.write(wav_bytes)

            model_path = self._get_model_path()
            if not model_path:
                self.logger.error("whisper.cpp model [%s] not found", self.model_key)
                return None

            cmd = self._build_command(model_path, tmp_path)
            self.logger.debug("Running: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
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
    def _wav_bytes(audio_data: np.ndarray) -> bytes:
        if len(audio_data.shape) > 1:
            audio_data = audio_data.ravel()
        if np.issubdtype(audio_data.dtype, np.floating):
            audio_data = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
        elif audio_data.dtype != np.int16:
            audio_data = audio_data.clip(-32768, 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(AudioRecorder.WHISPER_SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        return buffer.getvalue()
