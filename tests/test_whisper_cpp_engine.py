import logging
from unittest.mock import MagicMock

import numpy as np

from whisper_key import whisper_cpp_engine
from whisper_key.model_registry import ModelRegistry
from whisper_key.whisper_cpp_engine import WhisperCppEngine


def make_engine(model_key="base", language=None, initial_prompt=None, hotwords=None, models=None):
    engine = WhisperCppEngine.__new__(WhisperCppEngine)
    engine.model_key = model_key
    engine.language = language
    engine.beam_size = 5
    engine.initial_prompt = initial_prompt
    engine.hotwords = hotwords
    engine.logger = logging.getLogger("test")
    engine.registry = ModelRegistry(models or {"base": {}, "base.en": {}, "german": {"language": "de"}, "noprompt": {"supports_prompt": False}})
    engine._prompt_skip_logged_models = set()
    engine._binary = "whisper-cli"
    engine._cli_flags = None
    engine.cpu_threads = 6
    return engine


def test_cli_command_uses_forced_language_combined_prompt_and_supported_flags(monkeypatch):
    help_result = MagicMock(stdout="  -sns,      --suppress-nst  \n  --carry-initial-prompt\n", stderr="")
    monkeypatch.setattr(whisper_cpp_engine.subprocess, "run", MagicMock(return_value=help_result))
    engine = make_engine(model_key="german", initial_prompt="Meeting notes.", hotwords="Kubernetes, Helm")
    command = engine._build_command("model.bin", "audio.wav")
    assert command[command.index("-l") + 1] == "de"
    assert command[command.index("-t") + 1] == "6"
    assert command[command.index("--prompt") + 1] == "Meeting notes. Kubernetes, Helm"
    assert "--suppress-nst" in command and "--carry-initial-prompt" in command


def test_cli_command_skips_flags_an_older_binary_does_not_know(monkeypatch):
    monkeypatch.setattr(whisper_cpp_engine.subprocess, "run", MagicMock(return_value=MagicMock(stdout="usage", stderr="")))
    engine = make_engine(model_key="base.en", initial_prompt="Hi.")
    command = engine._build_command("model.bin", "audio.wav")
    assert command[command.index("-l") + 1] == "en"
    assert "--suppress-nst" not in command and "--carry-initial-prompt" not in command


def test_cli_help_is_read_once(monkeypatch):
    run = MagicMock(return_value=MagicMock(stdout="", stderr=""))
    monkeypatch.setattr(whisper_cpp_engine.subprocess, "run", run)
    engine = make_engine()
    engine._build_command("m", "a")
    engine._build_command("m", "a")
    assert run.call_count == 1


def test_prompt_skipped_for_models_without_prompt_support():
    engine = make_engine(model_key="noprompt", initial_prompt="Hi.", hotwords="x")
    assert engine._active_prompt() is None


def test_server_request_carries_prompt_and_suppresses_non_speech(monkeypatch):
    engine = make_engine(language="fr", initial_prompt="Bonjour.", hotwords="Kubernetes")
    engine._server_lock = MagicMock()
    engine._server_url = "http://127.0.0.1:1/x/inference"
    engine._server_process = MagicMock()
    sent = {}

    def fake_multipart(fields, file_field, file_name, file_bytes):
        sent.update(fields)
        return b"", "multipart/form-data"

    response = MagicMock()
    response.__enter__.return_value = response
    monkeypatch.setattr(whisper_cpp_engine, "_multipart_body", fake_multipart)
    monkeypatch.setattr(whisper_cpp_engine.json, "load", lambda _: {"text": " Salut "})
    monkeypatch.setattr(whisper_cpp_engine._LOCAL_OPENER, "open", MagicMock(return_value=response))
    assert engine._transcribe_with_server(engine._wav_bytes(np.zeros(160, dtype=np.float32)), 5) == "Salut"
    assert sent["language"] == "fr"
    assert sent["prompt"] == "Bonjour. Kubernetes"
    assert sent["carry_initial_prompt"] == "true" and sent["suppress_nst"] == "true"


def test_failed_help_probe_is_retried(monkeypatch):
    results = iter([OSError("busy"), MagicMock(stdout="--suppress-nst", stderr="")])

    def run(*args, **kwargs):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(whisper_cpp_engine.subprocess, "run", run)
    engine = make_engine()
    assert engine._supported_cli_flags() == set()
    assert engine._supported_cli_flags() == {"--suppress-nst"}
