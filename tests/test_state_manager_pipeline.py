import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from whisper_key.audio_recorder import Recording
from whisper_key.state_manager import StateManager


def make_state_manager(transcribe=None, precheck_enabled=True, precheck_result=True):
    config_manager = MagicMock()
    config_manager.get_setting.side_effect = lambda section, key: {
        ('vad', 'auto_trigger_enabled'): False,
        ('vad', 'auto_trigger_paste'): True,
        ('audio', 'host'): None,
    }.get((section, key))
    config_manager.get_logging_config.return_value = {'log_transcriptions': False}
    engine = MagicMock()
    engine.transcribe_audio.side_effect = transcribe or (lambda audio: "hello")
    vad_manager = MagicMock()
    vad_manager.is_precheck_enabled.return_value = precheck_enabled
    vad_manager.check_audio_for_speech.return_value = precheck_result
    clipboard_manager = MagicMock()
    clipboard_manager.deliver_transcription.return_value = True
    text_postprocessor = MagicMock()
    text_postprocessor.process.side_effect = lambda text: text
    voice_command_manager = MagicMock()
    voice_command_manager.match_in_dictation = False
    manager = StateManager(audio_recorder=None, whisper_engine=engine, clipboard_manager=clipboard_manager,
                           config_manager=config_manager, vad_manager=vad_manager, text_postprocessor=text_postprocessor,
                           voice_command_manager=voice_command_manager)
    manager.audio_recorder = MagicMock()
    manager.audio_recorder.get_recording_status.return_value = True
    manager.audio_recorder.continuous_vad = None
    manager._apply_auto_trigger = MagicMock(return_value=False)
    return manager, engine, vad_manager, clipboard_manager


def wait_for_pipeline(manager):
    manager._pipeline_executor.submit(lambda: None).result(timeout=5)


def test_stop_recording_returns_before_transcription_finishes():
    release = threading.Event()
    manager, engine, _, clipboard = make_state_manager(transcribe=lambda audio: release.wait(5) and "hello")
    manager.audio_recorder.stop_recording.return_value = Recording(np.ones(16000, dtype=np.float32))
    assert manager.stop_recording()
    assert manager.is_processing
    assert not manager.can_start_recording()
    release.set()
    wait_for_pipeline(manager)
    assert not manager.is_processing
    clipboard.deliver_transcription.assert_called_once_with("hello", False)


def test_realtime_speech_skips_the_precheck():
    manager, engine, vad_manager, _ = make_state_manager()
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32), realtime_speech_detected=True), False)
    wait_for_pipeline(manager)
    vad_manager.check_audio_for_speech.assert_not_called()
    engine.transcribe_audio.assert_called_once()


def test_precheck_silence_skips_transcription():
    manager, engine, vad_manager, _ = make_state_manager(precheck_result=False)
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    wait_for_pipeline(manager)
    engine.transcribe_audio.assert_not_called()
    assert not manager.is_processing


def test_disabled_precheck_is_not_run():
    manager, engine, vad_manager, _ = make_state_manager(precheck_enabled=False)
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    wait_for_pipeline(manager)
    vad_manager.check_audio_for_speech.assert_not_called()
    engine.transcribe_audio.assert_called_once()


def test_trailing_silence_is_trimmed_before_transcription():
    manager, engine, _, _ = make_state_manager()
    recording = Recording(np.ones(16000 * 10, dtype=np.float32), realtime_speech_detected=True, trailing_silence_seconds=6.0)
    manager._submit_transcription(recording, False)
    wait_for_pipeline(manager)
    assert len(engine.transcribe_audio.call_args[0][0]) == 16000 * 5


def test_pipeline_errors_reset_processing_state():
    manager, engine, _, _ = make_state_manager(transcribe=MagicMock(side_effect=RuntimeError("boom")))
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    wait_for_pipeline(manager)
    assert not manager.is_processing


def test_runtime_menu_does_not_wait_for_detection(monkeypatch):
    manager, _, _, _ = make_state_manager()
    release = threading.Event()
    from whisper_key import state_manager as state_manager_module
    monkeypatch.setattr(state_manager_module, "detect_runtimes", lambda binary: release.wait(5) and [])
    monkeypatch.setattr(state_manager_module, "whisper_cpp_runtime_paths", lambda: (None, None))
    monkeypatch.setattr("whisper_key.whisper_cpp_engine.find_whisper_cli", lambda: None)
    manager.start_runtime_detection()
    assert manager.get_menu_runtimes() == []
    assert manager.get_current_runtime(wait=False) is None
    assert not manager.runtimes_detected()
    release.set()
    deadline = threading.Event()
    for _ in range(100):
        if manager.runtimes_detected():
            break
        deadline.wait(0.02)
    assert manager.runtimes_detected()


def test_failed_transcription_can_be_retried():
    attempts = []

    def transcribe(audio):
        attempts.append(len(audio))
        if len(attempts) == 1:
            raise RuntimeError("CUDA out of memory")
        return "second try"

    manager, engine, _, clipboard = make_state_manager(transcribe=transcribe)
    manager.audio_recorder.get_recording_status.return_value = False
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), True)
    wait_for_pipeline(manager)
    assert manager.has_failed_recording()
    assert manager.retry_failed_recording()
    wait_for_pipeline(manager)
    assert not manager.has_failed_recording()
    clipboard.deliver_transcription.assert_called_once_with("second try", True)


def test_recording_is_kept_when_the_engine_is_switching():
    manager, engine, _, _ = make_state_manager()
    manager.is_model_loading = True
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    wait_for_pipeline(manager)
    engine.transcribe_audio.assert_not_called()
    assert manager.has_failed_recording()
    assert not manager.is_processing
