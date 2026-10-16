import threading
import time
from unittest.mock import ANY, MagicMock

import numpy as np

from whisper_key.audio_analysis import AnalysisSnapshot
from whisper_key.audio_recorder import Recording
from whisper_key.model_registry import ModelRegistry
from whisper_key.state_manager import StateManager


MODELS = {'tiny': {}, 'tiny.en': {}, 'parakeet': {'engine': 'onnx_asr', 'onnx_model_type': 'nemo-conformer-tdt'},
          'qwen': {'engine': 'onnx_asr', 'onnx_model_type': 'speech-llm'}}


def make_config_manager(settings=None):
    values = {
        ('vad', 'auto_trigger_enabled'): False,
        ('vad', 'auto_trigger_paste'): True,
        ('audio', 'host'): None,
        ('whisper', 'language'): 'auto',
        ('whisper', 'model_languages'): {},
        ('whisper', 'recent_languages'): [],
        **(settings or {}),
    }
    config_manager = MagicMock()
    config_manager.get_setting.side_effect = lambda section, key: values.get((section, key))
    config_manager.update_user_setting.side_effect = lambda section, key, value: values.__setitem__((section, key), value)
    config_manager.get_logging_config.return_value = {'log_transcriptions': False}
    return config_manager


def make_state_manager(transcribe=None, precheck_enabled=True, precheck_result=True, settings=None, model_key='tiny'):
    config_manager = make_config_manager(settings)
    transcribe_function = transcribe or (lambda audio: "hello")
    engine = MagicMock()
    engine.model_key = model_key
    engine.registry = ModelRegistry(MODELS)
    engine.transcribe_audio.side_effect = lambda audio, language=None: transcribe_function(audio)
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


def completed_analysis(speech_detected, trailing_silence_seconds):
    snapshot = AnalysisSnapshot()
    snapshot.complete((speech_detected, trailing_silence_seconds))
    return snapshot


def wait_for_pipeline(manager, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not manager.is_processing and manager._pipeline_jobs.empty() and manager._pipeline_job_started_at is None:
            return
        time.sleep(0.01)
    raise AssertionError("transcription pipeline did not finish")


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
    clipboard.deliver_transcription.assert_called_once_with("hello", False, ANY)


def test_realtime_speech_skips_the_precheck():
    manager, engine, vad_manager, _ = make_state_manager()
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32), completed_analysis(True, 0.0)), False)
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
    recording = Recording(np.ones(16000 * 10, dtype=np.float32), completed_analysis(True, 6.0))
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
    clipboard.deliver_transcription.assert_called_once_with("second try", True, ANY)


def test_recording_is_kept_when_the_engine_is_switching():
    manager, engine, _, _ = make_state_manager()
    manager.is_model_loading = True
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    wait_for_pipeline(manager)
    engine.transcribe_audio.assert_not_called()
    assert manager.has_failed_recording()
    assert not manager.is_processing


from whisper_key.languages import language_code


def test_language_code_normalizes_case_and_region():
    assert language_code("EN-us") == "en"
    assert language_code("auto") is None
    assert language_code(" Auto ") is None
    assert language_code("  ") is None
    assert language_code(None) is None


def test_job_mode_is_captured_when_it_is_submitted():
    seen = []
    release = threading.Event()
    manager, _, _, _ = make_state_manager()
    manager._transcription_pipeline = lambda request: seen.append(request) or release.wait(5)
    manager._command_mode = True
    manager._auto_triggered_recording = True
    manager._submit_transcription(Recording(np.ones(160, dtype=np.float32)), True)
    assert not manager._command_mode and not manager._auto_triggered_recording
    release.set()
    deadline = time.monotonic() + 5
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen[0].command_mode and seen[0].auto_triggered and seen[0].use_auto_enter


def test_retry_keeps_the_voice_activated_delivery_mode():
    manager, engine, _, clipboard = make_state_manager(transcribe=MagicMock(side_effect=[RuntimeError("boom"), "again"]))
    manager.audio_recorder.get_recording_status.return_value = False
    manager._auto_trigger_paste = False
    manager._auto_triggered_recording = True
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    wait_for_pipeline(manager)
    assert manager._failed_recording.auto_triggered
    assert manager.retry_failed_recording()
    wait_for_pipeline(manager)
    clipboard.copy_with_notification.assert_called_once_with("again")
    clipboard.deliver_transcription.assert_not_called()


def test_shutdown_waits_for_the_running_transcription_before_closing_the_engine():
    release = threading.Event()
    order = []

    def transcribe(audio):
        release.wait(5)
        order.append("transcribed")
        return "hello"

    manager, engine, _, _ = make_state_manager(transcribe=transcribe)
    engine.close.side_effect = lambda: order.append("closed")
    manager.audio_recorder.get_recording_status.return_value = False
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    threading.Timer(0.1, release.set).start()
    manager.shutdown()
    assert order == ["transcribed", "closed"]


def test_stuck_pipeline_is_logged(caplog):
    import logging
    manager, _, _, _ = make_state_manager()
    manager._pipeline_job_started_at = time.monotonic() - manager.PIPELINE_STUCK_WARNING_SECONDS - 1
    with caplog.at_level(logging.WARNING):
        manager._warn_if_pipeline_stuck()
    assert "Transcription has been running" in caplog.text
    manager._pipeline_job_started_at = None


def test_refreshing_runtimes_never_shows_them_as_undetected(monkeypatch):
    from whisper_key import state_manager as state_manager_module
    manager, _, _, _ = make_state_manager()
    monkeypatch.setattr(state_manager_module, "detect_runtimes", lambda binary: ["runtime"])
    monkeypatch.setattr(state_manager_module, "whisper_cpp_runtime_paths", lambda: (None, None))
    monkeypatch.setattr("whisper_key.whisper_cpp_engine.find_whisper_cli", lambda: None)
    manager.get_runtimes()
    manager.refresh_runtimes()
    assert manager.runtimes_detected() and manager.get_runtimes() == ["runtime"]


def test_retry_does_not_play_the_stop_cue_or_touch_output_audio():
    manager, engine, _, _ = make_state_manager(transcribe=MagicMock(side_effect=[RuntimeError("boom"), "again"]))
    manager.audio_recorder.get_recording_status.return_value = False
    manager.audio_feedback = MagicMock()
    manager.output_audio_control = MagicMock()
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    wait_for_pipeline(manager)
    manager.audio_feedback.reset_mock()
    manager.output_audio_control.reset_mock()
    assert manager.retry_failed_recording()
    wait_for_pipeline(manager)
    manager.audio_feedback.play_stop_sound.assert_not_called()
    manager.output_audio_control.release.assert_not_called()


def test_no_model_switch_or_monitoring_restart_during_shutdown():
    release = threading.Event()
    manager, engine, _, _ = make_state_manager(transcribe=lambda audio: release.wait(5) and "hello")
    manager.audio_recorder.get_recording_status.return_value = False
    manager._execute_model_change = MagicMock()
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    manager._pending_model_change = "base"
    threading.Timer(0.1, release.set).start()
    manager._apply_auto_trigger.reset_mock()
    manager.shutdown()
    manager._execute_model_change.assert_not_called()
    manager._apply_auto_trigger.assert_not_called()


def test_runtime_refresh_detects_outside_the_lock(monkeypatch):
    from whisper_key import state_manager as state_manager_module
    manager, _, _, _ = make_state_manager()
    lock_held = []
    monkeypatch.setattr(state_manager_module, "detect_runtimes", lambda binary: lock_held.append(manager._runtimes_lock.locked()) or [])
    monkeypatch.setattr(state_manager_module, "whisper_cpp_runtime_paths", lambda: (None, None))
    monkeypatch.setattr("whisper_key.whisper_cpp_engine.find_whisper_cli", lambda: None)
    manager.refresh_runtimes()
    assert lock_held == [False]


def test_language_setting_precedence():
    manager, _, _, _ = make_state_manager(settings={('whisper', 'language'): 'fr', ('whisper', 'model_languages'): {'qwen': 'de-AT'}})
    assert manager.get_language_setting('tiny') == 'fr'
    assert manager.get_language_setting('qwen') == 'de'
    assert manager.get_effective_language('tiny.en') == 'en'
    assert manager.get_effective_language('parakeet') is None


def test_choosing_a_language_saves_it_per_model_and_tracks_recent_ones():
    manager, _, _, _ = make_state_manager()
    assert manager.set_model_language('tiny', 'de')
    assert manager.set_model_language('qwen', 'fr')
    assert manager.set_model_language('tiny', 'nl')
    assert manager.set_model_language('tiny', 'de')
    assert manager.get_effective_language('tiny') == 'de' and manager.get_effective_language('qwen') == 'fr'
    assert manager.config_manager.get_setting('whisper', 'recent_languages') == ['de', 'nl', 'fr']
    assert manager.set_model_language('tiny', None)
    assert manager.get_effective_language('tiny') is None
    assert manager.config_manager.get_setting('whisper', 'model_languages')['tiny'] == 'auto'


def test_recent_languages_only_list_what_the_model_supports():
    manager, _, _, _ = make_state_manager(settings={('whisper', 'recent_languages'): ['sw', 'de']})
    assert manager.get_recent_languages('tiny') == ['sw', 'de']
    assert manager.get_recent_languages('qwen') == ['de']


def test_unsupported_or_fixed_choices_are_rejected():
    manager, _, _, _ = make_state_manager()
    assert not manager.set_model_language('qwen', 'sw')
    assert not manager.set_model_language('tiny.en', 'de')
    assert not manager.set_model_language('parakeet', 'de')
    manager.config_manager.update_user_setting.assert_not_called()


def test_each_job_uses_the_language_chosen_when_it_was_submitted():
    seen_languages = []
    release = threading.Event()
    manager, engine, _, _ = make_state_manager(settings={('whisper', 'model_languages'): {'tiny': 'de'}})
    engine.transcribe_audio.side_effect = lambda audio, language=None: seen_languages.append(language) or release.wait(5) and "hallo"
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    manager.set_model_language('tiny', 'fr')
    release.set()
    wait_for_pipeline(manager)
    assert seen_languages == ['de']


def test_retry_uses_the_language_chosen_at_retry_time():
    seen_languages = []
    manager, engine, _, _ = make_state_manager(settings={('whisper', 'model_languages'): {'tiny': 'de'}})
    manager.audio_recorder.get_recording_status.return_value = False
    attempts = iter([RuntimeError("boom"), "bonjour"])

    def transcribe(audio, language=None):
        seen_languages.append(language)
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    engine.transcribe_audio.side_effect = transcribe
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    wait_for_pipeline(manager)
    manager.set_model_language('tiny', 'fr')
    assert manager.retry_failed_recording()
    wait_for_pipeline(manager)
    assert seen_languages == ['de', 'fr']


def test_language_is_resolved_again_when_the_model_changed_before_the_job_ran():
    seen_languages = []
    manager, engine, _, _ = make_state_manager(settings={('whisper', 'model_languages'): {'tiny': 'sw', 'qwen': 'de'}})
    engine.transcribe_audio.side_effect = lambda audio, language=None: seen_languages.append(language) or "hallo"
    release = threading.Event()
    original_pipeline = manager._transcription_pipeline

    def gated_pipeline(request):
        release.wait(5)
        return original_pipeline(request)

    manager._transcription_pipeline = gated_pipeline
    manager._submit_transcription(Recording(np.ones(16000, dtype=np.float32)), False)
    engine.model_key = 'qwen'
    release.set()
    wait_for_pipeline(manager)
    assert seen_languages == ['de']


def test_fixed_language_warning_only_for_an_explicit_choice(capsys):
    manager, _, _, _ = make_state_manager(settings={('whisper', 'language'): 'de'})
    manager.get_model_download_state = MagicMock(return_value="unavailable")
    manager.request_model_change('tiny.en')
    assert 'only transcribes' not in capsys.readouterr().out
    manager.config_manager.get_setting.side_effect = lambda section, key: {
        ('whisper', 'language'): 'de', ('whisper', 'model_languages'): {'tiny.en': 'de'}}.get((section, key))
    manager.request_model_change('tiny.en')
    assert 'only transcribes' in capsys.readouterr().out
