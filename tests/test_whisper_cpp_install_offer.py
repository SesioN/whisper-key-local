from unittest.mock import MagicMock

import pytest

import whisper_key.state_manager as state_manager_module
from whisper_key.model_registry import ModelRegistry
from whisper_key.runtime_loader import CPU, VULKAN
from whisper_key.runtime_options import FASTER_WHISPER, INSTALLABLE, INSTALLED, UNSUPPORTED, WHISPER_CPP, Runtime
from whisper_key.state_manager import StateManager

MODELS = {
    'tiny': {'label': 'Tiny'},
    'large-v3-turbo-german': {'source': 'TheTobyB/whisper-large-v3-turbo-german-ct2', 'label': 'German'},
    'parakeet': {'engine': 'onnx_asr', 'onnx_model_type': 'nemo-conformer-tdt'},
}
CPU_RUNTIME = Runtime(CPU, "CPU", FASTER_WHISPER, "cpu", INSTALLED, compute_types=frozenset({"int8"}))


def make_manager(tmp_path, engine_type='onnx_asr', model_dir=None, vulkan_state=INSTALLABLE):
    config_manager = MagicMock()
    config_manager.get_setting.side_effect = lambda section, key: {
        ('whisper', 'runtime'): VULKAN, ('whisper', 'model_languages'): {}}.get((section, key))
    engine = MagicMock()
    engine.ENGINE_TYPE = engine_type
    engine.model_key = 'parakeet'
    engine.registry = ModelRegistry(MODELS)
    engine._is_model_cached.return_value = False
    manager = StateManager(audio_recorder=None, whisper_engine=engine, clipboard_manager=MagicMock(),
                           config_manager=config_manager, vad_manager=MagicMock(), text_postprocessor=MagicMock())
    manager.system_tray = MagicMock()
    manager.audio_recorder = MagicMock()
    manager.audio_recorder.get_recording_status.return_value = False
    manager.floating_widget = MagicMock()
    manager.terminal_title = MagicMock()
    manager._ggml_model_dir = MagicMock(return_value=str(model_dir) if model_dir else None)
    reason = "" if vulkan_state != UNSUPPORTED else "no Vulkan GPU driver"
    manager._runtimes = [CPU_RUNTIME, Runtime(VULKAN, "whisper.cpp", WHISPER_CPP, "vulkan", vulkan_state, reason)]
    manager._install_and_switch_runtime = MagicMock(return_value=True)
    manager.request_runtime_change = MagicMock(return_value=True)
    return manager


@pytest.fixture
def dialogs(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr(state_manager_module, 'dialogs', fake)
    return fake


def test_missing_whisper_cpp_offers_install_instead_of_unavailable(tmp_path):
    manager = make_manager(tmp_path)
    assert manager.get_model_download_state('tiny') == "install"


def test_model_without_whisper_cpp_version_falls_back_to_faster_whisper(tmp_path):
    manager = make_manager(tmp_path, model_dir=tmp_path)
    assert manager.get_model_download_state('large-v3-turbo-german') == "faster_whisper"


def test_installed_whisper_cpp_reports_download_and_ready(tmp_path):
    manager = make_manager(tmp_path, model_dir=tmp_path)
    assert manager.get_model_download_state('tiny') == "download"
    (tmp_path / "ggml-tiny.bin").write_bytes(b"")
    assert manager.get_model_download_state('tiny') == "ready"


def test_confirmed_install_installs_whisper_cpp_for_the_chosen_model(tmp_path, dialogs):
    manager = make_manager(tmp_path)
    dialogs.confirm.return_value = True
    manager._begin_engine_switch()
    manager._offer_whisper_cpp_install('tiny')
    runtime = manager._install_and_switch_runtime.call_args.args[0]
    assert runtime.key == VULKAN
    assert manager._install_and_switch_runtime.call_args.kwargs['model_key'] == 'tiny'


def test_declined_install_keeps_the_current_model(tmp_path, dialogs):
    manager = make_manager(tmp_path)
    dialogs.confirm.return_value = False
    manager._begin_engine_switch()
    manager._offer_whisper_cpp_install('tiny')
    manager._install_and_switch_runtime.assert_not_called()
    manager.request_runtime_change.assert_not_called()
    assert not manager._runtime_install_running


def test_unsupported_whisper_cpp_offers_faster_whisper_on_cpu(tmp_path, dialogs):
    manager = make_manager(tmp_path, vulkan_state=UNSUPPORTED)
    dialogs.confirm.return_value = True
    manager._begin_engine_switch()
    manager._offer_whisper_cpp_install('tiny')
    manager._install_and_switch_runtime.assert_not_called()
    manager.request_runtime_change.assert_called_once_with(CPU, model_key='tiny')
    assert not manager._runtime_install_running


def test_selecting_a_model_needing_whisper_cpp_starts_the_offer(tmp_path, dialogs):
    manager = make_manager(tmp_path)
    manager._offer_whisper_cpp_install = MagicMock()
    assert manager.request_model_change('tiny')
    for _ in range(100):
        if manager._offer_whisper_cpp_install.called:
            break
        __import__('time').sleep(0.01)
    manager._offer_whisper_cpp_install.assert_called_once_with('tiny')
