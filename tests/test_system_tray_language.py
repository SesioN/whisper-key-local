import logging
from unittest.mock import MagicMock

from whisper_key.languages import AUTO_LANGUAGE
from whisper_key.model_registry import ModelRegistry
from whisper_key.system_tray import SystemTray
from whisper_key.tray_menu_model import Choice, Info, Submenu

MODELS = {'tiny': {'label': 'Tiny'}, 'small.en': {'label': 'Small.En'},
          'parakeet': {'engine': 'onnx_asr', 'onnx_model_type': 'nemo-conformer-tdt', 'label': 'Parakeet'},
          'qwen': {'engine': 'onnx_asr', 'onnx_model_type': 'speech-llm', 'label': 'Qwen'}}


def make_tray(model_key, effective_language=None, recent_languages=()):
    tray = SystemTray.__new__(SystemTray)
    tray.logger = logging.getLogger("test")
    tray.model_registry = ModelRegistry(MODELS)
    tray.config_manager = MagicMock()
    tray.config_manager.get_setting.side_effect = lambda section, key: model_key
    tray.state_manager = MagicMock()
    tray.state_manager.get_effective_language.return_value = effective_language
    tray.state_manager.get_recent_languages.return_value = list(recent_languages)
    tray.state_manager.get_selectable_languages.side_effect = tray.model_registry.selectable_languages
    tray.refresh_menu = MagicMock()
    return tray


def test_selectable_model_lists_auto_then_recent_then_all_others_by_name():
    tray = make_tray('qwen', effective_language='de', recent_languages=['de', 'fr'])
    page = tray._build_language_page()
    assert len(page) == 1 and isinstance(page[0], Choice)
    options = page[0].resolved_options()
    values = [option.value for option in options]
    assert values[:3] == [AUTO_LANGUAGE, 'de', 'fr']
    assert len(values) == len(set(values)) == 1 + len(tray.model_registry.selectable_languages('qwen'))
    assert options[1].starts_group and options[3].starts_group
    others = [option.label for option in options[3:]]
    assert others == sorted(others, key=str.casefold)
    assert page[0].current() == 'de'
    assert tray._language_detail() == 'German'


def test_auto_detect_is_the_current_choice_without_a_language():
    tray = make_tray('tiny')
    assert tray._build_language_page()[0].current() == AUTO_LANGUAGE
    assert tray._language_detail() == 'Auto-detect'


def test_selecting_a_language_saves_it_for_the_selected_model():
    tray = make_tray('tiny')
    tray.state_manager.set_model_language.return_value = True
    tray._build_language_page()[0].on_select('nl')
    tray.state_manager.set_model_language.assert_called_once_with('tiny', 'nl')
    tray._build_language_page()[0].on_select(AUTO_LANGUAGE)
    tray.state_manager.set_model_language.assert_called_with('tiny', None)
    assert tray.refresh_menu.call_count == 2


def test_fixed_and_automatic_models_explain_instead_of_offering_choices():
    fixed_page = make_tray('small.en')._build_language_page()
    assert isinstance(fixed_page[0], Info) and fixed_page[0].value() == 'English, fixed by this model'
    assert make_tray('small.en')._language_detail() == 'English (fixed)'
    automatic_page = make_tray('parakeet')._build_language_page()
    assert isinstance(automatic_page[0], Info) and 'automatically' in automatic_page[0].value()
    assert make_tray('parakeet')._language_detail() == 'Automatic'


def test_model_page_has_a_language_row():
    tray = make_tray('tiny')
    tray.state_manager.get_application_state.return_value = {}
    tray.state_manager.runtimes_detected.return_value = False
    tray.state_manager.get_model_download_state.return_value = 'ready'
    page = tray._build_model_page()
    language_rows = [item for item in page if isinstance(item, Submenu) and item.label == 'Language']
    assert len(language_rows) == 1 and language_rows[0].detail() == 'Auto-detect'
