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


def language_page_parts(tray):
    page = tray._build_language_page()
    quick_choice, all_languages = page
    assert isinstance(quick_choice, Choice) and isinstance(all_languages, Submenu)
    full_choice = all_languages.resolved_items()[0]
    return quick_choice, all_languages, full_choice


def test_selectable_model_offers_auto_and_recent_languages_with_the_full_list_one_level_down():
    tray = make_tray('qwen', effective_language='de', recent_languages=['de', 'fr'])
    quick_choice, all_languages, full_choice = language_page_parts(tray)
    assert [option.value for option in quick_choice.resolved_options()] == [AUTO_LANGUAGE, 'de', 'fr']
    assert quick_choice.resolved_options()[1].starts_group
    full_values = [option.value for option in full_choice.resolved_options()]
    assert sorted(full_values) == sorted(tray.model_registry.selectable_languages('qwen'))
    labels = [option.label for option in full_choice.resolved_options()]
    assert labels == sorted(labels, key=str.casefold)
    assert quick_choice.current() == full_choice.current() == 'de'
    assert all_languages.detail() == ''
    assert tray._language_detail() == 'German'


def test_all_languages_row_names_a_current_language_that_is_not_recent():
    tray = make_tray('tiny', effective_language='nl', recent_languages=['de'])
    _, all_languages, _ = language_page_parts(tray)
    assert all_languages.detail() == 'Dutch'


def test_native_menu_keeps_the_long_list_in_a_nested_submenu():
    import types

    class FakeMenu:
        SEPARATOR = object()

        def __init__(self, *items):
            self.items = items

    class FakeMenuItem:
        def __init__(self, text, action, **kwargs):
            self.text = text
            self.submenu = action if isinstance(action, FakeMenu) else None

    fake_pystray = types.SimpleNamespace(Menu=FakeMenu, MenuItem=FakeMenuItem)
    from whisper_key.tray_menu_model import to_pystray
    tray = make_tray('tiny', effective_language='de', recent_languages=['de'])
    language_menu = to_pystray(tray._build_language_page(), fake_pystray)
    top_level = [item for item in language_menu.items if item is not FakeMenu.SEPARATOR]
    assert len(top_level) <= 6
    nested = [item for item in top_level if item.submenu is not None]
    assert len(nested) == 1 and nested[0].text == 'All languages'
    assert len([item for item in nested[0].submenu.items if item is not FakeMenu.SEPARATOR]) == 100


def test_auto_detect_is_the_current_choice_without_a_language():
    tray = make_tray('tiny')
    assert language_page_parts(tray)[0].current() == AUTO_LANGUAGE
    assert tray._language_detail() == 'Auto-detect'


def test_selecting_a_language_saves_it_for_the_selected_model():
    tray = make_tray('tiny')
    tray.state_manager.set_model_language.return_value = True
    language_page_parts(tray)[2].on_select('nl')
    tray.state_manager.set_model_language.assert_called_once_with('tiny', 'nl')
    language_page_parts(tray)[0].on_select(AUTO_LANGUAGE)
    tray.state_manager.set_model_language.assert_called_with('tiny', None)
    tray.refresh_menu.assert_not_called()


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
