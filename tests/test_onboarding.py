from unittest.mock import MagicMock

from whisper_key import onboarding
from whisper_key.onboarding import recommend_model, run_model_onboarding

MODELS = {name: {'label': name.title(), 'enabled': True}
          for name in ('tiny', 'base', 'base.en', 'small', 'small.en', 'large-v3-turbo')}


def test_recommendation_by_hardware_and_language():
    assert recommend_model(True, 2, None) == 'large-v3-turbo'
    assert recommend_model(False, 8, None) == 'small'
    assert recommend_model(False, 8, 'en') == 'small.en'
    assert recommend_model(False, 2, 'auto') == 'base'


def make_config_manager(created=True):
    config_manager = MagicMock()
    config_manager.created_user_settings = created
    return config_manager


def test_first_run_accepting_the_recommendation(monkeypatch):
    monkeypatch.setattr(onboarding, 'prompt_choice', lambda *args, **kwargs: onboarding.USE_RECOMMENDED_MODEL)
    config_manager = make_config_manager()
    config = run_model_onboarding(config_manager, {'model': 'tiny', 'language': 'en', 'models': MODELS}, False, 6)
    assert config['model'] == 'small.en'
    config_manager.update_user_setting.assert_called_once_with('whisper', 'model', 'small.en')


def test_existing_installs_and_custom_models_are_not_asked(monkeypatch):
    prompt = MagicMock()
    monkeypatch.setattr(onboarding, 'prompt_choice', prompt)
    run_model_onboarding(make_config_manager(created=False), {'model': 'tiny', 'models': MODELS}, False, 6)
    run_model_onboarding(make_config_manager(), {'model': 'medium', 'models': MODELS}, False, 6)
    prompt.assert_not_called()


def test_declining_keeps_tiny(monkeypatch):
    monkeypatch.setattr(onboarding, 'prompt_choice', lambda *args, **kwargs: 2)
    config_manager = make_config_manager()
    assert run_model_onboarding(config_manager, {'model': 'tiny', 'models': MODELS}, True, 6)['model'] == 'tiny'
    config_manager.update_user_setting.assert_not_called()
