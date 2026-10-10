import logging

from whisper_key.config_manager import ConfigManager, validate_config


def test_defaults_load_with_precheck_thresholds_and_model_languages():
    config = ConfigManager(use_user_settings=False).config
    assert config['vad']['vad_precheck_onset_threshold'] > config['vad']['vad_precheck_offset_threshold']
    assert config['whisper']['models']['distil-large-v3.5']['language'] == 'en'
    assert 'dtype' not in config['audio']


def test_offset_above_onset_resets_both_to_defaults():
    defaults = ConfigManager(use_user_settings=False).config
    import copy
    config = copy.deepcopy(defaults)
    config['vad']['vad_precheck_onset_threshold'] = 0.3
    config['vad']['vad_precheck_offset_threshold'] = 0.6
    validated = validate_config(config, copy.deepcopy(defaults), logging.getLogger("test"))
    assert validated['vad']['vad_precheck_onset_threshold'] == defaults['vad']['vad_precheck_onset_threshold']
    assert validated['vad']['vad_precheck_offset_threshold'] == defaults['vad']['vad_precheck_offset_threshold']
