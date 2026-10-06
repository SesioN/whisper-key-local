import os
import copy
import logging
from typing import Dict, Any, Optional
from io import StringIO

from ruamel.yaml import YAML

from .utils import resolve_asset_path, beautify_hotkey_bindings, get_user_app_data_path, get_version
from .platform import IS_MACOS
from .orb_skins import SKINS as ORB_SKINS

REPO_URL = "https://github.com/SesioN/whisper-key-local"


def _build_settings_header():
    version = get_version()
    ref = "master" if version.endswith("-dev") else f"v{version}"
    return (
        f"# Whisper Key {version} - User Settings\n"
        "#\n"
        f"# Available settings: {REPO_URL}/tree/{ref}?tab=readme-ov-file#%EF%B8%8F-configuration\n"
        f"# Defaults reference: {REPO_URL}/blob/{ref}/src/whisper_key/config.defaults.yaml\n"
        "\n"
    )

HOTKEY_ACTIONS = ('recording_hotkey', 'command_hotkey', 'stop_key', 'auto_send_key', 'cancel_combination')
HOTKEY_BINDING_SLOTS = 2
FLOATING_WIDGET_STYLES = ('button', 'orb')
ORB_SKIN_NAMES = tuple(ORB_SKINS)

EXTENSIBLE_PATHS = {'whisper.models', 'streaming.models', 'post_processing.corrections'}

def deep_merge_config(default_config: Dict[str, Any],
                      user_config: Dict[str, Any]) -> Dict[str, Any]:

    result = default_config.copy()

    for key, value in user_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_config(result[key], value)
        else:
            result[key] = value

    return result


def _to_plain(obj):
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def _compute_overrides(config, defaults, path_prefix=''):
    overrides = {}
    for key, value in config.items():
        current_path = f"{path_prefix}.{key}" if path_prefix else key
        if key not in defaults:
            if current_path in EXTENSIBLE_PATHS or path_prefix in EXTENSIBLE_PATHS:
                overrides[key] = value
            continue
        if isinstance(value, dict) and isinstance(defaults[key], dict):
            nested = _compute_overrides(value, defaults[key], current_path)
            if nested:
                overrides[key] = nested
        elif value != defaults[key]:
            overrides[key] = value
    return overrides


def _parse_platform_value(value: str) -> str:
    parts = value.split(' | macos:')
    default_value = parts[0].strip()
    macos_value = parts[1].strip() if len(parts) > 1 else default_value
    return macos_value if IS_MACOS else default_value


def _resolve_platform_values(config: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in config.items():
        if isinstance(value, dict):
            _resolve_platform_values(value)
        elif isinstance(value, str) and ' | macos:' in value:
            config[key] = _parse_platform_value(value)
        elif isinstance(value, list):
            config[key] = [_parse_platform_value(item) if isinstance(item, str) and ' | macos:' in item else item
                           for item in value]
    return config


def normalize_hotkey_bindings(value) -> list:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        value = []
    bindings = [item.lower().strip() if isinstance(item, str) else '' for item in value[:HOTKEY_BINDING_SLOTS]]
    return bindings + [''] * (HOTKEY_BINDING_SLOTS - len(bindings))


def _normalize_hotkey_section(config: Dict[str, Any]):
    hotkey_section = config.get('hotkey') or {}
    for action in HOTKEY_ACTIONS:
        hotkey_section[action] = normalize_hotkey_bindings(hotkey_section.get(action))


class ConfigManager:   
    def __init__(self, config_path: str = None, use_user_settings: bool = True):
        if config_path is None:
            config_path = resolve_asset_path("config.defaults.yaml")
        
        self.default_config_path = config_path
        self.use_user_settings = use_user_settings
        self.config = {}
        self.logger = logging.getLogger(__name__)
        
        self.config_path = self._determine_config_path(use_user_settings, config_path)
        
        self.config = self._load_config()
        self._print_config_status()

        self.logger.info("Configuration loaded successfully")
    
    def _determine_config_path(self, use_user_settings: bool, config_path: str) -> str:
        if use_user_settings:
            whisperkey_dir = get_user_app_data_path()
            self.user_settings_path = os.path.join(whisperkey_dir, 'user_settings.yaml')
            return self.user_settings_path
        else:
            return config_path
    
    
    def _ensure_user_settings_exist(self):
        user_settings_dir = os.path.dirname(self.user_settings_path)

        if not os.path.exists(user_settings_dir):
            os.makedirs(user_settings_dir, exist_ok=True)

        if not os.path.exists(self.user_settings_path):
            with open(self.user_settings_path, 'w', encoding='utf-8') as f:
                f.write(_build_settings_header())
            self.logger.info(f"Created user settings at {self.user_settings_path}")
    
    def _remove_unused_keys_from_user_config(self, user_config: Dict[str, Any], default_config: Dict[str, Any]):

        sections_to_remove = []

        for section, values in user_config.items():
            if section not in default_config:
                self.logger.info(f"Removed invalid config section: {section}")
                sections_to_remove.append(section)
            elif isinstance(values, dict) and isinstance(default_config[section], dict):
                keys_to_remove = []
                for key in values.keys():
                    if key not in default_config[section] and f"{section}.{key}" not in EXTENSIBLE_PATHS:
                        self.logger.info(f"Removed invalid config key: {section}.{key}")
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del values[key]

        for section in sections_to_remove:
            del user_config[section]
    
    def _fill_missing_runtime(self, user_config: Dict[str, Any]):
        from .runtime_loader import derive_runtime
        whisper_settings = user_config.get('whisper') or {}
        if 'runtime' in whisper_settings:
            return
        derived_runtime = derive_runtime(dict(whisper_settings), dict(user_config.get('onboarding') or {}))
        if derived_runtime != 'cpu':
            user_config.setdefault('whisper', {})['runtime'] = derived_runtime

    def _migrate_legacy_keys(self, user_config: Dict[str, Any]):
        clipboard = user_config.get('clipboard')
        if not isinstance(clipboard, dict) or 'type_also_copy_to_clipboard' not in clipboard:
            return
        legacy = clipboard.get('type_also_copy_to_clipboard')
        if 'copy_to_clipboard' not in clipboard and legacy and clipboard.get('delivery_method') == 'type':
            clipboard['copy_to_clipboard'] = True
            self.logger.info("Migrated clipboard.type_also_copy_to_clipboard to clipboard.copy_to_clipboard")

    def _load_config(self):

        default_config = self._load_default_config()
        self._defaults_baseline = validate_config(
            _resolve_platform_values(copy.deepcopy(default_config)),
            default_config,
            self.logger,
        )

        if self.use_user_settings:
            self._ensure_user_settings_exist()

            try:
                yaml = YAML()
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    user_config = yaml.load(file)

                if user_config is None:
                    user_config = {}

                self._migrate_legacy_keys(user_config)
                self._remove_unused_keys_from_user_config(user_config, default_config)
                self._fill_missing_runtime(user_config)
                merged_config = deep_merge_config(default_config, user_config)
                resolved_config = _resolve_platform_values(merged_config)
                self.logger.info(f"Loaded user configuration from {self.config_path}")

                validated_config = validate_config(resolved_config, default_config, self.logger)
                self.config = validated_config

                return validated_config

            except Exception as e:
                if "YAML" in str(e):
                    self.logger.error(f"Error parsing user YAML config: {e}")
                else:
                    self.logger.error(f"Error loading user config file: {e}")
                print(f"   ✗ Error loading user settings, using defaults: {e}")

        self.logger.info(f"Using default configuration from {self.default_config_path}")
        return _resolve_platform_values(default_config)
    
    def _load_default_config(self) -> Dict[str, Any]:
        try:
            yaml = YAML()
            with open(self.default_config_path, 'r', encoding='utf-8') as file:
                default_config = yaml.load(file)
            
            if default_config:
                self.logger.info(f"Loaded default configuration from {self.default_config_path}")
                return default_config
            else:
                self.logger.error(f"Default config file {self.default_config_path} is empty")
                raise ValueError("Default configuration is empty")
                
        except Exception as e:
            if "YAML" in str(e):
                self.logger.error(f"Error parsing default YAML config: {e}")
            else:
                self.logger.error(f"Error loading default config file: {e}")
            raise

    def _write_user_config(self, user_config):
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)

        body = StringIO()
        yaml.dump(_to_plain(user_config), body)

        with open(self.user_settings_path, 'w', encoding='utf-8') as f:
            f.write(_build_settings_header())
            f.write(body.getvalue())

    def _print_config_status(self):
        print("📁 Loading configuration...")

        if self.use_user_settings:
            config_dir = os.path.dirname(self.user_settings_path)
            display_dir = self._display_path(config_dir)
            settings_file = os.path.basename(self.user_settings_path)
            print(f"   ✓ Local settings: {display_dir}{os.sep}{settings_file}")

            if self.get_voice_commands_config().get('enabled', True):
                print(f"   ✓ Voice commands: {display_dir}{os.sep}commands.yaml")
            else:
                print(f"   ✓ Voice commands: disabled")

    def _display_path(self, path: str) -> str:
        if IS_MACOS:
            home = os.path.expanduser("~")
            if path.startswith(home):
                return "~" + path[len(home):]
        else:
            appdata = os.getenv('APPDATA', '')
            if appdata and path.startswith(appdata):
                return "%APPDATA%" + path[len(appdata):]
        return path
    
    def _get_stop_key_display(self) -> str:
        return beautify_hotkey_bindings(self.config['hotkey']['stop_key'])

    def print_stop_instructions_based_on_config(self):
        recording_mode = self.config['hotkey'].get('recording_mode', 'toggle')

        if recording_mode == 'push_to_talk':
            print("   Release key to stop and transcribe")
            return

        stop_key = self._get_stop_key_display()
        auto_paste_enabled = self.config['clipboard']['auto_paste']
        auto_send_key = beautify_hotkey_bindings(self.config['hotkey']['auto_send_key'])

        if auto_paste_enabled:
            print(f"   [{stop_key}] to stop and auto-paste")
        else:
            print(f"   [{stop_key}] to stop and copy to clipboard")

        if auto_paste_enabled and auto_send_key:
            print(f"   [{auto_send_key}] to auto-paste and send with ENTER")

    def print_startup_hotkey_instructions(self):
        recording_hotkey = beautify_hotkey_bindings(self.config['hotkey']['recording_hotkey'])
        recording_mode = self.config['hotkey'].get('recording_mode', 'toggle')
        mode_hint = " (hold to record)" if recording_mode == "push_to_talk" else ""
        print(f"   [{recording_hotkey}] for transcription{mode_hint}")

        if self.get_voice_commands_config().get('enabled', True):
            command_hotkey = beautify_hotkey_bindings(self.config['hotkey']['command_hotkey'])
            if command_hotkey:
                print(f"   [{command_hotkey}] for voice commands")

    def print_command_stop_instructions(self):
        stop_key = self._get_stop_key_display()
        auto_send_key = beautify_hotkey_bindings(self.config['hotkey']['auto_send_key'])
        keys = " / ".join(key for key in (stop_key, auto_send_key) if key)
        print(f"   [{keys}] to stop and execute command")
    
    def get_whisper_config(self) -> Dict[str, Any]:
        return self.config['whisper'].copy()

    def get_engine_type(self) -> str:
        return self.config['whisper'].get('engine_type', 'faster_whisper')
    
    def get_hotkey_config(self) -> Dict[str, Any]:
        return self.config['hotkey'].copy()

    def get_hotkey_bindings(self) -> Dict[str, list]:
        return {action: list(self.config['hotkey'][action]) for action in HOTKEY_ACTIONS}

    def get_default_hotkey_bindings(self) -> Dict[str, list]:
        return {action: list(self._defaults_baseline['hotkey'][action]) for action in HOTKEY_ACTIONS}

    def update_hotkey_bindings(self, hotkey_bindings: Dict[str, list]):
        for action in HOTKEY_ACTIONS:
            self.config['hotkey'][action] = normalize_hotkey_bindings(hotkey_bindings[action])
        self._save_user_overrides()
    
    def get_audio_config(self) -> Dict[str, Any]:
        return self.config['audio'].copy()

    def get_audio_host(self) -> Optional[str]:
        return self.config['audio'].get('host')
    
    def get_clipboard_config(self) -> Dict[str, Any]:
        return self.config['clipboard'].copy()
    
    def get_logging_config(self) -> Dict[str, Any]:
        return self.config['logging'].copy()
    
    def get_vad_config(self) -> Dict[str, Any]:
        return self.config['vad'].copy()
    
    def get_system_tray_config(self) -> Dict[str, Any]:
        return self.config['system_tray'].copy()
    
    def get_floating_widget_config(self) -> Dict[str, Any]:
        return self.config['floating_widget'].copy()

    def get_audio_feedback_config(self) -> Dict[str, Any]:
        return self.config['audio_feedback'].copy()

    def get_post_processing_config(self) -> Dict[str, Any]:
        return self.config.get('post_processing', {}).copy()

    def get_voice_commands_config(self) -> Dict[str, Any]:
        return self.config.get('voice_commands', {}).copy()

    def get_terminal_title_config(self) -> Dict[str, Any]:
        return self.config.get('terminal_title', {}).copy()

    def get_loading_screen_config(self) -> Dict[str, Any]:
        return self.config.get('loading_screen', {}).copy()

    def get_update_config(self) -> Dict[str, Any]:
        return self.config.get('update', {}).copy()

    def get_streaming_config(self) -> Dict[str, Any]:
        return self.config.get('streaming', {}).copy()

    def get_log_file_path(self) -> str:
        log_filename = self.config['logging']['file']['filename']
        return os.path.join(get_user_app_data_path(), log_filename)

    def get_setting(self, section: str, key: str) -> Any:
        return self.config[section][key]
    
    def _save_user_overrides(self):
        try:
            overrides = _compute_overrides(self.config, self._defaults_baseline)
            self._write_user_config(overrides)
            self.logger.info(f"User overrides saved to {self.user_settings_path}")
        except Exception as e:
            self.logger.error(f"Error saving user overrides to {self.user_settings_path}: {e}")
            raise
    
    def update_audio_host(self, host_name: Optional[str]):
        self.update_user_setting('audio', 'host', host_name)

    def update_user_setting(self, section: str, key: str, value: Any):
        try:
            old_value = None
            if section in self.config and key in self.config[section]:
                old_value = self.config[section][key]

                if old_value != value:
                    self.config[section][key] = value
                    self._save_user_overrides()

                    self.logger.debug(f"Updated setting {section}.{key}: {old_value} -> {value}")
            else:
                self.logger.error(f"Setting {section}:{key} does not exist")

        except Exception as e:
            self.logger.error(f"Error updating user setting {section}.{key}: {e}")
            raise


def _get_config_value_at_path(config_dict, path):
    keys = path.split('.')
    current = config_dict
    for key in keys:
        current = current[key]
    return current


def _set_config_value_at_path(config_dict, path, value):
    keys = path.split('.')
    current = config_dict
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def _set_to_default(config, default_config, path, prev_value, logger):
    default_value = _get_config_value_at_path(default_config, path)
    _set_config_value_at_path(config, path, default_value)
    logger.warning(f"{prev_value} value not validated for config {path}, setting to default")


def _validate_numeric_range(config, default_config, path, logger, min_val=None, max_val=None):
    current_value = _get_config_value_at_path(config, path)

    if not isinstance(current_value, (int, float)):
        logger.warning(f"{current_value} must be numeric")
        _set_to_default(config, default_config, path, current_value, logger)
    elif min_val is not None and current_value < min_val:
        logger.warning(f"{current_value} must be >= {min_val}")
        _set_to_default(config, default_config, path, current_value, logger)
    elif max_val is not None and current_value > max_val:
        logger.warning(f"{current_value} must be <= {max_val}")
        _set_to_default(config, default_config, path, current_value, logger)


def _resolve_hotkey_conflicts(config, default_config, logger):
    owners_by_binding = {}
    for action in HOTKEY_ACTIONS:
        bindings = _get_config_value_at_path(config, f'hotkey.{action}')
        had_bindings = any(bindings)
        _assign_free_bindings(action, bindings, owners_by_binding, logger)
        if had_bindings and not any(bindings):
            default_bindings = normalize_hotkey_bindings(
                _resolve_platform_values({'value': copy.deepcopy(default_config['hotkey'][action])})['value'])
            logger.warning(f"   ✗ {action} has no free binding left, falling back to its default")
            _assign_free_bindings(action, default_bindings, owners_by_binding, logger)
            _set_config_value_at_path(config, f'hotkey.{action}', default_bindings)


def _assign_free_bindings(action, bindings, owners_by_binding, logger):
    for slot, binding in enumerate(bindings):
        if not binding:
            continue
        owner = owners_by_binding.get(binding)
        if owner:
            logger.warning(f"   ✗ {action} binding '{binding}' disabled: already used by {owner}")
            bindings[slot] = ''
        else:
            owners_by_binding[binding] = action


def validate_config(config, default_config, logger):
    _validate_numeric_range(config, default_config, 'audio.max_duration', logger, min_val=0)

    _validate_numeric_range(config, default_config, 'vad.vad_onset_threshold', logger, min_val=0.0, max_val=1.0)
    _validate_numeric_range(config, default_config, 'vad.vad_offset_threshold', logger, min_val=0.0, max_val=1.0)
    _validate_numeric_range(config, default_config, 'vad.vad_min_speech_duration', logger, min_val=0.001, max_val=5.0)
    _validate_numeric_range(config, default_config, 'vad.vad_silence_timeout_seconds', logger, min_val=1.0, max_val=36000.0)
    _validate_numeric_range(config, default_config, 'vad.auto_trigger_silence_seconds', logger, min_val=0.2, max_val=60.0)

    runtime = _get_config_value_at_path(config, 'whisper.runtime')
    if runtime not in ('cpu', 'cuda', 'rocm', 'vulkan'):
        _set_to_default(config, default_config, 'whisper.runtime', runtime, logger)

    onnx_runtime = _get_config_value_at_path(config, 'whisper.onnx_runtime')
    if onnx_runtime not in ('onnx-cpu', 'onnx-directml', 'onnx-cuda'):
        _set_to_default(config, default_config, 'whisper.onnx_runtime', onnx_runtime, logger)

    auto_trigger_enabled = _get_config_value_at_path(config, 'vad.auto_trigger_enabled')
    if not isinstance(auto_trigger_enabled, bool):
        _set_to_default(config, default_config, 'vad.auto_trigger_enabled', auto_trigger_enabled, logger)

    auto_trigger_paste = _get_config_value_at_path(config, 'vad.auto_trigger_paste')
    if not isinstance(auto_trigger_paste, bool):
        _set_to_default(config, default_config, 'vad.auto_trigger_paste', auto_trigger_paste, logger)

    auto_trigger_silence = _get_config_value_at_path(config, 'vad.auto_trigger_silence_seconds')
    silence_timeout = _get_config_value_at_path(config, 'vad.vad_silence_timeout_seconds')
    if auto_trigger_silence >= silence_timeout:
        logger.warning(f"vad.auto_trigger_silence_seconds ({auto_trigger_silence}) must be below vad.vad_silence_timeout_seconds ({silence_timeout})")
        _set_to_default(config, default_config, 'vad.auto_trigger_silence_seconds', auto_trigger_silence, logger)

    engine_type = _get_config_value_at_path(config, 'whisper.engine_type')
    if engine_type not in ('faster_whisper', 'whisper_cpp'):
        _set_to_default(config, default_config, 'whisper.engine_type', engine_type, logger)

    recording_mode = _get_config_value_at_path(config, 'hotkey.recording_mode')
    if recording_mode not in ('toggle', 'push_to_talk'):
        _set_to_default(config, default_config, 'hotkey.recording_mode', recording_mode, logger)

    floating_widget_size = _get_config_value_at_path(config, 'floating_widget.size')
    if floating_widget_size not in ('small', 'medium', 'big'):
        _set_to_default(config, default_config, 'floating_widget.size', floating_widget_size, logger)

    if _get_config_value_at_path(config, 'floating_widget.style') not in FLOATING_WIDGET_STYLES:
        _set_to_default(config, default_config, 'floating_widget.style', _get_config_value_at_path(config, 'floating_widget.style'), logger)

    if _get_config_value_at_path(config, 'floating_widget.orb_skin') not in ORB_SKIN_NAMES:
        _set_to_default(config, default_config, 'floating_widget.orb_skin', _get_config_value_at_path(config, 'floating_widget.orb_skin'), logger)

    for floating_widget_flag in ('floating_widget.enabled', 'floating_widget.save_position', 'floating_widget.locked', 'floating_widget.orb_hide_on_fullscreen',
                                 'floating_widget.orb_lock_button', 'floating_widget.orb_mute_button'):
        flag_value = _get_config_value_at_path(config, floating_widget_flag)
        if not isinstance(flag_value, bool):
            _set_to_default(config, default_config, floating_widget_flag, flag_value, logger)

    for floating_widget_position_path in ('floating_widget.position', 'floating_widget.orb_position'):
        floating_widget_position = _get_config_value_at_path(config, floating_widget_position_path)
        if floating_widget_position is not None and not isinstance(floating_widget_position, str):
            _set_to_default(config, default_config, floating_widget_position_path, floating_widget_position, logger)

    _normalize_hotkey_section(config)
    _resolve_hotkey_conflicts(config, default_config, logger)

    return config
