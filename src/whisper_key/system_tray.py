import logging
import os
import signal
import threading
from typing import Optional, TYPE_CHECKING

from .utils import open_file
from .platform import IS_WINDOWS, permissions, icons, autostart
from .runtime_options import COMPUTE_TYPES
from .orb_skins import SKINS as ORB_SKINS
from .orb_skins.preview import render_skin_preview
from .tray_menu_model import Action, Choice, Footer, Header, Option, Section, Submenu, Toggle, to_pystray

try:
    import pystray
    from PIL import Image
    from .icon_effects import create_muted_icon
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    pystray = None
    Image = None

RECORDING_MODES = (("toggle", "Toggle"), ("push_to_talk", "Push to talk"))
FLOATING_WIDGET_STYLES = (("button", "Button"), ("orb", "Voice orb"))
FLOATING_WIDGET_SIZES = (("small", "Small"), ("medium", "Medium"), ("big", "Big"))
STATUS_TEXTS = {"idle": "Ready", "recording": "Recording", "processing": "Transcribing", "muted": "Microphone muted"}

ICON_MODEL = ""
ICON_MICROPHONE = ""
ICON_FLOATING_WIDGET = ""
ICON_KEYBOARD = ""
ICON_VOICE_COMMANDS = ""
ICON_FOLDER = ""
ICON_DOCUMENT = ""
ICON_EXIT = ""
ICON_SLIDERS = ""

if TYPE_CHECKING:
    from .state_manager import StateManager
    from .config_manager import ConfigManager

class SystemTray:
    def __init__(self,
                 state_manager: 'StateManager',
                 tray_config: dict = None,
                 config_manager: Optional['ConfigManager'] = None,
                 model_registry = None):

        self.state_manager = state_manager
        self.tray_config = tray_config or {}
        self.config_manager = config_manager
        self.model_registry = model_registry
        self.shortcut_manager_window = None
        self.voice_command_manager_window = None
        self.logger = logging.getLogger(__name__)

        self.recording_mode_changer = None
        self.icon = None  # pystray object, holds menu, state, etc.
        self.flyout = None
        self.is_running = False
        self.current_state = "idle"
        self.available = True
        self.audio_device_names = {}

        if self._check_tray_availability():
            self._load_icons_to_cache()

    def _check_tray_availability(self) -> bool:
        if not self.tray_config['enabled']:
            self.logger.warning("   ✗ System tray disabled in configuration")
            self.available = False

        elif not TRAY_AVAILABLE:
            self.logger.warning("   ✗ System tray not available - pystray or Pillow not installed")
            self.available = False

        return self.available

    def _load_icons_to_cache(self):
        try:
            self.icons = icons.get_tray_icons()
        except Exception as e:
            self.logger.error(f"Failed to load tray icons: {e}")
            self.icons = {
                "idle": self._create_fallback_icon("idle"),
                "recording": self._create_fallback_icon("recording"),
                "processing": self._create_fallback_icon("processing"),
                "muted": create_muted_icon(self._create_fallback_icon("idle")),
            }

    def _create_fallback_icon(self, state: str) -> Image.Image:
        colors = {
            'idle': (128, 128, 128),      # Gray
            'recording': (34, 139, 34),   # Green
            'processing': (255, 165, 0)   # Orange
        }

        color = colors.get(state, (128, 128, 128))  # Default to gray
        icon = Image.new('RGBA', (16, 16), color + (255,))

        return icon

    def build_menu(self) -> list:
        voice_commands_enabled = self.config_manager.get_setting('voice_commands', 'enabled')
        return [
            Header("Whisper Key", self._status_text, lambda: self.state_manager.is_muted, self._toggle_mute),
            Section("Recording"),
            Choice("Recording mode", [Option(mode, label) for mode, label in RECORDING_MODES],
                   self._current_recording_mode, self._set_recording_mode, style="segmented") if self.recording_mode_changer else None,
            Toggle("Voice-activated recording", lambda: self.state_manager.auto_trigger_enabled, self._set_auto_trigger) if self.state_manager.is_auto_trigger_available() else None,
            Action("Voice detection sensitivity...", self._open_vad_sensitivity_window, icon=ICON_SLIDERS) if self.state_manager.is_vad_sensitivity_window_available() else None,
            Toggle("Audio feedback", self._is_audio_feedback_enabled, self._set_audio_feedback),
            Section("Output"),
            Toggle("Auto-paste", self._is_auto_paste_enabled, self._set_transcription_mode),
            Toggle("Copy to clipboard", self._is_copy_enabled, self._set_copy_to_clipboard, enabled=self._is_auto_paste_enabled),
            Toggle("Voice commands in normal recordings", self._is_match_in_dictation_enabled, self._set_match_in_dictation) if self._is_voice_command_manager_enabled() else None,
            Section("Setup"),
            Submenu("Model", self._build_model_page, icon=ICON_MODEL, detail=self._model_detail),
            Submenu("Microphone", self._build_microphone_page, icon=ICON_MICROPHONE, detail=self._microphone_detail),
            Submenu("Floating button", self._build_floating_widget_page, icon=ICON_FLOATING_WIDGET, detail=self._floating_widget_detail) if self.state_manager.floating_widget_available else None,
            Action("Shortcuts...", self._open_shortcut_manager_window, icon=ICON_KEYBOARD) if self.shortcut_manager_window else None,
            Action("Voice commands...", self._open_voice_command_manager_window, icon=ICON_VOICE_COMMANDS) if self.voice_command_manager_window and voice_commands_enabled else None,
            Toggle("Start with Windows", self._is_autostart_enabled, self._set_autostart) if autostart.is_supported() else None,
            Submenu("Files and logs", self._build_files_page, icon=ICON_FOLDER),
            Footer([
                Action("Exit", self._quit_application_from_tray, icon=ICON_EXIT),
            ]),
        ]

    def _status_text(self) -> str:
        app_state = self.state_manager.get_application_state()
        if app_state.get('model_loading', False):
            status = "Loading model"
        elif self.state_manager.is_muted:
            status = STATUS_TEXTS["muted"]
        elif self.current_state == "idle" and self.state_manager.auto_trigger_enabled:
            status = "Listening for speech"
        else:
            status = STATUS_TEXTS.get(self.current_state, "Ready")
        return f"{status} · {self._model_detail()}"

    def _model_label(self, model_key: str) -> str:
        model = self.model_registry.get_model(model_key) if self.model_registry else None
        return model.label if model else model_key.title()

    def _model_detail(self) -> str:
        return self._model_label(self.config_manager.get_setting('whisper', 'model'))

    def _build_model_page(self) -> list:
        app_state = self.state_manager.get_application_state()
        is_model_loading = app_state.get('model_loading', False)
        is_busy = is_model_loading or app_state.get('processing', False)
        return [
            Choice("Model", self._model_options(is_model_loading),
                   lambda: self.config_manager.get_setting('whisper', 'model'), self._select_model),
            *self._build_runtime_items(is_busy),
        ]

    def _model_options(self, is_model_loading: bool) -> list:
        options = []
        if not self.model_registry:
            return options
        for group in self.model_registry.get_groups_ordered():
            models = self.model_registry.get_models_by_group(group)
            for index, model in enumerate(models):
                download_state = self.state_manager.get_model_download_state(model.key)
                options.append(Option(
                    model.key,
                    f"{model.label} (download)" if download_state == "download" else model.label,
                    enabled=download_state != "unavailable" and not is_model_loading,
                    starts_group=index == 0,
                ))
        return options

    def _build_runtime_items(self, is_busy: bool) -> list:
        current_runtime = self.state_manager.get_current_runtime()
        runtime_choice = Choice(
            "Runtime",
            [Option(runtime.key, runtime.menu_label, enabled=runtime.selectable and not is_busy)
             for runtime in self.state_manager.get_menu_runtimes()],
            lambda: getattr(self.state_manager.get_current_runtime(), 'key', None),
            self.state_manager.request_runtime_change,
        )
        if current_runtime and current_runtime.compute_types:
            precision_item = Choice(
                "Precision",
                [Option(compute_type, compute_type, enabled=not is_busy)
                 for compute_type in COMPUTE_TYPES if compute_type in current_runtime.compute_types],
                self.state_manager.get_current_compute_type,
                self.state_manager.request_compute_type_change,
            )
            return [runtime_choice, precision_item]
        return [runtime_choice, Section("Precision"), Action("Set by the model file", lambda: None, enabled=False)]

    def _microphone_detail(self) -> str:
        current_device = self.state_manager.get_current_audio_device_id()
        if current_device not in self.audio_device_names:
            self._audio_device_options()
        return self.audio_device_names.get(current_device) or self.state_manager.get_current_audio_host() or ""

    def _build_microphone_page(self) -> list:
        host_options = self._audio_host_options()
        return [
            Choice("Input device", self._audio_device_options, self.state_manager.get_current_audio_device_id,
                   lambda device_id: self._select_audio_device(device_id, self.audio_device_names.get(device_id, ""))),
        ] + ([Choice("Audio host", host_options, self.state_manager.get_current_audio_host, self._select_audio_host)]
             if host_options else [])

    def _audio_device_options(self) -> list:
        devices = self.state_manager.get_available_audio_devices(self.state_manager.get_current_audio_host()) or []
        self.audio_device_names = {device['id']: device['name'] for device in devices}
        return [Option(device['id'], device['name']) for device in devices]

    def _audio_host_options(self) -> list:
        return [Option(host['name'], host['name']) for host in self.state_manager.get_available_audio_hosts() or []]

    def _floating_widget_setting(self, key: str):
        return self.config_manager.get_setting('floating_widget', key)

    def _floating_widget_detail(self) -> str:
        if not self._floating_widget_setting('enabled'):
            return "Hidden"
        style = self._floating_widget_setting('style')
        if style == 'orb':
            skin_class = ORB_SKINS.get(self._floating_widget_setting('orb_skin'))
            return f"Orb · {skin_class.LABEL}" if skin_class else "Orb"
        return dict(FLOATING_WIDGET_STYLES).get(style, style)

    def _build_floating_widget_page(self) -> list:
        setting = self._floating_widget_setting
        items = [
            Toggle("Show", lambda: setting('enabled'), self._set_floating_widget_enabled),
            Choice("Style", [Option(style, label) for style, label in FLOATING_WIDGET_STYLES],
                   lambda: setting('style'), self._set_floating_widget_style, style="segmented"),
            Choice("Size", [Option(size, label) for size, label in FLOATING_WIDGET_SIZES],
                   lambda: setting('size'), self._set_floating_widget_size, style="segmented"),
        ]
        if setting('style') == 'orb':
            items += [
                Submenu("Orb skin", self._build_orb_skin_page,
                        detail=lambda: ORB_SKINS[setting('orb_skin')].LABEL if setting('orb_skin') in ORB_SKINS else ""),
                Section("Position and behavior"),
                Toggle("Remember position", lambda: setting('save_position'), self._set_floating_widget_save_position),
                Toggle("Lock position", lambda: setting('locked'), self._set_orb_locked),
                Toggle("Show lock button", lambda: setting('orb_lock_button'),
                       lambda enabled: self._set_orb_buttons(enabled, setting('orb_mute_button'))),
                Toggle("Show mute button", lambda: setting('orb_mute_button'),
                       lambda enabled: self._set_orb_buttons(setting('orb_lock_button'), enabled)),
                Toggle("Hide in fullscreen apps", lambda: setting('orb_hide_on_fullscreen'), self._set_orb_hide_on_fullscreen),
            ]
        else:
            items += [
                Section("Position"),
                Toggle("Remember position", lambda: setting('save_position'), self._set_floating_widget_save_position),
            ]
        return items

    def _build_orb_skin_page(self) -> list:
        return [Choice("Orb skin", [Option(skin, skin_class.LABEL) for skin, skin_class in ORB_SKINS.items()],
                       lambda: self._floating_widget_setting('orb_skin'), self._set_orb_skin,
                       preview=render_skin_preview)]

    def _build_files_page(self) -> list:
        voice_commands_enabled = self.config_manager.get_setting('voice_commands', 'enabled')
        return [
            Action("Settings file", self._open_config_file, icon=ICON_DOCUMENT),
            Action("Commands file", self._open_commands_file, icon=ICON_DOCUMENT) if voice_commands_enabled else None,
            Action("Config folder", self._open_config_folder, icon=ICON_FOLDER),
            Action("Log file", self._open_log_file, icon=ICON_DOCUMENT),
            Action("Model cache", self._open_model_cache, icon=ICON_FOLDER),
        ]

    def _set_floating_widget_style(self, style: str):
        self.state_manager.update_floating_widget_style(style)
        self.refresh_menu()

    def _set_orb_skin(self, skin: str):
        self.state_manager.update_orb_skin(skin)
        self.refresh_menu()

    def _set_orb_locked(self, locked: bool):
        self.state_manager.update_orb_locked(locked)

    def _set_orb_buttons(self, show_lock_button: bool, show_mute_button: bool):
        self.state_manager.update_orb_buttons(show_lock_button, show_mute_button)
        self.refresh_menu()

    def _set_orb_hide_on_fullscreen(self, hide_on_fullscreen: bool):
        self.state_manager.update_orb_hide_on_fullscreen(hide_on_fullscreen)
        self.refresh_menu()

    def _open_config_folder(self):
        try:
            config_dir = os.path.dirname(self.config_manager.user_settings_path)
            open_file(config_dir)
        except Exception as e:
            self.logger.error(f"Failed to open config folder: {e}")

    def _open_config_file(self):
        try:
            open_file(self.config_manager.user_settings_path)
        except Exception as e:
            self.logger.error(f"Failed to open config file: {e}")

    def get_commands_file_path(self) -> str:
        return os.path.join(os.path.dirname(self.config_manager.user_settings_path), "commands.yaml")

    def _open_commands_file(self):
        try:
            open_file(self.get_commands_file_path())
        except Exception as e:
            self.logger.error(f"Failed to open commands file: {e}")

    def _open_log_file(self):
        try:
            log_path = self.config_manager.get_log_file_path()
            open_file(log_path)
        except Exception as e:
            self.logger.error(f"Failed to open log file: {e}")

    def _open_model_cache(self):
        try:
            cache_path = self.model_registry.get_hf_cache_path()
            os.makedirs(cache_path, exist_ok=True)
            open_file(cache_path)
        except Exception as e:
            self.logger.error(f"Failed to open model cache: {e}")

    def _set_transcription_mode(self, auto_paste: bool):
        if auto_paste:
            if not permissions.check_accessibility_permission():
                if not permissions.handle_missing_permission(self.config_manager):
                    return
                auto_paste = False

        try:
            self.state_manager.update_transcription_mode(auto_paste)
        except Exception as e:
            self.logger.error(f"Error setting auto-paste to {auto_paste}: {e}")
        self.refresh_menu()

    def _is_audio_feedback_enabled(self):
        return bool(self.config_manager.get_setting('audio_feedback', 'enabled'))

    def _set_audio_feedback(self, enabled: bool):
        try:
            self.state_manager.update_audio_feedback(enabled)
        except Exception as e:
            self.logger.error(f"Error setting audio feedback to {enabled}: {e}")
        self.refresh_menu()

    def _is_voice_command_manager_enabled(self):
        voice_command_manager = self.state_manager.voice_command_manager
        return bool(voice_command_manager and voice_command_manager.enabled)

    def _is_match_in_dictation_enabled(self):
        return self.state_manager.voice_command_manager.match_in_dictation

    def _set_match_in_dictation(self, enabled: bool):
        try:
            self.state_manager.voice_command_manager.match_in_dictation = enabled
            self.config_manager.update_user_setting('voice_commands', 'match_in_dictation', enabled)
        except Exception as e:
            self.logger.error(f"Error setting voice commands in normal recordings to {enabled}: {e}")
        self.refresh_menu()

    def _is_auto_paste_enabled(self):
        return bool(self.config_manager.get_setting('clipboard', 'auto_paste'))

    def _is_copy_enabled(self):
        return bool(self.config_manager.get_setting('clipboard', 'copy_to_clipboard')) or not self._is_auto_paste_enabled()

    def _set_copy_to_clipboard(self, enabled: bool):
        try:
            self.state_manager.update_copy_to_clipboard(enabled)
        except Exception as e:
            self.logger.error(f"Error setting copy to clipboard to {enabled}: {e}")
        self.refresh_menu()

    def _select_model(self, model_key: str):
        try:
            success = self.state_manager.request_model_change(model_key)

            if success:
                self.refresh_menu()
            else:
                self.logger.warning(f"Request to change model to {model_key} was not accepted")

        except Exception as e:
            self.logger.error(f"Error selecting model {model_key}: {e}")

    def _set_auto_trigger(self, enabled: bool):
        threading.Thread(target=self._apply_auto_trigger_toggle, args=(enabled,), daemon=True).start()

    def _apply_auto_trigger_toggle(self, enabled: bool):
        try:
            self.state_manager.update_auto_trigger(enabled)
        except Exception as e:
            self.logger.error(f"Error toggling voice-activated recording: {e}")

    def _select_audio_host(self, host_name: str):
        try:
            success = self.state_manager.set_audio_host(host_name)
            if success:
                self.refresh_menu()
            else:
                self.logger.warning(f"Request to change audio host to {host_name} was not accepted")
        except Exception as e:
            self.logger.error(f"Error selecting audio host {host_name}: {e}")

    def attach_shortcut_manager_window(self, shortcut_manager_window):
        self.shortcut_manager_window = shortcut_manager_window
        self.refresh_menu()

    def _open_shortcut_manager_window(self):
        self.shortcut_manager_window.open()

    def attach_voice_command_manager_window(self, voice_command_manager_window):
        self.voice_command_manager_window = voice_command_manager_window
        self.refresh_menu()

    def _open_voice_command_manager_window(self):
        self.voice_command_manager_window.open()

    def _open_vad_sensitivity_window(self):
        self.state_manager.open_vad_sensitivity_window()

    def _select_audio_device(self, device_id: int, device_name: str):
        success = self.state_manager.request_audio_device_change(device_id, device_name)

        if success:
            self.config_manager.update_user_setting('audio', 'input_device', device_id)
            self.refresh_menu()
        else:
            self.logger.warning(f"Request to change audio device to {device_id} was not accepted")

    def attach_recording_mode_changer(self, recording_mode_changer):
        self.recording_mode_changer = recording_mode_changer
        self.refresh_menu()

    def _current_recording_mode(self):
        return self.config_manager.get_setting('hotkey', 'recording_mode')

    def _set_recording_mode(self, mode: str):
        try:
            if mode != self._current_recording_mode():
                self.recording_mode_changer(mode)
        except Exception as e:
            self.logger.error(f"Error setting recording mode to {mode}: {e}")
        self.refresh_menu()

    def _is_autostart_enabled(self):
        try:
            return autostart.is_enabled()
        except OSError as e:
            self.logger.error(f"Could not read autostart setting: {e}")
            return False

    def _set_autostart(self, enabled: bool):
        try:
            if enabled:
                autostart.enable()
                self.logger.info("Autostart enabled")
            else:
                autostart.disable()
                self.logger.info("Autostart disabled")
        except Exception as e:
            self.logger.error(f"Error changing autostart: {e}")
        self.refresh_menu()

    def _set_floating_widget_enabled(self, enabled: bool):
        self.state_manager.update_floating_widget_enabled(enabled)
        self.refresh_menu()

    def _set_floating_widget_save_position(self, save_position: bool):
        self.state_manager.update_floating_widget_save_position(save_position)
        self.refresh_menu()

    def _set_floating_widget_size(self, size: str):
        self.state_manager.update_floating_widget_size(size)
        self.refresh_menu()

    def _toggle_mute(self):
        self.state_manager.toggle_mute()

    def _quit_application_from_tray(self):
        signal.raise_signal(signal.SIGINT)

    def update_state(self, new_state: str):
        if not TRAY_AVAILABLE or not self.is_running:
            return

        self.current_state = new_state

        try:
            self.icon.icon = self.icons[new_state]
        except Exception as e:
            self.logger.error(f"Failed to update tray icon: {e}")
        self.refresh_menu()

    def set_status_text(self, text: Optional[str]):
        if not self.icon:
            return
        tooltip = self.tray_config.get('tooltip', 'Whisper Key')
        self.icon.title = f"{tooltip} - {text}"[:127] if text else tooltip

    def notify(self, message: str):
        if not self.icon or not self.is_running:
            return
        try:
            self.icon.notify(message, "Whisper Key")
        except Exception as e:
            self.logger.error(f"Failed to show tray notification: {e}")

    def refresh_menu(self):
        if not self.icon:
            return

        try:
            self.icon.title = self._get_title()
            if self._uses_flyout():
                self.flyout.refresh()
            else:
                self.icon.menu = self._build_native_menu()
        except Exception as e:
            self.logger.error(f"Failed to refresh tray menu: {e}")

    def _build_native_menu(self):
        return to_pystray(self.build_menu(), pystray)

    def _uses_flyout(self) -> bool:
        return self.flyout is not None and not self.flyout.failed

    def _get_title(self) -> str:
        if self.state_manager.auto_trigger_enabled:
            return "Whisper Key - listening for speech"
        return "Whisper Key"

    def start(self):
        if not self.available:
            return False

        if self.is_running:
            self.logger.warning("System tray is already running")
            return True

        try:
            idle_icon = self.icons.get("idle")

            self.icon = pystray.Icon(
                name="whisper-key",
                icon=idle_icon,
                title=self._get_title(),
                menu=self._build_native_menu()
            )
            if IS_WINDOWS:
                self._attach_flyout()

            self.icon.run_detached()

            self.is_running = True
            print("   ✓ System tray icon is running...")

            return True

        except Exception as e:
            self.logger.error(f"Failed to start system tray: {e}")
            return False

    def _attach_flyout(self):
        try:
            from pystray._util import win32 as pystray_win32
            from .tray_flyout import TrayFlyout
        except Exception as e:
            self.logger.error(f"Tray menu unavailable, using the native menu: {e}")
            return

        message_handlers = getattr(self.icon, '_message_handlers', None)
        if message_handlers is None or pystray_win32.WM_NOTIFY not in message_handlers:
            return
        native_notify_handler = message_handlers[pystray_win32.WM_NOTIFY]
        self.flyout = TrayFlyout(self.build_menu)

        def on_notify(wparam, lparam):
            if lparam not in (pystray_win32.WM_LBUTTONUP, pystray_win32.WM_RBUTTONUP):
                return native_notify_handler(wparam, lparam)
            if self.flyout.toggle(self.icon._hwnd):
                return None
            self.icon.menu = self._build_native_menu()
            return native_notify_handler(wparam, pystray_win32.WM_RBUTTONUP)

        message_handlers[pystray_win32.WM_NOTIFY] = on_notify

    def stop(self):
        if self.shortcut_manager_window:
            self.shortcut_manager_window.stop()
        if self.voice_command_manager_window:
            self.voice_command_manager_window.stop()
        if self.flyout:
            self.flyout.stop()

        if not self.is_running:
            return

        try:
            self.icon.stop()
            self.is_running = False

        except Exception as e:
            self.logger.error(f"Error stopping system tray: {e}")
