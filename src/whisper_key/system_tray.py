import logging
import os
import signal
from typing import Optional, TYPE_CHECKING
from pathlib import Path

from .utils import open_file
from .platform import permissions, icons, console
from .runtime_options import COMPUTE_TYPES

try:
    import pystray
    from PIL import Image
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    pystray = None
    Image = None

if TYPE_CHECKING:
    from .state_manager import StateManager
    from .config_manager import ConfigManager

class SystemTray:
    def __init__(self,
                 state_manager: 'StateManager',
                 tray_config: dict = None,
                 config_manager: Optional['ConfigManager'] = None,
                 model_registry = None,
                 console_config: dict = None):

        self.state_manager = state_manager
        self.tray_config = tray_config or {}
        self.config_manager = config_manager
        self.model_registry = model_registry
        self.console_config = console_config or {}
        self.logger = logging.getLogger(__name__)
               
        self.icon = None  # pystray object, holds menu, state, etc.
        self.is_running = False
        self.current_state = "idle"
        self.available = True
        
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
                "muted": self._create_fallback_icon("muted"),
            }
        
    def _create_fallback_icon(self, state: str) -> Image.Image:
        colors = {
            'idle': (128, 128, 128),      # Gray
            'recording': (34, 139, 34),   # Green  
            'processing': (255, 165, 0),  # Orange
            'muted': (200, 30, 30)
        }
        
        color = colors.get(state, (128, 128, 128))  # Default to gray
        icon = Image.new('RGBA', (16, 16), color + (255,))

        return icon
    
    def _build_model_menu_items(self, current_model: str, is_model_loading: bool) -> list:
        items = []

        if not self.model_registry:
            return items

        def make_model_selector(model_key):
            return lambda icon, item: self._select_model(model_key)

        def make_is_current(model_key):
            return lambda item: model_key == current_model

        def model_selection_enabled(item):
            return not is_model_loading

        first_group = True
        for group in self.model_registry.get_groups_ordered():
            models = self.model_registry.get_models_by_group(group)
            if not models:
                continue

            if not first_group:
                items.append(pystray.Menu.SEPARATOR)
            first_group = False

            for model in models:
                items.append(pystray.MenuItem(
                    model.label,
                    make_model_selector(model.key),
                    radio=True,
                    checked=make_is_current(model.key),
                    enabled=model_selection_enabled
                ))

        return items

    def _build_floating_widget_menu_items(self) -> list:
        if not self.state_manager.floating_widget_available:
            return []

        enabled = self.config_manager.get_setting('floating_widget', 'enabled')
        save_position = self.config_manager.get_setting('floating_widget', 'save_position')
        current_size = self.config_manager.get_setting('floating_widget', 'size')

        def make_size_selector(size):
            return lambda icon, item: self._set_floating_widget_size(size)

        def make_is_current_size(size):
            return lambda item: size == current_size

        items = [
            pystray.MenuItem("Show", lambda icon, item: self._set_floating_widget_enabled(not self.config_manager.get_setting('floating_widget', 'enabled')), checked=lambda item: enabled),
            pystray.MenuItem("Remember position", lambda icon, item: self._set_floating_widget_save_position(not self.config_manager.get_setting('floating_widget', 'save_position')), checked=lambda item: save_position),
            pystray.Menu.SEPARATOR,
        ]
        for size in ("small", "medium", "big"):
            items.append(pystray.MenuItem(size.title(), make_size_selector(size), radio=True, checked=make_is_current_size(size)))

        return items

    def _create_menu(self):
        try:
            app_state = self.state_manager.get_application_state()
            is_model_loading = app_state.get('model_loading', False)

            auto_paste_enabled = self.config_manager.get_setting('clipboard', 'auto_paste')
            current_model = self.config_manager.get_setting('whisper', 'model')

            available_hosts = self.state_manager.get_available_audio_hosts()
            current_host = self.state_manager.get_current_audio_host()

            def is_current_host(host_name):
                return lambda item: current_host == host_name

            def switch_host(host_name):
                return lambda icon, item: self._select_audio_host(host_name)

            audio_host_items = []
            if available_hosts:
                for host in available_hosts:
                    host_name = host['name']
                    audio_host_items.append(
                        pystray.MenuItem(
                            host_name,
                            switch_host(host_name),
                            radio=True,
                            checked=is_current_host(host_name)
                        )
                    )

            available_devices = self.state_manager.get_available_audio_devices(current_host)
            current_device = self.state_manager.get_current_audio_device_id()

            def is_current_device(dev_id):
                return lambda item: current_device == dev_id

            def switch_device(dev_id, dev_name):
                return lambda icon, item: self._select_audio_device(dev_id, dev_name)

            audio_device_items = []

            if available_devices:
                for device in available_devices:
                    device_id = device['id']
                    device_name = device['name']

                    audio_device_items.append(
                        pystray.MenuItem(
                            device_name,
                            switch_device(device_id, device['name']),
                            radio=True,
                            checked=is_current_device(device_id)
                        )
                    )

            model_sub_menu_items = self._build_model_menu_items(current_model, is_model_loading)
            floating_widget_menu_items = self._build_floating_widget_menu_items()

            voice_commands_enabled = self.config_manager.get_setting('voice_commands', 'enabled')
            auto_trigger_available = self.state_manager.is_auto_trigger_available()
            vad_sensitivity_window_available = self.state_manager.is_vad_sensitivity_window_available()

            menu_items = []

            if console.owns_console():
                menu_items.append(pystray.MenuItem("Show Console", self._show_console, default=True))
                menu_items.append(pystray.Menu.SEPARATOR)

            menu_items += [
                pystray.MenuItem("Mute microphone", self._toggle_mute, checked=lambda item: self.state_manager.is_muted),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Open log file...", self._open_log_file),
                pystray.MenuItem("Open model cache...", self._open_model_cache),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Open config folder...", self._open_config_folder),
                pystray.MenuItem("Open settings file...", self._open_config_file),
                pystray.MenuItem("Open commands file...", self._open_commands_file) if voice_commands_enabled else None,
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(
                    "Audio Host",
                    pystray.Menu(*audio_host_items)
                ) if audio_host_items else None,
                pystray.MenuItem(
                    f"Audio Source",
                    pystray.Menu(*audio_device_items)
                ),
                pystray.MenuItem("Audio feedback", lambda icon, item: self._set_audio_feedback(not self._is_audio_feedback_enabled()), checked=lambda item: self._is_audio_feedback_enabled()),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Auto-paste", lambda icon, item: self._set_transcription_mode(True), radio=True, checked=lambda item: auto_paste_enabled),
                pystray.MenuItem("Copy to clipboard", lambda icon, item: self._set_transcription_mode(False), radio=True, checked=lambda item: not auto_paste_enabled),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Floating button", pystray.Menu(*floating_widget_menu_items)) if floating_widget_menu_items else None,
                pystray.MenuItem(f"Model: {current_model.title()}", pystray.Menu(*model_sub_menu_items)),
                pystray.Menu.SEPARATOR if auto_trigger_available else None,
                pystray.MenuItem("Voice-activated recording", lambda icon, item: self._set_auto_trigger(not self.state_manager.auto_trigger_enabled), checked=lambda item: self.state_manager.auto_trigger_enabled) if auto_trigger_available else None,
                pystray.MenuItem("Voice detection sensitivity...", self._open_vad_sensitivity_window) if vad_sensitivity_window_available else None,
            ]

            menu_items += self._build_runtime_menu_items(is_model_loading or app_state.get('processing', False))

            menu_items.extend([
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Exit", self._quit_application_from_tray)
            ])

            menu = pystray.Menu(*[item for item in menu_items if item is not None])

            return menu 
                
        except Exception as e:
            self.logger.error(f"Error in _create_menu: {e}")
            raise

    def _build_runtime_menu_items(self, is_busy: bool) -> list:
        current_runtime = self.state_manager.get_current_runtime()
        current_compute_type = self.state_manager.get_current_compute_type()

        def make_runtime_selector(runtime_key):
            return lambda icon, item: self.state_manager.request_runtime_change(runtime_key)

        def make_compute_type_selector(compute_type):
            return lambda icon, item: self.state_manager.request_compute_type_change(compute_type)

        def make_is_current_runtime(runtime_key):
            return lambda item: current_runtime is not None and runtime_key == current_runtime.key

        def make_is_current_compute_type(compute_type):
            return lambda item: compute_type == current_compute_type

        runtime_items = [
            pystray.MenuItem(
                runtime.menu_label,
                make_runtime_selector(runtime.key),
                radio=True,
                checked=make_is_current_runtime(runtime.key),
                enabled=runtime.selectable and not is_busy
            )
            for runtime in self.state_manager.get_runtimes()
        ]

        if current_runtime and current_runtime.compute_types:
            precision_items = [
                pystray.MenuItem(
                    compute_type,
                    make_compute_type_selector(compute_type),
                    radio=True,
                    checked=make_is_current_compute_type(compute_type),
                    enabled=not is_busy
                )
                for compute_type in COMPUTE_TYPES if compute_type in current_runtime.compute_types
            ]
            precision_label = f"Precision: {current_compute_type}"
        else:
            precision_items = [pystray.MenuItem("Set by the model file", None, enabled=False)]
            precision_label = "Precision: model file"

        runtime_label = f"Runtime: {current_runtime.label}" if current_runtime else "Runtime"
        return [
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(runtime_label, pystray.Menu(*runtime_items)),
            pystray.MenuItem(precision_label, pystray.Menu(*precision_items)),
        ]

    def _open_config_folder(self, icon=None, item=None):
        try:
            config_dir = os.path.dirname(self.config_manager.user_settings_path)
            open_file(config_dir)
        except Exception as e:
            self.logger.error(f"Failed to open config folder: {e}")

    def _open_config_file(self, icon=None, item=None):
        try:
            open_file(self.config_manager.user_settings_path)
        except Exception as e:
            self.logger.error(f"Failed to open config file: {e}")

    def _open_commands_file(self, icon=None, item=None):
        try:
            commands_path = os.path.join(
                os.path.dirname(self.config_manager.user_settings_path),
                "commands.yaml"
            )
            open_file(commands_path)
        except Exception as e:
            self.logger.error(f"Failed to open commands file: {e}")

    def _open_log_file(self, icon=None, item=None):
        try:
            log_path = self.config_manager.get_log_file_path()
            open_file(log_path)
        except Exception as e:
            self.logger.error(f"Failed to open log file: {e}")

    def _open_model_cache(self, icon=None, item=None):
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

        self.state_manager.update_transcription_mode(auto_paste)
        self.icon.menu = self._create_menu()

    def _is_audio_feedback_enabled(self):
        return bool(self.config_manager.get_setting('audio_feedback', 'enabled'))

    def _set_audio_feedback(self, enabled: bool):
        try:
            self.state_manager.update_audio_feedback(enabled)
        except Exception as e:
            self.logger.error(f"Error setting audio feedback to {enabled}: {e}")
        self.icon.menu = self._create_menu()

    def _select_model(self, model_key: str):
        try:
            success = self.state_manager.request_model_change(model_key)

            if success:
                self.icon.menu = self._create_menu()
            else:
                self.logger.warning(f"Request to change model to {model_key} was not accepted")

        except Exception as e:
            self.logger.error(f"Error selecting model {model_key}: {e}")

    def _set_auto_trigger(self, enabled: bool):
        try:
            self.state_manager.update_auto_trigger(enabled)
            self.icon.menu = self._create_menu()
        except Exception as e:
            self.logger.error(f"Error toggling voice-activated recording: {e}")

    def _select_audio_host(self, host_name: str):
        try:
            success = self.state_manager.set_audio_host(host_name)
            if success:
                self.icon.menu = self._create_menu()
            else:
                self.logger.warning(f"Request to change audio host to {host_name} was not accepted")
        except Exception as e:
            self.logger.error(f"Error selecting audio host {host_name}: {e}")

    def _open_vad_sensitivity_window(self, icon=None, item=None):
        self.state_manager.open_vad_sensitivity_window()

    def _select_audio_device(self, device_id: int, device_name: str):
        success = self.state_manager.request_audio_device_change(device_id, device_name)

        if success:
            self.config_manager.update_user_setting('audio', 'input_device', device_id)
            self.icon.menu = self._create_menu()
        else:
            self.logger.warning(f"Request to change audio device to {device_id} was not accepted")

    def _set_floating_widget_enabled(self, enabled: bool):
        self.state_manager.update_floating_widget_enabled(enabled)
        self.icon.menu = self._create_menu()

    def _set_floating_widget_save_position(self, save_position: bool):
        self.state_manager.update_floating_widget_save_position(save_position)
        self.icon.menu = self._create_menu()

    def _set_floating_widget_size(self, size: str):
        self.state_manager.update_floating_widget_size(size)
        self.icon.menu = self._create_menu()

    def _toggle_mute(self, icon=None, item=None):
        self.state_manager.toggle_mute()

    def _show_console(self, icon=None, item=None):
        console.show()

    def apply_console_settings(self):
        if not console.owns_console() or not self.available:
            return
        if self.console_config.get('start_hidden', False):
            console.hide()
        console.start_minimize_monitor(console.hide)

    def _quit_application_from_tray(self, icon=None, item=None):        
        os.kill(os.getpid(), signal.SIGINT)
    
    def update_state(self, new_state: str):
        if not TRAY_AVAILABLE or not self.is_running:
            return
        
        self.current_state = new_state
        
        try:
            self.icon.icon = self.icons[new_state]
            self.icon.title = "Whisper Key (muted)" if new_state == "muted" else "Whisper Key"
            self.icon.menu = self._create_menu()
        except Exception as e:
            self.logger.error(f"Failed to update tray icon: {e}")

    def set_status_text(self, text: Optional[str]):
        if not self.icon:
            return
        tooltip = self.tray_config.get('tooltip', 'Whisper Key')
        self.icon.title = f"{tooltip} - {text}"[:127] if text else tooltip

    def refresh_menu(self):
        if not self.icon:
            return

        try:
            self.icon.menu = self._create_menu()
        except Exception as e:
            self.logger.error(f"Failed to refresh tray menu: {e}")
    
    def start(self):
        if not self.available:
            return False

        if self.is_running:
            self.logger.warning("System tray is already running")
            return True

        try:
            idle_icon = self.icons.get("idle")
            menu = self._create_menu()

            self.icon = pystray.Icon(
                name="whisper-key",
                icon=idle_icon,
                title="Whisper Key",
                menu=menu
            )

            self.icon.run_detached()

            self.is_running = True
            print("   ✓ System tray icon is running...")

            return True

        except Exception as e:
            self.logger.error(f"Failed to start system tray: {e}")
            return False
    
    def stop(self):
        if not self.is_running:
            return

        try:
            self.icon.stop()
            self.is_running = False

        except Exception as e:
            self.logger.error(f"Error stopping system tray: {e}")
