import logging
import threading

from .platform import hotkeys
from .state_manager import StateManager

HOTKEY_ACTIONS = ('recording_hotkey', 'command_hotkey', 'stop_key', 'auto_send_key', 'cancel_combination')

class HotkeyListener:
    def __init__(self, state_manager: StateManager, hotkey_bindings: dict, recording_mode: str = "toggle"):
        self.state_manager = state_manager
        self.hotkey_bindings_by_action = self._clean_bindings(hotkey_bindings)
        self.recording_mode = recording_mode
        self.keys_armed = True
        self.is_listening = False
        self.is_shut_down = False
        self._listening_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

        self._setup_hotkeys()

        self.start_listening()

    def _clean_bindings(self, hotkey_bindings: dict) -> dict:
        return {action: [binding.lower().strip() for binding in hotkey_bindings.get(action) or [] if binding and binding.strip()]
                for action in HOTKEY_ACTIONS}

    def _build_action_configs(self) -> dict:
        push_to_talk = self.recording_mode == "push_to_talk"
        return {
            'recording_hotkey': {
                'callback': self._standard_hotkey_pressed,
                'release_callback': self._push_to_talk_released if push_to_talk else self._arm_keys_on_release,
                'name': 'standard (push-to-talk)' if push_to_talk else 'standard'
            },
            'stop_key': {
                'callback': self._stop_key_pressed,
                'release_callback': self._arm_keys_on_release,
                'name': 'stop'
            },
            'auto_send_key': {
                'callback': self._auto_send_key_pressed,
                'release_callback': self._arm_keys_on_release,
                'name': 'auto-send'
            },
            'cancel_combination': {
                'callback': self._cancel_hotkey_pressed,
                'name': 'cancel'
            },
            'command_hotkey': {
                'callback': self._command_hotkey_pressed,
                'release_callback': self._push_to_talk_released if push_to_talk else None,
                'name': 'command (push-to-talk)' if push_to_talk else 'command'
            },
        }

    def _setup_hotkeys(self):
        hotkey_configs = []
        for action, action_config in self._build_action_configs().items():
            for combination in self.hotkey_bindings_by_action[action]:
                hotkey_configs.append({**action_config, 'combination': combination})

        hotkey_configs.sort(key=self._get_hotkey_combination_specificity, reverse=True)

        self.hotkey_bindings = []
        for config in hotkey_configs:
            hotkey = config['combination']
            self.hotkey_bindings.append([
                hotkey,
                config['callback'],
                config.get('release_callback') or None,
                False
            ])
            self.logger.info(f"Configured {config['name']} hotkey: {hotkey}")

        self.logger.info(f"Total hotkeys configured: {len(self.hotkey_bindings)}")

    def _get_hotkey_combination_specificity(self, hotkey_config: dict) -> int:
        combination = hotkey_config['combination'].lower()
        return len(combination.split('+'))

    def _standard_hotkey_pressed(self):
        self.logger.info("Standard hotkey pressed")
        self.keys_armed = False
        self.state_manager.start_recording()

    def _push_to_talk_released(self):
        self.logger.info("Push-to-talk key released")
        self.state_manager.stop_recording()

    def _stop_key_pressed(self):
        self.logger.debug(f"Stop key pressed, keys_armed={self.keys_armed}")

        if self.keys_armed:
            self.logger.info("Stop key activated")
            self.state_manager.stop_recording()
        else:
            self.logger.debug("Stop key ignored - waiting for key release first")

    def _auto_send_key_pressed(self):
        self.logger.debug(f"Auto-send key pressed, keys_armed={self.keys_armed}")

        if not self.state_manager.audio_recorder.get_recording_status():
            self.logger.debug("Auto-send key ignored - not currently recording")
            return

        if not self.keys_armed:
            self.logger.debug("Auto-send key ignored - waiting for key release first")
            return

        self.keys_armed = False

        self.state_manager.stop_recording(use_auto_enter=True)

    def _cancel_hotkey_pressed(self):
        self.logger.info("Cancel hotkey pressed")
        self.state_manager.cancel_recording_hotkey_pressed()

    def _command_hotkey_pressed(self):
        self.logger.info("Command hotkey pressed")
        self.keys_armed = False
        self.state_manager.start_command_recording()

    def _arm_keys_on_release(self):
        self.logger.debug("Key released - arming stop/auto-send keys")
        self.keys_armed = True

    def start_listening(self):
        with self._listening_lock:
            if self.is_listening or self.is_shut_down:
                return

            try:
                hotkeys.clear()
                hotkeys.register(self.hotkey_bindings)
                hotkeys.start()
                self.is_listening = True

            except Exception as e:
                self.logger.error(f"Failed to start hotkey listener: {e}")
                raise

    def stop_listening(self):
        with self._listening_lock:
            if not self.is_listening:
                return

            try:
                hotkeys.stop()
                self.is_listening = False
                self.logger.info("Hotkey listener stopped")

            except Exception as e:
                self.logger.error(f"Error stopping hotkey listener: {e}")

    def shutdown(self):
        with self._listening_lock:
            self.is_shut_down = True
            self.stop_listening()

    def apply_hotkey_bindings(self, hotkey_bindings: dict):
        with self._listening_lock:
            self._rebuild_hotkeys(lambda: setattr(self, 'hotkey_bindings_by_action', self._clean_bindings(hotkey_bindings)))

    def set_recording_mode(self, recording_mode: str):
        with self._listening_lock:
            if recording_mode == self.recording_mode:
                return
            self._rebuild_hotkeys(lambda: setattr(self, 'recording_mode', recording_mode))
            self.logger.info(f"Recording mode changed to {recording_mode}")

    def _rebuild_hotkeys(self, apply_change):
        was_listening = self.is_listening
        self.stop_listening()
        apply_change()
        self.keys_armed = True
        self._setup_hotkeys()
        if was_listening:
            self.start_listening()

    def is_active(self) -> bool:
        return self.is_listening
