import logging
from typing import Callable

FOCUS_CHECK_DELAY_MS = 50
MODIFIER_ORDER = ('ctrl', 'shift', 'alt', 'win')


def build_combination(pressed_keys: list) -> str:
    modifiers = [key for key in MODIFIER_ORDER if key in pressed_keys]
    other_keys = [key for key in pressed_keys if key not in MODIFIER_ORDER]
    return '+'.join(modifiers + other_keys)


class KeyCapture:
    def __init__(self, root,
                 pause_hotkeys: Callable[[], None],
                 resume_hotkeys: Callable[[], None],
                 key_name_for_virtual_key: Callable[[int], str],
                 on_capture_changed: Callable[[], None]):
        self.root = root
        self.pause_hotkeys = pause_hotkeys
        self.resume_hotkeys = resume_hotkeys
        self.key_name_for_virtual_key = key_name_for_virtual_key
        self.on_capture_changed = on_capture_changed
        self.logger = logging.getLogger(__name__)
        self.target = None
        self._on_captured = None
        self._held_keys = set()
        self._pressed_keys = []

        root.bind("<KeyPress>", self._on_key_press, add="+")
        root.bind("<KeyRelease>", self._on_key_release, add="+")
        root.bind("<FocusOut>", self._on_focus_out, add="+")

    @property
    def is_active(self) -> bool:
        return self.target is not None

    def toggle(self, target, on_captured: Callable[[str], None]):
        previous_target = self.target
        self.end()
        if previous_target != target:
            self._start(target, on_captured)
        self.on_capture_changed()

    def _start(self, target, on_captured: Callable[[str], None]):
        try:
            self.pause_hotkeys()
        except Exception as e:
            self.logger.error(f"Failed to pause hotkeys for key capture: {e}")
            return
        self.target = target
        self._on_captured = on_captured
        self._held_keys = set()
        self._pressed_keys = []
        self.root.focus_force()

    def end(self):
        if self.target is None:
            return
        self.target = None
        self._on_captured = None
        try:
            self.resume_hotkeys()
        except Exception as e:
            self.logger.error(f"Failed to resume hotkeys after key capture: {e}")

    def _finish(self):
        on_captured = self._on_captured
        combination = build_combination(self._pressed_keys)
        self.end()
        if combination:
            on_captured(combination)
        self.on_capture_changed()

    def _on_key_press(self, event):
        if self.target is None:
            return None
        key_name = self.key_name_for_virtual_key(event.keycode)
        if not key_name:
            self.logger.info(f"Unsupported key for capture: keycode={event.keycode} keysym={event.keysym}")
            return "break"
        self._held_keys.add(key_name)
        if key_name not in self._pressed_keys:
            self._pressed_keys.append(key_name)
        return "break"

    def _on_key_release(self, event):
        if self.target is None:
            return None
        key_name = self.key_name_for_virtual_key(event.keycode)
        if key_name not in self._held_keys:
            return "break"
        self._held_keys.discard(key_name)
        if not self._held_keys:
            self._finish()
        return "break"

    def _on_focus_out(self, event):
        if self.target is not None:
            self.root.after(FOCUS_CHECK_DELAY_MS, self._end_if_window_inactive)

    def _end_if_window_inactive(self):
        if self.target is None or self.root.focus_get() is not None:
            return
        if self._pressed_keys:
            self._finish()
        else:
            self.end()
            self.on_capture_changed()
