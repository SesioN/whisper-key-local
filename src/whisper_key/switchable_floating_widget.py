import threading
from typing import Callable, Dict

from .voice_orb import VoiceOrb


class SwitchableFloatingWidget:
    def __init__(self, widget_factories: Dict[str, Callable[[], object]], style: str):
        self.widget_factories = widget_factories
        self.style = style if style in widget_factories else next(iter(widget_factories))
        self._switch_lock = threading.RLock()
        self._shown = False
        self._state = "idle"
        self._muted = False
        self._widget = self.widget_factories[self.style]()

    def set_style(self, style: str):
        with self._switch_lock:
            if style == self.style or style not in self.widget_factories:
                return
            self._widget.stop()
            self.style = style
            self._widget = self.widget_factories[style]()
            self._widget.update_state(self._state)
            self._widget.set_muted(self._muted)
            if self._shown:
                self._widget.show()

    def show(self):
        with self._switch_lock:
            self._shown = True
            self._widget.show()

    def hide(self):
        with self._switch_lock:
            self._shown = False
            self._widget.hide()

    def update_state(self, new_state: str):
        with self._switch_lock:
            self._state = new_state
            self._widget.update_state(new_state)

    def set_muted(self, muted: bool):
        with self._switch_lock:
            self._muted = muted
            self._widget.set_muted(muted)

    def set_size(self, size: str):
        with self._switch_lock:
            self._widget.set_size(size)

    def set_save_position(self, save_position: bool):
        with self._switch_lock:
            self._widget.set_save_position(save_position)

    def set_level(self, level: float):
        widget = self._widget
        if isinstance(widget, VoiceOrb):
            widget.set_level(level)

    def set_orb_skin(self, skin: str):
        with self._switch_lock:
            if isinstance(self._widget, VoiceOrb):
                self._widget.set_skin(skin)

    def set_orb_locked(self, locked: bool):
        with self._switch_lock:
            if isinstance(self._widget, VoiceOrb):
                self._widget.set_locked(locked)

    def set_orb_hide_on_fullscreen(self, hide_on_fullscreen: bool):
        with self._switch_lock:
            if isinstance(self._widget, VoiceOrb):
                self._widget.set_hide_on_fullscreen(hide_on_fullscreen)

    def stop(self):
        with self._switch_lock:
            self._shown = False
            self._widget.stop()
