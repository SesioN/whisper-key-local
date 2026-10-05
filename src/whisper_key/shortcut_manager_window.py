import gc
import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable

from .utils import beautify_hotkey

QUEUE_POLL_INTERVAL_MS = 100
FOCUS_CHECK_DELAY_MS = 50
STOP_TIMEOUT_SECONDS = 3.0
BINDING_BUTTON_WIDTH = 22
ERROR_COLOR = "#C62828"
HINT_COLOR = "#616161"

RAISE = "raise"
CLOSE = "close"

ACTION_LABELS = {
    'recording_hotkey': "Start recording",
    'command_hotkey': "Voice command",
    'stop_key': "Stop recording",
    'auto_send_key': "Stop and send (ENTER)",
    'cancel_combination': "Cancel recording",
}
SLOT_LABELS = ("Primary", "Secondary")
MODIFIER_ORDER = ('ctrl', 'shift', 'alt', 'win')
ACTIONS_NEEDING_MODIFIER = ('recording_hotkey', 'command_hotkey')
TYPING_KEYS = ('space', 'enter', 'tab', 'backspace')
NOT_SET_TEXT = "Not set"
CAPTURE_TEXT = "Press keys... (click to cancel)"


def _is_typing_key(key_name: str) -> bool:
    return (len(key_name) == 1 and key_name.isalnum()) or key_name in TYPING_KEYS


def find_binding_problems(hotkey_bindings: dict) -> list:
    problems = []
    owners_by_binding = {}
    for action, bindings in hotkey_bindings.items():
        for binding in bindings:
            if not binding:
                continue
            owner = owners_by_binding.get(binding)
            if owner:
                if owner == action:
                    problems.append(f"{beautify_hotkey(binding)} is set twice for {ACTION_LABELS[action]}")
                else:
                    problems.append(f"{beautify_hotkey(binding)} is used by {ACTION_LABELS[owner]} and {ACTION_LABELS[action]}")
            else:
                owners_by_binding[binding] = action
            keys = binding.split('+')
            if action in ACTIONS_NEEDING_MODIFIER and len(keys) == 1 and _is_typing_key(keys[0]):
                problems.append(f"{ACTION_LABELS[action]} needs a modifier with {beautify_hotkey(binding)} (it would fire while typing)")
    return problems


def build_combination(pressed_keys: list) -> str:
    modifiers = [key for key in MODIFIER_ORDER if key in pressed_keys]
    other_keys = [key for key in pressed_keys if key not in MODIFIER_ORDER]
    return '+'.join(modifiers + other_keys)


class ShortcutManagerWindow:
    def __init__(self,
                 get_hotkey_bindings: Callable[[], dict],
                 get_default_hotkey_bindings: Callable[[], dict],
                 on_bindings_saved: Callable[[dict], None],
                 pause_hotkeys: Callable[[], None],
                 resume_hotkeys: Callable[[], None],
                 key_name_for_virtual_key: Callable[[int], str]):
        self.get_hotkey_bindings = get_hotkey_bindings
        self.get_default_hotkey_bindings = get_default_hotkey_bindings
        self.on_bindings_saved = on_bindings_saved
        self.pause_hotkeys = pause_hotkeys
        self.resume_hotkeys = resume_hotkeys
        self.key_name_for_virtual_key = key_name_for_virtual_key
        self.logger = logging.getLogger(__name__)

        self._command_queue = queue.Queue()
        self._window_thread = None
        self._window_closing = False
        self._thread_lock = threading.Lock()

    def open(self):
        with self._thread_lock:
            closing_thread = self._window_thread if self._window_closing else None
            if self._window_thread and not closing_thread:
                self._command_queue.put(RAISE)
                return
        if closing_thread:
            closing_thread.join(timeout=STOP_TIMEOUT_SECONDS)
        with self._thread_lock:
            if self._window_thread:
                self._command_queue.put(RAISE)
                return
            self._command_queue = queue.Queue()
            self._window_closing = False
            self._window_thread = threading.Thread(
                target=self._run_window_thread, args=(self._command_queue,), daemon=True, name="ShortcutManagerWindow"
            )
            self._window_thread.start()

    def stop(self):
        with self._thread_lock:
            window_thread = self._window_thread
            if window_thread:
                self._command_queue.put(CLOSE)
        if not window_thread:
            return
        window_thread.join(timeout=STOP_TIMEOUT_SECONDS)
        if window_thread.is_alive():
            self.logger.warning("Shortcut manager window did not close within timeout")

    def _run_window_thread(self, command_queue):
        try:
            self._run_window(command_queue)
        except Exception as e:
            self.logger.error(f"Shortcut manager window failed: {e}")
            print(f"❌ Could not open the shortcuts window: {e}")
        finally:
            with self._thread_lock:
                self._window_closing = True
            gc.collect()
            with self._thread_lock:
                self._window_thread = None

    def _run_window(self, command_queue):
        root = tk.Tk()
        editor = None
        try:
            editor = _ShortcutEditor(self, root, command_queue)
            root.mainloop()
        finally:
            if editor:
                editor.end_capture()
            try:
                root.destroy()
            except tk.TclError:
                pass


class _ShortcutEditor:
    def __init__(self, window: ShortcutManagerWindow, root: tk.Tk, command_queue):
        self.window = window
        self.root = root
        self.command_queue = command_queue
        self.logger = window.logger
        self.saved_bindings = self._copy_bindings(window.get_hotkey_bindings())
        self.bindings = self._copy_bindings(self.saved_bindings)
        self.default_bindings = self._copy_bindings(window.get_default_hotkey_bindings())
        self.capture = None
        self.binding_buttons = {}
        self.clear_buttons = {}

        self._build()
        self._refresh()
        self._process_command_queue()

    def _copy_bindings(self, hotkey_bindings: dict) -> dict:
        return {action: list(hotkey_bindings.get(action) or ['', ''])[:len(SLOT_LABELS)]
                for action in ACTION_LABELS}

    def _build(self):
        root = self.root
        root.title("Whisper Key - Shortcuts")
        root.resizable(False, False)

        frame = ttk.Frame(root, padding=20)
        frame.pack(fill=tk.BOTH)

        ttk.Label(frame, text="Action").grid(row=0, column=0, sticky="w", padx=(0, 15))
        for slot, slot_label in enumerate(SLOT_LABELS):
            ttk.Label(frame, text=slot_label).grid(row=0, column=1 + slot * 2, sticky="w")

        for row, (action, action_label) in enumerate(ACTION_LABELS.items(), start=1):
            ttk.Label(frame, text=action_label).grid(row=row, column=0, sticky="w", padx=(0, 15), pady=3)
            for slot in range(len(SLOT_LABELS)):
                binding_button = ttk.Button(frame, width=BINDING_BUTTON_WIDTH,
                                            command=lambda a=action, s=slot: self._toggle_capture(a, s))
                binding_button.grid(row=row, column=1 + slot * 2, sticky="we", pady=3)
                clear_button = ttk.Button(frame, text="✕", width=3,
                                          command=lambda a=action, s=slot: self._set_binding(a, s, ''))
                clear_button.grid(row=row, column=2 + slot * 2, padx=(2, 12), pady=3)
                self.binding_buttons[(action, slot)] = binding_button
                self.clear_buttons[(action, slot)] = clear_button
            ttk.Button(frame, text="Default", command=lambda a=action: self._reset_action(a)).grid(row=row, column=5, pady=3)

        self.problem_label = ttk.Label(frame, foreground=ERROR_COLOR, wraplength=560, justify=tk.LEFT)
        self.problem_label.grid(row=len(ACTION_LABELS) + 1, column=0, columnspan=6, sticky="w", pady=(10, 0))
        self.hint_label = ttk.Label(frame, foreground=HINT_COLOR, wraplength=560, justify=tk.LEFT)
        self.hint_label.grid(row=len(ACTION_LABELS) + 2, column=0, columnspan=6, sticky="w")

        button_row = ttk.Frame(frame)
        button_row.grid(row=len(ACTION_LABELS) + 3, column=0, columnspan=6, sticky="we", pady=(15, 0))
        ttk.Button(button_row, text="Restore all defaults", command=self._restore_all_defaults).pack(side=tk.LEFT)
        self.save_button = ttk.Button(button_row, text="Save", command=self._save)
        self.save_button.pack(side=tk.RIGHT)
        ttk.Button(button_row, text="Cancel", command=self._close).pack(side=tk.RIGHT, padx=(0, 8))

        root.bind("<KeyPress>", self._on_key_press)
        root.bind("<KeyRelease>", self._on_key_release)
        root.bind("<FocusOut>", self._on_focus_out)
        root.protocol("WM_DELETE_WINDOW", self._close)

    def _refresh(self):
        for (action, slot), binding_button in self.binding_buttons.items():
            binding = self.bindings[action][slot]
            capturing = self.capture is not None and self.capture['target'] == (action, slot)
            binding_button.config(text=CAPTURE_TEXT if capturing else (beautify_hotkey(binding) or NOT_SET_TEXT))
            self.clear_buttons[(action, slot)].config(state=tk.NORMAL if binding else tk.DISABLED)

        problems = find_binding_problems(self.bindings)
        self._show_message(self.problem_label, "\n".join(problems))
        disabled_actions = [ACTION_LABELS[action] for action, bindings in self.bindings.items() if not any(bindings)]
        self._show_message(self.hint_label, f"No shortcut, disabled: {', '.join(disabled_actions)}" if disabled_actions else "")
        can_save = not problems and self.bindings != self.saved_bindings and self.capture is None
        self.save_button.config(state=tk.NORMAL if can_save else tk.DISABLED)

    def _show_message(self, label, text: str):
        label.config(text=text)
        if text:
            label.grid()
        else:
            label.grid_remove()

    def _set_binding(self, action: str, slot: int, binding: str):
        self.end_capture()
        self.bindings[action][slot] = binding
        self._refresh()

    def _reset_action(self, action: str):
        self.end_capture()
        self.bindings[action] = list(self.default_bindings[action])
        self._refresh()

    def _restore_all_defaults(self):
        self.end_capture()
        self.bindings = self._copy_bindings(self.default_bindings)
        self._refresh()

    def _toggle_capture(self, action: str, slot: int):
        previous_target = self.capture['target'] if self.capture else None
        self.end_capture()
        if previous_target != (action, slot):
            self._start_capture(action, slot)
        self._refresh()

    def _start_capture(self, action: str, slot: int):
        try:
            self.window.pause_hotkeys()
        except Exception as e:
            self.logger.error(f"Failed to pause hotkeys for recording a shortcut: {e}")
            return
        self.capture = {'target': (action, slot), 'held_keys': set(), 'pressed_keys': []}
        self.root.focus_force()

    def end_capture(self):
        if self.capture is None:
            return
        self.capture = None
        try:
            self.window.resume_hotkeys()
        except Exception as e:
            self.logger.error(f"Failed to resume hotkeys after recording a shortcut: {e}")

    def _finish_capture(self):
        action, slot = self.capture['target']
        combination = build_combination(self.capture['pressed_keys'])
        self.end_capture()
        if combination:
            self.bindings[action][slot] = combination
        self._refresh()

    def _on_key_press(self, event):
        if self.capture is None:
            return None
        key_name = self.window.key_name_for_virtual_key(event.keycode)
        if not key_name:
            self.logger.info(f"Unsupported key for shortcuts: keycode={event.keycode} keysym={event.keysym}")
            return "break"
        self.capture['held_keys'].add(key_name)
        if key_name not in self.capture['pressed_keys']:
            self.capture['pressed_keys'].append(key_name)
        return "break"

    def _on_key_release(self, event):
        if self.capture is None:
            return None
        key_name = self.window.key_name_for_virtual_key(event.keycode)
        if key_name not in self.capture['held_keys']:
            return "break"
        self.capture['held_keys'].discard(key_name)
        if not self.capture['held_keys']:
            self._finish_capture()
        return "break"

    def _on_focus_out(self, event):
        if self.capture is not None:
            self.root.after(FOCUS_CHECK_DELAY_MS, self._end_capture_if_window_inactive)

    def _end_capture_if_window_inactive(self):
        if self.capture is None or self.root.focus_get() is not None:
            return
        if self.capture['pressed_keys']:
            self._finish_capture()
        else:
            self.end_capture()
            self._refresh()

    def _save(self):
        self.end_capture()
        if find_binding_problems(self.bindings):
            self._refresh()
            return
        try:
            self.window.on_bindings_saved(self._copy_bindings(self.bindings))
        except Exception as e:
            self.logger.error(f"Failed to save shortcuts: {e}")
            self._show_message(self.problem_label, f"Could not save shortcuts: {e}")
            return
        self._close()

    def _close(self):
        self.end_capture()
        self.root.quit()

    def _process_command_queue(self):
        try:
            while True:
                command = self.command_queue.get_nowait()
                if command == CLOSE:
                    self._close()
                    return
                if command == RAISE:
                    self.root.deiconify()
                    self.root.lift()
                    self.root.focus_force()
        except queue.Empty:
            pass
        self.root.after(QUEUE_POLL_INTERVAL_MS, self._process_command_queue)
