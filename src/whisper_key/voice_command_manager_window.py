import gc
import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont, ttk
from typing import Callable

from .key_capture import KeyCapture
from .platform import dialogs
from .themed_widgets import (ACCENT_BUTTON_STYLE, ERROR_STYLE, HINT_STYLE, apply_theme, apply_window_chrome,
                             center_window, style_text_widget)
from .utils import beautify_hotkey
from .voice_commands import entry_to_command, find_entry_problems, find_matching_commands

QUEUE_POLL_INTERVAL_MS = 100
STOP_TIMEOUT_SECONDS = 3.0
TEST_DELAY_SECONDS = 1.0
PREVIEW_LENGTH = 40
WRAP_LENGTH = 760
ROW_PADDING = 8

RAISE = "raise"
CLOSE = "close"

ACTION_LABELS = {'hotkey': "Hotkey", 'type': "Type text", 'run': "Run command"}
CAPTURE_TEXT = "Press keys... (click to cancel)"
RECORD_TEXT = "Click to record keys"
NEW_TRIGGER_TEXT = "new command"


def _preview(entry: dict) -> str:
    if entry['action'] == 'hotkey':
        return beautify_hotkey(entry['value'])
    single_line = " ".join(entry['value'].split())
    return single_line if len(single_line) <= PREVIEW_LENGTH else single_line[:PREVIEW_LENGTH - 1] + "…"


class VoiceCommandManagerWindow:
    def __init__(self,
                 load_command_entries: Callable[[], list],
                 save_command_entries: Callable[[list], None],
                 execute_command: Callable[[dict], None],
                 open_commands_file: Callable[[], None],
                 pause_hotkeys: Callable[[], None],
                 resume_hotkeys: Callable[[], None],
                 key_name_for_virtual_key: Callable[[int], str]):
        self.load_command_entries = load_command_entries
        self.save_command_entries = save_command_entries
        self.execute_command = execute_command
        self.open_commands_file = open_commands_file
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
                target=self._run_window_thread, args=(self._command_queue,), daemon=True, name="VoiceCommandManagerWindow"
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
            self.logger.warning("Voice command manager window did not close within timeout")

    def _run_window_thread(self, command_queue):
        try:
            self._run_window(command_queue)
        except Exception as e:
            self.logger.error(f"Voice command manager window failed: {e}")
            print(f"❌ Could not open the voice commands window: {e}")
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
            editor = _VoiceCommandEditor(self, root, command_queue)
            root.mainloop()
        finally:
            if editor:
                editor.key_capture.end()
            try:
                root.destroy()
            except tk.TclError:
                pass


class _VoiceCommandEditor:
    def __init__(self, window: VoiceCommandManagerWindow, root: tk.Tk, command_queue):
        self.window = window
        self.root = root
        self.command_queue = command_queue
        self.logger = window.logger
        self.entries = window.load_command_entries()
        self.saved_entries = [dict(entry) for entry in self.entries]
        self.selected_index = None
        self._loading_editor = False

        self._build()
        self._refresh_list()
        if self.entries:
            self._select(0)
        else:
            self._refresh_editor()
        self._process_command_queue()

    def _build(self):
        root = self.root
        root.title("Whisper Key - Voice commands")
        root.minsize(820, 480)
        theme = apply_theme(root)

        frame = ttk.Frame(root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(0, weight=1)

        list_frame = ttk.Frame(frame)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        row_height = tkfont.nametofont("TkDefaultFont", root=root).metrics("linespace") + ROW_PADDING
        ttk.Style(root).configure("VoiceCommands.Treeview", rowheight=row_height)
        self.command_list = ttk.Treeview(list_frame, style="VoiceCommands.Treeview", columns=("enabled", "trigger", "action", "value"),
                                         show="headings", selectmode="browse", height=16)
        for column, heading, width, stretch in (("enabled", "On", 40, False), ("trigger", "Trigger", 160, True),
                                                ("action", "Type", 135, False), ("value", "Action", 220, True)):
            self.command_list.heading(column, text=heading)
            self.command_list.column(column, width=width, stretch=stretch, anchor="center" if column == "enabled" else "w")
        self.command_list.grid(row=0, column=0, sticky="nsew")
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.command_list.yview)
        list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.command_list.configure(yscrollcommand=list_scrollbar.set)
        self.command_list.bind("<<TreeviewSelect>>", self._on_list_selected)

        list_buttons = ttk.Frame(list_frame)
        list_buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(list_buttons, text="Add", command=self._add_entry).pack(side=tk.LEFT)
        self.duplicate_button = ttk.Button(list_buttons, text="Duplicate", command=self._duplicate_entry)
        self.duplicate_button.pack(side=tk.LEFT, padx=(6, 0))
        self.delete_button = ttk.Button(list_buttons, text="Delete", command=self._delete_entry)
        self.delete_button.pack(side=tk.LEFT, padx=(6, 0))
        self.move_up_button = ttk.Button(list_buttons, text="↑", width=3, command=lambda: self._move_entry(-1))
        self.move_up_button.pack(side=tk.LEFT, padx=(12, 0))
        self.move_down_button = ttk.Button(list_buttons, text="↓", width=3, command=lambda: self._move_entry(1))
        self.move_down_button.pack(side=tk.LEFT, padx=(4, 0))

        editor_frame = ttk.LabelFrame(frame, text="Command", padding=10)
        editor_frame.grid(row=0, column=1, sticky="nsew")
        editor_frame.columnconfigure(0, weight=1)
        self.editor_frame = editor_frame

        self.enabled_variable = tk.BooleanVar(value=True)
        self.enabled_checkbox = ttk.Checkbutton(editor_frame, text="Enabled", variable=self.enabled_variable,
                                                command=self._on_editor_changed)
        self.enabled_checkbox.grid(row=0, column=0, sticky="w")

        ttk.Label(editor_frame, text="Trigger phrase").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.trigger_variable = tk.StringVar()
        self.trigger_variable.trace_add("write", lambda *args: self._on_editor_changed())
        self.trigger_entry = ttk.Entry(editor_frame, textvariable=self.trigger_variable)
        self.trigger_entry.grid(row=2, column=0, sticky="we")

        ttk.Label(editor_frame, text="Action").grid(row=3, column=0, sticky="w", pady=(10, 0))
        action_row = ttk.Frame(editor_frame)
        action_row.grid(row=4, column=0, sticky="w")
        self.action_variable = tk.StringVar(value='hotkey')
        self.action_radios = []
        for action, label in ACTION_LABELS.items():
            radio = ttk.Radiobutton(action_row, text=label, value=action, variable=self.action_variable,
                                    command=self._on_action_changed)
            radio.pack(side=tk.LEFT, padx=(0, 10))
            self.action_radios.append(radio)

        self.value_frame = ttk.Frame(editor_frame)
        self.value_frame.grid(row=5, column=0, sticky="nsew", pady=(8, 0))
        self.value_frame.columnconfigure(0, weight=1)
        editor_frame.rowconfigure(5, weight=1)

        self.hotkey_button = ttk.Button(self.value_frame, command=self._toggle_hotkey_capture)
        self.type_text = tk.Text(self.value_frame, height=6, wrap=tk.WORD, undo=True)
        style_text_widget(self.type_text, theme)
        self.type_text.bind("<<Modified>>", self._on_type_text_modified)
        self.run_variable = tk.StringVar()
        self.run_variable.trace_add("write", lambda *args: self._on_editor_changed())
        self.run_entry = ttk.Entry(self.value_frame, textvariable=self.run_variable)
        self.run_warning = ttk.Label(self.value_frame, style=HINT_STYLE, wraplength=300, justify=tk.LEFT,
                                     text="Runs in the shell (cmd.exe) with your user rights.")

        self.test_button = ttk.Button(editor_frame, text="Test (minimizes, runs in 1 s)", command=self._test_selected_entry)
        self.test_button.grid(row=6, column=0, sticky="w", pady=(10, 0))

        tester_frame = ttk.Frame(frame)
        tester_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(12, 0))
        tester_frame.columnconfigure(1, weight=1)
        ttk.Label(tester_frame, text="Phrase to test").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.phrase_variable = tk.StringVar()
        self.phrase_variable.trace_add("write", lambda *args: self._refresh_match())
        ttk.Entry(tester_frame, textvariable=self.phrase_variable).grid(row=0, column=1, sticky="we")
        self.match_label = ttk.Label(tester_frame, style=HINT_STYLE, wraplength=WRAP_LENGTH, justify=tk.LEFT)
        self.match_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.problem_label = ttk.Label(frame, style=ERROR_STYLE, wraplength=WRAP_LENGTH, justify=tk.LEFT)
        self.problem_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        button_row = ttk.Frame(frame)
        button_row.grid(row=3, column=0, columnspan=2, sticky="we", pady=(12, 0))
        ttk.Button(button_row, text="Open file...", command=self._open_commands_file).pack(side=tk.LEFT)
        self.save_button = ttk.Button(button_row, text="Save", style=ACCENT_BUTTON_STYLE, command=self._save)
        self.save_button.pack(side=tk.RIGHT)
        ttk.Button(button_row, text="Cancel", command=self._close).pack(side=tk.RIGHT, padx=(0, 8))

        self.key_capture = KeyCapture(root, self.window.pause_hotkeys, self.window.resume_hotkeys,
                                      self.window.key_name_for_virtual_key, self._on_capture_changed)
        root.protocol("WM_DELETE_WINDOW", self._close)
        apply_window_chrome(root, theme)
        center_window(root)

    def _selected_entry(self):
        if self.selected_index is None or self.selected_index >= len(self.entries):
            return None
        return self.entries[self.selected_index]

    def _refresh_list(self):
        self.command_list.delete(*self.command_list.get_children())
        for index, entry in enumerate(self.entries):
            self.command_list.insert("", tk.END, iid=str(index), values=(
                "✓" if entry['enabled'] else "", entry['trigger'], ACTION_LABELS[entry['action']], _preview(entry)))
        if self._selected_entry() is not None:
            self.command_list.selection_set(str(self.selected_index))
            self.command_list.see(str(self.selected_index))
        self._refresh_status()

    def _refresh_list_row(self):
        entry = self._selected_entry()
        if entry is None:
            return
        self.command_list.item(str(self.selected_index), values=(
            "✓" if entry['enabled'] else "", entry['trigger'], ACTION_LABELS[entry['action']], _preview(entry)))

    def _select(self, index):
        self.key_capture.end()
        self.selected_index = index
        if index is not None:
            self.command_list.selection_set(str(index))
            self.command_list.see(str(index))
        self._refresh_editor()

    def _on_list_selected(self, event):
        selection = self.command_list.selection()
        index = int(selection[0]) if selection else None
        if index != self.selected_index:
            self.key_capture.end()
            self.selected_index = index
            self._refresh_editor()

    def _refresh_editor(self):
        entry = self._selected_entry()
        self._loading_editor = True
        try:
            editor_state = tk.NORMAL if entry else tk.DISABLED
            for widget in [self.enabled_checkbox, self.trigger_entry, self.test_button, self.duplicate_button,
                           self.delete_button] + self.action_radios:
                widget.config(state=editor_state)
            self.move_up_button.config(state=tk.NORMAL if entry and self.selected_index > 0 else tk.DISABLED)
            self.move_down_button.config(state=tk.NORMAL if entry and self.selected_index < len(self.entries) - 1 else tk.DISABLED)
            self.enabled_variable.set(entry['enabled'] if entry else False)
            self.trigger_variable.set(entry['trigger'] if entry else "")
            self.action_variable.set(entry['action'] if entry else 'hotkey')
            self.run_variable.set(entry['value'] if entry and entry['action'] == 'run' else "")
            self.type_text.delete("1.0", tk.END)
            if entry and entry['action'] == 'type':
                self.type_text.insert("1.0", entry['value'])
            self.type_text.edit_modified(False)
            self._show_value_widgets(entry)
        finally:
            self._loading_editor = False

    def _show_value_widgets(self, entry):
        for widget in (self.hotkey_button, self.type_text, self.run_entry, self.run_warning):
            widget.grid_remove()
        if entry is None:
            return
        if entry['action'] == 'hotkey':
            self._refresh_hotkey_button()
            self.hotkey_button.grid(row=0, column=0, sticky="we")
        elif entry['action'] == 'type':
            self.type_text.grid(row=0, column=0, sticky="nsew")
        else:
            self.run_entry.grid(row=0, column=0, sticky="we")
            self.run_warning.grid(row=1, column=0, sticky="w", pady=(4, 0))

    def _refresh_hotkey_button(self):
        entry = self._selected_entry()
        if self.key_capture.is_active:
            self.hotkey_button.config(text=CAPTURE_TEXT)
        else:
            self.hotkey_button.config(text=beautify_hotkey(entry['value']) if entry and entry['value'] else RECORD_TEXT)

    def _on_editor_changed(self):
        entry = self._selected_entry()
        if self._loading_editor or entry is None:
            return
        entry['enabled'] = self.enabled_variable.get()
        entry['trigger'] = self.trigger_variable.get()
        if entry['action'] == 'run':
            entry['value'] = self.run_variable.get()
        elif entry['action'] == 'type':
            entry['value'] = self.type_text.get("1.0", "end-1c")
        self._refresh_list_row()
        self._refresh_status()

    def _on_type_text_modified(self, event):
        if not self.type_text.edit_modified():
            return
        self.type_text.edit_modified(False)
        self._on_editor_changed()

    def _on_action_changed(self):
        entry = self._selected_entry()
        if self._loading_editor or entry is None or entry['action'] == self.action_variable.get():
            return
        self.key_capture.end()
        entry['action'] = self.action_variable.get()
        entry['value'] = ""
        self._refresh_editor()
        self._refresh_list_row()
        self._refresh_status()

    def _toggle_hotkey_capture(self):
        if self._selected_entry() is None:
            return
        self.key_capture.toggle(self.selected_index, self._store_captured_hotkey)

    def _store_captured_hotkey(self, combination: str):
        entry = self._selected_entry()
        if entry is None or entry['action'] != 'hotkey':
            return
        entry['value'] = combination
        self._refresh_list_row()

    def _on_capture_changed(self):
        self._refresh_hotkey_button()
        self._refresh_status()

    def _add_entry(self):
        self._insert_entry({'trigger': NEW_TRIGGER_TEXT, 'action': 'hotkey', 'value': '', 'enabled': True, 'source_index': None})
        self.trigger_entry.focus_set()
        self.trigger_entry.select_range(0, tk.END)

    def _duplicate_entry(self):
        entry = self._selected_entry()
        if entry is not None:
            self._insert_entry({**entry, 'trigger': f"{entry['trigger']} copy", 'source_index': None})

    def _insert_entry(self, entry: dict):
        self.key_capture.end()
        insert_index = len(self.entries) if self.selected_index is None else self.selected_index + 1
        self.entries.insert(insert_index, entry)
        self.selected_index = insert_index
        self._refresh_list()
        self._select(insert_index)

    def _delete_entry(self):
        if self._selected_entry() is None:
            return
        self.key_capture.end()
        del self.entries[self.selected_index]
        self.selected_index = min(self.selected_index, len(self.entries) - 1) if self.entries else None
        self._refresh_list()
        self._select(self.selected_index)

    def _move_entry(self, offset: int):
        target_index = (self.selected_index or 0) + offset
        if self._selected_entry() is None or not 0 <= target_index < len(self.entries):
            return
        self.key_capture.end()
        self.entries[self.selected_index], self.entries[target_index] = self.entries[target_index], self.entries[self.selected_index]
        self.selected_index = target_index
        self._refresh_list()
        self._select(target_index)

    def _refresh_status(self):
        problems = find_entry_problems(self.entries)
        self._show_message(self.problem_label, "\n".join(problems))
        can_save = not problems and self.entries != self.saved_entries and not self.key_capture.is_active
        self.save_button.config(state=tk.NORMAL if can_save else tk.DISABLED)
        self._refresh_match()

    def _refresh_match(self):
        phrase = self.phrase_variable.get().strip()
        if not phrase:
            self.match_label.config(text="Type a phrase to see which command it would trigger.")
            return
        enabled_commands = [dict(entry_to_command(entry)) for entry in self.entries if entry['enabled'] and entry['trigger'].strip()]
        matches = find_matching_commands(enabled_commands, phrase)
        if not matches:
            self.match_label.config(text="No command matches.")
            return
        message = f"Runs: '{matches[0]['trigger']}'"
        if len(matches) > 1:
            other_triggers = ", ".join(f"'{match['trigger']}'" for match in matches[1:])
            message += f" (longest trigger wins; also contains {other_triggers})"
        self.match_label.config(text=message)

    def _show_message(self, label, text: str):
        label.config(text=text)
        if text:
            label.grid()
        else:
            label.grid_remove()

    def _test_selected_entry(self):
        entry = self._selected_entry()
        if entry is None or not entry['value'].strip():
            return
        if entry['action'] == 'run' and not dialogs.confirm(
                "Run command", f"Run this shell command now?\n\n{entry['value']}", parent=self.root):
            return
        command = entry_to_command(entry)
        self.root.iconify()
        threading.Thread(target=self._run_test_command, args=(command,), daemon=True, name="VoiceCommandTest").start()

    def _run_test_command(self, command: dict):
        time.sleep(TEST_DELAY_SECONDS)
        try:
            self.window.execute_command(command)
        except Exception as e:
            self.logger.error(f"Test of voice command '{command['trigger']}' failed: {e}")
        finally:
            self.command_queue.put(RAISE)

    def _open_commands_file(self):
        try:
            self.window.open_commands_file()
        except Exception as e:
            self.logger.error(f"Failed to open commands file: {e}")

    def _save(self):
        self.key_capture.end()
        if find_entry_problems(self.entries):
            self._refresh_status()
            return
        try:
            self.window.save_command_entries([dict(entry) for entry in self.entries])
        except Exception as e:
            self.logger.error(f"Failed to save voice commands: {e}")
            self._show_message(self.problem_label, f"Could not save voice commands: {e}")
            return
        self._close()

    def _close(self):
        self.key_capture.end()
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
