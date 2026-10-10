import logging
import queue
import sys
import threading
import time
from typing import Callable

WINDOW_WIDTH = 460
WINDOW_HEIGHT = 170
QUEUE_POLL_INTERVAL_MS = 200
CLOSE_TIMEOUT_SECONDS = 3.0
CLOSE_REQUEST = object()


class InstallProgressWindow:
    def __init__(self, title: str, on_cancel: Callable[[], None]):
        self.title = title
        self.on_cancel = on_cancel
        self.logger = logging.getLogger(__name__)
        self._message_queue = queue.Queue()
        self._window_thread = None
        self._status_text = "Starting..."
        self._started_at = time.monotonic()
        self._cancel_requested = False

    def show(self):
        if sys.platform != "win32":
            return
        self._window_thread = threading.Thread(target=self._run_window_thread, daemon=True, name="InstallProgress")
        self._window_thread.start()

    def set_status(self, text: str):
        if self._window_thread and self._window_thread.is_alive():
            self._message_queue.put(text)

    def close(self):
        if not self._window_thread:
            return
        self._message_queue.put(CLOSE_REQUEST)
        self._window_thread.join(timeout=CLOSE_TIMEOUT_SECONDS)
        if self._window_thread.is_alive():
            self.logger.warning("Install progress window did not close within timeout")
        self._window_thread = None

    def _run_window_thread(self):
        try:
            self._run_window()
        except Exception as e:
            self.logger.error(f"Install progress window failed: {e}", exc_info=True)

    def _run_window(self):
        import tkinter as tk
        from tkinter import ttk

        from .themed_widgets import HINT_STYLE, TITLE_STYLE, apply_theme, apply_window_chrome, center_window

        root = tk.Tk()
        try:
            root.title(self.title)
            root.resizable(False, False)
            theme = apply_theme(root)
            root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
            root.protocol("WM_DELETE_WINDOW", root.iconify)

            ttk.Label(root, text=self.title, style=TITLE_STYLE).pack(pady=(14, 6))
            status_label = ttk.Label(root, text=self._status_text, wraplength=WINDOW_WIDTH - 30, justify="center")
            status_label.pack(pady=4)
            ttk.Label(root, text="This can take a while. Keep Whisper Key running until it finishes.",
                      style=HINT_STYLE).pack(pady=4)
            cancel_button = ttk.Button(root, text="Cancel", width=10,
                                       command=lambda: self._request_cancel(cancel_button))
            cancel_button.pack(pady=8)

            apply_window_chrome(root, theme)
            center_window(root)
            root.attributes("-topmost", True)
            root.update()
            root.attributes("-topmost", False)
            self._process_message_queue(root, status_label)
            root.mainloop()
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass

    def _request_cancel(self, cancel_button):
        if self._cancel_requested:
            return
        self._cancel_requested = True
        cancel_button.configure(state="disabled", text="Cancelling...")
        threading.Thread(target=self.on_cancel, daemon=True).start()

    def _process_message_queue(self, root, status_label):
        try:
            while True:
                message = self._message_queue.get_nowait()
                if message is CLOSE_REQUEST:
                    root.quit()
                    return
                self._status_text = message
        except queue.Empty:
            pass
        elapsed = int(time.monotonic() - self._started_at)
        status_label.config(text=f"{self._status_text}\n({elapsed // 60}:{elapsed % 60:02d} elapsed)")
        root.after(QUEUE_POLL_INTERVAL_MS, self._process_message_queue, root, status_label)
