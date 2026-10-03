import gc
import logging
import queue
import threading
import time
import tkinter as tk

from PIL import Image, ImageTk

from .platform import icons

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 200
ICON_SIZE = 64
BACKGROUND_COLOR = "#222222"
STATUS_COLOR = "#FFFFFF"
VERSION_COLOR = "#AAAAAA"
FONT_FAMILY = "Segoe UI"
QUEUE_POLL_INTERVAL_MS = 50
CLOSE_TIMEOUT_SECONDS = 3.0
CLOSE_REQUEST = object()
ELAPSED_HINT_AFTER_SECONDS = 5


# All Tk objects are created, used and destroyed on the LoadingScreen thread only;
# other threads talk to it exclusively through the message queue.


class LoadingScreen:
    def __init__(self, version: str):
        self.version = version
        self.logger = logging.getLogger(__name__)
        self._message_queue = queue.Queue()
        self._window_thread = None
        self._icon_photo = None
        self._status_text = "Starting..."
        self._status_since = time.monotonic()

    def show(self):
        if self._window_thread and self._window_thread.is_alive():
            return
        self._message_queue = queue.Queue()
        self._window_thread = threading.Thread(target=self._run_window_thread, daemon=True, name="LoadingScreen")
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
            self.logger.warning("Loading screen did not close within timeout")
            return
        self._window_thread = None

    def _run_window_thread(self):
        try:
            self._run_window()
        except Exception as e:
            self.logger.error(f"Loading screen failed: {e}", exc_info=True)
        finally:
            gc.collect()

    def _run_window(self):
        root = tk.Tk()
        try:
            self._build_window(root)
            root.mainloop()
        finally:
            self._icon_photo = None
            try:
                root.destroy()
            except tk.TclError:
                pass

    def _build_window(self, root):
        root.overrideredirect(True)
        root.config(bg=BACKGROUND_COLOR)

        x = (root.winfo_screenwidth() - WINDOW_WIDTH) // 2
        y = (root.winfo_screenheight() - WINDOW_HEIGHT) // 2
        root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

        tk.Label(
            root,
            text=f"Whisper Key {self.version}",
            bg=BACKGROUND_COLOR,
            fg=VERSION_COLOR,
            font=(FONT_FAMILY, 8)
        ).pack(anchor="ne", padx=10, pady=5)

        icon_image = icons.get_tray_icons()["idle"].convert("RGBA").resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
        # Tk labels do not keep a Python reference; without this the icon is freed and goes blank
        self._icon_photo = ImageTk.PhotoImage(icon_image, master=root)
        tk.Label(root, image=self._icon_photo, bg=BACKGROUND_COLOR).pack(pady=10)

        status_label = tk.Label(
            root,
            text="Starting...",
            bg=BACKGROUND_COLOR,
            fg=STATUS_COLOR,
            font=(FONT_FAMILY, 10),
            wraplength=WINDOW_WIDTH - 20
        )
        status_label.pack(pady=10)

        root.bind("<Button-1>", lambda event: root.quit())

        self._bring_to_front_once(root)
        self._process_message_queue(root, status_label)

    def _bring_to_front_once(self, root):
        root.attributes("-topmost", True)
        root.update()
        root.attributes("-topmost", False)

    def _process_message_queue(self, root, status_label):
        try:
            while True:
                message = self._message_queue.get_nowait()
                if message is CLOSE_REQUEST:
                    root.quit()
                    return
                self._status_text = message
                self._status_since = time.monotonic()
        except queue.Empty:
            pass
        status_label.config(text=self._format_status())
        root.after(QUEUE_POLL_INTERVAL_MS, self._process_message_queue, root, status_label)

    def _format_status(self):
        elapsed = int(time.monotonic() - self._status_since)
        if elapsed < ELAPSED_HINT_AFTER_SECONDS:
            return self._status_text
        return f"{self._status_text} ({elapsed} s)"
