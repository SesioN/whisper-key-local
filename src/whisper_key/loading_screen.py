import ctypes
import gc
import logging
import os
import queue
import threading
import time
import tkinter as tk
from ctypes import wintypes

from PIL import Image, ImageTk

from .utils import resolve_asset_path

WINDOW_WIDTH = 440
WINDOW_HEIGHT = 260
FRAME_COUNT = 40
FRAME_INTERVAL_MS = 77
QUEUE_POLL_INTERVAL_MS = 50
STATUS_CENTER_Y = 199
STATUS_WRAP_WIDTH = 368
STATUS_COLOR = "#F2F5FF"
VERSION_COLOR = "#9AA6D6"
VERSION_MARGIN = 10
FONT_FAMILY = "Segoe UI"
STATUS_FONT_SIZE = 11
VERSION_FONT_SIZE = 8
SPLASH_FRAMES_ASSET = "platform/windows/assets/splash_frames.jpg"
SPLASH_EPOCH_VARIABLE = "WHISPER_KEY_SPLASH_EPOCH_MS"
CLOSE_TIMEOUT_SECONDS = 3.0
CLOSE_REQUEST = object()
ELAPSED_HINT_AFTER_SECONDS = 5
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_ROUND = 2


# All Tk objects are created, used and destroyed on the LoadingScreen thread only;
# other threads talk to it exclusively through the message queue.


def load_splash_frames(master):
    with Image.open(resolve_asset_path(SPLASH_FRAMES_ASSET)) as strip:
        strip = strip.convert("RGB")
        return [ImageTk.PhotoImage(strip.crop((index * WINDOW_WIDTH, 0, (index + 1) * WINDOW_WIDTH, WINDOW_HEIGHT)),
                                   master=master)
                for index in range(FRAME_COUNT)]


def splash_epoch_ms():
    try:
        return int(os.environ.pop(SPLASH_EPOCH_VARIABLE))
    except (KeyError, ValueError):
        return int(time.time() * 1000)


def current_frame_index(epoch_ms):
    elapsed_ms = max(0, int(time.time() * 1000) - epoch_ms)
    return (elapsed_ms // FRAME_INTERVAL_MS) % FRAME_COUNT


def round_window_corners(root):
    try:
        user32 = ctypes.WinDLL("user32")
        user32.GetParent.argtypes = [wintypes.HWND]
        user32.GetParent.restype = wintypes.HWND
        dwmapi = ctypes.WinDLL("dwmapi")
        dwmapi.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        window_handle = user32.GetParent(root.winfo_id()) or root.winfo_id()
        corner = ctypes.c_int(DWMWCP_ROUND)
        dwmapi.DwmSetWindowAttribute(window_handle, DWMWA_WINDOW_CORNER_PREFERENCE, ctypes.byref(corner),
                                     ctypes.sizeof(corner))
    except OSError:
        pass


class LoadingScreen:
    def __init__(self, version: str):
        self.version = version
        self.logger = logging.getLogger(__name__)
        self._message_queue = queue.Queue()
        self._window_thread = None
        self._frames = []
        self._animation_epoch_ms = 0
        self._status_text = "Starting Whisper Key..."
        self._status_since = time.monotonic()

    def show(self):
        if self._window_thread and self._window_thread.is_alive():
            return
        self._message_queue = queue.Queue()
        self._status_since = time.monotonic()
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
            self._frames = []
            try:
                root.destroy()
            except tk.TclError:
                pass

    def _build_window(self, root):
        root.overrideredirect(True)
        root.title("Whisper Key")

        x = (root.winfo_screenwidth() - WINDOW_WIDTH) // 2
        y = (root.winfo_screenheight() - WINDOW_HEIGHT) // 2
        root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

        canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, highlightthickness=0, bd=0, bg="#04060f")
        canvas.pack(fill="both", expand=True)

        self._frames = load_splash_frames(root)
        self._animation_epoch_ms = splash_epoch_ms()
        background = canvas.create_image(0, 0, anchor="nw", image=self._frames[0])
        canvas.create_text(WINDOW_WIDTH - VERSION_MARGIN, VERSION_MARGIN - 2, anchor="ne", text=self.version,
                           fill=VERSION_COLOR, font=(FONT_FAMILY, VERSION_FONT_SIZE))
        status = canvas.create_text(WINDOW_WIDTH // 2, STATUS_CENTER_Y, anchor="center", text=self._status_text,
                                    fill=STATUS_COLOR, font=(FONT_FAMILY, STATUS_FONT_SIZE),
                                    width=STATUS_WRAP_WIDTH, justify="center")

        root.bind("<Button-1>", lambda event: root.quit())

        root.update_idletasks()
        round_window_corners(root)
        self._bring_to_front_once(root)
        self._animate(root, canvas, background)
        self._process_message_queue(root, canvas, status)

    def _bring_to_front_once(self, root):
        root.attributes("-topmost", True)
        root.update()
        root.attributes("-topmost", False)

    def _animate(self, root, canvas, background):
        canvas.itemconfigure(background, image=self._frames[current_frame_index(self._animation_epoch_ms)])
        root.after(FRAME_INTERVAL_MS, self._animate, root, canvas, background)

    def _process_message_queue(self, root, canvas, status):
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
        canvas.itemconfigure(status, text=self._format_status())
        root.after(QUEUE_POLL_INTERVAL_MS, self._process_message_queue, root, canvas, status)

    def _format_status(self):
        elapsed = int(time.monotonic() - self._status_since)
        if elapsed < ELAPSED_HINT_AFTER_SECONDS:
            return self._status_text
        return f"{self._status_text} ({elapsed} s)"
