import gc
import logging
import queue
import re
import threading
import tkinter as tk
from typing import Callable, Optional

from PIL import Image, ImageTk

from .platform import icons, window_style

SIZES = {
    "small": {"icon": 32, "font": 8, "padding_x": 8, "padding_y": 4, "spacing": 5},
    "medium": {"icon": 48, "font": 9, "padding_x": 12, "padding_y": 6, "spacing": 8},
    "big": {"icon": 64, "font": 10, "padding_x": 15, "padding_y": 8, "spacing": 10},
}
DEFAULT_SIZE = "big"
DEFAULT_SCREEN_MARGIN = 150
DEFAULT_SCREEN_EDGE_GAP = 20
POSITION_PATTERN = re.compile(r"^\+(-?\d+)\+(-?\d+)$")
DRAG_THRESHOLD_PIXELS = 4
TRANSPARENT_KEY_COLOR = "#010203"
OPAQUE_ALPHA_THRESHOLD = 128
HIT_TARGET_ALPHA = 0.01
HIT_TARGET_COLOR = "#000000"
MOVABLE_APPEARANCE = {"text": "Movable", "fg": "#FFD700", "bg": "#222222"}
LOCKED_APPEARANCE = {"text": "Locked", "fg": "#FFFFFF", "bg": "#880000"}
LOCK_LABEL_WIDTH = 9
UNMUTED_APPEARANCE = {"text": "Mic on", "fg": "#7CDB7C", "bg": "#222222"}
MUTED_APPEARANCE = {"text": "Muted", "fg": "#FFFFFF", "bg": "#D32F2F"}
MUTE_LABEL_WIDTH = 7
FONT_FAMILY = "Segoe UI"
QUEUE_POLL_INTERVAL_MS = 100
HIDDEN_QUEUE_POLL_INTERVAL_MS = 500
STOP_TIMEOUT_SECONDS = 3.0

SHOW = "show"
HIDE = "hide"
REFRESH_ICON = "refresh_icon"
REFRESH_MUTE = "refresh_mute"
RESIZE = "resize"
SAVE_POSITION = "save_position"
CLOSE = "close"


class FloatingWidget:
    def __init__(self,
                 on_click: Callable[[], None],
                 on_position_changed: Callable[[str], None],
                 on_mute_click: Callable[[], None],
                 on_lock_changed: Callable[[bool], None],
                 size: str = DEFAULT_SIZE,
                 save_position: bool = False,
                 position: Optional[str] = None,
                 locked: bool = False):
        self.on_click = on_click
        self.on_position_changed = on_position_changed
        self.on_mute_click = on_mute_click
        self.on_lock_changed = on_lock_changed
        self.size = size if size in SIZES else DEFAULT_SIZE
        self.save_position = save_position
        self.position = position
        self.state = "idle"
        self.muted = False
        self.logger = logging.getLogger(__name__)

        self._command_queue = queue.Queue()
        self._window_thread = None
        self._thread_lock = threading.Lock()
        self._click_thread = None
        self._mute_click_thread = None

        self._root = None
        self._icon_label = None
        self._lock_label = None
        self._mute_label = None
        self._controls_frame = None
        self._hit_target = None
        self._icon_photos = {}
        self._locked = locked
        self._dragging = False
        self._press_x_root = 0
        self._press_y_root = 0
        self._drag_offset_x = 0
        self._drag_offset_y = 0

    def show(self):
        with self._thread_lock:
            if not self._window_thread:
                self._command_queue = queue.Queue()
                self._window_thread = threading.Thread(target=self._run_window_thread, daemon=True, name="FloatingWidget")
                self._window_thread.start()
            self._command_queue.put(SHOW)

    def hide(self):
        self._send_to_window(HIDE)

    def update_state(self, new_state: str):
        self.state = new_state
        self._send_to_window(REFRESH_ICON)

    def set_muted(self, muted: bool):
        self.muted = muted
        self._send_to_window(REFRESH_MUTE)

    def set_size(self, size: str):
        if size not in SIZES:
            return
        self.size = size
        self._send_to_window(RESIZE)

    def set_save_position(self, save_position: bool):
        self.save_position = save_position
        if save_position:
            self._send_to_window(SAVE_POSITION)

    def stop(self):
        with self._thread_lock:
            window_thread = self._window_thread
            self._window_thread = None
            if window_thread:
                self._command_queue.put(CLOSE)
        if not window_thread:
            return
        window_thread.join(timeout=STOP_TIMEOUT_SECONDS)
        if window_thread.is_alive():
            self.logger.warning("Floating widget did not close within timeout")

    def _send_to_window(self, command: str):
        with self._thread_lock:
            if self._window_thread:
                self._command_queue.put(command)

    def _run_window_thread(self):
        try:
            self._build_window()
            self._root.mainloop()
        except Exception as e:
            self.logger.error(f"Floating widget failed: {e}")
        finally:
            self._free_tk_objects_on_owning_thread()
            with self._thread_lock:
                if self._window_thread is threading.current_thread():
                    self._window_thread = None

    def _free_tk_objects_on_owning_thread(self):
        if self._root:
            try:
                self._root.destroy()
            except tk.TclError:
                pass
        self._root = None
        self._icon_label = None
        self._lock_label = None
        self._mute_label = None
        self._controls_frame = None
        self._hit_target = None
        self._icon_photos = {}
        gc.collect()

    def _build_window(self):
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-transparentcolor", TRANSPARENT_KEY_COLOR)
        self._root.config(bg=TRANSPARENT_KEY_COLOR)

        self._icon_label = tk.Label(self._root, bg=TRANSPARENT_KEY_COLOR, bd=0, highlightthickness=0, cursor="hand2")
        self._icon_label.pack()
        self._bind_icon_mouse_handlers(self._icon_label)

        self._controls_frame = tk.Frame(self._root, bg=TRANSPARENT_KEY_COLOR, bd=0, highlightthickness=0)
        self._controls_frame.pack()

        self._lock_label = tk.Label(self._controls_frame, bd=0, cursor="hand2", width=LOCK_LABEL_WIDTH)
        self._lock_label.pack(side=tk.LEFT)
        self._lock_label.bind("<Button-1>", self._on_lock_click)

        self._mute_label = tk.Label(self._controls_frame, bd=0, cursor="hand2", width=MUTE_LABEL_WIDTH)
        self._mute_label.pack(side=tk.LEFT)
        self._mute_label.bind("<Button-1>", self._on_mute_click)

        self._apply_size()
        self._apply_lock_appearance()
        self._apply_mute_appearance()
        self._root.update_idletasks()
        self._root.geometry(self._initial_position())
        self._root.update_idletasks()
        window_style.prevent_focus_steal(self._root.winfo_id())
        self._build_hit_target()

        self._root.after(QUEUE_POLL_INTERVAL_MS, self._process_command_queue)

    def _bind_icon_mouse_handlers(self, widget: tk.Misc):
        widget.bind("<Button-1>", self._on_icon_press)
        widget.bind("<B1-Motion>", self._on_icon_drag)
        widget.bind("<ButtonRelease-1>", self._on_icon_release)

    def _build_hit_target(self):
        self._hit_target = tk.Toplevel(self._root, bg=HIT_TARGET_COLOR, cursor="hand2")
        self._hit_target.withdraw()
        self._hit_target.overrideredirect(True)
        self._hit_target.attributes("-topmost", True)
        self._hit_target.attributes("-alpha", HIT_TARGET_ALPHA)
        self._bind_icon_mouse_handlers(self._hit_target)
        self._hit_target.update_idletasks()
        window_style.prevent_focus_steal(self._hit_target.winfo_id())

    def _sync_hit_target(self):
        if self._root.state() != "normal":
            self._hit_target.withdraw()
            return
        self._root.update_idletasks()
        self._place_hit_target(self._root.winfo_x(), self._root.winfo_y())
        if self._hit_target.state() != "normal":
            self._hit_target.deiconify()
            self._root.lift()

    def _place_hit_target(self, window_x: int, window_y: int):
        self._hit_target.geometry(
            f"{self._icon_label.winfo_width()}x{self._icon_label.winfo_height()}"
            f"+{window_x + self._icon_label.winfo_x()}+{window_y + self._icon_label.winfo_y()}"
        )

    def _initial_position(self) -> str:
        if self.save_position and self._saved_position_is_on_screen():
            return self.position
        x = self._root.winfo_screenwidth() - max(DEFAULT_SCREEN_MARGIN, self._root.winfo_reqwidth() + DEFAULT_SCREEN_EDGE_GAP)
        y = self._root.winfo_screenheight() - max(DEFAULT_SCREEN_MARGIN, self._root.winfo_reqheight() + DEFAULT_SCREEN_EDGE_GAP)
        return f"+{x}+{y}"

    def _saved_position_is_on_screen(self) -> bool:
        if not isinstance(self.position, str):
            return False
        match = POSITION_PATTERN.match(self.position)
        if not match:
            return False
        x, y = int(match.group(1)), int(match.group(2))
        right = x + self._root.winfo_reqwidth() - 1
        bottom = y + self._root.winfo_reqheight() - 1
        return window_style.is_point_on_screen(x, y) and window_style.is_point_on_screen(right, bottom)

    def _load_icon_photos(self, icon_size: int) -> dict:
        photos = {}
        for state, image in icons.get_tray_icons().items():
            resized = image.convert("RGBA").resize((icon_size, icon_size), Image.LANCZOS)
            hard_edged_alpha = resized.getchannel("A").point(lambda alpha: 255 if alpha >= OPAQUE_ALPHA_THRESHOLD else 0)
            resized.putalpha(hard_edged_alpha)
            photos[state] = ImageTk.PhotoImage(resized, master=self._root)
        return photos

    def _apply_size(self):
        dimensions = SIZES[self.size]
        self._icon_photos = self._load_icon_photos(dimensions["icon"])
        self._icon_label.pack_configure(pady=(0, dimensions["spacing"]))
        for control_label in (self._lock_label, self._mute_label):
            control_label.config(
                font=(FONT_FAMILY, dimensions["font"], "bold"),
                padx=dimensions["padding_x"],
                pady=dimensions["padding_y"]
            )
        self._mute_label.pack_configure(padx=(dimensions["spacing"] // 2, 0))
        self._apply_icon()

    def _apply_icon(self):
        photo = self._icon_photos.get(self.state, self._icon_photos["idle"])
        self._icon_label.config(image=photo)

    def _apply_lock_appearance(self):
        self._lock_label.config(**(LOCKED_APPEARANCE if self._locked else MOVABLE_APPEARANCE))

    def _apply_mute_appearance(self):
        self._mute_label.config(**(MUTED_APPEARANCE if self.muted else UNMUTED_APPEARANCE))

    def _process_command_queue(self):
        try:
            while True:
                command = self._command_queue.get_nowait()
                if command == CLOSE:
                    self._root.quit()
                    return
                self._handle_command_safely(command)
        except queue.Empty:
            pass
        poll_interval = QUEUE_POLL_INTERVAL_MS if self._root.state() == "normal" else HIDDEN_QUEUE_POLL_INTERVAL_MS
        self._root.after(poll_interval, self._process_command_queue)

    def _handle_command_safely(self, command: str):
        try:
            self._handle_command(command)
        except Exception as e:
            self.logger.error(f"Floating widget command '{command}' failed: {e}")

    def _handle_command(self, command: str):
        if command == SHOW:
            self._root.deiconify()
            self._sync_hit_target()
        elif command == HIDE:
            self._root.withdraw()
            self._sync_hit_target()
        elif command == REFRESH_ICON:
            self._apply_icon()
        elif command == REFRESH_MUTE:
            self._apply_mute_appearance()
        elif command == RESIZE:
            self._apply_size()
            self._sync_hit_target()
        elif command == SAVE_POSITION:
            self._report_position()

    def _report_position(self):
        if self._root.state() != "normal":
            return
        self.position = f"+{self._root.winfo_x()}+{self._root.winfo_y()}"
        self.on_position_changed(self.position)

    def _on_lock_click(self, event):
        self._locked = not self._locked
        self._apply_lock_appearance()
        try:
            self.on_lock_changed(self._locked)
        except Exception:
            self.logger.exception("Floating widget lock handler failed")

    def _on_mute_click(self, event):
        if self._mute_click_thread and self._mute_click_thread.is_alive():
            return
        self._mute_click_thread = threading.Thread(target=self._run_mute_callback, daemon=True, name="FloatingWidgetMute")
        self._mute_click_thread.start()

    def _run_mute_callback(self):
        try:
            self.on_mute_click()
        except Exception:
            self.logger.exception("Floating widget mute handler failed")

    def _on_icon_press(self, event):
        self._dragging = False
        self._press_x_root = event.x_root
        self._press_y_root = event.y_root
        self._drag_offset_x = event.x_root - self._root.winfo_x()
        self._drag_offset_y = event.y_root - self._root.winfo_y()

    def _on_icon_drag(self, event):
        if not self._dragging:
            moved = max(abs(event.x_root - self._press_x_root), abs(event.y_root - self._press_y_root))
            if moved < DRAG_THRESHOLD_PIXELS:
                return
            self._dragging = True
        if not self._locked:
            window_x = event.x_root - self._drag_offset_x
            window_y = event.y_root - self._drag_offset_y
            self._root.geometry(f"+{window_x}+{window_y}")
            self._place_hit_target(window_x, window_y)

    def _on_icon_release(self, event):
        self._root.lift()
        if self._dragging:
            self._dragging = False
            if self.save_position and not self._locked:
                self._handle_command_safely(SAVE_POSITION)
            return
        if self._click_thread and self._click_thread.is_alive():
            return
        self._click_thread = threading.Thread(target=self._run_click_callback, daemon=True, name="FloatingWidgetClick")
        self._click_thread.start()

    def _run_click_callback(self):
        try:
            self.on_click()
        except Exception:
            self.logger.exception("Floating widget click handler failed")
