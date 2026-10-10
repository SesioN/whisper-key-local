import ctypes
import logging
import queue
import threading
import time
import tkinter as tk
from ctypes import wintypes
from tkinter import font as tkfont
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageTk

from .ui_theme import load_theme, rgb_color
from .tray_menu_model import Action, Choice, Footer, Header, Section, Slider, Submenu, Toggle, resolve

TOGGLE = "toggle"
REFRESH = "refresh"
STOP = "stop"

POLL_INTERVAL_MS = 25
FOCUS_GRACE_SECONDS = 0.3
REOPEN_GUARD_SECONDS = 0.35
START_TIMEOUT_SECONDS = 3.0
STOP_TIMEOUT_SECONDS = 3.0

WIDTH = 336
EDGE_MARGIN = 10
ROW_HEIGHT = 36
SECTION_HEIGHT = 30
GROUP_GAP_HEIGHT = 9
LINE_HEIGHT = 9
HEADER_HEIGHT = 64
BACK_ROW_HEIGHT = 44
SEGMENTED_HEIGHT = 68
SLIDER_HEIGHT = 60
SLIDER_TRACK_THICKNESS = 4
SLIDER_GRADIENT_THICKNESS = 8
SLIDER_THUMB_SIZE = 18
ROW_INSET = 4
CONTENT_PADDING = 12
ICON_COLUMN = 30
PREVIEW_SIZE = 26
CORNER_RADIUS = 5
SUPERSAMPLE = 4

ICON_BACK = ""
ICON_CHEVRON = ""
ICON_CHECK = ""
ICON_MICROPHONE = ""
ICON_MICROPHONE_OFF = ""

MONITOR_DEFAULTTONEAREST = 2
MDT_EFFECTIVE_DPI = 0
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_BORDER_COLOR = 34
DWMWCP_ROUND = 2
ASFW_ANY = 0xFFFFFFFF
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
user32.GetCursorPos.restype = wintypes.BOOL
user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
user32.MonitorFromPoint.restype = ctypes.c_void_p
user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MONITORINFO)]
user32.GetMonitorInfoW.restype = wintypes.BOOL
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.AllowSetForegroundWindow.argtypes = [wintypes.DWORD]
user32.AllowSetForegroundWindow.restype = wintypes.BOOL
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p

shcore = ctypes.WinDLL("shcore")
shcore.GetDpiForMonitor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
shcore.GetDpiForMonitor.restype = ctypes.c_long

dwmapi = ctypes.WinDLL("dwmapi")
dwmapi.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long


class TrayFlyout:
    def __init__(self, build_menu: Callable[[], list]):
        self.build_menu = build_menu
        self.logger = logging.getLogger(__name__)
        self.failed = False
        self._commands = queue.Queue()
        self._callbacks = queue.Queue()
        self._thread = None
        self._ready = threading.Event()
        self._thread_lock = threading.Lock()
        threading.Thread(target=self._run_callbacks, daemon=True, name="TrayFlyoutActions").start()

    def toggle(self, tray_window: int = 0) -> bool:
        if self.failed or not self._ensure_thread():
            return False
        if tray_window:
            user32.SetForegroundWindow(tray_window)
        user32.AllowSetForegroundWindow(ASFW_ANY)
        self._commands.put(TOGGLE)
        return True

    def refresh(self):
        if self._thread:
            self._commands.put(REFRESH)

    def run_callback(self, callback: Callable[[], None]):
        self._callbacks.put(callback)

    def stop(self):
        with self._thread_lock:
            thread = self._thread
        if not thread:
            return
        self._commands.put(STOP)
        thread.join(timeout=STOP_TIMEOUT_SECONDS)
        if thread.is_alive():
            self.logger.warning("Tray menu did not close within timeout")

    def _ensure_thread(self) -> bool:
        with self._thread_lock:
            if not self._thread:
                self._ready.clear()
                self._thread = threading.Thread(target=self._run_thread, daemon=True, name="TrayFlyout")
                self._thread.start()
        self._ready.wait(START_TIMEOUT_SECONDS)
        return not self.failed

    def _run_thread(self):
        root = None
        try:
            user32.SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
            root = tk.Tk()
            _FlyoutView(self, root, self._commands)
            self._ready.set()
            root.mainloop()
        except Exception as e:
            self.logger.error(f"Tray menu failed: {e}")
            self.failed = True
        finally:
            self._ready.set()
            if root:
                try:
                    root.destroy()
                except tk.TclError:
                    pass
            with self._thread_lock:
                self._thread = None

    def _run_callbacks(self):
        while True:
            callback = self._callbacks.get()
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Tray menu action failed: {e}")
            self.refresh()


class _FlyoutView:
    def __init__(self, flyout: TrayFlyout, root: tk.Tk, commands: queue.Queue):
        self.flyout = flyout
        self.root = root
        self.commands = commands
        self.logger = flyout.logger
        self.visible = False
        self.stopping = False
        self.had_focus = False
        self.shown_at = 0.0
        self.hidden_at = 0.0
        self.keyboard_mode = False
        self.page_stack = []
        self.rows = []
        self.focusables = []
        self.highlighted = None
        self.container = None
        self.body_canvas = None
        self.body_scrollable = False
        self.cursor = (0, 0)
        self.work_area = (0, 0, 0, 0)
        self.scale = 1.0
        self.theme = load_theme()
        self.image_cache = {}
        self.preview_cache = {}

        root.withdraw()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.title("Whisper Key")
        root.update_idletasks()
        self.hwnd = int(root.wm_frame(), 16)
        self.font_families = set(tkfont.families(root))
        root.bind_all("<Key>", self._on_key)
        root.bind_all("<MouseWheel>", self._on_wheel)
        root.bind_all("<Motion>", self._on_motion)
        root.after(POLL_INTERVAL_MS, self._poll)

    def px(self, value: float) -> int:
        return int(round(value * self.scale))

    @property
    def width(self) -> int:
        return self.px(WIDTH)

    def _poll(self):
        try:
            self._process_commands()
            if self.visible:
                self._close_when_focus_lost()
        except Exception as e:
            self.logger.error(f"Tray menu update failed: {e}")
        finally:
            if not self.stopping:
                self.root.after(POLL_INTERVAL_MS, self._poll)

    def _process_commands(self):
        try:
            while True:
                command = self.commands.get_nowait()
                if command == STOP:
                    self.stopping = True
                    self.root.quit()
                    return
                if command == TOGGLE:
                    if self.visible:
                        self.hide()
                    elif time.monotonic() - self.hidden_at > REOPEN_GUARD_SECONDS:
                        self.show()
                elif command == REFRESH and self.visible:
                    self.refresh()
        except queue.Empty:
            pass

    def _close_when_focus_lost(self):
        if user32.GetForegroundWindow() == self.hwnd:
            self.had_focus = True
            return
        if time.monotonic() - self.shown_at <= FOCUS_GRACE_SECONDS:
            return
        if self.had_focus or self._mouse_pressed_outside():
            self.hide()

    def _mouse_pressed_outside(self) -> bool:
        if not (user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000 or user32.GetAsyncKeyState(VK_RBUTTON) & 0x8000):
            return False
        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        left, top = self.root.winfo_rootx(), self.root.winfo_rooty()
        inside = left <= point.x < left + self.root.winfo_width() and top <= point.y < top + self.root.winfo_height()
        return not inside

    def show(self):
        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        self.cursor = (point.x, point.y)
        monitor = user32.MonitorFromPoint(point, MONITOR_DEFAULTTONEAREST)
        info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
        user32.GetMonitorInfoW(monitor, ctypes.byref(info))
        self.work_area = (info.rcWork.left, info.rcWork.top, info.rcWork.right, info.rcWork.bottom)
        dpi_x, dpi_y = ctypes.c_uint(96), ctypes.c_uint(96)
        shcore.GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
        new_scale = dpi_x.value / 96.0
        new_theme = load_theme()
        if new_scale != self.scale or new_theme != self.theme:
            self.image_cache.clear()
            self.preview_cache.clear()
        self.scale = new_scale
        self.theme = new_theme
        self._create_fonts()
        self._apply_window_style()
        self.page_stack = []
        self.highlighted = None
        self.keyboard_mode = False
        self._build_page()
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        user32.SetForegroundWindow(self.hwnd)
        self.visible = True
        self.had_focus = False
        self.shown_at = time.monotonic()

    def hide(self):
        if not self.visible:
            return
        self.visible = False
        self.hidden_at = time.monotonic()
        self.root.withdraw()

    def _create_fonts(self):
        text_family = "Segoe UI Variable Text" if "Segoe UI Variable Text" in self.font_families else "Segoe UI"
        display_family = "Segoe UI Variable Display" if "Segoe UI Variable Display" in self.font_families else "Segoe UI"
        icon_family = "Segoe Fluent Icons" if "Segoe Fluent Icons" in self.font_families else "Segoe MDL2 Assets"
        self.uses_fluent_icons = icon_family == "Segoe Fluent Icons"
        self.font_body = tkfont.Font(root=self.root, family=text_family, size=-self.px(14))
        self.font_small = tkfont.Font(root=self.root, family=text_family, size=-self.px(12))
        self.font_section = tkfont.Font(root=self.root, family=text_family, size=-self.px(12), weight="bold")
        self.font_title = tkfont.Font(root=self.root, family=display_family, size=-self.px(17), weight="bold")
        self.font_page_title = tkfont.Font(root=self.root, family=display_family, size=-self.px(15), weight="bold")
        self.font_icon = tkfont.Font(root=self.root, family=icon_family, size=-self.px(16))
        self.font_icon_small = tkfont.Font(root=self.root, family=icon_family, size=-self.px(12))
        self.font_check = tkfont.Font(root=self.root, family=icon_family, size=-self.px(14))

    def microphone_icon(self, muted: bool) -> str:
        if muted and self.uses_fluent_icons:
            return ICON_MICROPHONE_OFF
        return ICON_MICROPHONE

    def _apply_window_style(self):
        corner = ctypes.c_int(DWMWCP_ROUND)
        result = dwmapi.DwmSetWindowAttribute(self.hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, ctypes.byref(corner), ctypes.sizeof(corner))
        self.dwm_frame = result == 0
        if self.dwm_frame:
            r, g, b = rgb_color(self.theme.border)
            border = wintypes.DWORD(r | (g << 8) | (b << 16))
            dwmapi.DwmSetWindowAttribute(self.hwnd, DWMWA_BORDER_COLOR, ctypes.byref(border), ctypes.sizeof(border))

    def fit_text(self, font: tkfont.Font, text: str, max_width: int) -> str:
        if max_width <= 0:
            return ""
        if font.measure(text) <= max_width:
            return text
        ellipsis = "…"
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if font.measure(text[:middle].rstrip() + ellipsis) <= max_width:
                low = middle
            else:
                high = middle - 1
        return text[:low].rstrip() + ellipsis

    def rounded_image(self, width: int, height: int, radius: int, fill: str, outline: Optional[str] = None,
                      outline_width: int = 0) -> ImageTk.PhotoImage:
        key = ("rounded", width, height, radius, fill, outline, outline_width)
        image = self.image_cache.get(key)
        if image is None:
            big = Image.new("RGBA", (width * SUPERSAMPLE, height * SUPERSAMPLE), (0, 0, 0, 0))
            draw = ImageDraw.Draw(big)
            draw.rounded_rectangle((0, 0, big.width - 1, big.height - 1), radius=radius * SUPERSAMPLE,
                                   fill=rgb_color(fill) + (255,),
                                   outline=rgb_color(outline) + (255,) if outline else None,
                                   width=outline_width * SUPERSAMPLE)
            image = ImageTk.PhotoImage(big.resize((width, height), Image.LANCZOS), master=self.root)
            self.image_cache[key] = image
        return image

    def switch_image(self, on: bool, enabled: bool) -> ImageTk.PhotoImage:
        theme = self.theme
        key = ("switch", on, enabled, self.scale)
        image = self.image_cache.get(key)
        if image is None:
            width, height = self.px(40), self.px(20)
            s = SUPERSAMPLE
            big = Image.new("RGBA", (width * s, height * s), (0, 0, 0, 0))
            draw = ImageDraw.Draw(big)
            box = (0, 0, big.width - 1, big.height - 1)
            if on:
                track = theme.accent if enabled else theme.disabled
                draw.rounded_rectangle(box, radius=big.height // 2, fill=rgb_color(track) + (255,))
                knob_radius = height * s * 0.3
                knob_x = big.width - big.height / 2
                knob_color = theme.on_accent
            else:
                stroke = theme.secondary if enabled else theme.disabled
                draw.rounded_rectangle(box, radius=big.height // 2, fill=rgb_color(theme.bg) + (255,),
                                       outline=rgb_color(stroke) + (255,), width=max(1, self.px(1)) * s)
                knob_radius = height * s * 0.24
                knob_x = big.height / 2
                knob_color = stroke
            knob_y = big.height / 2
            draw.ellipse((knob_x - knob_radius, knob_y - knob_radius, knob_x + knob_radius, knob_y + knob_radius),
                         fill=rgb_color(knob_color) + (255,))
            image = ImageTk.PhotoImage(big.resize((width, height), Image.LANCZOS), master=self.root)
            self.image_cache[key] = image
        return image

    def slider_thumb_image(self) -> ImageTk.PhotoImage:
        theme = self.theme
        key = ("slider_thumb", self.scale)
        image = self.image_cache.get(key)
        if image is None:
            size = self.px(SLIDER_THUMB_SIZE)
            s = SUPERSAMPLE
            big = Image.new("RGBA", (size * s, size * s), (0, 0, 0, 0))
            draw = ImageDraw.Draw(big)
            draw.ellipse((0, 0, big.width - 1, big.height - 1), fill=rgb_color(theme.surface_hover) + (255,),
                         outline=rgb_color(theme.border) + (255,), width=max(1, self.px(1)) * s)
            inner = big.width * 0.28
            center = big.width / 2
            draw.ellipse((center - inner, center - inner, center + inner, center + inner),
                         fill=rgb_color(theme.accent) + (255,))
            image = ImageTk.PhotoImage(big.resize((size, size), Image.LANCZOS), master=self.root)
            self.image_cache[key] = image
        return image

    def gradient_image(self, width: int, height: int, colors: tuple) -> ImageTk.PhotoImage:
        key = ("gradient", width, height, colors)
        image = self.image_cache.get(key)
        if image is None:
            for stale_key in [cached for cached in self.image_cache if cached[0] == "gradient"]:
                del self.image_cache[stale_key]
            strip = Image.new("RGB", (len(colors), 1))
            strip.putdata([rgb_color(color) for color in colors])
            stretched = strip.resize((width * SUPERSAMPLE, height * SUPERSAMPLE), Image.BILINEAR).convert("RGBA")
            mask = Image.new("L", stretched.size, 0)
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, mask.width - 1, mask.height - 1),
                                                   radius=mask.height // 2, fill=255)
            stretched.putalpha(mask)
            image = ImageTk.PhotoImage(stretched.resize((width, height), Image.LANCZOS), master=self.root)
            self.image_cache[key] = image
        return image

    def preview_image(self, choice: Choice, value) -> Optional[ImageTk.PhotoImage]:
        key = (choice.label, value, resolve(choice.preview_key))
        if key not in self.preview_cache:
            for stale_key in [cached for cached in self.preview_cache if cached[0] == key[0] and cached[2] != key[2]]:
                del self.preview_cache[stale_key]
            try:
                picture = choice.preview(value, self.px(PREVIEW_SIZE))
                self.preview_cache[key] = ImageTk.PhotoImage(picture, master=self.root) if picture else None
            except Exception as e:
                self.logger.error(f"Could not render preview for {value}: {e}")
                self.preview_cache[key] = None
        return self.preview_cache[key]

    def _current_page(self):
        items = [item for item in self.flyout.build_menu() if item is not None]
        title = None
        resolved_stack = []
        for label in self.page_stack:
            submenu = next((item for item in items if isinstance(item, Submenu) and item.label == label), None)
            if submenu is None:
                break
            items = submenu.resolved_items()
            title = label
            resolved_stack.append(label)
        self.page_stack = resolved_stack
        return title, items

    def _describe(self, title, items):
        top, body, footer = [], [], []
        if title is not None:
            top.append((_BackRow, title))
        for item in items:
            if isinstance(item, Header):
                top.append((_HeaderRow, item))
            elif isinstance(item, Footer):
                footer += [(_ActionRow, action) for action in item.items if action is not None]
            elif isinstance(item, Section):
                body.append((_SectionRow, item.label))
            elif isinstance(item, Action):
                body.append((_ActionRow, item))
            elif isinstance(item, Toggle):
                body.append((_ToggleRow, item))
            elif isinstance(item, Submenu):
                body.append((_NavRow, item))
            elif isinstance(item, Slider):
                body.append((_SliderRow, item))
            elif isinstance(item, Choice):
                options = item.resolved_options()
                if item.style == "segmented":
                    body.append((_SegmentedRow, (item, options)))
                    continue
                if item.label and len(items) > 1:
                    body.append((_SectionRow, item.label))
                for index, option in enumerate(options):
                    if option.starts_group and index > 0:
                        body.append((_GapRow, None))
                    body.append((_RadioRow, (item, option)))
        return top, body, footer

    @staticmethod
    def _signature(descriptions):
        return [(row_class, row_class.key(data)) for row_class, data in descriptions]

    def _build_page(self, keep_highlight_index: Optional[int] = None, keep_scroll: float = 0.0):
        title, items = self._current_page()
        top, body, footer = self._describe(title, items)
        theme = self.theme

        self.highlighted = None
        if self.container:
            self.container.destroy()
        border = 0 if self.dwm_frame else 1
        self.container = tk.Frame(self.root, bg=theme.border, bd=0, padx=border, pady=border)
        self.container.pack(fill="both", expand=True)
        inner = tk.Frame(self.container, bg=theme.bg, bd=0)
        inner.pack(fill="both", expand=True)

        self.rows = []
        top_frame = tk.Frame(inner, bg=theme.bg)
        top_frame.pack(fill="x")
        for row_class, data in top:
            self.rows.append(row_class(self, top_frame, data))
        if top:
            _LineRow(self, top_frame, None)

        self.body_canvas = tk.Canvas(inner, width=self.width, bg=theme.bg, highlightthickness=0, bd=0,
                                     yscrollincrement=1)
        self.body_canvas.pack(fill="x")
        body_frame = tk.Frame(self.body_canvas, bg=theme.bg)
        self.body_canvas.create_window(0, 0, window=body_frame, anchor="nw")
        tk.Frame(body_frame, bg=theme.bg, height=self.px(4), width=self.width).pack(fill="x")
        for row_class, data in body:
            self.rows.append(row_class(self, body_frame, data))
        tk.Frame(body_frame, bg=theme.bg, height=self.px(4), width=self.width).pack(fill="x")

        footer_frame = tk.Frame(inner, bg=theme.bg)
        footer_frame.pack(fill="x")
        if footer:
            _LineRow(self, footer_frame, None)
            for row_class, data in footer:
                self.rows.append(row_class(self, footer_frame, data))
            tk.Frame(footer_frame, bg=theme.bg, height=self.px(4)).pack(fill="x")

        self.page_signature = self._signature(top + body + footer)
        self.focusables = [row for row in self.rows if row.focusable]

        self.root.update_idletasks()
        content_height = body_frame.winfo_reqheight()
        left, top_edge, right, bottom = self.work_area
        margin = self.px(EDGE_MARGIN)
        chrome_height = top_frame.winfo_reqheight() + footer_frame.winfo_reqheight() + 2 * border
        max_body_height = max(self.px(120), bottom - top_edge - 2 * margin - chrome_height)
        body_height = min(content_height, max_body_height)
        self.body_scrollable = content_height > body_height
        self.body_canvas.configure(height=body_height, scrollregion=(0, 0, self.width, content_height))
        self.body_canvas.yview_moveto(keep_scroll)

        width = self.width + 2 * border
        height = chrome_height + body_height
        x = min(max(self.cursor[0] - width, left + margin), right - margin - width)
        y = min(max(self.cursor[1] - height, top_edge + margin), bottom - margin - height)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        if keep_highlight_index is not None and self.focusables:
            self.set_highlight(self.focusables[min(keep_highlight_index, len(self.focusables) - 1)])
        elif self.keyboard_mode:
            self._highlight_first()

    def refresh(self):
        title, items = self._current_page()
        top, body, footer = self._describe(title, items)
        descriptions = top + body + footer
        highlight_index = self.focusables.index(self.highlighted) if self.highlighted in self.focusables else None
        if self._signature(descriptions) == self.page_signature:
            for row, (row_class, data) in zip(self.rows, descriptions):
                row.update(data)
            return
        scroll = self.body_canvas.yview()[0] if self.body_canvas else 0.0
        self._build_page(highlight_index, scroll)

    def push_page(self, label: str):
        self.page_stack.append(label)
        self.highlighted = None
        self._build_page()

    def pop_page(self):
        if not self.page_stack:
            return
        left_label = self.page_stack.pop()
        self.highlighted = None
        self._build_page()
        if self.keyboard_mode:
            for row in self.focusables:
                if isinstance(row, _NavRow) and row.item.label == left_label:
                    self.set_highlight(row)
                    break

    def set_highlight(self, row):
        if row is self.highlighted:
            return
        previous, self.highlighted = self.highlighted, row
        if previous:
            previous.set_highlighted(False)
        if row:
            row.set_highlighted(True)
            self._scroll_into_view(row)

    def _highlight_first(self):
        candidates = [row for row in self.focusables if row.enabled() and not isinstance(row, _BackRow)]
        if candidates:
            self.set_highlight(candidates[0])

    def _scroll_into_view(self, row):
        if not self.body_scrollable or row.canvas.master.master is not self.body_canvas:
            return
        content_height = float(self.body_canvas.cget("scrollregion").split()[3])
        view_top, view_bottom = (fraction * content_height for fraction in self.body_canvas.yview())
        row_top = row.canvas.winfo_y()
        row_bottom = row_top + row.canvas.winfo_reqheight()
        if row_top < view_top:
            self.body_canvas.yview_moveto(row_top / content_height)
        elif row_bottom > view_bottom:
            self.body_canvas.yview_moveto((row_bottom - (view_bottom - view_top)) / content_height)

    def apply_pending_choice(self, choice: Choice, value):
        for row in self.rows:
            if isinstance(row, _RadioRow) and row.choice.label == choice.label:
                row.pending_value = value
                row.draw()

    def _on_motion(self, _event):
        self.keyboard_mode = False

    def _on_wheel(self, event):
        if self.visible and self.body_scrollable:
            self.body_canvas.yview_scroll(int(-event.delta / 120 * self.px(ROW_HEIGHT) * 3), "units")

    def _on_key(self, event):
        if not self.visible:
            return
        self.keyboard_mode = True
        key = event.keysym
        if key == "Escape":
            if self.page_stack:
                self.pop_page()
            else:
                self.hide()
        elif key in ("Up", "Down"):
            self._move_highlight(1 if key == "Down" else -1)
        elif key in ("Return", "space"):
            if self.highlighted and self.highlighted.enabled():
                self.highlighted.activate()
        elif key in ("Left", "Right"):
            if self.highlighted and self.highlighted.handle_arrow(key):
                return
            if key == "Left":
                self.pop_page()
            elif isinstance(self.highlighted, _NavRow):
                self.highlighted.activate()
        elif key == "BackSpace":
            self.pop_page()

    def _move_highlight(self, step: int):
        candidates = [row for row in self.focusables if row.enabled()]
        if not candidates:
            return
        if self.highlighted not in candidates:
            self.set_highlight(candidates[0] if step > 0 else candidates[-1])
            return
        index = (candidates.index(self.highlighted) + step) % len(candidates)
        self.set_highlight(candidates[index])


class _Row:
    focusable = True
    height = ROW_HEIGHT

    def __init__(self, view: _FlyoutView, parent: tk.Misc, data):
        self.view = view
        self.highlighted = False
        self.set_data(data)
        self.canvas = tk.Canvas(parent, width=view.width, height=view.px(self.row_height()), bg=view.theme.bg,
                                highlightthickness=0, bd=0)
        self.canvas.pack(fill="x")
        if self.focusable:
            self.canvas.bind("<Enter>", self._on_enter)
            self.canvas.bind("<ButtonRelease-1>", self._on_click)
        else:
            self.canvas.bind("<Enter>", lambda _event: view.set_highlight(None))
        self.draw()

    @staticmethod
    def key(data):
        return getattr(data, "label", data)

    def row_height(self) -> int:
        return self.height

    def set_data(self, data):
        self.data = data

    def update(self, data):
        self.set_data(data)
        self.draw()

    def enabled(self) -> bool:
        return True

    def set_highlighted(self, highlighted: bool):
        if highlighted != self.highlighted:
            self.highlighted = highlighted
            self.draw()

    def handle_arrow(self, key: str) -> bool:
        return False

    def activate(self, event=None):
        pass

    def _on_enter(self, _event):
        if not self.view.keyboard_mode:
            self.view.set_highlight(self if self.enabled() else None)

    def _on_click(self, event):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if 0 <= event.x < width and 0 <= event.y < height and self.enabled():
            self.activate(event)

    def draw(self):
        canvas, view = self.canvas, self.view
        canvas.delete("all")
        if self.highlighted and self.enabled():
            self.draw_hover()
        self.draw_content()

    def draw_hover(self):
        self.draw_background(self.view.theme.hover)

    def draw_background(self, color: str):
        view = self.view
        inset = view.px(ROW_INSET)
        height = view.px(self.row_height())
        image = view.rounded_image(view.width - 2 * inset, height - view.px(2), view.px(CORNER_RADIUS), color)
        self.canvas.create_image(inset, view.px(1), image=image, anchor="nw")

    def draw_content(self):
        pass

    def text_color(self) -> str:
        return self.view.theme.text if self.enabled() else self.view.theme.disabled

    def middle(self) -> int:
        return self.view.px(self.row_height()) // 2


class _SectionRow(_Row):
    focusable = False
    height = SECTION_HEIGHT

    def draw_content(self):
        view = self.view
        self.canvas.create_text(view.px(ROW_INSET + CONTENT_PADDING), view.px(SECTION_HEIGHT - 8), anchor="sw",
                                text=self.data.upper(), font=view.font_section, fill=view.theme.secondary)


class _GapRow(_Row):
    focusable = False
    height = GROUP_GAP_HEIGHT

    @staticmethod
    def key(data):
        return None


class _LineRow(_Row):
    focusable = False
    height = LINE_HEIGHT

    @staticmethod
    def key(data):
        return None

    def draw_content(self):
        view = self.view
        y = view.px(LINE_HEIGHT) // 2
        self.canvas.create_line(view.px(ROW_INSET), y, view.width - view.px(ROW_INSET), y, fill=view.theme.border)


class _HeaderRow(_Row):
    height = HEADER_HEIGHT

    @staticmethod
    def key(data):
        return "header"

    def set_data(self, data):
        self.data = data
        self.pending_muted = None

    def muted(self) -> bool:
        return self.data.muted() if self.pending_muted is None else self.pending_muted

    def draw_hover(self):
        pass

    def button_box(self):
        view = self.view
        width, height = view.px(44), view.px(34)
        right = view.width - view.px(ROW_INSET + CONTENT_PADDING)
        top = (view.px(HEADER_HEIGHT) - height) // 2
        return right - width, top, width, height

    def draw_content(self):
        view, theme = self.view, self.view.theme
        left = view.px(ROW_INSET + CONTENT_PADDING)
        x, y, width, height = self.button_box()
        self.canvas.create_text(left, view.px(26), anchor="w", text=self.data.title, font=view.font_title,
                                fill=theme.text)
        self.canvas.create_text(left, view.px(46), anchor="w", font=view.font_small, fill=theme.secondary,
                                text=view.fit_text(view.font_small, self.data.subtitle(), x - left - view.px(8)))
        muted = self.muted()
        if muted:
            fill, glyph_color = theme.danger, theme.on_danger
        else:
            fill, glyph_color = (theme.surface_hover if self.highlighted else theme.surface), theme.text
        self.canvas.create_image(x, y, anchor="nw", image=view.rounded_image(width, height, view.px(CORNER_RADIUS), fill))
        self.canvas.create_text(x + width // 2, y + height // 2, text=view.microphone_icon(muted), font=view.font_icon,
                                fill=glyph_color)

    def _on_enter(self, _event):
        pass

    def _on_click(self, event):
        x, y, width, height = self.button_box()
        if x <= event.x < x + width and y <= event.y < y + height:
            self.activate()

    def activate(self, event=None):
        self.pending_muted = not self.muted()
        self.draw()
        self.view.flyout.run_callback(self.data.on_toggle_mute)


class _BackRow(_Row):
    height = BACK_ROW_HEIGHT

    def draw_content(self):
        view = self.view
        left = view.px(ROW_INSET + CONTENT_PADDING)
        middle = self.middle()
        self.canvas.create_text(left, middle, anchor="w", text=ICON_BACK, font=view.font_icon_small, fill=view.theme.text)
        self.canvas.create_text(left + view.px(ICON_COLUMN - 6), middle, anchor="w", text=self.data,
                                font=view.font_page_title, fill=view.theme.text)

    def activate(self, event=None):
        self.view.pop_page()


class _ActionRow(_Row):
    def enabled(self) -> bool:
        return bool(resolve(self.data.enabled))

    def draw_content(self):
        view = self.view
        left = view.px(ROW_INSET + CONTENT_PADDING)
        middle = self.middle()
        if self.data.icon:
            self.canvas.create_text(left, middle, anchor="w", text=self.data.icon, font=view.font_icon,
                                    fill=self.text_color())
        text_left = left + view.px(ICON_COLUMN) if self.data.icon else left
        self.canvas.create_text(text_left, middle, anchor="w", font=view.font_body, fill=self.text_color(),
                                text=view.fit_text(view.font_body, self.data.label, view.width - text_left - left))

    def activate(self, event=None):
        self.view.hide()
        self.view.flyout.run_callback(self.data.on_select)


class _NavRow(_Row):
    def set_data(self, data):
        self.data = data
        self.item = data

    def draw_content(self):
        view, theme = self.view, self.view.theme
        left = view.px(ROW_INSET + CONTENT_PADDING)
        right = view.width - left
        middle = self.middle()
        if self.item.icon:
            self.canvas.create_text(left, middle, anchor="w", text=self.item.icon, font=view.font_icon, fill=theme.text)
        text_left = left + view.px(ICON_COLUMN) if self.item.icon else left
        self.canvas.create_text(right, middle, anchor="e", text=ICON_CHEVRON, font=view.font_icon_small,
                                fill=theme.secondary)
        label_width = view.font_body.measure(self.item.label)
        detail_right = right - view.px(20)
        if self.item.detail:
            detail_left = text_left + label_width + view.px(16)
            detail = view.fit_text(view.font_small, self.item.detail(), detail_right - detail_left)
            self.canvas.create_text(detail_right, middle, anchor="e", text=detail, font=view.font_small,
                                    fill=theme.secondary)
        self.canvas.create_text(text_left, middle, anchor="w", font=view.font_body, fill=theme.text,
                                text=view.fit_text(view.font_body, self.item.label, detail_right - text_left))

    def activate(self, event=None):
        self.view.push_page(self.item.label)


class _ToggleRow(_Row):
    def set_data(self, data):
        self.data = data
        self.pending_on = None

    def is_on(self) -> bool:
        return self.data.is_on() if self.pending_on is None else self.pending_on

    def enabled(self) -> bool:
        return bool(resolve(self.data.enabled))

    def draw_content(self):
        view = self.view
        left = view.px(ROW_INSET + CONTENT_PADDING)
        middle = self.middle()
        switch = view.switch_image(self.is_on(), self.enabled())
        switch_left = view.width - left - switch.width()
        self.canvas.create_image(switch_left, middle, anchor="w", image=switch)
        self.canvas.create_text(left, middle, anchor="w", font=view.font_body, fill=self.text_color(),
                                text=view.fit_text(view.font_body, self.data.label, switch_left - left - view.px(12)))

    def activate(self, event=None):
        target = not self.is_on()
        self.pending_on = target
        self.draw()
        self.view.flyout.run_callback(lambda: self.data.on_change(target))


class _RadioRow(_Row):
    @staticmethod
    def key(data):
        choice, option = data
        return choice.label, option.value, option.label

    def set_data(self, data):
        self.data = data
        self.choice, self.option = data
        self.pending_value = None
        self.preview = self.view.preview_image(self.choice, self.option.value) if self.choice.preview else None

    def row_height(self) -> int:
        return PREVIEW_SIZE + 14 if self.choice.preview else ROW_HEIGHT

    def draw(self):
        self.canvas.delete("all")
        if self.highlighted and self.enabled():
            self.draw_hover()
        elif self.choice.preview and self.selected():
            self.draw_background(self.view.theme.surface)
        self.draw_content()

    def selected(self) -> bool:
        current = self.choice.current() if self.pending_value is None else self.pending_value
        return current == self.option.value

    def enabled(self) -> bool:
        return bool(self.option.enabled)

    def draw_content(self):
        view, theme = self.view, self.view.theme
        left = view.px(ROW_INSET + CONTENT_PADDING)
        right = view.width - left
        middle = self.middle()
        selected = self.selected()
        if self.choice.preview:
            if self.preview:
                self.canvas.create_image(left, middle, anchor="w", image=self.preview)
            text_left = left + view.px(PREVIEW_SIZE + 12)
            if selected:
                self.canvas.create_text(right, middle, anchor="e", text=ICON_CHECK, font=view.font_check,
                                        fill=theme.accent)
            text_right = right - view.px(24)
        else:
            if selected:
                self.canvas.create_text(left, middle, anchor="w", text=ICON_CHECK, font=view.font_check,
                                        fill=theme.accent if self.enabled() else theme.disabled)
            text_left = left + view.px(ICON_COLUMN)
            text_right = right
        self.canvas.create_text(text_left, middle, anchor="w", font=view.font_body, fill=self.text_color(),
                                text=view.fit_text(view.font_body, self.option.label, text_right - text_left))

    def activate(self, event=None):
        if self.selected():
            return
        value = self.option.value
        self.view.apply_pending_choice(self.choice, value)
        self.view.flyout.run_callback(lambda: self.choice.on_select(value))


class _SliderRow(_Row):
    height = SLIDER_HEIGHT

    def __init__(self, view: _FlyoutView, parent: tk.Misc, data):
        super().__init__(view, parent, data)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)

    def set_data(self, data):
        self.data = data
        self.pending_value = None
        self.dragging = False

    def update(self, data):
        if self.dragging:
            self.data = data
            return
        super().update(data)

    def current(self) -> float:
        return self.data.current() if self.pending_value is None else self.pending_value

    def track_span(self):
        view = self.view
        left = view.px(ROW_INSET + CONTENT_PADDING) + view.px(SLIDER_THUMB_SIZE) // 2
        right = view.width - left
        return left, right, view.px(42)

    def fraction(self, value: float) -> float:
        slider = self.data
        span = slider.maximum - slider.minimum
        return 0.0 if span <= 0 else (value - slider.minimum) / span

    def draw_content(self):
        view, theme, slider = self.view, self.view.theme, self.data
        text_left = view.px(ROW_INSET + CONTENT_PADDING)
        value = self.current()
        value_text = slider.format_value(value)
        value_right = view.width - text_left
        self.canvas.create_text(value_right, view.px(17), anchor="e", text=value_text, font=view.font_small,
                                fill=theme.secondary)
        label_width = value_right - text_left - view.font_small.measure(value_text) - view.px(12)
        self.canvas.create_text(text_left, view.px(17), anchor="w", font=view.font_body, fill=theme.text,
                                text=view.fit_text(view.font_body, slider.label, label_width))
        left, right, middle = self.track_span()
        thumb_x = left + int(round((right - left) * self.fraction(value)))
        colors = tuple(slider.track_colors()) if slider.track_colors else None
        if colors:
            thickness = view.px(SLIDER_GRADIENT_THICKNESS)
            self.canvas.create_image(left, middle, anchor="w",
                                     image=view.gradient_image(right - left, thickness, colors))
        else:
            thickness = view.px(SLIDER_TRACK_THICKNESS)
            self.canvas.create_line(left, middle, right, middle, width=thickness, capstyle=tk.ROUND,
                                    fill=theme.surface_hover if self.highlighted else theme.border)
            self.canvas.create_line(left, middle, thumb_x, middle, width=thickness, capstyle=tk.ROUND,
                                    fill=theme.accent)
        mark_reach = view.px(SLIDER_THUMB_SIZE) // 2
        marks = list(slider.snap_points) + ([slider.default_value] if slider.default_value is not None else [])
        for mark_value in marks:
            mark_x = left + int(round((right - left) * self.fraction(mark_value)))
            self.canvas.create_line(mark_x, middle - mark_reach, mark_x, middle + mark_reach,
                                    width=max(1, view.px(2)), fill=theme.secondary)
        self.canvas.create_image(thumb_x, middle, image=view.slider_thumb_image())

    def draw_hover(self):
        pass

    def value_at(self, x: int) -> float:
        left, right, _ = self.track_span()
        fraction = max(0.0, min(1.0, (x - left) / max(1, right - left)))
        slider = self.data
        return slider.snap(slider.minimum + fraction * (slider.maximum - slider.minimum))

    def _on_press(self, event):
        _, _, middle = self.track_span()
        if abs(event.y - middle) > self.view.px(SLIDER_THUMB_SIZE):
            return
        self.dragging = True
        self._preview(self.value_at(event.x))

    def _on_drag(self, event):
        if self.dragging:
            self._preview(self.value_at(event.x))

    def _on_click(self, event):
        if not self.dragging:
            return
        self.dragging = False
        self._commit(self.value_at(event.x))

    def _preview(self, value: float):
        if value == self.pending_value:
            return
        self.pending_value = value
        self.draw()
        slider = self.data
        self.view.flyout.run_callback(lambda: slider.on_change(value))

    def _commit(self, value: float):
        self.pending_value = value
        self.draw()
        slider = self.data
        self.view.flyout.run_callback(lambda: slider.on_commit(value))

    def handle_arrow(self, key: str) -> bool:
        slider = self.data
        value = slider.snap_to_step(self.current() + (slider.step if key == "Right" else -slider.step))
        if value != self.current():
            self._commit(value)
        return True


class _SegmentedRow(_Row):
    height = SEGMENTED_HEIGHT

    @staticmethod
    def key(data):
        choice, options = data
        return choice.label, tuple((option.value, option.label) for option in options)

    def set_data(self, data):
        self.data = data
        self.choice, self.options = data
        self.pending_value = None

    def current(self):
        return self.choice.current() if self.pending_value is None else self.pending_value

    def track_box(self):
        view = self.view
        left = view.px(ROW_INSET + CONTENT_PADDING)
        top = view.px(30)
        return left, top, view.width - 2 * left, view.px(30)

    def draw_content(self):
        view, theme = self.view, self.view.theme
        left, top, width, height = self.track_box()
        self.canvas.create_text(left, view.px(16), anchor="w", text=self.choice.label, font=view.font_body,
                                fill=theme.text)
        track_fill = theme.surface_hover if self.highlighted else theme.surface
        self.canvas.create_image(left, top, anchor="nw", image=view.rounded_image(width, height, view.px(CORNER_RADIUS), track_fill))
        segment_width = width / max(1, len(self.options))
        current = self.current()
        pad = view.px(2)
        for index, option in enumerate(self.options):
            segment_left = left + int(round(index * segment_width))
            segment_right = left + int(round((index + 1) * segment_width))
            if option.value == current:
                pill = view.rounded_image(segment_right - segment_left - 2 * pad, height - 2 * pad,
                                          view.px(CORNER_RADIUS - 1), theme.accent)
                self.canvas.create_image(segment_left + pad, top + pad, anchor="nw", image=pill)
                color = theme.on_accent
            else:
                color = theme.text if option.enabled else theme.disabled
            label = view.fit_text(view.font_small, option.label, segment_right - segment_left - view.px(8))
            self.canvas.create_text((segment_left + segment_right) // 2, top + height // 2, text=label,
                                    font=view.font_small, fill=color)

    def draw_hover(self):
        pass

    def _on_click(self, event):
        left, top, width, height = self.track_box()
        if not (left <= event.x < left + width and top <= event.y < top + height):
            return
        index = int((event.x - left) / (width / len(self.options)))
        self._select(self.options[min(index, len(self.options) - 1)])

    def handle_arrow(self, key: str) -> bool:
        values = [option.value for option in self.options]
        current = self.current()
        index = values.index(current) if current in values else 0
        step = 1 if key == "Right" else -1
        while 0 <= index + step < len(self.options):
            index += step
            if self.options[index].enabled:
                self._select(self.options[index])
                break
        return True

    def activate(self, event=None):
        self.handle_arrow("Right")

    def _select(self, option):
        if not option.enabled or option.value == self.current():
            return
        self.pending_value = option.value
        self.draw()
        value = option.value
        self.view.flyout.run_callback(lambda: self.choice.on_select(value))
