import ctypes
import logging
import math
import re
import threading
import time
from ctypes import wintypes
from typing import Callable, Optional

from .orb_skins import DEFAULT_SKIN, SKINS, get_skin
from .orb_skins.badges import BadgeRenderer
from .orb_skins.base import DEFAULT_IDLE_OPACITY, appearance_matrix, apply_color_matrix

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
WPARAM = ctypes.c_size_t
LPARAM = ctypes.c_ssize_t

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
ULW_ALPHA = 0x02
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01
BI_RGB = 0
DIB_RGB_COLORS = 0
PM_REMOVE = 0x0001
QS_ALLINPUT = 0x04FF
MWMO_INPUTAVAILABLE = 0x0004
MONITOR_DEFAULTTONULL = 0
MONITOR_DEFAULTTOPRIMARY = 1
MONITOR_DEFAULTTONEAREST = 2
IDC_HAND = 32649
ERROR_CLASS_ALREADY_EXISTS = 1410

WM_NULL = 0x0000
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_MOUSEWHEEL = 0x020A
WM_CAPTURECHANGED = 0x0215
WM_MOUSELEAVE = 0x02A3
TME_LEAVE = 0x00000002
WM_DPICHANGED = 0x02E0

DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)

SHELL_WINDOW_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class SIZE(ctypes.Structure):
    _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class TRACKMOUSEEVENT(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("hwndTrack", wintypes.HWND), ("dwHoverTime", wintypes.DWORD)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]


WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, WPARAM, LPARAM)


class WNDCLASS(ctypes.Structure):
    _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]


def _bind():
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
    user32.DefWindowProcW.restype = LRESULT
    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, WPARAM, LPARAM]
    user32.RegisterClassW.restype = wintypes.ATOM
    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]
    user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.UpdateLayeredWindow.argtypes = [
        wintypes.HWND, wintypes.HDC, ctypes.POINTER(POINT), ctypes.POINTER(SIZE),
        wintypes.HDC, ctypes.POINTER(POINT), wintypes.DWORD,
        ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD]
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.MonitorFromWindow.restype = wintypes.HANDLE
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromPoint.restype = wintypes.HANDLE
    user32.MonitorFromPoint.argtypes = [POINT, wintypes.DWORD]
    user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFO)]
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
    user32.SetCapture.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.LoadCursorW.restype = wintypes.HANDLE
    user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, ctypes.c_void_p]
    user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                    wintypes.UINT, wintypes.UINT, wintypes.UINT]
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.MsgWaitForMultipleObjectsEx.restype = wintypes.DWORD
    user32.MsgWaitForMultipleObjectsEx.argtypes = [wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                                   wintypes.DWORD, wintypes.DWORD]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, WPARAM, LPARAM]
    user32.TrackMouseEvent.argtypes = [ctypes.POINTER(TRACKMOUSEEVENT)]
    user32.GetDpiForWindow.restype = wintypes.UINT
    user32.GetDpiForWindow.argtypes = [wintypes.HWND]
    user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.CreateDIBSection.argtypes = [
        wintypes.HDC, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
        ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]


_bind()

SIZES = {"small": 44, "medium": 60, "big": 80}
SIZE_ORDER = ("small", "medium", "big")
DEFAULT_SIZE = "big"
POSITION_PATTERN = re.compile(r"^\+(-?\d+)\+(-?\d+)$")

CLICK_SLOP = 4
SNAP_DISTANCE = 16
FLASH_DURATION = 0.45
OUTCOME_SECONDS = {"success": 1.2, "error": 2.0}
ERROR_FLASH_COLOR = (255, 60, 50)
OPACITY_MUTED_SKIN = 0.85
OPACITY_MUTED = 0.3
OPACITY_LOADING = 0.35
FPS_IDLE = 8
FPS_ACTIVE = 30
HIDDEN_POLL_SECONDS = 0.25
CPU_BUDGET = 0.15
MAX_BUDGET_WAIT_SECONDS = 0.25
TOPMOST_REFRESH_SECONDS = 2.0
FULLSCREEN_CHECK_SECONDS = 1.0
STOP_TIMEOUT_SECONDS = 3.0
BADGE_FADE_IN_SECONDS = 0.15
BADGE_FADE_OUT_SECONDS = 0.4
BADGE_LINGER_SECONDS = 0.3


class VoiceOrb:

    def __init__(self,
                 on_click: Callable[[], None],
                 on_position_changed: Callable[[str], None],
                 on_size_changed: Callable[[str], None],
                 size: str = DEFAULT_SIZE,
                 skin: str = DEFAULT_SKIN,
                 save_position: bool = False,
                 position: Optional[str] = None,
                 locked: bool = False,
                 hide_on_fullscreen: bool = True,
                 on_lock_click: Optional[Callable[[bool], None]] = None,
                 on_mute_click: Optional[Callable[[], None]] = None,
                 show_lock_button: bool = True,
                 show_mute_button: bool = True,
                 opacity: float = DEFAULT_IDLE_OPACITY,
                 vibrancy: float = 1.0,
                 hue: float = 0.0):
        self.on_click = on_click
        self.on_lock_click = on_lock_click
        self.on_mute_click = on_mute_click
        self.show_lock_button = show_lock_button
        self.show_mute_button = show_mute_button
        self.class_name = f"WhisperKeyVoiceOrb{id(self)}"
        self.on_position_changed = on_position_changed
        self.on_size_changed = on_size_changed
        self.logger = logging.getLogger(__name__)

        self.size = size if size in SIZES else DEFAULT_SIZE
        if skin not in SKINS:
            self.logger.warning(f"Unknown orb skin '{skin}', using {DEFAULT_SKIN}")
            skin = DEFAULT_SKIN
        self.skin = skin
        self.save_position = save_position
        self.position = position
        self.locked = locked
        self.hide_on_fullscreen = hide_on_fullscreen
        self.appearance_opacity = opacity
        self.color_matrix = appearance_matrix(hue, vibrancy)

        self.state = "idle"
        self.muted = False
        self.visible = False
        self._level = 0.0
        self._level_smooth = 0.0
        self._flash_until = 0.0
        self._flash_color = None
        self._outcome = None
        self._outcome_until = 0.0

        self._thread_lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._rebuild_requested = False
        self._click_thread = None

        self._hwnd = None
        self._hinstance = None
        self._class_registered = False
        self._wndproc_ref = None
        self._renderer = None
        self._surface = None
        self._dpi = 96
        self._pos = (0, 0)
        self._shown = False
        self._hidden_by_fullscreen = False
        self._dragging = False
        self._hovered = False
        self._hover_left_at = 0.0
        self._badge_visibility = 0.0
        self._last_badge_update = None
        self._redraw_now = False
        self._pressed_badge = None
        self._badges = None
        self._drag_origin = (0, 0)
        self._window_origin = (0, 0)
        self._drag_distance = 0
        self._drag_target = None
        self._last_topmost = 0.0
        self._last_fullscreen_check = 0.0


    def show(self):
        self.visible = True
        with self._thread_lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="VoiceOrb")
            self._thread.start()

    def hide(self):
        self.visible = False

    def update_state(self, new_state: str):
        if new_state == "idle" and self.state in ("processing", "recording") and not self._skin_handles_outcomes():
            self._flash_color = None
            self._flash_until = time.monotonic() + FLASH_DURATION
        self.state = new_state
        if new_state != "recording":
            self._level = 0.0

    def show_outcome(self, outcome: str):
        if outcome not in OUTCOME_SECONDS:
            return
        if self._skin_handles_outcomes():
            self._outcome = outcome
            self._outcome_until = time.monotonic() + OUTCOME_SECONDS[outcome]
        elif outcome == "error":
            self._flash_color = ERROR_FLASH_COLOR
            self._flash_until = time.monotonic() + FLASH_DURATION * 2

    def _skin_handles_outcomes(self) -> bool:
        return get_skin(self.skin).HANDLES_OUTCOMES

    def _active_outcome(self):
        if self._outcome and time.monotonic() < self._outcome_until:
            return self._outcome
        return None

    def set_level(self, level: float):
        self._level = level

    def set_muted(self, muted: bool):
        self.muted = muted

    def set_size(self, size: str):
        if size in SIZES and size != self.size:
            self.size = size
            self._rebuild_requested = True

    def set_skin(self, skin: str):
        if skin in SKINS and skin != self.skin:
            self.skin = skin
            self._rebuild_requested = True

    def set_appearance(self, opacity: float, vibrancy: float, hue: float):
        self.appearance_opacity = opacity
        self.color_matrix = appearance_matrix(hue, vibrancy)
        self._redraw_now = True

    def set_save_position(self, save_position: bool):
        self.save_position = save_position
        if save_position and self._hwnd:
            self._report_position()

    def set_locked(self, locked: bool):
        self.locked = locked

    def set_hide_on_fullscreen(self, hide_on_fullscreen: bool):
        self.hide_on_fullscreen = hide_on_fullscreen

    def set_buttons(self, show_lock_button: bool, show_mute_button: bool):
        self.show_lock_button = show_lock_button
        self.show_mute_button = show_mute_button

    def stop(self):
        with self._thread_lock:
            thread, self._thread = self._thread, None
        self._stop_event.set()
        if not thread:
            return
        hwnd = self._hwnd
        if hwnd:
            user32.PostMessageW(hwnd, WM_NULL, 0, 0)
        thread.join(timeout=STOP_TIMEOUT_SECONDS)
        if thread.is_alive():
            self.logger.warning("Voice orb did not close within timeout")


    def _run(self):
        try:
            self._set_thread_dpi_awareness()
            self._create_window()
            self._loop()
        except Exception:
            self.logger.exception("Voice orb failed")
        finally:
            self._destroy()

    def _set_thread_dpi_awareness(self):
        try:
            user32.SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
        except AttributeError:
            pass

    def _register_class(self):
        self._hinstance = kernel32.GetModuleHandleW(None)
        self._wndproc_ref = WNDPROC(self._wndproc)
        wc = WNDCLASS()
        wc.lpfnWndProc = self._wndproc_ref
        wc.hInstance = self._hinstance
        wc.hCursor = user32.LoadCursorW(None, ctypes.c_void_p(IDC_HAND))
        wc.lpszClassName = self.class_name
        if not user32.RegisterClassW(ctypes.byref(wc)):
            error = ctypes.get_last_error()
            if error != ERROR_CLASS_ALREADY_EXISTS:
                raise ctypes.WinError(error)
            user32.UnregisterClassW(self.class_name, self._hinstance)
            if not user32.RegisterClassW(ctypes.byref(wc)):
                raise ctypes.WinError(ctypes.get_last_error())
        self._class_registered = True

    def _create_window(self):
        self._register_class()
        ex_style = WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        self._hwnd = user32.CreateWindowExW(ex_style, self.class_name, "Whisper Key", WS_POPUP,
                                            0, 0, 10, 10, None, None, self._hinstance, None)
        if not self._hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self._dpi = self._window_dpi()
        self._build_surface()
        self._move_to(*self._initial_position())

    def _window_dpi(self) -> int:
        try:
            return int(user32.GetDpiForWindow(self._hwnd)) or 96
        except AttributeError:
            return 96

    def _orb_radius(self) -> float:
        return SIZES[self.size] * self._dpi / 96.0 / 2.0 * get_skin(self.skin).VISUAL_SCALE

    def _canvas_size(self) -> int:
        return max(48, int(round(2.0 * self._orb_radius() * get_skin(self.skin).CANVAS_FACTOR)))

    def _build_surface(self):
        import numpy as np

        self._release_surface()
        canvas = self._canvas_size()
        hdc_screen = user32.GetDC(None)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = canvas
        bmi.bmiHeader.biHeight = -canvas
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        bits = ctypes.c_void_p()
        hbmp = gdi32.CreateDIBSection(hdc_mem, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(bits), None, 0)
        if not hbmp:
            error = ctypes.get_last_error()
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(None, hdc_screen)
            raise ctypes.WinError(error)

        old = gdi32.SelectObject(hdc_mem, hbmp)
        pixels = np.ctypeslib.as_array(ctypes.cast(bits, ctypes.POINTER(ctypes.c_ubyte)), shape=(canvas, canvas, 4))
        self._surface = {"canvas": canvas, "hdc_screen": hdc_screen, "hdc_mem": hdc_mem,
                         "hbmp": hbmp, "old": old, "pixels": pixels}
        skin = get_skin(self.skin)
        self._renderer = skin(canvas, self._orb_radius())
        self._badges = BadgeRenderer(canvas, self._orb_radius() / skin.VISUAL_SCALE, self._dpi)

    def _release_surface(self):
        surface, self._surface = self._surface, None
        if not surface:
            return
        gdi32.SelectObject(surface["hdc_mem"], surface["old"])
        gdi32.DeleteObject(surface["hbmp"])
        gdi32.DeleteDC(surface["hdc_mem"])
        user32.ReleaseDC(None, surface["hdc_screen"])

    def _destroy(self):
        if self._hwnd:
            user32.DestroyWindow(self._hwnd)
            self._hwnd = None
        self._release_surface()
        self._renderer = None
        if self._class_registered:
            user32.UnregisterClassW(self.class_name, self._hinstance)
            self._class_registered = False
        self._shown = False


    def _monitor_rects(self, hmon):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        user32.GetMonitorInfoW(hmon, ctypes.byref(info))
        return info.rcMonitor, info.rcWork

    def _initial_position(self):
        canvas = self._surface["canvas"]
        match = POSITION_PATTERN.match(self.position) if isinstance(self.position, str) else None
        if self.save_position and match:
            x, y = int(match.group(1)), int(match.group(2))
            if user32.MonitorFromPoint(POINT(x + canvas // 2, y + canvas // 2), MONITOR_DEFAULTTONULL):
                return x, y
        _, work = self._monitor_rects(user32.MonitorFromPoint(POINT(0, 0), MONITOR_DEFAULTTOPRIMARY))
        return (work.left + (work.right - work.left - canvas) // 2, work.bottom - canvas - 12)

    def _move_to(self, x, y):
        self._pos = (x, y)
        canvas = self._surface["canvas"]
        user32.SetWindowPos(self._hwnd, None, x, y, canvas, canvas, SWP_NOZORDER | SWP_NOACTIVATE)

    def _apply_drag_target(self):
        target, self._drag_target = self._drag_target, None
        if target is None or target == self._pos:
            return
        self._pos = target
        user32.UpdateLayeredWindow(self._hwnd, None, ctypes.byref(POINT(*target)), None,
                                   None, None, 0, None, 0)

    def _snap(self, x, y):
        canvas = self._surface["canvas"]
        hmon = user32.MonitorFromPoint(POINT(x + canvas // 2, y + canvas // 2), MONITOR_DEFAULTTONEAREST)
        for rect in self._monitor_rects(hmon)[::-1]:
            if abs(x - rect.left) <= SNAP_DISTANCE:
                x = rect.left
            elif abs(x + canvas - rect.right) <= SNAP_DISTANCE:
                x = rect.right - canvas
            if abs(y - rect.top) <= SNAP_DISTANCE:
                y = rect.top
            elif abs(y + canvas - rect.bottom) <= SNAP_DISTANCE:
                y = rect.bottom - canvas
        return x, y

    def _report_position(self):
        self.position = f"+{self._pos[0]}+{self._pos[1]}"
        try:
            self.on_position_changed(self.position)
        except Exception:
            self.logger.exception("Saving the orb position failed")

    def _rebuild_centered(self):
        old_canvas = self._surface["canvas"]
        center = (self._pos[0] + old_canvas // 2, self._pos[1] + old_canvas // 2)
        self._build_surface()
        canvas = self._surface["canvas"]
        self._move_to(center[0] - canvas // 2, center[1] - canvas // 2)


    def _wndproc(self, hwnd, msg, wparam, lparam):
        try:
            if self._handle_message(msg, wparam, lparam):
                return 0
        except Exception:
            self.logger.exception("Voice orb message handling failed")
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _enabled_badges(self):
        kinds = []
        if self.show_lock_button and self.on_lock_click:
            kinds.append("lock")
        if self.show_mute_button and self.on_mute_click:
            kinds.append("mute")
        return kinds

    def _client_point(self, lparam):
        return ctypes.c_short(lparam & 0xFFFF).value, ctypes.c_short((lparam >> 16) & 0xFFFF).value

    def _badge_at(self, lparam):
        if not self._badges:
            return None
        return self._badges.hit(*self._client_point(lparam), self._enabled_badges())

    def _note_hover(self):
        if self._hovered:
            return
        self._hovered = True
        self._redraw_now = True
        track = TRACKMOUSEEVENT(ctypes.sizeof(TRACKMOUSEEVENT), TME_LEAVE, self._hwnd, 0)
        user32.TrackMouseEvent(ctypes.byref(track))

    def _press_badge(self, badge):
        if badge == "lock":
            self.locked = not self.locked
            locked = self.locked
            self._start_click_thread(lambda: self.on_lock_click(locked))
        elif badge == "mute":
            self._start_click_thread(self.on_mute_click)
        self._redraw_now = True

    def _handle_message(self, msg, wparam, lparam) -> bool:
        if msg == WM_LBUTTONDOWN and self._badge_at(lparam):
            self._pressed_badge = self._badge_at(lparam)
            user32.SetCapture(self._hwnd)
            return True

        if msg == WM_LBUTTONUP and self._pressed_badge:
            pressed, self._pressed_badge = self._pressed_badge, None
            user32.ReleaseCapture()
            if self._badge_at(lparam) == pressed:
                self._press_badge(pressed)
            return True

        if msg == WM_MOUSEMOVE:
            self._note_hover()

        if msg == WM_MOUSELEAVE:
            self._hovered = False
            self._hover_left_at = time.monotonic()
            return True

        if msg == WM_LBUTTONDOWN:
            pt = POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            self._dragging = True
            self._drag_origin = (pt.x, pt.y)
            self._window_origin = self._pos
            self._drag_distance = 0
            user32.SetCapture(self._hwnd)
            return True

        if msg == WM_MOUSEMOVE and not self._dragging:
            return True

        if msg == WM_MOUSEMOVE and self._dragging:
            pt = POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            dx, dy = pt.x - self._drag_origin[0], pt.y - self._drag_origin[1]
            self._drag_distance = max(self._drag_distance, abs(dx) + abs(dy))
            if not self.locked and self._drag_distance > CLICK_SLOP:
                self._drag_target = (self._window_origin[0] + dx, self._window_origin[1] + dy)
            return True

        if msg == WM_LBUTTONUP and self._dragging:
            self._dragging = False
            self._apply_drag_target()
            user32.ReleaseCapture()
            if self._drag_distance <= CLICK_SLOP:
                self._start_click_thread(self.on_click)
            elif not self.locked:
                self._move_to(*self._snap(*self._pos))
                if self.save_position:
                    self._report_position()
            return True

        if msg == WM_CAPTURECHANGED:
            self._dragging = False
            self._pressed_badge = None
            self._drag_target = None
            return False

        if msg == WM_MOUSEWHEEL:
            delta = ctypes.c_short((wparam >> 16) & 0xFFFF).value
            self._step_size(1 if delta > 0 else -1)
            return True

        if msg == WM_DPICHANGED:
            self._dpi = (wparam & 0xFFFF) or 96
            self._rebuild_centered()
            return True

        if msg == WM_CLOSE:
            return True

        return False

    def _start_click_thread(self, callback: Callable[[], None]):
        if self._click_thread and self._click_thread.is_alive():
            return
        self._click_thread = threading.Thread(target=self._run_click_callback, args=(callback,),
                                              daemon=True, name="VoiceOrbClick")
        self._click_thread.start()

    def _run_click_callback(self, callback: Callable[[], None]):
        try:
            callback()
        except Exception:
            self.logger.exception("Voice orb click handler failed")

    def _step_size(self, direction: int):
        index = SIZE_ORDER.index(self.size) + direction
        if 0 <= index < len(SIZE_ORDER):
            self.size = SIZE_ORDER[index]
            self._rebuild_centered()
            try:
                self.on_size_changed(self.size)
            except Exception:
                self.logger.exception("Saving the orb size failed")


    def _loop(self):
        start = time.monotonic()
        next_frame_at = start
        while not self._stop_event.is_set():
            self._pump_messages()

            now = time.monotonic()
            if self._rebuild_requested:
                self._rebuild_requested = False
                self._rebuild_centered()
            self._maintain_window(now)

            if not self._shown:
                self._wait_for_messages(HIDDEN_POLL_SECONDS)
                continue

            if now >= next_frame_at or self._redraw_now:
                self._redraw_now = False
                self._draw(now - start)
                elapsed = time.monotonic() - now
                budget_wait = min(elapsed * (1.0 / CPU_BUDGET - 1.0), MAX_BUDGET_WAIT_SECONDS)
                if self._dragging:
                    budget_wait = 0.0
                next_frame_at = now + max(1.0 / self._target_fps() - elapsed, budget_wait, 0.005) + elapsed
            self._wait_for_messages(next_frame_at - time.monotonic())

    def _badge_target(self, now) -> float:
        if self._hovered or self._pressed_badge:
            return 1.0
        return 1.0 if now - self._hover_left_at < BADGE_LINGER_SECONDS else 0.0

    def _badges_animating(self) -> bool:
        if not self._enabled_badges():
            return False
        return self._badge_visibility != self._badge_target(time.monotonic())

    def _update_badge_visibility(self):
        now = time.monotonic()
        elapsed = 0.0 if self._last_badge_update is None else now - self._last_badge_update
        self._last_badge_update = now
        target = self._badge_target(now)
        if target > self._badge_visibility:
            self._badge_visibility = min(target, self._badge_visibility + elapsed / BADGE_FADE_IN_SECONDS)
        else:
            self._badge_visibility = max(target, self._badge_visibility - elapsed / BADGE_FADE_OUT_SECONDS)
        return self._badge_visibility

    def _target_fps(self) -> int:
        if self.state == "recording" or self._active_outcome() or self._badges_animating():
            return FPS_ACTIVE
        if self.state == "processing":
            return FPS_IDLE * 2
        return get_skin(self.skin).IDLE_FPS or FPS_IDLE

    def _pump_messages(self):
        msg = wintypes.MSG()
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        self._apply_drag_target()

    def _wait_for_messages(self, seconds: float):
        if seconds <= 0 or self._stop_event.is_set():
            return
        user32.MsgWaitForMultipleObjectsEx(0, None, max(1, int(seconds * 1000)),
                                           QS_ALLINPUT, MWMO_INPUTAVAILABLE)

    def _maintain_window(self, now):
        if self.hide_on_fullscreen and now - self._last_fullscreen_check > FULLSCREEN_CHECK_SECONDS:
            self._last_fullscreen_check = now
            self._hidden_by_fullscreen = self._foreground_is_fullscreen()
        elif not self.hide_on_fullscreen:
            self._hidden_by_fullscreen = False

        should_show = self.visible and not self._hidden_by_fullscreen
        if should_show != self._shown:
            self._shown = should_show
            user32.ShowWindow(self._hwnd, SW_SHOWNOACTIVATE if should_show else SW_HIDE)

        if self._shown and now - self._last_topmost > TOPMOST_REFRESH_SECONDS:
            self._last_topmost = now
            user32.SetWindowPos(self._hwnd, wintypes.HWND(HWND_TOPMOST), 0, 0, 0, 0,
                                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)

    def _foreground_is_fullscreen(self) -> bool:
        foreground = user32.GetForegroundWindow()
        if not foreground or foreground == self._hwnd:
            return False
        class_name = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(foreground, class_name, len(class_name))
        if class_name.value in SHELL_WINDOW_CLASSES:
            return False
        rect = RECT()
        if not user32.GetWindowRect(foreground, ctypes.byref(rect)):
            return False
        monitor, _ = self._monitor_rects(user32.MonitorFromWindow(foreground, MONITOR_DEFAULTTONEAREST))
        return (rect.left <= monitor.left and rect.top <= monitor.top
                and rect.right >= monitor.right and rect.bottom >= monitor.bottom)

    def _draw(self, t):
        target = min(1.0, (max(self._level, 0.0) * 9.0) ** 0.6)
        rate = 0.55 if target > self._level_smooth else 0.12
        self._level_smooth += (target - self._level_smooth) * rate

        skin = type(self._renderer)
        outcome = self._active_outcome()
        grayscale = self.state == "loading"
        if grayscale:
            state, level, opacity = "idle", 0.0, OPACITY_LOADING
        elif outcome and self.state != "recording":
            state, level, opacity = outcome, 0.0, 1.0
        elif self.muted or self.state == "muted":
            if skin.HANDLES_MUTED:
                state, level, opacity = "muted", 0.0, OPACITY_MUTED_SKIN
            else:
                state, level, opacity = "idle", 0.0, OPACITY_MUTED
        else:
            state, level = self.state, self._level_smooth
            opacity = 1.0 if state in ("recording", "processing") else DEFAULT_IDLE_OPACITY
        flash_length = FLASH_DURATION * 2 if self._flash_color else FLASH_DURATION
        flash = max(0.0, (self._flash_until - time.monotonic()) / flash_length)

        opacity = min(1.0, opacity * self.appearance_opacity / DEFAULT_IDLE_OPACITY)
        color_matrix = self.color_matrix
        surface = self._surface
        self._renderer.render_into(surface["pixels"], state, level, t, opacity, flash, self._flash_color, grayscale,
                                   color_matrix)
        visibility = self._update_badge_visibility()
        badges = self._enabled_badges()
        if badges:
            colors = tuple(apply_color_matrix(color, color_matrix) for color in self._renderer.badge_colors(state))
            active = {"lock": self.locked, "mute": self.muted}
            self._badges.composite(surface["pixels"], visibility * opacity,
                                   [(kind, active[kind], colors) for kind in badges])
        gdi32.GdiFlush()
        self._pump_messages()
        if self._surface is not surface:
            return

        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        size = SIZE(surface["canvas"], surface["canvas"])
        user32.UpdateLayeredWindow(self._hwnd, surface["hdc_screen"], ctypes.byref(POINT(*self._pos)),
                                   ctypes.byref(size), surface["hdc_mem"], ctypes.byref(POINT(0, 0)),
                                   0, ctypes.byref(blend), ULW_ALPHA)
