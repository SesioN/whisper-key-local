import ctypes
import logging
import os
import time
from ctypes import wintypes
from dataclasses import dataclass

from .keyboard import send_empty_input

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_BORDER_COLOR = 34
DWMWCP_ROUND = 2
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
DWMWA_EXTENDED_FRAME_BOUNDS = 9
MONITOR_DEFAULTTONEAREST = 2
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)


SW_RESTORE = 9
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
FOCUS_POLL_SECONDS = 0.03
logger = logging.getLogger(__name__)


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD), ("hwndActive", wintypes.HWND),
                ("hwndFocus", wintypes.HWND), ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
                ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND), ("rcCaret", wintypes.RECT)]


@dataclass(frozen=True)
class FocusTarget:
    window: int
    focused_control: int
    thread_id: int
    window_class: str = ""
    process_name: str = ""


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


user32 = ctypes.WinDLL("user32")
user32.GetParent.argtypes = [wintypes.HWND]
user32.GetParent.restype = wintypes.HWND
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = wintypes.LONG
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
user32.SetWindowLongW.restype = wintypes.LONG
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsWindow.restype = wintypes.BOOL
user32.IsIconic.argtypes = [wintypes.HWND]
user32.IsIconic.restype = wintypes.BOOL
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.c_void_p]
user32.GetGUIThreadInfo.restype = wintypes.BOOL
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
user32.AttachThreadInput.restype = wintypes.BOOL
user32.SetFocus.argtypes = [wintypes.HWND]
user32.SetFocus.restype = wintypes.HWND
user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.BringWindowToTop.restype = wintypes.BOOL
user32.IsChild.argtypes = [wintypes.HWND, wintypes.HWND]
user32.IsChild.restype = wintypes.BOOL
kernel32 = ctypes.WinDLL("kernel32")
kernel32.GetCurrentThreadId.restype = wintypes.DWORD
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
user32.GetCursorPos.restype = wintypes.BOOL
user32.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
user32.MonitorFromPoint.restype = wintypes.HMONITOR
user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFO)]
user32.GetMonitorInfoW.restype = wintypes.BOOL
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, wintypes.UINT]
user32.SetWindowPos.restype = wintypes.BOOL
user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p

dwmapi = ctypes.WinDLL("dwmapi")
dwmapi.DwmSetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long
dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long


def _set_dwm_attribute(window_handle: int, attribute: int, value: int):
    data = ctypes.c_int(value)
    return dwmapi.DwmSetWindowAttribute(window_handle, attribute, ctypes.byref(data), ctypes.sizeof(data))


def apply_window_chrome(tk_window_id: int, dark: bool, border_rgb: tuple):
    window_handle = user32.GetParent(tk_window_id) or tk_window_id
    _set_dwm_attribute(window_handle, DWMWA_USE_IMMERSIVE_DARK_MODE, int(dark))
    _set_dwm_attribute(window_handle, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)
    red, green, blue = border_rgb
    _set_dwm_attribute(window_handle, DWMWA_BORDER_COLOR, red | (green << 8) | (blue << 16))


def prevent_focus_steal(tk_window_id: int):
    window_handle = user32.GetParent(tk_window_id) or tk_window_id
    extended_style = user32.GetWindowLongW(window_handle, GWL_EXSTYLE)
    user32.SetWindowLongW(window_handle, GWL_EXSTYLE, extended_style | WS_EX_NOACTIVATE)


def _visible_frame(window_handle) -> wintypes.RECT:
    frame = wintypes.RECT()
    result = dwmapi.DwmGetWindowAttribute(window_handle, DWMWA_EXTENDED_FRAME_BOUNDS,
                                          ctypes.byref(frame), ctypes.sizeof(frame))
    if result != 0:
        user32.GetWindowRect(window_handle, ctypes.byref(frame))
    return frame


def center_window(tk_window_id: int):
    window_handle = user32.GetParent(tk_window_id) or tk_window_id
    previous_context = user32.SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    try:
        window_rect = wintypes.RECT()
        if not user32.GetWindowRect(window_handle, ctypes.byref(window_rect)):
            return
        frame = _visible_frame(window_handle)
        cursor = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(cursor))
        monitor_info = MONITORINFO()
        monitor_info.cbSize = ctypes.sizeof(MONITORINFO)
        if not user32.GetMonitorInfoW(user32.MonitorFromPoint(cursor, MONITOR_DEFAULTTONEAREST),
                                      ctypes.byref(monitor_info)):
            return
        work = monitor_info.rcWork
        frame_width = frame.right - frame.left
        frame_height = frame.bottom - frame.top
        target_left = work.left + max(0, (work.right - work.left - frame_width) // 2)
        target_top = work.top + max(0, (work.bottom - work.top - frame_height) // 2)
        user32.SetWindowPos(window_handle, None,
                            window_rect.left + target_left - frame.left,
                            window_rect.top + target_top - frame.top,
                            0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)
    finally:
        if previous_context:
            user32.SetThreadDpiAwarenessContext(previous_context)


def get_foreground_window() -> int:
    return user32.GetForegroundWindow() or 0


def give_back_foreground(previous_window: int, tk_window_ids: list[int]):
    own_windows = {user32.GetParent(tk_window_id) or tk_window_id for tk_window_id in tk_window_ids}
    if previous_window in own_windows or not user32.IsWindow(previous_window):
        return
    if get_foreground_window() in own_windows:
        user32.SetForegroundWindow(previous_window)


def _thread_of(window: int) -> int:
    return user32.GetWindowThreadProcessId(window, None) if window else 0


def _class_name(window: int) -> str:
    if not window:
        return ""
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(window, buffer, len(buffer))
    return buffer.value


def _process_name(process_id: int) -> str:
    process_handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
    if not process_handle:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buffer))
        if not kernel32.QueryFullProcessImageNameW(process_handle, 0, buffer, ctypes.byref(size)):
            return ""
        return os.path.basename(buffer.value)
    finally:
        kernel32.CloseHandle(process_handle)


def _thread_focused_control(thread_id: int) -> int:
    info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
    if thread_id and user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
        return info.hwndFocus or 0
    return 0


def capture_focus_target():
    window = get_foreground_window()
    if not window:
        return None
    process_id = wintypes.DWORD()
    thread_id = user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
    if process_id.value == os.getpid():
        return None
    return FocusTarget(window, _thread_focused_control(thread_id), thread_id,
                       _class_name(window), _process_name(process_id.value))


def describe_focus_target(target) -> str:
    if not target:
        return "none"
    return f"{target.process_name or '?'} [{target.window_class}] window={target.window:#x} control={target.focused_control:#x}"


def describe_foreground() -> str:
    window = get_foreground_window()
    process_id = wintypes.DWORD()
    thread_id = user32.GetWindowThreadProcessId(window, ctypes.byref(process_id)) if window else 0
    return (f"{_process_name(process_id.value) or '?'} [{_class_name(window)}] window={window:#x} "
            f"control={_thread_focused_control(thread_id):#x}")


def is_focus_target_active(target) -> bool:
    return bool(target and user32.IsWindow(target.window) and get_foreground_window() == target.window)


def _focused_control_restored(target) -> bool:
    if not target.focused_control or not user32.IsWindow(target.focused_control) \
            or not user32.IsChild(target.window, target.focused_control):
        return True
    return _thread_focused_control(target.thread_id) == target.focused_control


def _restore_focused_control(target):
    if _focused_control_restored(target):
        return
    user32.SetFocus(target.focused_control)


def _attempt_activation(target, unlock_foreground: bool):
    own_thread = kernel32.GetCurrentThreadId()
    foreground_thread = _thread_of(get_foreground_window())
    if unlock_foreground:
        send_empty_input()
    attached = []
    for thread_id in {foreground_thread, target.thread_id} - {0, own_thread}:
        if user32.AttachThreadInput(own_thread, thread_id, True):
            attached.append(thread_id)
    try:
        user32.BringWindowToTop(target.window)
        foreground_set = user32.SetForegroundWindow(target.window)
        _restore_focused_control(target)
    finally:
        for thread_id in attached:
            user32.AttachThreadInput(own_thread, thread_id, False)
    return bool(foreground_set)


def restore_focus_target(target, settle_seconds: float = 0.2, timeout_seconds: float = 0.5) -> bool:
    if not target or not user32.IsWindow(target.window):
        logger.info("Focus target is gone")
        return False
    if is_focus_target_active(target) and _focused_control_restored(target):
        return True
    logger.info(f"Refocusing {describe_focus_target(target)}; foreground is {describe_foreground()}")
    if user32.IsIconic(target.window):
        user32.ShowWindow(target.window, SW_RESTORE)
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        foreground_set = _attempt_activation(target, unlock_foreground=attempt > 1)
        time.sleep(FOCUS_POLL_SECONDS)
        if is_focus_target_active(target) and _focused_control_restored(target):
            break
        logger.info(f"Refocus attempt {attempt} not confirmed (SetForegroundWindow={foreground_set}); "
                    f"foreground is {describe_foreground()}")
        if time.monotonic() >= deadline:
            return False
    time.sleep(settle_seconds)
    active = is_focus_target_active(target) and _focused_control_restored(target)
    outcome = "confirmed" if active else f"lost again during settle, foreground is {describe_foreground()}"
    logger.info(f"Refocus after {attempt} attempt(s) {outcome}")
    return active


def is_point_on_screen(x: int, y: int) -> bool:
    left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    return left <= x < left + width and top <= y < top + height
