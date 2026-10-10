import ctypes
from ctypes import wintypes

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


def is_point_on_screen(x: int, y: int) -> bool:
    left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    return left <= x < left + width and top <= y < top + height
