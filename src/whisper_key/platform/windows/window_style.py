import ctypes
from ctypes import wintypes

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

user32 = ctypes.WinDLL("user32")
user32.GetParent.argtypes = [wintypes.HWND]
user32.GetParent.restype = wintypes.HWND
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = wintypes.LONG
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
user32.SetWindowLongW.restype = wintypes.LONG
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int


def prevent_focus_steal(tk_window_id: int):
    window_handle = user32.GetParent(tk_window_id) or tk_window_id
    extended_style = user32.GetWindowLongW(window_handle, GWL_EXSTYLE)
    user32.SetWindowLongW(window_handle, GWL_EXSTYLE, extended_style | WS_EX_NOACTIVATE)


def is_point_on_screen(x: int, y: int) -> bool:
    left = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    top = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    return left <= x < left + width and top <= y < top + height
