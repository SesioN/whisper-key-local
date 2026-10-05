import ctypes
from ctypes import wintypes

MB_OK = 0x0
MB_OKCANCEL = 0x1
MB_YESNOCANCEL = 0x3
MB_ICONERROR = 0x10
MB_ICONQUESTION = 0x20
MB_SETFOREGROUND = 0x10000
MB_TOPMOST = 0x40000
IDOK = 1
IDYES = 6
IDNO = 7

user32 = ctypes.WinDLL("user32")
user32.MessageBoxW.argtypes = (wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT)
user32.MessageBoxW.restype = ctypes.c_int


def _message_box(title: str, text: str, flags: int) -> int:
    return user32.MessageBoxW(None, text, title, flags | MB_SETFOREGROUND | MB_TOPMOST)


def confirm(title: str, text: str) -> bool:
    return _message_box(title, text, MB_OKCANCEL | MB_ICONQUESTION) == IDOK


def choose(title: str, text: str):
    result = _message_box(title, text, MB_YESNOCANCEL | MB_ICONQUESTION)
    if result == IDYES:
        return True
    if result == IDNO:
        return False
    return None


def show_error(title: str, text: str):
    _message_box(title, text, MB_OK | MB_ICONERROR)
