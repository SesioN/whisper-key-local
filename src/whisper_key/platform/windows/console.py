import ctypes
import ctypes.wintypes as wintypes
import os
import sys

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

STD_OUTPUT_HANDLE = -11
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
ENABLE_PROCESSED_OUTPUT = 0x0001

LF_FACESIZE = 32
FF_MODERN = 0x30
FIXED_PITCH = 0x01
TMPF_TRUETYPE = 0x04

_hwnd = None


class COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class CONSOLE_FONT_INFOEX(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.ULONG),
        ("nFont", wintypes.DWORD),
        ("dwFontSize", COORD),
        ("FontFamily", ctypes.c_uint),
        ("FontWeight", ctypes.c_uint),
        ("FaceName", ctypes.c_wchar * LF_FACESIZE),
    ]


def _get_hwnd():
    global _hwnd
    if _hwnd is None:
        _hwnd = kernel32.GetConsoleWindow()
    return _hwnd


WM_SETICON = 0x0080
ICON_SMALL = 0
ICON_BIG = 1


def _set_icon():
    hwnd = _get_hwnd()
    exe_path = os.environ.get("PYAPP", "")
    if not hwnd or not exe_path:
        return
    shell32 = ctypes.windll.shell32
    icon_handle = shell32.ExtractIconW(0, exe_path, 0)
    if icon_handle and icon_handle != 1:
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, icon_handle)
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, icon_handle)


def _configure_console():
    kernel32.SetConsoleTitleW("Whisper Key")
    _set_icon()

    handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

    mode = wintypes.DWORD()
    kernel32.GetConsoleMode(handle, ctypes.byref(mode))
    mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT
    kernel32.SetConsoleMode(handle, mode)

    font = CONSOLE_FONT_INFOEX()
    font.cbSize = ctypes.sizeof(CONSOLE_FONT_INFOEX)
    font.dwFontSize.X = 0
    font.dwFontSize.Y = 22
    font.FontFamily = FF_MODERN | FIXED_PITCH | TMPF_TRUETYPE
    font.FontWeight = 400
    font.FaceName = "Cascadia Mono"
    kernel32.SetCurrentConsoleFontEx(handle, False, ctypes.byref(font))


def _redirect_missing_streams_to_devnull():
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r")
    if sys.stdout is None or sys.stderr is None:
        devnull = open(os.devnull, "w", encoding="utf-8", errors="replace")
        if sys.stdout is None:
            sys.stdout = devnull
        if sys.stderr is None:
            sys.stderr = devnull


def setup():
    _redirect_missing_streams_to_devnull()
    if os.environ.get("PYAPP") and not os.environ.get("WHISPER_KEY_NO_TERMINAL") and _get_hwnd():
        _configure_console()
