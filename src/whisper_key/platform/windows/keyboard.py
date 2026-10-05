import ctypes
import ctypes.wintypes as wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
advapi32.OpenProcessToken.argtypes = (wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE))
advapi32.GetTokenInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD))
advapi32.GetSidSubAuthorityCount.argtypes = (ctypes.c_void_p,)
advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
advapi32.GetSidSubAuthority.argtypes = (ctypes.c_void_p, wintypes.DWORD)
advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TOKEN_INTEGRITY_LEVEL = 25

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

VK_RETURN = 0x0D
VK_TAB = 0x09


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]

    _anonymous_ = ("_union",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("_union", _INPUT_UNION),
    ]


VK_MAP = {
    "enter": VK_RETURN, "return": VK_RETURN,
    "tab": VK_TAB,
    "backspace": 0x08, "delete": 0x2E, "escape": 0x1B, "esc": 0x1B,
    "space": 0x20,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "insert": 0x2D,
    "ctrl": 0xA2, "control": 0xA2, "lctrl": 0xA2, "rctrl": 0xA3,
    "shift": 0xA0, "lshift": 0xA0, "rshift": 0xA1,
    "alt": 0xA4, "lalt": 0xA4, "ralt": 0xA5,
    "win": 0x5B, "lwin": 0x5B, "rwin": 0x5C,
    "cmd": 0x5B, "command": 0x5B,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
}

for c in "abcdefghijklmnopqrstuvwxyz":
    VK_MAP[c] = ord(c.upper())
for d in "0123456789":
    VK_MAP[d] = ord(d)
for function_key_number in range(1, 25):
    VK_MAP[f"f{function_key_number}"] = 0x6F + function_key_number
VK_MAP.update({"page_up": 0x21, "page_down": 0x22, "del": 0x2E})

def can_send_key(key: str) -> bool:
    return key.lower() in VK_MAP

def _make_vk_input(vk, flags=0):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ki.wVk = vk
    inp.ki.dwFlags = flags
    return inp


def _make_unicode_input(char_code, flags=0):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ki.wVk = 0
    inp.ki.wScan = char_code
    inp.ki.dwFlags = KEYEVENTF_UNICODE | flags
    return inp


def _send(inputs):
    n = len(inputs)
    array = (INPUT * n)(*inputs)
    user32.SendInput(n, array, ctypes.sizeof(INPUT))


def _process_integrity_level(process_handle):
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(process_handle, TOKEN_QUERY, ctypes.byref(token)):
        return None
    try:
        buffer = ctypes.create_string_buffer(64)
        returned = wintypes.DWORD()
        if not advapi32.GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL, buffer, len(buffer), ctypes.byref(returned)):
            return None
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        count = advapi32.GetSidSubAuthorityCount(sid)[0]
        return advapi32.GetSidSubAuthority(sid, count - 1)[0]
    finally:
        kernel32.CloseHandle(token)


def is_foreground_input_blocked() -> bool:
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return False

    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return False

    own_level = _process_integrity_level(kernel32.GetCurrentProcess())
    if own_level is None:
        return False

    process = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not process:
        return True
    try:
        target_level = _process_integrity_level(process)
    finally:
        kernel32.CloseHandle(process)

    if target_level is None:
        return True
    return target_level > own_level


def validate_delivery_method(method: str) -> str:
    return method


def set_delay(delay: float):
    pass


def send_key(key: str):
    vk = VK_MAP.get(key.lower())
    if vk is None:
        return
    _send([_make_vk_input(vk), _make_vk_input(vk, KEYEVENTF_KEYUP)])


def send_hotkey(*keys: str):
    down = []
    up = []
    for key in keys:
        vk = VK_MAP.get(key.lower())
        if vk is None:
            continue
        down.append(_make_vk_input(vk))
        up.insert(0, _make_vk_input(vk, KEYEVENTF_KEYUP))
    _send(down + up)


def type_text(text: str):
    inputs = []
    for char in text:
        if char == "\n":
            inputs.append(_make_vk_input(VK_RETURN))
            inputs.append(_make_vk_input(VK_RETURN, KEYEVENTF_KEYUP))
        elif char == "\t":
            inputs.append(_make_vk_input(VK_TAB))
            inputs.append(_make_vk_input(VK_TAB, KEYEVENTF_KEYUP))
        else:
            code = ord(char)
            if code > 0xFFFF:
                high = 0xD800 + ((code - 0x10000) >> 10)
                low = 0xDC00 + ((code - 0x10000) & 0x3FF)
                inputs.append(_make_unicode_input(high))
                inputs.append(_make_unicode_input(high, KEYEVENTF_KEYUP))
                inputs.append(_make_unicode_input(low))
                inputs.append(_make_unicode_input(low, KEYEVENTF_KEYUP))
            else:
                inputs.append(_make_unicode_input(code))
                inputs.append(_make_unicode_input(code, KEYEVENTF_KEYUP))

    if inputs:
        _send(inputs)
