import os
import sys
import winreg
from pathlib import Path

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "WhisperKey"
CONSOLE_SUFFIX = "-console"


def is_supported() -> bool:
    return get_launch_command() is not None


def _no_terminal_exe(pyapp_exe: Path) -> Path:
    if not pyapp_exe.stem.endswith(CONSOLE_SUFFIX):
        return pyapp_exe
    candidate = pyapp_exe.with_name(f"{pyapp_exe.stem.removesuffix(CONSOLE_SUFFIX)}{pyapp_exe.suffix}")
    return candidate if candidate.is_file() else pyapp_exe


def get_launch_command():
    pyapp_exe = os.environ.get("PYAPP", "")
    if os.path.isfile(pyapp_exe):
        return f'"{_no_terminal_exe(Path(pyapp_exe))}"'

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if pythonw.is_file():
        return f'"{pythonw}" -m whisper_key'
    return None


def _read_command():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, VALUE_NAME)
            return value
    except FileNotFoundError:
        return None


def is_enabled() -> bool:
    return _read_command() is not None


def enable():
    command = get_launch_command()
    if command is None:
        raise RuntimeError("No launcher found for autostart")
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command)


def disable():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, VALUE_NAME)
    except FileNotFoundError:
        pass


def refresh():
    if not os.path.isfile(os.environ.get("PYAPP", "")):
        return False
    current = _read_command()
    command = get_launch_command()
    if current is not None and command is not None and current != command:
        enable()
        return True
    return False
