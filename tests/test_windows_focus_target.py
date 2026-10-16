import importlib.util
import os
import sys

import pytest

from conftest import PLATFORM_DIR

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


def load_windows_window_style():
    spec = importlib.util.spec_from_file_location("windows_window_style", PLATFORM_DIR / "windows" / "window_style.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_own_process_window_is_not_captured(monkeypatch):
    window_style = load_windows_window_style()

    def fake_thread_process_id(window, process_id_pointer):
        if process_id_pointer is not None:
            process_id_pointer._obj.value = os.getpid()
        return 42

    monkeypatch.setattr(window_style, "get_foreground_window", lambda: 1234)
    monkeypatch.setattr(window_style.user32, "GetWindowThreadProcessId", fake_thread_process_id)
    assert window_style.capture_focus_target() is None
