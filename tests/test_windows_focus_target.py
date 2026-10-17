import importlib.util
import os
import sys
import types

import pytest

from conftest import PLATFORM_DIR

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


@pytest.fixture
def window_style(monkeypatch):
    package = types.ModuleType("windows_platform_under_test")
    package.__path__ = [str(PLATFORM_DIR / "windows")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for module_name in ("keyboard", "window_style"):
        full_name = f"{package.__name__}.{module_name}"
        spec = importlib.util.spec_from_file_location(full_name, PLATFORM_DIR / "windows" / f"{module_name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module


def test_own_process_window_is_not_captured(window_style, monkeypatch):
    def fake_thread_process_id(window, process_id_pointer):
        if process_id_pointer is not None:
            process_id_pointer._obj.value = os.getpid()
        return 42

    monkeypatch.setattr(window_style, "get_foreground_window", lambda: 1234)
    monkeypatch.setattr(window_style.user32, "GetWindowThreadProcessId", fake_thread_process_id)
    assert window_style.capture_focus_target() is None


def test_child_control_from_another_process_must_get_focus_back(window_style, monkeypatch):
    target = window_style.FocusTarget(1, 2, 10, "Tauri Window", "swarmterm.exe")
    focused = {"control": 1}
    set_focus_calls = []
    monkeypatch.setattr(window_style.user32, "IsWindow", lambda window: True)
    monkeypatch.setattr(window_style.user32, "IsChild", lambda parent, child: True)
    monkeypatch.setattr(window_style.user32, "SetFocus", lambda control: set_focus_calls.append(control) or focused.update(control=control))
    monkeypatch.setattr(window_style, "_thread_focused_control", lambda thread_id: focused["control"])
    assert not window_style._focused_control_restored(target)
    window_style._restore_focused_control(target)
    assert set_focus_calls == [2]
    assert window_style._focused_control_restored(target)


def test_control_no_longer_inside_the_window_is_ignored(window_style, monkeypatch):
    target = window_style.FocusTarget(1, 2, 10, "Notepad", "notepad.exe")
    set_focus_calls = []
    monkeypatch.setattr(window_style.user32, "IsWindow", lambda window: True)
    monkeypatch.setattr(window_style.user32, "IsChild", lambda parent, child: False)
    monkeypatch.setattr(window_style.user32, "SetFocus", lambda control: set_focus_calls.append(control))
    window_style._restore_focused_control(target)
    assert set_focus_calls == []


def test_refocus_retries_with_foreground_unlock_until_confirmed(window_style, monkeypatch):
    target = window_style.FocusTarget(1, 0, 10, "Notepad", "notepad.exe")
    active_states = iter([False, False, True, True])
    attempts = []
    monkeypatch.setattr(window_style.user32, "IsWindow", lambda window: True)
    monkeypatch.setattr(window_style.user32, "IsIconic", lambda window: False)
    monkeypatch.setattr(window_style, "is_focus_target_active", lambda checked_target: next(active_states))
    monkeypatch.setattr(window_style, "describe_foreground", lambda: "other")
    monkeypatch.setattr(window_style, "_attempt_activation",
                        lambda checked_target, unlock_foreground: attempts.append(unlock_foreground) or True)
    assert window_style.restore_focus_target(target, settle_seconds=0)
    assert attempts == [False, True]


def test_refocus_gives_up_after_the_timeout(window_style, monkeypatch):
    target = window_style.FocusTarget(1, 0, 10, "Notepad", "notepad.exe")
    clock = iter(range(100))
    monkeypatch.setattr(window_style.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(window_style.user32, "IsWindow", lambda window: True)
    monkeypatch.setattr(window_style.user32, "IsIconic", lambda window: False)
    monkeypatch.setattr(window_style, "is_focus_target_active", lambda checked_target: False)
    monkeypatch.setattr(window_style, "describe_foreground", lambda: "other")
    monkeypatch.setattr(window_style, "_attempt_activation", lambda checked_target, unlock_foreground: False)
    assert not window_style.restore_focus_target(target, settle_seconds=0, timeout_seconds=3)


def test_active_window_with_moved_control_is_refocused(window_style, monkeypatch):
    target = window_style.FocusTarget(1, 2, 10, "Tauri Window", "swarmterm.exe")
    focused = {"control": 1}
    attempts = []
    monkeypatch.setattr(window_style.user32, "IsWindow", lambda window: True)
    monkeypatch.setattr(window_style.user32, "IsChild", lambda parent, child: True)
    monkeypatch.setattr(window_style.user32, "IsIconic", lambda window: False)
    monkeypatch.setattr(window_style, "is_focus_target_active", lambda checked_target: True)
    monkeypatch.setattr(window_style, "describe_foreground", lambda: "swarmterm")
    monkeypatch.setattr(window_style, "_thread_focused_control", lambda thread_id: focused["control"])

    def activate(checked_target, unlock_foreground):
        attempts.append(unlock_foreground)
        focused["control"] = 2
        return True

    monkeypatch.setattr(window_style, "_attempt_activation", activate)
    assert window_style.restore_focus_target(target, settle_seconds=0)
    assert attempts == [False]


@pytest.fixture
def activation_calls(window_style, monkeypatch):
    calls = {"attach": [], "detach": [], "empty_input": 0}

    def attach(own_thread, thread_id, attach_flag):
        if attach_flag:
            calls["attach"].append(thread_id)
            return thread_id != 30
        calls["detach"].append(thread_id)
        return True

    monkeypatch.setattr(window_style.kernel32, "GetCurrentThreadId", lambda: 5)
    monkeypatch.setattr(window_style, "get_foreground_window", lambda: 77)
    monkeypatch.setattr(window_style, "_thread_of", lambda window: 30)
    monkeypatch.setattr(window_style.user32, "AttachThreadInput", attach)
    monkeypatch.setattr(window_style.user32, "BringWindowToTop", lambda window: True)
    monkeypatch.setattr(window_style, "_restore_focused_control", lambda target: None)
    monkeypatch.setattr(window_style, "send_empty_input",
                        lambda: calls.__setitem__("empty_input", calls["empty_input"] + 1))
    return calls


def test_activation_detaches_only_successful_attaches(window_style, activation_calls, monkeypatch):
    target = window_style.FocusTarget(1, 0, 10, "Notepad", "notepad.exe")
    monkeypatch.setattr(window_style.user32, "SetForegroundWindow", lambda window: True)
    assert window_style._attempt_activation(target, unlock_foreground=False)
    assert sorted(activation_calls["attach"]) == [10, 30]
    assert activation_calls["detach"] == [10]
    assert activation_calls["empty_input"] == 0


def test_activation_detaches_when_activation_raises_and_unlocks_on_request(window_style, activation_calls, monkeypatch):
    target = window_style.FocusTarget(1, 0, 10, "Notepad", "notepad.exe")

    def fail(window):
        raise OSError("denied")

    monkeypatch.setattr(window_style.user32, "SetForegroundWindow", fail)
    with pytest.raises(OSError):
        window_style._attempt_activation(target, unlock_foreground=True)
    assert activation_calls["detach"] == [10]
    assert activation_calls["empty_input"] == 1
