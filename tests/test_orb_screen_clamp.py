import importlib.util
import sys
import types
from pathlib import Path

import pytest

from whisper_key.voice_orb import RECT, clamp_canvas_to_rect

SCREEN = RECT(0, 0, 1920, 1080)
CANVAS = 200
INSET = 40


@pytest.mark.parametrize("position, expected", [
    ((500, 400), (500, 400)),
    ((-1000, 400), (-INSET, 400)),
    ((5000, 400), (1920 - CANVAS + INSET, 400)),
    ((500, -1000), (500, -INSET)),
    ((500, 5000), (500, 1080 - CANVAS + INSET)),
    ((5000, 5000), (1920 - CANVAS + INSET, 1080 - CANVAS + INSET)),
])
def test_clamp_keeps_the_visible_orb_on_screen(position, expected):
    assert clamp_canvas_to_rect(*position, CANVAS, INSET, SCREEN) == expected


def test_clamp_respects_monitor_offset():
    second_monitor = RECT(1920, -200, 3840, 880)
    assert clamp_canvas_to_rect(0, -1000, CANVAS, INSET, second_monitor) == (1920 - INSET, -200 - INSET)


def _real_window_style(monkeypatch):
    # conftest stubs the platform package, so load the real Windows module from its file
    if sys.platform != "win32":
        pytest.skip("Windows only")
    path = Path(__file__).parent.parent / "src" / "whisper_key" / "platform" / "windows" / "window_style.py"
    package = types.ModuleType("_real_windows")
    package.__path__ = [str(path.parent)]
    monkeypatch.setitem(sys.modules, "_real_windows", package)
    monkeypatch.setitem(sys.modules, "_real_windows.keyboard", types.SimpleNamespace(send_empty_input=lambda: None))
    spec = importlib.util.spec_from_file_location("_real_windows.window_style", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeUser32:
    def __init__(self, work=None):
        self.work = work

    def MonitorFromPoint(self, point, flags):
        return 1

    def GetCursorPos(self, point_ref):
        return True

    def GetMonitorInfoW(self, hmon, info_ref):
        if self.work is None:
            return False
        work = info_ref._obj.rcWork
        work.left, work.top, work.right, work.bottom = self.work.left, self.work.top, self.work.right, self.work.bottom
        return True


def test_orb_inset_keeps_edge_padding_between_orb_and_screen(monkeypatch):
    from whisper_key import voice_orb
    orb = voice_orb.VoiceOrb.__new__(voice_orb.VoiceOrb)
    orb._surface = {"canvas": CANVAS}
    orb._dpi = 96
    orb.size = 100
    orb.skin = voice_orb.DEFAULT_SKIN
    monkeypatch.setattr(voice_orb, "user32", _FakeUser32())
    monkeypatch.setattr(orb, "_monitor_rects", lambda hmon: (SCREEN, SCREEN))
    inset = CANVAS // 2 - orb.size // 2 - voice_orb.SCREEN_EDGE_PADDING
    assert orb._clamp_to_monitor(None, -1000, 5000) == (-inset, 1080 - CANVAS + inset)


def test_orb_edge_padding_scales_with_dpi(monkeypatch):
    from whisper_key import voice_orb
    orb = voice_orb.VoiceOrb.__new__(voice_orb.VoiceOrb)
    orb._surface = {"canvas": CANVAS * 2}
    orb._dpi = 192
    orb.size = 100
    orb.skin = voice_orb.DEFAULT_SKIN
    monkeypatch.setattr(voice_orb, "user32", _FakeUser32())
    monkeypatch.setattr(orb, "_monitor_rects", lambda hmon: (SCREEN, SCREEN))
    inset = CANVAS - orb.size - 2 * voice_orb.SCREEN_EDGE_PADDING
    assert orb._clamp_to_monitor(None, -1000, 400) == (-inset, 400)


@pytest.mark.parametrize("position, expected", [
    ((500, 400), (500, 400)),
    ((-50, -50), (0, 0)),
    ((5000, 5000), (1920 - 120, 1040 - 80)),
])
def test_widget_clamps_to_work_area(monkeypatch, position, expected):
    window_style = _real_window_style(monkeypatch)
    monkeypatch.setattr(window_style, "user32", _FakeUser32(RECT(0, 0, 1920, 1040)))
    assert window_style.clamp_to_monitor(*position, 120, 80) == expected


def test_widget_keeps_position_without_monitor_info(monkeypatch):
    window_style = _real_window_style(monkeypatch)
    monkeypatch.setattr(window_style, "user32", _FakeUser32(None))
    assert window_style.clamp_to_monitor(5000, 5000, 120, 80) == (5000, 5000)
