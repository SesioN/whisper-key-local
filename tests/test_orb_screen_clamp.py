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
