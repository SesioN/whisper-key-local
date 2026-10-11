import numpy as np
import pytest

from whisper_key.orb_skins import SKINS
from whisper_key.orb_skins.base import DEFAULT_IDLE_OPACITY, solidity_for_opacity

ORB_RADIUS = 30.0


def _renderer(skin_class):
    if not skin_class.is_prepared(ORB_RADIUS):
        skin_class.prepare(ORB_RADIUS)
    return skin_class(int(ORB_RADIUS * 2 * skin_class.CANVAS_FACTOR), ORB_RADIUS)


def _body_alpha(renderer, pixels):
    body = renderer._downsample(renderer.body_mask()) >= 0.999
    return pixels[..., 3][body]


@pytest.mark.parametrize("skin_name", sorted(SKINS))
def test_full_opacity_makes_the_orb_body_solid(skin_name):
    renderer = _renderer(SKINS[skin_name])
    pixels = renderer.render("idle", 0.0, 1.0, 1.0, solidity=1.0)
    assert _body_alpha(renderer, pixels).min() >= 254


@pytest.mark.parametrize("skin_name", sorted(SKINS))
def test_default_opacity_keeps_the_skin_unchanged(skin_name):
    renderer = _renderer(SKINS[skin_name])
    plain = renderer.render("idle", 0.0, 1.0, 1.0).copy()
    with_zero_solidity = renderer.render("idle", 0.0, 1.0, 1.0, solidity=solidity_for_opacity(DEFAULT_IDLE_OPACITY))
    assert np.array_equal(plain, with_zero_solidity)


def test_solidity_ramps_from_default_to_full_opacity():
    assert solidity_for_opacity(0.2) == 0.0
    assert solidity_for_opacity(DEFAULT_IDLE_OPACITY) == 0.0
    assert solidity_for_opacity(0.85) == pytest.approx(0.5)
    assert solidity_for_opacity(1.0) == 1.0
