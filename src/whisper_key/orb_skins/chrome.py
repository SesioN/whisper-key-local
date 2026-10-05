import math

import numpy as np

from .base import smoothstep
from .blob import BlobSkin

COLOR_SKY = (206, 222, 242)
COLOR_GROUND = (62, 71, 88)
COLOR_BOUNCE = (158, 174, 196)
COLOR_BAND = (255, 251, 240)
COLOR_RIM = (158, 184, 220)


class ChromeSkin(BlobSkin):

    LABEL = "Chrome"

    CANVAS_FACTOR = 2.0

    BADGE_FILL = (38, 44, 56)
    BADGE_RING = COLOR_RIM
    BADGE_GLYPH = COLOR_SKY
    SUPERSAMPLE = 2

    HARMONIC_WEIGHT = (0.032, 0.022, 0.014, 0.009)

    SPEC_ANGLE = -2.36
    SPEC_DISTANCE = 0.44
    SPEC_WIDTH = 0.19
    HORIZON_HEIGHT = 0.38

    def __init__(self, canvas: int, orb_radius: float):
        super().__init__(canvas, orb_radius)
        self.sky = np.array(COLOR_SKY, dtype=np.float32)
        self.ground = np.array(COLOR_GROUND, dtype=np.float32)
        self.bounce_col = np.array(COLOR_BOUNCE, dtype=np.float32)
        self.band_col = np.array(COLOR_BAND, dtype=np.float32)
        self.rim_col = np.array(COLOR_RIM, dtype=np.float32)
        self.white = np.float32(255.0)

    def shade(self, state: str, level: float, t: float):
        if state == "recording":
            deform = 0.50 + 1.8 * level
            scale = 1.0 + 0.10 * level
            slosh = 0.26 * level
            spec_angle = self.SPEC_ANGLE
            glow_strength = 0.16 + 0.14 * level
        elif state == "processing":
            deform = 0.70
            scale = 1.0 + 0.030 * math.sin(t * 4.2)
            slosh = 0.05
            spec_angle = self.SPEC_ANGLE - t * 1.5
            glow_strength = 0.18
        else:
            deform = 0.26
            scale = 1.0 + 0.030 * math.sin(t * 2.0 * math.pi / 5.0)
            slosh = 0.0
            spec_angle = self.SPEC_ANGLE
            glow_strength = 0.12

        radius = self._radius_field(t, deform, scale)

        body = smoothstep(1.2 * self.px, -1.2 * self.px, self.rr - radius)
        outside = np.maximum(self.rr - radius, 0.0) / (self.r0 * 0.26)
        glow = np.exp(-outside * outside) * glow_strength * (1.0 - body)

        u = np.clip(self.rr / np.maximum(radius, 1e-6), 0.0, 1.0)
        nz = np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
        ny = np.clip(self.dy / np.maximum(radius, 1e-6), -1.0, 1.0)

        horizon = (self.HORIZON_HEIGHT + 0.05 * math.sin(t * 0.55)
                   + slosh * math.sin(t * 2.3))
        h = 2.0 * nz * ny + horizon
        sky_mix = smoothstep(0.34, -0.62, h)[..., None]
        bounce = smoothstep(0.86, 1.34, h)[..., None]
        band = np.exp(-(h / 0.15) ** 2)

        rgb = self.ground + (self.sky - self.ground) * sky_mix
        rgb = rgb + (self.bounce_col - rgb) * (0.65 * bounce)
        rgb = rgb + (self.band_col - rgb) * (0.85 * band)[..., None]

        fresnel = (1.0 - nz) ** 3
        rgb = rgb + (self.rim_col - rgb) * (0.55 * fresnel)[..., None]

        sx = self.SPEC_DISTANCE * self.r0 * math.cos(spec_angle)
        sy = self.SPEC_DISTANCE * self.r0 * math.sin(spec_angle)
        d2 = (self.dx - sx) ** 2 + (self.dy - sy) ** 2
        spec = np.exp(-d2 / (self.SPEC_WIDTH * self.r0) ** 2)
        rgb = rgb + (self.white - rgb) * np.clip(spec, 0.0, 1.0)[..., None]

        return rgb, body + glow
