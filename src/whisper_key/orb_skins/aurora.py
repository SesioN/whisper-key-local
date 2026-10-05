import math

import numpy as np

from .base import OrbSkin

COLOR_GREEN = (86, 236, 168)
COLOR_TEAL = (66, 180, 238)
COLOR_VIOLET = (176, 122, 250)
COLOR_VEIL = (222, 255, 240)


class AuroraSkin(OrbSkin):

    LABEL = "Aurora"

    CANVAS_FACTOR = 2.2

    BADGE_FILL = (8, 28, 34)
    BADGE_RING = COLOR_TEAL
    BADGE_GLYPH = COLOR_VEIL

    def __init__(self, canvas: int, orb_radius: float):
        super().__init__(canvas, orb_radius)
        self.nx = self.dx / self.r0
        self.ny = self.dy / self.r0
        rn = self.rr / self.r0

        self.disc = np.exp(-(rn / 0.80) ** 3)
        self.glow = np.exp(-(rn / 1.25) ** 2) * 0.22

        self.green = np.array(COLOR_GREEN, dtype=np.float32)
        self.teal = np.array(COLOR_TEAL, dtype=np.float32)
        self.violet = np.array(COLOR_VIOLET, dtype=np.float32)
        self.veil_col = np.array(COLOR_VEIL, dtype=np.float32)

    def _projection(self, angle: float):
        return 0.5 + 0.5 * (self.nx * math.cos(angle) + self.ny * math.sin(angle))

    def shade(self, state: str, level: float, t: float):
        if state == "recording":
            speed = 0.85 + 1.5 * level
            gain = 0.88 + 0.40 * level
            veil_gain = 0.55 + 0.45 * level
        elif state == "processing":
            speed = 1.6
            gain = 1.00
            veil_gain = 0.75
        else:
            speed = 0.40
            gain = 0.86
            veil_gain = 0.52

        hue = self._projection(0.34 * speed * t)
        lum = self._projection(1.9 - 0.21 * speed * t)

        rgb = self.green + (self.teal - self.green) * np.clip(hue / 0.5, 0.0, 1.0)[..., None]
        rgb = rgb + (self.violet - rgb) * np.clip((hue - 0.5) / 0.5, 0.0, 1.0)[..., None]
        rgb = rgb * (0.80 + 0.45 * lum)[..., None]

        veil = np.exp(-((lum - 0.62) / 0.15) ** 2) * veil_gain
        rgb = rgb + (self.veil_col - rgb) * np.clip(0.75 * veil, 0.0, 1.0)[..., None]

        alpha = self.disc * (0.78 + 0.34 * veil) * gain + self.glow * gain
        return rgb, alpha
