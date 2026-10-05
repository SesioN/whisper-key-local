import math

import numpy as np

from .base import smoothstep
from .blob import BlobSkin

COLOR_BODY = (206, 230, 245)
COLOR_RIM_WARM = (255, 232, 204)
COLOR_RIM_COOL = (198, 222, 255)
COLOR_CONTOUR = (52, 72, 92)
COLOR_CAUSTIC = (255, 246, 220)


class GlassSkin(BlobSkin):

    LABEL = "Glass drop"

    CANVAS_FACTOR = 2.0
    SUPERSAMPLE = 2

    HARMONIC_WEIGHT = (0.042, 0.029, 0.019, 0.012)

    RIM_U = 0.94
    RIM_W = 0.070
    CONTOUR_U = 0.78
    CONTOUR_W = 0.105
    SPEC_ANGLE = -2.36

    def __init__(self, canvas: int, orb_radius: float):
        super().__init__(canvas, orb_radius)
        self.body_col = np.array(COLOR_BODY, dtype=np.float32)
        self.rim_warm = np.array(COLOR_RIM_WARM, dtype=np.float32)
        self.rim_cool = np.array(COLOR_RIM_COOL, dtype=np.float32)
        self.contour_col = np.array(COLOR_CONTOUR, dtype=np.float32)
        self.caustic_col = np.array(COLOR_CAUSTIC, dtype=np.float32)
        self.white = np.array((255.0, 255.0, 255.0), dtype=np.float32)

    def shade(self, state: str, level: float, t: float):
        if state == "recording":
            deform = 0.60 + 2.0 * level
            scale = 1.0 + 0.11 * level
            body_alpha = 0.13 + 0.09 * level
            rim_gain = 0.75 + 0.30 * level
            sweep = 0.0
        elif state == "processing":
            deform = 0.85
            scale = 1.0 + 0.035 * math.sin(t * 4.2)
            body_alpha = 0.15
            rim_gain = 0.85
            sweep = 1.3
        else:
            deform = 0.30
            scale = 1.0 + 0.040 * math.sin(t * 2.0 * math.pi / 5.0)
            body_alpha = 0.11
            rim_gain = 0.62
            sweep = 0.0

        radius = self._radius_field(t, deform, scale)
        edge = smoothstep(1.2 * self.px, -1.2 * self.px, self.rr - radius)
        un = self.rr / np.maximum(radius, 1e-6)

        rim = np.exp(-((un - self.RIM_U) / self.RIM_W) ** 2)
        if sweep > 0.0:
            rim = rim * (1.0 + sweep * np.clip(np.cos(self.th - t * 2.2), 0.0, 1.0) ** 6)
        rim = rim * rim_gain

        rim_col = self.rim_warm + (self.rim_cool - self.rim_warm) * smoothstep(
            self.RIM_U - self.RIM_W, self.RIM_U + self.RIM_W, un)[..., None]

        contour = np.exp(-((un - self.CONTOUR_U) / self.CONTOUR_W) ** 2) * 0.34

        sx = 0.46 * self.r0 * math.cos(self.SPEC_ANGLE)
        sy = 0.46 * self.r0 * math.sin(self.SPEC_ANGLE)
        spec = np.exp(-((self.dx - sx) ** 2 + (self.dy - sy) ** 2) / (0.15 * self.r0) ** 2)

        caustic = np.exp(-((self.dx * 0.7) ** 2 + (self.dy - 0.46 * self.r0) ** 2)
                         / (0.44 * self.r0) ** 2) * edge * 0.22

        w_body = edge * body_alpha
        w_sum = w_body + rim + contour + spec + caustic

        rgb = rim_col * rim[..., None]
        rgb += self.body_col * w_body[..., None]
        rgb += self.contour_col * contour[..., None]
        rgb += self.white * spec[..., None]
        rgb += self.caustic_col * caustic[..., None]
        rgb /= np.maximum(w_sum, 1e-4)[..., None]

        over = np.clip(w_sum - 1.0, 0.0, 1.0)[..., None] * 0.55
        rgb *= 1.0 - over
        rgb += self.white * over
        return rgb, w_sum
