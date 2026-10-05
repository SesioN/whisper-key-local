import math

import numpy as np

from .base import OrbSkin, smoothstep


class BlobSkin(OrbSkin):

    CANVAS_FACTOR = 2.2

    COLOR_CORE = (240, 217, 138)
    COLOR_MID = (200, 168, 85)
    COLOR_EDGE = (184, 152, 63)

    SHIMMER_STRENGTH = 0.38
    SHIMMER_SHARPNESS = 1.6

    HARMONICS = (2, 3, 5, 7)
    HARMONIC_SPEED = (0.61, -0.43, 0.29, -0.19)
    HARMONIC_WEIGHT = (0.055, 0.038, 0.026, 0.017)

    def badge_colors(self, state: str):
        if self.BADGE_RING is not OrbSkin.BADGE_RING:
            return super().badge_colors(state)
        fill = tuple(int(channel * 0.16) for channel in self.COLOR_EDGE)
        return fill, self.COLOR_MID, self.COLOR_CORE

    def __init__(self, canvas: int, orb_radius: float):
        super().__init__(canvas, orb_radius)
        self.light = np.clip((-self.dx - self.dy) / (2.0 * self.r0) * 0.5 + 0.5, 0.0, 1.0)
        self.core = np.array(self.COLOR_CORE, dtype=np.float32)
        self.mid = np.array(self.COLOR_MID, dtype=np.float32)
        self.edge = np.array(self.COLOR_EDGE, dtype=np.float32)
        self.harmonic_basis = [(np.sin(k * self.th), np.cos(k * self.th)) for k in self.HARMONICS]
        self.th2_basis = (np.sin(2.0 * self.th), np.cos(2.0 * self.th))
        self.lighting = (0.80 + 0.30 * self.light)[..., None].astype(np.float32)

    def _radius_field(self, t: float, deform: float, scale: float):
        r = np.zeros_like(self.th)
        for (sin_k, cos_k), speed, weight in zip(self.harmonic_basis, self.HARMONIC_SPEED, self.HARMONIC_WEIGHT):
            phase = speed * t * 2.0 * math.pi
            r += sin_k * np.float32(weight * math.cos(phase))
            r += cos_k * np.float32(weight * math.sin(phase))
        r *= np.float32(deform)
        r += np.float32(1.0)
        r *= np.float32(self.r0 * scale)
        return r

    def _state_params(self, state: str, level: float, t: float):
        if state == "recording":
            return (0.55 + 2.2 * level,
                    1.0 + 0.13 * level,
                    0.32 + 0.32 * level,
                    self.r0 * (0.32 + 0.16 * level))
        if state == "processing":
            return (0.85,
                    1.0 + 0.035 * math.sin(t * 4.2),
                    0.30,
                    self.r0 * 0.34)
        return (0.32,
                1.0 + 0.040 * math.sin(t * 2.0 * math.pi / 5.0),
                0.20,
                self.r0 * 0.30)

    def shade(self, state: str, level: float, t: float):
        deform, scale, glow_strength, glow_width = self._state_params(state, level, t)
        radius = self._radius_field(t, deform, scale)

        body = smoothstep(1.2 * self.px, -1.2 * self.px, self.rr - radius)
        outside = np.maximum(self.rr - radius, 0.0) / glow_width
        glow = np.exp(-outside * outside) * glow_strength * (1.0 - body)

        u = np.clip(self.rr / np.maximum(radius, 1e-6), 0.0, 1.0)
        flow = 0.5 + 0.5 * (self.th2_basis[0] * np.float32(math.cos(t * 1.1))
                            + self.th2_basis[1] * np.float32(math.sin(t * 1.1))) * np.sin(u * 3.4 - t * 0.8)
        u = np.clip(u * (0.86 + 0.20 * flow), 0.0, 1.0)

        w = u[..., None]
        inner = self.core + (self.mid - self.core) * np.clip(w / 0.55, 0.0, 1.0)
        outer = self.mid + (self.edge - self.mid) * np.clip((w - 0.55) / 0.45, 0.0, 1.0)
        rgb = np.where(w < 0.55, inner, outer)
        rgb *= self.lighting

        if state == "processing":
            shimmer = np.clip(np.cos(self.th - t * 2.6), 0.0, 1.0) ** self.SHIMMER_SHARPNESS
            rgb = rgb * (1.0 + self.SHIMMER_STRENGTH * shimmer[..., None])

        return rgb, body + glow
