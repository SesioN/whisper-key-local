import math

import numpy as np

from .base import OrbSkin, smoothstep

COLOR_RING = (255, 246, 224)
COLOR_DISK_INNER = (255, 201, 104)
COLOR_DISK_OUTER = (216, 96, 26)
COLOR_DISK_EDGE = (112, 30, 8)
COLOR_LENS = (255, 208, 128)
COLOR_HORIZON = (6, 4, 8)
COLOR_WHITE = (255, 250, 240)


class BlackHoleSkin(OrbSkin):

    LABEL = "Black hole"

    CANVAS_FACTOR = 2.1
    SUPERSAMPLE = 2

    TILT = 0.52
    HORIZON = 0.46
    DISK_IN = 0.62
    DISK_OUT = 1.30
    PHOTON_RING = 1.10
    LENS_R = 0.72
    LENS_W = 0.115
    LENS_TILT = 0.88
    DOPPLER = 0.85
    SPIRAL = 1.1

    def __init__(self, canvas: int, orb_radius: float):
        super().__init__(canvas, orb_radius)
        r0 = self.r0
        r_h = self.HORIZON * r0
        r_in = self.DISK_IN * r0
        r_out = self.DISK_OUT * r0

        dyt = self.dy / self.TILT
        self.rd = np.sqrt(self.dx * self.dx + dyt * dyt)
        self.phi = np.arctan2(dyt, self.dx)
        self.rd_n = self.rd / r0

        psi0 = self.phi + self.SPIRAL * np.log(np.maximum(self.rd, r_in * 0.5) / r_in)
        self.band2 = (np.sin(2.0 * psi0 + 1.7), np.cos(2.0 * psi0 + 1.7))
        self.band3 = (np.sin(3.0 * psi0 - 0.6), np.cos(3.0 * psi0 - 0.6))
        self.wave = (np.sin(self.rd_n * 5.5), np.cos(self.rd_n * 5.5))
        self.lens_wave = (np.sin(self.th * 2.0), np.cos(self.th * 2.0))

        u = np.clip((self.rd - r_in) / max(r_out - r_in, 1e-6), 0.0, 1.0)
        profile = (smoothstep(r_in * 0.86, r_in, self.rd)
                   * smoothstep(r_out, r_out * 0.62, self.rd)
                   * (1.35 - 0.55 * u))
        halo = np.exp(-(np.maximum(self.rd - r_out, 0.0) / (0.30 * r0)) ** 2) * 0.07

        beam = 1.0 + self.DOPPLER * -np.cos(self.phi)
        self.prof_beam = profile * beam
        self.prof_halo = halo * beam

        self.front_mask = smoothstep(-1.5 * self.px, 1.5 * self.px, self.dy)
        self.back_mask = 1.0 - self.front_mask
        self.horizon = smoothstep(1.0 * self.px, -1.0 * self.px, self.rr - r_h)

        sigma = max(0.9 * self.px, 0.038 * r0)
        self.photon = np.exp(-((self.rr - self.PHOTON_RING * r_h) / sigma) ** 2)

        dyl = self.dy / self.LENS_TILT
        rl = np.sqrt(self.dx * self.dx + dyl * dyl)
        lens = np.exp(-((rl - self.LENS_R * r0) / (self.LENS_W * r0)) ** 2)
        self.lens_prof = lens * (0.62 + 0.55 * self.back_mask) * (1.0 - 0.30 * np.cos(self.th))

        w = u[..., None]
        inner = np.array(COLOR_DISK_INNER, dtype=np.float32)
        outer = np.array(COLOR_DISK_OUTER, dtype=np.float32)
        edge = np.array(COLOR_DISK_EDGE, dtype=np.float32)
        self.white = np.array(COLOR_WHITE, dtype=np.float32)
        col = np.where(w < 0.5,
                       inner + (outer - inner) * np.clip(w / 0.5, 0.0, 1.0),
                       outer + (edge - outer) * np.clip((w - 0.5) / 0.5, 0.0, 1.0))
        hot = np.clip((beam - 1.0) / self.DOPPLER, 0.0, 1.0)[..., None]
        self.disk_col = col + (self.white - col) * (0.40 * hot)

        self.lens_col = np.array(COLOR_LENS, dtype=np.float32)
        self.ring_col = np.array(COLOR_RING, dtype=np.float32)
        self.horizon_col = np.array(COLOR_HORIZON, dtype=np.float32)

    def shade(self, state: str, level: float, t: float):
        if state == "recording":
            spin = 0.45 + 1.40 * level
            gain = 0.85 + 0.55 * level
            ring_gain = 1.05 + 0.45 * level
        elif state == "processing":
            spin = 0.90
            gain = 0.95
            ring_gain = 1.05 + 0.16 * math.sin(t * 2.4)
        else:
            spin = 0.35
            gain = 0.66
            ring_gain = 0.95 + 0.06 * math.sin(t * 2.0 * math.pi / 5.0)

        s2, c2 = self.band2
        s3, c3 = self.band3
        sw, cw = self.wave
        a2, a3, aw = 2.0 * spin * t, 3.0 * spin * t, 0.7 * t
        bands = (0.55
                 + 0.32 * (s2 * math.cos(a2) + c2 * math.sin(a2))
                 + 0.20 * (s3 * math.cos(a3) + c3 * math.sin(a3))
                 + 0.10 * (sw * math.cos(aw) - cw * math.sin(aw)))

        disk = (self.prof_beam * np.clip(bands, 0.0, 1.8) + self.prof_halo) * gain
        w_disk = disk * self.front_mask + disk * self.back_mask * (1.0 - self.horizon)

        sl, cl = self.lens_wave
        al = 0.5 * t
        lens = (self.lens_prof
                * (0.58 + 0.42 * (sl * math.cos(al) - cl * math.sin(al)))
                * gain * (1.0 - self.horizon))
        ring = self.photon * ring_gain

        w_sum = w_disk + lens + ring
        rgb = self.disk_col * w_disk[..., None]
        rgb += self.lens_col * lens[..., None]
        rgb += self.ring_col * ring[..., None]
        rgb /= np.maximum(w_sum, 1e-4)[..., None]

        over = np.clip(w_sum - 1.0, 0.0, 1.0)[..., None] * 0.45
        rgb *= 1.0 - over
        rgb += self.white * over

        emissive = np.clip(w_sum, 0.0, 1.0)
        cover = self.horizon * (1.0 - emissive)
        rgb = rgb * (1.0 - cover[..., None]) + self.horizon_col * cover[..., None]
        return rgb, emissive + cover
