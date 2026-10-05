import numpy as np

ALPHA_FLOOR = 0.035

def smoothstep(edge0, edge1, x):
    span = edge1 - edge0
    if abs(span) < 1e-6:
        span = 1e-6 if span >= 0 else -1e-6
    t = np.clip((x - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class OrbSkin:

    CANVAS_FACTOR = 2.2

    SUPERSAMPLE = 1
    SS_BUDGET = 102_400

    LABEL = ""

    HANDLES_MUTED = False
    HANDLES_OUTCOMES = False
    IDLE_FPS = None

    def __init__(self, canvas: int, orb_radius: float):
        self.canvas = canvas

        factor = self.SUPERSAMPLE
        while factor > 1 and (canvas * factor) ** 2 > self.SS_BUDGET:
            factor -= 1
        self.ss = factor

        self.px = float(factor)

        grid = canvas * factor
        self.r0 = orb_radius * factor
        c = (grid - 1) / 2.0
        ys, xs = np.mgrid[0:grid, 0:grid].astype(np.float32)
        self.dx = xs - c
        self.dy = ys - c
        self.rr = np.sqrt(self.dx * self.dx + self.dy * self.dy)
        self.th = np.arctan2(self.dy, self.dx)
        self.buffer = np.zeros((canvas, canvas, 4), dtype=np.float32)

    def shade(self, state: str, level: float, t: float):
        raise NotImplementedError

    def _downsample(self, a):
        s = self.ss
        if s == 1:
            return a
        if s == 2:
            return (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) * 0.25
        n = self.canvas
        return a.reshape((n, s, n, s) + a.shape[2:]).mean(axis=(1, 3))

    def render_into(self, pixels, state: str, level: float, t: float, opacity: float, flash: float = 0.0,
                    flash_color=None):
        rgb, alpha = self.shade(state, level, t)

        if flash > 0.0:
            if flash_color is None:
                rgb = rgb * np.float32(1.0 + 0.85 * flash)
            else:
                tint = np.array(flash_color, dtype=np.float32)
                rgb = rgb + (tint - rgb) * np.float32(0.6 * flash)

        alpha = np.clip(alpha, 0.0, 1.0)
        premul = np.clip(rgb, 0.0, 255.0)
        premul *= alpha[..., None]

        alpha = self._downsample(alpha) * np.float32(opacity * 255.0)
        premul = self._downsample(premul) * np.float32(opacity)

        visible = alpha >= ALPHA_FLOOR * 255.0
        alpha *= visible
        premul *= visible[..., None]

        out = self.buffer
        out[..., :3] = premul[..., ::-1]
        out[..., 3] = alpha
        np.copyto(pixels, out, casting="unsafe")

    def render(self, state: str, level: float, t: float, opacity: float, flash: float = 0.0):
        pixels = np.empty((self.canvas, self.canvas, 4), dtype=np.uint8)
        self.render_into(pixels, state, level, t, opacity, flash)
        return pixels
