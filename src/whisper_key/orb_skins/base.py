import numpy as np

ALPHA_FLOOR = 0.035
DEFAULT_IDLE_OPACITY = 0.7
GRAYSCALE_WEIGHTS = (0.299, 0.587, 0.114)

def smoothstep(edge0, edge1, x):
    span = edge1 - edge0
    if abs(span) < 1e-6:
        span = 1e-6 if span >= 0 else -1e-6
    t = np.clip((x - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


RGB_TO_YIQ = np.array([[0.299, 0.587, 0.114],
                       [0.596, -0.274, -0.322],
                       [0.211, -0.523, 0.312]], dtype=np.float64)
YIQ_TO_RGB = np.linalg.inv(RGB_TO_YIQ)


def appearance_matrix(hue_degrees: float, vibrancy: float):
    if hue_degrees % 360 == 0 and vibrancy == 1.0:
        return None
    angle = np.radians(hue_degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    yiq_adjust = np.array([[1.0, 0.0, 0.0],
                           [0.0, vibrancy * cos_a, -vibrancy * sin_a],
                           [0.0, vibrancy * sin_a, vibrancy * cos_a]])
    return (YIQ_TO_RGB @ yiq_adjust @ RGB_TO_YIQ).astype(np.float32)


def apply_color_matrix(color, color_matrix):
    adjusted = np.array(color, dtype=np.float32)
    if color_matrix is not None:
        adjusted = color_matrix @ adjusted
    return tuple(int(round(channel)) for channel in np.clip(adjusted, 0.0, 255.0))


class OrbSkin:

    CANVAS_FACTOR = 2.2
    VISUAL_SCALE = 1.0

    SUPERSAMPLE = 1
    SS_BUDGET = 102_400

    LABEL = ""

    HANDLES_MUTED = False
    HANDLES_OUTCOMES = False
    IDLE_FPS = None

    BADGE_FILL = (16, 18, 24)
    BADGE_RING = (200, 204, 214)
    BADGE_GLYPH = (236, 238, 244)

    @classmethod
    def is_prepared(cls, orb_radius: float) -> bool:
        return True

    @classmethod
    def prepare(cls, orb_radius: float):
        pass

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

    def badge_colors(self, state: str):
        return self.BADGE_FILL, self.BADGE_RING, self.BADGE_GLYPH

    def _downsample(self, a):
        s = self.ss
        if s == 1:
            return a
        if s == 2:
            return (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) * 0.25
        n = self.canvas
        return a.reshape((n, s, n, s) + a.shape[2:]).mean(axis=(1, 3))

    def render_into(self, pixels, state: str, level: float, t: float, opacity: float, flash: float = 0.0,
                    flash_color=None, grayscale: bool = False, color_matrix=None):
        rgb, alpha = self.shade(state, level, t)

        if color_matrix is not None:
            rgb = rgb @ color_matrix.T

        if grayscale:
            luminance = rgb @ np.array(GRAYSCALE_WEIGHTS, dtype=np.float32)
            rgb = np.repeat(luminance[..., None], 3, axis=-1)

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

    def render(self, state: str, level: float, t: float, opacity: float, flash: float = 0.0, color_matrix=None):
        pixels = np.empty((self.canvas, self.canvas, 4), dtype=np.uint8)
        self.render_into(pixels, state, level, t, opacity, flash, color_matrix=color_matrix)
        return pixels
