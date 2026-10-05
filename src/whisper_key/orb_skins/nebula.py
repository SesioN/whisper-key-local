import math

import numpy as np

from .base import OrbSkin, smoothstep
from .glyphs import check_mark, exclamation, mic_slash, ring_badge

U_MAX = 1.7
TEXTURE_RADIAL = 384
TEXTURE_ANGULAR = 1024
FADE_SECONDS = 0.28
SPIRAL_TWIST = 1.35

PALETTES = {
    "idle": {"warm": (70, 190, 255), "cool": (150, 110, 255), "haze": (40, 80, 180),
             "core": (2, 5, 16), "glyph": (110, 220, 255)},
    "recording": {"warm": (70, 205, 255), "cool": (175, 115, 255), "haze": (46, 90, 200),
                  "core": (2, 5, 16), "glyph": (120, 235, 255)},
    "processing": {"warm": (150, 95, 255), "cool": (215, 160, 255), "haze": (60, 32, 140),
                   "core": (8, 3, 22), "glyph": (235, 205, 255)},
    "muted": {"warm": (165, 200, 255), "cool": (190, 170, 255), "haze": (60, 90, 170),
              "core": (12, 22, 48), "glyph": (200, 220, 255)},
    "success": {"warm": (60, 235, 150), "cool": (160, 255, 200), "haze": (14, 70, 46),
                "core": (1, 12, 7), "glyph": (110, 255, 165)},
    "error": {"warm": (255, 45, 40), "cool": (255, 105, 80), "haze": (80, 12, 10),
              "core": (14, 1, 1), "glyph": (255, 80, 65)},
}

_TEXTURE_CACHE = {}


def _polar_grid():
    u = ((np.arange(TEXTURE_RADIAL, dtype=np.float32) + 0.5) / TEXTURE_RADIAL * U_MAX)[:, None]
    theta = (np.arange(TEXTURE_ANGULAR, dtype=np.float32) / TEXTURE_ANGULAR * 2.0 * np.pi - np.pi)[None, :]
    return u, theta


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _orbit_texture(seed, count, width, radius_range, aspect_range):
    rng = np.random.default_rng(seed)
    u, theta = _polar_grid()
    texture = np.zeros((TEXTURE_RADIAL, TEXTURE_ANGULAR), dtype=np.float32)
    for _ in range(count):
        major = rng.uniform(*radius_range)
        minor = major * rng.uniform(*aspect_range)
        tilt = rng.uniform(0.0, np.pi)
        local = theta - tilt
        radius = major * minor / np.sqrt((minor * np.cos(local)) ** 2 + (major * np.sin(local)) ** 2)
        line = np.exp(-((u - radius) / width) ** 2)
        facing = 0.45 + 0.55 * np.cos(local - rng.uniform(0.0, 2.0 * np.pi)) ** 2
        beads = 0.55 + 0.45 * np.cos(theta * rng.integers(40, 130) + rng.uniform(0.0, 6.3)) ** 8
        texture += line * facing * beads * rng.uniform(0.45, 1.0)
    return texture


def _spiral_texture(seed, width):
    rng = np.random.default_rng(seed)
    u, theta = _polar_grid()
    texture = np.zeros((TEXTURE_RADIAL, TEXTURE_ANGULAR), dtype=np.float32)
    log_u = np.log(np.maximum(u, 0.02))
    for arm in range(4):
        for strand in range(4):
            offset = arm * np.pi / 2.0 + rng.uniform(-0.22, 0.22)
            twist = SPIRAL_TWIST * rng.uniform(0.85, 1.15)
            angle = _wrap(theta - twist * log_u - offset)
            spread = width / np.maximum(u, 0.08) * rng.uniform(0.9, 1.6)
            beads = 0.5 + 0.5 * np.cos(log_u * rng.uniform(30, 60) + rng.uniform(0.0, 6.3)) ** 6
            texture += np.exp(-(angle / spread) ** 2) * beads * rng.uniform(0.45, 1.0)
    texture *= smoothstep(0.04, 0.22, u) * smoothstep(1.12, 0.80, u)
    return texture


def _haze_texture(seed):
    rng = np.random.default_rng(seed)
    u, theta = _polar_grid()
    field = np.zeros((TEXTURE_RADIAL, TEXTURE_ANGULAR), dtype=np.float32)
    for k in range(2, 9):
        field += (np.cos(k * theta + rng.uniform(0.0, 6.3))
                  * np.cos(u * rng.uniform(4.0, 14.0) + rng.uniform(0.0, 6.3)) / k)
    field -= field.min()
    field /= field.max()
    return (0.35 + 0.65 * field).astype(np.float32)


def _lump_profile(seed):
    rng = np.random.default_rng(seed)
    theta = np.arange(TEXTURE_ANGULAR, dtype=np.float32) / TEXTURE_ANGULAR * 2.0 * np.pi
    profile = np.zeros(TEXTURE_ANGULAR, dtype=np.float32)
    for k in range(3, 11):
        profile += rng.uniform(0.4, 1.0) / k * np.cos(k * theta + rng.uniform(0.0, 6.3))
    return profile / np.abs(profile).max()


def _textures(width):
    key = round(width, 3)
    if key not in _TEXTURE_CACHE:
        _TEXTURE_CACHE[key] = {
            "orbit_forward": _orbit_texture(11, 16, width, (0.72, 1.04), (0.55, 0.95)),
            "orbit_reverse": _orbit_texture(23, 12, width, (0.66, 1.00), (0.60, 0.98)),
            "rings": (_orbit_texture(37, 14, width, (0.60, 1.02), (0.90, 1.0))
                      + 0.6 * _orbit_texture(41, 6, width, (0.50, 0.62), (0.97, 1.0))),
            "spiral": _spiral_texture(53, width * 1.6),
            "haze": _haze_texture(67),
            "lumps_a": _lump_profile(71),
            "lumps_b": _lump_profile(89),
        }
    return _TEXTURE_CACHE[key]


class NebulaSkin(OrbSkin):

    LABEL = "Nebula"

    CANVAS_FACTOR = 1.55
    VISUAL_SCALE = 1.15
    SUPERSAMPLE = 1
    HANDLES_MUTED = True
    HANDLES_OUTCOMES = True
    IDLE_FPS = 15

    WAVE_BARS = 21

    def __init__(self, canvas: int, orb_radius: float):
        super().__init__(canvas, orb_radius)
        self.u = (self.rr / self.r0).astype(np.float32)
        self.gx = (self.dx / self.r0).astype(np.float32)
        self.gy = (self.dy / self.r0).astype(np.float32)
        self.pixel_u = np.float32(1.0 / self.r0)
        self.cos_th = np.cos(self.th)
        self.sin_th = np.sin(self.th)

        line_width = max(0.011, 1.15 / self.r0)
        thickness_gain = np.float32((0.011 / line_width) ** 0.75)
        self.textures = {name: (tex.ravel() if tex.ndim == 2 else tex)
                         for name, tex in _textures(line_width).items()}
        for name in ("orbit_forward", "orbit_reverse", "rings", "spiral"):
            self.textures[name] = self.textures[name] * thickness_gain
        radial_index = np.clip((self.u / U_MAX * TEXTURE_RADIAL).astype(np.int32), 0, TEXTURE_RADIAL - 1)
        self.radial_offset = radial_index * TEXTURE_ANGULAR
        self.angle_index = (((self.th + np.pi) / (2.0 * np.pi) * TEXTURE_ANGULAR).astype(np.int32)
                            % TEXTURE_ANGULAR)

        self.palettes = {state: {name: np.array(color, dtype=np.float32) for name, color in palette.items()}
                         for state, palette in PALETTES.items()}
        self.white = np.array((255, 255, 255), dtype=np.float32)

        pixel = self.pixel_u
        self.glyphs = {
            "ring": ring_badge(self.gx, self.gy, pixel),
            "mic": mic_slash(self.gx, self.gy, pixel),
            "check": check_mark(self.gx, self.gy, pixel),
            "exclamation": exclamation(self.gx, self.gy, pixel),
        }
        self._build_waveform_fields()
        self._hex = None

        self._state = None
        self._previous_state = None
        self._state_since = 0.0
        self._last_t = None
        self._phase = {"forward": 0.0, "reverse": 0.0, "haze": 0.0, "spiral": 0.0, "lumps": 0.0, "hue": 0.0}

    def badge_colors(self, state: str):
        palette = PALETTES.get(state, PALETTES["idle"])
        return palette["core"], palette["warm"], palette["glyph"]

    def _build_waveform_fields(self):
        span = 0.44
        rows = np.abs(self.gy[:, 0]) < 0.36
        cols = np.abs(self.gx[0, :]) < span + 0.03
        self.wave_rows = slice(int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1])))
        self.wave_cols = slice(int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1])))
        x = self.gx[self.wave_rows, self.wave_cols]
        y = self.gy[self.wave_rows, self.wave_cols]
        pitch = 2.0 * span / self.WAVE_BARS
        position = (x + span) / pitch
        index = np.clip(np.floor(position).astype(np.int32), 0, self.WAVE_BARS - 1)
        local = (position - index - 0.5) * pitch
        inside = (np.abs(x) <= span).astype(np.float32)
        self.wave_index = index
        self.wave_bar = np.clip((pitch * 0.36 - np.abs(local)) / self.pixel_u + 0.5, 0.0, 1.0) * inside
        self.wave_abs_y = np.abs(y)
        self.wave_hue = np.clip((x + span) / (2.0 * span), 0.0, 1.0)[..., None]
        centers = (np.arange(self.WAVE_BARS) + 0.5) * pitch - span
        self.wave_envelope = np.exp(-(centers / 0.24) ** 2).astype(np.float32)
        rng = np.random.default_rng(97)
        self.wave_speed = rng.uniform(5.0, 13.0, self.WAVE_BARS).astype(np.float32)
        self.wave_offset = rng.uniform(0.0, 6.3, self.WAVE_BARS).astype(np.float32)

    def _hex_fields(self):
        if self._hex is not None:
            return self._hex
        sphere_radius = 0.84
        x, y = self.gx / sphere_radius, self.gy / sphere_radius
        rho2 = x * x + y * y
        z = np.sqrt(np.maximum(1.0 - rho2, 0.0))
        warp = 1.0 / (0.45 + 0.55 * z)
        px, py = x * warp * 4.2, y * warp * 4.2
        sqrt3 = math.sqrt(3.0)
        ax = np.mod(px, 1.0) - 0.5
        ay = np.mod(py, sqrt3) - sqrt3 / 2.0
        bx = np.mod(px - 0.5, 1.0) - 0.5
        by = np.mod(py - sqrt3 / 2.0, sqrt3) - sqrt3 / 2.0
        use_a = ax * ax + ay * ay < bx * bx + by * by
        gx = np.where(use_a, ax, bx)
        gy = np.where(use_a, ay, by)
        hex_distance = np.maximum(np.abs(gx) * 0.5 + np.abs(gy) * sqrt3 / 2.0, np.abs(gx))
        edge = 0.5 - hex_distance
        edge_px = edge / warp / 4.2 * sphere_radius / self.pixel_u
        lines = np.exp(-(np.maximum(edge_px, 0.0) / 0.9) ** 2)
        cell_x, cell_y = px - gx, py - gy
        cell_phase = np.mod(np.sin(cell_x * 12.9898 + cell_y * 78.233) * 43758.5453, 1.0) * 6.283
        rho = np.sqrt(rho2)
        inside = smoothstep(1.0 + 1.5 * self.pixel_u, 1.0 - 1.5 * self.pixel_u, rho)
        rim = np.exp(-((rho - 1.0) * sphere_radius / 0.035) ** 2)
        fresnel = np.clip(rho, 0.0, 1.0) ** 4
        light = np.clip(0.5 + 0.5 * (-x - y) / 1.414, 0.0, 1.0)
        self._hex = {"lines": (lines * inside * (0.35 + 0.65 * fresnel + 0.25)).astype(np.float32),
                     "phase": cell_phase.astype(np.float32),
                     "inside": inside.astype(np.float32),
                     "rim": rim.astype(np.float32),
                     "fresnel": (fresnel * inside).astype(np.float32),
                     "light": light.astype(np.float32)}
        return self._hex

    def _sample(self, name, phase):
        shift = int(round(phase / (2.0 * math.pi) * TEXTURE_ANGULAR))
        index = self.radial_offset + ((self.angle_index - shift) & (TEXTURE_ANGULAR - 1))
        return self.textures[name].take(index)

    def _sample_ring(self, name, phase):
        shift = int(round(phase / (2.0 * math.pi) * TEXTURE_ANGULAR))
        return self.textures[name].take((self.angle_index - shift) & (TEXTURE_ANGULAR - 1))

    def _advance(self, state, level, t):
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), 0.2)
        self._last_t = t
        speeds = {
            "idle": (0.16, 0.10, 0.05, 0.0),
            "recording": (0.45 + 1.3 * level, 0.30 + 0.9 * level, 0.12, 0.0),
            "processing": (0.25, 0.18, 0.10, 1.25),
            "muted": (0.10, 0.07, 0.04, 0.0),
            "success": (0.55, 0.40, 0.10, 0.0),
            "error": (0.75, 0.55, 0.20, 0.0),
        }[state]
        self._phase["forward"] += speeds[0] * dt
        self._phase["reverse"] += speeds[1] * dt
        self._phase["haze"] += speeds[2] * dt
        self._phase["spiral"] += speeds[3] * dt
        self._phase["lumps"] += (0.25 + 0.5 * level) * dt
        self._phase["hue"] += (0.35 if state == "recording" else 0.12) * dt

    def _envelope(self, scale, ragged):
        lumps = (self._sample_ring("lumps_a", self._phase["lumps"]) * 0.6
                 + self._sample_ring("lumps_b", -self._phase["lumps"] * 0.7) * 0.4)
        edge = scale * (1.0 + ragged * lumps)
        body = np.clip((edge + 0.10 - self.u) / 0.18, 0.0, 1.0)
        body = body * body * (3.0 - 2.0 * body)
        wisp = np.exp(-(np.maximum(self.u - edge, 0.0) / 0.09) ** 2) * (1.0 - body)
        return body, wisp

    def _gradient(self, palette, hue_phase):
        mix = 0.5 + 0.5 * (self.cos_th * math.cos(hue_phase) + self.sin_th * math.sin(hue_phase))
        return palette["warm"] + (palette["cool"] - palette["warm"]) * mix[..., None]

    def _vortex(self, palette, gain, haze_amount, core_radius, core_alpha, texture_names=("orbit_forward", "orbit_reverse"),
                scale=1.0, ragged=0.07):
        body, wisp = self._envelope(scale, ragged)
        filaments = (self._sample(texture_names[0], self._phase["forward"])
                     + 0.8 * self._sample(texture_names[1], -self._phase["reverse"]))
        filaments *= body * gain
        haze = self._sample("haze", self._phase["haze"]) * body * haze_amount
        wisp = wisp * (0.25 + 0.35 * self._sample("haze", -self._phase["haze"])) * gain

        color = self._gradient(palette, self._phase["hue"])
        hot = np.clip(filaments - 0.9, 0.0, 1.0)[..., None]
        emission = color * (filaments[..., None] + wisp[..., None]) + palette["haze"] * haze[..., None]
        emission += (self.white - color) * hot * 0.5
        alpha = np.clip(filaments + haze + wisp, 0.0, 1.0)

        core = smoothstep(core_radius, core_radius - 0.16, self.u) * core_alpha
        emission *= (1.0 - 0.75 * core)[..., None]
        emission += palette["core"] * core[..., None]
        alpha = alpha + core * (1.0 - alpha)
        return emission, alpha

    def _overlay(self, emission, alpha, mask, color, strength=1.0):
        weight = mask * strength
        emission = emission * (1.0 - weight)[..., None] + color * weight[..., None]
        return emission, alpha + weight * (1.0 - alpha)

    def _waveform(self, emission, alpha, palette, level, t):
        jitter = 0.35 + 0.65 * np.abs(np.sin(self.wave_speed * t + self.wave_offset))
        heights = 0.018 + 0.30 * min(level * 1.3, 1.0) * self.wave_envelope * jitter
        bar_height = heights[self.wave_index]
        mask = self.wave_bar * np.clip((bar_height - self.wave_abs_y) / self.pixel_u + 0.5, 0.0, 1.0)
        color = palette["glyph"] + (palette["cool"] - palette["glyph"]) * self.wave_hue
        region = (self.wave_rows, self.wave_cols)
        emission[region] = emission[region] * (1.0 - mask)[..., None] + color * mask[..., None]
        alpha[region] = alpha[region] + mask * (1.0 - alpha[region])
        return emission, alpha

    def _shade_state(self, state, level, t, age):
        palette = self.palettes[state]
        if state == "idle":
            return self._vortex(palette, 0.40, 0.42, 0.50, 0.82, ragged=0.05)

        if state == "recording":
            emission, alpha = self._vortex(palette, 0.70 + 0.50 * level, 0.55, 0.52, 0.90,
                                           scale=1.0 + 0.05 * level, ragged=0.06 + 0.05 * level)
            return self._waveform(emission, alpha, palette, level, t)

        if state == "processing":
            emission, alpha = self._vortex(palette, 0.45, 0.55, 0.20, 0.55,
                                           texture_names=("orbit_forward", "orbit_forward"), ragged=0.06)
            spiral = self._sample("spiral", self._phase["spiral"]) * 1.1
            spiral_color = self._gradient(palette, -self._phase["spiral"] * 0.5)
            emission += spiral_color * spiral[..., None]
            alpha = np.clip(alpha + spiral, 0.0, 1.0)
            cycle = (t % 1.6) / 1.6
            ring_radius = 0.025 + 0.11 * cycle
            pulse = (np.exp(-((self.u - ring_radius) / 0.022) ** 2) * (0.4 + 0.6 * cycle)
                     + np.exp(-(self.u / (0.045 + 0.02 * (1.0 - cycle))) ** 2) * (1.0 - 0.6 * cycle))
            return self._overlay(emission, alpha, np.clip(pulse, 0.0, 1.0), palette["glyph"])

        if state == "muted":
            hex_fields = self._hex_fields()
            emission, alpha = self._vortex(palette, 0.30, 0.18, 0.0, 0.0, scale=1.08, ragged=0.06)
            inside = hex_fields["inside"]
            shimmer = 0.55 + 0.45 * np.sin(hex_fields["phase"] + t * 1.4)
            sweep = 0.5 + 0.5 * (self.cos_th * math.cos(t * 0.7) + self.sin_th * math.sin(t * 0.7))
            glass_alpha = inside * (0.62 + 0.25 * hex_fields["fresnel"])
            glass = palette["core"] + palette["haze"] * (0.15 + 0.35 * hex_fields["light"] + 1.1 * hex_fields["fresnel"])[..., None]
            emission = emission * (1.0 - glass_alpha)[..., None] + glass * glass_alpha[..., None]
            alpha = alpha + glass_alpha * (1.0 - alpha)
            lines = hex_fields["lines"] * shimmer
            rim = np.clip(hex_fields["rim"] * (0.85 + 0.5 * sweep) + hex_fields["fresnel"] * 0.55, 0.0, 1.0)
            emission, alpha = self._overlay(emission, alpha, np.clip(lines * 1.1, 0.0, 1.0), palette["warm"] * 0.85)
            emission, alpha = self._overlay(emission, alpha, rim, palette["glyph"] + 30.0)
            emission, alpha = self._overlay(emission, alpha, self.glyphs["ring"], palette["glyph"], 0.75)
            return self._overlay(emission, alpha, self.glyphs["mic"], palette["glyph"], 0.95)

        if state == "success":
            onset = min(age / 0.25, 1.0)
            boost = 1.0 + 0.6 * max(0.0, 1.0 - age / 0.6)
            emission, alpha = self._vortex(palette, 0.75 * boost, 0.50, 0.52, 0.92,
                                           texture_names=("rings", "orbit_reverse"), ragged=0.05)
            emission, alpha = self._overlay(emission, alpha, self.glyphs["ring"], palette["glyph"], 0.9 * onset)
            return self._overlay(emission, alpha, self.glyphs["check"], palette["glyph"] + 40.0, onset)

        flicker = 0.85 + 0.15 * math.sin(t * 9.0) * math.sin(t * 3.7)
        boost = 1.0 + 0.8 * max(0.0, 1.0 - age / 0.5)
        emission, alpha = self._vortex(palette, 0.85 * flicker * boost, 0.50, 0.52, 0.92, ragged=0.10)
        emission, alpha = self._overlay(emission, alpha, self.glyphs["ring"], palette["glyph"], 0.9)
        return self._overlay(emission, alpha, self.glyphs["exclamation"], palette["glyph"] + 40.0)

    def shade(self, state: str, level: float, t: float):
        if state not in PALETTES:
            state = "idle"
        if state != self._state:
            self._previous_state = self._state
            self._state = state
            self._state_since = t
        self._advance(state, level, t)

        age = t - self._state_since
        emission, alpha = self._shade_state(state, level, t, age)
        if self._previous_state and age < FADE_SECONDS:
            mix = float(smoothstep(0.0, FADE_SECONDS, age))
            old_emission, old_alpha = self._shade_state(self._previous_state, level, t, age + 10.0)
            emission = old_emission + (emission - old_emission) * mix
            alpha = old_alpha + (alpha - old_alpha) * mix

        rgb = emission / np.maximum(alpha, 1e-3)[..., None]
        return rgb, alpha
