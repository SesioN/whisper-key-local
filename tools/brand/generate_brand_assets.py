import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_ASSETS = REPO_ROOT / "src" / "whisper_key" / "platform" / "windows" / "assets"
FONTS_DIR = Path("C:/Windows/Fonts")
WORDMARK_FONT = FONTS_DIR / "seguisb.ttf"

SPLASH_WIDTH = 440
SPLASH_HEIGHT = 260
SPLASH_FRAMES = 40
SPLASH_JPEG_QUALITY = 88
CARD_BOX = (20, 170, 420, 242)
CARD_RADIUS = 10
CARD_FILL = (8, 11, 26)
CARD_OPACITY = 0.82
CARD_BORDER = (120, 150, 230)
CARD_BORDER_OPACITY = 0.28
ACTIVITY_BAR_Y = 232
ACTIVITY_BAR_HEIGHT = 3
ACTIVITY_BAR_INSET = 16
WORDMARK_TEXT = "Whisper Key"
WORDMARK_SIZE = 44
LOGO_SIZE = 64
LOGO_GAP = 12
WORDMARK_CENTER_Y = 92

NOISE_SIZE = 512
STAR_COUNT = 70

ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
TRAY_ICON_SIZE = 64
ICON_SUPERSAMPLE = 8

DEEP_SPACE = np.array((4, 6, 18), dtype=np.float32)
NEBULA_BLUE = np.array((70, 190, 255), dtype=np.float32)
NEBULA_VIOLET = np.array((150, 110, 255), dtype=np.float32)
NEBULA_HAZE = np.array((40, 80, 180), dtype=np.float32)
NEBULA_PINK = np.array((215, 120, 255), dtype=np.float32)

ICON_PALETTES = {
    "idle": {"ring_start": (70, 190, 255), "ring_end": (150, 110, 255), "core": (6, 10, 30),
             "core_edge": (30, 50, 120), "glow": (60, 120, 255), "glyph": (235, 245, 255)},
    "recording": {"ring_start": (90, 225, 255), "ring_end": (120, 160, 255), "core": (4, 14, 34),
                  "core_edge": (20, 80, 150), "glow": (70, 200, 255), "glyph": (255, 255, 255)},
    "processing": {"ring_start": (160, 100, 255), "ring_end": (225, 160, 255), "core": (12, 4, 30),
                   "core_edge": (70, 36, 150), "glow": (150, 90, 255), "glyph": (245, 230, 255)},
}
RECORDING_DOT = (255, 64, 72)


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def periodic_noise(seed, size, falloff):
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((size, size))
    frequencies = np.fft.fftfreq(size)
    radius = np.sqrt(frequencies[:, None] ** 2 + frequencies[None, :] ** 2)
    radius[0, 0] = 1.0
    spectrum = np.fft.fft2(white) / radius ** falloff
    spectrum[0, 0] = 0.0
    field = np.real(np.fft.ifft2(spectrum))
    field -= field.min()
    field /= field.max()
    return field.astype(np.float32)


def sample_wrapped(field, offset_x, offset_y, scale):
    size = field.shape[0]
    ys, xs = np.mgrid[0:SPLASH_HEIGHT, 0:SPLASH_WIDTH].astype(np.float32)
    sample_x = (xs * scale + offset_x) % size
    sample_y = (ys * scale + offset_y) % size
    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    fx = sample_x - x0
    fy = sample_y - y0
    x0 %= size
    y0 %= size
    x1 = (x0 + 1) % size
    y1 = (y0 + 1) % size
    top = field[y0, x0] * (1 - fx) + field[y0, x1] * fx
    bottom = field[y1, x0] * (1 - fx) + field[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


class SplashRenderer:
    def __init__(self):
        self.cloud_large = periodic_noise(3, NOISE_SIZE, 1.35)
        self.cloud_fine = periodic_noise(7, NOISE_SIZE, 1.3)
        self.cloud_hue = periodic_noise(11, NOISE_SIZE, 1.6)
        ys, xs = np.mgrid[0:SPLASH_HEIGHT, 0:SPLASH_WIDTH].astype(np.float32)
        center_x, center_y = SPLASH_WIDTH * 0.5, SPLASH_HEIGHT * 0.38
        distance = np.sqrt(((xs - center_x) / SPLASH_WIDTH) ** 2 + ((ys - center_y) / SPLASH_HEIGHT) ** 2)
        self.vignette = (1.0 - smoothstep(0.15, 0.75, distance) * 0.75)[..., None]
        rng = np.random.default_rng(19)
        self.stars = [(rng.uniform(0, SPLASH_WIDTH), rng.uniform(0, SPLASH_HEIGHT), rng.uniform(0.3, 1.0),
                       rng.integers(1, 4), rng.uniform(0, 2 * math.pi)) for _ in range(STAR_COUNT)]
        self.logo = np.asarray(render_icon("idle", LOGO_SIZE), dtype=np.float32)
        self.wordmark_mask, self.wordmark_box = self._wordmark_mask()
        self.wordmark_glow = np.asarray(
            Image.fromarray((self.wordmark_mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(9)),
            dtype=np.float32) / 255.0
        self.card_mask, self.card_border = self._card_masks()

    def _wordmark_mask(self):
        font = ImageFont.truetype(str(WORDMARK_FONT), WORDMARK_SIZE)
        image = Image.new("L", (SPLASH_WIDTH, SPLASH_HEIGHT), 0)
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = draw.textbbox((0, 0), WORDMARK_TEXT, font=font)
        x = (SPLASH_WIDTH - (right - left) + LOGO_SIZE + LOGO_GAP) / 2 - left
        self.logo_left = int(x + left - LOGO_SIZE - LOGO_GAP)
        y = WORDMARK_CENTER_Y - (bottom - top) / 2 - top
        draw.text((x, y), WORDMARK_TEXT, fill=255, font=font)
        box = (x + left, y + top, x + right, y + bottom)
        return np.asarray(image, dtype=np.float32) / 255.0, box

    def _card_masks(self):
        scale = 4
        size = (SPLASH_WIDTH * scale, SPLASH_HEIGHT * scale)
        box = tuple(value * scale for value in CARD_BOX)
        fill = Image.new("L", size, 0)
        ImageDraw.Draw(fill).rounded_rectangle(box, radius=CARD_RADIUS * scale, fill=255)
        border = Image.new("L", size, 0)
        ImageDraw.Draw(border).rounded_rectangle(box, radius=CARD_RADIUS * scale, outline=255, width=scale)
        downscale = (SPLASH_WIDTH, SPLASH_HEIGHT)
        return (np.asarray(fill.resize(downscale, Image.LANCZOS), dtype=np.float32) / 255.0,
                np.asarray(border.resize(downscale, Image.LANCZOS), dtype=np.float32) / 255.0)

    def frame(self, index):
        phase = index / SPLASH_FRAMES
        angle = 2 * math.pi * phase
        large = sample_wrapped(self.cloud_large, 22 * math.cos(angle), 22 * math.sin(angle), 0.55)
        fine = sample_wrapped(self.cloud_fine, -30 * math.cos(angle) + 200, 30 * math.sin(angle), 0.9)
        hue = sample_wrapped(self.cloud_hue, 15 * math.sin(angle), 15 * math.cos(angle) + 100, 0.4)

        density = smoothstep(0.38, 0.9, large * 0.8 + fine * 0.3)[..., None]
        wisps = smoothstep(0.6, 0.95, fine * 0.6 + large * 0.4)[..., None]
        hue = smoothstep(0.25, 0.75, hue)[..., None]
        cloud_color = np.where(hue < 0.5,
                               NEBULA_BLUE + (NEBULA_VIOLET - NEBULA_BLUE) * (hue * 2),
                               NEBULA_VIOLET + (NEBULA_PINK - NEBULA_VIOLET) * (hue * 2 - 1))
        rgb = DEEP_SPACE + NEBULA_HAZE * density * 0.45 + cloud_color * density * 0.5 + cloud_color * wisps * 0.45
        rgb *= self.vignette

        image = rgb.copy()
        for star_x, star_y, brightness, period, star_phase in self.stars:
            twinkle = 0.55 + 0.45 * math.sin(2 * math.pi * period * phase + star_phase)
            ix, iy = int(star_x), int(star_y)
            image[iy, ix] = np.minimum(image[iy, ix] + 255 * brightness * twinkle, 255)

        glow_strength = 1.15 + 0.25 * math.sin(angle)
        image += (NEBULA_BLUE * 0.6 + NEBULA_VIOLET * 0.4) * self.wordmark_glow[..., None] * glow_strength
        logo_alpha = self.logo[..., 3:4] / 255.0
        logo_top = int(WORDMARK_CENTER_Y - LOGO_SIZE / 2)
        logo_region = image[logo_top:logo_top + LOGO_SIZE, self.logo_left:self.logo_left + LOGO_SIZE]
        logo_region[:] = logo_region * (1 - logo_alpha) + self.logo[..., :3] * logo_alpha

        left, top, right, bottom = self.wordmark_box
        ys, xs = np.mgrid[0:SPLASH_HEIGHT, 0:SPLASH_WIDTH].astype(np.float32)
        span = right - left
        sweep_center = left - span * 0.4 + phase * span * 1.8
        sweep = np.exp(-(((xs + (ys - top) * 0.35) - sweep_center) / 22.0) ** 2)
        vertical = np.clip((ys - top) / max(bottom - top, 1), 0, 1)[..., None]
        text_color = np.array((255, 255, 255), dtype=np.float32) * (1 - vertical * 0.25) \
            + np.array((190, 205, 255), dtype=np.float32) * vertical * 0.25
        text_color *= 0.84
        shine = sweep[..., None] * 140
        mask = self.wordmark_mask[..., None]
        image = image * (1 - mask) + np.minimum(text_color + shine, 255) * mask

        card = self.card_mask[..., None] * CARD_OPACITY
        image = image * (1 - card) + np.array(CARD_FILL, dtype=np.float32) * card
        border = self.card_border[..., None] * CARD_BORDER_OPACITY
        image = image * (1 - border) + np.array(CARD_BORDER, dtype=np.float32) * border

        bar_left = CARD_BOX[0] + ACTIVITY_BAR_INSET
        bar_right = CARD_BOX[2] - ACTIVITY_BAR_INSET
        bar_width = bar_right - bar_left
        track = image[ACTIVITY_BAR_Y:ACTIVITY_BAR_Y + ACTIVITY_BAR_HEIGHT, bar_left:bar_right]
        track[:] = track * 0.6 + np.array((40, 50, 90), dtype=np.float32) * 0.4
        bar_xs = np.arange(bar_width, dtype=np.float32)
        head = -bar_width * 0.35 + phase * bar_width * 1.35
        segment = smoothstep(0.0, 18.0, bar_xs - head) * smoothstep(0.0, 18.0, head + bar_width * 0.35 - bar_xs)
        gradient = bar_xs / bar_width
        bar_color = NEBULA_BLUE[None, :] * (1 - gradient[:, None]) + NEBULA_VIOLET[None, :] * gradient[:, None]
        track[:] = track * (1 - segment[None, :, None]) + bar_color[None] * segment[None, :, None]

        return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB")

    def strip(self):
        strip = Image.new("RGB", (SPLASH_WIDTH * SPLASH_FRAMES, SPLASH_HEIGHT))
        for index in range(SPLASH_FRAMES):
            strip.paste(self.frame(index), (index * SPLASH_WIDTH, 0))
        return strip


def angular_gradient(xs, ys, start, end):
    angle = (np.arctan2(ys, xs) + math.pi) / (2 * math.pi)
    blend = (0.5 - 0.5 * np.cos(angle * 2 * math.pi))[..., None]
    return np.array(start, dtype=np.float32) * (1 - blend) + np.array(end, dtype=np.float32) * blend


def waveform_mask(xs, ys, pixel, bar_count, span, heights):
    pitch = 2 * span / bar_count
    position = (xs + span) / pitch
    index = np.clip(np.floor(position).astype(np.int32), 0, bar_count - 1)
    local = (position - index - 0.5) * pitch
    half_width = pitch * 0.30
    height = np.array(heights, dtype=np.float32)[index]
    capsule_y = np.maximum(np.abs(ys) - (height - half_width), 0)
    mask = np.clip(0.5 - (np.sqrt(local ** 2 + capsule_y ** 2) - half_width) / pixel, 0, 1)
    return mask * (np.abs(xs) <= span + half_width)


def processing_mask(xs, ys, pixel):
    mask = np.zeros_like(xs)
    for dot_index in range(3):
        dot_x = (dot_index - 1) * 0.30
        radius = 0.10 + 0.02 * (1 - abs(dot_index - 1))
        mask = np.maximum(mask, np.clip(0.5 - (np.sqrt((xs - dot_x) ** 2 + ys ** 2) - radius) / pixel, 0, 1))
    return mask


def render_icon(state, size):
    palette = ICON_PALETTES[state]
    grid = size * ICON_SUPERSAMPLE
    coordinates = (np.arange(grid, dtype=np.float32) + 0.5) / grid * 2 - 1
    xs, ys = np.meshgrid(coordinates, coordinates)
    radius = np.sqrt(xs * xs + ys * ys)
    pixel = 2.0 / size

    small = size <= 24
    ring_outer = 0.96 if small else 0.90
    ring_inner = 0.66 if small else 0.64
    glow_extent = 0.0 if small else 0.08

    ring = np.clip((ring_outer - radius) / pixel + 0.5, 0, 1) * np.clip((radius - ring_inner) / pixel + 0.5, 0, 1)
    disc = np.clip((ring_inner + pixel * 0.5 - radius) / pixel + 0.5, 0, 1)
    glow = np.exp(-((radius - ring_outer) / max(glow_extent, 1e-3)) ** 2) * (radius > ring_outer) \
        if glow_extent else np.zeros_like(radius)

    ring_color = angular_gradient(xs, ys, palette["ring_start"], palette["ring_end"])
    ring_shade = (0.80 + 0.20 * np.clip(-ys, -1, 1))[..., None]
    ring_highlight = np.exp(-((radius - (ring_inner + ring_outer) / 2) / ((ring_outer - ring_inner) * 0.28)) ** 2)
    ring_rgb = np.minimum(ring_color * ring_shade + 60 * ring_highlight[..., None], 255)

    core_mix = smoothstep(0.0, ring_inner, radius)[..., None]
    core_rgb = np.array(palette["core"], dtype=np.float32) * (1 - core_mix) \
        + np.array(palette["core_edge"], dtype=np.float32) * core_mix

    if not small:
        rng = np.random.default_rng(5)
        streaks = np.zeros_like(radius)
        for _ in range(9):
            major = rng.uniform(ring_inner - 0.02, ring_outer + 0.02)
            minor = major * rng.uniform(0.7, 0.95)
            tilt = rng.uniform(0, math.pi)
            local = np.arctan2(ys, xs) - tilt
            orbit = major * minor / np.sqrt((minor * np.cos(local)) ** 2 + (major * np.sin(local)) ** 2)
            streaks += np.exp(-((radius - orbit) / (pixel * 0.9)) ** 2) * rng.uniform(0.4, 1.0)
        ring_rgb = np.minimum(ring_rgb + np.clip(streaks, 0, 1)[..., None] * 70 * (size >= 48), 255)

    glyph_scale = ring_inner * (1.0 if small else 0.95)
    glyph_xs, glyph_ys = xs / glyph_scale, ys / glyph_scale
    glyph_pixel = pixel / glyph_scale
    if state == "processing":
        glyph = processing_mask(glyph_xs, glyph_ys, glyph_pixel)
    else:
        bar_heights = (0.22, 0.48, 0.70, 0.48, 0.22) if size <= 32 else (0.16, 0.34, 0.56, 0.72, 0.56, 0.34, 0.16)
        glyph = waveform_mask(glyph_xs, glyph_ys, glyph_pixel, len(bar_heights), 0.58, bar_heights)
    glyph = glyph * disc

    rgb = core_rgb * disc[..., None] + ring_rgb * ring[..., None]
    rgb = rgb * (1 - glyph[..., None]) + np.array(palette["glyph"], dtype=np.float32) * glyph[..., None]
    alpha = np.clip(disc + ring, 0, 1)

    glow_rgb = np.array(palette["glow"], dtype=np.float32)
    glow_alpha = glow * 0.55 * (1 - alpha)
    total_alpha = alpha + glow_alpha
    rgb = np.where(total_alpha[..., None] > 0,
                   (rgb * alpha[..., None] + glow_rgb * glow_alpha[..., None]) / np.maximum(total_alpha, 1e-6)[..., None],
                   0)

    if state == "recording":
        dot_x, dot_y, dot_radius = 0.62, 0.62, 0.30 if small else 0.24
        dot_distance = np.sqrt((xs - dot_x) ** 2 + (ys - dot_y) ** 2)
        outline = np.clip((dot_radius + pixel * 1.2 - dot_distance) / pixel + 0.5, 0, 1)
        dot = np.clip((dot_radius - dot_distance) / pixel + 0.5, 0, 1)
        rgb = rgb * (1 - outline[..., None]) + np.array((10, 12, 24), dtype=np.float32) * outline[..., None]
        rgb = rgb * (1 - dot[..., None]) + np.array(RECORDING_DOT, dtype=np.float32) * dot[..., None]
        total_alpha = np.maximum(total_alpha, outline)

    image = np.dstack([np.clip(rgb, 0, 255), np.clip(total_alpha, 0, 1) * 255]).astype(np.float32)
    premultiplied = image.copy()
    premultiplied[..., :3] *= image[..., 3:4] / 255.0
    premultiplied = premultiplied.reshape(size, ICON_SUPERSAMPLE, size, ICON_SUPERSAMPLE, 4).mean(axis=(1, 3))
    alpha_channel = premultiplied[..., 3:4]
    color = np.where(alpha_channel > 0, premultiplied[..., :3] * 255.0 / np.maximum(alpha_channel, 1e-6), 0)
    return Image.fromarray(np.dstack([np.clip(color, 0, 255), alpha_channel]).astype(np.uint8), "RGBA")


def write_icons(output_dir):
    icon_images = [render_icon("idle", size) for size in ICON_SIZES]
    icon_images[-1].save(output_dir / "whisperkey-icon.ico", sizes=[(size, size) for size in ICON_SIZES],
                         append_images=icon_images[:-1])
    for state in ICON_PALETTES:
        render_icon(state, TRAY_ICON_SIZE).save(output_dir / f"tray_{state}.png")


def write_splash(output_dir):
    SplashRenderer().strip().save(output_dir / "splash_frames.jpg", quality=SPLASH_JPEG_QUALITY,
                                  optimize=True, progressive=False, subsampling=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=WINDOWS_ASSETS)
    parser.add_argument("--only", choices=("icons", "splash"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.only != "splash":
        write_icons(args.output)
    if args.only != "icons":
        write_splash(args.output)


if __name__ == "__main__":
    main()
