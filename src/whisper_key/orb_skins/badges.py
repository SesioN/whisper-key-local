import math

import numpy as np

from .glyphs import lock, mic, mic_slash

BADGE_ANGLES = {"lock": math.radians(128.0), "mute": math.radians(52.0)}
BADGE_DISTANCE = 0.95
BADGE_RADIUS = 0.27
MIN_BADGE_RADIUS_PX = 9.0
GLOW_EXTENT = 0.45
GLYPH_SCALE = 0.40
MUTED_RING = (255, 92, 112)
MUTED_GLYPH = (255, 182, 190)


class BadgeRenderer:

    def __init__(self, canvas: int, orb_radius: float, dpi: int = 96):
        self.canvas = canvas
        self.radius = max(BADGE_RADIUS * orb_radius, MIN_BADGE_RADIUS_PX * dpi / 96.0)
        center = (canvas - 1) / 2.0
        extent = self.radius * (1.0 + GLOW_EXTENT)
        self.slots = {}
        for kind, angle in BADGE_ANGLES.items():
            cx = center + math.cos(angle) * BADGE_DISTANCE * orb_radius
            cy = center + math.sin(angle) * BADGE_DISTANCE * orb_radius
            cy = min(cy, canvas - 1 - extent)
            x0, x1 = max(0, int(math.floor(cx - extent))), min(canvas, int(math.ceil(cx + extent)) + 1)
            y0, y1 = max(0, int(math.floor(cy - extent))), min(canvas, int(math.ceil(cy + extent)) + 1)
            ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            local_x = (xs - cx) / self.radius
            local_y = (ys - cy) / self.radius
            self.slots[kind] = {"center": (cx, cy), "region": (slice(y0, y1), slice(x0, x1)),
                                "x": local_x, "y": local_y,
                                "distance": np.sqrt(local_x * local_x + local_y * local_y)}
        self._sprites = {}

    def hit(self, x: float, y: float, kinds):
        for kind in kinds:
            cx, cy = self.slots[kind]["center"]
            if (x - cx) ** 2 + (y - cy) ** 2 <= (self.radius * 1.1) ** 2:
                return kind
        return None

    def _sprite(self, kind, active, colors):
        key = (kind, active, colors)
        sprite = self._sprites.get(key)
        if sprite is not None:
            return sprite
        if len(self._sprites) > 32:
            self._sprites.clear()

        slot = self.slots[kind]
        distance = slot["distance"]
        pixel = 1.0 / self.radius
        fill_color, ring_color, glyph_color = (np.array(c, dtype=np.float32) for c in colors)
        if kind == "mute" and active:
            ring_color = np.array(MUTED_RING, dtype=np.float32)
            glyph_color = np.array(MUTED_GLYPH, dtype=np.float32)
        if kind == "lock" and active:
            ring_color = ring_color + (255.0 - ring_color) * 0.35
        ring_strength = 1.0 if active else 0.7

        disc = np.clip((1.0 - distance) / pixel + 0.5, 0.0, 1.0)
        shade = (1.0 - 0.35 * np.clip(distance, 0.0, 1.0) ** 2) * (1.0 + 0.25 * np.clip(-slot["y"], 0.0, 1.0))
        ring = np.clip((0.07 - np.abs(distance - 0.90)) / pixel + 0.5, 0.0, 1.0) * ring_strength
        glow = np.exp(-(np.maximum(distance - 0.95, 0.0) / 0.22) ** 2) * (1.0 - disc) * 0.55 * ring_strength

        gx, gy = slot["x"] * GLYPH_SCALE / 0.90, slot["y"] * GLYPH_SCALE / 0.90
        glyph_pixel = pixel * GLYPH_SCALE / 0.90
        if kind == "lock":
            glyph = lock(gx, gy, glyph_pixel, active, 0.02)
        else:
            glyph = (mic_slash if active else mic)(gx, gy, glyph_pixel, 0.02)

        alpha = disc * 0.86
        rgb = fill_color * shade[..., None]
        rgb = rgb + (ring_color - rgb) * ring[..., None]
        alpha = np.maximum(alpha, ring)
        rgb = rgb + (glyph_color - rgb) * glyph[..., None]
        alpha = np.maximum(alpha, glyph * disc)
        premul = rgb * alpha[..., None] + ring_color * glow[..., None]
        alpha = np.clip(alpha + glow, 0.0, 1.0)

        sprite = np.empty(alpha.shape + (4,), dtype=np.float32)
        sprite[..., :3] = np.minimum(premul[..., ::-1], alpha[..., None] * 255.0)
        sprite[..., 3] = alpha * 255.0
        hit_mask = (disc > 0.5).astype(np.float32)
        self._sprites[key] = (sprite, hit_mask)
        return self._sprites[key]

    def composite(self, pixels, visibility: float, badges):
        for kind, active, colors in badges:
            sprite, hit_mask = self._sprite(kind, active, colors)
            region = self.slots[kind]["region"]
            target = pixels[region]
            if visibility > 0.0:
                layer = sprite * np.float32(visibility)
                inverse = 1.0 - layer[..., 3:4] / 255.0
                blended = layer + target.astype(np.float32) * inverse
            else:
                blended = target.astype(np.float32)
            blended[..., 3] = np.maximum(blended[..., 3], hit_mask)
            np.copyto(target, blended + 0.5, casting="unsafe")
