from functools import lru_cache

import numpy as np
from PIL import Image

from . import get_skin

PREVIEW_SUPERSAMPLE = 2
PREVIEW_FILL = 0.82
PREVIEW_TIME = 2.0
PALETTE_SAMPLE_SIZE = 32


def render_skin_preview(skin_name: str, size: int, color_matrix=None) -> Image.Image:
    skin_class = get_skin(skin_name)
    big_size = size * PREVIEW_SUPERSAMPLE
    canvas = big_size * 2
    renderer = skin_class(canvas, big_size * 0.5 / skin_class.VISUAL_SCALE * PREVIEW_FILL)
    pixels = renderer.render("idle", 0.0, PREVIEW_TIME, 1.0, color_matrix=color_matrix)
    alpha = pixels[..., 3].astype(np.float32)
    rgb = pixels[..., 2::-1].astype(np.float32)
    covered = alpha > 0
    rgb[covered] = rgb[covered] * 255.0 / alpha[covered, None]
    image = Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha]).astype(np.uint8), "RGBA")
    offset = (canvas - big_size) // 2
    image = image.crop((offset, offset, offset + big_size, offset + big_size))
    return image.resize((size, size), Image.LANCZOS)


@lru_cache(maxsize=None)
def skin_palette_color(skin_name: str) -> tuple:
    pixels = np.asarray(render_skin_preview(skin_name, PALETTE_SAMPLE_SIZE), dtype=np.float32)
    weights = pixels[..., 3] / 255.0
    total = max(float(weights.sum()), 1e-6)
    return tuple(float((pixels[..., channel] * weights).sum() / total) for channel in range(3))
