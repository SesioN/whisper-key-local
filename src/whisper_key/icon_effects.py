from PIL import Image, ImageDraw, ImageEnhance

MUTED_ICON_SIZE = 128
MUTED_BRIGHTNESS = 0.75
MUTED_SLASH_COLOR = (220, 38, 38, 255)
MUTED_SLASH_OUTLINE_COLOR = (255, 255, 255, 255)
MUTED_SLASH_INSET_RATIO = 0.14
MUTED_SLASH_WIDTH_RATIO = 0.11
MUTED_OUTLINE_WIDTH_RATIO = 0.06


def create_muted_icon(idle_icon: Image.Image) -> Image.Image:
    size = MUTED_ICON_SIZE
    base = idle_icon.convert("RGBA").resize((size, size), Image.LANCZOS)
    alpha = base.getchannel("A")
    grey = ImageEnhance.Brightness(base.convert("L")).enhance(MUTED_BRIGHTNESS)
    muted = Image.merge("RGBA", (grey, grey, grey, alpha))

    inset = size * MUTED_SLASH_INSET_RATIO
    start = (inset, inset)
    end = (size - inset, size - inset)
    slash_width = round(size * MUTED_SLASH_WIDTH_RATIO)
    outline_width = slash_width + 2 * round(size * MUTED_OUTLINE_WIDTH_RATIO)

    draw = ImageDraw.Draw(muted)
    _draw_round_line(draw, start, end, outline_width, MUTED_SLASH_OUTLINE_COLOR)
    _draw_round_line(draw, start, end, slash_width, MUTED_SLASH_COLOR)
    return muted


def _draw_round_line(draw: ImageDraw.ImageDraw, start: tuple, end: tuple, width: int, color: tuple):
    draw.line([start, end], fill=color, width=width)
    radius = width / 2
    for x, y in (start, end):
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
