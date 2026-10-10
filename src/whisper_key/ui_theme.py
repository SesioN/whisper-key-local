from dataclasses import dataclass

from .platform import theme as platform_theme


@dataclass
class Theme:
    dark: bool
    bg: str
    hover: str
    surface: str
    surface_hover: str
    border: str
    text: str
    secondary: str
    disabled: str
    accent: str
    on_accent: str
    danger: str
    on_danger: str


def hex_color(rgb) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


def rgb_color(color: str):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def mix_colors(color_a: str, color_b: str, amount: float) -> str:
    a, b = rgb_color(color_a), rgb_color(color_b)
    return hex_color([a[i] + (b[i] - a[i]) * amount for i in range(3)])


def load_theme() -> Theme:
    apps_light, accent_rgb = platform_theme.read_system_appearance()
    accent = hex_color(accent_rgb)
    if apps_light:
        return Theme(dark=False, bg="#f9f9f9", hover="#ebebeb", surface="#ededed", surface_hover="#e2e2e2",
                     border="#d9d9d9", text="#1b1b1b", secondary="#5f5f5f", disabled="#a3a3a3",
                     accent=mix_colors(accent, "#000000", 0.1), on_accent="#ffffff", danger="#c42b1c", on_danger="#ffffff")
    return Theme(dark=True, bg="#2c2c2c", hover="#3a3a3a", surface="#383838", surface_hover="#454545",
                 border="#454545", text="#ffffff", secondary="#a8a8a8", disabled="#6b6b6b",
                 accent=mix_colors(accent, "#ffffff", 0.45), on_accent="#000000", danger="#ff99a4", on_danger="#000000")
