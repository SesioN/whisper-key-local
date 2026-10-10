import tkinter as tk
from tkinter import font as tkfont, ttk

from .platform import window_style
from .ui_theme import mix_colors, rgb_color, load_theme

TEXT_FAMILIES = ("Segoe UI Variable Text", "Segoe UI")
BODY_FONT_SIZE = 10
TITLE_FONT_SIZE = 12
CONTROL_PADDING = (12, 6)

HINT_STYLE = "Hint.TLabel"
ERROR_STYLE = "Error.TLabel"
TITLE_STYLE = "Title.TLabel"
ACCENT_BUTTON_STYLE = "Accent.TButton"


def text_family(root) -> str:
    available = set(tkfont.families(root))
    fallback = tkfont.nametofont("TkDefaultFont", root=root).actual("family")
    return next((family for family in TEXT_FAMILIES if family in available), fallback)


def apply_theme(root: tk.Misc):
    theme = load_theme()
    family = text_family(root)
    for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont"):
        tkfont.nametofont(font_name, root=root).configure(family=family, size=BODY_FONT_SIZE)
    title_font = (family, TITLE_FONT_SIZE, "bold")

    root.configure(bg=theme.bg)
    root.option_add("*Font", "TkDefaultFont")

    pressed = mix_colors(theme.surface_hover, theme.border, 0.5)
    accent_hover = mix_colors(theme.accent, theme.text, 0.12)
    accent_pressed = mix_colors(theme.accent, theme.text, 0.24)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=theme.bg, foreground=theme.text, bordercolor=theme.border,
                    lightcolor=theme.bg, darkcolor=theme.bg, troughcolor=theme.surface,
                    focuscolor=theme.accent, selectbackground=theme.accent, selectforeground=theme.on_accent,
                    insertcolor=theme.text, relief="flat", borderwidth=0)
    style.map(".", foreground=[("disabled", theme.disabled)])

    style.configure("TFrame", background=theme.bg)
    style.configure("TLabel", background=theme.bg, foreground=theme.text)
    style.configure(HINT_STYLE, foreground=theme.secondary)
    style.configure(ERROR_STYLE, foreground=theme.danger)
    style.configure(TITLE_STYLE, font=title_font)
    style.configure("TLabelframe", background=theme.bg, bordercolor=theme.border, lightcolor=theme.border,
                    darkcolor=theme.border, borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background=theme.bg, foreground=theme.secondary)

    style.configure("TButton", background=theme.surface, foreground=theme.text, bordercolor=theme.border,
                    lightcolor=theme.surface, darkcolor=theme.surface, borderwidth=1, padding=CONTROL_PADDING,
                    anchor="center")
    style.map("TButton",
              background=[("disabled", theme.bg), ("pressed", pressed), ("active", theme.surface_hover)],
              bordercolor=[("disabled", theme.border), ("focus", theme.accent)],
              lightcolor=[("pressed", pressed), ("active", theme.surface_hover), ("disabled", theme.bg)],
              darkcolor=[("pressed", pressed), ("active", theme.surface_hover), ("disabled", theme.bg)])
    style.configure(ACCENT_BUTTON_STYLE, background=theme.accent, foreground=theme.on_accent,
                    bordercolor=theme.accent, lightcolor=theme.accent, darkcolor=theme.accent)
    style.map(ACCENT_BUTTON_STYLE,
              background=[("disabled", theme.surface), ("pressed", accent_pressed), ("active", accent_hover)],
              foreground=[("disabled", theme.disabled)],
              bordercolor=[("disabled", theme.border), ("pressed", accent_pressed), ("active", accent_hover)],
              lightcolor=[("disabled", theme.surface), ("pressed", accent_pressed), ("active", accent_hover)],
              darkcolor=[("disabled", theme.surface), ("pressed", accent_pressed), ("active", accent_hover)])

    style.configure("TEntry", fieldbackground=theme.surface, foreground=theme.text, bordercolor=theme.border,
                    lightcolor=theme.border, darkcolor=theme.border, borderwidth=1, padding=5)
    style.map("TEntry", bordercolor=[("focus", theme.accent)], lightcolor=[("focus", theme.accent)],
              darkcolor=[("focus", theme.accent)], fieldbackground=[("disabled", theme.bg)])

    for indicator_style in ("TCheckbutton", "TRadiobutton"):
        style.configure(indicator_style, background=theme.bg, foreground=theme.text,
                        indicatorbackground=theme.surface, indicatorforeground=theme.on_accent,
                        upperbordercolor=theme.border, lowerbordercolor=theme.border, padding=4)
        style.map(indicator_style,
                  background=[("active", theme.bg)],
                  indicatorbackground=[("selected", theme.accent), ("disabled", theme.bg), ("active", theme.surface_hover)])

    style.configure("TScrollbar", background=theme.surface_hover, troughcolor=theme.bg, bordercolor=theme.bg,
                    lightcolor=theme.surface_hover, darkcolor=theme.surface_hover, arrowcolor=theme.secondary,
                    gripcount=0, borderwidth=0)
    style.map("TScrollbar", background=[("active", theme.secondary)])

    style.configure("Horizontal.TScale", background=theme.bg, troughcolor=theme.surface, bordercolor=theme.bg,
                    lightcolor=theme.accent, darkcolor=theme.accent)
    style.map("Horizontal.TScale", background=[("active", theme.accent)])

    style.configure("Treeview", background=theme.surface, fieldbackground=theme.surface, foreground=theme.text,
                    bordercolor=theme.border, lightcolor=theme.surface, darkcolor=theme.surface, borderwidth=1)
    style.map("Treeview", background=[("selected", theme.accent)], foreground=[("selected", theme.on_accent)])
    style.configure("Treeview.Heading", background=theme.bg, foreground=theme.secondary, bordercolor=theme.border,
                    lightcolor=theme.bg, darkcolor=theme.bg, relief="flat", padding=(6, 6))
    style.map("Treeview.Heading", background=[("active", theme.hover)])

    return theme


def style_text_widget(widget: tk.Text, theme):
    widget.configure(bg=theme.surface, fg=theme.text, insertbackground=theme.text,
                     selectbackground=theme.accent, selectforeground=theme.on_accent,
                     relief="flat", borderwidth=0, highlightthickness=1,
                     highlightbackground=theme.border, highlightcolor=theme.accent,
                     padx=6, pady=6, font="TkDefaultFont")


def apply_window_chrome(root: tk.Misc, theme):
    root.update_idletasks()
    window_style.apply_window_chrome(root.winfo_id(), theme.dark, rgb_color(theme.border))
