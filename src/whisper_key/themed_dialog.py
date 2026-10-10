import tkinter as tk
from tkinter import ttk

from .themed_widgets import ACCENT_BUTTON_STYLE, apply_theme, apply_window_chrome

DIALOG_WIDTH = 440
TEXT_WRAP_LENGTH = DIALOG_WIDTH - 48
BUTTON_MIN_WIDTH = 9


def show_dialog(title: str, text: str, buttons: list, dismiss_result, parent: tk.Misc = None):
    window = tk.Toplevel(parent) if parent else tk.Tk()
    result = [dismiss_result]
    try:
        window.withdraw()
        window.title(title)
        window.resizable(False, False)
        theme = apply_theme(window)

        frame = ttk.Frame(window, padding=24)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text=text, wraplength=TEXT_WRAP_LENGTH, justify=tk.LEFT).pack(anchor="w")

        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, pady=(24, 0))

        def finish(value):
            result[0] = value
            window.destroy()

        def activate_focused_button(event):
            focused = window.focus_get()
            if isinstance(focused, ttk.Button):
                focused.invoke()
            else:
                finish(buttons[0][1])

        primary_button = None
        for index, (label, value) in reversed(list(enumerate(buttons))):
            button = ttk.Button(button_row, text=label, width=BUTTON_MIN_WIDTH,
                                style=ACCENT_BUTTON_STYLE if index == 0 else "TButton",
                                command=lambda chosen=value: finish(chosen))
            button.pack(side=tk.RIGHT, padx=(8, 0))
            if index == 0:
                primary_button = button

        window.protocol("WM_DELETE_WINDOW", lambda: finish(dismiss_result))
        window.bind("<Escape>", lambda event: finish(dismiss_result))
        window.bind("<Return>", activate_focused_button)

        apply_window_chrome(window, theme)
        window.update_idletasks()
        width = max(DIALOG_WIDTH, window.winfo_reqwidth())
        height = window.winfo_reqheight()
        x = (window.winfo_screenwidth() - width) // 2
        y = (window.winfo_screenheight() - height) // 3
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.attributes("-topmost", True)
        window.deiconify()
        window.lift()
        window.focus_force()
        primary_button.focus_set()
        if parent:
            window.transient(parent)
            window.wait_visibility()
            window.grab_set()
            parent.wait_window(window)
        else:
            window.mainloop()
    finally:
        try:
            window.destroy()
        except tk.TclError:
            pass
    return result[0]
