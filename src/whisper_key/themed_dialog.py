import tkinter as tk
from tkinter import ttk

from .themed_widgets import ACCENT_BUTTON_STYLE, apply_theme, apply_window_chrome

DIALOG_WIDTH = 440
TEXT_WRAP_LENGTH = DIALOG_WIDTH - 48
BUTTON_MIN_WIDTH = 9


def show_dialog(title: str, text: str, buttons: list, dismiss_result):
    root = tk.Tk()
    result = [dismiss_result]
    try:
        root.title(title)
        root.resizable(False, False)
        theme = apply_theme(root)

        frame = ttk.Frame(root, padding=24)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text=text, wraplength=TEXT_WRAP_LENGTH, justify=tk.LEFT).pack(anchor="w")

        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, pady=(24, 0))

        def finish(value):
            result[0] = value
            root.quit()

        primary_button = None
        for index, (label, value) in reversed(list(enumerate(buttons))):
            button = ttk.Button(button_row, text=label, width=BUTTON_MIN_WIDTH,
                                style=ACCENT_BUTTON_STYLE if index == 0 else "TButton",
                                command=lambda chosen=value: finish(chosen))
            button.pack(side=tk.RIGHT, padx=(8, 0))
            if index == 0:
                primary_button = button

        root.protocol("WM_DELETE_WINDOW", lambda: finish(dismiss_result))
        root.bind("<Escape>", lambda event: finish(dismiss_result))
        root.bind("<Return>", lambda event: finish(buttons[0][1]))

        apply_window_chrome(root, theme)
        root.update_idletasks()
        width = max(DIALOG_WIDTH, root.winfo_reqwidth())
        height = root.winfo_reqheight()
        x = (root.winfo_screenwidth() - width) // 2
        y = (root.winfo_screenheight() - height) // 3
        root.geometry(f"{width}x{height}+{x}+{y}")
        root.attributes("-topmost", True)
        root.focus_force()
        primary_button.focus_set()
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
    return result[0]
