import tkinter as tk
import threading
import os
import platform
import ctypes
from PIL import Image, ImageTk
from .utils import resolve_asset_path
from .platform import IS_MACOS

class LoadingScreen:
    def __init__(self):
        self.root = None
        self.thread = None
        self._running = False

    def _setup_ui(self):
        self.root = tk.Tk()
        # Ensure it's not hidden initially
        self.root.deiconify()
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        
        # Window styling
        self.root.config(bg="#222222")
        self.root.geometry("200x150")
        
        # Center on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - 100
        y = (screen_height // 2) - 75
        self.root.geometry(f"200x150+{x}+{y}")

        # Determine icon path
        plat = "macos" if IS_MACOS else "windows"
        try:
            # Use a PNG for better compatibility if possible
            # Check if there's a tray icon (PNG) instead of ico
            icon_path = resolve_asset_path(f"platform/{plat}/assets/tray_idle.png")
            icon_img = Image.open(icon_path).convert("RGBA")
            icon_img = icon_img.resize((64, 64), Image.Resampling.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_img)
        except Exception:
            icon_photo = None

        if icon_photo:
            label = tk.Label(self.root, image=icon_photo, bg="#222222")
            label.pack(pady=(20, 10))
            self.root.icon_photo = icon_photo
        
        text = tk.Label(self.root, text="Loading...", bg="#222222", fg="white", font=("Arial", 12))
        text.pack()
        
        # Explicit update
        self.root.update()

    def show(self):
        self._running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        self._setup_ui()
        self.root.mainloop()

    def hide(self):
        if self.root:
            self.root.after(0, self.root.destroy)
        self._running = False
