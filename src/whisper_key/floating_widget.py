import tkinter as tk
from PIL import Image, ImageTk
import queue
import platform
import logging
import ctypes

from .utils import resolve_asset_path
from .platform import IS_MACOS

class FloatingWidget:
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        self.queue = queue.Queue()
        self._is_running = False
        self.root = None
        self.icons = {}

    def _setup_ui(self):
        self.root = tk.Tk()
        self.is_dragging = False
        self._drag_start_x = 0
        self._drag_start_y = 0
        
        self._setup_window()
        self._load_icons()
        self._create_label()
        self._bind_events()
        
        self.current_state = "idle"
        self.update_icon("idle")
        
        # Start polling queue
        self.root.after(100, self._process_queue)

    def _setup_window(self):
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self.root.title("Whisper Key Floating Widget")
        
        # Set transparency based on platform
        if platform.system() == "Windows":
            # Use a specific color for transparency
            self.bg_color = "magenta"
            self.root.wm_attributes("-transparentcolor", self.bg_color)
            self.root.config(bg=self.bg_color)

            # WS_EX_NOACTIVATE = 0x08000000 (Prevents window from taking focus when clicked)
            # GWL_EXSTYLE = -20
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x08000000)
            
        elif IS_MACOS:
            self.bg_color = "systemTransparent"
            self.root.wm_attributes("-transparent", True)
            self.root.config(bg=self.bg_color)
        else:
            # Linux/other fallback
            self.bg_color = "black" 
            # Linux transparency is tricky, usually -alpha works best for whole window
            self.root.wait_visibility(self.root)
            self.root.wm_attributes("-alpha", 0.9)

        # Initial position (bottom right-ish)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        # Default position: 100px from right, 100px from bottom
        x = screen_width - 150
        y = screen_height - 150
        self.root.geometry(f"+{x}+{y}")

    def _load_icons(self):
        # Determine platform folder for assets
        plat = "macos" if IS_MACOS else "windows"
        
        def load_icon(name):
             try:
                 path = resolve_asset_path(f"platform/{plat}/assets/tray_{name}.png")
                 img = Image.open(path).convert("RGBA")
                 # Resize to a reasonable widget size (e.g. 64x64)
                 img = img.resize((64, 64), Image.Resampling.LANCZOS)
                 return ImageTk.PhotoImage(img)
             except Exception as e:
                 self.logger.error(f"Failed to load icon {name}: {e}")
                 # Create a fallback colored square
                 img = Image.new('RGBA', (64, 64), color='red')
                 return ImageTk.PhotoImage(img)

        self.icons = {
            "idle": load_icon("idle"),
            "recording": load_icon("recording"),
            "processing": load_icon("processing"),
        }

    def _create_label(self):
        self.label = tk.Label(
            self.root, 
            image=self.icons["idle"], 
            bg=self.bg_color, 
            bd=0,
            highlightthickness=0
        )
        self.label.pack()

    def _bind_events(self):
        self.label.bind("<Button-1>", self._on_press)
        self.label.bind("<B1-Motion>", self._on_drag)
        self.label.bind("<ButtonRelease-1>", self._on_release)
        
        # Also bind right click to close? Or context menu? 
        # For now, keep it simple. User can close via tray.

    def _on_press(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self.is_dragging = False

    def _on_drag(self, event):
        self.is_dragging = True
        # Calculate delta
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        
        x = self.root.winfo_x() + dx
        y = self.root.winfo_y() + dy
        self.root.geometry(f"+{x}+{y}")

    def _on_release(self, event):
        # If mouse didn't move much, treat as click
        if not self.is_dragging:
             self._handle_click()
        self.is_dragging = False

    def _handle_click(self):
        current_app_state = self.state_manager.get_current_state()
        
        if current_app_state == "idle":
            self.logger.info("Floating widget clicked: Start Recording")
            # We must not block the GUI thread with long operations
            # start_recording itself is fast (just signals), but if it blocks, we should run in thread.
            # StateManager.start_recording() calls audio_recorder.start() which might be slowish?
            # It's better to run it via the existing hotkey/trigger mechanism logic if possible.
            # But calling directly is fine for now as start_recording is designed to be called from hotkeys.
            self.state_manager.start_recording()
            
        elif current_app_state == "recording":
            self.logger.info("Floating widget clicked: Stop Recording")
            self.state_manager.stop_recording()
            
        elif current_app_state == "processing":
            self.logger.info("Floating widget clicked while processing (ignored)")

    def update_state(self, new_state):
        # Thread-safe update via queue
        self.queue.put(("state", new_state))

    def _process_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                if msg_type == "state":
                    self.update_icon(data)
                elif msg_type == "quit":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        
        # Check if we should check for external shutdown signals? 
        # main.py handles shutdown_event. 
        # We can add a check here if we had access to shutdown_event.
        # But main.py will likely call stop() on us.
        
        if self._is_running:
            self.root.after(100, self._process_queue)

    def update_icon(self, state):
        self.current_state = state
        icon = self.icons.get(state, self.icons["idle"])
        self.label.configure(image=icon)
    
    def start(self, shutdown_event=None):
        self._is_running = True
        self._setup_ui()
        
        if shutdown_event:
            self._check_shutdown(shutdown_event)
            
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
            
    def _check_shutdown(self, shutdown_event):
        if shutdown_event.is_set():
            self.stop()
            return
        self.root.after(200, lambda: self._check_shutdown(shutdown_event))

    def stop(self):
        self._is_running = False
        self.queue.put(("quit", None))
