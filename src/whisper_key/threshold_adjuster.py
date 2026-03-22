import tkinter as tk
from tkinter import ttk
import threading
import logging

class ThresholdAdjuster:
    """
    Persistent Threshold Adjuster UI for Voice Activity Detection.
    """
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.logger = logging.getLogger(__name__)
        
        self.root = None
        self.canvas = None
        self.slider = None
        self.meter_rect = 0
        self.threshold_line = 0
        
        self.current_db = -60.0  # Default to silence
        self.threshold = state_manager.config_manager.get_vad_config().get('vad_onset_threshold', 0.7)
        
        self.canvas_width = 300
        self.canvas_height = 30
        
        self._lock = threading.Lock() # Protects self.root
        self._db_lock = threading.Lock() # Protects self.current_db
        
    def show(self):
        with self._lock:
            if self.root:
                self.root.deiconify()
                self.root.lift()
                self.state_manager.audio_recorder.start_monitoring()
                return

            threading.Thread(target=self._run_ui, daemon=True).start()

    def _run_ui(self):
        root = tk.Tk()
        root.title("Threshold Settings")
        root.protocol("WM_DELETE_WINDOW", self.close)
        
        with self._lock:
            self.root = root
        
        frame = ttk.Frame(root, padding=20)
        frame.pack()

        ttk.Label(frame, text="Sensitivity Threshold:").pack()
        self.slider = ttk.Scale(
            frame, from_=0.0, to=1.0, 
            orient=tk.HORIZONTAL, 
            command=self._on_slider_change
        )
        self.slider.set(self.threshold)
        self.slider.pack(fill=tk.X, pady=5)
        
        self.canvas = tk.Canvas(frame, width=self.canvas_width, height=self.canvas_height, bg="#222")
        self.canvas.pack(pady=10)
        
        self.meter_rect = self.canvas.create_rectangle(0, 0, 0, self.canvas_height, fill="#00FF00", outline="")
        self.threshold_line = self.canvas.create_line(0, 0, 0, self.canvas_height, fill="red", width=2)
        
        self.state_manager.audio_recorder.start_monitoring()
        
        self._update_ui()
        root.mainloop()

    def _on_slider_change(self, val):
        self.threshold = float(val)
        offset = max(0.05, self.threshold - 0.15)
        self.state_manager.update_vad_thresholds(self.threshold, offset)
        self._draw_threshold()

    def _update_ui(self):
        if not self.root or not self.canvas or self.meter_rect is None or self.threshold_line is None:
            return
        
        with self._db_lock:
            db_val = self.current_db
        
        db_normalized = max(0.0, min(1.0, (db_val + 60) / 60))
        fill_x = db_normalized * self.canvas_width
        self.canvas.coords(self.meter_rect, 0, 0, fill_x, self.canvas_height)
        
        self._draw_threshold()
        
        try:
            self.root.after(30, self._update_ui)
        except tk.TclError:
            pass

    def _draw_threshold(self):
        if not self.canvas: return
        x = self.threshold * self.canvas_width
        self.canvas.coords(self.threshold_line, x, 0, x, self.canvas_height)

    def update_db(self, db):
        with self._db_lock:
            self.current_db = db
        
    def update_probability(self, probability):
        with self._db_lock:
            self.current_db = (probability * 60) - 60

    def close(self):
        with self._lock:
            if self.root:
                self.root.withdraw()
                self.state_manager.audio_recorder.stop_monitoring()
