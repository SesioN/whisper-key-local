import gc
import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable

METER_WIDTH = 300
METER_HEIGHT = 30
METER_BACKGROUND_COLOR = "#222222"
METER_SPEECH_COLOR = "#00C853"
METER_SILENCE_COLOR = "#757575"
THRESHOLD_LINE_COLOR = "#FF5252"
METER_REFRESH_INTERVAL_MS = 30
MIN_ONSET_THRESHOLD = 0.05
MAX_ONSET_THRESHOLD = 0.95
QUEUE_POLL_INTERVAL_MS = 100
STOP_TIMEOUT_SECONDS = 3.0

RAISE = "raise"
CLOSE = "close"


class VadSensitivityWindow:
    def __init__(self,
                 onset_threshold: float,
                 on_threshold_selected: Callable[[float], None],
                 on_opened: Callable[[], None],
                 on_closed: Callable[[], None]):
        self.onset_threshold = self._clamp_threshold(onset_threshold)
        self.applied_threshold = self.onset_threshold
        self.on_threshold_selected = on_threshold_selected
        self.on_opened = on_opened
        self.on_closed = on_closed
        self.speech_probability = 0.0
        self.logger = logging.getLogger(__name__)

        self._command_queue = queue.Queue()
        self._window_thread = None
        self._thread_lock = threading.Lock()

    def open(self):
        with self._thread_lock:
            if self._window_thread:
                self._command_queue.put(RAISE)
                return
            self._command_queue = queue.Queue()
            self._window_thread = threading.Thread(
                target=self._run_window_thread, args=(self._command_queue,), daemon=True, name="VadSensitivityWindow"
            )
            self._window_thread.start()

    def _clamp_threshold(self, threshold: float) -> float:
        return round(max(MIN_ONSET_THRESHOLD, min(MAX_ONSET_THRESHOLD, float(threshold))), 2)

    def update_probability(self, probability: float):
        self.speech_probability = probability

    def stop(self):
        with self._thread_lock:
            window_thread = self._window_thread
            if window_thread:
                self._command_queue.put(CLOSE)
        if not window_thread:
            return
        window_thread.join(timeout=STOP_TIMEOUT_SECONDS)
        if window_thread.is_alive():
            self.logger.warning("VAD sensitivity window did not close within timeout")

    def _run_window_thread(self, command_queue):
        try:
            self.speech_probability = 0.0
            self.on_opened()
            self._run_window(command_queue)
        except Exception as e:
            self.logger.error(f"VAD sensitivity window failed: {e}")
        finally:
            gc.collect()
            self.speech_probability = 0.0
            self._notify_closed()
            with self._thread_lock:
                self._window_thread = None

    def _notify_closed(self):
        try:
            self.on_closed()
        except Exception as e:
            self.logger.error(f"VAD sensitivity window close handler failed: {e}")

    def _run_window(self, command_queue):
        root = tk.Tk()
        try:
            self._build_window(root, command_queue)
            root.mainloop()
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass

    def _build_window(self, root, command_queue):
        root.title("Whisper Key - Voice detection sensitivity")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", root.quit)

        frame = ttk.Frame(root, padding=20)
        frame.pack()

        ttk.Label(frame, text="Speech detection threshold").pack(anchor="w")
        threshold_label = ttk.Label(frame, text=f"{self.onset_threshold:.2f}")
        threshold_label.pack(anchor="e")

        meter = tk.Canvas(frame, width=METER_WIDTH, height=METER_HEIGHT, bg=METER_BACKGROUND_COLOR, highlightthickness=0)
        level_bar = meter.create_rectangle(0, 0, 0, METER_HEIGHT, fill=METER_SILENCE_COLOR, outline="")
        threshold_line = meter.create_line(0, 0, 0, METER_HEIGHT, fill=THRESHOLD_LINE_COLOR, width=2)

        def show_threshold(value):
            self.onset_threshold = self._clamp_threshold(value)
            threshold_label.config(text=f"{self.onset_threshold:.2f}")
            line_x = self.onset_threshold * METER_WIDTH
            meter.coords(threshold_line, line_x, 0, line_x, METER_HEIGHT)

        def select_threshold(event):
            if self.onset_threshold == self.applied_threshold:
                return
            try:
                self.on_threshold_selected(self.onset_threshold)
                self.applied_threshold = self.onset_threshold
            except Exception as e:
                self.logger.error(f"Failed to apply VAD threshold: {e}")

        slider = ttk.Scale(
            frame, from_=MIN_ONSET_THRESHOLD, to=MAX_ONSET_THRESHOLD,
            orient=tk.HORIZONTAL, length=METER_WIDTH, command=show_threshold
        )
        slider.set(self.onset_threshold)
        slider.bind("<ButtonRelease-1>", select_threshold)
        slider.bind("<KeyRelease>", select_threshold)
        slider.pack(pady=(5, 10))
        meter.pack()

        ttk.Label(frame, text="Speak: the bar should pass the red line only while you talk").pack(pady=(10, 0))

        def refresh_meter():
            probability = max(0.0, min(1.0, self.speech_probability))
            color = METER_SPEECH_COLOR if probability > self.onset_threshold else METER_SILENCE_COLOR
            meter.coords(level_bar, 0, 0, probability * METER_WIDTH, METER_HEIGHT)
            meter.itemconfig(level_bar, fill=color)
            root.after(METER_REFRESH_INTERVAL_MS, refresh_meter)

        def process_command_queue():
            try:
                while True:
                    command = command_queue.get_nowait()
                    if command == CLOSE:
                        root.quit()
                        return
                    if command == RAISE:
                        root.deiconify()
                        root.lift()
            except queue.Empty:
                pass
            root.after(QUEUE_POLL_INTERVAL_MS, process_command_queue)

        show_threshold(self.onset_threshold)
        refresh_meter()
        process_command_queue()
