import tkinter as tk
import logging
import threading
# ...

class LoadingScreen:
    def __init__(self):
        self.root = None
        self.thread = None
        self._running = False
        self.logger = logging.getLogger(__name__)
        self.destroyed_event = threading.Event()


    def _setup_ui(self):
        self.root = tk.Tk()
        # ... existing ...
        
        # Add a custom event or flag for quitting
        self.root.bind("<<Quit>>", lambda e: self.root.destroy())

    def hide(self):
        if self.root:
            self.root.event_generate("<<Quit>>", when="tail")
            if not self.destroyed_event.wait(timeout=3):
                self.logger.warning("Loading screen destruction timeout!")
        self._running = False
        self.logger.info("Loading screen hidden")

