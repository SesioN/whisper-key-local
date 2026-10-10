import logging
import queue
import threading

from .platform import media


class OutputAudioControl:
    ENGAGE_DELAY_SECONDS = 0.4  # Lets the start sound finish before output is muted

    def __init__(self, mute_output_enabled=False, pause_media_enabled=False):
        self.mute_output_enabled = mute_output_enabled
        self.pause_media_enabled = pause_media_enabled
        self.logger = logging.getLogger(__name__)
        self._muted_by_us = False
        self._paused_media = []
        self._engaged = False
        self._tasks = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            media.init_thread()
        except Exception as e:
            self.logger.warning(f"Output audio control unavailable: {e}")
        while True:
            task = self._tasks.get()
            try:
                task()
            except Exception as e:
                self.logger.warning(f"Output audio control failed: {e}")

    def engage(self):
        self._engaged = True
        self._tasks.put(self._engage)

    def release(self):
        self._engaged = False
        self._tasks.put(self._release)

    def _engage(self):
        threading.Event().wait(self.ENGAGE_DELAY_SECONDS)
        if not self._engaged:
            return
        if self.pause_media_enabled and not self._paused_media:
            self._paused_media = media.pause_playing_media()
        if self.mute_output_enabled and not self._muted_by_us and not media.is_output_muted():
            media.set_output_muted(True)
            self._muted_by_us = True

    def _release(self):
        if self._muted_by_us:
            self._muted_by_us = False
            media.set_output_muted(False)
        if self._paused_media:
            paused_media, self._paused_media = self._paused_media, []
            media.resume_media(paused_media)

    def release_blocking(self, timeout=3.0):
        self.release()
        done = threading.Event()
        self._tasks.put(done.set)
        done.wait(timeout)
