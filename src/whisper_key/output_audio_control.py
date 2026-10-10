import logging
import queue
import threading

from .platform import media


class OutputAudioControl:
    MUTE_DELAY_SECONDS = 0.4

    def __init__(self, mute_output_enabled=False, pause_media_enabled=False):
        self.mute_output_enabled = mute_output_enabled
        self.pause_media_enabled = pause_media_enabled
        self.logger = logging.getLogger(__name__)
        self._muted_by_us = False
        self._paused_media = []
        self._engaged = False
        self._tasks = queue.Queue()
        self._worker_lock = threading.Lock()
        self._worker_started = False
        self._worker_alive = False

    def _ensure_worker(self):
        with self._worker_lock:
            if not self._worker_started:
                self._worker_started = True
                self._worker_alive = True
                threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            media.init_thread()
        except Exception as e:
            self.logger.warning(f"Output audio control unavailable: {e}")
            self._worker_alive = False
            return
        while True:
            task = self._tasks.get()
            try:
                task()
            except Exception as e:
                self.logger.warning(f"Output audio control failed: {e}")

    def engage(self):
        if not (self.mute_output_enabled or self.pause_media_enabled):
            return
        self._ensure_worker()
        self._engaged = True
        self._tasks.put(self._engage)

    def release(self):
        self._engaged = False
        if not self._worker_started:
            return
        self._tasks.put(self._release)

    def set_mute_output_enabled(self, enabled: bool):
        self.mute_output_enabled = enabled

    def set_pause_media_enabled(self, enabled: bool):
        self.pause_media_enabled = enabled

    def _run_step(self, step):
        try:
            step()
        except Exception as e:
            self.logger.warning(f"Output audio control failed: {e}")

    def _engage(self):
        if not self._engaged:
            return
        self._run_step(self._pause_media)
        threading.Event().wait(self.MUTE_DELAY_SECONDS)
        if not self._engaged:
            return
        self._run_step(self._mute_output)

    def _pause_media(self):
        if self.pause_media_enabled and not self._paused_media:
            self._paused_media = media.pause_playing_media()

    def _mute_output(self):
        if self.mute_output_enabled and not self._muted_by_us and not media.is_output_muted():
            media.set_output_muted(True)
            self._muted_by_us = True

    def _release(self):
        self._run_step(self._unmute_output)
        self._run_step(self._resume_media)

    def _unmute_output(self):
        if self._muted_by_us:
            if media.is_output_muted():
                media.set_output_muted(False)
            self._muted_by_us = False

    def _resume_media(self):
        if self._paused_media:
            media.resume_media(self._paused_media)
            self._paused_media = []

    def release_blocking(self, timeout=3.0):
        if not self._worker_alive:
            return
        self.release()
        done = threading.Event()
        self._tasks.put(done.set)
        done.wait(timeout)
