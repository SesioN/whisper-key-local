import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import soxr

ANALYSIS_SAMPLE_RATE = 16000
VAD_FRAME_SAMPLES = 256
MAX_QUEUED_SECONDS = 2.0
WORKER_JOIN_TIMEOUT_SECONDS = 2.0
OVERFLOW_REPORT_INTERVAL_SECONDS = 10.0

_RESET = object()
_STOP = object()


class VadFramer:
    def __init__(self, input_rate: int):
        self.input_rate = input_rate
        self.reset()

    def reset(self):
        self._resampler = None
        if self.input_rate != ANALYSIS_SAMPLE_RATE:
            self._resampler = soxr.ResampleStream(self.input_rate, ANALYSIS_SAMPLE_RATE, 1, dtype='float32')
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, block: np.ndarray) -> tuple:
        samples = np.ascontiguousarray(block.reshape(-1), dtype=np.float32)
        if self._resampler is not None:
            samples = self._resampler.resample_chunk(samples)
        buffered = np.concatenate((self._pending, samples)) if len(self._pending) else samples
        complete_samples = len(buffered) // VAD_FRAME_SAMPLES * VAD_FRAME_SAMPLES
        frames = buffered[:complete_samples].reshape(-1, VAD_FRAME_SAMPLES)
        self._pending = buffered[complete_samples:].copy()
        return samples, frames


class AudioAnalysisWorker:
    def __init__(self, input_rate: int, block_samples: int,
                 process_vad_frame: Optional[Callable[[np.ndarray], None]],
                 process_streaming_audio: Optional[Callable[[np.ndarray], None]],
                 reset_consumers: Callable[[bool, bool], None]):
        self.logger = logging.getLogger(__name__)
        self._framer = VadFramer(input_rate)
        self._process_vad_frame = process_vad_frame
        self._process_streaming_audio = process_streaming_audio
        self._reset_consumers = reset_consumers
        self._max_queued_blocks = max(1, int(MAX_QUEUED_SECONDS * input_rate / max(block_samples, 1)))
        self._queue = queue.SimpleQueue()
        self._dropped_blocks = 0
        self._overflow_count = 0
        self._reported_overflow_count = 0
        self._last_overflow_report = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AudioAnalysis")

    def start(self):
        self._thread.start()

    def stop(self):
        self._queue.put(_STOP)
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
        self._report_overflows(force=True)

    def is_running(self) -> bool:
        return self._thread.is_alive()

    def submit(self, block: np.ndarray, feed_streaming: bool):
        if self._queue.qsize() >= self._max_queued_blocks:
            self._dropped_blocks += 1
            return
        self._queue.put((block, feed_streaming))

    def request_reset(self, reset_vad: bool, reset_streaming: bool):
        self._queue.put((_RESET, reset_vad, reset_streaming))

    def record_overflow(self):
        self._overflow_count += 1

    def _run(self):
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            try:
                if item[0] is _RESET:
                    _, reset_vad, reset_streaming = item
                    if reset_vad:
                        self._framer.reset()
                    self._reset_consumers(reset_vad, reset_streaming)
                else:
                    self._analyze(*item)
            except Exception as e:
                self.logger.error(f"Audio analysis failed: {e}")
            self._report_overflows()

    def _analyze(self, block: np.ndarray, feed_streaming: bool):
        audio_16k, frames = self._framer.push(block)
        if self._process_vad_frame:
            for frame in frames:
                self._process_vad_frame(frame)
        if feed_streaming and self._process_streaming_audio and len(audio_16k):
            self._process_streaming_audio(audio_16k)

    def _report_overflows(self, force: bool = False):
        now = time.monotonic()
        if not force and now - self._last_overflow_report < OVERFLOW_REPORT_INTERVAL_SECONDS:
            return
        self._last_overflow_report = now
        new_overflows = self._overflow_count - self._reported_overflow_count
        dropped_blocks, self._dropped_blocks = self._dropped_blocks, 0
        self._reported_overflow_count = self._overflow_count
        if new_overflows:
            self.logger.warning(f"Microphone input overflowed {new_overflows} times (audio was lost); "
                                f"the system may be too busy or the audio driver buffer too small")
        if dropped_blocks:
            self.logger.warning(f"Audio analysis fell behind, skipped {dropped_blocks} blocks for voice detection")
