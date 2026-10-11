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
REPORT_INTERVAL_SECONDS = 10.0

_RESET = object()
_SNAPSHOT = object()
_STOP = object()
INCOMPLETE_ANALYSIS = (False, 0.0)


class AnalysisSnapshot:
    def __init__(self):
        self._ready = threading.Event()
        self._result = INCOMPLETE_ANALYSIS

    def complete(self, result: tuple):
        self._result = result
        self._ready.set()

    def abandon(self):
        self._ready.set()

    def result(self, timeout: float) -> tuple:
        if not self._ready.wait(timeout):
            return INCOMPLETE_ANALYSIS
        return self._result


def abandoned_snapshot() -> AnalysisSnapshot:
    snapshot = AnalysisSnapshot()
    snapshot.abandon()
    return snapshot


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
                 streaming_attached: Callable[[], bool],
                 reset_consumers: Callable[[bool, bool], None],
                 consumer_lock: threading.Lock):
        self.logger = logging.getLogger(__name__)
        self._framer = VadFramer(input_rate)
        self._process_vad_frame = process_vad_frame
        self._process_streaming_audio = process_streaming_audio
        self._streaming_attached = streaming_attached
        self._reset_consumers = reset_consumers
        self._consumer_lock = consumer_lock
        self._max_queued_blocks = max(1, int(MAX_QUEUED_SECONDS * input_rate / block_samples))
        self._queue = queue.SimpleQueue()
        self._queue_lock = threading.Lock()
        self._stopping = False
        self.submitted_blocks = 0
        self.dropped_blocks = 0
        self._reported_dropped_blocks = 0
        self._last_report = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AudioAnalysis")

    def start(self):
        self._thread.start()

    def stop(self):
        with self._queue_lock:
            self._stopping = True
            pending_items = []
            while True:
                try:
                    pending_items.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            for item in self._items_needed_for_snapshots(pending_items):
                self._queue.put(item)
            self._queue.put((_STOP,))
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
        if not self._thread.is_alive():
            self._abandon_queued_snapshots()
        self._report_dropped_blocks()

    def _abandon_queued_snapshots(self):
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item[0] is _SNAPSHOT:
                item[1].abandon()

    @staticmethod
    def _items_needed_for_snapshots(pending_items: list) -> list:
        last_snapshot_index = max((index for index, item in enumerate(pending_items) if item[0] is _SNAPSHOT), default=-1)
        return pending_items[:last_snapshot_index + 1]

    def is_accepting(self) -> bool:
        return self._thread.is_alive() and not self._stopping

    def submit(self, block: np.ndarray, feed_streaming: bool) -> bool:
        with self._queue_lock:
            if self._stopping:
                return False
            if self._queue.qsize() >= self._max_queued_blocks:
                self.dropped_blocks += 1
                return False
            self.submitted_blocks += 1
            self._queue.put((block, feed_streaming))
            return True

    def request_reset(self, reset_vad: bool, reset_streaming: bool) -> bool:
        with self._queue_lock:
            if self._stopping:
                return False
            self._queue.put((_RESET, reset_vad, reset_streaming))
            return True

    def request_snapshot(self, take_snapshot: Callable[[], tuple]) -> AnalysisSnapshot:
        snapshot = AnalysisSnapshot()
        with self._queue_lock:
            if self._stopping:
                snapshot.abandon()
            else:
                self._queue.put((_SNAPSHOT, snapshot, take_snapshot))
        return snapshot

    def _run(self):
        while True:
            try:
                item = self._queue.get(timeout=REPORT_INTERVAL_SECONDS)
            except queue.Empty:
                self._report_dropped_blocks()
                continue
            if item[0] is _STOP:
                return
            try:
                with self._consumer_lock:
                    if item[0] is _RESET:
                        _, reset_vad, reset_streaming = item
                        if reset_vad:
                            self._framer.reset()
                        self._reset_consumers(reset_vad, reset_streaming)
                    elif item[0] is _SNAPSHOT:
                        _, snapshot, take_snapshot = item
                        snapshot.complete(take_snapshot())
                    else:
                        self._analyze(*item)
            except Exception as e:
                self.logger.error(f"Audio analysis failed: {e}")
                if item[0] is _SNAPSHOT:
                    item[1].abandon()
            if time.monotonic() - self._last_report >= REPORT_INTERVAL_SECONDS:
                self._report_dropped_blocks()

    def _analyze(self, block: np.ndarray, feed_streaming: bool):
        stream_audio = feed_streaming and self._process_streaming_audio is not None and self._streaming_attached()
        if self._process_vad_frame is None and not stream_audio:
            return
        audio_16k, frames = self._framer.push(block)
        if self._process_vad_frame:
            for frame in frames:
                self._process_vad_frame(frame)
        if stream_audio and len(audio_16k):
            self._process_streaming_audio(audio_16k)

    def _report_dropped_blocks(self):
        self._last_report = time.monotonic()
        dropped_blocks = self.dropped_blocks
        newly_dropped = dropped_blocks - self._reported_dropped_blocks
        self._reported_dropped_blocks = dropped_blocks
        if newly_dropped:
            self.logger.warning(f"Audio analysis fell behind, skipped {newly_dropped} blocks for voice detection")
