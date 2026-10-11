import logging
import threading

import numpy as np
import pytest

from whisper_key.audio_analysis import INCOMPLETE_ANALYSIS, AudioAnalysisWorker, VadFramer


@pytest.mark.parametrize("rate", [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 192000])
def test_framer_emits_exact_256_sample_frames_for_every_rate(rate):
    framer = VadFramer(rate)
    block_samples = int(256 * rate / 16000)
    block_count = rate // block_samples * 2
    frames = [frame for _ in range(block_count) for frame in framer.push(np.random.randn(block_samples, 1).astype(np.float32))[1]]
    expected_samples = block_count * block_samples * 16000 / rate
    assert frames and all(frame.shape == (256,) for frame in frames)
    assert abs(len(frames) * 256 - expected_samples) <= 4 * 256


def test_framer_passes_16k_audio_through_unchanged():
    framer = VadFramer(16000)
    block = np.arange(256, dtype=np.float32).reshape(-1, 1)
    audio_16k, frames = framer.push(block)
    assert np.array_equal(audio_16k, block.reshape(-1)) and np.array_equal(frames[0], block.reshape(-1))


def make_worker(process_vad_frame=None, process_streaming_audio=None, streaming_attached=lambda: True,
                rate=16000, block_samples=256, consumer_lock=None):
    events = []
    worker = AudioAnalysisWorker(
        input_rate=rate, block_samples=block_samples,
        process_vad_frame=process_vad_frame or (lambda frame: events.append(("vad", len(frame)))),
        process_streaming_audio=process_streaming_audio or (lambda audio: events.append(("stream", len(audio)))),
        streaming_attached=streaming_attached,
        reset_consumers=lambda vad, streaming: events.append(("reset", vad, streaming)),
        consumer_lock=consumer_lock or threading.Lock(),
    )
    return worker, events


def block():
    return np.zeros((256, 1), dtype=np.float32)


def drain(worker):
    return worker.request_snapshot(lambda: (True, 0.0)).result(5) == (True, 0.0)


def test_worker_processes_blocks_and_resets_in_submission_order():
    worker, events = make_worker()
    worker.start()
    worker.submit(block(), feed_streaming=False)
    assert worker.request_reset(True, True)
    worker.submit(block(), feed_streaming=True)
    assert drain(worker)
    worker.stop()
    assert events == [("vad", 256), ("reset", True, True), ("vad", 256), ("stream", 256)]


def test_worker_drops_blocks_when_it_falls_behind(caplog):
    gate = threading.Event()
    worker, _ = make_worker(process_vad_frame=lambda frame: gate.wait(5), block_samples=16000)
    worker.start()
    try:
        for _ in range(10):
            worker.submit(block(), feed_streaming=False)
        assert worker.dropped_blocks > 0
    finally:
        gate.set()
    with caplog.at_level(logging.WARNING):
        worker.stop()
    assert "skipped" in caplog.text


def test_stop_discards_backlog_and_rejects_new_blocks():
    gate = threading.Event()
    processed = []

    def slow_frame(frame):
        processed.append(1)
        gate.wait(5)

    worker, _ = make_worker(process_vad_frame=slow_frame, block_samples=64)
    worker.start()
    for _ in range(20):
        worker.submit(block(), feed_streaming=False)
    stopper = threading.Thread(target=worker.stop)
    stopper.start()
    gate.set()
    stopper.join(timeout=5)
    assert not stopper.is_alive()
    assert len(processed) < 20
    assert not worker.submit(block(), feed_streaming=False)
    assert not worker.is_accepting()


def test_snapshot_is_taken_exactly_after_the_blocks_submitted_before_it():
    frames_seen = []
    worker, _ = make_worker(process_vad_frame=lambda frame: frames_seen.append(1))
    worker.start()
    worker.submit(block(), feed_streaming=False)
    worker.submit(block(), feed_streaming=False)
    snapshot = worker.request_snapshot(lambda: (True, float(len(frames_seen))))
    worker.submit(block(), feed_streaming=False)
    assert snapshot.result(5) == (True, 2.0)
    assert drain(worker)
    worker.stop()
    assert len(frames_seen) == 3


def test_snapshot_times_out_as_incomplete_while_the_worker_lags():
    gate = threading.Event()
    worker, _ = make_worker(process_vad_frame=lambda frame: gate.wait(5))
    worker.start()
    try:
        worker.submit(block(), feed_streaming=False)
        assert worker.request_snapshot(lambda: (True, 1.0)).result(0) == INCOMPLETE_ANALYSIS
    finally:
        gate.set()
    worker.stop()


def test_stop_still_completes_snapshots_queued_before_it_and_refuses_new_requests():
    gate = threading.Event()
    frames_seen = []

    def slow_frame(frame):
        gate.wait(5)
        frames_seen.append(1)

    worker, _ = make_worker(process_vad_frame=slow_frame)
    worker.start()
    worker.submit(block(), feed_streaming=False)
    worker.submit(block(), feed_streaming=False)
    pending = worker.request_snapshot(lambda: (True, float(len(frames_seen))))
    worker.submit(block(), feed_streaming=False)
    stopper = threading.Thread(target=worker.stop)
    stopper.start()
    gate.set()
    stopper.join(timeout=5)
    assert pending.result(1) == (True, 2.0)
    assert worker.request_snapshot(lambda: (True, 1.0)).result(0) == INCOMPLETE_ANALYSIS
    assert not worker.request_reset(True, True)


def test_snapshots_of_a_worker_that_never_ran_are_abandoned():
    worker, _ = make_worker()
    pending = worker.request_snapshot(lambda: (True, 1.0))
    worker.stop()
    assert pending.result(1) == INCOMPLETE_ANALYSIS


def test_streaming_only_worker_skips_audio_until_a_recognizer_is_attached():
    attached = {"value": False}
    streamed = []
    worker = AudioAnalysisWorker(16000, 256, None, lambda audio: streamed.append(len(audio)),
                                 lambda: attached["value"], lambda vad, streaming: None, threading.Lock())
    worker.start()
    worker.submit(block(), feed_streaming=True)
    assert drain(worker)
    attached["value"] = True
    worker.submit(block(), feed_streaming=True)
    assert drain(worker)
    worker.stop()
    assert streamed == [256]


def test_consumers_run_under_the_shared_lock():
    consumer_lock = threading.Lock()
    held = []
    worker, _ = make_worker(process_vad_frame=lambda frame: held.append(consumer_lock.locked()), consumer_lock=consumer_lock)
    worker.start()
    worker.submit(block(), feed_streaming=False)
    assert drain(worker)
    worker.stop()
    assert held == [True]


def test_worker_survives_consumer_errors():
    calls = []

    def failing(frame):
        calls.append(1)
        raise RuntimeError("bad frame")

    worker, _ = make_worker(process_vad_frame=failing)
    worker.start()
    worker.submit(block(), False)
    worker.submit(block(), False)
    assert drain(worker)
    worker.stop()
    assert len(calls) == 2


def test_superseded_worker_stops_touching_consumers():
    consumer_lock = threading.Lock()
    gate = threading.Event()
    frames_seen = []

    def slow_frame(frame):
        gate.wait(5)
        frames_seen.append(1)

    worker, _ = make_worker(process_vad_frame=slow_frame, consumer_lock=consumer_lock)
    worker.start()
    for _ in range(5):
        worker.submit(block(), feed_streaming=False)
    pending = worker.request_snapshot(lambda: (True, 1.0))
    gate.set()
    with consumer_lock:
        worker.supersede()
        seen_when_superseded = len(frames_seen)
    worker.stop()
    assert len(frames_seen) == seen_when_superseded
    assert pending.result(1) == INCOMPLETE_ANALYSIS
