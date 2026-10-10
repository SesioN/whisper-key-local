import logging
import threading
import time

import numpy as np
import pytest

from whisper_key.audio_analysis import AudioAnalysisWorker, VadFramer


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


def make_worker(rate=16000):
    events = []
    worker = AudioAnalysisWorker(
        input_rate=rate, block_samples=256,
        process_vad_frame=lambda frame: events.append(("vad", len(frame))),
        process_streaming_audio=lambda audio: events.append(("stream", len(audio))),
        reset_consumers=lambda vad, streaming: events.append(("reset", vad, streaming)),
    )
    return worker, events


def test_worker_processes_blocks_and_resets_in_submission_order():
    worker, events = make_worker()
    worker.start()
    block = np.zeros((256, 1), dtype=np.float32)
    worker.submit(block, feed_streaming=False)
    worker.request_reset(True, True)
    worker.submit(block, feed_streaming=True)
    worker.stop()
    assert events == [("vad", 256), ("reset", True, True), ("vad", 256), ("stream", 256)]


def test_worker_drops_blocks_when_it_falls_behind(caplog):
    gate = threading.Event()
    worker = AudioAnalysisWorker(input_rate=16000, block_samples=16000, process_vad_frame=lambda frame: gate.wait(5),
                                 process_streaming_audio=None, reset_consumers=lambda vad, streaming: None)
    worker.start()
    block = np.zeros((256, 1), dtype=np.float32)
    for _ in range(10):
        worker.submit(block, feed_streaming=False)
    assert worker._dropped_blocks > 0
    gate.set()
    with caplog.at_level(logging.WARNING):
        worker.stop()
    assert "skipped" in caplog.text


def test_worker_reports_input_overflows(caplog):
    worker, _ = make_worker()
    worker.start()
    worker.record_overflow()
    worker.record_overflow()
    with caplog.at_level(logging.WARNING):
        worker.stop()
    assert "overflowed 2 times" in caplog.text


def test_worker_survives_consumer_errors():
    calls = []

    def failing(frame):
        calls.append(1)
        raise RuntimeError("bad frame")

    worker = AudioAnalysisWorker(16000, 256, failing, None, lambda vad, streaming: None)
    worker.start()
    worker.submit(np.zeros((256, 1), dtype=np.float32), False)
    worker.submit(np.zeros((256, 1), dtype=np.float32), False)
    worker.stop()
    assert len(calls) == 2
