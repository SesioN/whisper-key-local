import numpy as np

from whisper_key.onnx_asr_engine import SAMPLE_RATE, split_at_quiet_points


def test_short_audio_is_one_chunk():
    audio = np.ones(SAMPLE_RATE * 5, dtype=np.float32)
    chunks = split_at_quiet_points(audio, 28)
    assert len(chunks) == 1 and len(chunks[0]) == len(audio)


def test_long_audio_splits_at_quiet_point_and_keeps_every_sample():
    audio = np.ones(SAMPLE_RATE * 40, dtype=np.float32)
    quiet_start = SAMPLE_RATE * 22
    audio[quiet_start:quiet_start + SAMPLE_RATE // 10] = 0.0
    chunks = split_at_quiet_points(audio, 28)
    assert sum(len(chunk) for chunk in chunks) == len(audio)
    assert all(len(chunk) <= SAMPLE_RATE * 28 for chunk in chunks)
    assert quiet_start <= len(chunks[0]) <= quiet_start + SAMPLE_RATE // 10
