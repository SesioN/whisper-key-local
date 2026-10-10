import numpy as np

from whisper_key.audio_recorder import to_mono_block


def test_stereo_block_is_mixed_to_mono_column():
    stereo = np.array([[0.2, 0.4], [-1.0, 1.0]], dtype=np.float32)
    mono = to_mono_block(stereo)
    assert mono.shape == (2, 1)
    assert np.allclose(mono[:, 0], [0.3, 0.0])


def test_mono_block_is_copied():
    block = np.ones((4, 1), dtype=np.float32)
    mono = to_mono_block(block)
    block[:] = 0
    assert mono.shape == (4, 1) and mono.sum() == 4


from whisper_key.audio_recorder import trim_trailing_silence


def test_trailing_silence_is_trimmed_down_to_the_kept_margin():
    audio = np.ones(16000 * 10, dtype=np.float32)
    trimmed = trim_trailing_silence(audio, trailing_silence_seconds=6.0, keep_seconds=1.0,
                                    min_remaining_seconds=0.5, sample_rate=16000)
    assert len(trimmed) == 16000 * 5


def test_short_silence_is_kept():
    audio = np.ones(16000 * 10, dtype=np.float32)
    assert len(trim_trailing_silence(audio, 0.8, 1.0, 0.5, 16000)) == len(audio)


def test_trim_never_leaves_too_little_audio():
    audio = np.ones(16000 * 2, dtype=np.float32)
    assert len(trim_trailing_silence(audio, 30.0, 1.0, 0.5, 16000)) == len(audio)
