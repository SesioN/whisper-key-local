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
