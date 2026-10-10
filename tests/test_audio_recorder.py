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


import threading
import time
import types

from whisper_key import audio_recorder as audio_recorder_module
from whisper_key.audio_recorder import AudioRecorder
from whisper_key.voice_activity_detection import ContinuousVoiceDetector, VAD_HOP_DURATION_SEC


class FakeInputStream:
    def __init__(self, samplerate, channels, callback, dtype, blocksize, device):
        self.samplerate = samplerate
        self.channels = channels
        self.callback = callback
        self.blocksize = blocksize or 256
        self.active = True
        self._thread = threading.Thread(target=self._feed, daemon=True)

    def _feed(self):
        status = types.SimpleNamespace(input_overflow=False, __bool__=lambda self: False)
        while self.active:
            block = np.full((self.blocksize, self.channels), 0.1, dtype=np.float32)
            self.callback(block, self.blocksize, None, None)
            time.sleep(self.blocksize / self.samplerate / 4)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *args):
        self.active = False
        self._thread.join(timeout=2)


def install_fake_sounddevice(monkeypatch, rate, hostapi="Windows WASAPI"):
    fake_sd = types.SimpleNamespace(
        query_devices=lambda device=None, kind=None: {"name": "Fake mic", "hostapi": 0, "default_samplerate": rate,
                                                       "max_input_channels": 2, "index": 0},
        query_hostapis=lambda index=None: {"name": hostapi},
        InputStream=FakeInputStream,
        sleep=lambda ms: time.sleep(ms / 1000),
        PortAudioError=RuntimeError,
    )
    monkeypatch.setattr(audio_recorder_module, "sd", fake_sd)


class AlwaysSpeech:
    def __init__(self):
        self.frame_lengths = set()

    def process(self, audio_int16):
        self.frame_lengths.add(len(audio_int16))
        return 0.95, 1


def make_recorder(monkeypatch, rate, channels=1):
    install_fake_sounddevice(monkeypatch, rate)
    ten_vad = AlwaysSpeech()
    detector = ContinuousVoiceDetector(ten_vad, 0.7, 0.55, 30.0, VAD_HOP_DURATION_SEC, 0.05, 1.5)
    vad_manager = types.SimpleNamespace(is_available=lambda: True,
                                        create_continuous_detector=lambda event_callback, probability_callback: detector)
    recorder = AudioRecorder(on_vad_event=lambda event: None, channels=channels, max_duration=0,
                             vad_manager=vad_manager, pre_roll_seconds=0)
    return recorder, ten_vad


def test_recording_through_the_analysis_worker_at_22050_hz(monkeypatch):
    recorder, ten_vad = make_recorder(monkeypatch, 22050, channels=2)
    assert recorder.start_recording()
    time.sleep(0.4)
    recording = recorder.stop_recording()
    assert recording is not None and recording.audio.ndim == 1
    assert recording.realtime_speech_detected
    assert ten_vad.frame_lengths == {256}
    assert recording.duration_seconds > 0.2


def test_monitoring_survives_recording_stop(monkeypatch):
    recorder, _ = make_recorder(monkeypatch, 48000)
    assert recorder.start_monitoring()
    assert recorder.start_recording()
    time.sleep(0.2)
    assert recorder.stop_recording() is not None
    assert recorder._stream_is_running()
    recorder.stop_monitoring()
    assert not recorder._stream_is_running()
    assert not recorder._analysis_worker.is_running()
