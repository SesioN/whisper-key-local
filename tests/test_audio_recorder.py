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


def wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class FakeInputStream:
    instances = []
    overflow_blocks = 0

    def __init__(self, samplerate, channels, callback, dtype, blocksize, device):
        self.samplerate = samplerate
        self.channels = channels
        self.callback = callback
        self.blocksize = blocksize or 256
        self.active = True
        self.blocks_delivered = 0
        self._thread = threading.Thread(target=self._feed, daemon=True)
        FakeInputStream.instances.append(self)

    def _feed(self):
        while self.active:
            block = np.full((self.blocksize, self.channels), 0.1, dtype=np.float32)
            overflow = self.blocks_delivered < FakeInputStream.overflow_blocks
            status = types.SimpleNamespace(input_overflow=overflow) if overflow else None
            self.callback(block, self.blocksize, None, status)
            self.blocks_delivered += 1
            time.sleep(self.blocksize / self.samplerate / 4)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *args):
        self.active = False
        self._thread.join(timeout=2)


def install_fake_sounddevice(monkeypatch, rate, hostapi="Windows WASAPI"):
    FakeInputStream.instances = []
    FakeInputStream.overflow_blocks = 0
    fake_sd = types.SimpleNamespace(
        query_devices=lambda device=None, kind=None: {"name": "Fake mic", "hostapi": 0, "default_samplerate": rate,
                                                       "max_input_channels": 2, "index": 0},
        query_hostapis=lambda index=None: {"name": hostapi},
        InputStream=FakeInputStream,
        sleep=lambda ms: time.sleep(ms / 1000),
        PortAudioError=RuntimeError,
    )
    monkeypatch.setattr(audio_recorder_module, "sd", fake_sd)


class ScriptedSpeech:
    def __init__(self, probability=0.95, delay=0.0):
        self.frame_lengths = set()
        self.probability = probability
        self.delay = delay

    def process(self, audio_int16):
        self.frame_lengths.add(len(audio_int16))
        if self.delay:
            time.sleep(self.delay)
        return self.probability, 1


def make_recorder(monkeypatch, rate, channels=1, ten_vad=None, vad_available=True, streaming_manager=None):
    install_fake_sounddevice(monkeypatch, rate)
    ten_vad = ten_vad or ScriptedSpeech()
    detector = ContinuousVoiceDetector(ten_vad, 0.7, 0.55, 30.0, VAD_HOP_DURATION_SEC, 0.05, 1.5)
    vad_manager = types.SimpleNamespace(is_available=lambda: vad_available,
                                        create_continuous_detector=lambda event_callback, probability_callback: detector)
    recorder = AudioRecorder(on_vad_event=lambda event: None, channels=channels, max_duration=0,
                             vad_manager=vad_manager, pre_roll_seconds=0, streaming_manager=streaming_manager)
    return recorder, ten_vad


def delivered_blocks():
    return sum(stream.blocks_delivered for stream in FakeInputStream.instances)


def test_recording_through_the_analysis_worker_at_22050_hz(monkeypatch):
    recorder, ten_vad = make_recorder(monkeypatch, 22050, channels=2)
    assert recorder.start_recording()
    assert wait_until(lambda: delivered_blocks() >= 30)
    recording = recorder.stop_recording()
    assert recording is not None and recording.audio.ndim == 1
    assert recording.realtime_speech_detected
    assert ten_vad.frame_lengths == {256}
    assert recording.duration_seconds > 0.2


def test_monitoring_survives_recording_stop(monkeypatch):
    recorder, _ = make_recorder(monkeypatch, 48000)
    assert recorder.start_monitoring()
    assert recorder.start_recording()
    assert wait_until(lambda: delivered_blocks() >= 10)
    assert recorder.stop_recording() is not None
    assert recorder._stream_is_running()
    recorder.stop_monitoring()
    assert not recorder._stream_is_running()
    assert not recorder._analysis_worker.is_accepting()


def test_restarting_the_stream_never_runs_two_workers_on_the_detector(monkeypatch):
    recorder, _ = make_recorder(monkeypatch, 16000, ten_vad=ScriptedSpeech(delay=0.002))
    active_consumers = []
    concurrent = []
    original = recorder.continuous_vad.process_chunk

    def tracking_process_chunk(frame):
        active_consumers.append(1)
        if len(active_consumers) > 1:
            concurrent.append(True)
        try:
            return original(frame)
        finally:
            active_consumers.pop()

    recorder.continuous_vad.process_chunk = tracking_process_chunk
    for _ in range(5):
        assert recorder.start_monitoring()
        assert wait_until(lambda: delivered_blocks() >= 5)
        recorder.stop_monitoring()
    assert not concurrent


def test_incomplete_analysis_disables_trim_and_realtime_speech(monkeypatch):
    recorder, _ = make_recorder(monkeypatch, 16000, ten_vad=ScriptedSpeech(delay=0.05))
    assert recorder.start_recording()
    assert wait_until(lambda: delivered_blocks() >= 20)
    recorder.ANALYSIS_CATCH_UP_SECONDS = 0.0
    recording = recorder.stop_recording()
    assert not recording.realtime_speech_detected
    assert recording.trailing_silence_seconds == 0.0


def test_streaming_without_vad_feeds_the_recognizer_once_attached(monkeypatch):
    streamed = []
    recognizer = types.SimpleNamespace(process_chunk=lambda audio: streamed.append(len(audio)), reset=lambda: None)
    streaming_manager = types.SimpleNamespace(streaming_enabled=True, is_available=lambda: bool(streamed is not None and attach["ready"]),
                                              create_continuous_recognizer=lambda result_callback: recognizer)
    attach = {"ready": False}
    recorder, _ = make_recorder(monkeypatch, 16000, vad_available=False, streaming_manager=streaming_manager)
    assert recorder.continuous_vad is None and recorder.continuous_streaming is None
    assert recorder.start_recording()
    assert wait_until(lambda: delivered_blocks() >= 5)
    assert streamed == []
    attach["ready"] = True
    recorder.attach_streaming()
    assert wait_until(lambda: len(streamed) >= 3)
    recorder.stop_recording()


def test_input_overflows_are_reported_without_an_analysis_worker(monkeypatch, caplog):
    import logging
    recorder, _ = make_recorder(monkeypatch, 16000, vad_available=False)
    FakeInputStream.overflow_blocks = 3
    recorder.pre_roll_seconds = 1.0
    with caplog.at_level(logging.WARNING):
        assert recorder.start_buffering()
        assert wait_until(lambda: delivered_blocks() >= 5)
        recorder.stop_buffering()
    assert "overflowed 3 times" in caplog.text


def test_resets_are_queued_while_streaming_and_direct_when_stopped(monkeypatch):
    recorder, _ = make_recorder(monkeypatch, 16000)
    queued = []
    recorder.continuous_vad.speech_detected_since_reset = True
    recorder._reset_realtime_analysis(stream_is_running=False, reset_vad=True, reset_streaming=False)
    assert not recorder.continuous_vad.speech_detected_since_reset

    assert recorder.start_monitoring()
    worker = recorder._analysis_worker
    monkeypatch.setattr(worker, "request_reset", lambda reset_vad, reset_streaming: queued.append((reset_vad, reset_streaming)))
    recorder._reset_realtime_analysis(stream_is_running=True, reset_vad=True, reset_streaming=True)
    assert queued == [(True, True)]
    recorder.stop_monitoring()
