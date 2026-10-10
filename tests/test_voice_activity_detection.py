import numpy as np

from whisper_key.voice_activity_detection import ContinuousVoiceDetector, Hysteresis, VadEvent, VAD_HOP_DURATION_SEC


class ScriptedVad:
    def __init__(self, probabilities):
        self.probabilities = list(probabilities)

    def process(self, audio_chunk):
        return self.probabilities.pop(0), 0


def make_detector(probabilities, silence_timeout=1.0, speech_end_silence=0.1, min_speech=0.032):
    events = []
    detector = ContinuousVoiceDetector(
        ten_vad=ScriptedVad(probabilities),
        vad_onset_threshold=0.7,
        vad_offset_threshold=0.55,
        vad_silence_timeout_seconds=silence_timeout,
        frame_duration_sec=VAD_HOP_DURATION_SEC,
        min_speech_duration_seconds=min_speech,
        speech_end_silence_seconds=speech_end_silence,
    )
    detector._dispatch_event = events.append
    return detector, events


def frame():
    return np.zeros((256, 1), dtype=np.float32)


def test_hysteresis_holds_speech_until_offset_threshold():
    hysteresis = Hysteresis(high_threshold=0.7, low_threshold=0.5, frame_duration_sec=VAD_HOP_DURATION_SEC)
    assert [hysteresis.detect_speech(p) for p in (0.6, 0.8, 0.6, 0.4)] == [False, True, True, False]


def test_hysteresis_requires_minimum_speech_run():
    hysteresis = Hysteresis(high_threshold=0.7, low_threshold=0.5, frame_duration_sec=VAD_HOP_DURATION_SEC)
    assert not hysteresis.detect_speech_in_probabilities([0.9, 0.9, 0.1, 0.9, 0.9], 0.048)
    hysteresis = Hysteresis(high_threshold=0.7, low_threshold=0.5, frame_duration_sec=VAD_HOP_DURATION_SEC)
    assert hysteresis.detect_speech_in_probabilities([0.1, 0.9, 0.9, 0.9], 0.048)


def test_detector_emits_speech_start_then_speech_end():
    detector, events = make_detector([0.9] * 3 + [0.1] * 8)
    for _ in range(11):
        detector.process_chunk(frame())
    assert events == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]


def test_detector_emits_silence_timeout():
    frames_for_timeout = int(1.0 / VAD_HOP_DURATION_SEC)
    detector, events = make_detector([0.1] * frames_for_timeout)
    for _ in range(frames_for_timeout):
        detector.process_chunk(frame())
    assert events == [VadEvent.SILENCE_TIMEOUT]


def test_detector_reports_trailing_silence_duration():
    detector, _ = make_detector([0.9] * 3 + [0.1] * 5)
    for _ in range(8):
        detector.process_chunk(frame())
    assert abs(detector.get_silence_duration() - 5 * VAD_HOP_DURATION_SEC) < 1e-9


class CountingVad:
    instances = []

    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0

    def process(self, chunk):
        assert len(chunk) == 256
        probability = self.probabilities[min(self.calls, len(self.probabilities) - 1)]
        self.calls += 1
        return probability, 0


def install_counting_vad(monkeypatch, probabilities):
    from whisper_key import voice_activity_detection
    CountingVad.instances = []

    def factory():
        vad = CountingVad(probabilities)
        CountingVad.instances.append(vad)
        return vad

    monkeypatch.setattr(voice_activity_detection, "TenVad", factory)
    manager = voice_activity_detection.VadManager.__new__(voice_activity_detection.VadManager)
    manager.vad_precheck_enabled = True
    manager.vad_realtime_enabled = False
    manager.vad_precheck_onset_threshold = 0.7
    manager.vad_precheck_offset_threshold = 0.55
    manager.vad_min_speech_duration = 0.1
    manager.ten_vad = object()
    import logging
    manager.logger = logging.getLogger("test")
    return manager


def test_precheck_stops_at_first_speech_run(monkeypatch):
    manager = install_counting_vad(monkeypatch, [0.1] * 10 + [0.9] * 1000)
    assert manager.check_audio_for_speech(np.zeros(16000 * 60, dtype=np.float32))
    assert CountingVad.instances[0].calls == 16


def test_precheck_scans_everything_when_silent(monkeypatch):
    manager = install_counting_vad(monkeypatch, [0.1])
    assert not manager.check_audio_for_speech(np.zeros(16000, dtype=np.float32))
    assert CountingVad.instances[0].calls == 63


def test_precheck_uses_fresh_vad_state_each_time(monkeypatch):
    manager = install_counting_vad(monkeypatch, [0.1])
    manager.check_audio_for_speech(np.zeros(512, dtype=np.float32))
    manager.check_audio_for_speech(np.zeros(512, dtype=np.float32))
    assert len(CountingVad.instances) == 2


def test_recording_analysis_reports_speech_and_trailing_silence():
    detector, _ = make_detector([0.9] * 3 + [0.1] * 5)
    for _ in range(8):
        detector.process_chunk(frame())
    speech_detected, trailing_silence = detector.get_recording_analysis()
    assert speech_detected
    assert abs(trailing_silence - 5 * VAD_HOP_DURATION_SEC) < 1e-9
    detector.reset()
    assert detector.get_recording_analysis() == (False, 0.0)


def test_recording_analysis_reports_no_trailing_silence_without_speech():
    detector, _ = make_detector([0.1] * 8)
    for _ in range(8):
        detector.process_chunk(frame())
    assert detector.get_recording_analysis() == (False, 0.0)
