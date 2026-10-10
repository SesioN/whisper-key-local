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
