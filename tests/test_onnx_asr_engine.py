import logging
import threading

import numpy as np

from whisper_key.model_registry import ModelRegistry
from whisper_key.onnx_asr_engine import OnnxAsrEngine, SAMPLE_RATE


class RecordingModel:
    def __init__(self):
        self.calls = []

    def recognize(self, audio, sample_rate, **kwargs):
        self.calls.append((len(audio), kwargs))
        return f"part{len(self.calls)}"


def make_engine(model_key, models, language=None):
    engine = OnnxAsrEngine.__new__(OnnxAsrEngine)
    engine.model_key = model_key
    engine.language = language
    engine.registry = ModelRegistry(models)
    engine.vad_manager = None
    engine.logger = logging.getLogger("test")
    engine._model_lock = threading.Lock()
    engine.model = RecordingModel()
    engine._model_type = engine.registry.get_model(model_key).onnx_model_type
    engine._max_chunk_seconds = engine._chunk_seconds(model_key)
    return engine


PARAKEET = {"parakeet": {"engine": "onnx_asr", "onnx_model_type": "nemo-conformer-tdt"}}
QWEN = {"qwen": {"engine": "onnx_asr", "onnx_model_type": "speech-llm"},
        "qwen-int8": {"engine": "onnx_asr", "onnx_model_type": "speech-llm", "max_chunk_seconds": 14}}


def test_nemo_models_are_chunked_for_long_recordings():
    engine = make_engine("parakeet", PARAKEET)
    assert engine._max_chunk_seconds == 60
    text = engine._recognize(np.ones(SAMPLE_RATE * 150, dtype=np.float32))
    assert len(engine.model.calls) >= 3
    assert sum(length for length, _ in engine.model.calls) == SAMPLE_RATE * 150
    assert all(length <= SAMPLE_RATE * 60 for length, _ in engine.model.calls)
    assert all(kwargs == {} for _, kwargs in engine.model.calls)
    assert text.startswith("part1 part2 part3")


def test_speech_llm_chunk_limits_and_language():
    assert make_engine("qwen", QWEN)._max_chunk_seconds == 28
    engine = make_engine("qwen-int8", QWEN, language="de")
    assert engine._max_chunk_seconds == 14
    engine._recognize(np.ones(SAMPLE_RATE * 10, dtype=np.float32))
    assert engine.model.calls[0][1] == {"language": "German"}
