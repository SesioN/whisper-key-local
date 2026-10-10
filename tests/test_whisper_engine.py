import types

import numpy as np
import pytest

from whisper_key import whisper_engine
from whisper_key.model_registry import ModelRegistry


class FakeWhisperModel:
    created = []
    fail_sources = set()
    transcribe_error = None

    def __init__(self, source, device, compute_type):
        if source in self.fail_sources:
            raise RuntimeError(f"cannot load {source}")
        self.source = source
        self.calls = []
        FakeWhisperModel.created.append(self)

    def transcribe(self, audio, **kwargs):
        if FakeWhisperModel.transcribe_error and len(audio) > 16000:
            raise FakeWhisperModel.transcribe_error
        self.calls.append(kwargs)
        segment = types.SimpleNamespace(text=" hello")
        return iter([segment]), types.SimpleNamespace(language="en", language_probability=1.0)


@pytest.fixture
def fake_model(monkeypatch):
    FakeWhisperModel.created = []
    FakeWhisperModel.fail_sources = set()
    FakeWhisperModel.transcribe_error = None
    monkeypatch.setattr(whisper_engine, "WhisperModel", FakeWhisperModel)
    return FakeWhisperModel


def make_engine(device="cpu", model_key="tiny", **kwargs):
    registry = ModelRegistry({"tiny": {}, "base": {}, "broken": {"source": "broken"}})
    registry.get_load_source = lambda key: registry.get_source(key)
    return whisper_engine.WhisperEngine(model_key=model_key, device=device, model_registry=registry, **kwargs)


def run_model_change(engine, model_key):
    messages = []
    engine.change_model(model_key, messages.append)
    engine._loading_thread.join(timeout=5) if engine._loading_thread else None
    return messages


def test_gpu_model_change_releases_old_model_before_loading(fake_model, monkeypatch):
    engine = make_engine(device="cuda")
    old_model = engine.model
    model_at_creation = []
    original_create = engine._create_model

    def tracking_create(model_key):
        model_at_creation.append(engine.model)
        return original_create(model_key)

    monkeypatch.setattr(engine, "_create_model", tracking_create)
    messages = run_model_change(engine, "base")
    assert model_at_creation == [None]
    assert engine.model is not old_model and engine.model_key == "base"
    assert messages[-1] == "Model ready!"


def test_failed_gpu_model_change_restores_previous_model(fake_model):
    engine = make_engine(device="cuda")
    fake_model.fail_sources = {"broken"}
    messages = run_model_change(engine, "broken")
    assert engine.model is not None and engine.model_key == "tiny"
    assert messages[-1].startswith("Failed to load model")


def test_cpu_model_change_keeps_old_model_until_new_one_is_ready(fake_model):
    engine = make_engine(device="cpu")
    old_model = engine.model
    fake_model.fail_sources = {"broken"}
    run_model_change(engine, "broken")
    assert engine.model is old_model and engine.model_key == "tiny"


def test_transcription_errors_propagate(fake_model):
    engine = make_engine()
    fake_model.transcribe_error = RuntimeError("CUDA failed")
    with pytest.raises(RuntimeError):
        engine.transcribe_audio(np.ones(32000, dtype=np.float32))
