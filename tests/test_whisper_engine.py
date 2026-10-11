import types

import numpy as np
import pytest

from whisper_key import whisper_engine
from whisper_key.model_registry import ModelRegistry


class FakeWhisperModel:
    created = []
    fail_sources = set()
    transcribe_error = None

    def __init__(self, source, device, compute_type, cpu_threads):
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
    monkeypatch.setattr(whisper_engine, "download_model", lambda source: source)
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

    def tracking_create(model_key, model_source=None):
        model_at_creation.append(engine.model)
        return original_create(model_key, model_source)

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


def test_forced_model_language_overrides_auto_detect(fake_model):
    registry = ModelRegistry({"distil": {"source": "distil", "language": "en"}})
    registry.get_load_source = lambda key: registry.get_source(key)
    engine = whisper_engine.WhisperEngine(model_key="distil", language="auto", model_registry=registry)
    engine.transcribe_audio(np.ones(16000, dtype=np.float32))
    assert engine.model.calls[-1]["language"] == "en"


def test_long_prompted_audio_is_chunked_so_every_chunk_gets_the_prompt(fake_model):
    engine = make_engine(initial_prompt="Glossary: Kubernetes")
    languages = iter(["de"] + ["en"] * 10)

    def transcribe(audio, **kwargs):
        engine.model.calls.append(kwargs)
        info = types.SimpleNamespace(language=next(languages), language_probability=0.9)
        return iter([types.SimpleNamespace(text=f" part{len(engine.model.calls)}")]), info

    engine.model.calls.clear()
    engine.model.transcribe = transcribe
    text = engine.transcribe_audio(np.ones(16000 * 70, dtype=np.float32))
    assert len(engine.model.calls) >= 3
    assert all(call["initial_prompt"] == "Glossary: Kubernetes" for call in engine.model.calls)
    assert [call["language"] for call in engine.model.calls] == [None] + ["de"] * (len(engine.model.calls) - 1)
    assert text.startswith("part1 part2 part3")


def test_long_audio_without_prompt_is_one_call(fake_model):
    engine = make_engine()
    engine.model.calls.clear()
    engine.transcribe_audio(np.ones(16000 * 70, dtype=np.float32))
    assert len(engine.model.calls) == 1


def test_cpu_threads_default_to_physical_cores(fake_model):
    engine = make_engine()
    assert engine.cpu_threads == 4
    assert make_engine(cpu_threads=3).cpu_threads == 3


def test_timestamps_are_skipped_only_for_single_window_audio(fake_model):
    engine = make_engine()
    seen = []
    original = engine.model.transcribe

    def transcribe(audio, **kwargs):
        seen.append((len(audio), kwargs["without_timestamps"]))
        return original(audio, **kwargs)

    engine.model.transcribe = transcribe
    engine.transcribe_audio(np.ones(16000 * 10, dtype=np.float32))
    engine.transcribe_audio(np.ones(16000 * 45, dtype=np.float32))
    assert seen == [(16000 * 10, True), (16000 * 45, False)]


def test_gpu_model_change_downloads_before_releasing_the_old_model(fake_model, monkeypatch):
    engine = make_engine(device="cuda")
    order = []
    monkeypatch.setattr(whisper_engine, "download_model", lambda source: order.append(("download", engine.model is not None)) or source)
    run_model_change(engine, "base")
    assert order == [("download", True)]


def test_failed_restore_reports_that_no_model_is_loaded(fake_model):
    engine = make_engine(device="cuda")
    fake_model.fail_sources = {"broken", "tiny"}
    messages = run_model_change(engine, "broken")
    assert engine.model is None
    assert "no model loaded" in messages[-1]


def test_transcription_keeps_using_the_model_it_started_with(fake_model):
    engine = make_engine(initial_prompt="Prompt")
    original_model = engine.model
    calls = []

    def transcribe(audio, **kwargs):
        calls.append(1)
        engine.model = None
        return iter([types.SimpleNamespace(text=" part")]), types.SimpleNamespace(language="en", language_probability=0.9)

    original_model.transcribe = transcribe
    assert engine.transcribe_audio(np.ones(16000 * 70, dtype=np.float32)).startswith("part")
    assert len(calls) >= 3


def test_uncertain_language_is_detected_per_chunk(fake_model):
    engine = make_engine(initial_prompt="Prompt")
    languages = []

    def transcribe(audio, **kwargs):
        languages.append(kwargs["language"])
        return iter([types.SimpleNamespace(text=" x")]), types.SimpleNamespace(language="de", language_probability=0.3)

    engine.model.transcribe = transcribe
    engine.transcribe_audio(np.ones(16000 * 70, dtype=np.float32))
    assert set(languages) == {None}
