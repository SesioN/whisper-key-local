import threading

from whisper_key.streaming_manager import StreamingManager


def test_background_initialize_notifies_when_ready(monkeypatch):
    manager = StreamingManager(streaming_enabled=True, streaming_model="tiny")
    monkeypatch.setattr(manager, "_load_model", lambda: True)
    ready = threading.Event()
    manager.start_background_initialize(on_ready=ready.set)
    assert ready.wait(2)


def test_background_initialize_skipped_when_disabled():
    manager = StreamingManager(streaming_enabled=False)
    called = []
    manager.start_background_initialize(on_ready=lambda: called.append(True))
    assert called == []
