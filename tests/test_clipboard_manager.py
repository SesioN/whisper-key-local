import threading
import time
from unittest.mock import MagicMock

import pytest

from whisper_key import clipboard_manager as clipboard_module


@pytest.fixture
def manager(monkeypatch):
    clipboard_contents = {"text": "original"}
    platform_clipboard = MagicMock()
    platform_clipboard.snapshot.side_effect = lambda: [("snapshot", clipboard_contents["text"])]
    platform_clipboard.restore.side_effect = lambda snapshot: clipboard_contents.update(text=snapshot[0][1])
    monkeypatch.setattr(clipboard_module, "clipboard", platform_clipboard)
    monkeypatch.setattr(clipboard_module.pyperclip, "copy", lambda text: clipboard_contents.update(text=text))
    monkeypatch.setattr(clipboard_module.pyperclip, "paste", lambda: clipboard_contents["text"])
    manager = clipboard_module.ClipboardManager(
        auto_paste=True, delivery_method="paste", paste_hotkey="ctrl+v", paste_pre_paste_delay=0,
        paste_preserve_clipboard=True, paste_clipboard_restore_delay=0.2, copy_to_clipboard=False,
        type_auto_enter_delay=0, type_auto_enter_delay_per_100_chars=0, macos_key_simulation_delay=0)
    manager.delivery_method = "paste"
    return manager, clipboard_contents, platform_clipboard


def test_paste_returns_without_waiting_for_the_restore(manager):
    manager, contents, _ = manager
    start = time.monotonic()
    assert manager._clipboard_paste("dictated")
    assert time.monotonic() - start < 0.15
    assert contents["text"] == "dictated"
    time.sleep(0.35)
    assert contents["text"] == "original"


def test_quick_second_paste_restores_the_first_original(manager):
    manager, contents, platform_clipboard = manager
    manager._clipboard_paste("first")
    manager._clipboard_paste("second")
    time.sleep(0.35)
    assert contents["text"] == "original"
    assert platform_clipboard.snapshot.call_count == 1
    assert platform_clipboard.restore.call_count == 1


def test_empty_snapshot_is_never_restored(manager):
    manager, contents, platform_clipboard = manager
    platform_clipboard.snapshot.side_effect = lambda: None
    manager._clipboard_paste("dictated")
    time.sleep(0.35)
    platform_clipboard.restore.assert_not_called()
    assert contents["text"] == "dictated"
