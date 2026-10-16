import threading
import time
from unittest.mock import MagicMock

import pytest

from whisper_key import clipboard_manager as clipboard_module


@pytest.fixture
def manager(monkeypatch):
    clipboard_contents = {"text": "original"}
    clipboard_contents["changes"] = 0
    platform_clipboard = MagicMock()
    platform_clipboard.snapshot.side_effect = lambda: [("snapshot", clipboard_contents["text"])]
    platform_clipboard.restore.side_effect = lambda snapshot: clipboard_contents.update(text=snapshot[0][1]) or True
    platform_clipboard.change_count.side_effect = lambda: clipboard_contents["changes"]
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


def test_restore_that_already_fired_does_not_run_early_after_a_new_paste(manager):
    manager, contents, platform_clipboard = manager
    manager._clipboard_paste("first")
    first_pending = manager._pending_restore
    with manager._clipboard_lock:
        first_pending.timer.cancel()
        waiting_restore = threading.Thread(target=manager._restore_clipboard, args=(first_pending,))
        waiting_restore.start()
        manager._clipboard_paste("second")
    waiting_restore.join(timeout=2)
    assert contents["text"] == "second"
    time.sleep(0.35)
    assert contents["text"] == "original"


def test_content_copied_during_the_delay_is_kept(manager):
    manager, contents, platform_clipboard = manager
    manager._clipboard_paste("dictated")
    contents["text"] = "copied by the user"
    contents["changes"] += 1
    time.sleep(0.35)
    platform_clipboard.restore.assert_not_called()
    assert contents["text"] == "copied by the user"


def test_paste_still_happens_when_the_clipboard_cannot_be_saved(manager):
    manager, contents, platform_clipboard = manager
    platform_clipboard.snapshot.side_effect = OSError("clipboard busy")
    assert manager._clipboard_paste("dictated")
    assert contents["text"] == "dictated"


def test_failed_restore_is_logged(manager, caplog):
    import logging
    manager, contents, platform_clipboard = manager
    platform_clipboard.restore.side_effect = lambda snapshot: False
    with caplog.at_level(logging.WARNING):
        manager._clipboard_paste("dictated")
        time.sleep(0.35)
    assert "could not be restored" in caplog.text


def test_delivery_restores_focus_target_before_paste(manager, monkeypatch):
    manager, _, _ = manager
    calls = []
    window_style = MagicMock()
    window_style.restore_focus_target.side_effect = lambda target: calls.append("restore") or True
    monkeypatch.setattr(clipboard_module, "window_style", window_style)
    monkeypatch.setattr(manager, "execute_delivery", lambda text: calls.append("paste") or True)
    assert manager.deliver_transcription("dictated", focus_target="target")
    assert calls == ["restore", "paste"]


def test_unreachable_focus_target_copies_instead_of_pasting(manager, monkeypatch):
    manager, contents, _ = manager
    window_style = MagicMock()
    window_style.restore_focus_target.return_value = False
    monkeypatch.setattr(clipboard_module, "window_style", window_style)
    execute_delivery = MagicMock()
    monkeypatch.setattr(manager, "execute_delivery", execute_delivery)
    manager.on_delivery_blocked = MagicMock()
    assert not manager.deliver_transcription("dictated", focus_target="target")
    execute_delivery.assert_not_called()
    assert contents["text"] == "dictated"
    manager.on_delivery_blocked.assert_called_once()


def test_disabled_restore_focus_skips_refocus(manager, monkeypatch):
    manager, _, _ = manager
    window_style = MagicMock()
    monkeypatch.setattr(clipboard_module, "window_style", window_style)
    monkeypatch.setattr(manager, "execute_delivery", lambda text: True)
    manager.update_restore_focus(False)
    assert manager.deliver_transcription("dictated", focus_target="target")
    window_style.restore_focus_target.assert_not_called()
