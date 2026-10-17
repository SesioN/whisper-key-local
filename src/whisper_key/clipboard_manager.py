import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import pyperclip

from .platform import clipboard, keyboard, window_style
from .utils import parse_hotkey

MODIFIER_RELEASE_TIMEOUT_SECONDS = 1.0

@dataclass(eq=False)
class PendingRestore:
    original_clipboard: Any
    clipboard_change_count: Optional[int]
    timer: Optional[threading.Timer] = None


class ClipboardManager:
    def __init__(self, auto_paste, delivery_method, paste_hotkey,
                 paste_pre_paste_delay, paste_preserve_clipboard,
                 paste_clipboard_restore_delay,
                 copy_to_clipboard, type_auto_enter_delay,
                 type_auto_enter_delay_per_100_chars,
                 macos_key_simulation_delay, restore_focus=True,
                 restore_focus_settle_delay=0.2):
        self.logger = logging.getLogger(__name__)
        self.auto_paste = auto_paste
        self.delivery_method = keyboard.validate_delivery_method(delivery_method)
        self.paste_hotkey = paste_hotkey
        self.paste_keys = parse_hotkey(paste_hotkey)
        self.paste_pre_paste_delay = paste_pre_paste_delay
        self.paste_preserve_clipboard = paste_preserve_clipboard
        self.paste_clipboard_restore_delay = paste_clipboard_restore_delay
        self.copy_to_clipboard = copy_to_clipboard
        self.type_auto_enter_delay = type_auto_enter_delay
        self.type_auto_enter_delay_per_100_chars = type_auto_enter_delay_per_100_chars
        self.restore_focus = restore_focus
        self.restore_focus_settle_delay = restore_focus_settle_delay
        self.on_delivery_blocked = None
        self._clipboard_lock = threading.RLock()
        self._pending_restore = None
        keyboard.set_delay(macos_key_simulation_delay)
        if self.delivery_method == "paste":
            self._test_clipboard_access()
        self._print_status()

    def _test_clipboard_access(self):
        try:
            pyperclip.paste()
            self.logger.info("Clipboard access test successful")

        except Exception as e:
            self.logger.error(f"Clipboard access test failed: {e}")
            raise

    def _print_status(self):
        hotkey_display = self.paste_hotkey.upper()
        if self.auto_paste:
            if self.delivery_method == "type":
                print(f"   ✓ Auto-paste is ENABLED using direct text injection")
            else:
                print(f"   ✓ Auto-paste is ENABLED using clipboard paste ({hotkey_display})")
        else:
            print(f"   ✗ Auto-paste is DISABLED - paste manually with {hotkey_display}")
        if self.auto_paste and self.copy_to_clipboard:
            print("   ✓ Transcription is also kept on the clipboard")

    def copy_text(self, text: str) -> bool:
        if not text:
            return False

        try:
            self.logger.info(f"Copying text to clipboard ({len(text)} chars)")
            pyperclip.copy(text)
            return True

        except Exception as e:
            self.logger.error(f"Failed to copy text to clipboard: {e}")
            return False

    def get_clipboard_content(self) -> Optional[str]:
        try:
            clipboard_content = pyperclip.paste()

            if clipboard_content:
                return clipboard_content
            else:
                return None

        except Exception as e:
            self.logger.error(f"Failed to paste text from clipboard: {e}")
            return None

    def copy_with_notification(self, text: str) -> bool:
        if not text:
            return False

        success = self.copy_text(text)

        if success:
            print("   ✓ Copied to clipboard")
            print("   ✓ You can now paste with Ctrl+V in any application!")

        return success

    def clear_clipboard(self) -> bool:
        try:
            pyperclip.copy("")
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear clipboard: {e}")
            return False

    def _type_delivery(self, text: str) -> bool:
        try:
            keyboard.type_text(text)
            if self.copy_to_clipboard:
                pyperclip.copy(text)
            print(f"   ✓ Auto-pasted via text injection")
            return True
        except Exception as e:
            self.logger.error(f"Failed to inject text: {e}")
            return False

    def _take_original_clipboard(self):
        pending = self._pending_restore
        self._pending_restore = None
        if pending:
            pending.timer.cancel()
            return pending.original_clipboard
        try:
            return clipboard.snapshot()
        except Exception as e:
            self.logger.warning(f"Could not save the clipboard, it is not restored after this paste: {e}")
            return None

    def _schedule_restore(self, original_clipboard):
        pending = PendingRestore(original_clipboard, self._clipboard_change_count())
        pending.timer = threading.Timer(self.paste_clipboard_restore_delay, self._restore_clipboard, args=(pending,))
        pending.timer.daemon = True
        self._pending_restore = pending
        pending.timer.start()

    def _clipboard_change_count(self) -> Optional[int]:
        try:
            return clipboard.change_count()
        except Exception as e:
            self.logger.debug(f"Could not read the clipboard change count: {e}")
            return None

    def _restore_clipboard(self, pending: PendingRestore):
        with self._clipboard_lock:
            if self._pending_restore is not pending:
                return
            self._pending_restore = None
            if self._clipboard_change_count() != pending.clipboard_change_count:
                self.logger.info("Clipboard changed after the paste, keeping the new content instead of restoring")
                return
            try:
                if not clipboard.restore(pending.original_clipboard):
                    self.logger.warning("The previous clipboard content could not be restored")
            except Exception as e:
                self.logger.error(f"Failed to restore the clipboard: {e}")

    def _clipboard_paste(self, text: str) -> bool:
        with self._clipboard_lock:
            original_clipboard = None
            try:
                if self.paste_preserve_clipboard and not self.copy_to_clipboard:
                    original_clipboard = self._take_original_clipboard()

                if not self.copy_text(text):
                    return False

                time.sleep(self.paste_pre_paste_delay)
                keyboard.send_hotkey(*self.paste_keys)

                print("   ✓ Auto-pasted via key simulation")
                return True

            except Exception as e:
                self.logger.error(f"Failed to simulate paste keypress: {e}")
                return False

            finally:
                if original_clipboard:
                    self._schedule_restore(original_clipboard)

    def _copy_and_notify(self, text: str, copied_message: str, failed_message: str) -> bool:
        message = copied_message if self.copy_text(text) else failed_message
        print(f"   ⚠ {message}")
        if self.on_delivery_blocked:
            try:
                self.on_delivery_blocked(message)
            except Exception as e:
                self.logger.error(f"Failed to report blocked delivery: {e}")
        return False

    def _deliver_to_blocked_target(self, text: str) -> bool:
        self.logger.warning("Foreground window runs at a higher privilege level - auto-paste blocked")
        return self._copy_and_notify(
            text, f"Target window runs as administrator - press {self.paste_hotkey.upper()} to paste.",
            "Target window runs as administrator - auto-paste blocked and copying to the clipboard failed.")

    def _report_unreachable_target(self, text: str) -> bool:
        self.logger.warning("Original window could not be focused again - auto-paste skipped")
        return self._copy_and_notify(
            text, f"The original window is no longer available - press {self.paste_hotkey.upper()} to paste.",
            "The original window is no longer available and copying to the clipboard failed.")

    def _focus_target_ready(self, focus_target) -> bool:
        if not self.restore_focus or focus_target is None:
            return True
        try:
            return window_style.restore_focus_target(focus_target, settle_seconds=self.restore_focus_settle_delay)
        except Exception as e:
            self.logger.error(f"Failed to restore focus: {e}")
            return False

    def _wait_for_modifiers_released(self):
        started_at = time.monotonic()
        released = keyboard.wait_for_modifiers_released(MODIFIER_RELEASE_TIMEOUT_SECONDS)
        waited_seconds = time.monotonic() - started_at
        if not released:
            self.logger.warning(f"Modifier keys still held after {waited_seconds:.2f}s - pasting anyway")
        elif waited_seconds >= 0.01:
            self.logger.info(f"Waited {waited_seconds:.2f}s for modifier keys to be released")

    def _focus_target_still_active(self, focus_target) -> bool:
        if not self.restore_focus or focus_target is None:
            return True
        return window_style.is_focus_target_active(focus_target)

    def execute_delivery(self, text: str) -> bool:
        if keyboard.is_foreground_input_blocked():
            return self._deliver_to_blocked_target(text)
        if self.delivery_method == "type":
            return self._type_delivery(text)
        else:
            return self._clipboard_paste(text)

    def send_enter_key(self) -> bool:
        try:
            self.logger.info("Sending ENTER key to active application")
            keyboard.send_key('enter')
            print("   ✓ Text submitted with ENTER!")

            return True

        except Exception as e:
            self.logger.error(f"Failed to send ENTER key: {e}")
            return False

    def deliver_transcription(self,
                              transcribed_text: str,
                              use_auto_enter: bool = False,
                              focus_target=None) -> bool:

        try:
            if self.auto_paste:
                self._wait_for_modifiers_released()
                if not self._focus_target_ready(focus_target):
                    return self._report_unreachable_target(transcribed_text)
                self.logger.info(f"Delivering to {window_style.describe_foreground()} "
                                 f"(recording started in {window_style.describe_focus_target(focus_target)})")
                success = self.execute_delivery(transcribed_text)
                if success and use_auto_enter:
                    if self.delivery_method == "type":
                        delay = self.type_auto_enter_delay + len(transcribed_text) / 100 * self.type_auto_enter_delay_per_100_chars
                        if delay > 0:
                            time.sleep(delay)
                    if self._focus_target_still_active(focus_target):
                        success = self.send_enter_key()
                    else:
                        self.logger.warning("Focus left the original window - ENTER not sent")
                        print("   ⚠ Focus left the original window - ENTER was not sent")
            else:
                print("📋 Copying to clipboard...")
                success = self.copy_with_notification(transcribed_text)

            return success

        except Exception as e:
            self.logger.error(f"Delivery workflow failed: {e}")
            return False

    def update_auto_paste(self, enabled: bool):
        self.auto_paste = enabled
        self._print_status()

    def update_restore_focus(self, enabled: bool):
        self.restore_focus = enabled

    def update_copy_to_clipboard(self, enabled: bool):
        self.copy_to_clipboard = enabled
        self._print_status()
