import logging
import struct
import time

import win32clipboard
import win32con

logger = logging.getLogger(__name__)

OPEN_ATTEMPTS = 10
OPEN_RETRY_SECONDS = 0.02
MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
DROPFILES_HEADER_SIZE = 20
SYNTHESIZED_OR_HANDLE_FORMATS = {
    win32con.CF_TEXT, win32con.CF_OEMTEXT, win32con.CF_LOCALE, win32con.CF_BITMAP, win32con.CF_METAFILEPICT,
    win32con.CF_PALETTE, win32con.CF_ENHMETAFILE, win32con.CF_OWNERDISPLAY, win32con.CF_DSPBITMAP,
    win32con.CF_DSPMETAFILEPICT, win32con.CF_DSPENHMETAFILE, win32con.CF_DSPTEXT,
}


def _open_clipboard() -> bool:
    for _ in range(OPEN_ATTEMPTS):
        try:
            win32clipboard.OpenClipboard()
            return True
        except Exception:
            time.sleep(OPEN_RETRY_SECONDS)
    return False


def _dropfiles_bytes(file_names) -> bytes:
    paths = "\0".join(file_names) + "\0\0"
    return struct.pack("<IiiII", DROPFILES_HEADER_SIZE, 0, 0, 0, 1) + paths.encode("utf-16-le")


def _format_bytes(clipboard_format: int, data) -> bytes:
    if clipboard_format == win32con.CF_HDROP:
        return _dropfiles_bytes(data)
    if isinstance(data, str):
        return (data + "\0").encode("utf-16-le")
    return bytes(data)


def snapshot():
    if not _open_clipboard():
        logger.warning("Clipboard is busy, its content is not restored after this paste")
        return None
    formats = []
    total_bytes = 0
    try:
        clipboard_format = win32clipboard.EnumClipboardFormats(0)
        while clipboard_format:
            if clipboard_format not in SYNTHESIZED_OR_HANDLE_FORMATS:
                try:
                    data = _format_bytes(clipboard_format, win32clipboard.GetClipboardData(clipboard_format))
                    total_bytes += len(data)
                    formats.append((clipboard_format, data))
                except Exception as e:
                    logger.debug(f"Skipping clipboard format {clipboard_format}: {e}")
            if total_bytes > MAX_SNAPSHOT_BYTES:
                logger.warning("Clipboard content is too large to keep, it is not restored after this paste")
                return None
            clipboard_format = win32clipboard.EnumClipboardFormats(clipboard_format)
    finally:
        win32clipboard.CloseClipboard()
    return formats or None


def restore(formats) -> bool:
    if not formats or not _open_clipboard():
        return False
    try:
        win32clipboard.EmptyClipboard()
        for clipboard_format, data in formats:
            try:
                win32clipboard.SetClipboardData(clipboard_format, data)
            except Exception as e:
                logger.debug(f"Could not restore clipboard format {clipboard_format}: {e}")
        return True
    finally:
        win32clipboard.CloseClipboard()
