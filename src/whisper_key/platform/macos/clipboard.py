import logging

logger = logging.getLogger(__name__)

MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
SKIPPED_TYPE_PREFIXES = ("com.apple.NSFilePromiseItemMetaData", "com.apple.pasteboard.promised-file-")


def _general_pasteboard():
    from AppKit import NSPasteboard
    return NSPasteboard.generalPasteboard()


def _is_skipped_type(pasteboard_type) -> bool:
    return str(pasteboard_type).startswith(SKIPPED_TYPE_PREFIXES)


def change_count():
    return _general_pasteboard().changeCount()


def snapshot():
    try:
        items = []
        total_bytes = 0
        for item in _general_pasteboard().pasteboardItems() or []:
            item_data = {}
            for pasteboard_type in item.types():
                if _is_skipped_type(pasteboard_type):
                    continue
                data = item.dataForType_(pasteboard_type)
                if data is None:
                    continue
                total_bytes += data.length()
                if total_bytes > MAX_SNAPSHOT_BYTES:
                    logger.warning("Clipboard content is too large to keep, it is not restored after this paste")
                    return None
                item_data[pasteboard_type] = data
            if item_data:
                items.append(item_data)
        return items or None
    except Exception as e:
        logger.warning(f"Could not read the clipboard, it is not restored after this paste: {e}")
        return None


def restore(items) -> bool:
    if not items:
        return False
    try:
        from AppKit import NSPasteboardItem
        pasteboard = _general_pasteboard()
        pasteboard_items = []
        for item_data in items:
            pasteboard_item = NSPasteboardItem.alloc().init()
            for pasteboard_type, data in item_data.items():
                pasteboard_item.setData_forType_(data, pasteboard_type)
            pasteboard_items.append(pasteboard_item)
        pasteboard.clearContents()
        return bool(pasteboard.writeObjects_(pasteboard_items))
    except Exception as e:
        logger.warning(f"Could not restore the clipboard: {e}")
        return False
