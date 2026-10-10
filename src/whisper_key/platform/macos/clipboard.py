import logging

logger = logging.getLogger(__name__)


def _general_pasteboard():
    from AppKit import NSPasteboard
    return NSPasteboard.generalPasteboard()


def snapshot():
    try:
        items = []
        for item in _general_pasteboard().pasteboardItems() or []:
            item_data = {}
            for pasteboard_type in item.types():
                data = item.dataForType_(pasteboard_type)
                if data is not None:
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
