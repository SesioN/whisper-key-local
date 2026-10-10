def _show(title: str, text: str, buttons: list, dismiss_result, parent=None):
    from ...themed_dialog import show_dialog
    return show_dialog(title, text, buttons, dismiss_result, parent)


def confirm(title: str, text: str, parent=None) -> bool:
    return _show(title, text, [("OK", True), ("Cancel", False)], False, parent)


def choose(title: str, text: str):
    return _show(title, text, [("Yes", True), ("No", False), ("Cancel", None)], None)


def show_error(title: str, text: str):
    _show(title, text, [("OK", None)], None)
