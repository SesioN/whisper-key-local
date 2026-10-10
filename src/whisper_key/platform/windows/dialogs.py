def _show(title: str, text: str, buttons: list, dismiss_result):
    from ...themed_dialog import show_dialog
    return show_dialog(title, text, buttons, dismiss_result)


def confirm(title: str, text: str) -> bool:
    return _show(title, text, [("OK", True), ("Cancel", False)], False)


def choose(title: str, text: str):
    return _show(title, text, [("Yes", True), ("No", False), ("Cancel", None)], None)


def show_error(title: str, text: str):
    _show(title, text, [("OK", None)], None)
