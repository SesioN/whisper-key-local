def is_supported() -> bool:
    return False


def is_blocked_by_system() -> bool:
    return False


def open_system_settings():
    pass


def is_enabled() -> bool:
    return False


def enable():
    raise NotImplementedError("Autostart is not supported on macOS yet")


def disable():
    pass


def refresh():
    return False
