def is_supported() -> bool:
    return False


def is_enabled() -> bool:
    return False


def enable():
    raise NotImplementedError("Autostart is not supported on macOS yet")


def disable():
    pass


def refresh():
    return False
