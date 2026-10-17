def apply_window_chrome(tk_window_id: int, dark: bool, border_rgb: tuple):
    pass


def center_window(tk_window_id: int):
    pass


def prevent_focus_steal(tk_window_id: int):
    pass


def is_point_on_screen(x: int, y: int) -> bool:
    return True


def get_foreground_window() -> int:
    return 0


def give_back_foreground(previous_window: int, tk_window_ids: list[int]):
    pass


def capture_focus_target():
    return None


def is_focus_target_active(target) -> bool:
    return True


def describe_focus_target(target) -> str:
    return "none"


def describe_foreground() -> str:
    return "unknown"


def restore_focus_target(target, settle_seconds: float = 0.2, timeout_seconds: float = 0.5) -> bool:
    return True
