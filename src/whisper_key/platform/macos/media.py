import subprocess

MEDIA_APPS = ("Spotify", "Music")


def init_thread():
    pass


def _osascript(script: str) -> str:
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2)
    except subprocess.TimeoutExpired:
        return ""
    return result.stdout.strip()


def _is_app_running(app_name: str) -> bool:
    return _osascript(f'application "{app_name}" is running') == "true"


def is_output_muted() -> bool:
    return _osascript("output muted of (get volume settings)") == "true"


def set_output_muted(muted: bool):
    _osascript(f"set volume output muted {'true' if muted else 'false'}")


def _pause_app_if_playing(app_name: str) -> bool:
    if not _is_app_running(app_name):
        return False
    script = (
        f'tell application "{app_name}"\n'
        f'  if player state is playing then\n'
        f'    pause\n'
        f'    return "paused"\n'
        f'  end if\n'
        f'end tell'
    )
    return _osascript(script) == "paused"


def pause_playing_media() -> list:
    return [app_name for app_name in MEDIA_APPS if _pause_app_if_playing(app_name)]


def resume_media(paused_media: list):
    for app_name in paused_media:
        if _is_app_running(app_name):
            _osascript(f'tell application "{app_name}" to play')
