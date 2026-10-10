import subprocess

MEDIA_APPS = ("Spotify", "Music")


def init_thread():
    pass


def _osascript(script: str) -> str:
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
    return result.stdout.strip()


def is_output_muted() -> bool:
    return _osascript("output muted of (get volume settings)") == "true"


def set_output_muted(muted: bool):
    _osascript(f"set volume output muted {'true' if muted else 'false'}")


def _pause_app_if_playing(app_name: str) -> bool:
    script = (
        f'if application "{app_name}" is running then\n'
        f'  tell application "{app_name}"\n'
        f'    if player state is playing then\n'
        f'      pause\n'
        f'      return "paused"\n'
        f'    end if\n'
        f'  end tell\n'
        f'end if'
    )
    return _osascript(script) == "paused"


def pause_playing_media() -> list:
    return [app_name for app_name in MEDIA_APPS if _pause_app_if_playing(app_name)]


def resume_media(paused_media: list):
    for app_name in paused_media:
        _osascript(f'tell application "{app_name}" to play')
