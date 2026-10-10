import winreg

PERSONALIZE_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
DWM_KEY = r"Software\Microsoft\Windows\DWM"
DEFAULT_ACCENT_ABGR = 0xFFC06700


def _read_user_dword(path: str, name: str, default: int) -> int:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            value, _ = winreg.QueryValueEx(key, name)
            return int(value)
    except OSError:
        return default


def read_system_appearance():
    apps_light = _read_user_dword(PERSONALIZE_KEY, "AppsUseLightTheme", 1)
    accent_abgr = _read_user_dword(DWM_KEY, "AccentColor", DEFAULT_ACCENT_ABGR)
    accent_rgb = (accent_abgr & 0xFF, (accent_abgr >> 8) & 0xFF, (accent_abgr >> 16) & 0xFF)
    return bool(apps_light), accent_rgb
