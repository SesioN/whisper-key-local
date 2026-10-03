from pathlib import Path
from PIL import Image

from ...utils import resolve_asset_path
from ...icon_effects import create_muted_icon

ASSETS_DIR = Path(resolve_asset_path("platform/windows/assets"))

def get_tray_icons() -> dict:
    idle_icon = Image.open(ASSETS_DIR / "tray_idle.png")
    return {
        "idle": idle_icon,
        "muted": create_muted_icon(idle_icon),
        "recording": Image.open(ASSETS_DIR / "tray_recording.png"),
        "processing": Image.open(ASSETS_DIR / "tray_processing.png"),
    }
