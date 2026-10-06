from .aurora import AuroraSkin
from .base import OrbSkin
from .blackhole import BlackHoleSkin
from .blob import BlobSkin
from .chrome import ChromeSkin
from .glass import GlassSkin
from .nebula import NebulaSkin
from .gold import GoldSkin
from .silver import SilverSkin

SKINS = {
    "gold": GoldSkin,
    "silver": SilverSkin,
    "chrome": ChromeSkin,
    "glass": GlassSkin,
    "aurora": AuroraSkin,
    "blackhole": BlackHoleSkin,
    "nebula": NebulaSkin,
}

DEFAULT_SKIN = "nebula"


def get_skin(name: str):
    return SKINS.get(name, SKINS[DEFAULT_SKIN])


__all__ = ["OrbSkin", "BlobSkin", "SKINS", "DEFAULT_SKIN", "get_skin",
           "GoldSkin", "SilverSkin", "ChromeSkin", "GlassSkin", "AuroraSkin",
           "BlackHoleSkin", "NebulaSkin"]
