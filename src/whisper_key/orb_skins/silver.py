from .blob import BlobSkin


class SilverSkin(BlobSkin):

    LABEL = "Silver orb"

    COLOR_CORE = (238, 244, 250)
    COLOR_MID = (176, 188, 201)
    COLOR_EDGE = (140, 152, 168)

    SHIMMER_STRENGTH = 0.30
    SHIMMER_SHARPNESS = 3.0
