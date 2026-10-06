MIN_SIZE = 16
MAX_SIZE = 80
DEFAULT_SIZE = 80
NAMED_SIZES = (("Small", 44), ("Medium", 60), ("Big", 80))
SNAP_DISTANCE = 3


def clamp_size(size) -> float:
    if isinstance(size, bool) or not isinstance(size, (int, float)):
        return DEFAULT_SIZE
    return round(max(MIN_SIZE, min(MAX_SIZE, float(size))), 1)


def size_label(size: float) -> str:
    name = next((label for label, named_size in NAMED_SIZES if abs(size - named_size) < 0.05), None)
    return name or f"{size:.0f} px"
