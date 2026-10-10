import logging
import subprocess
from typing import Optional

SYSCTL_TIMEOUT_SECONDS = 2

logger = logging.getLogger(__name__)


def _sysctl_int(name: str) -> Optional[int]:
    try:
        result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=SYSCTL_TIMEOUT_SECONDS)
        return int(result.stdout.strip()) if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def physical_core_count() -> Optional[int]:
    count = _sysctl_int("hw.perflevel0.physicalcpu") or _sysctl_int("hw.physicalcpu")
    if count is None:
        logger.debug("Could not count physical CPU cores")
    return count
