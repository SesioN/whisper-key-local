import ctypes
import logging
import struct
from ctypes import wintypes
from typing import Optional

RELATION_PROCESSOR_CORE = 0
ENTRY_HEADER = struct.Struct("<II")

logger = logging.getLogger(__name__)


def physical_core_count() -> Optional[int]:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        length = wintypes.DWORD(0)
        kernel32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, None, ctypes.byref(length))
        buffer = ctypes.create_string_buffer(length.value)
        if not kernel32.GetLogicalProcessorInformationEx(RELATION_PROCESSOR_CORE, buffer, ctypes.byref(length)):
            return None
        core_count = 0
        offset = 0
        while offset + ENTRY_HEADER.size <= length.value:
            relationship, entry_size = ENTRY_HEADER.unpack_from(buffer, offset)
            if entry_size == 0:
                break
            if relationship == RELATION_PROCESSOR_CORE:
                core_count += 1
            offset += entry_size
        return core_count or None
    except Exception as e:
        logger.debug(f"Could not count physical CPU cores: {e}")
        return None
