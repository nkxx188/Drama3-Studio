"""Windows host memory snapshot without loading a compute runtime."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class WindowsMemoryStatus:
    ram_total_bytes: int
    ram_available_bytes: int
    commit_limit_bytes: int
    commit_available_bytes: int


def read_windows_memory_status() -> WindowsMemoryStatus | None:
    if os.name != "nt":
        return None

    from ctypes import wintypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", wintypes.ULARGE_INTEGER),
            ("ullAvailPhys", wintypes.ULARGE_INTEGER),
            ("ullTotalPageFile", wintypes.ULARGE_INTEGER),
            ("ullAvailPageFile", wintypes.ULARGE_INTEGER),
            ("ullTotalVirtual", wintypes.ULARGE_INTEGER),
            ("ullAvailVirtual", wintypes.ULARGE_INTEGER),
            ("ullAvailExtendedVirtual", wintypes.ULARGE_INTEGER),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
    except (AttributeError, OSError):
        return None

    # The PageFile fields measure commit capacity, not free disk space.
    return WindowsMemoryStatus(status.ullTotalPhys, status.ullAvailPhys,
                               status.ullTotalPageFile, status.ullAvailPageFile)
