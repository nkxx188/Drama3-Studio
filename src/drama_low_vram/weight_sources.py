"""Immutable checkpoint views for CPU embedding and temporary weight parking.

Only tensors supplied by a checkpoint loader are registered here. Mutable
sequence state and runtime buffers never acquire a file-backed substitute.
Consumers may read or copy these tensors, but must not modify their storage.
"""

from __future__ import annotations

import torch


def reclaimable_file_bytes(tensors, needed, *, max_pages=262_144):
    """Count actual resident, unlocked 4KiB pages without reading their payload.

    Call only for verified read-only file-backed tensors. The capped Windows
    query provides a conservative credit: unsampled or nonresident pages count
    as zero, and a page shared by adjacent views is counted once.
    """
    import os
    if os.name != "nt" or needed <= 0:
        return 0
    import ctypes
    from ctypes import wintypes

    class Page(ctypes.Structure):
        _fields_ = [("address", ctypes.c_void_p), ("attributes", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    query = ctypes.WinDLL("psapi", use_last_error=True).QueryWorkingSetEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    handle = kernel.GetCurrentProcess()
    pages = (Page * 8192)()
    seen = set()
    count = resident = examined = 0

    def flush():
        nonlocal resident, examined, count
        if not count:
            return True
        if not query(handle, pages, count * ctypes.sizeof(Page)):
            return False
        resident += sum(
            4096 for index in range(count)
            if pages[index].attributes & 1 and not pages[index].attributes & (1 << 22)
        )
        examined += count
        count = 0
        return True

    for tensor in tensors:
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            continue
        start = (tensor.data_ptr() + 4095) // 4096 * 4096
        end = tensor.data_ptr() + tensor.numel() * tensor.element_size()
        for address in range(start, end - 4095, 4096):
            if address in seen:
                continue
            seen.add(address)
            pages[count].address = address
            pages[count].attributes = 0
            count += 1
            if count == len(pages) or examined + count >= max_pages:
                if not flush():
                    return 0
                if resident >= needed or examined >= max_pages:
                    return resident
    return resident if flush() else 0


class ImmutableWeightSources:
    def __init__(self):
        self._values = {}
        self.cpu_embedding_bytes = 0

    def remember(self, root, name, source):
        """Retain an exact CPU source only when no dtype conversion is needed."""
        path, _, leaf = name.rpartition(".")
        module = root.get_submodule(path) if path else root
        if leaf in module._parameters:
            kind = "parameter"
            target = module._parameters[leaf]
        elif leaf in module._buffers:
            kind = "buffer"
            target = module._buffers[leaf]
        else:
            raise ValueError(f"Checkpoint source has no target: {name}")
        if target is None or target.shape != source.shape:
            raise ValueError(f"Checkpoint source shape mismatch: {name}")
        if source.device.type != "cpu" or source.dtype != target.dtype:
            return False
        value = (
            torch.nn.Parameter(source.detach(), requires_grad=False)
            if kind == "parameter" else source.detach()
        )
        self._values[(id(module), kind, leaf)] = value
        return True

    def get(self, module, kind, name):
        return self._values.get((id(module), kind, name))

    def is_source(self, key, value):
        return self._values.get(key) is value

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in self._values.values())

    def as_dict(self):
        return {
            "tensors": len(self._values), "mapped_bytes": self.nbytes,
            "cpu_embedding_bytes": self.cpu_embedding_bytes,
            "parking_source_bytes": self.nbytes - self.cpu_embedding_bytes,
        }

    def clear(self):
        self._values.clear()
        self.cpu_embedding_bytes = 0
