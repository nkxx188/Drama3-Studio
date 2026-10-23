"""CPU LRU of deterministic, whole-chunk reference-prefill states."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import threading

import torch

from .sequence_state import SequenceSnapshot


def prefix_execution_key(model):
    """Keep arithmetic/batching settings, excluding expert residency capacity."""
    manager = getattr(model, "_drama_low_vram_manager", None)
    backend = getattr(manager, "backend", None)
    return (
        type(backend).__name__, str(model.backbone.dtype),
        str(model.audio_decoder.codebook_embeddings.weight.dtype),
        getattr(backend, "decode_specialization", None),
        getattr(manager, "max_staged_experts", None),
        getattr(manager, "max_pair_rows", None),
        torch.__version__, torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32,
    )


class ReferencePrefixCache:
    def __init__(self, budget_bytes=256 * 1024**2, *, admission=None):
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError("Prefix cache budget must be a non-negative integer")
        self.budget_bytes = budget_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.stores = self.evictions = self.skips = 0
        self._lock = threading.RLock()
        self._suspended = 0
        self.admission = admission

    @property
    def enabled(self):
        with self._lock:
            return bool(self.budget_bytes and not self._suspended)

    @contextmanager
    def suspend(self):
        with self._lock:
            self._suspended += 1
        try:
            yield
        finally:
            with self._lock:
                self._suspended -= 1

    def key(self, model, prompt, chunk_size, execution_key=()):
        """First system-message EOS precedes all target/history turns.

        Only whole existing prefill chunks fit the cache. The following row is
        included in the identity because tower row p is labelled by frame p+1.
        A boundary at/after the system EOS is deliberately excluded.
        """
        with self._lock:
            if not self.enabled:
                return None, 0
        if chunk_size < 1 or prompt.device.type != "cpu" or prompt.dtype != torch.int64:
            raise ValueError("Prefix lookup requires CPU int64 prompt and positive chunk size")
        ends = torch.nonzero(prompt[:, 0] == model.config.eos_token_id).reshape(-1)
        if not len(ends):
            return None, 0
        # Exclude the EOS itself and require its label-lookahead to remain in
        # the system prefix. Text-only system prefixes normally fit no chunk.
        boundary = (int(ends[0]) // chunk_size) * chunk_size
        if not boundary or boundary >= len(prompt):
            return None, 0
        digest = hashlib.sha256(prompt[:boundary + 1].contiguous().numpy().tobytes()).hexdigest()
        return (id(model), chunk_size, boundary, tuple(execution_key), digest), boundary

    def get(self, key):
        with self._lock:
            value = self.entries.get(key)
            if value is None:
                self.misses += 1
            else:
                self.entries.move_to_end(key)
                self.hits += 1
            return value

    def peek(self, key):
        """Check residency without changing LRU order or hit/miss statistics."""
        with self._lock:
            return self.entries.get(key)

    def can_capture(self, model, boundary):
        with self._lock:
            if not self.enabled:
                return False
            size = SequenceSnapshot.estimate_bytes(model, boundary, boundary)
            if size > self.budget_bytes:
                return False
            admission = self.admission
        return admission is None or admission(size)

    def put(self, key, snapshot):
        with self._lock:
            if snapshot.nbytes > self.budget_bytes or self._suspended:
                self.skips += 1
                return False
            if key in self.entries:
                self.bytes -= self.entries.pop(key).nbytes
            while self.entries and self.bytes + snapshot.nbytes > self.budget_bytes:
                _, previous = self.entries.popitem(last=False)
                self.bytes -= previous.nbytes
                self.evictions += 1
            self.entries[key] = snapshot
            self.bytes += snapshot.nbytes
            self.stores += 1
            return True

    def capture(self, key, model, boundary):
        with self._lock:
            size = SequenceSnapshot.estimate_bytes(model, boundary, boundary)
            if size > self.budget_bytes or self._suspended:
                self.skips += 1
                return False
            # Free LRU entries before allocating, so the tensor budget also
            # bounds transient snapshot residency.
            while self.entries and self.bytes + size > self.budget_bytes:
                _, previous = self.entries.popitem(last=False)
                self.bytes -= previous.nbytes
                self.evictions += 1
                del previous
            if self.admission is not None and not self.admission(size):
                self.skips += 1
                return False
            try:
                snapshot = SequenceSnapshot.capture(
                    model, backbone_rows=boundary, tower_rows=boundary, max_bytes=self.budget_bytes)
            except MemoryError:
                self.skips += 1
                return False
            return self.put(key, snapshot)

    def clear(self):
        with self._lock:
            self.entries.clear()
            self.bytes = 0

    def report(self):
        with self._lock:
            return {
                "budget_bytes": self.budget_bytes, "resident_bytes": self.bytes,
                "entries": len(self.entries), "hits": self.hits, "misses": self.misses,
                "stores": self.stores, "evictions": self.evictions, "skips": self.skips,
            }
