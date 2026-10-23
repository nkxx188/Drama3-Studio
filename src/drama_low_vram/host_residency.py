"""Budgeted pageable RAM experts above an immutable mmap fallback.

The resident set is selected at request boundaries, not on a decode miss. This
follows Strata's profile-ranked resident complement: a bounded copy contains
selected experts and every other expert retains its original mapped fallback.
The caller serializes configuration against inference. Returned views are
read-only by contract, just as the underlying checkpoint mapping is.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .offload import EXPERT_BUFFER_NAMES, FP8_DTYPE, ExpertStore

ExpertKey = tuple[int, int]


@dataclass
class HostResidencyStatistics:
    """Logical source requests; mapped bytes are not measured physical disk I/O."""

    requests: int = 0
    expert_reads: int = 0
    resident_hits: int = 0
    mapped_fallback_reads: int = 0
    resident_read_bytes: int = 0
    mapped_fallback_bytes: int = 0
    observed_routes: int = 0
    rebuilds: int = 0
    copied_experts: int = 0
    copied_bytes: int = 0
    rebuild_seconds: float = 0.0
    allocation_fallbacks: int = 0
    reclaim_failures: int = 0

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _budget(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Host resident budget must be an integer >= 0")
    return value


class HostResidency:
    """One bounded FP8 arena, with per-expert source resolution.

    Sources must have identical expert geometry. ``configure`` may discard the
    old arena before allocating the replacement, so old and new allocations
    never coexist in this object. Callers must not retain views across that
    boundary. A failed allocation leaves the mapped fallback usable.

    Mapped source pages can temporarily coexist with their anonymous RAM copy.
    They remain reclaimable by the OS; ``reclaim_copied`` lets a platform layer
    discard only copied file ranges after a rebuild. This class never trims the
    whole process working set or claims that logical RAM bytes equal RSS.
    """

    def __init__(
        self,
        stores: Mapping[int, ExpertStore] | Sequence[ExpertStore],
        *,
        reclaim_copied: Callable[[tuple[ExpertKey, ...]], None] | None = None,
    ) -> None:
        if isinstance(stores, Mapping):
            self._stores = dict(stores)
        else:
            self._stores = dict(enumerate(stores))
        if not self._stores:
            raise ValueError("Host residency needs at least one expert layer")
        geometry = None
        for layer, store in self._stores.items():
            if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
                raise ValueError("Layer indices must be non-negative integers")
            store.validate()
            current = (store.hidden, store.intermediate)
            if geometry is not None and geometry != current:
                raise ValueError("One host arena requires uniform expert geometry")
            geometry = current
        self._layers = tuple(sorted(self._stores))
        first = self._stores[self._layers[0]]
        self.expert_bytes = first.nbytes // first.num_experts
        self.total_experts = sum(store.num_experts for store in self._stores.values())
        self.total_source_bytes = self.expert_bytes * self.total_experts
        self.stats = HostResidencyStatistics()
        self._observed: Counter[ExpertKey] = Counter()
        self._arena: dict[str, Tensor] = {}
        self._keys: tuple[ExpertKey, ...] = ()
        self._slots: dict[ExpertKey, int] = {}
        self._excluded: set[ExpertKey] = set()
        self._budget_bytes = 0
        self._closed = False
        self._reclaim_copied = reclaim_copied
        self.last_error: str | None = None

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    @property
    def resident_bytes(self) -> int:
        return len(self._keys) * self.expert_bytes

    @property
    def resident_keys(self) -> tuple[ExpertKey, ...]:
        return self._keys

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError("Host residency is closed")

    def _key(self, key: ExpertKey) -> ExpertKey:
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise ValueError("An expert key must be (layer, expert)")
        layer, expert = key
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer not in self._stores
            or isinstance(expert, bool)
            or not isinstance(expert, int)
            or not 0 <= expert < self._stores[layer].num_experts
        ):
            raise IndexError(f"Expert key outside the source bank: {key!r}")
        return layer, expert

    def ranked_keys(self, *, include_unseen: bool = True) -> tuple[ExpertKey, ...]:
        """Observed routes first, then unseen experts interleaved over layers."""
        self._open()
        observed = sorted(self._observed, key=lambda key: (-self._observed[key], key))
        if not include_unseen:
            return tuple(observed)
        result = list(observed)
        seen = set(observed)
        for expert in range(max(store.num_experts for store in self._stores.values())):
            for layer in self._layers:
                key = (layer, expert)
                if expert < self._stores[layer].num_experts and key not in seen:
                    result.append(key)
        return tuple(result)

    def configure(
        self,
        budget_bytes: int,
        ranked_keys: Iterable[ExpertKey] | None = None,
        *,
        excluded_keys: Iterable[ExpertKey] = (),
    ) -> dict[str, int | float | str | None]:
        """Build a stable ranked set fitting the budget; never promote on a miss.

        Exclusions are suitable only for experts guaranteed to remain elsewhere
        (for example a stable GPU tier). A changing GPU cache needs coordinated
        complement exchanges before its entries can be excluded safely.
        """
        self._open()
        budget = _budget(budget_bytes)
        excluded = {self._key(key) for key in excluded_keys}
        rank = self.ranked_keys() if ranked_keys is None else ranked_keys
        selected = []
        seen = set(excluded)
        capacity = min(budget // self.expert_bytes, self.total_experts)
        # Validate the complete rank before replacing a usable configuration.
        for key in rank:
            key = self._key(key)
            if key in seen:
                continue
            seen.add(key)
            if len(selected) < capacity:
                selected.append(key)
        keys = tuple(selected)
        self._budget_bytes = budget
        self._excluded = excluded
        if keys == self._keys:
            return self.as_dict()
        self._arena.clear()
        self._keys = ()
        self._slots.clear()
        self.last_error = None
        if not keys:
            return self.as_dict()
        started = time.perf_counter()
        self.stats.rebuilds += 1
        first = self._stores[self._layers[0]]
        try:
            arena = {
                name: torch.empty(
                    (len(keys), *getattr(first, name).shape[1:]),
                    dtype=getattr(first, name).dtype,
                    device="cpu",
                )
                for name in EXPERT_BUFFER_NAMES
            }
            for slot, (layer, expert) in enumerate(keys):
                store = self._stores[layer]
                for name in EXPERT_BUFFER_NAMES:
                    source = getattr(store, name)[expert]
                    target = arena[name][slot]
                    if source.dtype == FP8_DTYPE:
                        source, target = source.view(torch.uint8), target.view(torch.uint8)
                    target.copy_(source)
        except (MemoryError, RuntimeError) as exc:
            # Allocation or copying can fail under changing host pressure.
            # No partial arena is published; the original read-only bank stays.
            self.stats.allocation_fallbacks += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self.as_dict()
        finally:
            self.stats.rebuild_seconds += time.perf_counter() - started
        self._arena = arena
        self._keys = keys
        self._slots = {key: slot for slot, key in enumerate(keys)}
        self.stats.copied_experts += len(keys)
        self.stats.copied_bytes += self.resident_bytes
        if self._reclaim_copied is not None:
            try:
                self._reclaim_copied(keys)
            except Exception as exc:
                self.stats.reclaim_failures += 1
                self.last_error = f"Source page reclaim failed: {type(exc).__name__}: {exc}"
        return self.as_dict()

    def set_budget(self, budget_bytes: int) -> dict[str, int | float | str | None]:
        """Resize at a safe boundary, retaining existing priorities when possible."""
        self._open()
        rank = self._keys + self.ranked_keys()
        return self.configure(budget_bytes, rank, excluded_keys=self._excluded)

    def observe(self, layer: int, expert_ids: Tensor) -> None:
        """Record all router choices, including routes already served by GPU RAM."""
        self._open()
        if (
            not isinstance(expert_ids, Tensor)
            or expert_ids.device.type != "cpu"
            or expert_ids.dtype != torch.int64
            or expert_ids.ndim not in (1, 2)
            or expert_ids.numel() == 0
        ):
            raise ValueError("Observed expert IDs must be a non-empty int64 CPU vector or matrix")
        keys = [self._key((layer, expert)) for expert in expert_ids.reshape(-1).tolist()]
        self._observed.update(keys)
        self.stats.observed_routes += len(keys)

    def _record(self, keys: Sequence[ExpertKey], *, observe: bool = True) -> None:
        hits = sum(key in self._slots for key in keys)
        if observe:
            self._observed.update(keys)
            self.stats.observed_routes += len(keys)
        self.stats.requests += 1
        self.stats.expert_reads += len(keys)
        self.stats.resident_hits += hits
        self.stats.mapped_fallback_reads += len(keys) - hits
        self.stats.resident_read_bytes += hits * self.expert_bytes
        self.stats.mapped_fallback_bytes += (len(keys) - hits) * self.expert_bytes

    def get(self, layer: int, expert: int) -> dict[str, Tensor]:
        """Read-only single-expert views, preserving all official FP8 bytes."""
        self._open()
        key = self._key((layer, expert))
        self._record((key,))
        slot = self._slots.get(key)
        if slot is not None:
            return {name: self._arena[name][slot] for name in EXPERT_BUFFER_NAMES}
        store = self._stores[layer]
        return {name: getattr(store, name)[expert] for name in EXPERT_BUFFER_NAMES}

    def copy_to(
        self,
        layer: int,
        expert_ids: Tensor,
        targets: Mapping[str, Tensor],
        *,
        observe: bool = True,
    ) -> None:
        """Gather into staging buffers without touching mapped copies of RAM hits."""
        self._open()
        if (
            not isinstance(expert_ids, Tensor)
            or expert_ids.device.type != "cpu"
            or expert_ids.dtype != torch.int64
            or expert_ids.ndim != 1
            or expert_ids.numel() == 0
        ):
            raise ValueError("Expert IDs must be a non-empty one-dimensional int64 CPU tensor")
        ids = expert_ids.tolist()
        keys = [self._key((layer, expert)) for expert in ids]
        store = self._stores[layer]
        count = len(ids)
        for name in EXPERT_BUFFER_NAMES:
            target = targets.get(name)
            source = getattr(store, name)
            if (
                not isinstance(target, Tensor)
                or target.device.type != "cpu"
                or target.dtype != source.dtype
                or target.ndim != source.ndim
                or target.shape[0] < count
                or target.shape[1:] != source.shape[1:]
                # Packed staging leaves gaps between expert rows, while each
                # projection stays contiguous and non-overlapping.
                or not target[0].is_contiguous()
                or target.stride(0) < target[0].numel()
            ):
                raise ValueError(f"Invalid CPU staging target for {name}")
        self._record(keys, observe=observe)
        # Consecutive outputs from the same tier can be gathered directly into
        # their final output slice, without a temporary or a cold-bank overread.
        starts = [0]
        resident = [key in self._slots for key in keys]
        starts.extend(i for i in range(1, count) if resident[i] != resident[i - 1])
        starts.append(count)
        for start, end in zip(starts, starts[1:]):
            if resident[start]:
                indices = torch.tensor(
                    [self._slots[key] for key in keys[start:end]], dtype=torch.int64
                )
                sources = self._arena
            else:
                indices = expert_ids[start:end]
                sources = {name: getattr(store, name) for name in EXPERT_BUFFER_NAMES}
            for name in EXPERT_BUFFER_NAMES:
                source, target = sources[name], targets[name][start:end]
                if source.dtype == FP8_DTYPE:
                    source, target = source.view(torch.uint8), target.view(torch.uint8)
                torch.index_select(source, 0, indices, out=target)

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "budget_bytes": self._budget_bytes,
            "resident_bytes": self.resident_bytes,
            "resident_experts": len(self._keys),
            "source_bytes": self.total_source_bytes,
            "source_experts": self.total_experts,
            "expert_bytes": self.expert_bytes,
            "last_error": self.last_error,
            **self.stats.as_dict(),
        }

    def close(self) -> None:
        self._arena.clear()
        self._keys = ()
        self._slots.clear()
        self._observed.clear()
        self._excluded.clear()
        self._stores.clear()
        self._budget_bytes = 0
        self._closed = True


__all__ = ["ExpertKey", "HostResidency", "HostResidencyStatistics"]
