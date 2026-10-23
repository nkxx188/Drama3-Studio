"""Monotonic WebUI request timing: loading and waiting never count as generation."""

from __future__ import annotations

import time
import threading


class JobClock:
    PHASES = ("generation", "model_load", "initialization", "waiting")

    def __init__(self, *, queued_seconds=0.0, clock=time.perf_counter):
        self._clock = clock
        self._lock = threading.RLock()
        self._at = clock()
        self._seconds = dict.fromkeys(self.PHASES, 0.0)
        self._stack = []
        self.phase = "initialization"
        self.queued_seconds = max(0.0, queued_seconds)

    def _advance(self):
        now = self._clock()
        if self.phase in self._seconds:
            self._seconds[self.phase] += max(0.0, now - self._at)
        self._at = now

    def switch(self, phase):
        with self._lock:
            if self.phase == "finished":
                return
            if phase not in self.PHASES:
                raise ValueError("Unknown job timing phase")
            self._advance()
            self.phase = phase

    def begin(self, phase):
        with self._lock:
            if self.phase == "finished":
                return
            self._stack.append((phase, self.phase))
            self.switch(phase)

    def end(self, phase):
        with self._lock:
            if self._stack and self._stack[-1][0] == phase:
                _, previous = self._stack.pop()
                self.switch(previous)

    def finish(self):
        with self._lock:
            self._advance()
            self.phase = "finished"
            self._stack.clear()

    def snapshot(self):
        with self._lock:
            self._advance()
            return {
                "schema": 1, "phase": self.phase,
                **{name + "_seconds": value for name, value in self._seconds.items()},
                "queue_seconds": self.queued_seconds,
                "total_seconds": self.queued_seconds + sum(self._seconds.values()),
                "measured_at": time.time(),
            }
