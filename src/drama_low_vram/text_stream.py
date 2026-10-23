"""Bounded one-chunk-lookahead sessions for live text input.

The producer can push while a consumer runs. Audio is yielded after each whole
generated chunk; low-memory model parking remains the engine's responsibility.
This is not a claim of frame-level real-time synthesis.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from threading import Condition, Event

from .codec import concatenate_wavs
from .protocols import PreviousTake, validate_content
from .workflows import DramaWorkflows, load_take


class TextStreamSession:
    def __init__(self, workflows: DramaWorkflows, output_dir, *, max_pending=16,
                 max_chunk_chars=4000, options=None):
        if (isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 2
                or isinstance(max_chunk_chars, bool) or not isinstance(max_chunk_chars, int)
                or max_chunk_chars < 1):
            raise ValueError("Text stream requires a positive text limit and at least two queue slots")
        self.workflows, self.output_dir = workflows, Path(output_dir).resolve()
        self.max_pending, self.max_chunk_chars = max_pending, max_chunk_chars
        self.options = dict(options or {})
        forbidden = set(self.options) & {"mode", "text", "output", "lookahead",
                                        "previous_takes", "cancelled", "segment_ready"}
        if forbidden:
            raise ValueError("Session options cannot override stream-owned state")
        self._queue = deque()
        self._condition, self._cancelled = Condition(), Event()
        self._finished, self._consumer, self._running, self._done = False, False, False, False
        self.results = []
        self.status = "waiting"

    def push(self, text):
        validate_content(text, "stream input")
        if len(text) > self.max_chunk_chars:
            raise ValueError("Text stream chunk exceeds its character limit")
        with self._condition:
            if self._finished or self._done or self._cancelled.is_set():
                raise RuntimeError("Text stream no longer accepts input")
            if len(self._queue) >= self.max_pending:
                raise BufferError("Text stream queue is full; wait for the consumer")
            self._queue.append(text)
            self._condition.notify_all()

    def finish(self):
        with self._condition:
            self._finished = True
            self._condition.notify_all()

    def cancel(self):
        self._cancelled.set()
        with self._condition:
            self._queue.clear()
            self._condition.notify_all()

    @property
    def pending(self):
        with self._condition:
            return len(self._queue)

    def run(self, *, progress=None):
        """Yield completed audio in order; one active consumer per session."""
        with self._condition:
            if self._consumer:
                raise RuntimeError("Text stream already has a consumer")
            self._consumer = self._running = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        history, first = [], None
        try:
            while not self._cancelled.is_set():
                with self._condition:
                    self.status = "waiting"
                    self._condition.wait_for(
                        lambda: self._cancelled.is_set() or len(self._queue) >= 2
                        or self._finished)
                    if self._cancelled.is_set() or not self._queue:
                        break
                    current = self._queue.popleft()
                    lookahead = self._queue[0] if self._queue else ""
                index = len(self.results)
                self.status = "generating"
                recent = [first] if first is not None else []
                if history and history[-1] is not first:
                    recent.append(history[-1])
                result = self.workflows.run(
                    "stream", current, self.output_dir / f"chunk-{index + 1:05d}.wav",
                    lookahead=lookahead, previous_takes=recent,
                    cancelled=self._cancelled.is_set, progress=progress, **self.options)
                self.results.append(result)
                if result["status"] != "complete":
                    self.status = result["status"]
                    return
                if getattr(self.workflows.engine, "is_gguf", False):
                    take, _, _ = self.workflows.load_take(result["take_file"])
                else:
                    config = self.workflows.engine.model.config
                    take, _, _ = load_take(result["take_file"], config)
                history[:] = [take]
                if first is None:
                    first = take
                yield {
                    "index": index, "text": current, "lookahead": lookahead,
                    "path": result["output"], "take_file": result["take_file"],
                    "audio": result["audio"],
                }
            self.status = "cancelled" if self._cancelled.is_set() else "complete"
        finally:
            self._running, self._done = False, True
            if self._cancelled.is_set():
                self.status = "cancelled"

    def export(self, output):
        """Export exactly the yielded PCM chunks after successful completion."""
        if self.status != "complete" or not self.results:
            raise RuntimeError("Only a completed nonempty text stream can be exported")
        return concatenate_wavs([result["output"] for result in self.results], output)
