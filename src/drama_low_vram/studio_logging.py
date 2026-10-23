"""Concise Studio console messages with bounded, local diagnostic output."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
import sys
import threading
import time

_active = None
_ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def brief(value, limit=220):
    text = " ".join(_ansi.sub("", str(value)).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


class _FileHandler(RotatingFileHandler):
    def __init__(self, *args, console, **kwargs):
        self.console, self.warned = console, False
        super().__init__(*args, **kwargs)

    def handleError(self, record):
        # Logging errors must not recurse through redirected stderr or fail a job.
        if not self.warned:
            self.warned = True
            try:
                self.console.write("Warning: detailed logging is unavailable. Console status remains active.\n")
                self.console.flush()
            except (OSError, ValueError):
                pass


class _DiagnosticStream:
    encoding = "utf-8"
    errors = "backslashreplace"

    def __init__(self, logger, channel):
        self.logger, self.channel = logger, channel
        self.pending = {}
        self.lock = threading.RLock()
        self.closed = False

    def write(self, value):
        if self.closed or not value:
            return len(value)
        with self.lock:
            key = threading.get_ident()
            rows = (self.pending.pop(key, "") + value.replace("\r", "\n")).split("\n")
            tail = rows.pop()
            if len(tail) > 16384:
                rows.append(tail)
            elif tail:
                self.pending[key] = tail
            for row in rows:
                if row.strip():
                    self.logger.debug("[%s] %s", self.channel, _ansi.sub("", row))
        return len(value)

    def flush(self):
        with self.lock:
            for row in self.pending.values():
                if row.strip():
                    self.logger.debug("[%s] %s", self.channel, _ansi.sub("", row))
            self.pending.clear()

    def isatty(self):
        return False

    def writable(self):
        return not self.closed


class StudioLog:
    def __init__(self, root, console, max_bytes, backup_count):
        self.path = Path(root) / "diagnostics/studio.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.console, self.lock = console, threading.RLock()
        self.logger = logging.Logger("drama3.studio", logging.DEBUG)
        self.handler = _FileHandler(
            self.path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
            console=console)
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(self.handler)
        self.stdout = _DiagnosticStream(self.logger, "stdout")
        self.stderr = _DiagnosticStream(self.logger, "stderr")

    def status(self, text, *args):
        text = text % args if args else str(text)
        self.logger.info(text)
        with self.lock:
            try:
                self.console.write(time.strftime("[%H:%M:%S] ") + brief(text) + "\n")
                self.console.flush()
            except (OSError, ValueError):
                pass

    def close(self):
        self.stdout.flush()
        self.stderr.flush()
        self.stdout.closed = self.stderr.closed = True
        self.logger.removeHandler(self.handler)
        self.handler.close()


@contextmanager
def console_session(root, *, console=None, max_bytes=4 * 1024**2, backup_count=2):
    global _active
    if _active is not None:
        yield _active
        return
    log = StudioLog(root, console if console is not None else sys.stdout, max_bytes, backup_count)
    _active = log
    try:
        with redirect_stdout(log.stdout), redirect_stderr(log.stderr):
            try:
                yield log
            except BaseException:
                log.logger.debug("Studio session ended with an exception", exc_info=True)
                raise
    finally:
        log.close()
        _active = None


def status(text, *args):
    if _active is not None:
        _active.status(text, *args)


def detail(text, *args, **kwargs):
    if _active is not None:
        _active.logger.debug(text, *args, **kwargs)
