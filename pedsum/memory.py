"""Best-effort memory limit for every pedsum subcommand (ADR 0004 §3).

A daemon thread samples the process RSS about once a second. When it passes
the limit, the thread runs the registered breach callbacks, logs the running
phase, and ends the process with exit code 3. Native allocation continues
between samples and while the callbacks run, so this narrows the window for
the kernel OOM killer; it does not close it.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from pedsum.base import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

EXIT_MEMORY_LIMIT = 3

#: Share of the measured headroom the default limit allows; the rest absorbs
#: what native code allocates between samples.
DEFAULT_LIMIT_FRACTION = 0.8

GiB = 2**30

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_SIZE_UNITS = {"": 1, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}


def parse_size(text: str) -> int:
    """Parse ``500M``, ``12G`` or a plain byte count into bytes (binary units); ``0`` disables."""
    number = text.strip().upper().rstrip("B")
    unit = number[-1:] if number[-1:] in _SIZE_UNITS else ""
    try:
        size = float(number.removesuffix(unit)) * _SIZE_UNITS[unit]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (use e.g. 500M, 12G, or 0 to disable)") from None
    if size < 0:
        raise argparse.ArgumentTypeError(f"size must not be negative: {text!r}")
    return int(size)


def read_rss_bytes() -> int:
    """Resident set size of this process in bytes, from ``/proc/self/statm``."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * _PAGE_SIZE


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _mem_available(proc_root: Path) -> int | None:
    try:
        lines = (proc_root / "meminfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _cgroup_headrooms(proc_root: Path, cgroup_root: Path) -> list[int]:
    """``memory.max - memory.current`` for this process's cgroup and every ancestor that sets a limit.

    cgroup v2 limits are hierarchical, so a parent scope's ``memory.max`` binds
    even when the leaf reads ``max``.
    """
    try:
        lines = (proc_root / "self" / "cgroup").read_text().splitlines()
    except OSError:
        return []
    rel = next((line[3:] for line in lines if line.startswith("0::")), None)
    if rel is None:
        return []
    leaf = cgroup_root / rel.lstrip("/")
    headrooms = []
    for level in (leaf, *leaf.parents):
        limit = _read_int(level / "memory.max")
        current = _read_int(level / "memory.current")
        if limit is not None and current is not None:
            headrooms.append(max(limit - current, 0))
        if level == cgroup_root:
            break
    return headrooms


def memory_headroom_bytes(
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int | None:
    """The smallest of host ``MemAvailable`` and every cgroup headroom; None when nothing is readable."""
    sources = _cgroup_headrooms(proc_root, cgroup_root)
    available = _mem_available(proc_root)
    if available is not None:
        sources.append(available)
    return min(sources) if sources else None


def resolve_limit(max_memory: int | None) -> int | None:
    """The limit in bytes for ``--max-memory`` (None = default, 0 = off); None means no watchdog."""
    if max_memory == 0:
        logger.info("memory limit off (--max-memory 0)")
        return None
    if max_memory is not None:
        logger.info("memory limit %.1f GiB (--max-memory)", max_memory / GiB)
        return max_memory
    headroom = memory_headroom_bytes()
    if headroom is None:
        logger.info("memory limit inactive: could not read available memory from /proc or /sys/fs/cgroup")
        return None
    limit = int(DEFAULT_LIMIT_FRACTION * headroom)
    logger.info(
        "memory limit %.1f GiB (%d%% of %.1f GiB available; --max-memory to change)",
        limit / GiB,
        round(DEFAULT_LIMIT_FRACTION * 100),
        headroom / GiB,
    )
    return limit


class MemoryWatchdog:
    """Stop the process with exit code 3 once its RSS passes ``limit_bytes``.

    A ``limit_bytes`` of None makes an inactive watchdog: no thread starts,
    callbacks never run, and ``disarm()`` only takes the lock.

    Final publication and the breach path share one lock. The main thread
    publishes inside ``disarm()``; a breach that arrives after it only logs a
    warning. A breach that takes the lock first runs the callbacks and exits,
    so exactly one of the two writes the output.
    """

    def __init__(
        self,
        limit_bytes: int | None,
        *,
        poll_s: float = 1.0,
        rss_reader: Callable[[], int] = read_rss_bytes,
        exit_fn: Callable[[int], object] = os._exit,
        phase: Callable[[], str | None] = lambda: None,
    ) -> None:
        """Set the limit; ``phase`` names the running step in log lines, the other seams serve tests."""
        self.limit_bytes = limit_bytes
        self._poll_s = poll_s
        self._rss_reader = rss_reader
        self._exit_fn = exit_fn
        self._phase = phase
        self._lock = threading.Lock()
        self._armed = True
        self._callbacks: list[Callable[[int, int], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> MemoryWatchdog:
        """Start sampling when a limit is set."""
        if self.limit_bytes is not None:
            self._thread = threading.Thread(target=self._watch, name="pedsum-memory-watchdog", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop sampling and wait for the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def on_breach(self, fn: Callable[[int, int], None]) -> None:
        """Register ``fn(rss_bytes, limit_bytes)`` to run on a breach; it must write only a few KB."""
        self._callbacks.append(fn)

    @contextmanager
    def disarm(self) -> Iterator[None]:
        """Hold the publication lock for the block; a later breach no longer stops the process."""
        with self._lock:
            self._armed = False
            yield

    def _watch(self) -> None:
        assert self.limit_bytes is not None
        while not self._stop.wait(self._poll_s):
            rss = self._rss_reader()
            if rss > self.limit_bytes:
                self._breach(rss, self.limit_bytes)
                return

    def _breach(self, rss: int, limit: int) -> None:
        phase = self._phase() or "pedsum"
        with self._lock:
            if not self._armed:
                logger.warning(
                    "%s used %.1f GiB RSS, over the %.1f GiB limit, after its output was published",
                    phase,
                    rss / GiB,
                    limit / GiB,
                )
                return
            logger.error(
                "%s used %.1f GiB RSS, over the %.1f GiB limit; stopping. "
                "Pass --max-memory SIZE to raise the limit, or --max-memory 0 to remove it",
                phase,
                rss / GiB,
                limit / GiB,
            )
            for fn in self._callbacks:
                try:
                    fn(rss, limit)
                except Exception:
                    logger.exception("memory-limit callback %r failed", fn)
            for handler in logging.getLogger().handlers:
                handler.flush()
            sys.stdout.flush()
            sys.stderr.flush()
            self._exit_fn(EXIT_MEMORY_LIMIT)
