"""What a ``--safe-attempt`` run may log.

A log line built on a research machine can carry an input ID or a data value,
and a log is a file a user may transfer. Under ``--safe-attempt`` every log
record is rebuilt before any handler sees it. A record of pedsum's own logger
keeps its message, and an argument is printed only when its type says it
holds no ID and no value drawn from the data rows:

- a :class:`Public` string or number, a path, an ``OSError``;
- an exception with a ``public`` text (:class:`pedsum.base.PedigreeError`), as that text.

Every other argument (a plain string, an unmarked number, a list, any
object) prints as ``[withheld]``, and a traceback is dropped. So a new log
call that formats an ID or a count as an argument withholds it without
anyone marking it. The message itself must be a literal, which
``tests/test_safe_log.py`` checks over the package.

A record of any other logger (pedigree-graph, a Python warning, any library)
prints as ``[withheld]`` whole, since nothing checks that its message is a
literal.

``--id-log`` writes each record that lost something, in full, to
``<command>.ids.local.log`` in ``--out`` and nowhere else.
"""

from __future__ import annotations

import contextlib
import logging
import numbers
import re
import warnings
from collections.abc import Mapping
from pathlib import PurePath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path
    from typing import TextIO

WITHHELD = "[withheld]"

#: The logger whose messages are literals (``pedsum.base.logger``) and its children.
_OWN_LOGGER = "pedigree_summary"

_SPEC = re.compile(r"%(?:\((?P<key>[^)]*)\))?[#0\- +]*\d*(?:\.\d+)?[hlL]?(?P<conv>[diouxXeEfFgGcrsa%])")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"


class Public:
    """A log argument that names no individual and is not drawn from the data rows.

    A check, column, estimator, phase or file name; an option value, a thread
    count, the minimum cell, elapsed seconds, a memory cap. A
    ``--safe-attempt`` log prints it; an unmarked string or number is
    withheld. ``Public(value)`` is a ``str``, an ``int`` or a ``float`` like
    ``value``, so an ordinary run formats it as before.
    """

    __slots__ = ()

    def __new__(cls, value: object) -> Public:
        """``value`` as the ``Public`` subclass of its own type."""
        if cls is not Public:
            return super().__new__(cls, value)  # ty: ignore[too-many-positional-arguments]
        if isinstance(value, str):
            return _PublicStr(value)
        if isinstance(value, bool):
            return _PublicStr(str(value))
        if isinstance(value, numbers.Integral):
            return _PublicInt(int(value))
        if isinstance(value, numbers.Real):
            return _PublicFloat(float(value))
        raise TypeError(f"Public takes a string or a number, not {type(value).__name__}")


class _PublicStr(Public, str):
    __slots__ = ()


class _PublicInt(Public, int):
    __slots__ = ()


class _PublicFloat(Public, float):
    __slots__ = ()


def detail(value: object) -> str:
    """``value`` as text, for a log argument drawn from the data: an extreme, a p-value, a depth.

    A ``--safe-attempt`` log withholds it, as it does any unmarked argument.
    """
    return str(value)


class _Scrubber:
    """The log record factory of a ``--safe-attempt`` run; ``make_record`` is the factory it replaced."""

    def __init__(self, make_record: Callable[..., logging.LogRecord]) -> None:
        self.make_record = make_record
        self.id_log: logging.FileHandler | None = None
        self.withheld = False

    def _shown(self, arg: object) -> object | None:
        """``arg`` as it may be printed, or None to withhold it."""
        if isinstance(arg, (Public, PurePath, OSError)):
            return arg
        public = getattr(arg, "public", None) if isinstance(arg, Exception) else None
        return None if public is None else Public(public)

    def _render(self, msg: object, args: object) -> tuple[str, bool]:
        """The scrubbed message text and whether anything was withheld."""
        if not isinstance(msg, str):
            msg, args = "%s", (msg,)
        if not args:
            return msg, False
        # LogRecord's own rule for a single mapping argument.
        if isinstance(args, tuple) and len(args) == 1 and isinstance(args[0], Mapping) and args[0]:
            args = args[0]
        positional = iter(args) if isinstance(args, tuple) else None
        lost = False

        def one(m: re.Match[str]) -> str:
            nonlocal lost
            if m["conv"] == "%":
                return "%"
            key = m["key"]
            if key is not None:
                assert isinstance(args, Mapping)
                arg = args[key]
            else:
                assert positional is not None
                arg = next(positional)
            shown = self._shown(arg)
            if shown is None:
                lost = True
                return WITHHELD
            if isinstance(arg, Exception) and str(shown) != str(arg):
                lost = True
            if isinstance(shown, str):
                return str(shown)
            spec = m[0] if key is None else m[0].replace(f"({key})", "", 1)
            return spec % (shown,)

        try:
            return _SPEC.sub(one, msg), lost
        except Exception:
            # A format this parser does not follow: nothing of it is printed.
            return WITHHELD, True

    def __call__(self, name, level, fn, lno, msg, args, exc_info, func=None, sinfo=None, **kwargs) -> logging.LogRecord:
        own = name == _OWN_LOGGER or name.startswith(_OWN_LOGGER + ".")
        text, lost = self._render(msg, args) if own else (WITHHELD, True)
        if lost or exc_info or sinfo:
            self.withheld = True
            if self.id_log is not None:
                self.id_log.handle(self.make_record(name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs))
        return self.make_record(name, level, fn, lno, text, (), None, func, None, **kwargs)


def _current() -> _Scrubber | None:
    factory = logging.getLogRecordFactory()
    return factory if isinstance(factory, _Scrubber) else None


def _log_warning(
    message: Warning | str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """``warnings.showwarning`` while scrubbing: the category prints, the text is withheld like any string."""
    logging.getLogger(_OWN_LOGGER).warning(
        "a library warned (%s): %s",
        Public(category.__name__),
        warnings.formatwarning(message, category, filename, lineno, line),
    )


@contextlib.contextmanager
def scrubbing() -> Generator[None]:
    """Scrub every log record and every Python warning inside the block.

    On exit the ID log is closed, and the log record factory and
    ``warnings.showwarning`` are the ones found on entry.
    """
    make_record = logging.getLogRecordFactory()
    showwarning = warnings.showwarning
    logging.setLogRecordFactory(_Scrubber(make_record))
    warnings.showwarning = _log_warning  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        close_id_log()
        logging.setLogRecordFactory(make_record)
        warnings.showwarning = showwarning


def open_id_log(path: Path) -> None:
    """Start ``path`` empty and write every record that loses something to it, in full."""
    scrubber = _current()
    assert scrubber is not None
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    scrubber.id_log = handler


def close_id_log() -> None:
    """Close the ID log, if one is open."""
    scrubber = _current()
    if scrubber is not None and scrubber.id_log is not None:
        scrubber.id_log.close()
        scrubber.id_log = None


def id_log_is_open() -> bool:
    """Whether ``--id-log`` opened its file in this run."""
    scrubber = _current()
    return scrubber is not None and scrubber.id_log is not None


def write_id_detail(text: str) -> None:
    """Append ``text`` to the ID log; nothing without ``--id-log``."""
    scrubber = _current()
    if scrubber is None or scrubber.id_log is None:
        return
    handler = scrubber.id_log
    handler.acquire()
    try:
        assert handler.stream is not None
        handler.stream.write(text)
        handler.flush()
    finally:
        handler.release()


def anything_withheld() -> bool:
    """Whether this run's log has withheld an argument or a traceback."""
    scrubber = _current()
    return scrubber is not None and scrubber.withheld
