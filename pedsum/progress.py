"""A terminal progress bar for pedigree-graph's long relationship calls.

pedigree-graph reports a running relationship call through its ``progress=``
keyword (its ADR 0017). :func:`relationship_progress` yields what to pass
there: a callable that draws a tqdm bar on an interactive stderr, or ``None``
to keep pedigree-graph's progress line every 30 s, which suits log files and
redirected runs.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, TextIO

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pedigree_graph import RelationshipProgress

# No rate or remaining-time field: row cost is far from uniform (on the horse
# pedigree, 14% of the rows took the first 30 s of an 1,833 s call), so any
# estimate would mislead.
_WALK_FORMAT = "{desc} {percentage:3.0f}%|{bar}| {n:,}/{total:,} rows [{elapsed}]"

#: Per phase: the suffix after the label, and the bar format. The total is
#: unknown while preparing, so that phase shows only the clock.
_PHASES = {
    "preparing": (": preparing", "{desc} [{elapsed}]"),
    "walking": ("", _WALK_FORMAT),
    "finishing": (": assembling", _WALK_FORMAT),
}


@contextmanager
def relationship_progress(
    label: str, stream: TextIO | None = None
) -> Iterator[Callable[[RelationshipProgress], None] | None]:
    """Yield the ``progress=`` value for one pedigree-graph relationship call.

    The bar is drawn only when *stream* is a terminal and the
    ``pedigree_graph`` logger is enabled for INFO, so ``--quiet`` hides it as
    it hides the log lines. It appears on the first progress report, about a
    second into the call, so a short call draws nothing. While the bar is
    open, console log lines are written above it rather than through it.

    Args:
        label: The bar's description, usually the method name.
        stream: Where the bar is drawn; ``sys.stderr`` when ``None``.

    Yields:
        A callable that takes a ``RelationshipProgress`` and draws the bar, or
        ``None`` for pedigree-graph's default progress logging.
    """
    stream = sys.stderr if stream is None else stream
    if not (stream.isatty() and logging.getLogger("pedigree_graph").isEnabledFor(logging.INFO)):
        yield None
        return

    bar: tqdm | None = None

    def draw(p: RelationshipProgress) -> None:
        nonlocal bar
        suffix, bar_format = _PHASES[p.phase]
        if bar is None:
            bar = tqdm(
                desc=label + suffix,
                total=p.rows_total,
                initial=p.rows_done,
                file=stream,
                bar_format=bar_format,
            )
            # Count from the call's start, about a second before this report.
            bar.start_t -= p.elapsed
        bar.bar_format = bar_format
        bar.set_description_str(label + suffix, refresh=False)
        bar.total = p.rows_total
        bar.n = p.rows_done
        bar.refresh()

    try:
        with logging_redirect_tqdm():
            yield draw
        # The call can return between two reports, leaving the bar short of
        # its total; a completed call should not read 97%.
        if bar is not None and bar.total:
            bar.bar_format = _WALK_FORMAT
            bar.set_description_str(label, refresh=False)
            bar.n = bar.total
            bar.refresh()
    finally:
        if bar is not None:
            bar.close()
