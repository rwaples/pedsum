"""Small-cell redaction primitives for ``summarize --safe-attempt``.

Best effort, not a safe-harbor guarantee. Each summary section's rule lives
beside the code that produces the section and is registered as
``SectionSpec.redact`` (``pedsum.schema``); ``report.apply_safe_attempt``
drives them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Counts and strata below this many individuals are nulled.
SAFE_MIN_CELL = 5

#: A section's rule: redact the section in place, given the minimum cell
#: size and the pedigree's row count.
type Redact = Callable[[Any, int, int], None]


def null_below(d: dict, keys: Iterable[str], min_cell: int) -> None:
    """Null each of ``keys`` in ``d`` whose int value is in ``(0, min_cell)``.

    Missing / ``None`` values collapse to 0 via ``or 0`` and are left
    untouched (the ``0 <`` guard fails).
    """
    for k in keys:
        if 0 < int(d.get(k, 0) or 0) < min_cell:
            d[k] = None


def redact_small_group(row: dict, keep: tuple[str, ...], min_cell: int, small_keys: Iterable[str]) -> None:
    """Small-cell redaction for a grouped row keyed by ``n``.

    If the group's ``n`` is below ``min_cell``, null every key except those
    in ``keep``. Otherwise null only the ``small_keys`` whose own value is
    below ``min_cell`` (via :func:`null_below`).
    """
    if int(row.get("n", 0)) < min_cell:
        for k in list(row):
            if k not in keep:
                row[k] = None
    else:
        null_below(row, small_keys, min_cell)
