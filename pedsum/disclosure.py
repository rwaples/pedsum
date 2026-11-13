"""Small-cell disclosure engine for ``--safe-attempt``: declared tables, merge or blank, taint, release list.

Pure functions over plain data, shared by every command. A command declares
a :class:`ReleaseSpec`: the leaves of its payload that may leave the research
machine, each with the reason it is safe. :func:`build_release` copies those
leaves into a new payload and nothing else, so a key no rule names is never
released.

Best effort, not a safe-harbor guarantee. The engine protects each declared
table and withholds every statistic derived from a table it had to merge or
blank. It cannot see that two declared tables share a cell: a spec lists
each fact once and declares what remains shared as a :class:`Link`, which
``tests/disclosure_oracle.py`` checks by exact integer programming.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pedsum.sections import Distribution

#: The minimum cell ``--safe-attempt`` protects at.
DEFAULT_MIN_CELL = 5

#: The smallest minimum cell the engine accepts. Below 3 a blanked count has
#: one possible value.
MIN_CELL_FLOOR = 3

#: A payload leaf: the keys from the payload root down to it.
type Path = tuple[str, ...]

#: A bin of a 1-D table, a ``(row, column)`` cell of a contingency table, or
#: ``None`` for a table's total or a scalar count.
type CellLabel = str | tuple[str, str] | None

#: One cell of one declared table or count.
type CellRef = tuple[Path, CellLabel]


class Status(Enum):
    """What protection did to a table or count."""

    INTACT = "intact"
    MERGED = "merged"
    BLANKED = "blanked"


def check_min_cell(min_cell: int) -> None:
    """Raise ``ValueError`` unless ``min_cell`` is an int of at least :data:`MIN_CELL_FLOOR`."""
    if not _is_count(min_cell) or min_cell < MIN_CELL_FLOOR:
        raise ValueError(f"min_cell must be an integer >= {MIN_CELL_FLOOR}, got {min_cell!r}")


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _check_counts(what: str, values: Sequence[object]) -> None:
    """Raise ``TypeError`` unless every value is a non-negative int: a share passed as a count must not pass the gate."""
    bad = [v for v in values if not _is_count(v)]
    if bad:
        raise TypeError(f"{what} must be non-negative integers, got {bad[0]!r}")


@dataclass(frozen=True)
class FreqTable:
    """A 1-D frequency table: labelled integer counts, in order.

    ``support``, when given, is each bin's number of distinct supporting
    units (the families behind a pair count); the minimum-cell test then
    reads it instead of the count. A count below its support cannot come
    from real data and is tested as the count.
    """

    bins: tuple[tuple[str, int], ...]
    ordered: bool
    support: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the counts, the labels and the support's length."""
        labels = [label for label, _ in self.bins]
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate bin labels in {labels}")
        _check_counts("bin counts", [count for _, count in self.bins])
        if self.support is not None:
            if len(self.support) != len(self.bins):
                raise ValueError(f"support has {len(self.support)} entries for {len(self.bins)} bins")
            _check_counts("support counts", self.support)


@dataclass(frozen=True)
class ContingencyTable:
    """A 2-D table of integer counts, ``counts[i][j]`` for ``rows[i]`` and ``cols[j]``, with optional per-cell support."""

    rows: tuple[str, ...]
    cols: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...]
    support: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        """Validate the shape and the counts."""
        for name, grid in (("counts", self.counts), ("support", self.support)):
            if grid is None:
                continue
            if len(grid) != len(self.rows) or any(len(row) != len(self.cols) for row in grid):
                raise ValueError(f"{name} is not {len(self.rows)} x {len(self.cols)}")
            _check_counts(name, [cell for row in grid for cell in row])


@dataclass(frozen=True)
class ReleasedBin:
    """One released bin; ``covers`` names the source bins it sums, more than one after a merge."""

    label: str
    count: int
    covers: tuple[str, ...]


@dataclass(frozen=True)
class ProtectedFreq:
    """A protected 1-D table. ``bins`` is empty when blanked; ``total`` is ``None`` when the total itself is small."""

    status: Status
    bins: tuple[ReleasedBin, ...]
    total: int | None
    labels: tuple[str, ...]

    def render(self) -> dict[str, Any]:
        """The released mapping: status, total and, unless blanked, the bins."""
        out: dict[str, Any] = {"status": self.status.value, "total": self.total}
        if self.status is not Status.BLANKED:
            out["bins"] = {b.label: b.count for b in self.bins}
        return out


@dataclass(frozen=True)
class ProtectedContingency:
    """A protected contingency table: every cell, or (blanked) only the total."""

    status: Status
    rows: tuple[str, ...]
    cols: tuple[str, ...]
    counts: tuple[tuple[int, ...], ...] | None
    total: int | None

    def render(self) -> dict[str, Any]:
        """The released mapping: status, total and, unless blanked, the cells by row then column."""
        out: dict[str, Any] = {"status": self.status.value, "total": self.total}
        if self.counts is not None:
            out["cells"] = {
                r: dict(zip(self.cols, row, strict=True)) for r, row in zip(self.rows, self.counts, strict=True)
            }
        return out


@dataclass(frozen=True)
class ProtectedCount:
    """A protected scalar count: its value, or ``None`` when blanked."""

    status: Status
    value: int | None

    def render(self) -> int | None:
        """The released value."""
        return self.value


type Protected = ProtectedFreq | ProtectedContingency | ProtectedCount


def protect(table: FreqTable | ContingencyTable, min_cell: int) -> ProtectedFreq | ProtectedContingency:
    """Protect one table at ``min_cell``.

    An ordered 1-D table merges tail bins inward, from each end whose
    outermost nonzero bin is small, until the merged bin is not small. It is
    blanked if a small bin remains between the tails or fewer than two bins
    would be left. An unordered 1-D table is blanked by any small bin. A
    zero bin of a 1-D table is not small and stays visible.

    A contingency table is blanked by any small cell. A zero cell is not
    small, yet a zero in a row or column whose total is below ``min_cell``
    is never released: that margin holds a nonzero cell at most as large
    as its total, which blanks the table.

    A blanked table keeps only its total, and not that when the total is
    small.

    Raises:
        ValueError: If ``min_cell`` is below :data:`MIN_CELL_FLOOR`.
    """
    check_min_cell(min_cell)
    if isinstance(table, ContingencyTable):
        return _protect_contingency(table, min_cell)
    return _protect_freq(table, min_cell)


def protect_count(count: int, min_cell: int, support: int | None = None) -> ProtectedCount:
    """Release a scalar count that is 0 or backed by at least ``min_cell`` units (``support`` if given, else itself).

    Raises:
        ValueError: If ``min_cell`` is below :data:`MIN_CELL_FLOOR`.
    """
    check_min_cell(min_cell)
    _check_counts("a count and its support", [count] if support is None else [count, support])
    gate = count if support is None else min(count, support)
    if count == 0 or gate >= min_cell:
        return ProtectedCount(Status.INTACT, count)
    return ProtectedCount(Status.BLANKED, None)


def _protect_freq(table: FreqTable, min_cell: int) -> ProtectedFreq:
    labels = tuple(label for label, _ in table.bins)
    counts = [count for _, count in table.bins]
    k = len(counts)

    def small(lo: int, hi: int) -> bool:
        """Whether bins ``lo:hi``, taken as one bin, are nonzero and below ``min_cell``."""
        count = sum(counts[lo:hi])
        # Families can back several bins, so a merged bin's distinct support
        # is only known to be at least its largest part.
        gate = count if table.support is None else min(count, max(table.support[lo:hi], default=0))
        return count > 0 and gate < min_cell

    total = sum(counts)
    released_total = None if small(0, k) else total
    blanked = ProtectedFreq(Status.BLANKED, (), released_total, labels)
    singles = [(i, i + 1) for i in range(k)]
    if not any(small(lo, hi) for lo, hi in singles):
        return ProtectedFreq(Status.INTACT, _released(table, singles, k), released_total, labels)
    if not table.ordered:
        return blanked

    nonzero = [i for i, count in enumerate(counts) if count]
    lower_end, upper_end = 0, k
    if small(nonzero[0], nonzero[0] + 1):
        lower_end = nonzero[0] + 1
        while lower_end < k and small(0, lower_end):
            lower_end += 1
    if small(nonzero[-1], nonzero[-1] + 1):
        upper_end = nonzero[-1]
        while upper_end > 0 and small(upper_end, k):
            upper_end -= 1
    if lower_end > upper_end:
        return blanked
    groups = [(0, lower_end)] if lower_end else []
    groups += singles[lower_end:upper_end]
    groups += [(upper_end, k)] if upper_end < k else []
    if len(groups) < 2 or any(small(lo, hi) for lo, hi in groups):
        return blanked
    return ProtectedFreq(Status.MERGED, _released(table, groups, k), released_total, labels)


def _released(table: FreqTable, groups: Sequence[tuple[int, int]], k: int) -> tuple[ReleasedBin, ...]:
    out = []
    for lo, hi in groups:
        covers = tuple(label for label, _ in table.bins[lo:hi])
        label = covers[0] if hi - lo == 1 else _merged_label(covers, upper=hi == k)
        out.append(ReleasedBin(label, sum(count for _, count in table.bins[lo:hi]), covers))
    return tuple(out)


def _merged_label(covers: tuple[str, ...], *, upper: bool) -> str:
    """Name a merged run of bins: ``"4+"`` for an upper tail of integers, ``"0-1"`` for a lower one, else ``first..last``."""
    first, last = covers[0], covers[-1]
    if upper and first.isdecimal():
        return f"{first}+"
    if first.isdecimal() and last.isdecimal():
        return f"{first}-{last}"
    return f"{first}..{last}"


def _protect_contingency(table: ContingencyTable, min_cell: int) -> ProtectedContingency:
    counts = table.counts
    gates = counts
    if table.support is not None:
        gates = tuple(
            tuple(min(c, s) for c, s in zip(count_row, support_row, strict=True))
            for count_row, support_row in zip(counts, table.support, strict=True)
        )
    total = sum(cell for row in counts for cell in row)
    any_small = any(
        count > 0 and gate < min_cell
        for count_row, gate_row in zip(counts, gates, strict=True)
        for count, gate in zip(count_row, gate_row, strict=True)
    )
    # As for a merged bin: the total's distinct support is only known to be at least its largest cell's.
    total_gate = total if table.support is None else max((cell for row in gates for cell in row), default=0)
    released_total = None if total > 0 and total_gate < min_cell else total
    if any_small:
        return ProtectedContingency(Status.BLANKED, table.rows, table.cols, None, released_total)
    return ProtectedContingency(Status.INTACT, table.rows, table.cols, counts, released_total)


def release_derived[T](value: T, sources: Sequence[Protected]) -> T | None:
    """``value`` if every table or count it was computed from is intact, else ``None``."""
    return value if all(source.status is Status.INTACT for source in sources) else None


def release_distribution(dist: Distribution, min_cell: int, histogram: ProtectedFreq | None = None) -> dict[str, float]:
    """The fields of ``dist`` that may be released; never ``min``, ``max`` or ``nz``.

    A continuous variable (no ``histogram``) releases ``n``, ``mean`` and
    ``std`` when ``n`` is 0 or at least ``min_cell``, and the quartiles when
    ``n >= 4 * min_cell``, so no quartile cut falls within ``min_cell``
    values of an end.

    An integer-count variable passes the protected full-resolution
    ``histogram`` of its values. Its mean, sd and quartiles are released
    only if that histogram is intact; ``n`` is the histogram's total and is
    not repeated.

    Raises:
        ValueError: If ``min_cell`` is below :data:`MIN_CELL_FLOOR`.
    """
    check_min_cell(min_cell)
    moments = {"mean": dist.mean, "std": dist.std}
    quartiles = {"q1": dist.q1, "median": dist.median, "q3": dist.q3}
    if histogram is not None:
        return moments | quartiles if histogram.status is Status.INTACT else {}
    if dist.n == 0:
        return {"n": 0}
    if dist.n < min_cell:
        return {}
    return {"n": dist.n} | moments | (quartiles if dist.n >= 4 * min_cell else {})


# ---------------------------------------------------------------------------
# Release list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """One released leaf: where it is in the payload and why releasing it is safe."""

    path: Path
    reason: str

    def __post_init__(self) -> None:
        """Require a path and a written reason."""
        if not self.path or not all(isinstance(key, str) and key for key in self.path):
            raise ValueError(f"a rule path is a non-empty tuple of keys, got {self.path!r}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(f"rule {'.'.join(self.path)} has no reason")


@dataclass(frozen=True)
class Plain(Rule):
    """Copied as is: metadata that describes the run, not the data."""


@dataclass(frozen=True, kw_only=True)
class Count(Rule):
    """A scalar count; ``support`` is the payload path of its number of distinct supporting units."""

    support: Path | None = None


@dataclass(frozen=True, kw_only=True)
class Table(Rule):
    """A 1-D table, a mapping of label to count in bin order; ``support`` is the path of a parallel mapping.

    ``requires`` names tables this one repeats cells of (a finer table it
    is a margin of). It is released only while each of them is intact, and
    is otherwise withheld down to its total.
    """

    ordered: bool
    support: Path | None = None
    requires: tuple[Path, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Contingency(Rule):
    """A 2-D table, a mapping of row to column to count; ``support`` and ``requires`` are as for :class:`Table`."""

    support: Path | None = None
    requires: tuple[Path, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Derived(Rule):
    """A statistic computed from the tables and counts at ``sources``, released only if all of them are intact."""

    sources: tuple[Path, ...]

    def __post_init__(self) -> None:
        """Require at least one source: a statistic with none is not derived, and nothing would gate it."""
        super().__post_init__()
        if not self.sources:
            raise ValueError(f"derived rule {'.'.join(self.path)} names no source")


@dataclass(frozen=True, kw_only=True)
class Continuous(Rule):
    """A ``Distribution``; ``histogram`` is the path of the table of its values when it summarises integer counts."""

    histogram: Path | None = None


@dataclass(frozen=True)
class Link:
    """A declared relation between released tables: the ``whole`` cell is the sum of the ``parts`` cells.

    With no parts the whole is zero by construction (no individual of the
    deepest depth has offspring). The engine does not act on links. They tell the disclosure oracle which
    cells of different tables are tied, so it can prove no blanked cell is
    pinned.
    """

    whole: CellRef
    parts: tuple[CellRef, ...]
    reason: str


def shared_total(reason: str, *paths: Path) -> tuple[Link, ...]:
    """Links saying the tables and counts at ``paths`` split one population, so their totals are equal."""
    first, *rest = paths
    return tuple(Link((first, None), ((other, None),), reason) for other in rest)


_SOURCE_RULES = (Count, Table, Contingency)
_TABLE_RULES = (Table, Contingency)


@dataclass(frozen=True)
class ReleaseSpec:
    """A command's release list: its rules, and the links between its tables."""

    rules: tuple[Rule, ...]
    links: tuple[Link, ...] = ()

    def __post_init__(self) -> None:
        """Reject clashing paths and references to tables the spec does not declare."""
        by_path: dict[Path, Rule] = {}
        for rule in self.rules:
            if rule.path in by_path:
                raise ValueError(f"two rules for {'.'.join(rule.path)}")
            by_path[rule.path] = rule
        for path in by_path:
            for depth in range(1, len(path)):
                if path[:depth] in by_path:
                    raise ValueError(f"rule {'.'.join(path[:depth])} is a parent of rule {'.'.join(path)}")

        def require(owner: str, path: Path, kinds: tuple[type, ...]) -> None:
            if not isinstance(by_path.get(path), kinds):
                names = " or ".join(kind.__name__ for kind in kinds)
                raise ValueError(f"{owner} refers to {'.'.join(path)}, which is not a {names} rule")

        for rule in self.rules:
            if isinstance(rule, Derived):
                for source in rule.sources:
                    require(f"derived rule {'.'.join(rule.path)}", source, _SOURCE_RULES)
            if isinstance(rule, Continuous) and rule.histogram is not None:
                require(f"continuous rule {'.'.join(rule.path)}", rule.histogram, (Table,))
            if isinstance(rule, _TABLE_RULES):
                for required in rule.requires:
                    require(f"table rule {'.'.join(rule.path)}", required, _TABLE_RULES)
                    target = by_path[required]
                    if isinstance(target, _TABLE_RULES) and target.requires:
                        raise ValueError(
                            f"table rule {'.'.join(rule.path)} requires {'.'.join(required)}, "
                            "which has requirements of its own"
                        )
        for link in self.links:
            if not link.reason.strip():
                raise ValueError("a link has no reason")
            for path, _ in (link.whole, *link.parts):
                require("a link", path, _SOURCE_RULES)


_ABSENT = object()


def _lookup(payload: Mapping[str, Any], path: Path) -> Any:
    """The value at ``path``, which may be ``None``, or ``_ABSENT`` when a key is missing."""
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return _ABSENT
        node = node[key]
    return node


def _protect_rule(
    rule: Count | Table | Contingency, value: Any, payload: Mapping[str, Any], min_cell: int
) -> Protected:
    """Protect the count or table ``value``; a rule that names a support the payload lacks is withheld whole."""
    support = None if rule.support is None else _lookup(payload, rule.support)
    unsupported = rule.support is not None and (support is _ABSENT or support is None)
    match rule:
        case Count():
            if unsupported:
                return ProtectedCount(Status.BLANKED, None)
            return protect_count(value, min_cell, support)
        case Table():
            if unsupported:
                return ProtectedFreq(Status.BLANKED, (), None, tuple(value))
            aligned = None if support is None else tuple(support[label] for label in value)
            return protect(FreqTable(tuple(value.items()), rule.ordered, aligned), min_cell)
        case Contingency():
            rows = tuple(value)
            cols = tuple(value[rows[0]]) if rows else ()
            if unsupported:
                return ProtectedContingency(Status.BLANKED, rows, cols, None, None)

            def grid(cells: Mapping[str, Mapping[str, int]]) -> tuple[tuple[int, ...], ...]:
                return tuple(tuple(cells[r][c] for c in cols) for r in rows)

            return protect(
                ContingencyTable(rows, cols, grid(value), None if support is None else grid(support)), min_cell
            )
    raise TypeError(f"{type(rule).__name__} is not a table or count rule")


def _withheld(table: ProtectedFreq | ProtectedContingency) -> ProtectedFreq | ProtectedContingency:
    """``table`` with its cells withheld and its total kept."""
    if isinstance(table, ProtectedFreq):
        return replace(table, status=Status.BLANKED, bins=())
    return replace(table, status=Status.BLANKED, counts=None)


def protect_payload(payload: Mapping[str, Any], spec: ReleaseSpec, min_cell: int) -> dict[Path, Protected]:
    """Protect every table and count ``spec`` declares that ``payload`` holds, keyed by rule path.

    A table is then withheld down to its total unless every table it
    ``requires`` is in ``payload`` and intact.

    Raises:
        ValueError: If ``min_cell`` is below :data:`MIN_CELL_FLOOR`.
    """
    check_min_cell(min_cell)
    protected = {}
    for rule in spec.rules:
        if isinstance(rule, _SOURCE_RULES):
            value = _lookup(payload, rule.path)
            if value is not _ABSENT and value is not None:
                protected[rule.path] = _protect_rule(rule, value, payload, min_cell)
    for rule in spec.rules:
        table = protected.get(rule.path)
        if isinstance(rule, _TABLE_RULES) and isinstance(table, (ProtectedFreq, ProtectedContingency)):
            intact = all(
                required in protected and protected[required].status is Status.INTACT for required in rule.requires
            )
            if not intact:
                protected[rule.path] = _withheld(table)
    return protected


def build_release(payload: Mapping[str, Any], spec: ReleaseSpec, min_cell: int) -> dict[str, Any]:
    """A new payload holding only the leaves ``spec`` lists, each protected at ``min_cell``.

    A key no rule names is absent from the result, and so is a listed leaf
    whose key ``payload`` lacks. A listed leaf that is withheld is
    ``None``, and so is one ``payload`` holds as ``None``: a statistic that
    could not be computed reads as a withheld one, so which keys a release
    has says nothing about the data. A derived statistic whose source is
    missing from ``payload`` is withheld.

    Raises:
        ValueError: If ``min_cell`` is below :data:`MIN_CELL_FLOOR`.
    """
    protected = protect_payload(payload, spec, min_cell)
    out: dict[str, Any] = {}
    for rule in spec.rules:
        value = _lookup(payload, rule.path)
        if value is _ABSENT:
            continue
        match rule:
            case _ if value is None:
                released = None
            case Plain():
                released = value
            case Count() | Table() | Contingency():
                released = protected[rule.path].render()
            case Derived():
                sources = [protected[source] for source in rule.sources if source in protected]
                released = release_derived(value, sources) if len(sources) == len(rule.sources) else None
            case Continuous():
                histogram = None if rule.histogram is None else protected.get(rule.histogram)
                if rule.histogram is None:
                    released = release_distribution(value, min_cell) or None
                elif isinstance(histogram, ProtectedFreq):
                    released = release_distribution(value, min_cell, histogram) or None
                else:
                    released = None
            case _:
                raise TypeError(f"no release for rule type {type(rule).__name__}")
        node = out
        for key in rule.path[:-1]:
            node = node.setdefault(key, {})
        node[rule.path[-1]] = released
    return out
