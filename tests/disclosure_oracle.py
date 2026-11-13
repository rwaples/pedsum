"""Exact integer oracle for the disclosure engine: can a withheld cell be recomputed from what was released?

A :class:`System` holds what a reader of the released output knows: one
unknown non-negative integer per source cell, the linear relations among
them (a released bin, a total, a declared link, a moment of a released
statistic) and, for a statistic that is not linear in the cells, the
alternatives it leaves. A cell is pinned when every integer solution gives
it the same value. ``scipy.optimize.milp`` minimises and maximises each
cell; a linear-programming relaxation is not enough (see
``test_lp_relaxation_misses_the_integer_pin``).

The oracle covers the relations a spec declares. It does not model what
the fact of blanking a table says about its cells; a blanked scalar count
is bounded to ``1..min_cell-1``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from pedsum.disclosure import ProtectedContingency, ProtectedCount, ProtectedFreq, Status

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

    from pedsum.disclosure import CellRef, Path, Protected, ReleaseSpec

#: Upper bound of a cell nothing else bounds.
DEFAULT_CAP = 10**6

# Constraint sums are integers, so a bound within this of an integer is that integer.
_EPS = 1e-9

type Constraint = tuple[dict[Hashable, int], float, float]


@dataclass(frozen=True)
class Branch:
    """One alternative left by a nonlinear statistic: cells fixed to values, plus extra linear constraints."""

    fixes: Mapping[Hashable, int] = field(default_factory=dict)
    constraints: tuple[Constraint, ...] = ()


class InconsistentSystemError(Exception):
    """No integer solution: the released view contradicts itself, so the system was built wrong."""


class System:
    """Unknown integer cells and what the released output says about them."""

    def __init__(self) -> None:
        """Start with no cells."""
        self._bounds: dict[Hashable, tuple[int, int]] = {}
        self._constraints: list[Constraint] = []
        self._choices: list[Sequence[Branch]] = []

    def var(self, name: Hashable, lo: int = 0, hi: int = DEFAULT_CAP) -> None:
        """Declare the cell ``name`` with ``lo <= name <= hi``."""
        if name in self._bounds:
            raise ValueError(f"cell {name!r} declared twice")
        self._bounds[name] = (lo, hi)

    def fix(self, name: Hashable, value: int) -> None:
        """A released cell: its value is known."""
        self.add({name: 1}, value, value)

    def add(self, coeffs: Mapping[Hashable, int], lo: float, hi: float) -> None:
        """Require ``lo <= sum(coeff * cell) <= hi``."""
        unknown = [name for name in coeffs if name not in self._bounds]
        if unknown:
            raise KeyError(f"constraint on undeclared cells {unknown}")
        self._constraints.append((dict(coeffs), lo, hi))

    def choose(self, branches: Sequence[Branch]) -> None:
        """Require that one of ``branches`` holds."""
        self._choices.append(branches)

    def value_range(self, name: Hashable, *, integral: bool = True) -> tuple[float, float]:
        """The smallest and largest value of ``name`` over all solutions.

        Raises:
            InconsistentSystemError: If there is no solution.
        """
        names = list(self._bounds)
        index = {n: i for i, n in enumerate(names)}
        objective = np.zeros(len(names))
        objective[index[name]] = 1.0
        lo_all, hi_all = math.inf, -math.inf
        for combo in itertools.product(*self._choices):
            lower = np.array([self._bounds[n][0] for n in names], dtype=float)
            upper = np.array([self._bounds[n][1] for n in names], dtype=float)
            constraints = list(self._constraints)
            for branch in combo:
                constraints += branch.constraints
                for fixed, value in branch.fixes.items():
                    lower[index[fixed]] = max(lower[index[fixed]], value)
                    upper[index[fixed]] = min(upper[index[fixed]], value)
            if (lower > upper).any():
                continue
            matrix = np.zeros((len(constraints), len(names)))
            lows, highs = [], []
            for row, (coeffs, lo, hi) in enumerate(constraints):
                for cell, coeff in coeffs.items():
                    matrix[row, index[cell]] = coeff
                lows.append(math.ceil(lo - _EPS))
                highs.append(math.floor(hi + _EPS))
            if any(lo > hi for lo, hi in zip(lows, highs, strict=True)):
                continue
            linear = LinearConstraint(matrix, lows, highs) if constraints else ()
            ends = []
            for sign in (1.0, -1.0):
                res = milp(
                    sign * objective,
                    constraints=linear,
                    integrality=np.full(len(names), int(integral)),
                    bounds=Bounds(lower, upper),
                )
                if res.status == 2:
                    break
                if not res.success:
                    raise RuntimeError(f"milp did not solve: {res.message}")
                ends.append(float(res.x[index[name]]))
            else:
                lo_all, hi_all = min(lo_all, ends[0]), max(hi_all, ends[1])
        if lo_all > hi_all:
            raise InconsistentSystemError(f"no solution when solving for {name!r}")
        return (round(lo_all), round(hi_all)) if integral else (lo_all, hi_all)

    def pinned(self, targets: Iterable[Hashable], *, integral: bool = True) -> list[Hashable]:
        """The ``targets`` every solution gives the same value."""
        out = []
        for name in targets:
            lo, hi = self.value_range(name, integral=integral)
            if hi - lo < 1e-6:
                out.append(name)
        return out


def add_moments(
    system: System, values: Mapping[Hashable, int], n: int, mean: float, sd: float, half_width: float
) -> None:
    """Add what a released mean and sample sd of ``n`` integer values say about how many values equal each integer.

    ``values`` maps each cell (a count of values) to the integer its values
    equal. ``mean`` and ``sd`` are known to within ``half_width``, the
    serialised precision. The sum of squares depends on the square of the
    sum, so each integer sum the mean allows is its own branch.
    """
    total = dict(values)
    squares = {cell: v * v for cell, v in values.items()}
    sum_lo = math.ceil(n * (mean - half_width) - _EPS)
    sum_hi = math.floor(n * (mean + half_width) + _EPS)
    sd_lo, sd_hi = max(sd - half_width, 0.0), sd + half_width
    system.choose(
        [
            Branch(
                constraints=(
                    (total, s, s),
                    (squares, (n - 1) * sd_lo**2 + s * s / n, (n - 1) * sd_hi**2 + s * s / n),
                )
            )
            for s in range(sum_lo, sum_hi + 1)
        ]
    )


def add_nonlinear(
    system: System, cells: Sequence[Hashable], candidates: Iterable[Sequence[int]], consistent: Callable[..., bool]
) -> None:
    """Add a released statistic that is not linear in ``cells``: keep the ``candidates`` it is ``consistent`` with."""
    system.choose([Branch(fixes=dict(zip(cells, c, strict=True))) for c in candidates if consistent(*c)])


def close_to(released: float, half_width: float) -> Callable[[float], bool]:
    """Whether a recomputed statistic serialises to ``released``."""
    return lambda value: abs(value - released) <= half_width + _EPS


def _table_cells(path: Path, table: Protected) -> list[CellRef]:
    if isinstance(table, ProtectedFreq):
        return [(path, label) for label in table.labels]
    if isinstance(table, ProtectedContingency):
        return [(path, (r, c)) for r in table.rows for c in table.cols]
    return [(path, None)]


def system_from_release(
    spec: ReleaseSpec,
    protected: Mapping[Path, Protected],
    min_cell: int,
    cap: int = DEFAULT_CAP,
    unwritten: Callable[[Path], bool] = lambda path: False,
) -> tuple[System, list[CellRef]]:
    """The system a reader can write down from a protected payload, and the cells that were withheld.

    Every source cell of every table is an unknown. A released bin fixes
    its cell, a merged bin and a released total fix a sum, and each
    ``spec.links`` entry ties cells of different tables, or with no parts
    fixes a cell at zero. A table or count at an ``unwritten`` path is in
    no output file, whatever its status: the reader knows its cells only
    through the links, and all of them count as withheld.
    """
    system = System()
    hidden: list[CellRef] = []
    totals: dict[Path, dict[CellRef, int]] = {}
    for path, table in protected.items():
        cells = _table_cells(path, table)
        totals[path] = dict.fromkeys(cells, 1)
        if unwritten(path):
            for cell in cells:
                system.var(cell, 0, cap)
            hidden += cells
            continue
        if isinstance(table, ProtectedCount):
            if table.value is None:
                system.var((path, None), 1, min_cell - 1)
                hidden.append((path, None))
            else:
                system.var((path, None), 0, cap)
                system.fix((path, None), table.value)
            continue
        for cell in cells:
            system.var(cell, 0, cap)
        if table.total is not None:
            system.add(totals[path], table.total, table.total)
        if table.status is Status.BLANKED:
            hidden += cells
        elif isinstance(table, ProtectedFreq):
            for released in table.bins:
                system.add({(path, label): 1 for label in released.covers}, released.count, released.count)
                if len(released.covers) > 1:
                    hidden += [(path, label) for label in released.covers]
        else:
            for r, row in zip(table.rows, table.counts or (), strict=True):
                for c, count in zip(table.cols, row, strict=True):
                    system.fix((path, (r, c)), count)

    def side(ref: CellRef) -> dict[CellRef, int]:
        path, label = ref
        return totals[path] if label is None else {ref: 1}

    for link in spec.links:
        if any(path not in protected for path, _ in (link.whole, *link.parts)):
            continue
        coeffs: dict[CellRef, int] = dict(side(link.whole))
        for part in link.parts:
            for cell, coeff in side(part).items():
                coeffs[cell] = coeffs.get(cell, 0) - coeff
        system.add(coeffs, 0, 0)
    return system, hidden
