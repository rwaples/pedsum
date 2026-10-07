"""Low-level pedigree array helpers (parent rows, mating pairs, depth)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    import polars as pl


def _id_list(ids, max_show: int = 5) -> str:
    ids = list(ids)
    if len(ids) <= max_show:
        return ", ".join(str(i) for i in ids)
    return ", ".join(str(i) for i in ids[:max_show]) + f", ... ({len(ids)} total)"


def unique_ints(values: np.ndarray) -> np.ndarray:
    """Sorted distinct values of a 1-D integer array, identical to ``np.unique(values)``."""
    # A flagless np.unique takes numpy's hash-table path (numpy 2.5.2), ~30x
    # slower than sorting on high-cardinality int64 and superlinear in n.
    ordered = np.sort(values)
    if ordered.size == 0:
        return ordered
    keep = np.empty(ordered.size, dtype=bool)
    keep[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=keep[1:])
    return ordered[keep]


class IdIndex:
    """ID → row-position lookup over an int array (argsort + searchsorted).

    Replaces the pandas ``Index.get_indexer`` idiom: ``get_indexer(values)``
    returns, for each value, the row position of that ID in the original
    array, or ``-1`` when absent. With duplicate IDs the first occurrence
    (lowest row) wins — callers that care about duplicates gate on the
    ``duplicate_ids`` check first.
    """

    def __init__(self, ids) -> None:
        """Build the lookup from ``ids`` (any int-convertible 1-D sequence)."""
        self._ids = np.asarray(ids, dtype=np.int64)
        self._order = np.argsort(self._ids, kind="stable")
        self._sorted = self._ids[self._order]

    def __len__(self) -> int:
        """Number of indexed IDs."""
        return len(self._ids)

    def to_numpy(self) -> np.ndarray:
        """Return the original (unsorted) ID array."""
        return self._ids

    def get_indexer(self, values) -> np.ndarray:
        """Row position of each value in the original ID array; -1 if absent."""
        vals = np.asarray(values, dtype=np.int64)
        out = np.full(vals.shape, -1, dtype=np.int64)
        n = self._sorted.size
        if n == 0 or vals.size == 0:
            return out
        pos = np.searchsorted(self._sorted, vals)
        ok = pos < n
        cand = np.where(ok, pos, 0)
        match = ok & (self._sorted[cand] == vals)
        out[match] = self._order[cand[match]]
        return out


@dataclass(frozen=True)
class ParentRefs:
    """The distinct ids one parent column references (sorted) and each one's own row, -1 when it has none."""

    ids: np.ndarray
    rows: np.ndarray

    @classmethod
    def of(cls, parents: np.ndarray, id_index: IdIndex) -> ParentRefs:
        """Collect the references in ``parents`` (``-1`` is no parent) and look up their rows."""
        ids = unique_ints(parents[parents != -1])
        return cls(ids, id_index.get_indexer(ids))


def _parent_rows(parents: np.ndarray, id_index: IdIndex) -> tuple[np.ndarray, np.ndarray]:
    """Map parent IDs to row indices; -1 for missing. Returns (row_index, present_mask)."""
    out = np.full(len(parents), -1, dtype=np.int64)
    mask = parents != -1
    if mask.any():
        out[mask] = id_index.get_indexer(parents[mask])
    return out, mask


@dataclass(frozen=True)
class MatingPairs:
    """Children with both parents known, grouped by **Mating Pair**.

    Pairs are numbered in ascending (mother id, father id) order and
    ``children`` is sorted by pair, so ``pair`` is non-decreasing.
    """

    children: np.ndarray  # row of each child with both parents known
    pair: np.ndarray  # pair number of each entry of ``children``
    sizes: np.ndarray  # children per pair
    mother_rows: np.ndarray  # mother row of each pair
    father_rows: np.ndarray  # father row of each pair


def _group_mating_pairs(df: pl.DataFrame, mother_rows: np.ndarray, father_rows: np.ndarray) -> MatingPairs:
    """Group the children of ``df`` by (mother, father); parent rows are ``-1`` when unknown."""
    # Sorting by parent id rather than row keeps the per-pair order, and with
    # it every float summed over ``sizes``, independent of row order.
    mothers = df["mother"].to_numpy()
    fathers = df["father"].to_numpy()
    both = np.flatnonzero((mother_rows >= 0) & (father_rows >= 0))
    children = both[np.lexsort((fathers[both], mothers[both]))]
    m, f = mothers[children], fathers[children]
    starts = np.ones(len(children), dtype=bool)
    starts[1:] = (m[1:] != m[:-1]) | (f[1:] != f[:-1])
    pair = np.cumsum(starts) - 1
    first_child = children[starts]
    return MatingPairs(
        children=children,
        pair=pair,
        sizes=np.bincount(pair),
        mother_rows=mother_rows[first_child],
        father_rows=father_rows[first_child],
    )


def _build_children_csr(mother_rows: np.ndarray, father_rows: np.ndarray) -> sp.csr_matrix | None:
    """Build the parent→child CSR from parent rows (``-1`` when unknown). None if no edges."""
    n = len(mother_rows)
    has_mom = mother_rows >= 0
    has_dad = father_rows >= 0
    parent_rows = np.concatenate([mother_rows[has_mom], father_rows[has_dad]])
    if len(parent_rows) == 0:
        return None
    child_rows = np.concatenate([np.where(has_mom)[0], np.where(has_dad)[0]])
    return sp.csr_matrix(
        (np.ones(len(parent_rows), dtype=np.int8), (parent_rows, child_rows)),
        shape=(n, n),
    )


def _concat_ranges(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """``np.concatenate([np.arange(a, b) for a, b in zip(starts, stops)])`` without the loop."""
    lengths = stops - starts
    return np.arange(lengths.sum()) + np.repeat(starts - (np.cumsum(lengths) - lengths), lengths)


def _structural_depth(mother_rows: np.ndarray, father_rows: np.ndarray) -> np.ndarray:
    """Per-row topological depth, tolerant of any input row order.

    Founders have depth 0, other rows ``max(parent_depth) + 1``.  Kahn's
    algorithm one level at a time: a row enters the frontier in the round its
    last parent leaves it, which is its depth.  Rows in a cycle, or descended
    from one, never enter it and keep depth -1.  Returns ``np.int32``.
    """
    n = len(mother_rows)
    parents = np.concatenate([mother_rows, father_rows])
    children = np.tile(np.arange(n), 2)
    known = parents >= 0
    parents, children = parents[known], children[known]
    children = children[np.argsort(parents, kind="stable")]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(parents, minlength=n), out=indptr[1:])
    unplaced = np.bincount(children, minlength=n)  # known parents not yet placed

    depth = np.full(n, -1, dtype=np.int32)
    frontier = np.flatnonzero(unplaced == 0)
    level = 0
    while frontier.size:
        depth[frontier] = level
        reached = children[_concat_ranges(indptr[frontier], indptr[frontier + 1])]
        reached, hits = np.unique(reached, return_counts=True)
        unplaced[reached] -= hits
        frontier = reached[unplaced[reached] == 0]
        level += 1
    return depth
