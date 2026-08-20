"""Low-level pedigree array helpers (parent rows, sib groups, depth)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from pedsum.base import PedigreeError

if TYPE_CHECKING:
    import polars as pl


def _id_list(ids, max_show: int = 5) -> str:
    ids = list(ids)
    if len(ids) <= max_show:
        return ", ".join(str(i) for i in ids)
    return ", ".join(str(i) for i in ids[:max_show]) + f", ... ({len(ids)} total)"


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


def _parent_rows(parents: np.ndarray, id_index: IdIndex) -> tuple[np.ndarray, np.ndarray]:
    """Map parent IDs to row indices; -1 for missing. Returns (row_index, present_mask)."""
    out = np.full(len(parents), -1, dtype=np.int64)
    mask = parents != -1
    if mask.any():
        out[mask] = id_index.get_indexer(parents[mask])
    return out, mask


def _group_sizes_per_row(keys: np.ndarray) -> np.ndarray:
    """Size of each row's group, where ``keys`` labels the group of each row.

    Vectorized ``groupby(key).size()`` + per-row lookup: rows sharing a key
    value all receive that key's total count. ``keys`` may be any 1-D or 2-D
    (row-wise composite key) integer array.
    """
    if keys.ndim == 1:
        _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    else:
        _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return counts[inv]


def _full_sib_groups(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row full-sib counts plus underlying mating-pair group sizes.

    Returns (fs_count, fs_group_sizes, both_present) where fs_count[i] is the
    number of full sibs of row i (0 if either parent is unknown),
    fs_group_sizes is the array of distinct (mother, father) group sizes, and
    both_present is the boolean row mask for rows with both parents known.
    """
    n = len(df)
    fs_count = np.zeros(n, dtype=np.int64)
    mothers = df["mother"].to_numpy()
    fathers = df["father"].to_numpy()
    both_present = (mothers != -1) & (fathers != -1)
    if not both_present.any():
        return fs_count, np.array([], dtype=np.int64), both_present
    pair_keys = np.column_stack([mothers[both_present], fathers[both_present]])
    _, inv, counts = np.unique(pair_keys, axis=0, return_inverse=True, return_counts=True)
    fs_count[np.where(both_present)[0]] = counts[inv] - 1
    return fs_count, counts, both_present


def _grandparent_arrays(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mm, mf, fm, ff) arrays of grandparent IDs (-1 for unknown)."""
    id_index = IdIndex(df["id"].to_numpy())
    mothers = df["mother"].to_numpy()
    fathers = df["father"].to_numpy()
    parent_cols = {"mother": mothers, "father": fathers}

    def _lookup(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
        rows, present = _parent_rows(outer, id_index)
        out = np.full(len(outer), -1, dtype=np.int64)
        hit = present & (rows != -1)
        out[hit] = inner[rows[hit]]
        return out

    return tuple(  # ty: ignore[invalid-return-type]
        _lookup(parent_cols[outer], parent_cols[inner])
        for outer, inner in (
            ("mother", "mother"),
            ("mother", "father"),
            ("father", "mother"),
            ("father", "father"),
        )
    )


def _build_children_csr(
    m_row: np.ndarray,
    mask_m: np.ndarray,
    f_row: np.ndarray,
    mask_f: np.ndarray,
    n: int,
) -> sp.csr_matrix | None:
    """Build parent→child CSR from parent-row arrays. None if no edges."""
    parent_rows = np.concatenate([m_row[mask_m], f_row[mask_f]])
    if len(parent_rows) == 0:
        return None
    child_rows = np.concatenate([np.where(mask_m)[0], np.where(mask_f)[0]])
    return sp.csr_matrix(
        (np.ones(len(parent_rows), dtype=np.int8), (parent_rows, child_rows)),
        shape=(n, n),
    )


def _compute_depth_unordered(
    mother_rows: np.ndarray,
    father_rows: np.ndarray,
    n: int,
) -> np.ndarray:
    """Per-row topological depth, tolerant of any input row order.

    Vectorized fixed-point sweep: founders have depth 0, other rows have
    ``max(parent_depth) + 1``.  Iterates until depths stabilise — unlike
    a single Kahn pass, this does not require parents to already precede
    children in row index.  Returns ``np.int32``.  Raises ``PedigreeError``
    when no progress can be made (true cycle).
    """
    depth = np.full(n, -1, dtype=np.int32)
    depth[(mother_rows < 0) & (father_rows < 0)] = 0
    while True:
        todo = depth < 0
        if not todo.any():
            return depth
        m_safe = np.maximum(mother_rows, 0)
        f_safe = np.maximum(father_rows, 0)
        m_resolved = (mother_rows < 0) | (depth[m_safe] >= 0)
        f_resolved = (father_rows < 0) | (depth[f_safe] >= 0)
        eligible = todo & m_resolved & f_resolved
        if not eligible.any():
            unresolved = np.where(todo)[0][:5].tolist()
            raise PedigreeError(
                f"pedigree contains a cycle: {int(todo.sum())} individual(s) "
                f"could not be topologically ordered (e.g. rows {unresolved})",
            )
        md = np.where(mother_rows >= 0, depth[m_safe], 0)
        fd = np.where(father_rows >= 0, depth[f_safe], 0)
        depth[eligible] = np.maximum(md[eligible], fd[eligible]) + 1
