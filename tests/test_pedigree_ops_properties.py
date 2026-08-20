"""Property-based tests for low-level pedigree array helpers in ``pedsum.pedigree_ops``.

Covers the order-tolerant topological-depth sweep (row-order invariance,
idempotence, cycle detection), full-sib counting (conservation per mating pair),
and parent-row resolution (mask/row consistency). Inputs are random acyclic
pedigrees with ids ``0..n-1`` emitted in topological order.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pedsum.base import PedigreeError
from pedsum.pedigree_ops import IdIndex, _compute_depth_unordered, _full_sib_groups, _parent_rows


@st.composite
def _pedigrees(draw: st.DrawFn) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Build an acyclic pedigree (ids 0..n-1 in topological order) as id/parent arrays."""
    n = draw(st.integers(min_value=1, max_value=12))
    mothers: list[int] = []
    fathers: list[int] = []
    for i in range(n):
        if i == 0:
            mothers.append(-1)
            fathers.append(-1)
            continue
        # Parents are drawn from strictly earlier ids, so the pedigree is acyclic.
        mothers.append(draw(st.one_of(st.just(-1), st.integers(min_value=0, max_value=i - 1))))
        fathers.append(draw(st.one_of(st.just(-1), st.integers(min_value=0, max_value=i - 1))))
    ids = list(range(n))
    return ids, np.array(mothers, dtype=np.int64), np.array(fathers, dtype=np.int64)


def _depth_by_id(order: list[int], mothers: np.ndarray, fathers: np.ndarray) -> dict[int, int]:
    """Compute per-id depth for a given row ``order`` of the same pedigree."""
    n = len(order)
    pos = {id_: row for row, id_ in enumerate(order)}
    m_rows = np.array([pos[int(mothers[i])] if mothers[i] != -1 else -1 for i in order], dtype=np.int64)
    f_rows = np.array([pos[int(fathers[i])] if fathers[i] != -1 else -1 for i in order], dtype=np.int64)
    depth = _compute_depth_unordered(m_rows, f_rows, n)
    return {id_: int(depth[row]) for row, id_ in enumerate(order)}


@settings(deadline=None)
@given(ped=_pedigrees(), data=st.data())
def test_depth_row_order_invariant_and_monotone(ped: tuple, data: st.DataObject) -> None:
    """Depth is invariant to row order and idempotent; founders are 0; children outrank parents."""
    ids, mothers, fathers = ped
    base = _depth_by_id(ids, mothers, fathers)

    permuted = data.draw(st.permutations(ids))
    assert _depth_by_id(permuted, mothers, fathers) == base
    assert _depth_by_id(ids, mothers, fathers) == base  # idempotence

    for i in ids:
        is_founder = mothers[i] == -1 and fathers[i] == -1
        assert (base[i] == 0) == is_founder
        if mothers[i] != -1:
            assert base[i] > base[int(mothers[i])]
        if fathers[i] != -1:
            assert base[i] > base[int(fathers[i])]


def test_depth_detects_cycle() -> None:
    """A 2-row mutual-parent cycle has no founder, so the sweep raises ``PedigreeError``."""
    mothers = np.array([1, 0], dtype=np.int64)
    fathers = np.array([-1, -1], dtype=np.int64)
    with pytest.raises(PedigreeError):
        _compute_depth_unordered(mothers, fathers, 2)


@settings(deadline=None)
@given(ped=_pedigrees())
def test_full_sib_counts_conserved(ped: tuple) -> None:
    """fs_count is non-negative, zero when a parent is missing, and sums to n(n-1) per mating pair."""
    ids, mothers, fathers = ped
    df = pl.DataFrame({"id": ids, "mother": mothers, "father": fathers})
    fs_count, _, _ = _full_sib_groups(df)

    assert (fs_count >= 0).all()
    missing_parent = (mothers == -1) | (fathers == -1)
    assert (fs_count[missing_parent] == 0).all()
    assert (~missing_parent)[fs_count > 0].all()  # fs_count > 0 implies both parents present

    both_present_rows = np.where(~missing_parent)[0]
    pair_keys = np.column_stack([mothers[both_present_rows], fathers[both_present_rows]])
    if both_present_rows.size:
        _, inv = np.unique(pair_keys, axis=0, return_inverse=True)
        for group in range(int(inv.max()) + 1):
            rows = both_present_rows[inv == group]
            size = len(rows)
            assert fs_count[rows].sum() == size * (size - 1)


@settings(deadline=None)
@given(ped=_pedigrees())
def test_parent_rows_mask_and_index(ped: tuple) -> None:
    """The present-mask matches ``!= -1``; resolved rows are in range and map back to the parent id."""
    ids, mothers, fathers = ped
    id_index = IdIndex(ids)
    n = len(ids)
    for parents in (mothers, fathers):
        rows, mask = _parent_rows(parents, id_index)
        assert (mask == (parents != -1)).all()
        assert (rows[~mask] == -1).all()
        present = rows[mask]
        assert ((present >= 0) & (present < n)).all()
        assert (id_index.to_numpy()[present] == parents[mask]).all()
