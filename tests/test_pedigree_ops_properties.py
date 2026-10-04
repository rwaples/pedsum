"""Property-based tests for low-level pedigree array helpers in ``pedsum.pedigree_ops``.

Covers the order-tolerant topological depth (row-order invariance,
idempotence, cycle marking), mating-pair grouping (agreement with a
``np.unique`` grouping), and parent-row resolution (mask/row consistency). Inputs are random acyclic
pedigrees with ids ``0..n-1`` emitted in topological order.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from pedsum.pedigree_ops import IdIndex, _group_mating_pairs, _parent_rows, _structural_depth


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
    pos = {id_: row for row, id_ in enumerate(order)}
    m_rows = np.array([pos[int(mothers[i])] if mothers[i] != -1 else -1 for i in order], dtype=np.int64)
    f_rows = np.array([pos[int(fathers[i])] if fathers[i] != -1 else -1 for i in order], dtype=np.int64)
    depth = _structural_depth(m_rows, f_rows)
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


def test_depth_marks_cycle_and_descendants() -> None:
    """Rows in a cycle, and rows descended from one, keep depth -1; the rest are placed."""
    # rows 0 and 1 are each other's mother; row 3 has row 0 as mother and founder row 2 as father.
    mothers = np.array([1, 0, -1, 0], dtype=np.int64)
    fathers = np.array([-1, -1, -1, 2], dtype=np.int64)
    np.testing.assert_array_equal(_structural_depth(mothers, fathers), [-1, -1, 0, -1])


@settings(deadline=None)
@given(ped=_pedigrees(), data=st.data())
def test_mating_pairs_match_unique_grouping(ped: tuple, data: st.DataObject) -> None:
    """Pairs, sizes, and parent rows agree with ``np.unique(axis=0)``, in any row order."""
    ids, mothers, fathers = ped
    order = data.draw(st.permutations(ids))
    df = pl.DataFrame({"id": ids, "mother": mothers, "father": fathers})[order]
    id_index = IdIndex(df["id"].to_numpy())
    m_rows, _ = _parent_rows(df["mother"].to_numpy(), id_index)
    f_rows, _ = _parent_rows(df["father"].to_numpy(), id_index)
    mating = _group_mating_pairs(df, m_rows, f_rows)

    both = np.flatnonzero((m_rows >= 0) & (f_rows >= 0))
    assert sorted(mating.children.tolist()) == both.tolist()
    assert (np.diff(mating.pair) >= 0).all()
    keys = np.column_stack([df["mother"].to_numpy()[both], df["father"].to_numpy()[both]])
    if both.size:
        _, sizes = np.unique(keys, axis=0, return_counts=True)
        np.testing.assert_array_equal(mating.sizes, sizes)
    else:
        assert mating.sizes.size == 0
    np.testing.assert_array_equal(mating.mother_rows[mating.pair], m_rows[mating.children])
    np.testing.assert_array_equal(mating.father_rows[mating.pair], f_rows[mating.children])


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
