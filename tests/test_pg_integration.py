"""Regression tests for the pedsum → pedigree-graph consolidation.

These pin behaviors that could silently regress when the pedigree-graph
pin is later bumped:

- ``df["ped_depth"]`` after PG construction matches ``pg.depth``.
- ``descendant_path_counts`` numerically matches the pre-refactor pedsum
  implementation (path-count snapshot on the bundled example).
- ``distinct_ancestor_counts`` matches the same snapshot.
- The compact PG preserves sex (catches a silent regression of the
  ``from_arrays(sex=...)`` extension).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

import pedigree_summary as ps

EXAMPLE = Path(__file__).resolve().parents[1] / "example_pedigree.tsv"

# Snapshot of pre-refactor pedsum outputs on example_pedigree.tsv,
# captured before the consolidation landed.  These check both
# numerical agreement of the moved primitives and the implied claim
# that the pedigree-graph algorithms reproduce pedsum's behavior.
_SNAPSHOT = {
    "n_ancestors_sum": 1813,
    "n_ancestors_max": 29,
    "n_descendants_sum": 1980,
    "n_descendants_max": 74,
    "ped_depth_max": 4,
}


def _build_pg_and_df():
    df, _ = ps.load_and_validate(EXAMPLE)
    pg = ps._build_pedigree_graph(df)
    df = df.with_columns(pl.Series("ped_depth", np.asarray(pg.depth, dtype=np.int32)))
    return pg, df


def test_ped_depth_matches_pg_depth():
    """``df['ped_depth']`` after construction equals ``pg.depth``."""
    pg, df = _build_pg_and_df()
    np.testing.assert_array_equal(df["ped_depth"].to_numpy(), pg.depth)
    assert int(df["ped_depth"].max()) == _SNAPSHOT["ped_depth_max"]


def test_no_generation_labels_without_a_generation_column():
    """No generation column is supplied, so depth is the only label available."""
    pg, _ = _build_pg_and_df()
    assert pg.generation_labels is None


def test_descendant_path_counts_snapshot():
    """``pg.descendant_path_counts`` matches the pre-refactor snapshot totals."""
    pg, _ = _build_pg_and_df()
    n_desc = pg.descendant_path_counts()
    # int64 upstream: a path count multiplies along inbreeding loops and is not
    # bounded by the row count the way a distinct-individual count is.
    assert n_desc.dtype == np.int64
    assert int(n_desc.sum()) == _SNAPSHOT["n_descendants_sum"]
    assert int(n_desc.max()) == _SNAPSHOT["n_descendants_max"]


def test_distinct_ancestor_counts_snapshot():
    """``pg.distinct_ancestor_counts`` matches the pre-refactor snapshot totals."""
    pg, _ = _build_pg_and_df()
    n_anc = pg.distinct_ancestor_counts()
    assert n_anc.dtype == np.int32
    assert int(n_anc.sum()) == _SNAPSHOT["n_ancestors_sum"]
    assert int(n_anc.max()) == _SNAPSHOT["n_ancestors_max"]


def test_sex_preserved_through_compaction():
    """`_build_pedigree_graph` must thread `sex` through to the graph."""
    pg, df = _build_pg_and_df()
    # Compact ordering matches input ordering (compaction preserves
    # row order, only IDs are remapped to 0..n-1).
    np.testing.assert_array_equal(
        pg.sex.astype(np.int64),
        df["sex"].to_numpy().astype(np.int64),
    )
    # Spot-check the male/female totals match (a silent zeros default
    # would make pg.sex.sum() == 0).
    assert int(pg.sex.sum()) == int((df["sex"].to_numpy() == ps.SEX_MALE).sum())
