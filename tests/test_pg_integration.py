"""Regression tests for the pedsum → pedigree-graph consolidation.

These pin behaviors that could silently regress when the pedigree-graph
pin is later bumped:

- ``df["ped_depth"]`` after PG construction matches ``pg.generation``.
- ``compute_n_descendants`` numerically matches the pre-refactor pedsum
  implementation (path-count snapshot on the bundled example).
- The compact PG preserves sex (catches a silent regression of the
  ``from_arrays(sex=...)`` extension).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    df["ped_depth"] = np.asarray(pg.generation, dtype=np.int32)
    return pg, df


def test_ped_depth_matches_pg_generation():
    """``df['ped_depth']`` after construction equals ``pg.generation``."""
    pg, df = _build_pg_and_df()
    np.testing.assert_array_equal(df["ped_depth"].to_numpy(), pg.generation)
    assert int(df["ped_depth"].max()) == _SNAPSHOT["ped_depth_max"]


def test_compute_n_descendants_snapshot():
    """``pg.compute_n_descendants`` matches the pre-refactor snapshot totals."""
    pg, _ = _build_pg_and_df()
    n_desc = pg.compute_n_descendants()
    assert n_desc.dtype == np.int32
    assert int(n_desc.sum()) == _SNAPSHOT["n_descendants_sum"]
    assert int(n_desc.max()) == _SNAPSHOT["n_descendants_max"]


def test_compute_n_ancestors_snapshot():
    """``pg.compute_n_ancestors`` matches the pre-refactor snapshot totals."""
    pg, _ = _build_pg_and_df()
    n_anc = pg.compute_n_ancestors()
    assert n_anc.dtype == np.int32
    assert int(n_anc.sum()) == _SNAPSHOT["n_ancestors_sum"]
    assert int(n_anc.max()) == _SNAPSHOT["n_ancestors_max"]


def test_sex_preserved_through_compaction():
    """`_build_pedigree_graph` must thread `sex` through to the graph."""
    pg, df = _build_pg_and_df()
    # Compact ordering matches input ordering (compaction preserves
    # row order, only IDs are remapped to 0..n-1).
    np.testing.assert_array_equal(
        pg.sex.astype(np.int64), df["sex"].to_numpy().astype(np.int64),
    )
    # Spot-check the male/female totals match (a silent zeros default
    # would make pg.sex.sum() == 0).
    assert int(pg.sex.sum()) == int((df["sex"] == ps.SEX_MALE).sum())
