"""Tests for the ``--inbreeding`` gating of distinct-ancestor aggregates.

Pin that the default summarize path leaves the per-individual
``n_distinct_ancestors`` distribution and the genealogy / depth_summary
ancestor aggregates empty, and that ``--inbreeding`` flips them on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedigree_summary import (
    SEX_FEMALE,
    SEX_MALE,
    _build_individual_data,
    compute_aggregate_sections,
    compute_founder_summary,
)


def _individual_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "sex": [SEX_MALE, SEX_FEMALE, SEX_MALE],
            "mother": [-1, -1, 2],
            "father": [-1, -1, 1],
            "ped_depth": [0, 0, 1],
            "is_founder": [True, True, False],
            "F": [0.0, 0.0, 0.25],
            "n_full_sibs": [0, 0, 0],
            "n_mat_half_sibs": [0, 0, 0],
            "n_pat_half_sibs": [0, 0, 0],
            "n_offspring": [1, 1, 0],
            "n_mates": [1, 1, 0],
            "component_id": [0, 0, 0],
            "n_grandparents": [0, 0, 2],
            "n_grandchildren": [1, 1, 0],
            "n_uncles_aunts": [0, 0, 0],
            "n_first_cousins": [0, 0, 0],
            "n_founder_ancestors": np.array([1, 1, 2], dtype=np.int32),
            "n_distinct_ancestors": np.array([0, 0, 2], dtype=np.int32),
            "n_descendant_paths": np.array([1, 1, 0], dtype=np.int32),
        }
    )


def test_default_aggregate_sections_mark_ancestor_stats_unavailable() -> None:
    """Default mode emits None for genealogy/depth-summary ancestor stats."""
    idf = _individual_df()

    founder_summary, _ = compute_founder_summary(idf)
    aggregates = compute_aggregate_sections(
        idf,
        founder_summary=founder_summary,
        include_inbreeding=False,
    )

    assert aggregates["genealogy"]["distinct_ancestors"] is None
    assert all(row["mean_distinct_ancestors"] is None for row in aggregates["depth_summary"])


def test_default_individual_data_omits_ancestor_distribution() -> None:
    """Default mode omits ``F`` and ``n_distinct_ancestors`` from the distribution block."""
    idf = _individual_df()

    out = _build_individual_data(
        idf,
        Path("ped.tsv"),
        "pedigree_summary.py summarize --in ped.tsv --out out",
        include_inbreeding=False,
    )

    assert "F" not in out["distributions"]
    assert "n_distinct_ancestors" not in out["distributions"]


def test_inbreeding_mode_reports_ancestor_stats() -> None:
    """``--inbreeding`` populates genealogy, depth_summary, and per-individual ancestor stats."""
    idf = _individual_df()

    founder_summary, _ = compute_founder_summary(idf)
    aggregates = compute_aggregate_sections(
        idf,
        founder_summary=founder_summary,
        include_inbreeding=True,
    )
    out = _build_individual_data(
        idf,
        Path("ped.tsv"),
        "pedigree_summary.py summarize --in ped.tsv --out out --inbreeding",
        include_inbreeding=True,
    )

    assert aggregates["genealogy"]["distinct_ancestors"]["max"] == 2
    depth1 = next(row for row in aggregates["depth_summary"] if row["depth"] == 1)
    assert depth1["mean_distinct_ancestors"] == 2.0
    assert out["distributions"]["n_distinct_ancestors"]["max"] == 2
