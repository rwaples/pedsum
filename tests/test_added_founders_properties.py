"""Property tests for validate-output added founder synthesis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from pedsum.report import _build_added_founders

FounderReferences = tuple[np.ndarray, np.ndarray, np.ndarray]


@st.composite
def founder_reference_arrays(draw: st.DrawFn) -> FounderReferences:
    """Generate present IDs and parent-reference arrays with optional missing parent IDs."""
    ids = sorted(draw(st.lists(st.integers(min_value=0, max_value=80), unique=True, max_size=25)))
    n = len(ids)
    missing_candidates = sorted(
        set(draw(st.lists(st.integers(min_value=0, max_value=100), unique=True, max_size=20))) - set(ids)
    )
    parent_choices = [-1, *ids, *missing_candidates]
    mothers = draw(st.lists(st.sampled_from(parent_choices), min_size=n, max_size=n))
    fathers = draw(st.lists(st.sampled_from(parent_choices), min_size=n, max_size=n))
    return (
        np.array(ids, dtype=np.int64),
        np.array(mothers, dtype=np.int64),
        np.array(fathers, dtype=np.int64),
    )


def _missing_role_sets(
    ids: np.ndarray,
    mothers: np.ndarray,
    fathers: np.ndarray,
) -> tuple[set[int], set[int], set[int]]:
    """Return missing mother-only, father-only, and role-conflict parent IDs."""
    present = set(ids.tolist())
    moms = set(mothers[mothers != -1].tolist()) - present
    dads = set(fathers[fathers != -1].tolist()) - present
    conflicts = moms & dads
    return moms - conflicts, dads - conflicts, conflicts


@given(founder_reference_arrays(), st.booleans())
def test_added_founder_ids_are_missing_parent_references(
    generated: FounderReferences,
    no_sex_check: bool,
) -> None:
    """Added founder IDs are exactly missing parent references allowed by the sex-check flag."""
    ids, mothers, fathers = generated
    moms_only, dads_only, conflicts = _missing_role_sets(ids, mothers, fathers)
    expected = moms_only | dads_only
    if no_sex_check:
        expected |= conflicts

    founders = _build_added_founders(mothers, fathers, pd.Index(ids), no_sex_check=no_sex_check)
    added_ids = [row["id"] for row in founders]

    assert added_ids == sorted(expected)
    assert len(added_ids) == len(set(added_ids))
    assert not (set(added_ids) & set(ids.tolist()))


@given(founder_reference_arrays())
def test_added_founder_sex_follows_unambiguous_parent_role(generated: FounderReferences) -> None:
    """Mother-only missing IDs become F and father-only missing IDs become M."""
    ids, mothers, fathers = generated
    moms_only, dads_only, conflicts = _missing_role_sets(ids, mothers, fathers)
    founders = _build_added_founders(mothers, fathers, pd.Index(ids), no_sex_check=True)
    by_id = {row["id"]: row for row in founders}

    for mid in moms_only:
        assert by_id[mid]["sex"] == "F"
    for did in dads_only:
        assert by_id[did]["sex"] == "M"
    for cid in conflicts:
        assert by_id[cid]["sex"] == "F"
        assert "--no-sex-check" in by_id[cid]["reason"]


@given(founder_reference_arrays())
def test_role_conflict_founders_require_no_sex_check(generated: FounderReferences) -> None:
    """Missing IDs used as both mother and father are omitted unless sex checks are disabled."""
    ids, mothers, fathers = generated
    _, _, conflicts = _missing_role_sets(ids, mothers, fathers)

    strict_ids = {row["id"] for row in _build_added_founders(mothers, fathers, pd.Index(ids), no_sex_check=False)}
    permissive_ids = {row["id"] for row in _build_added_founders(mothers, fathers, pd.Index(ids), no_sex_check=True)}

    assert strict_ids.isdisjoint(conflicts)
    assert conflicts <= permissive_ids
