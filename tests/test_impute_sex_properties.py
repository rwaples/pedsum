"""Property-based tests for role-based sex imputation in ``pedsum.validate``.

``_impute_sex_from_roles`` tags every row with a provenance category and may fill
unknown sex from mother/father usage. These tests assert the provenance partition
and its invariants (unresolved iff still unknown; ``input`` rows untouched;
override-off leaves asserted sexes alone) over random role/sex combinations.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN
from pedsum.validate import _impute_sex_from_roles

_CATEGORIES = {"input", "imputed_from_missing", "imputed_from_role", "unresolved"}


@st.composite
def _impute_inputs(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Build (ids, mothers, fathers, sex, override) with parents drawn from the id set."""
    n = draw(st.integers(min_value=1, max_value=15))
    ids = np.arange(n, dtype=np.int64)
    parent = st.one_of(st.just(-1), st.integers(min_value=0, max_value=n - 1))
    mothers = np.array([draw(parent) for _ in range(n)], dtype=np.int64)
    fathers = np.array([draw(parent) for _ in range(n)], dtype=np.int64)
    sex = np.array([draw(st.sampled_from([SEX_FEMALE, SEX_MALE, SEX_UNKNOWN])) for _ in range(n)], dtype=np.int8)
    override = draw(st.booleans())
    return ids, mothers, fathers, sex, override


@settings(deadline=None)
@given(data=_impute_inputs())
def test_impute_partition_and_invariants(data: tuple) -> None:
    """sex_source partitions rows; unresolved iff still unknown; input/override rules hold."""
    ids, mothers, fathers, sex, override = data
    res = _impute_sex_from_roles(sex, ids, mothers, fathers, override_asserted_sex=override)
    src = res.sex_source
    imp = res.imputed_sex
    n = len(ids)

    assert all(tag in _CATEGORIES for tag in src)
    for i in range(n):
        assert (imp[i] == SEX_UNKNOWN) == (src[i] == "unresolved")
        if src[i] == "input":
            assert imp[i] == sex[i]
        if src[i] == "imputed_from_missing":
            assert sex[i] == SEX_UNKNOWN

    if not override:
        assert "imputed_from_role" not in set(src)
        for i in range(n):
            if sex[i] != SEX_UNKNOWN:
                assert imp[i] == sex[i]


def test_impute_role_directions() -> None:
    """An unknown id used only as mother becomes female; only as father becomes male."""
    ids = np.array([0, 1, 2, 3], dtype=np.int64)
    sex = np.array([SEX_UNKNOWN, SEX_UNKNOWN, SEX_FEMALE, SEX_MALE], dtype=np.int8)
    mothers = np.array([-1, -1, 0, -1], dtype=np.int64)  # id 0 used only as mother
    fathers = np.array([-1, -1, -1, 1], dtype=np.int64)  # id 1 used only as father
    res = _impute_sex_from_roles(sex, ids, mothers, fathers)
    assert res.imputed_sex[0] == SEX_FEMALE
    assert res.imputed_sex[1] == SEX_MALE


def test_impute_ambiguous_role_stays_unresolved() -> None:
    """An unknown id used as both mother and father stays unresolved and is flagged ambiguous."""
    ids = np.array([0, 1, 2], dtype=np.int64)
    sex = np.array([SEX_UNKNOWN, SEX_UNKNOWN, SEX_UNKNOWN], dtype=np.int8)
    mothers = np.array([-1, 0, -1], dtype=np.int64)  # id 0 used as mother
    fathers = np.array([-1, -1, 0], dtype=np.int64)  # id 0 also used as father
    res = _impute_sex_from_roles(sex, ids, mothers, fathers)
    assert res.imputed_sex[0] == SEX_UNKNOWN
    assert res.sex_source[0] == "unresolved"
    assert res.ambiguous_mask[0]
