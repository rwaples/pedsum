"""Property-based tests for the slim/extra YAML split machinery in ``pedsum.schema``.

The split functions claim a *partition* contract: every leaf key of the input
appears in exactly one of (slim, extra, intentional-drop), with none lost and
none duplicated. These tests assert that contract via key-set algebra over
Hypothesis-generated dicts, so the routing rules are exercised on structures the
example-driven suite never produces.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from pedsum.schema import (
    _PAIRS_SLIM_KEYS,
    _SEX_CONCORDANCE_SLIM_KEYS,
    _SEX_SUMMARY_SLIM_KEYS,
    INDIVIDUAL_SLIM_COLS,
    INDIVIDUAL_SLIM_DIST_KEYS,
    _is_effective_size_array_key,
    _split_effective_size,
    _split_individual_distributions,
    _split_offspring_sex_concordance,
    _split_sex_summary,
    _split_summary,
)

_SCALARS = st.integers(min_value=-1000, max_value=1000)

# Effective-size key pools: names/suffixes that route to extra vs slim.
_ES_ARRAY_KEYS = [
    "ne_per_gen",
    "ne_per_transition",
    "x_per_cohort",
    "cohort_years",
    "age_table",
    "v_mm",
    "v_ff",
    "cov_m",
]
_ES_SCALAR_KEYS = ["ne", "se", "ci_low", "ci_high", "cohort_window", "method", "n"]
_ES_NAMES = ["ne_inbreeding", "ne_variance_family_size", "ne_hill_overlapping"]

_SEX_EXTRA_KEYS = ["offspring_count", "mate_count", "F_dist", "age"]

_DIST_KEYS = ["mean", "median", "min", "q1", "q3", "max", "std", "nz"]
_OTHER_DIST_COLS = ["age", "depth", "custom"]

_PAIRS_EXTRA_KEYS = ["foo", "bar", "baz", "quux"]

_CONCORDANCE_GROUPINGS = ["sibship", "maternal_offspring_group", "paternal_offspring_group"]
_CONCORDANCE_EXTRA_KEYS = ["z", "p_analytical", "n_groups_total", "by_group_size", "all_resolved"]


@st.composite
def _es_dicts(draw: st.DrawFn) -> dict:
    """Build an effective_size dict mapping estimators to mixed array/scalar keys."""
    names = draw(st.lists(st.sampled_from(_ES_NAMES), unique=True, max_size=3))
    out: dict = {}
    for name in names:
        keys = draw(st.lists(st.sampled_from(_ES_ARRAY_KEYS + _ES_SCALAR_KEYS), unique=True, max_size=6))
        out[name] = {k: draw(_SCALARS) for k in keys}
    return out


@st.composite
def _sex_dicts(draw: st.DrawFn) -> dict:
    """Build a sex_summary dict mapping strata to mixed scalar/distribution keys."""
    strata = draw(st.lists(st.sampled_from(["all", "male", "female"]), unique=True, max_size=3))
    out: dict = {}
    for stratum in strata:
        keys = draw(
            st.lists(st.sampled_from([*_SEX_SUMMARY_SLIM_KEYS, *_SEX_EXTRA_KEYS]), unique=True, max_size=6),
        )
        out[stratum] = {k: draw(_SCALARS) for k in keys}
    return out


@st.composite
def _dist_dicts(draw: st.DrawFn) -> dict:
    """Build a per-individual distributions dict mapping columns to quantile keys."""
    cols = draw(st.lists(st.sampled_from([*INDIVIDUAL_SLIM_COLS, *_OTHER_DIST_COLS]), unique=True, max_size=5))
    out: dict = {}
    for col in cols:
        keys = draw(st.lists(st.sampled_from(_DIST_KEYS), unique=True, max_size=8))
        out[col] = {k: draw(_SCALARS) for k in keys}
    return out


@st.composite
def _rel_pairs(draw: st.DrawFn) -> dict:
    """Build a relationship_pairs dict mixing slim keys, extra keys, and by_degree."""
    slim = draw(st.lists(st.sampled_from(list(_PAIRS_SLIM_KEYS)), unique=True, max_size=5))
    extra = draw(st.lists(st.sampled_from(_PAIRS_EXTRA_KEYS), unique=True, max_size=3))
    out: dict = {k: draw(_SCALARS) for k in slim}
    out.update({k: draw(_SCALARS) for k in extra})
    if draw(st.booleans()):
        out["by_degree"] = {0: draw(_SCALARS)}
    return out


@settings(deadline=None)
@given(es=_es_dicts())
def test_split_effective_size_partitions_keys(es: dict) -> None:
    """Every estimator key lands in slim or extra (disjoint), routed by array-ness."""
    slim, extra = _split_effective_size(es)
    array_keys = set(_ES_ARRAY_KEYS)
    for name, value in es.items():
        slim_keys = set(slim.get(name, {}))
        extra_keys = set(extra.get(name, {}))
        assert slim_keys & extra_keys == set()
        assert slim_keys | extra_keys == set(value)
        assert all(_is_effective_size_array_key(k) for k in extra_keys)
        assert not any(_is_effective_size_array_key(k) for k in slim_keys)
        # Every array-named key is routed correctly regardless of the pool above.
        assert {k for k in value if k in array_keys} <= extra_keys


def test_split_effective_size_ne_coancestry_none_stub() -> None:
    """``ne_coancestry`` with ``ne`` None collapses to a slim-only ``{ne: None}`` stub."""
    es = {"ne_coancestry": {"ne": None, "ne_per_gen": [1, 2], "se": 0.1}}
    slim, extra = _split_effective_size(es)
    assert slim["ne_coancestry"] == {"ne": None}
    assert "ne_coancestry" not in extra


@settings(deadline=None)
@given(sex=_sex_dicts())
def test_split_sex_summary_partitions_keys(sex: dict) -> None:
    """Per stratum, scalar keys go to slim and the rest to extra, with no overlap."""
    slim, extra = _split_sex_summary(sex)
    slim_keyset = set(_SEX_SUMMARY_SLIM_KEYS)
    for stratum, value in sex.items():
        slim_keys = set(slim.get(stratum, {}))
        extra_keys = set(extra.get(stratum, {}))
        assert slim_keys & extra_keys == set()
        assert slim_keys | extra_keys == set(value)
        assert slim_keys <= slim_keyset
        assert extra_keys & slim_keyset == set()


@st.composite
def _concordance_dicts(draw: st.DrawFn) -> dict:
    """Build an offspring_sex_concordance dict: null_model plus grouping blocks."""
    groupings = draw(st.lists(st.sampled_from(_CONCORDANCE_GROUPINGS), unique=True, max_size=3))
    out: dict = {}
    if draw(st.booleans()):
        out["null_model"] = {"method": "fixed_margin_exchangeability", "min_group_size": 2}
    for name in groupings:
        keys = draw(
            st.lists(
                st.sampled_from([*_SEX_CONCORDANCE_SLIM_KEYS, *_CONCORDANCE_EXTRA_KEYS]),
                unique=True,
                max_size=8,
            ),
        )
        out[name] = {k: draw(_SCALARS) for k in keys}
    return out


@settings(deadline=None)
@given(conc=_concordance_dicts())
def test_split_offspring_sex_concordance_partitions_keys(conc: dict) -> None:
    """Per grouping, slim keys go to slim and the rest to extra; null_model stays slim."""
    slim, extra = _split_offspring_sex_concordance(conc)
    slim_keyset = set(_SEX_CONCORDANCE_SLIM_KEYS)
    if "null_model" in conc:
        assert slim["null_model"] == conc["null_model"]
        assert "null_model" not in extra
    for name, value in conc.items():
        if name == "null_model":
            continue
        slim_keys = set(slim.get(name, {}))
        extra_keys = set(extra.get(name, {}))
        assert slim_keys & extra_keys == set()
        assert slim_keys | extra_keys == set(value)
        assert slim_keys <= slim_keyset
        assert extra_keys & slim_keyset == set()


@settings(deadline=None)
@given(dists=_dist_dicts())
def test_split_individual_distributions_partitions_with_fmax_drop(dists: dict) -> None:
    """Per column, slim keeps mean/median for headline cols; extra holds the rest, less F.max."""
    slim, extra = _split_individual_distributions(dists)
    slim_cols = set(INDIVIDUAL_SLIM_COLS)
    slim_dist_keys = set(INDIVIDUAL_SLIM_DIST_KEYS)
    for col, dist in dists.items():
        keys = set(dist)
        slim_keys = set(slim.get(col, {}))
        extra_keys = set(extra.get(col, {}))
        assert slim_keys & extra_keys == set()
        if col in slim_cols:
            assert slim_keys == keys & slim_dist_keys
            expected_extra = keys - slim_dist_keys
            if col == "F":
                expected_extra -= {"max"}
            assert extra_keys == expected_extra
        else:
            assert slim_keys == set()
            assert extra_keys == keys


@settings(deadline=None)
@given(
    rp=_rel_pairs(),
    size=st.dictionaries(st.sampled_from(["n", "n_components", "x"]), _SCALARS, min_size=1),
)
def test_split_summary_relationship_pairs_split_and_by_degree_drop(rp: dict, size: dict) -> None:
    """relationship_pairs splits by slim keys; by_degree is dropped; spec-less sections stay slim."""
    nested = {
        "relatedness": {"relationship_pairs": dict(rp)},
        "structure": {"size_structure": dict(size)},
    }
    slim, extra = _split_summary(nested)
    pairs_slim_set = set(_PAIRS_SLIM_KEYS)

    expected_slim = {k: v for k, v in rp.items() if k in pairs_slim_set}
    expected_extra = {k: v for k, v in rp.items() if k not in pairs_slim_set and k != "by_degree"}
    got_slim = slim.get("relatedness", {}).get("relationship_pairs", {})
    got_extra = extra.get("relatedness", {}).get("relationship_pairs", {})

    assert got_slim == expected_slim
    assert got_extra == expected_extra
    assert "by_degree" not in got_slim
    assert "by_degree" not in got_extra

    # A section with no slim_keys spec goes wholesale to slim, never to extra.
    assert slim["structure"]["size_structure"] == size
    assert "structure" not in extra
