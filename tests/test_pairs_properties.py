"""Property-based tests for relationship-pair count aggregation in ``pedsum.pairs``.

``_augment_pair_counts`` derives ``PO`` and a ``by_degree`` rollup from a dict of
raw relationship-code counts. These tests assert the conservation invariants
(``PO == MO + FO``; the degree buckets reproduce the input total) over arbitrary
count dicts, which the example tests (driven by a single bundled pedigree) cannot.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st
from pedigree_graph import REL_REGISTRY

from pedsum.pairs import _augment_pair_counts

# Raw registry codes only: these include MO/FO but NOT the derived PO/engine,
# which _augment_pair_counts adds itself (and which are absent from REL_REGISTRY,
# so feeding them in would KeyError on REL_REGISTRY[code].degree).
_CODES = sorted(REL_REGISTRY.keys())


@given(named=st.dictionaries(st.sampled_from(_CODES), st.integers(min_value=0, max_value=1_000_000)))
def test_augment_pair_counts_conserves(named: dict) -> None:
    """PO equals MO+FO; by_degree buckets reproduce the input total; codes are preserved."""
    original = dict(named)
    out = _augment_pair_counts(named)

    assert out["PO"] == named.get("MO", 0) + named.get("FO", 0)
    for code, count in named.items():
        assert out[code] == int(count)

    by_degree = out["by_degree"]
    assert set(by_degree.keys()) == set(range(6))
    assert sum(by_degree.values()) == sum(int(v) for v in named.values())
    for degree in range(6):
        expected = sum(int(named[c]) for c in named if REL_REGISTRY[c].degree == degree)
        assert by_degree[degree] == expected

    # The function must not mutate its input dict.
    assert named == original
