"""Property tests for relationship-summary conservation rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pedigree_graph import REL_REGISTRY

from pedsum.sections import compute_relationship_summary

PairLists = dict[str, tuple[np.ndarray, np.ndarray]]
RelationshipInput = tuple[pd.DataFrame, PairLists]

_CODE_BY_DEGREE = {
    degree: next(code for code, rel in REL_REGISTRY.items() if rel.degree == degree) for degree in range(1, 6)
}


@st.composite
def relationship_inputs(draw: st.DrawFn) -> RelationshipInput:
    """Generate row-index pair lists, including duplicates, reversals, and self-pairs."""
    n = draw(st.integers(min_value=0, max_value=25))
    depths = draw(st.lists(st.integers(min_value=0, max_value=5), min_size=n, max_size=n))
    df = pd.DataFrame({"ped_depth": np.array(depths, dtype=np.int32)})
    if n == 0:
        return df, {}

    triples = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=5),
                st.integers(min_value=0, max_value=n - 1),
                st.integers(min_value=0, max_value=n - 1),
            ),
            max_size=100,
        )
    )
    pair_parts: dict[str, tuple[list[int], list[int]]] = {}
    for degree, a, b in triples:
        code = _CODE_BY_DEGREE[degree]
        left, right = pair_parts.setdefault(code, ([], []))
        left.append(a)
        right.append(b)
    pair_lists = {
        code: (np.array(left, dtype=np.int64), np.array(right, dtype=np.int64))
        for code, (left, right) in pair_parts.items()
    }
    return df, pair_lists


def _closest_pair_degrees(pair_lists: PairLists) -> dict[tuple[int, int], int]:
    """Return the minimum observed relationship degree for each unordered non-self pair."""
    closest: dict[tuple[int, int], int] = {}
    for code, (left, right) in pair_lists.items():
        degree = REL_REGISTRY[code].degree
        for left_row, right_row in zip(left, right, strict=True):
            a = int(left_row)
            b = int(right_row)
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            closest[key] = min(closest.get(key, degree), degree)
    return closest


@given(st.lists(st.integers(min_value=0, max_value=5), max_size=25))
def test_relationship_summary_skip_branch_uses_individual_pair_name(depths: list[int]) -> None:
    """Skipped summaries still report the canonical Individual Pair denominator."""
    df = pd.DataFrame({"ped_depth": np.array(depths, dtype=np.int32)})
    summary = compute_relationship_summary(df, None)
    assert summary["computed"] is False
    assert "n_possible_pairs" not in summary
    assert summary["n_individual_pairs"] == len(depths) * (len(depths) - 1) // 2


@given(relationship_inputs())
def test_relationship_summary_conserves_individual_pairs(generated: RelationshipInput) -> None:
    """Related and unrelated Individual Pair counts partition all unordered pairs."""
    df, pair_lists = generated
    summary = compute_relationship_summary(df, pair_lists)
    n = len(df)
    n_possible = n * (n - 1) // 2
    closest = _closest_pair_degrees(pair_lists)

    assert "n_possible_pairs" not in summary
    assert "related_pair_density_by_generation" not in summary
    assert "related_pair_density_by_depth" in summary
    assert summary["n_individual_pairs"] == n_possible
    assert summary["n_related_pairs"] == len(closest)
    assert summary["n_related_pairs"] + summary["n_unrelated_pairs"] == n_possible
    expected_density = (len(closest) / n_possible) if n_possible else 0.0
    assert summary["related_pair_density"] == pytest.approx(expected_density)
    assert sum(summary["related_pairs_by_closest_degree"].values()) == len(closest)
    assert sum(summary["closest_relationship_per_individual"].values()) == n


@given(relationship_inputs())
def test_relationship_summary_uses_closest_degree_after_deduplication(generated: RelationshipInput) -> None:
    """Duplicate and reversed rows collapse to each pair's closest relationship degree."""
    df, pair_lists = generated
    summary = compute_relationship_summary(df, pair_lists)
    closest = _closest_pair_degrees(pair_lists)
    expected = {str(degree): 0 for degree in range(1, 6)}
    for degree in closest.values():
        expected[str(degree)] += 1
    assert summary["related_pairs_by_closest_degree"] == expected


@given(relationship_inputs())
def test_relationship_summary_depth_rows_conserve_pairs(generated: RelationshipInput) -> None:
    """Each per-Depth density row partitions same-Depth Individual Pairs."""
    df, pair_lists = generated
    summary = compute_relationship_summary(df, pair_lists)
    for row in summary["related_pair_density_by_depth"]:
        assert row["n_individual_pairs"] == row["n"] * (row["n"] - 1) // 2
        assert row["n_related_pairs"] + row["n_unrelated_pairs"] == row["n_individual_pairs"]
        expected_density = row["n_related_pairs"] / row["n_individual_pairs"] if row["n_individual_pairs"] else 0.0
        assert row["related_pair_density"] == pytest.approx(expected_density)
