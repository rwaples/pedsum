"""Offspring Sex Concordance: the analytical moments are exact.

The conditional moments are checked against exhaustive enumeration as
rationals, and the integer guards that keep them exact (the ``S`` pre-shift
bound, ``choose(P, 2)`` in Python ``int``) are pinned at the configurations
that break the wrong version.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sex_concordance_helpers import offspring_frame, uniform_groups

from pedsum import sex_concordance as sc
from pedsum.base import SEX_MALE


def _enumerate_moments(sizes: list[int], n_male: int) -> tuple[Fraction, Fraction]:
    """Exact ``(E[C], Var(C))`` over every fixed-margin assignment, as rationals."""
    n_total = sum(sizes)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    n_draws = 0
    s1 = Fraction(0)
    s2 = Fraction(0)
    for positions in itertools.combinations(range(n_total), n_male):
        labels = np.zeros(n_total, dtype=np.int64)
        labels[list(positions)] = 1
        male_counts = np.add.reduceat(labels, starts)
        c = sc.concordant_pairs(male_counts, sizes_arr)
        n_draws += 1
        s1 += c
        s2 += c * c
    mean = s1 / n_draws
    return mean, s2 / n_draws - mean * mean


# Sizes / male-margin configurations small enough to enumerate exhaustively,
# including single-group and all-one-sex degenerate cases.
_EXACT_CONFIGS = [
    ([2, 2], 2),
    ([2, 2], 1),
    ([3, 2], 2),
    ([3, 3], 3),
    ([2, 2, 2], 3),
    ([4], 2),
    ([5], 0),
    ([5], 5),
    ([4, 3, 2], 4),
    ([2, 2, 2, 2], 4),
    ([6, 2], 3),
]


@pytest.mark.parametrize(("sizes", "n_male"), _EXACT_CONFIGS)
def test_moments_match_exhaustive_enumeration(sizes, n_male):
    """E[C] and Var(C) equal the enumerated values *exactly*, as rationals."""
    moments = sc.analytical_moments(np.asarray(sizes, dtype=np.int64), n_male, sum(sizes) - n_male)
    expected, variance = _enumerate_moments(sizes, n_male)
    assert moments.expected == expected
    assert moments.variance == variance


def test_concordant_pairs_hand_checked():
    """C counts same-sex within-group Individual Pairs and nothing else."""
    sizes = np.array([4, 3], dtype=np.int64)
    # Group A: 3 male + 1 female -> C(3,2)=3. Group B: 1 male + 2 female -> C(2,2)=1.
    assert sc.concordant_pairs(np.array([3, 1], dtype=np.int64), sizes) == 4


def test_expected_and_excess_hand_checked():
    """Expected/excess/direction on a config whose exact values are hand-computable."""
    sizes = np.array([2, 2], dtype=np.int64)
    moments = sc.analytical_moments(sizes, 2, 2)
    # q2 = ([2]_2 + [2]_2) / [4]_2 = (2 + 2) / 12 = 1/3; P = 2 -> E[C] = 2/3.
    assert moments.n_pairs == 2
    assert moments.expected == Fraction(2, 3)
    # Both groups single-sex -> C = 2, so the excess is 2 - 2/3 = 4/3 pairs.
    assert Fraction(sc.concordant_pairs(np.array([2, 0], dtype=np.int64), sizes)) - moments.expected == Fraction(4, 3)


def test_pairs_and_triples_matches_exact_python_sum():
    """The vectorized ``(P, S)`` path agrees with a plain Python sum."""
    sizes = np.array([2, 3, 4, 7, 11], dtype=np.int64)
    n_pairs, n_shared = sc._pairs_and_triples(sizes, int(sizes.sum()))
    assert n_pairs == sum(int(n) * (int(n) - 1) // 2 for n in sizes)
    assert n_shared == sum(int(n) * (int(n) - 1) * (int(n) - 2) // 2 for n in sizes)


def test_pairs_and_triples_empty():
    """No eligible groups yields ``(0, 0)`` rather than an ndarray reduction error."""
    assert sc._pairs_and_triples(np.empty(0, dtype=np.int64), 0) == (0, 0)


def test_s_guard_routes_dominant_group_to_exact_fallback():
    """``[2_150_000, 2]`` must not take the vectorized path.

    A ``// 2``-halved overflow guard admits this configuration, where the
    *pre-shift* intermediate ``n(n-1)(n-2)`` wraps int64 before the shift ever
    happens. The guard bounds ``N * n_max**2`` with no halving for exactly this
    reason. Cheap to pin: the helper takes a sizes array, so no pedigree of
    that size is built.
    """
    sizes = np.array([2_150_000, 2], dtype=np.int64)
    true_s = 2_150_000 * 2_149_999 * 2_149_998 // 2
    _, n_shared = sc._pairs_and_triples(sizes, int(sizes.sum()))
    assert n_shared == true_s

    # And the vectorized expression really does wrap here — otherwise this test
    # would pass against the buggy guard too.
    n = sizes
    wrapped = int((((n * (n - 1)) * (n - 2)) >> 1).sum())
    assert wrapped != true_s
    assert wrapped < 0


def test_s_guard_admits_the_largest_safe_group():
    """``n_max = 2.0M`` is still int64-safe pre-shift and stays vectorized."""
    sizes = np.array([2_000_000, 2], dtype=np.int64)
    _, n_shared = sc._pairs_and_triples(sizes, int(sizes.sum()))
    assert n_shared == 2_000_000 * 1_999_999 * 1_999_998 // 2


def test_choose_p_2_is_computed_in_python_int():
    """``D = choose(P, 2) - S`` must not wrap int64 for a high-``P`` pedigree."""
    sizes = np.array([200_000], dtype=np.int64)
    moments = sc.analytical_moments(sizes, 100_000, 100_000)
    n_pairs = 200_000 * 199_999 // 2
    assert moments.n_pairs == n_pairs
    assert moments.n_disjoint_pairs == n_pairs * (n_pairs - 1) // 2 - moments.n_shared_index_pairs
    assert moments.n_disjoint_pairs > 2**63  # would have wrapped in int64


def test_single_group_variance_is_exactly_zero():
    """One eligible group makes ``C`` deterministic; the variance must equal 0.

    In float64 this configuration returns a large *negative* variance, so a
    float guard never fires and ``z`` comes out NaN or garbage.
    """
    moments = sc.analytical_moments(np.array([200_000], dtype=np.int64), 100_000, 100_000)
    assert moments.variance == 0
    assert isinstance(moments.variance, Fraction)


@pytest.mark.parametrize("n_male", [0, 6])
def test_single_sex_variance_is_zero(n_male):
    """All-male / all-female eligible sets have zero conditional variance."""
    assert sc.analytical_moments(np.array([3, 3], dtype=np.int64), n_male, 6 - n_male).variance == 0


def test_fewer_than_two_groups_skips():
    """A single eligible group is reported as not computable, with a reason."""
    mothers, fathers = uniform_groups(1, 4)
    out = sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, [1, 1, 0, 0]))
    block = out["sibship"]
    assert block["computed"] is False
    assert block["skip_reason"] == "fewer_than_two_eligible_groups"
    assert block["n_groups_eligible"] == 1  # eligibility metadata is retained


def test_single_sex_skips_with_its_own_reason():
    """A single-sex eligible set skips as ``single_sex``, not ``zero_variance``."""
    mothers, fathers = uniform_groups(3, 2)
    out = sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, [SEX_MALE] * 6))
    assert out["sibship"]["skip_reason"] == "single_sex"


@settings(deadline=None, max_examples=40)
@given(
    sizes=st.lists(st.integers(min_value=2, max_value=6), min_size=2, max_size=8),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_row_permutation_and_label_swap_invariance(sizes, seed):
    """C, E[C], Var(C) are invariant to input row order and to swapping M/F."""
    rng = np.random.default_rng(seed)
    mothers = np.repeat(np.arange(1, len(sizes) + 1) * 10, sizes)
    fathers = mothers + 1
    sexes = rng.integers(0, 2, mothers.size).astype(np.int8)

    base = sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, sexes))["sibship"]

    order = rng.permutation(mothers.size)
    shuffled = sc.compute_offspring_sex_concordance(
        offspring_frame(mothers[order], fathers[order], sexes[order]),
    )["sibship"]
    assert shuffled == base

    swapped = sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, 1 - sexes))["sibship"]
    assert swapped["observed_pair_concordance"] == base["observed_pair_concordance"]
    assert swapped["expected_pair_concordance"] == base["expected_pair_concordance"]
    assert swapped["z"] == base["z"]
