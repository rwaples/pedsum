"""Offspring Sex Concordance: the permutation samplers and the permutation p-value."""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest
from sex_concordance_helpers import GROUPINGS, offspring_frame, uniform_groups
from sex_concordance_oracles import sample_concordance_numpy

from pedsum import sex_concordance as sc


def _pmf_of_c(sizes: list[int], n_male: int) -> dict[int, Fraction]:
    """Exact pmf of ``C`` over every fixed-margin assignment."""
    n_total = sum(sizes)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    counts: dict[int, int] = {}
    total = 0
    for positions in itertools.combinations(range(n_total), n_male):
        labels = np.zeros(n_total, dtype=np.int64)
        labels[list(positions)] = 1
        c = sc.concordant_pairs(np.add.reduceat(labels, starts), sizes_arr)
        counts[c] = counts.get(c, 0) + 1
        total += 1
    return {c: Fraction(k, total) for c, k in counts.items()}


def test_marginals_allocation_preserves_sizes_and_margin():
    """The fixed-margin allocation the NumPy sampler relies on is exactly that."""
    rng = np.random.default_rng(0)
    sizes = np.array([2, 3, 5, 7, 4], dtype=np.int64)
    n_male = 11
    for _ in range(200):
        male = rng.multivariate_hypergeometric(sizes, n_male)
        assert male.sum() == n_male
        assert np.all(male <= sizes)
        assert np.all(male >= 0)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_sampler_reproduces_the_exact_fixed_margin_distribution(backend):
    """Both samplers hit the enumerated support with the enumerated frequencies.

    A sampler that failed to preserve group sizes or the global male margin
    would put mass outside this support, so this doubles as the margin check
    for the compiled path (whose per-group counts are not otherwise visible).
    """
    sizes = np.array([3, 3], dtype=np.int64)
    n_male, n_total, n_draws = 3, 6, 40_000
    if backend == "numba":
        draws = sc.load_numba_sampler()(sizes, n_male, n_total, n_draws, 11)
    else:
        draws = sample_concordance_numpy(sizes, n_male, n_draws, 11)

    pmf = _pmf_of_c([3, 3], n_male)
    assert set(np.unique(draws)) <= set(pmf)
    for value, probability in pmf.items():
        assert np.mean(draws == value) == pytest.approx(float(probability), abs=0.01)


def test_permutation_moments_match_the_analytical_moments():
    """Empirical mean/variance of the sampler agree with the exact moments."""
    sizes = np.full(200, 5, dtype=np.int64)
    n_total = int(sizes.sum())
    n_male = n_total // 2
    moments = sc.analytical_moments(sizes, n_male, n_total - n_male)
    draws = sample_concordance_numpy(sizes, n_male, 8_000, 99).astype(np.float64)
    assert draws.mean() == pytest.approx(float(moments.expected), rel=0.01)
    assert draws.var(ddof=1) == pytest.approx(float(moments.variance), rel=0.10)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_sampler_is_deterministic_per_backend(backend):
    """A fixed seed reproduces the stream — within one backend, not across them."""
    sizes = np.full(50, 4, dtype=np.int64)
    if backend == "numba":
        sampler = sc.load_numba_sampler()
        first = sampler(sizes, 100, 200, 64, 5)
        second = sampler(sizes, 100, 200, 64, 5)
    else:
        first = sample_concordance_numpy(sizes, 100, 64, 5)
        second = sample_concordance_numpy(sizes, 100, 64, 5)
    assert np.array_equal(first, second)


def test_permutations_always_use_the_compiled_sampler():
    """With numba a dependency, production has one sampler: the block names no backend and holds the compiled draws' p."""
    assert not hasattr(sc, "sample_concordance_numpy")
    sizes = np.full(20, 4, dtype=np.int64)
    block = sc._run_permutations(sizes, 40, 30, 28.0, 50, 3)
    assert "backend" not in block
    draws = sc.load_numba_sampler()(sizes, 40, 80, 50, 3)
    n_extreme = int((np.abs(draws - 28.0) >= 2.0 * (1.0 - 1e-12)).sum())
    assert block["p_raw"] == (n_extreme + 1) / 51


def test_permutation_p_uses_the_finite_monte_carlo_form():
    """``p = (b+1)/(B+1)``: never zero, always an exact multiple of ``1/(B+1)``."""
    sizes = np.full(30, 4, dtype=np.int64)
    n_permutations = 199
    block = sc._run_permutations(sizes, 60, 120, 44.0, n_permutations, 1)
    assert block["completed"] == n_permutations
    assert block["p_raw"] >= 1 / (n_permutations + 1)
    numerator = block["p_raw"] * (n_permutations + 1)
    assert numerator == pytest.approx(round(numerator))


def test_permutation_two_sided_extremeness_counts_both_tails():
    """An observation far *below* E[C] is as extreme as one far above."""
    sizes = np.full(30, 4, dtype=np.int64)
    n_total = 120
    n_male = 60
    expected = float(sc.analytical_moments(sizes, n_male, n_total - n_male).expected)
    low = sc._run_permutations(sizes, n_male, 0, expected, 200, 4)
    high = sc._run_permutations(sizes, n_male, 2 * round(expected), expected, 200, 4)
    assert low["p_raw"] == pytest.approx(high["p_raw"])


def test_sampler_size_limit_refuses_cleanly():
    """Beyond the sampler's bound, permutations are skipped — analytics survive."""
    bound = sc._NUMBA_SAMPLER_MAX_N
    # Two enormous group sizes reach the bound without allocating a big array.
    sizes = np.array([bound // 2, bound - bound // 2], dtype=np.int64)
    block = sc._run_permutations(sizes, 4, 3, 2.0, 10, 0)
    assert block["completed"] == 0
    assert block["p_raw"] is None
    assert block["skip_reason"] == "sampler_size_limit_numba"
    assert block["requested"] == 10


def test_zero_permutations_never_touches_the_compiled_sampler(monkeypatch):
    """``permutations=0`` must not pay a JIT compile it will not use."""

    def _boom():
        raise AssertionError("numba sampler loaded for a zero-permutation run")

    monkeypatch.setattr(sc, "load_numba_sampler", _boom)
    mothers, fathers = uniform_groups(10, 4)
    rng = np.random.default_rng(1)
    out = sc.compute_offspring_sex_concordance(offspring_frame(mothers, fathers, rng.integers(0, 2, 40)))
    assert all(out[name]["permutations"] is None for name in GROUPINGS)


def test_permutation_results_are_invariant_to_input_row_order():
    """Canonical group sorting makes the permutation stream order-independent."""
    rng = np.random.default_rng(2)
    mothers, fathers = uniform_groups(30, 4)
    sexes = rng.integers(0, 2, mothers.size)
    base = sc.compute_offspring_sex_concordance(
        offspring_frame(mothers, fathers, sexes),
        permutations=200,
        seed=17,
    )["sibship"]
    order = rng.permutation(mothers.size)
    shuffled = sc.compute_offspring_sex_concordance(
        offspring_frame(mothers[order], fathers[order], sexes[order]),
        permutations=200,
        seed=17,
    )["sibship"]
    assert shuffled == base


def test_permutation_p_is_preferred_when_present():
    """``p_source`` / ``p_holm`` switch to the permutation p once it exists."""
    rng = np.random.default_rng(6)
    mothers, fathers = uniform_groups(30, 4)
    out = sc.compute_offspring_sex_concordance(
        offspring_frame(mothers, fathers, rng.integers(0, 2, mothers.size)),
        permutations=100,
        seed=1,
    )
    block = out["sibship"]
    assert block["p_source"] == "permutation"
    assert block["p_holm"] == block["permutations"]["p_holm"]
    # The analytical p is still reported alongside, unadjusted and adjusted —
    # preferring the permutation p does not discard it.
    assert block["p_analytical"] is not None
    assert block["p_analytical_holm"] is not None
    assert block["permutations"]["p_raw"] != block["p_analytical"]
