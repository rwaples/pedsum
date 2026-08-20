"""Tests for Offspring Sex Concordance (``pedsum.sex_concordance``).

Four layers, in increasing distance from the formula:

1. **Exactness** — the conditional moments are checked against exhaustive
   enumeration *as rationals*, and the integer guards that keep them exact
   (the ``S`` pre-shift bound, ``choose(P, 2)`` in Python ``int``) are pinned
   at the configurations that break the wrong version.
2. **Calibration** — simulation, not algebra. Both regressions guarded here
   were found by simulation during design review and neither is visible to a
   formula test: the reproduction-conditional provenance artefact, and the
   far-tail liberality of the normal approximation.
3. **Projection** — eligibility, provenance, and group membership.
4. **Wiring** — CLI flags, slim/extra routing, safe-attempt redaction.
"""

from __future__ import annotations

import itertools
import math
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import EXAMPLE, write_ped
from conftest import load_summary_extra_yaml as _load_extra
from conftest import load_summary_yaml as _load_yaml
from conftest import run_pedsum as _run
from hypothesis import given, settings
from hypothesis import strategies as st

from pedsum import sex_concordance as sc
from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN
from pedsum.schema import _SEX_CONCORDANCE_SLIM_KEYS

_SECTION = "offspring_sex_concordance"
_GROUPINGS = ("sibship", "maternal_offspring_group", "paternal_offspring_group")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame(mothers, fathers, sexes, sources=None) -> pd.DataFrame:
    """Minimal validated-frame stand-in: the four columns the analysis reads."""
    n = len(mothers)
    return pd.DataFrame(
        {
            "mother": np.asarray(mothers, dtype=np.int64),
            "father": np.asarray(fathers, dtype=np.int64),
            "sex": np.asarray(sexes, dtype=np.int8),
            "sex_source": np.asarray(["input"] * n if sources is None else sources, dtype=object),
        },
    )


def _uniform_groups(n_groups: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (mothers, fathers) for ``n_groups`` disjoint groups of ``size``."""
    mothers = np.repeat(np.arange(1, n_groups + 1) * 10, size)
    return mothers, mothers + 1


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


# ---------------------------------------------------------------------------
# 1. Exact analytical correctness
# ---------------------------------------------------------------------------

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
    mothers, fathers = _uniform_groups(1, 4)
    out = sc.compute_offspring_sex_concordance(_frame(mothers, fathers, [1, 1, 0, 0]))
    block = out["sibship"]
    assert block["computed"] is False
    assert block["skip_reason"] == "fewer_than_two_eligible_groups"
    assert block["n_groups_eligible"] == 1  # eligibility metadata is retained


def test_single_sex_skips_with_its_own_reason():
    """A single-sex eligible set skips as ``single_sex``, not ``zero_variance``."""
    mothers, fathers = _uniform_groups(3, 2)
    out = sc.compute_offspring_sex_concordance(_frame(mothers, fathers, [SEX_MALE] * 6))
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

    base = sc.compute_offspring_sex_concordance(_frame(mothers, fathers, sexes))["sibship"]

    order = rng.permutation(mothers.size)
    shuffled = sc.compute_offspring_sex_concordance(
        _frame(mothers[order], fathers[order], sexes[order]),
    )["sibship"]
    assert shuffled == base

    swapped = sc.compute_offspring_sex_concordance(_frame(mothers, fathers, 1 - sexes))["sibship"]
    assert swapped["observed_pair_concordance"] == base["observed_pair_concordance"]
    assert swapped["expected_pair_concordance"] == base["expected_pair_concordance"]
    assert swapped["z"] == base["z"]


# ---------------------------------------------------------------------------
# 2. Calibration (simulation, not algebra)
# ---------------------------------------------------------------------------


def test_null_calibration_with_complete_input_provenance():
    """Under iid sex with complete ``input`` provenance the test holds its size.

    400 groups of 6, 2000 replicates, nominal 5%. Monte-Carlo se is ~0.5pp, so
    the band is wide enough not to flake and tight enough to catch a real
    mis-calibration.
    """
    rng = np.random.default_rng(20260820)
    n_groups, size = 400, 6
    sizes = np.full(n_groups, size, dtype=np.int64)
    starts = np.arange(0, n_groups * size, size)
    n_reject = 0
    n_reps = 2000
    for _ in range(n_reps):
        labels = rng.integers(0, 2, n_groups * size)
        male_counts = np.add.reduceat(labels, starts)
        n_male = int(male_counts.sum())
        moments = sc.analytical_moments(sizes, n_male, n_groups * size - n_male)
        z = (sc.concordant_pairs(male_counts, sizes) - float(moments.expected)) / math.sqrt(float(moments.variance))
        n_reject += abs(z) > 1.96
    assert 0.035 <= n_reject / n_reps <= 0.070


def _selection_replicate(rng, *, group_varying: bool, mixed: bool) -> dict:
    """One replicate where eligibility is conditional on having reproduced.

    True sex is iid at p=0.5 with **no** real concordance. An offspring is
    resolved either because its sex was supplied (``input``) or because it
    reproduced (``imputed_from_role``); the reproduction rate is sex-specific
    and, under ``group_varying``, differs from group to group — the realistic
    case, since retaining a dam's daughters and culling her sons is decided per
    group.
    """
    n_groups, size = 400, 6
    mothers, fathers = _uniform_groups(n_groups, size)
    n = n_groups * size
    true_sex = rng.integers(0, 2, n)
    male_rate = rng.uniform(0.1, 0.9, n_groups) if group_varying else np.full(n_groups, 0.5)
    rate = np.where(true_sex == SEX_MALE, np.repeat(male_rate, size), np.repeat(1 - male_rate, size))
    reproduced = rng.random(n) < rate
    has_input = (rng.random(n) < 0.5) if mixed else np.zeros(n, dtype=bool)
    sources = np.where(has_input, "input", np.where(reproduced, "imputed_from_role", "unresolved"))
    sexes = np.where(sources == "unresolved", SEX_UNKNOWN, true_sex)
    return sc.compute_offspring_sex_concordance(_frame(mothers, fathers, sexes, sources))["sibship"]


def _rejection_rates(group_varying: bool, n_reps: int = 200) -> tuple[float, float]:
    """Return ``(headline, all_resolved)`` rejection rates at nominal 5%."""
    rng = np.random.default_rng(4242)
    headline = all_resolved = 0
    for _ in range(n_reps):
        block = _selection_replicate(rng, group_varying=group_varying, mixed=True)
        if block["computed"] and block["p_analytical"] < 0.05:
            headline += 1
        sens = block["all_resolved"]
        if sens["computed"] and sens["p_analytical"] < 0.05:
            all_resolved += 1
    return headline / n_reps, all_resolved / n_reps


def test_uniform_selection_rates_are_harmless():
    """A *uniform* marginal sex bias in reproduction is absorbed by conditioning.

    This is the case that makes ``sex_source`` counts insufficient on their own:
    it has the same provenance counts as the group-varying case below and the
    opposite calibration.
    """
    headline, all_resolved = _rejection_rates(group_varying=False)
    assert headline < 0.15
    assert all_resolved < 0.15


def test_group_varying_selection_inflates_all_resolved_only():
    """Group-level heterogeneity in selection breaks all-resolved, not the headline."""
    headline, all_resolved = _rejection_rates(group_varying=True)
    assert headline < 0.15, f"input-only headline should stay calibrated, got {headline}"
    assert all_resolved > 0.15, f"all-resolved should inflate, got {all_resolved}"
    assert all_resolved > 3 * headline


def test_all_imputed_pedigree_refuses_the_headline():
    """With no input sex at all the headline skips rather than reporting the artefact."""
    rng = np.random.default_rng(7)
    block = _selection_replicate(rng, group_varying=True, mixed=False)
    assert block["computed"] is False
    assert block["skip_reason"] == "no_input_sex"
    assert "reproduced" in block["skip_message"]


def test_far_tail_is_liberal_for_a_dominant_group():
    """Pin the known far-tail anti-conservatism so an "improvement" can't hide it.

    One group of 600 plus 400 groups of 6 (``max_group_pair_share`` ~0.97).
    At the nominal 0.1% level the analytical p rejects ~1.6% of the time, and
    the excess is entirely in the positive (over-concordant) tail. At the
    conventional 5% level the same configuration is sound — which is why the
    feature warns rather than refusing on dominance grounds.
    """
    sizes = np.concatenate(([600], np.full(400, 6))).astype(np.int64)
    n_total = int(sizes.sum())
    n_male = n_total // 2
    moments = sc.analytical_moments(sizes, n_male, n_total - n_male)
    draws = sc.sample_concordance_numpy(sizes, n_male, 20_000, 1)
    z = (draws - float(moments.expected)) / math.sqrt(float(moments.variance))

    assert 0.04 <= np.mean(np.abs(z) > 1.96) <= 0.07, "should stay sound at the 5% level"
    far_tail = float(np.mean(np.abs(z) > 3.29))
    assert far_tail > 0.005, f"far-tail liberality lost its bite: {far_tail}"
    assert np.mean(z < -3.29) < 0.002, "the excess must be in the positive tail"


def test_max_group_pair_share_tracks_dominance():
    """``max_group_pair_share`` is the diagnostic that predicts the tail behaviour."""
    mothers, fathers = _uniform_groups(400, 6)
    flat = sc.compute_offspring_sex_concordance(
        _frame(mothers, fathers, np.resize([SEX_MALE, SEX_FEMALE], mothers.size)),
    )["sibship"]
    assert flat["max_group_pair_share"] < 0.01

    dominant_m = np.concatenate((np.full(600, 10), mothers[: 400 * 6]))
    dominant_f = dominant_m + 1
    dominant = sc.compute_offspring_sex_concordance(
        _frame(dominant_m, dominant_f, np.resize([SEX_MALE, SEX_FEMALE], dominant_m.size)),
    )["sibship"]
    assert dominant["max_group_pair_share"] > 0.9


# ---------------------------------------------------------------------------
# 3. Group eligibility and provenance
# ---------------------------------------------------------------------------


def test_groupings_differ_under_remating():
    """Half-sibs partition differently into Sibships and maternal/paternal groups."""
    # Mother 10 remated: two Sibships of 2, one Maternal Offspring Group of 4,
    # two Paternal Offspring Groups of 2.
    mothers = [10, 10, 10, 10]
    fathers = [11, 11, 12, 12]
    out = sc.compute_offspring_sex_concordance(_frame(mothers, fathers, [1, 0, 1, 0]))
    assert out["sibship"]["n_groups_eligible"] == 2
    assert out["maternal_offspring_group"]["n_groups_eligible"] == 1
    assert out["maternal_offspring_group"]["n_offspring_eligible"] == 4
    assert out["paternal_offspring_group"]["n_groups_eligible"] == 2


def test_missing_parent_changes_eligibility_per_grouping():
    """A missing father removes an offspring from Sibship/paternal, not maternal."""
    mothers = [10, 10, 20, 20]
    fathers = [11, -1, 21, 21]
    out = sc.compute_offspring_sex_concordance(_frame(mothers, fathers, [1, 0, 1, 0]))
    assert out["sibship"]["n_offspring_total"] == 3
    assert out["maternal_offspring_group"]["n_offspring_total"] == 4
    assert out["paternal_offspring_group"]["n_offspring_total"] == 3


def test_eligible_members_retained_when_a_sibling_is_not():
    """An ineligible offspring is dropped; its eligible siblings keep the group."""
    mothers, fathers = _uniform_groups(2, 3)
    sources = ["input", "input", "unresolved", "input", "input", "input"]
    sexes = [SEX_MALE, SEX_FEMALE, SEX_UNKNOWN, SEX_MALE, SEX_FEMALE, SEX_MALE]
    block = sc.compute_offspring_sex_concordance(_frame(mothers, fathers, sexes, sources))["sibship"]
    assert block["n_groups_eligible"] == 2
    assert block["n_groups_incomplete"] == 1
    assert block["n_offspring_eligible"] == 5
    assert block["n_offspring_sex_excluded"] == 1


def test_headline_is_input_only_and_sensitivity_admits_imputed():
    """The two passes differ only in which provenances they admit."""
    mothers, fathers = _uniform_groups(2, 3)
    sources = ["input", "input", "imputed_from_role", "input", "input", "imputed_from_missing"]
    sexes = [SEX_MALE, SEX_FEMALE, SEX_MALE, SEX_MALE, SEX_FEMALE, SEX_FEMALE]
    block = sc.compute_offspring_sex_concordance(_frame(mothers, fathers, sexes, sources))["sibship"]

    assert block["n_offspring_eligible"] == 4
    assert block["eligible_by_sex_source"] == {"input": 4, "imputed_from_missing": 0, "imputed_from_role": 0}

    sens = block["all_resolved"]
    assert sens["n_offspring_eligible"] == 6
    assert sens["eligible_by_sex_source"] == {"input": 4, "imputed_from_missing": 1, "imputed_from_role": 1}
    # Provenance counts reconcile: every offspring the sensitivity admits is
    # either headline-eligible or one of the imputed categories.
    assert sum(sens["eligible_by_sex_source"].values()) == sens["n_offspring_eligible"]


@pytest.mark.parametrize("grouping", _GROUPINGS)
def test_offspring_and_group_counts_conserve(grouping):
    """Total offspring split exactly into excluded / too-small / eligible."""
    rng = np.random.default_rng(3)
    mothers = rng.integers(1, 30, 400) * 10
    fathers = rng.integers(1, 4, 400) + mothers
    sexes = rng.integers(0, 2, 400)
    sources = rng.choice(["input", "imputed_from_role", "unresolved"], 400)
    sexes = np.where(sources == "unresolved", SEX_UNKNOWN, sexes)
    block = sc.compute_offspring_sex_concordance(_frame(mothers, fathers, sexes, sources))[grouping]
    assert (
        block["n_offspring_total"]
        == block["n_offspring_sex_excluded"] + block["n_offspring_in_small_groups"] + block["n_offspring_eligible"]
    )
    assert block["n_groups_total"] == block["n_groups_eligible"] + block["n_groups_too_small"]
    assert block["n_groups_incomplete"] <= block["n_groups_eligible"]


def test_by_group_size_rows_reconcile_with_totals():
    """``by_group_size`` covers every observed eligible size and sums to the totals."""
    mothers = np.concatenate((np.repeat([10, 20], 2), np.repeat([30], 4)))
    fathers = mothers + 1
    block = sc.compute_offspring_sex_concordance(
        _frame(mothers, fathers, [1, 0, 1, 1, 0, 0, 1, 0]),
    )["sibship"]
    rows = block["by_group_size"]
    assert [r["group_size"] for r in rows] == [2, 4]
    assert sum(r["n_groups"] for r in rows) == block["n_groups_eligible"]
    assert sum(r["n_offspring"] for r in rows) == block["n_offspring_eligible"]
    # Every row's expected value is the pooled null, identical across rows.
    assert len({r["expected_pair_concordance"] for r in rows}) == 1


# ---------------------------------------------------------------------------
# 4. Holm correction
# ---------------------------------------------------------------------------


def test_holm_pinned_example():
    """Three raw p-values give the textbook step-down result."""
    adjusted = sc.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == pytest.approx(0.03)  # 3 * 0.01
    assert adjusted["c"] == pytest.approx(0.06)  # max(0.03, 2 * 0.03)
    assert adjusted["b"] == pytest.approx(0.06)  # max(0.06, 1 * 0.04)


def test_holm_is_monotone_and_bounded():
    """Adjusted values never decrease with raw p and never exceed 1."""
    raw = {"a": 0.4, "b": 0.5, "c": 0.9}
    adjusted = sc.holm_adjust(raw)
    ordered = [adjusted[k] for k in sorted(raw, key=raw.get)]
    assert ordered == sorted(ordered)
    assert all(0.0 <= v <= 1.0 for v in adjusted.values())


def test_holm_handles_ties():
    """Tied raw p-values receive the same adjusted value."""
    adjusted = sc.holm_adjust({"a": 0.02, "b": 0.02, "c": 0.5})
    assert adjusted["a"] == adjusted["b"] == pytest.approx(0.06)


def test_holm_single_hypothesis_is_identity():
    """With one available hypothesis the adjustment is a no-op."""
    assert sc.holm_adjust({"a": 0.017}) == {"a": pytest.approx(0.017)}


def test_holm_drops_skipped_analyses_from_the_family():
    """A skipped grouping is not counted in ``m``, so it cannot inflate the others."""
    assert sc.holm_adjust({"a": 0.02, "b": None}) == {"a": pytest.approx(0.02)}


def test_holm_is_across_groupings_not_group_sizes():
    """The family is the computable groupings; ``by_group_size`` is untested."""
    mothers, fathers = _uniform_groups(20, 3)
    rng = np.random.default_rng(5)
    out = sc.compute_offspring_sex_concordance(_frame(mothers, fathers, rng.integers(0, 2, 60)))
    computed = [name for name in _GROUPINGS if out[name]["computed"]]
    for name in computed:
        block = out[name]
        assert block["p_analytical_holm"] >= block["p_analytical"]
        assert block["p_analytical_holm"] <= len(computed) * block["p_analytical"] + 1e-12
        assert all("p_analytical" not in row for row in block["by_group_size"])


# ---------------------------------------------------------------------------
# 5. Permutation implementations
# ---------------------------------------------------------------------------


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
        sampler = sc.load_numba_sampler()
        if sampler is None:
            pytest.skip("numba is not installed")
        draws = sampler(sizes, n_male, n_total, n_draws, 11)
    else:
        draws = sc.sample_concordance_numpy(sizes, n_male, n_draws, 11)

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
    draws = sc.sample_concordance_numpy(sizes, n_male, 8_000, 99).astype(np.float64)
    assert draws.mean() == pytest.approx(float(moments.expected), rel=0.01)
    assert draws.var(ddof=1) == pytest.approx(float(moments.variance), rel=0.10)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_sampler_is_deterministic_per_backend(backend):
    """A fixed seed reproduces the stream — within one backend, not across them."""
    sizes = np.full(50, 4, dtype=np.int64)
    if backend == "numba":
        sampler = sc.load_numba_sampler()
        if sampler is None:
            pytest.skip("numba is not installed")
        first = sampler(sizes, 100, 200, 64, 5)
        second = sampler(sizes, 100, 200, 64, 5)
    else:
        first = sc.sample_concordance_numpy(sizes, 100, 64, 5)
        second = sc.sample_concordance_numpy(sizes, 100, 64, 5)
    assert np.array_equal(first, second)


def _force_numpy_backend(monkeypatch) -> None:
    """Make ``import numba`` fail and clear the memoised probe."""
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setattr(sc, "_NUMBA_PROBED", False)
    monkeypatch.setattr(sc, "_NUMBA_SAMPLER", None)


def test_numpy_fallback_when_numba_is_unimportable(monkeypatch):
    """A missing numba degrades to the NumPy sampler, and says so in ``backend``."""
    _force_numpy_backend(monkeypatch)
    assert sc.load_numba_sampler() is None

    sizes = np.full(20, 4, dtype=np.int64)
    block = sc._run_permutations(sizes, 40, 30, 28.0, 50, 3)
    assert block["backend"] == "numpy"
    assert block["completed"] == 50
    assert block["p_raw"] is not None


def test_backend_is_recorded_for_the_compiled_path():
    """With numba importable the compiled sampler is used and recorded."""
    if sc.load_numba_sampler() is None:
        pytest.skip("numba is not installed")
    sizes = np.full(20, 4, dtype=np.int64)
    assert sc._run_permutations(sizes, 40, 30, 28.0, 50, 3)["backend"] == "numba"


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
    """Beyond the backend's bound, permutations are skipped — analytics survive."""
    backend = "numba" if sc.load_numba_sampler() is not None else "numpy"
    bound = sc._sampler_bound(backend)
    # Two enormous group sizes reach the bound without allocating a big array.
    sizes = np.array([bound // 2, bound - bound // 2], dtype=np.int64)
    block = sc._run_permutations(sizes, 4, 3, 2.0, 10, 0)
    assert block["completed"] == 0
    assert block["p_raw"] is None
    assert block["skip_reason"] == f"sampler_size_limit_{backend}"
    assert block["requested"] == 10


def test_zero_permutations_never_touches_the_compiled_sampler(monkeypatch):
    """``permutations=0`` must not pay a JIT compile it will not use."""

    def _boom():
        raise AssertionError("numba sampler loaded for a zero-permutation run")

    monkeypatch.setattr(sc, "load_numba_sampler", _boom)
    mothers, fathers = _uniform_groups(10, 4)
    rng = np.random.default_rng(1)
    out = sc.compute_offspring_sex_concordance(_frame(mothers, fathers, rng.integers(0, 2, 40)))
    assert all(out[name]["permutations"] is None for name in _GROUPINGS)


def test_permutation_results_are_invariant_to_input_row_order():
    """Canonical group sorting makes the permutation stream order-independent."""
    rng = np.random.default_rng(2)
    mothers, fathers = _uniform_groups(30, 4)
    sexes = rng.integers(0, 2, mothers.size)
    base = sc.compute_offspring_sex_concordance(
        _frame(mothers, fathers, sexes),
        permutations=200,
        seed=17,
    )["sibship"]
    order = rng.permutation(mothers.size)
    shuffled = sc.compute_offspring_sex_concordance(
        _frame(mothers[order], fathers[order], sexes[order]),
        permutations=200,
        seed=17,
    )["sibship"]
    assert shuffled == base


def test_permutation_p_is_preferred_when_present():
    """``p_source`` / ``p_holm`` switch to the permutation p once it exists."""
    rng = np.random.default_rng(6)
    mothers, fathers = _uniform_groups(30, 4)
    out = sc.compute_offspring_sex_concordance(
        _frame(mothers, fathers, rng.integers(0, 2, mothers.size)),
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


# ---------------------------------------------------------------------------
# 6. CLI, schema, and privacy
# ---------------------------------------------------------------------------


def _concordant_pedigree(tmp_path, sibship_sizes):
    """Pedigree whose Sibships are perfectly sex-concordant (p is tiny).

    ``sibship_sizes`` gives one size per Sibship, so a caller can arrange the
    ``by_group_size`` rows it wants to see.
    """
    rows = []
    next_id = 1
    for s, size in enumerate(sibship_sizes):
        mother, father = next_id, next_id + 1
        rows.append({"id": mother, "sex": 2, "mother": -1, "father": -1})
        rows.append({"id": father, "sex": 1, "mother": -1, "father": -1})
        next_id += 2
        child_sex = 1 if s % 2 == 0 else 2
        for _ in range(size):
            rows.append({"id": next_id, "sex": child_sex, "mother": mother, "father": father})
            next_id += 1
    return write_ped(tmp_path / "concordant.tsv", rows)


def _two_sibship_pedigree(tmp_path):
    """Eight rows forming exactly two eligible Sibships of two."""
    rows = [
        {"id": 1, "sex": 1, "mother": -1, "father": -1},
        {"id": 2, "sex": 2, "mother": -1, "father": -1},
        {"id": 3, "sex": 1, "mother": -1, "father": -1},
        {"id": 4, "sex": 2, "mother": -1, "father": -1},
        {"id": 5, "sex": 1, "mother": 2, "father": 1},
        {"id": 6, "sex": 2, "mother": 2, "father": 1},
        {"id": 7, "sex": 1, "mother": 4, "father": 3},
        {"id": 8, "sex": 2, "mother": 4, "father": 3},
    ]
    return write_ped(tmp_path / "two_sibships.tsv", rows)


def test_default_run_omits_the_section(tmp_path):
    """The feature is opt-in: a bare ``summarize`` emits nothing for it."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir)])
    assert res.returncode == 0, res.stderr
    assert _SECTION not in _load_yaml(out_dir)["pedigree"]["demography"]
    assert _SECTION not in _load_extra(out_dir)["pedigree"].get("demography", {})
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "annotated.tsv.gz",
        "summary.extra.yaml",
        "summary.yaml",
    ]


def test_flag_adds_the_section(tmp_path):
    """``--sex-concordance`` emits ``null_model`` plus all three groupings."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance"])
    assert res.returncode == 0, res.stderr
    section = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]
    assert section["null_model"]["method"] == "fixed_margin_exchangeability"
    assert section["null_model"]["headline_eligibility"] == "sex_source_input"
    assert all(name in section for name in _GROUPINGS)


def test_null_model_key_is_not_a_bare_null(tmp_path):
    """The block is ``null_model:``; a bare ``null:`` would parse as a None key."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance"])
    assert res.returncode == 0, res.stderr
    raw = yaml.safe_load((out_dir / "summary.yaml").read_text())
    section = raw["pedigree"]["demography"][_SECTION]
    assert None not in section
    assert "null_model" in section


def test_permutations_flag_implies_enable(tmp_path):
    """``--sex-concordance-permutations`` alone turns the analysis on."""
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(EXAMPLE),
            "--out",
            str(out_dir),
            "--sex-concordance-permutations",
            "50",
            "--sex-concordance-seed",
            "3",
        ],
    )
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    assert slim["p_source"] == "permutation"
    perms = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]["permutations"]
    assert perms["requested"] == 50
    assert perms["completed"] == 50
    assert perms["seed"] == 3
    assert perms["backend"] in {"numba", "numpy"}


def test_seed_without_permutations_warns(tmp_path):
    """A seed with zero permutations is inert, and says so."""
    out_dir = tmp_path / "out"
    res = _run(
        ["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance", "--sex-concordance-seed", "9"],
    )
    assert res.returncode == 0, res.stderr
    assert "--sex-concordance-seed has no effect" in res.stderr


def test_low_analytical_p_without_permutations_warns(tmp_path):
    """Below p = 0.01 the unpermuted analytical p must not be claimed silently."""
    ped = _concordant_pedigree(tmp_path, [4] * 40)
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(ped), "--out", str(out_dir), "--sex-concordance"])
    assert res.returncode == 0, res.stderr
    assert "--sex-concordance-permutations before claiming this" in res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    assert slim["direction"] == "over_concordant"


def test_no_warning_once_permutations_have_run(tmp_path):
    """The warning is about *unpermuted* claims, so permutations silence it."""
    ped = _concordant_pedigree(tmp_path, [4] * 10)
    out_dir = tmp_path / "out"
    res = _run(
        ["summarize", "--in", str(ped), "--out", str(out_dir), "--sex-concordance-permutations", "100"],
    )
    assert res.returncode == 0, res.stderr
    assert "before claiming this" not in res.stderr


def test_slim_extra_partition_has_no_overlap(tmp_path):
    """Each grouping's leaf keys land in exactly one of slim / extra."""
    out_dir = tmp_path / "out"
    res = _run(
        ["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance-permutations", "20"],
    )
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]
    extra = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]
    assert "null_model" in slim
    assert "null_model" not in extra
    for name in _GROUPINGS:
        assert set(slim[name]) & set(extra[name]) == set()
        assert set(slim[name]) <= set(_SEX_CONCORDANCE_SLIM_KEYS)
        assert "by_group_size" in extra[name]
        assert "all_resolved" in extra[name]
        assert "z" in extra[name]


def test_slim_yaml_stays_under_budget_with_the_feature_on(tmp_path):
    """Even opted in, the example pedigree's slim YAML stays inside the budget."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance"])
    assert res.returncode == 0, res.stderr
    assert len((out_dir / "summary.yaml").read_text().splitlines()) <= 500


def test_long_tsv_carries_aggregates_but_no_group_identities(tmp_path):
    """The optional TSV gains aggregate rows; no mother/father id is emitted."""
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(EXAMPLE), "--out", str(out_dir), "--sex-concordance", "--tsv"])
    assert res.returncode == 0, res.stderr
    rows = [
        line.split("\t")
        for line in (out_dir / "summary.pedigree.tsv").read_text().splitlines()[1:]
        if line.startswith(_SECTION)
    ]
    assert rows, "expected aggregate concordance rows in the long TSV"
    assert {r[1] for r in rows} == {"null_model", *_GROUPINGS}
    forbidden = ("mother", "father", "id")
    assert not any(any(tok in r[2] for tok in forbidden) for r in rows)


def test_safe_attempt_suppresses_inference_below_five_groups(tmp_path):
    """Under five eligible groups: eligibility metadata stays, inference goes."""
    ped = _two_sibship_pedigree(tmp_path)
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--sex-concordance-permutations",
            "20",
            "--safe-attempt",
        ],
    )
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    extra = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]

    assert slim["computed"] is True  # the analysis ran; its results are redacted
    assert slim["excess_concordance"] is None
    assert slim["direction"] is None
    assert slim["p_holm"] is None
    assert slim["max_group_pair_share"] is None
    assert extra["z"] is None
    assert extra["male_proportion_distribution"] is None
    assert extra["by_group_size"] == []
    assert extra["all_resolved"]["excess_concordance"] is None
    # Counts of 1-4 are nulled; the procedure metadata may remain.
    assert slim["n_groups_eligible"] is None
    assert extra["permutations"]["completed"] == 20
    assert extra["permutations"]["p_raw"] is None


def test_safe_attempt_redacts_small_by_group_size_rows(tmp_path):
    """Per-size rows below the cell threshold keep only size and group count."""
    ped = _concordant_pedigree(tmp_path, [3] * 12 + [4] * 2)
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(ped), "--out", str(out_dir), "--sex-concordance", "--safe-attempt"])
    assert res.returncode == 0, res.stderr
    rows = {
        r["group_size"]: r for r in _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]["by_group_size"]
    }
    # 12 Sibships of 3 survive; the 2 Sibships of 4 are a small cell.
    assert rows[3]["n_groups"] == 12
    assert rows[3]["observed_pair_concordance"] is not None
    assert rows[4]["n_groups"] == 2
    assert rows[4]["observed_pair_concordance"] is None
    assert rows[4]["excess_concordance"] is None
    assert rows[4]["n_offspring"] is None


def test_safe_attempt_drops_male_proportion_extrema(tmp_path):
    """The male-proportion distribution loses min/max like every other one."""
    ped = _concordant_pedigree(tmp_path, [3] * 12 + [4] * 2)
    out_dir = tmp_path / "out"
    res = _run(["summarize", "--in", str(ped), "--out", str(out_dir), "--sex-concordance", "--safe-attempt"])
    assert res.returncode == 0, res.stderr
    dist = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]["male_proportion_distribution"]
    assert dist is not None
    assert "min" not in dist
    assert "max" not in dist


def test_real_imputation_feeds_the_provenance_tally(tmp_path):
    """The three ``sex_source`` values ``validate`` emits land in the right slots.

    Guards a cross-module coupling the fabricated-frame tests cannot: they
    supply ``sex_source`` themselves, so they would still pass if ``validate``
    renamed a provenance — while every real pedigree silently reported zeros.

    The Sibship of six under (1, 2) is built so pedsum's own imputation
    produces all three: two offspring with input sex, two whose sex is missing
    and who are each used as exactly one parent role (``imputed_from_missing``),
    and two whose asserted sex is contradicted by their role
    (``imputed_from_role``).
    """
    rows = [
        {"id": 1, "sex": 2, "mother": -1, "father": -1},
        {"id": 2, "sex": 1, "mother": -1, "father": -1},
        {"id": 3, "sex": 1, "mother": 1, "father": 2},
        {"id": 4, "sex": 2, "mother": 1, "father": 2},
        {"id": 5, "sex": -1, "mother": 1, "father": 2},  # used only as a mother
        {"id": 6, "sex": -1, "mother": 1, "father": 2},  # used only as a father
        {"id": 7, "sex": 1, "mother": 1, "father": 2},  # asserted M, used as a mother
        {"id": 8, "sex": 2, "mother": 1, "father": 2},  # asserted F, used as a father
        {"id": 9, "sex": 1, "mother": -1, "father": -1},
        {"id": 10, "sex": 2, "mother": -1, "father": -1},
        {"id": 11, "sex": 1, "mother": -1, "father": -1},
        {"id": 12, "sex": 2, "mother": -1, "father": -1},
    ]
    next_id = 13
    for mother, father in ((5, 9), (10, 6), (7, 11), (12, 8)):
        for sex in (1, 2):
            rows.append({"id": next_id, "sex": sex, "mother": mother, "father": father})
            next_id += 1
    ped = write_ped(tmp_path / "mixed_provenance.tsv", rows)

    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--sex-concordance",
            "--no-effective-size",
            "--no-inbreeding",
        ],
    )
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    extra = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]

    # Headline admits only the input-sex offspring; the imputed slots stay zero.
    assert extra["eligible_by_sex_source"] == {"input": 10, "imputed_from_missing": 0, "imputed_from_role": 0}
    assert slim["n_offspring_eligible"] == 10

    sensitivity = extra["all_resolved"]
    assert sensitivity["eligible_by_sex_source"] == {"input": 10, "imputed_from_missing": 2, "imputed_from_role": 2}
    assert sensitivity["n_offspring_eligible"] == 14
    assert sum(sensitivity["eligible_by_sex_source"].values()) == sensitivity["n_offspring_eligible"]


def test_no_input_sex_pedigree_skips_with_its_message(tmp_path):
    """A pedigree whose every resolved sex is imputed refuses the headline."""
    rows = [
        {"id": 1, "sex": -1, "mother": -1, "father": -1},
        {"id": 2, "sex": -1, "mother": -1, "father": -1},
    ]
    rows += [{"id": i, "sex": -1, "mother": 1, "father": 2} for i in range(3, 7)]
    rows += [{"id": i, "sex": -1, "mother": -1, "father": -1} for i in range(7, 11)]
    next_id = 11
    for k, parent in enumerate(range(3, 7)):
        for _ in range(2):
            rows.append({"id": next_id, "sex": -1, "mother": parent, "father": 7 + k})
            next_id += 1
    ped = write_ped(tmp_path / "no_input_sex.tsv", rows)
    out_dir = tmp_path / "out"
    res = _run(
        [
            "summarize",
            "--in",
            str(ped),
            "--out",
            str(out_dir),
            "--sex-concordance",
            "--allow-missing-sex",
            "--no-effective-size",
            "--no-inbreeding",
        ],
    )
    assert res.returncode == 0, res.stderr
    slim = _load_yaml(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    extra = _load_extra(out_dir)["pedigree"]["demography"][_SECTION]["sibship"]
    assert slim["computed"] is False
    assert slim["skip_reason"] == "no_input_sex"
    assert "reproduced" in extra["skip_message"]
