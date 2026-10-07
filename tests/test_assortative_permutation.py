"""The score-statistic permutation test (plan v3, decision 17): exact tail counts, invariances, and agreement with the refit null."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import assortative_oracles as oracles
import numpy as np
import polars as pl
import pytest
from conftest import four_thread_json
from scipy.special import ndtr
from scipy.stats import norm

from pedsum import assortative_kernels as kernels
from pedsum.assortative_mating import (
    CELL_ESTIMATORS,
    INFERENCE,
    SEQUENTIAL_H,
    CellPairs,
    Fit,
    Trait,
    Undefined,
    _Cell,
    _mother_scores,
    _PermutationInput,
    _standardised,
    _Stopping,
    bvn_cdf,
    compute_assortative_mating,
    estimate_value,
    permutation_donors,
    permutation_record,
    pooled,
)

if TYPE_CHECKING:
    from collections.abc import Callable

K = {"continuous": None, "binary": 2, "ordinal": 4}
LATENT_CELLS = [
    ("binary", "binary"),
    ("ordinal", "ordinal"),
    ("continuous", "binary"),
    ("binary", "continuous"),
    ("continuous", "ordinal"),
    ("ordinal", "continuous"),
]
#: Score-statistic p against the refit-ρ̂ permutation p on the same donors at P = 199. The two statistics
#: are almost monotone in each other (corr >= 0.9999 on 81 cell x form tests at P = 499, where max |Δp|
#: was 0.010), so they differ only where near-tied draws cross the observed value; 0.03 is six draws of 200.
REFIT_TOL = 0.03
REFIT_PERMUTATIONS = 199


# ---------------------------------------------------------------------------
# Fixtures: remated cells with cohort shifts, and the production statistic on explicit donors
# ---------------------------------------------------------------------------


def _remated(
    seed: int, n: int, m_kind: str, f_kind: str, r: float = 0.0, n_strata: int = 3
) -> tuple[CellPairs, np.ndarray, np.ndarray, np.ndarray]:
    """``n`` pairs over about ``0.8 n`` fathers (some remating), each side shifted by its own stratum.

    Returns ``(pairs, father_values, pair_father, father_stratum)`` with one
    value and stratum per distinct father.
    """
    rng = np.random.default_rng(seed)
    pair_father = np.unique(rng.integers(0, int(0.8 * n), n), return_inverse=True)[1]
    n_fathers = int(pair_father.max()) + 1
    f_stratum_of = rng.integers(0, n_strata, n_fathers)
    m_stratum = rng.integers(0, n_strata, n)
    liab_f_of = rng.normal(size=n_fathers)
    liab_f = liab_f_of[pair_father]
    liab_m = r * liab_f + np.sqrt(1 - r * r) * rng.normal(size=n) + 0.4 * m_stratum
    liab_f_of = liab_f_of - 0.5 * f_stratum_of
    cuts = np.quantile(np.concatenate([liab_m, liab_f_of[pair_father]]), [0.25, 0.5, 0.75])

    def code(x, kind, cut):
        if kind == "continuous":
            return x
        if kind == "binary":
            return (x > cut).astype(float)
        return np.digitize(x, cuts).astype(float)

    m, f_of = code(liab_m, m_kind, 0.3), code(liab_f_of, f_kind, -0.2)
    pairs = CellPairs(m, f_of[pair_father], m_stratum, f_stratum_of[pair_father]).with_levels(K[m_kind], K[f_kind])
    return pairs, f_of, pair_father, f_stratum_of


def _statistics(
    pairs: CellPairs, m_kind: str, f_of: np.ndarray, pair_father: np.ndarray, donors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(T, cross, ss_f, status)`` of each donor row, crude and stratified, through the production kernel path.

    Streams are packed as float64 so the statistic can be checked to 1e-12; the
    float32 production storage is tested against this path separately.
    """
    cell = _Cell(pairs, 0, pair_father, _mother_scores(pairs, K[m_kind]), [])
    packed = _PermutationInput.pack([cell], f_of[None, :], np.zeros(f_of.size, dtype=np.int64), np.float64)
    cross, ss_f, status = packed.statistics(donors)
    return _standardised(cross, ss_f, status, packed.ss_m)[:, 0], cross[:, 0], ss_f[:, 0], status[:, 0]


def _reference_scores(values: np.ndarray, stratum: np.ndarray, k: int | None) -> np.ndarray:
    """NumPy reference for ``latent_scores``: z within stratum, or ``E[η | level]`` from that stratum's thresholds."""
    if k is None:
        return oracles.standardise(values, stratum)
    n_strata = int(stratum.max()) + 1
    margin = np.bincount(stratum * k + values.astype(np.int64), minlength=n_strata * k).reshape(n_strata, k)
    tau = oracles.thresholds(margin)
    pdf = np.where(np.isfinite(tau), norm.pdf(np.where(np.isfinite(tau), tau, 0.0)), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(margin > 0, (pdf[:, :-1] - pdf[:, 1:]) * margin.sum(axis=1, keepdims=True) / margin, 0.0)
    return e[stratum, values.astype(np.int64)]


def _reference_statistic(pairs: CellPairs, m_kind: str, f_kind: str) -> float:
    """``Σ e_m e_f / √(Σ e_m² Σ e_f²)`` from the reference scores."""
    e_m = _reference_scores(pairs.m, pairs.m_stratum, K[m_kind])
    e_f = _reference_scores(pairs.f, pairs.f_stratum, K[f_kind])
    return float(np.dot(e_m, e_f) / np.sqrt(np.dot(e_m, e_m) * np.dot(e_f, e_f)))


def _oracle_nll(pairs: CellPairs, m_kind: str, f_kind: str) -> Callable[[float], float]:
    """The two-step latent log-likelihood of ``pairs`` as a function of ρ, with margins from these pairs.

    Polychoric: Olsson (1979) eqs 3-4 on the corner grids of every stratum
    table, thresholds from each stratum's margin. Polyserial: the conditional
    term of Olsson, Drasgow & Dorans (1982) eq 20 with the eq 19 probabilities.
    """
    if m_kind != "continuous" and f_kind != "continuous":
        n = oracles.count_table(pairs)
        a, b = oracles.thresholds(n.sum(axis=(1, 3))), oracles.thresholds(n.sum(axis=(0, 2)))
        combos = np.argwhere(n.sum(axis=(2, 3)) > 0)
        counts = n[combos[:, 0], combos[:, 1]]
        h, k = a[combos[:, 0]][:, :, None], b[combos[:, 1]][:, None, :]

        def nll(rho: float) -> float:
            cdf = bvn_cdf(h, k, rho)
            pi = cdf[:, 1:, 1:] - cdf[:, :-1, 1:] - cdf[:, 1:, :-1] + cdf[:, :-1, :-1]
            return -float(np.sum(counts[counts > 0] * np.log(pi[counts > 0])))

        return nll
    if m_kind == "continuous":
        x, y, xs, ys, levels = pairs.m, pairs.f, pairs.m_stratum, pairs.f_stratum, pairs.f_levels
    else:
        x, y, xs, ys, levels = pairs.f, pairs.m, pairs.f_stratum, pairs.m_stratum, pairs.m_levels
    z = oracles.standardise(x, xs)
    k = levels.shape[1]
    codes = y.astype(np.int64)
    margin = np.bincount(ys * k + codes, minlength=levels.shape[0] * k).reshape(levels.shape[0], k)
    tau = oracles.thresholds(margin)
    upper, lower = tau[ys, codes + 1], tau[ys, codes]

    def nll(rho: float) -> float:
        scale = np.sqrt(1 - rho * rho)
        cdf_upper = np.where(
            np.isfinite(upper), ndtr((np.where(np.isfinite(upper), upper, 0.0) - rho * z) / scale), 1.0
        )
        cdf_lower = np.where(
            np.isfinite(lower), ndtr((np.where(np.isfinite(lower), lower, 0.0) - rho * z) / scale), 0.0
        )
        return -float(np.sum(np.log(cdf_upper - cdf_lower)))

    return nll


# ---------------------------------------------------------------------------
# The counter-based generator
# ---------------------------------------------------------------------------


def test_splitmix64_matches_a_pure_python_reference():
    """``stream_state`` and ``next_below`` follow SplitMix64 keyed by ``(seed, draw)``."""
    mask = (1 << 64) - 1

    def mix(z: int) -> int:
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return z ^ (z >> 31)

    for seed, draw in [(0, 0), (7, 3), (2**31, 998), (123456789, 1)]:
        state = int(kernels.stream_state(seed, draw))
        assert state == mix((mix(seed) + draw) & mask)
        for n in (1, 2, 10, 1_000_003):
            nxt, j = kernels.next_below(np.uint64(state), n)
            expected_state = (state + 0x9E3779B97F4A7C15) & mask
            assert int(nxt) == expected_state
            assert j == min(int((mix(expected_state) >> 11) * 2.0**-53 * n), n - 1)
            state = expected_state


def test_block_permutations_are_uniform():
    """Every within-block permutation of a 3 + 2 + 1 father layout appears about equally often."""
    donors = permutation_donors(np.array([0, 0, 0, 1, 1, 2]), 12_000, 5)
    _, counts = np.unique(donors, axis=0, return_counts=True)
    assert counts.size == 12
    # 12 cells of expectation 1000 (sd 30): a chi-square above 32 (p < 0.001) would fail.
    assert float(np.sum((counts - 1000.0) ** 2) / 1000.0) < 32


# ---------------------------------------------------------------------------
# Exactness and correctness of the statistic
# ---------------------------------------------------------------------------


def _enumerate_donors(blocks: np.ndarray) -> np.ndarray:
    """Every within-block permutation, identity first."""
    within = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    donors = []
    for orders in itertools.product(*(itertools.permutations(members) for members in within)):
        donor = np.empty(len(blocks), dtype=np.int64)
        for members, order in zip(within, orders, strict=True):
            donor[members] = order
        donors.append(donor)
    donors = np.array(donors)
    identity = np.flatnonzero((donors == np.arange(len(blocks))).all(axis=1))
    return np.concatenate([donors[identity], np.delete(donors, identity, axis=0)])


@pytest.mark.parametrize(
    ("m_kind", "f_kind", "m", "f_of"),
    [
        ("continuous", "continuous", [0.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0], [2.0, 1.0, 0.0, 1.0, 2.0]),
        ("binary", "binary", [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 1.0]),
        ("continuous", "ordinal", [0.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0], [0.0, 0.0, 0.0, 2.0, 1.0]),
    ],
)
def test_exact_tail_count_matches_enumeration(m_kind, f_kind, m, f_of):
    """Fed every non-identity within-block permutation, the p-value equals the exact permutation p of the statistic."""
    pair_father = np.array([0, 0, 1, 2, 3, 4, 4])
    blocks = np.array([0, 0, 0, 1, 1])
    m, f_of = np.array(m), np.array(f_of)
    k_f = {"continuous": None, "binary": 2, "ordinal": 3}[f_kind]
    zeros = np.zeros(m.size, dtype=np.int64)
    pairs = CellPairs(m, f_of[pair_father], zeros, zeros).with_levels(K[m_kind], k_f)
    donors = _enumerate_donors(blocks)
    stats, _, _, status = _statistics(pairs, m_kind, f_of, pair_father, donors)
    assert not status.any()
    crude = stats[:, 0]
    for donor, stat in zip(donors, crude, strict=True):
        permuted = replace(pairs, f=f_of[donor[pair_father]])
        assert stat == pytest.approx(_reference_statistic(permuted, m_kind, f_kind), abs=1e-12)
    observed = crude[0]
    n_extreme = int(np.sum(np.abs(crude) > abs(observed) - 1e-12))
    exact_p = n_extreme / len(donors)
    record = permutation_record(observed, crude[1:].tolist(), len(donors) - 1, 0, 0, "score_at_zero")
    assert (len(donors), record["permutations"]["valid"]) == (12, 11)
    assert record["p_perm"] == exact_p
    assert not record["permutations"]["stopped_early"]
    assert 1 < n_extreme < len(donors)


@pytest.mark.parametrize(("m_kind", "f_kind"), LATENT_CELLS)
@pytest.mark.parametrize("stratified", [False, True])
def test_score_at_zero_is_the_derivative_of_the_oracle_log_likelihood(m_kind, f_kind, stratified):
    """``Σ e_m e_f`` equals ``∂l/∂ρ`` at ρ = 0 of the oracle two-step log-likelihood, margins refit on the same pairs.

    Checked on the observed pairs and on three permutations, so the refit
    father margins enter.
    """
    pairs, f_of, pair_father, f_stratum_of = _remated(11, 800, m_kind, f_kind, r=0.3)
    donors = permutation_donors(f_stratum_of, 3, 2)
    donors = np.concatenate([np.arange(f_of.size)[None, :], donors])
    stats, cross, ss_f, status = _statistics(pairs, m_kind, f_of, pair_father, donors)
    assert not status.any()
    form = int(stratified)
    h = 1e-5
    for donor, score, stat, ssf in zip(donors, cross[:, form], stats[:, form], ss_f[:, form], strict=True):
        permuted = replace(pairs, f=f_of[donor[pair_father]])
        sample = permuted if stratified else pooled(permuted)
        nll = _oracle_nll(sample, m_kind, f_kind)
        derivative = (nll(-h) - nll(h)) / (2 * h)
        assert score == pytest.approx(derivative, rel=1e-6), (m_kind, f_kind, stratified)
        e_f = _reference_scores(sample.f, sample.f_stratum, K[f_kind])
        assert ssf == pytest.approx(np.dot(e_f, e_f), rel=1e-9)
        assert stat == pytest.approx(_reference_statistic(sample, m_kind, f_kind), abs=1e-12)


def test_pearson_cells_use_pearson_r():
    """For two continuous sides the statistic is the Pearson r, crude on the pooled values and stratified on the standardised ones."""
    pairs, f_of, pair_father, f_stratum_of = _remated(3, 300, "continuous", "continuous", r=0.2)
    donors = np.concatenate([np.arange(f_of.size)[None, :], permutation_donors(f_stratum_of, 5, 1)])
    stats, _, _, status = _statistics(pairs, "continuous", f_of, pair_father, donors)
    assert not status.any()
    for donor, (crude, stratified) in zip(donors, stats, strict=True):
        permuted = replace(pairs, f=f_of[donor[pair_father]])
        assert crude == pytest.approx(oracles.pearson(permuted.m, permuted.f), abs=1e-12)
        assert stratified == pytest.approx(oracles.stratified_pearson(permuted), abs=1e-12)


def test_failures_in_a_permuted_sample_are_reported():
    """A donor from outside the cell that empties a shown level fails that form with the estimators' reason."""
    m = np.array([0.0, 1.0, 2.0, 1.5, 0.5, 2.5])
    pair_father = np.array([0, 1, 2, 3, 4, 5])
    f_of = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    f_stratum_of = np.array([0, 0, 0, 1, 1, 1])
    pairs = CellPairs(m, f_of[pair_father], np.zeros(6, dtype=np.int64), f_stratum_of[pair_father])
    # Only the first four pairs are analysed: fathers 4 and 5 can donate values into the cell.
    analysed = pairs.take(np.arange(4)).with_levels(None, 2)
    donors = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 4, 3, 5], [3, 1, 2, 0, 4, 5], [5, 1, 2, 3, 4, 0]])
    _, _, _, status = _statistics(analysed, "continuous", f_of, pair_father[:4], donors)
    # Row 1: father 3's stratum (1) now shows only level 1 in the cell: stratified empty_category, crude still fine.
    # Row 2: swapping 0 and 3 keeps both margins. Row 3: stratum 0 gets 0, 1, 1 (fine) and stratum 1 keeps 0.
    np.testing.assert_array_equal(status[:, 0], [0, 0, 0, 0])
    np.testing.assert_array_equal(status[:, 1], [0, kernels.STATUS_EMPTY_CATEGORY, 0, 0])
    all_ones = np.array([[1, 1, 2, 4, 3, 5]])
    _, _, _, status = _statistics(
        analysed, "continuous", np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0]), pair_father[:4], all_ones
    )
    # Level 0 is shown but gone from both margins: ``empty_category`` comes before ``constant_margin``, as in ``_check_levels``.
    np.testing.assert_array_equal(status[0], [kernels.STATUS_EMPTY_CATEGORY, kernels.STATUS_EMPTY_CATEGORY])


def test_a_constant_continuous_father_stratum_is_degenerate():
    """A draw that leaves one father stratum constant fails the stratified form only, without dividing by its zero SD."""
    m = np.array([0.0, 1.0, 2.0, 1.5, 0.5, 2.5])
    pair_father = np.arange(6)
    f_of = np.array([1.0, 1.0, 1.0, 0.5, 2.0, 3.0])
    f_stratum_of = np.array([0, 0, 0, 1, 1, 1])
    pairs = CellPairs(m, f_of, np.zeros(6, dtype=np.int64), f_stratum_of).with_levels(None, None)
    _, _, ss_f, status = _statistics(pairs, "continuous", f_of, pair_father, np.arange(6)[None, :])
    np.testing.assert_array_equal(status[0], [0, kernels.STATUS_DEGENERATE_STRATUM])
    assert ss_f[0, 0] == 6.0


# ---------------------------------------------------------------------------
# Agreement with the refit-ρ̂ permutation and null calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("m_kind", "f_kind"), LATENT_CELLS)
def test_score_p_tracks_the_refit_permutation_p(m_kind, f_kind):
    """On the same donors, the score-statistic p is within ``REFIT_TOL`` of the p from refitting ρ̂ per permutation."""
    for seed, r in ((1, 0.0), (2, 0.04)):
        pairs, f_of, pair_father, f_stratum_of = _remated(seed, 1500, m_kind, f_kind, r=r)
        estimators = CELL_ESTIMATORS[m_kind, f_kind]
        donors = permutation_donors(f_stratum_of, REFIT_PERMUTATIONS, seed)
        identity = np.arange(f_of.size)[None, :]
        observed = _statistics(pairs, m_kind, f_of, pair_father, identity)[0][0]
        stats, _, _, status = _statistics(pairs, m_kind, f_of, pair_father, donors)
        assert not status.any()
        for form, estimator in enumerate((estimators.crude[0], estimators.stratified)):
            fit = estimator.fn(pairs, np.ones(pairs.m.size), None)
            assert isinstance(fit, Fit)
            refit = []
            for donor in donors:
                draw = estimator.fn(replace(pairs, f=f_of[donor[pair_father]]), np.ones(pairs.m.size), fit.value)
                refit.append(estimate_value(draw) if not isinstance(draw, Undefined) else draw)
            # No sequential stop: the fixed-size p compares the two statistics over every draw.
            h = REFIT_PERMUTATIONS + 1
            p_refit = permutation_record(fit.value, refit, REFIT_PERMUTATIONS, seed, 0, "refit", h)["p_perm"]
            p_score = permutation_record(observed[form], stats[:, form].tolist(), REFIT_PERMUTATIONS, seed, 0, "s", h)
            valid = np.array([d for d in refit if not isinstance(d, Undefined)])
            assert np.corrcoef(valid, stats[:, form])[0, 1] > 0.999, (m_kind, f_kind, form)
            assert abs(p_score["p_perm"] - p_refit) <= REFIT_TOL, (m_kind, f_kind, form, p_score["p_perm"], p_refit)


@pytest.mark.parametrize(("m_kind", "f_kind"), [("binary", "binary"), ("continuous", "ordinal")])
def test_null_rejection_rate_is_about_alpha(m_kind, f_kind):
    """Over 100 null datasets at P = 99, the share of ``p_perm <= 0.05`` stays within three binomial SDs of 0.05."""
    alpha, n_sets, permutations = 0.05, 100, 99
    rejections = np.zeros(2)
    for seed in range(n_sets):
        pairs, f_of, pair_father, f_stratum_of = _remated(1000 + seed, 400, m_kind, f_kind)
        observed = _statistics(pairs, m_kind, f_of, pair_father, np.arange(f_of.size)[None, :])[0][0]
        stats, _, _, status = _statistics(
            pairs, m_kind, f_of, pair_father, permutation_donors(f_stratum_of, permutations, seed)
        )
        for form in range(2):
            draws = [Undefined("x") if st else float(t) for t, st in zip(stats[:, form], status[:, form], strict=True)]
            p = permutation_record(observed[form], draws, permutations, seed, 0, "score_at_zero")["p_perm"]
            rejections[form] += p <= alpha
    upper = alpha + 3 * np.sqrt(alpha * (1 - alpha) / n_sets)
    assert (rejections / n_sets <= upper).all(), rejections
    assert rejections.sum() > 0


# ---------------------------------------------------------------------------
# Invariances through the payload
# ---------------------------------------------------------------------------


def _frame(seed: int, n_pairs: int, remate_every: int = 3) -> tuple[pl.DataFrame, list[Trait]]:
    """A pedigree of ``n_pairs`` Mating Pairs with remating, birth years, and a continuous plus a binary trait."""
    rng = np.random.default_rng(seed)
    pairs, father = [], 1000
    for k in range(n_pairs):
        if k % remate_every:
            father += 1
        pairs.append((k + 1, father))
    parents = sorted({m for m, _ in pairs} | {f for _, f in pairs})
    rows = [(p, 0 if p <= n_pairs else 1, -1, -1, 0, int(rng.integers(1950, 1980))) for p in parents]
    next_id = max(parents) + 1
    for m, f in pairs:
        rows.append((next_id, 0, m, f, 1, 2000))
        next_id += 1
    ids, sexes, mothers, fathers, depths, years = (list(col) for col in zip(*rows, strict=True))
    df = pl.DataFrame(
        {"id": ids, "sex": sexes, "mother": mothers, "father": fathers, "ped_depth": depths, "birth_year": years},
        schema={
            "id": pl.Int64,
            "sex": pl.Int8,
            "mother": pl.Int64,
            "father": pl.Int64,
            "ped_depth": pl.Int32,
            "birth_year": pl.Int32,
        },
    )
    x = rng.normal(size=len(df))
    x[rng.random(len(df)) < 0.1] = np.nan
    b = np.where(np.isnan(x), np.nan, (x + rng.normal(size=len(df)) > 0.5).astype(float))
    b[rng.random(len(df)) < 0.1] = np.nan
    return df, [Trait("x", "continuous", "stated", None, x), Trait("b", "binary", "inferred", ("0", "1"), b)]


def _payload(threads: int) -> dict:
    import numba

    numba.set_num_threads(threads)
    df, traits = _frame(21, 300)
    return compute_assortative_mating(
        df, traits, permutations=199, bootstrap=0, seed=4, stratify_by="birth_year", birth_year_bin=10
    )


def test_thread_count_invariance():
    """The same seed gives identical permutation records on 1 thread and on 4 (a subprocess, since the suite pins numba to 1)."""
    one = _payload(1)
    assert one["settings"]["threads"] == 1
    four = four_thread_json("from test_assortative_permutation import _payload; print(json.dumps(_payload(4)))")
    assert (one["settings"].pop("threads"), four["settings"].pop("threads")) == (1, 4)
    assert json.loads(json.dumps(one)) == four
    primaries = [record for cell in four["mate_correlation"] for record in cell["crude"].values() if "p_perm" in record]
    assert len(primaries) == 4
    assert all(record["permutations"]["valid"] > 0 for record in primaries)
    stopped = [record["permutations"] for record in primaries if record["permutations"]["stopped_early"]]
    assert stopped, "null cells should stop early, so the 4-thread run exercised the batch scan"
    assert all(block["draws_used"] < 199 for block in stopped)


def test_permutation_statistic_is_recorded_per_cell():
    """Pearson cells record ``pearson``, latent cells ``score_at_zero``, crude and stratified alike; the inference block names both."""
    df, traits = _frame(8, 200)
    out = compute_assortative_mating(
        df, traits, permutations=49, bootstrap=0, seed=1, stratify_by="birth_year", birth_year_bin=10
    )
    assert out["inference"]["permutation_statistic"] == INFERENCE["permutation_statistic"]
    expected = {("x", "x"): ("pearson", "pearson"), ("x", "b"): ("biserial", "score_at_zero")}
    expected |= {("b", "x"): ("biserial", "score_at_zero"), ("b", "b"): ("tetrachoric", "score_at_zero")}
    for cell in out["mate_correlation"]:
        name, statistic = expected[cell["mother"], cell["father"]]
        for record in (cell["crude"][name], cell["stratified"][name]):
            assert record["permutation_statistic"] == statistic
            assert record["p_perm"] is not None
        for secondary in set(cell["crude"]) - {name, "table"}:
            assert "permutation_statistic" not in cell["crude"][secondary]


def _both_dtypes(
    m_kind: str, f_kind: str, n: int, seed: int, offset: float = 0.0
) -> tuple[_PermutationInput, _PermutationInput, np.ndarray]:
    """One remated cell packed with float32 and with float64 streams, its fathers' values shifted by ``offset``; also the values."""
    pairs, f_of, pair_father, f_stratum_of = _remated(seed, n, m_kind, f_kind, r=0.1)
    f_of = f_of + offset
    pairs = replace(pairs, f=f_of[pair_father])
    cell = _Cell(pairs, 0, pair_father, _mother_scores(pairs, K[m_kind]), [])
    f32, f64 = (
        _PermutationInput.pack([cell], f_of[None, :], f_stratum_of, dtype) for dtype in (np.float32, np.float64)
    )
    return f32, f64, f_of


def _p_value(observed: float, null: np.ndarray, status: np.ndarray) -> float:
    draws = [float(t) if st == 0 else Undefined("x") for t, st in zip(null, status, strict=True)]
    return permutation_record(float(observed), draws, null.size, 3, 0, "score_at_zero")["p_perm"]


@pytest.mark.parametrize(
    ("m_kind", "f_kind", "offset"),
    [
        *[(m, f, 0.0) for m, f in [*LATENT_CELLS, ("continuous", "continuous")]],
        *[(m, "continuous", 2e7) for m in K],
    ],
)
def test_float32_streams_track_the_float64_path(m_kind, f_kind, offset):
    """float32 storage of the streamed arrays moves a statistic by about its rounding: observed within 1e-6, null within 1e-5, same failures.

    A continuous father trait far from zero (values ~2e7, SD ~1) keeps its
    variation, and a discrete one is streamed as its codes. The p-values agree
    unless a null draw ties the observed statistic to within 1e-5 (a discrete
    cell repeats its table under many permutations, so exact ties are common
    there), where rounding decides which side of the tie the draw falls.
    """
    narrow, wide, f_of = _both_dtypes(m_kind, f_kind, 3000, 5, offset)
    assert (narrow.rows.dtype, wide.rows.dtype) == (np.float32, np.float64)
    if K[f_kind] is not None:
        np.testing.assert_array_equal(narrow.rows[:, 0], f_of[narrow.order])
    observed_32, observed_64 = narrow.observed(), wide.observed()
    np.testing.assert_allclose(observed_32, observed_64, rtol=0, atol=1e-6)
    null_32, status_32 = narrow.permuted(199, 3)
    null_64, status_64 = wide.permuted(199, 3)
    np.testing.assert_array_equal(status_32, status_64)
    valid = status_32 == 0
    assert valid.sum() > 300
    assert np.abs(null_32[valid] - null_64[valid]).max() <= 1e-5
    for form in range(2):
        p_32 = _p_value(observed_32[0, form], null_32[:, 0, form], status_32[:, 0, form])
        p_64 = _p_value(observed_64[0, form], null_64[:, 0, form], status_64[:, 0, form])
        if p_32 != p_64:
            tied = np.abs(np.abs(null_64[valid[:, 0, form], 0, form]) - abs(observed_64[0, form])) <= 1e-5
            assert tied.any(), (form, p_32, p_64)


# ---------------------------------------------------------------------------
# Sequential stopping (Besag & Clifford 1991, closed scheme)
# ---------------------------------------------------------------------------


def _record(draws, requested=None, h=3) -> dict:
    requested = len(draws) if requested is None else requested
    return permutation_record(1.0, draws, requested, 0, 0, "pearson", h)


def test_sequential_p_on_hand_built_sequences():
    """``p = h / l`` at the ``h``-th valid exceedance (ties count, failures do not), else ``(g + 1) / (B_valid + 1)``."""
    hit, miss, fail = 1.5, 0.2, Undefined("constant_margin")
    stop = _record([miss, hit, miss, miss, hit, -1.0, miss, hit, hit])
    assert stop["p_perm"] == 3 / 6
    assert stop["permutations"] | {"failure_reasons": {}} == stop["permutations"]
    assert {k: stop["permutations"][k] for k in ("valid", "failed", "draws_used", "stopped_early", "sequential_h")} == {
        "valid": 6,
        "failed": 0,
        "draws_used": 6,
        "stopped_early": True,
        "sequential_h": 3,
    }
    with_failures = _record([fail, hit, fail, hit, miss, fail, hit, miss, miss])
    assert with_failures["p_perm"] == 3 / 4
    assert {k: with_failures["permutations"][k] for k in ("valid", "failed", "draws_used", "failure_reasons")} == {
        "valid": 4,
        "failed": 3,
        "draws_used": 7,
        "failure_reasons": {"constant_margin": 3},
    }
    tie = _record([hit, 1.0 - 1e-13, miss, 1.0, miss])
    assert (tie["p_perm"], tie["permutations"]["draws_used"]) == (3 / 4, 4)
    last_draw = _record([miss, hit, miss, hit, hit])
    assert (last_draw["p_perm"], last_draw["permutations"]["stopped_early"]) == (3 / 5, False)
    no_stop = _record([hit, miss, miss, hit, fail, miss])
    assert no_stop["p_perm"] == (2 + 1) / (5 + 1)
    assert {k: no_stop["permutations"][k] for k in ("valid", "draws_used", "stopped_early")} == {
        "valid": 5,
        "draws_used": 6,
        "stopped_early": False,
    }
    assert SEQUENTIAL_H == 20 == INFERENCE["permutation_stop_h"]
    assert INFERENCE["permutation_stopping"] == "besag_clifford_closed"


def _sequential_cell(
    seed: int, r: float, permutations: int, h: int
) -> tuple[list[dict], _Stopping, _PermutationInput, np.ndarray, np.ndarray]:
    """A Pearson cell's sequential record and the ``_Stopping`` behind it, through the batched kernel path."""
    pairs, f_of, pair_father, f_stratum_of = _remated(seed, 400, "continuous", "continuous", r=r)
    cell = _Cell(pairs, 0, pair_father, _mother_scores(pairs, None), [])
    packed = _PermutationInput.pack([cell], f_of[None, :], f_stratum_of)
    observed = packed.observed()
    stopping = _Stopping.start(observed, np.ones((1, 2), dtype=bool), h)
    null, status = packed.permuted(permutations, seed, stopping)
    records = []
    for form in range(2):
        used = stopping.used[0, form]
        draws = [
            float(t) if st == 0 else Undefined("x")
            for t, st in zip(null[:used, 0, form], status[:used, 0, form], strict=True)
        ]
        records.append(permutation_record(float(observed[0, form]), draws, permutations, seed, 0, "pearson", h))
    return records, stopping, packed, null, status


def test_sequential_p_is_exact_under_the_null():
    """Over 300 null cells at P = 99, h = 5, the sequential p lies in the paper's support set and ``P(p <= a) <= a`` at a in {0.05, 0.1, 0.5}."""
    permutations, h, n_sets = 99, 5, 300
    n = permutations + 1
    support = {h / size for size in range(h, n)} | {k / n for k in range(1, h + 1)} | {1.0}
    p_values = np.array(
        [[rec["p_perm"] for rec in _sequential_cell(seed, 0.0, permutations, h)[0]] for seed in range(n_sets)]
    )
    assert all(any(abs(p - s) < 1e-12 for s in support) for p in p_values.ravel())
    for alpha in (0.05, 0.1, 0.5):
        rate = (p_values <= alpha + 1e-12).mean(axis=0)
        assert (rate <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / n_sets)).all(), (alpha, rate)
    assert (p_values <= 0.5 + 1e-12).mean() > 0.35
    # p = 1 is the stop at l = h, the observed value above its first h draws: probability 1 / (h + 1).
    assert abs((p_values == 1.0).mean() - 1 / (h + 1)) <= 3 * np.sqrt((1 / 6) * (5 / 6) / (2 * n_sets))


def test_stopping_reads_a_prefix_of_the_full_run():
    """With stopping, each form's used draws equal the first draws of the run without it, and a null cell stops inside the first batches."""
    permutations, seed = 199, 7
    records, stopping, packed, null, status = _sequential_cell(seed, 0.0, permutations, SEQUENTIAL_H)
    full_null, full_status = packed.permuted(permutations, seed)
    for form, record in enumerate(records):
        used = record["permutations"]["draws_used"]
        assert used == stopping.used[0, form] < permutations
        assert record["permutations"]["stopped_early"]
        assert record["p_perm"] == SEQUENTIAL_H / record["permutations"]["valid"]
        np.testing.assert_array_equal(null[:used, 0, form], full_null[:used, 0, form])
        np.testing.assert_array_equal(status[:used, 0, form], full_status[:used, 0, form])
        full = permutation_record(
            float(packed.observed()[0, form]), full_null[:, 0, form].tolist(), permutations, seed, 0, "pearson"
        )
        assert full["permutations"] == record["permutations"]
        assert full["p_perm"] == record["p_perm"]
    computed = status[:, 0, 0] != kernels.STATUS_NO_PAIRS
    launched = int(computed.sum())
    assert computed[:launched].all(), "the computed draws are a prefix"
    assert stopping.used.max() <= launched < permutations, "the stop saved draws whatever the launch schedule"


def test_a_strong_signal_runs_every_draw_and_a_null_cell_stops():
    """A cell with r = 0.8 never sees 20 exceedances, so it uses all 199 draws; the same cell under the null stops early."""
    signal, _, _, _, _ = _sequential_cell(3, 0.8, 199, SEQUENTIAL_H)
    for record in signal:
        block = record["permutations"]
        assert (block["stopped_early"], block["draws_used"], block["valid"]) == (False, 199, 199)
        assert record["p_perm"] == 1 / 200
    null, _, _, _, _ = _sequential_cell(3, 0.0, 199, SEQUENTIAL_H)
    for record in null:
        assert record["permutations"]["stopped_early"]
        assert record["permutations"]["draws_used"] < 199


def _signal_frame(seed: int, n_pairs: int) -> tuple[pl.DataFrame, list[Trait]]:
    df, traits = _frame(seed, n_pairs)
    x = traits[0].values
    index = {int(i): k for k, i in enumerate(df["id"].to_numpy())}
    rng = np.random.default_rng(seed)
    for m, f in zip(df["mother"].to_numpy(), df["father"].to_numpy(), strict=True):
        if m > 0 and not np.isnan(x[index[m]]):
            x[index[f]] = 0.9 * x[index[m]] + 0.3 * rng.normal()
    return df, traits


def test_payload_records_stopping_per_cell():
    """Through ``compute_assortative_mating``: the inference block names the scheme, a correlated x x x cell runs every draw, the null x x b cell (b thresholded from the father's own x) stops."""
    df, traits = _signal_frame(5, 300)
    out = compute_assortative_mating(df, traits, permutations=199, bootstrap=0, seed=2)
    assert out["inference"]["permutation_stopping"] == "besag_clifford_closed"
    assert out["inference"]["permutation_stop_h"] == 20
    cells = {(c["mother"], c["father"]): c for c in out["mate_correlation"]}
    xx = cells["x", "x"]["crude"]["pearson"]["permutations"]
    assert (xx["stopped_early"], xx["draws_used"], xx["requested"], xx["sequential_h"]) == (False, 199, 199, 20)
    xb = cells["x", "b"]["crude"]["biserial"]["permutations"]
    assert xb["stopped_early"]
    assert xb["draws_used"] < 199
    assert cells["x", "b"]["crude"]["biserial"]["p_perm"] == 20 / xb["valid"]
