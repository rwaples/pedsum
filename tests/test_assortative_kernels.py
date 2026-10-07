"""The numba kernels against the v2 NumPy oracles (``assortative_oracles``) and against finite differences."""

from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING

import assortative_oracles as oracles
import numpy as np
import pytest
from conftest import four_thread_json
from scipy.special import ndtri
from scipy.stats import rankdata
from test_assortative_mating import _from_table

from pedsum import assortative_kernels as kernels
from pedsum.assortative_mating import (
    CELL_ESTIMATORS,
    CellPairs,
    Fit,
    Undefined,
    bvn_pdf_and_drho,
    degenerate_strata,
    odds_ratio,
    pearson,
    polychoric,
    polyserial,
    pooled,
    spearman,
    standardise,
    stratified_pearson,
    thresholds,
    unit_weights,
)

if TYPE_CHECKING:
    from pedsum.assortative_mating import Estimate

#: Kernel estimates agree with the oracles to this (the oracles stop Brent at xatol 1e-7, so the
#: comparison measures the kernel against Brent's stopping point as much as against the optimum).
TOL = 1e-7


def test_ndtri_matches_scipy():
    """AS 241 agrees with ``scipy.special.ndtri`` to 1e-15 relative over the whole open interval, ±inf at 0 and 1."""
    p = np.concatenate(
        [np.logspace(-300, -1, 2000), np.linspace(1e-3, 1 - 1e-3, 5001), 1 - np.logspace(-16, -1, 2000), [0.075, 0.925]]
    )
    got = np.array([kernels.ndtri(v) for v in p])
    expected = ndtri(p)
    np.testing.assert_allclose(got, expected, rtol=1e-15, atol=0)
    assert kernels.ndtri(0.0) == -np.inf
    assert kernels.ndtri(1.0) == np.inf
    assert kernels.ndtri(0.5) == 0.0


def test_thresholds_match_oracle():
    """Per-stratum thresholds equal the ``norm.ppf`` oracle, with -inf rows for empty strata and +inf for empty top levels."""
    margin = np.array([[10, 20, 30, 5], [0, 0, 0, 0], [7, 0, 3, 0], [4, 4, 0, 0]], dtype=np.float64)
    got, expected = thresholds(margin), oracles.thresholds(margin)
    finite = np.isfinite(expected)
    np.testing.assert_allclose(got[finite], expected[finite], rtol=1e-14, atol=0)
    np.testing.assert_array_equal(got[~finite], expected[~finite])


@pytest.mark.parametrize("rho", [-0.9, -0.3, 0.0, 0.45, 0.95])
def test_polyserial_terms_match_finite_differences(rho):
    """The fused kernel's score and Hessian are the ρ-derivatives of its own NLL (central differences)."""
    rng = np.random.default_rng(5)
    n = 400
    draws = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], size=n)
    x_stratum = rng.integers(0, 3, n)
    x = draws[:, 0] + x_stratum
    y = np.digitize(draws[:, 1], [-0.8, 0.1, 1.2]).astype(float)
    y_stratum = rng.integers(0, 2, n)
    w = rng.integers(0, 3, n).astype(float)
    _, mean, var, _, _ = kernels.stratum_moments(x, x_stratum, w, 3)
    margin = kernels.margin(y, y_stratum, w, 2, 4)
    args = (x, x_stratum, mean, 1 / np.sqrt(var), y, y_stratum, kernels.thresholds(margin), w)
    nll, grad, hess = kernels.polyserial_terms(rho, *args)
    assert nll == kernels.polyserial_nll(rho, *args)
    h = 1e-5
    fd_grad = (kernels.polyserial_nll(rho + h, *args) - kernels.polyserial_nll(rho - h, *args)) / (2 * h)
    fd_hess = (kernels.polyserial_terms(rho + h, *args)[1] - kernels.polyserial_terms(rho - h, *args)[1]) / (2 * h)
    assert grad == pytest.approx(fd_grad, rel=1e-6, abs=1e-6)
    assert hess == pytest.approx(fd_hess, rel=1e-5, abs=1e-4)


@pytest.mark.parametrize("rho", [-0.6, 0.0, 0.3, 0.6])
def test_bvn_pdf_drho_matches_finite_difference(rho):
    """``∂φ2/∂ρ`` (Olsson 1979 A2, corrected) is the central difference of φ2; both vanish at infinite arguments."""
    h = np.array([0.2, -1.1, 0.0, 1.5, np.inf, -np.inf, 0.3])
    k = np.array([-0.5, 0.7, 0.0, -2.0, 0.4, 0.4, np.inf])
    pdf, drho = bvn_pdf_and_drho(h, k, rho)
    eps = 1e-6
    fd = (bvn_pdf_and_drho(h, k, rho + eps)[0] - bvn_pdf_and_drho(h, k, rho - eps)[0]) / (2 * eps)
    np.testing.assert_allclose(drho, fd, rtol=1e-6, atol=1e-9)
    finite = np.isfinite(h) & np.isfinite(k)
    q = 1 - rho * rho
    expected = np.exp(-(h[finite] ** 2 - 2 * rho * h[finite] * k[finite] + k[finite] ** 2) / (2 * q)) / (
        2 * np.pi * np.sqrt(q)
    )
    np.testing.assert_allclose(pdf[finite], expected, rtol=1e-14)
    assert (pdf[~finite] == 0).all()
    assert (drho[~finite] == 0).all()


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_ranks_match_rankdata_on_the_expanded_multiset(ties):
    """Ranks by cumulative weight (average rank in a tie group) equal ``rankdata`` over the pairs repeated ``w`` times."""
    rng = np.random.default_rng(11)
    n = 200
    v = rng.integers(0, 12, n).astype(float) if ties else rng.normal(size=n)
    w = rng.integers(0, 4, n).astype(float)
    ranks = kernels.weighted_ranks(np.argsort(v), v, w)
    idx = np.repeat(np.arange(n), w.astype(int))
    np.testing.assert_allclose(ranks[idx], rankdata(v[idx]), atol=1e-12)
    assert (ranks[w == 0] == 0).all()


def test_stratum_moments_and_degenerate_strata():
    """Weighted moments skip zero-weight pairs; a stratum is degenerate when its weighted values are constant."""
    x = np.array([1.0, 2.0, 2.0, 5.0, 7.0, 3.0])
    code = np.array([0, 0, 1, 1, 2, 2])
    w = np.array([1.0, 2.0, 1.0, 0.0, 1.0, 1.0])
    total, mean, var, lo, hi = kernels.stratum_moments(x, code, w, 4)
    np.testing.assert_allclose(total, [3, 1, 2, 0])
    np.testing.assert_allclose(mean, [5 / 3, 2.0, 5.0, 0.0])
    np.testing.assert_allclose(var, [2 / 9, 0.0, 4.0, 0.0])
    np.testing.assert_array_equal(degenerate_strata(x, code, w), [False, True, False])
    np.testing.assert_array_equal(degenerate_strata(x, code), oracles.degenerate_strata(x, code))
    assert lo[3] == np.inf
    assert hi[3] == -np.inf


# ---------------------------------------------------------------------------
# Kernel estimators against the oracles, every cell type x crude/stratified x observed/draw/permutation
# ---------------------------------------------------------------------------

KINDS = ("continuous", "binary", "ordinal")


def _mates(rng, n, r_mf=0.45, k_ordinal=4) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Mother and father values of each kind, with cohort shifts per stratum so stratified fits differ from crude."""
    draws = rng.multivariate_normal([0, 0], [[1, r_mf], [r_mf, 1]], size=n)
    m_stratum, f_stratum = rng.integers(0, 3, n), rng.integers(0, 2, n)
    shift_m, shift_f = 0.4 * m_stratum, -0.5 * f_stratum
    liab_m, liab_f = draws[:, 0] + shift_m, draws[:, 1] + shift_f
    cuts = np.quantile(np.concatenate([liab_m, liab_f]), np.linspace(0, 1, k_ordinal + 1)[1:-1])
    values = {
        "continuous": (liab_m, liab_f),
        "binary": ((liab_m > 0.3).astype(float), (liab_f > -0.2).astype(float)),
        "ordinal": (np.digitize(liab_m, cuts).astype(float), np.digitize(liab_f, cuts).astype(float)),
    }
    return values, m_stratum, f_stratum


def _cell(values, m_kind, f_kind, m_stratum, f_stratum) -> CellPairs:
    k = {"continuous": None, "binary": 2, "ordinal": 4}
    return CellPairs(values[m_kind][0], values[f_kind][1], m_stratum, f_stratum).with_levels(k[m_kind], k[f_kind])


def _oracle_estimate(name, pairs, kinds) -> Estimate:
    """The oracle's value of estimator ``name`` on ``pairs`` (already resampled by ``take``)."""
    m_kind = kinds[0]
    if name in ("pearson", "phi", "point_biserial"):
        return oracles.pearson(pairs.m, pairs.f) if _pooled_form(pairs) else oracles.stratified_pearson(pairs)
    if name == "spearman":
        return oracles.spearman(pairs.m, pairs.f)
    if name == "odds_ratio":
        return oracles.odds_ratio(pairs)
    if name in ("tetrachoric", "polychoric"):
        return oracles.polychoric(pairs)
    assert name in ("biserial", "polyserial")
    if m_kind == "continuous":
        return oracles.polyserial(pairs.m, pairs.f, pairs.m_stratum, pairs.f_stratum, pairs.f_levels)
    return oracles.polyserial(pairs.f, pairs.m, pairs.f_stratum, pairs.m_stratum, pairs.m_levels)


def _pooled_form(pairs) -> bool:
    return not pairs.m_stratum.any() and not pairs.f_stratum.any()


def _assert_agree(got, expected, context) -> None:
    if isinstance(expected, Undefined):
        assert got == expected, context
        return
    assert not isinstance(got, Undefined), (context, got, expected)
    got_value = got.value if isinstance(got, Fit) else got
    expected_value = expected.value if isinstance(expected, Fit) else expected
    assert got_value == pytest.approx(expected_value, abs=TOL), context
    if isinstance(expected, Fit):
        assert isinstance(got, Fit), context
        assert got.boundary == expected.boundary, context


@pytest.mark.parametrize(("m_kind", "f_kind"), list(itertools.product(KINDS, KINDS)))
def test_every_estimator_matches_its_oracle(m_kind, f_kind):
    """Observed fit, five weighted bootstrap draws and three father permutations agree with the oracles to 1e-7.

    Crude estimators run on the pooled pairs, the stratified primary on the
    strata; the oracle sees each draw as ``take`` with repeats.
    """
    rng = np.random.default_rng(hash((m_kind, f_kind)) % 2**32)
    n = 600
    values, m_stratum, f_stratum = _mates(rng, n)
    pairs = _cell(values, m_kind, f_kind, m_stratum, f_stratum)
    estimators = CELL_ESTIMATORS[m_kind, f_kind]
    labels = rng.integers(0, n // 2, n)
    draws = [np.bincount(rng.integers(0, n // 2, n // 2), minlength=n // 2)[labels].astype(float) for _ in range(5)]
    permutations = [rng.permutation(n) for _ in range(3)]
    kinds = (m_kind, f_kind)
    checks = 0
    for estimator, stratified in [(e, False) for e in estimators.crude] + [(estimators.stratified, True)]:
        form = pairs if stratified else pooled(pairs)
        observed = estimator.fn(pairs, unit_weights(n), None)
        _assert_agree(observed, _oracle_estimate(estimator.name, form, kinds), (estimator.name, stratified, "obs"))
        start = observed.value if isinstance(observed, Fit) else None
        for d, w in enumerate(draws):
            idx = np.repeat(np.arange(n), w.astype(int))
            got = estimator.fn(pairs, w, start)
            _assert_agree(got, _oracle_estimate(estimator.name, form.take(idx), kinds), (estimator.name, stratified, d))
            checks += 1
        for perm in permutations:
            permuted = CellPairs(
                pairs.m, pairs.f[perm], pairs.m_stratum, pairs.f_stratum, pairs.m_levels, pairs.f_levels
            )
            got = estimator.fn(permuted, unit_weights(n), start)
            expected = _oracle_estimate(estimator.name, permuted if stratified else pooled(permuted), kinds)
            _assert_agree(got, expected, (estimator.name, stratified, "perm"))
            checks += 1
    assert checks == 8 * (len(estimators.crude) + 1)


@pytest.mark.parametrize(
    "table",
    [
        [[13, 6, 0], [69, 113, 22], [41, 132, 104]],  # Olsson 1979 Table 7
        [[30, 10], [0, 20]],  # plateau to the bound: boundary
        [[0, 20], [30, 10]],  # the negative mirror
        [[10, 10, 10], [10, 0, 10], [10, 10, 10]],  # empty centre, interior ρ̂ = 0
        [[10, 10], [0, 0]],  # constant margin
        [[0, 0], [0, 0]],  # no pairs
        [[50, 0, 0], [0, 50, 0], [0, 0, 50]],  # diagonal, boundary
        [[1, 0], [0, 1]],
    ],
)
def test_polychoric_edge_tables_match_oracle(table):
    """Zero cells, boundary plateaus, constant margins and empty tables give the oracle's value, flag or reason."""
    pairs = _from_table(table)
    got, expected = polychoric(pairs), oracles.polychoric(pairs)
    _assert_agree(got, expected, table)
    _assert_agree(
        odds_ratio(pairs) if np.asarray(table).shape == (2, 2) else got,
        oracles.odds_ratio(pairs) if np.asarray(table).shape == (2, 2) else expected,
        table,
    )


def test_stratified_polychoric_with_a_zero_stratum_matches_oracle():
    """One stratum's sub-table has a zero while the shared ρ stays interior; both forms agree with the oracle."""
    tables = np.array([[[15, 5], [0, 10]], [[10, 10], [10, 10]]])
    pairs = _from_table(tables, strata=np.array([0, 1]))
    _assert_agree(polychoric(pairs), oracles.polychoric(pairs), "stratified")
    _assert_agree(polychoric(pooled(pairs)), oracles.polychoric(pooled(pairs)), "pooled")


def test_polyserial_edge_cases_match_oracle():
    """Empty category in a draw, a constant continuous side, a degenerate stratum and no pairs give the oracle's reasons."""
    rng = np.random.default_rng(2)
    n = 120
    x = rng.normal(size=n)
    y = np.digitize(0.5 * x + rng.normal(size=n), [-0.5, 0.9]).astype(float)
    one = np.zeros(n, dtype=np.int64)
    levels = np.ones((1, 3), dtype=bool)
    w = np.where(y == 2, 0.0, 1.0)
    idx = np.flatnonzero(w)
    assert polyserial(x, y, one, one, levels, w) == oracles.polyserial(x[idx], y[idx], one[idx], one[idx], levels)
    assert polyserial(x, y, one, one, levels, w) == Undefined("empty_category")
    assert polyserial(np.ones(n), y, one, one, levels) == oracles.polyserial(np.ones(n), y, one, one, levels)
    two = (np.arange(n) % 2).astype(np.int64)
    x_degenerate = np.where(two == 1, 3.0, x)
    assert polyserial(x_degenerate, y, two, one, levels) == Undefined("degenerate_stratum")
    assert polyserial(x_degenerate, y, two, one, levels) == oracles.polyserial(x_degenerate, y, two, one, levels)
    assert polyserial(x[:0], y[:0], one[:0], one[:0], levels) == Undefined("no_complete_pairs")
    assert polyserial(x, y, one, one, levels, np.zeros(n)) == Undefined("no_complete_pairs")
    boundary = polyserial(x, (x > 0).astype(float), one, one, np.ones((1, 2), dtype=bool))
    assert isinstance(boundary, Fit)
    assert boundary.boundary
    _assert_agree(boundary, oracles.polyserial(x, (x > 0).astype(float), one, one, np.ones((1, 2), dtype=bool)), "cut")


def test_pearson_spearman_and_standardise_match_oracle_with_weights_and_ties():
    """Weighted Pearson, Spearman (ties included) and within-stratum standardisation equal the oracles on the repeats."""
    rng = np.random.default_rng(9)
    n = 300
    m = rng.normal(size=n)
    f = 0.4 * m + rng.normal(size=n)
    f[:50] = f[0]
    w = rng.integers(0, 3, n).astype(float)
    idx = np.repeat(np.arange(n), w.astype(int))
    assert pearson(m, f, w) == pytest.approx(oracles.pearson(m[idx], f[idx]), abs=1e-12)
    assert spearman(m, f, w) == pytest.approx(oracles.spearman(m[idx], f[idx]), abs=1e-12)
    code = rng.integers(0, 4, n)
    np.testing.assert_allclose(standardise(m, code, w)[idx], oracles.standardise(m[idx], code[idx]), atol=1e-12)
    pairs = CellPairs(m, f, code, rng.integers(0, 2, n))
    assert stratified_pearson(pairs, w) == pytest.approx(oracles.stratified_pearson(pairs.take(idx)), abs=1e-12)
    assert pearson(m, f, np.zeros(n)) == Undefined("no_complete_pairs")
    assert pearson(np.ones(n), f) == Undefined("constant_margin")
    assert spearman(m, np.ones(n)) == Undefined("constant_margin")
    one_pair_in_stratum_3 = np.where(code == 3, 0.0, 1.0)
    one_pair_in_stratum_3[np.flatnonzero(code == 3)[0]] = 1.0
    assert stratified_pearson(pairs, one_pair_in_stratum_3) == Undefined("degenerate_stratum")


# ---------------------------------------------------------------------------
# Blocked reductions: the same numbers on every thread count and on the serial twins
# ---------------------------------------------------------------------------


def _reduction_fixture(n: int) -> dict:
    rng = np.random.default_rng(11)
    x = rng.normal(size=n)
    y = (x + rng.normal(size=n) > 0.3).astype(np.float64)
    m = rng.integers(0, 3, size=n).astype(np.float64)
    xs = rng.integers(0, 3, size=n)
    ys = rng.integers(0, 2, size=n)
    w = rng.integers(0, 3, size=n).astype(np.float64)
    return {"x": x, "y": y, "m": m, "xs": xs, "ys": ys, "w": w}


def _reduction_results(n: int) -> dict:
    """Every blocked kernel on one fixture of ``n`` pairs, as JSON-ready lists (bit-exact through ``repr``)."""
    f = _reduction_fixture(n)
    x, y, m, xs, ys, w = f["x"], f["y"], f["m"], f["xs"], f["ys"], f["w"]
    total, mean, var, lo, hi = kernels.stratum_moments(x, xs, w, 3)
    margin = kernels.margin(y, ys, w, 2, 2)
    tau = kernels.thresholds(margin)
    inv_sd = 1 / np.sqrt(var)
    args = (x, xs, mean, inv_sd, y, ys, tau, w)
    influence, a_rr = kernels.polyserial_influence(0.3, x, xs, mean, var, y, ys, tau, w)
    _, mean_y, var_y, _, _ = kernels.stratum_moments(y, ys, w, 2)
    table = kernels.count_table(xs, ys, m, y, w, 3, 2, 3, 2)
    out = {
        "moments": [total, mean, var, lo, hi],
        "pearson": kernels.pearson(x, y, w),
        "margin": margin,
        "table": table,
        "nll": kernels.polyserial_nll(0.3, *args),
        "terms": kernels.polyserial_terms(0.3, *args),
        "influence": (influence, a_rr),
        "pearson_influence": kernels.pearson_influence(x, y, xs, ys, mean, var, mean_y, var_y, w),
        "gather": kernels.gather_influence(table, xs, ys, m, y, w),
    }
    return json.loads(json.dumps(out, default=lambda a: a.tolist()))


def test_serial_twins_equal_the_parallel_kernels():
    """The draw kernels' serial twins reduce the same fixed blocks, so they return the parallel kernels' bits."""
    f = _reduction_fixture(3 * kernels._BLOCK + 101)
    x, y, xs, ys, w = f["x"], f["y"], f["xs"], f["ys"], f["w"]
    for got, expected in zip(
        kernels.stratum_moments_serial(x, xs, w, 3), kernels.stratum_moments(x, xs, w, 3), strict=True
    ):
        np.testing.assert_array_equal(got, expected)
    assert kernels.pearson_serial(x, y, w) == kernels.pearson(x, y, w)
    np.testing.assert_array_equal(kernels.margin_serial(y, ys, w, 2, 2), kernels.margin(y, ys, w, 2, 2))
    _, mean, var, _, _ = kernels.stratum_moments(x, xs, w, 3)
    tau = kernels.thresholds(kernels.margin(y, ys, w, 2, 2))
    args = (x, xs, mean, 1 / np.sqrt(var), y, ys, tau, w)
    assert kernels.polyserial_terms_serial(0.3, *args) == kernels.polyserial_terms(0.3, *args)
    assert kernels.polyserial_terms(0.3, *args)[0] == kernels.polyserial_nll(0.3, *args)


def test_blocked_reductions_are_thread_count_invariant():
    """Every parallel reduction gives identical bits on 1 thread and on 4 (a subprocess, since the suite pins numba to 1)."""
    n = 3 * kernels._BLOCK + 101
    one = _reduction_results(n)
    four = four_thread_json(
        "import numba; numba.set_num_threads(4); from test_assortative_kernels import _reduction_results; "
        f"print(json.dumps({{'threads': numba.get_num_threads(), 'out': _reduction_results({n})}}))"
    )
    assert four["threads"] == 4
    assert four["out"] == one
