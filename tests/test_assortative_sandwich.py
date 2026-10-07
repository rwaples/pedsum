"""The Mate-Network cluster-robust two-step sandwich SE and the Wald CI it feeds (plan v3, decision 16)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import assortative_oracles as oracles
import numpy as np
import pytest
from test_assortative_mating import _binary, _pedigree, _trait

from pedsum.assortative_mating import (
    CELL_ESTIMATORS,
    CI_LEVEL,
    CellPairs,
    Fit,
    Undefined,
    bootstrap_networks,
    cluster_se,
    compute_assortative_mating,
    estimate_value,
    pearson_influence,
    polychoric_influence,
    pooled,
    sandwich_se,
    unit_weights,
    wald_ci,
)

if TYPE_CHECKING:
    from pedsum.assortative_mating import Estimator

#: Production SE against the finite-difference oracle. Central differences with step 1e-5 on smooth
#: equations leave about 1e-9; 1e-6 is three orders above that and far below any use of an SE.
ORACLE_TOL = 1e-6
#: Production SE against the refit bootstrap SD at B = 600 draws on 400 pairs in 200 networks. The
#: bootstrap SD itself has Monte Carlo error 1/sqrt(2(B-1)) = 2.9%, and both are O(1/G)-biased
#: estimates of the same variance; measured ratios on two seeds were 0.93-1.06, so 15% is about
#: twice the worst observed deviation and more than four Monte Carlo SDs.
BOOTSTRAP_TOL = 0.15
BOOTSTRAP_DRAWS = 600

KINDS = ("continuous", "binary", "ordinal")
PRIMARY_CELLS = [
    ("continuous", "continuous"),
    ("binary", "binary"),
    ("continuous", "binary"),
    ("binary", "continuous"),
    ("ordinal", "ordinal"),
    ("continuous", "ordinal"),
    ("ordinal", "continuous"),
]


def _remated_cell(
    seed: int, n: int, mother_kind: str, father_kind: str, network_effect: float = 0.0
) -> tuple[CellPairs, np.ndarray]:
    """A cell of ``n`` pairs in ``n/2`` two-pair Mate Networks with mild cohort shifts; ``(pairs, labels)``."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n // 2), 2)
    rng.shuffle(labels)
    m_stratum, f_stratum = rng.integers(0, 2, n), rng.integers(0, 3, n)
    shared = rng.normal(size=n // 2)[labels] * network_effect
    latent = rng.multivariate_normal([0, 0], [[1, 0.35], [0.35, 1]], n) + shared[:, None]
    latent += np.column_stack([0.3 * m_stratum, 0.2 * f_stratum])

    def side(kind: str, v: np.ndarray, cuts: dict) -> tuple[np.ndarray, int | None]:
        if kind == "continuous":
            return v, None
        return np.digitize(v, cuts[kind]).astype(np.float64), len(cuts[kind]) + 1

    m, k_m = side(mother_kind, latent[:, 0], {"binary": [0.3], "ordinal": [-0.5, 0.6]})
    f, k_f = side(father_kind, latent[:, 1], {"binary": [0.0], "ordinal": [-0.2, 0.9]})
    return CellPairs(m, f, m_stratum, f_stratum).with_levels(k_m, k_f), labels


def _oracle_se(estimator: Estimator, pairs: CellPairs, value: float, labels: np.ndarray, mother_kind: str) -> float:
    if estimator.name in ("pearson", "phi", "point_biserial"):
        psi, theta = oracles.pearson_equations(pairs)
    elif estimator.name in ("tetrachoric", "polychoric"):
        psi, theta = oracles.polychoric_equations(pairs, value)
    elif estimator.name == "odds_ratio":
        psi, theta = oracles.odds_ratio_equations(pairs)
    elif mother_kind == "continuous":
        psi, theta = oracles.polyserial_equations(
            pairs.m, pairs.f, pairs.m_stratum, pairs.f_stratum, pairs.f_levels.shape[1], value
        )
    else:
        psi, theta = oracles.polyserial_equations(
            pairs.f, pairs.m, pairs.f_stratum, pairs.m_stratum, pairs.m_levels.shape[1], value
        )
    return oracles.sandwich_se(psi, theta, labels)


def _estimators(mother_kind: str, father_kind: str) -> list[tuple[Estimator, bool]]:
    cell = CELL_ESTIMATORS[mother_kind, father_kind]
    return [(e, False) for e in cell.crude if e.influence is not None] + [(cell.stratified, True)]


@pytest.mark.parametrize(("mother_kind", "father_kind"), [(a, b) for a in KINDS for b in KINDS])
def test_sandwich_matches_the_numerical_oracle(mother_kind, father_kind):
    """Every estimator's clustered sandwich SE, crude and stratified, equals the finite-difference oracle to 1e-6."""
    pairs, labels = _remated_cell(3, 240, mother_kind, father_kind)
    for estimator, stratified in _estimators(mother_kind, father_kind):
        observed = estimator.fn(pairs, unit_weights(pairs.m.size), None)
        assert not isinstance(observed, Undefined)
        se = sandwich_se(estimator, pairs, observed, labels)
        assert isinstance(se, float), (estimator.name, se)
        sample = pairs if stratified else pooled(pairs)
        expected = _oracle_se(estimator, sample, estimate_value(observed), labels, mother_kind)
        assert se == pytest.approx(expected, rel=ORACLE_TOL), (estimator.name, stratified)


@pytest.mark.parametrize(("mother_kind", "father_kind"), PRIMARY_CELLS)
def test_sandwich_tracks_the_refit_bootstrap_sd(mother_kind, father_kind):
    """On remated data with a shared network effect, the sandwich SE is within 15% of the refit Mate-Network bootstrap SD."""
    pairs, labels = _remated_cell(0, 400, mother_kind, father_kind, network_effect=0.5)
    n = pairs.m.size
    for estimator, _ in _estimators(mother_kind, father_kind):
        observed = estimator.fn(pairs, unit_weights(n), None)
        value = estimate_value(observed)
        se = sandwich_se(estimator, pairs, observed, labels)
        start = value if isinstance(observed, Fit) else None
        draws = [estimator.fn(pairs, w, start) for w in bootstrap_networks(labels, BOOTSTRAP_DRAWS, 7)]
        values = np.array([estimate_value(d) for d in draws if not isinstance(d, Undefined)])
        if estimator.ci_scale == "log":
            values = np.log(values)
        assert values.size == BOOTSTRAP_DRAWS
        assert se == pytest.approx(values.std(ddof=1), rel=BOOTSTRAP_TOL), estimator.name


def test_cluster_se_normalisation():
    """``Var = G/(G−1) Σ_g (Σ_{i∈g} IF_i)²``: singleton clusters give the i.i.d. sandwich times sqrt(N/(N−1))."""
    rng = np.random.default_rng(1)
    influence = rng.normal(size=50) / 50
    iid = float(np.sqrt(np.sum(influence**2)))
    assert cluster_se(influence, np.arange(50)) == pytest.approx(iid * np.sqrt(50 / 49), rel=1e-12)
    labels = np.repeat(np.arange(10), 5)
    sums = influence.reshape(10, 5).sum(axis=1)
    assert cluster_se(influence, labels) == pytest.approx(np.sqrt(10 / 9 * np.sum(sums**2)), rel=1e-12)


def test_no_remating_equals_the_iid_sandwich_scaled():
    """With every pair its own Mate Network the clustered SE is the unclustered sandwich times sqrt(G/(G−1))."""
    pairs, _ = _remated_cell(5, 200, "ordinal", "ordinal")
    singletons = np.arange(200)
    for estimator, stratified in _estimators("ordinal", "ordinal"):
        observed = estimator.fn(pairs, unit_weights(200), None)
        influence = estimator.influence(pairs, unit_weights(200), estimate_value(observed))
        iid = np.sqrt(np.sum(influence**2))
        assert sandwich_se(estimator, pairs, observed, singletons) == pytest.approx(iid * np.sqrt(200 / 199), rel=1e-12)
        sample = pairs if stratified else pooled(pairs)
        expected = _oracle_se(estimator, sample, estimate_value(observed), singletons, "ordinal")
        assert iid * np.sqrt(200 / 199) == pytest.approx(expected, rel=ORACLE_TOL)


def test_crude_pearson_influence_is_the_classic_formula():
    """One stratum per side reduces the kernel to ``(z_a z_b − r (z_a² + z_b²) / 2) / N``."""
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(2, 60))
    b += 0.5 * a
    zeros = np.zeros(60, dtype=np.int64)
    pairs = CellPairs(a, b, zeros, zeros)
    za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    r = np.mean(za * zb)
    expected = (za * zb - r * (za**2 + zb**2) / 2) / 60
    np.testing.assert_allclose(pearson_influence(pairs, unit_weights(60), r), expected, rtol=1e-12, atol=1e-15)


def test_weighted_influence_matches_the_expanded_sample():
    """Frequency weights give each pair the influence it has in the sample that repeats it ``w`` times (phase 1's contract)."""
    pairs, _ = _remated_cell(8, 120, "binary", "ordinal")
    rng = np.random.default_rng(8)
    w = rng.integers(0, 3, 120).astype(np.float64)
    expanded = pairs.take(np.repeat(np.arange(120), w.astype(np.int64)))
    for influence in (polychoric_influence, pearson_influence):
        rho = estimate_value(CELL_ESTIMATORS["binary", "ordinal"].stratified.fn(pairs, w, None))
        weighted = influence(pairs, w, rho)
        unweighted = influence(expanded, unit_weights(expanded.m.size), rho)
        np.testing.assert_allclose(weighted[w > 0], unweighted[np.cumsum(w.astype(np.int64))[w > 0] - 1], rtol=1e-12)
        assert not weighted[w == 0].any()


def test_olsson_table_7_sandwich_matches_the_oracle():
    """The i.i.d. two-step sandwich on Olsson's Table 7 (a zero cell included) matches the oracle; no printed two-step SE exists."""
    table = np.array([[13, 6, 0], [69, 113, 22], [41, 132, 104]])
    m, f = (np.repeat(np.arange(3), table.sum(axis=1)), np.concatenate([np.repeat(np.arange(3), row) for row in table]))
    zeros = np.zeros(500, dtype=np.int64)
    pairs = CellPairs(m.astype(np.float64), f.astype(np.float64), zeros, zeros).with_levels(3, 3)
    estimator = CELL_ESTIMATORS["ordinal", "ordinal"].crude[0]
    observed = estimator.fn(pairs, unit_weights(500), None)
    se = sandwich_se(estimator, pairs, observed, np.arange(500))
    assert se == pytest.approx(
        _oracle_se(estimator, pairs, estimate_value(observed), np.arange(500), "ordinal"), rel=ORACLE_TOL
    )
    # Olsson's printed .048 is the full-ML expected-information SE (equations.md C1.5); the two-step
    # sandwich is of the same order, which is all the paper claims (p. 458). Not a primary-source assertion.
    assert 0.03 < se < 0.07


@pytest.mark.parametrize("rho", [0.0, 0.6, -0.95, 0.99])
def test_fisher_z_interval_stays_inside_the_unit_interval(rho):
    """Even with a large SE near ±1 the Wald interval is inside (−1, 1) and contains the estimate."""
    lo, hi = wald_ci(rho, 0.05, "fisher_z")
    assert -1 < lo < rho < hi < 1
    lo, hi = wald_ci(rho, 1e-9, "fisher_z")
    assert lo == pytest.approx(rho, abs=1e-8)
    assert hi == pytest.approx(rho, abs=1e-8)


def test_log_scale_interval_for_the_odds_ratio():
    """The odds-ratio interval is ``exp(log OR ± z·se)`` at the 95% level."""
    lo, hi = wald_ci(4.0, 0.25, "log")
    z = 1.959963984540054
    assert (lo, hi) == pytest.approx((4.0 * np.exp(-z * 0.25), 4.0 * np.exp(z * 0.25)))
    assert CI_LEVEL == 0.95


# ---------------------------------------------------------------------------
# Payload: reasons and the ci_method switch
# ---------------------------------------------------------------------------


def _binary_cell_payload(rows: list[tuple[int, int]], **options) -> dict:
    pairs = [(k + 1, 1000 + k) for k in range(len(rows))]
    values = {m: float(mv) for (m, _), (mv, _) in zip(pairs, rows, strict=True)}
    values |= {f: float(fv) for (_, f), (_, fv) in zip(pairs, rows, strict=True)}
    df = _pedigree(pairs)
    return compute_assortative_mating(df, [_binary(df, values)], permutations=0, seed=0, **options)


def test_boundary_and_infinite_odds_ratio_withhold_the_sandwich_ci():
    """A tetrachoric fit at the bound and an infinite odds ratio report null CIs with their reasons; phi keeps its CI."""
    out = _binary_cell_payload([(0, 0)] * 30 + [(0, 1)] * 10 + [(1, 1)] * 20, bootstrap=0)
    crude = out["mate_correlation"][0]["crude"]
    assert out["settings"]["ci_method"] == "sandwich"
    assert crude["tetrachoric"]["boundary"] is True
    assert (crude["tetrachoric"]["se"], crude["tetrachoric"]["ci"], crude["tetrachoric"]["ci_method"]) == (
        None,
        None,
        None,
    )
    assert crude["tetrachoric"]["ci_unavailable_reason"] == "boundary"
    assert crude["odds_ratio"]["value"] == np.inf
    assert (crude["odds_ratio"]["se"], crude["odds_ratio"]["ci"]) == (None, None)
    assert crude["odds_ratio"]["ci_unavailable_reason"] == "infinite_odds_ratio"
    assert crude["phi"]["ci_method"] == "sandwich"
    assert crude["phi"]["ci"][0] < crude["phi"]["r"] < crude["phi"]["ci"][1]
    assert "bootstrap" not in crude["phi"]


def test_sandwich_ci_is_the_fisher_z_wald_interval():
    """A regular binary cell reports ``se`` and the Wald CI on the Fisher-z (tetrachoric, phi) and log (odds ratio) scales."""
    rng = np.random.default_rng(4)
    latent = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], 200)
    rows = [(int(a > 0), int(b > 0.2)) for a, b in latent]
    crude = _binary_cell_payload(rows, bootstrap=0)["mate_correlation"][0]["crude"]
    for name, key in (("tetrachoric", "rho"), ("phi", "r")):
        record = crude[name]
        assert (record["ci_method"], record["ci_unavailable_reason"]) == ("sandwich", None)
        assert record["ci"] == pytest.approx(wald_ci(record[key], record["se"], "fisher_z"))
    odds = crude["odds_ratio"]
    assert odds["ci"] == pytest.approx(wald_ci(odds["value"], odds["se"], "log"))


def test_bootstrap_request_keeps_the_percentile_ci_and_reports_the_sandwich_se():
    """With ``--bootstrap N > 0`` the CI is the refit bootstrap's and ``se`` is still the sandwich."""
    rng = np.random.default_rng(6)
    latent = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], 150)
    rows = [(int(a > 0), int(b > 0.2)) for a, b in latent]
    with_bootstrap = _binary_cell_payload(rows, bootstrap=50)
    without = _binary_cell_payload(rows, bootstrap=0)
    assert with_bootstrap["settings"]["ci_method"] == "bootstrap"
    boot, sand = (p["mate_correlation"][0]["crude"]["tetrachoric"] for p in (with_bootstrap, without))
    assert (boot["ci_method"], boot["bootstrap"]["requested"]) == ("bootstrap", 50)
    assert boot["se"] == sand["se"]
    assert boot["ci"] != sand["ci"]


def test_single_network_and_spearman_reasons():
    """One Mate Network withholds every CI; Spearman has no sandwich and says the bootstrap was not requested."""
    chain = [(1, 100), (1, 101), (2, 101), (2, 102), (3, 102)]
    df = _pedigree(chain)
    values = {1: 0.0, 2: 1.0, 3: 3.0, 100: 0.5, 101: 2.0, 102: 1.0}
    trait = _trait(df, values)
    crude = compute_assortative_mating(df, [trait], permutations=0, bootstrap=0, seed=0)["mate_correlation"][0]["crude"]
    assert crude["pearson"]["ci_unavailable_reason"] == "single_mate_network"
    assert crude["pearson"]["se"] is None

    pairs = [(k + 1, 1000 + k) for k in range(40)]
    df = _pedigree(pairs)
    rng = np.random.default_rng(9)
    values = {p: float(v) for p, v in zip([p for pair in pairs for p in pair], rng.normal(size=80), strict=True)}
    trait = _trait(df, values)
    crude = compute_assortative_mating(df, [trait], permutations=0, bootstrap=0, seed=0)["mate_correlation"][0]["crude"]
    assert crude["spearman"] == {
        "r": crude["spearman"]["r"],
        "se": None,
        "ci": None,
        "ci_method": None,
        "ci_unavailable_reason": "bootstrap_not_requested",
    }
    assert crude["pearson"]["ci_method"] == "sandwich"
