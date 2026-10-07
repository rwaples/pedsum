"""The opt-in one-step Mate-Network bootstrap (plan v3, decision 16): agreement with the full refit, exactness of the closed forms, invariances."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from conftest import four_thread_json
from test_assortative_permutation import _frame
from test_assortative_sandwich import KINDS, PRIMARY_CELLS, _remated_cell

from pedsum import assortative_kernels as kernels
from pedsum import assortative_mating as am
from pedsum.assortative_mating import (
    CELL_ESTIMATORS,
    CI_LEVEL,
    FIT_OUTCOMES,
    INFERENCE,
    Fit,
    Undefined,
    _draw_estimates,
    bootstrap_networks,
    compute_assortative_mating,
    estimate_value,
    fit_cell,
    network_weights,
)

if TYPE_CHECKING:
    from pedsum.assortative_mating import CellPairs

#: One-step CI bound against the full-refit CI bound on the same draws, 2,500 remated pairs, B = 400.
#: The one-step error is O(1/n): over 4 seeds x 7 latent cells x 2 forms the worst bound difference
#: was 0.0012 at n = 2500, 0.0018 at 2000 and 0.0036 at 1000 (development measurement), so 0.002
#: is 1.7x the worst observed at this n and the design's 1e-3 tolerance is met from n ~ 5000.
ONE_STEP_TOL = 0.002
#: A closed-form estimator is recomputed, not approximated, on a draw: only summation order differs.
EXACT_TOL = 1e-9
DRAWS = 400
N_PAIRS = 2500
SEED = 7


def _ci(values: np.ndarray) -> np.ndarray:
    """The percentile CI as ``bootstrap_record`` takes it (its tail is ``0.025 + 2e-17``, one order statistic off a literal 0.025 at B = 400)."""
    tail = (1 - CI_LEVEL) / 2
    return np.quantile(values, [tail, 1 - tail], method="inverted_cdf")


def _refit_draws(estimator, pairs: CellPairs, labels: np.ndarray, observed) -> list:
    start = estimate_value(observed) if isinstance(observed, Fit) else None
    return [estimator.fn(pairs, w, start) for w in bootstrap_networks(labels, DRAWS, SEED)]


def _values(draws: list) -> np.ndarray:
    return np.array([estimate_value(d) for d in draws if not isinstance(d, Undefined)])


@pytest.mark.parametrize(
    ("mother_kind", "father_kind"), [c for c in PRIMARY_CELLS if c != ("continuous", "continuous")]
)
def test_one_step_ci_matches_the_full_refit_ci(mother_kind, father_kind):
    """Crude and stratified latent CIs from one Newton step per draw are within 0.002 per bound of the refit CIs on the same draws."""
    pairs, labels = _remated_cell(0, N_PAIRS, mother_kind, father_kind, network_effect=0.5)
    cell = CELL_ESTIMATORS[mother_kind, father_kind]
    observed, records = fit_cell(cell, pairs, labels, DRAWS, SEED, True)
    fitted = cell.fitted(True)
    for slot in (0, len(fitted) - 1):
        refit = _values(_refit_draws(fitted[slot], pairs, labels, observed[slot]))
        assert refit.size == DRAWS
        assert records[slot]["bootstrap"]["valid"] == DRAWS
        assert records[slot]["ci_method"] == "bootstrap"
        np.testing.assert_allclose(records[slot]["ci"], _ci(refit), atol=ONE_STEP_TOL, rtol=0)


@pytest.mark.parametrize(("mother_kind", "father_kind"), [(a, b) for a in KINDS for b in KINDS])
def test_closed_form_estimators_are_exact_on_every_draw(mother_kind, father_kind):
    """Pearson, stratified Pearson, Spearman, phi, point-biserial and the odds ratio equal their refit on each draw."""
    pairs, labels = _remated_cell(1, 300, mother_kind, father_kind)
    cell = CELL_ESTIMATORS[mother_kind, father_kind]
    fitted = cell.fitted(True)
    observed = [e.fn(pairs, np.ones(pairs.m.size), None) for e in fitted]
    starts = [float("nan") if isinstance(o, Undefined) else estimate_value(o) for o in observed]
    values, status = cell.draws(pairs, labels, 60, SEED, starts)
    checked = 0
    for k, estimator in enumerate(fitted):
        if estimator.name in ("tetrachoric", "polychoric", "biserial", "polyserial"):
            continue
        for d, w in enumerate(bootstrap_networks(labels, 60, SEED)):
            expected = estimator.fn(pairs, w, None)
            if isinstance(expected, Undefined):
                assert status[d, k] != 0
            else:
                assert status[d, k] == 0
                assert values[d, k] == pytest.approx(expected, abs=EXACT_TOL), (estimator.name, d)
        checked += 1
    assert checked == len(fitted) - (0 if cell.crude[0].name == "pearson" else 2)


@pytest.mark.parametrize(("mother_kind", "father_kind"), [("binary", "binary"), ("ordinal", "ordinal")])
def test_table_draws_run_in_bounded_chunks_with_identical_draws(monkeypatch, mother_kind, father_kind):
    """The draws' count tables are built a bounded number of draws at a time, and the draws equal one launch bit for bit."""
    pairs, labels = _remated_cell(1, 300, mother_kind, father_kind)
    cell = CELL_ESTIMATORS[mother_kind, father_kind]
    starts = [estimate_value(e.fn(pairs, np.ones(pairs.m.size), None)) for e in cell.fitted(True)]
    launches: list[int] = []
    table_draws = kernels.table_draws

    def spy(*args):
        tables = table_draws(*args)
        launches.append(tables.shape[0])
        return tables

    monkeypatch.setattr(kernels, "table_draws", spy)
    one_launch = cell.draws(pairs, labels, 10, SEED, starts)
    assert launches == [10]
    launches.clear()
    monkeypatch.setattr(am, "_TABLE_CHUNK", 3 * am.count_table(pairs).size)
    chunked = cell.draws(pairs, labels, 10, SEED, starts)
    assert launches == [3, 3, 3, 1]
    for a, b in zip(one_launch, chunked, strict=True):
        np.testing.assert_array_equal(a, b)


def test_network_weights_are_whole_network_multiplicities_in_their_own_stream():
    """A draw weights whole networks by how often each was drawn among ``G`` draws; the stream is not the permutations'."""
    labels = np.array([0, 1, 0, 2, 2, 2, 3])
    for d, w in enumerate(bootstrap_networks(labels, 40, 3)):
        assert w.dtype == np.float64
        np.testing.assert_array_equal(w, network_weights(labels, 3, d))
        per_network = [np.unique(w[labels == label]) for label in range(4)]
        assert all(len(m) == 1 for m in per_network)
        multiplicities = np.concatenate(per_network)
        assert np.all(multiplicities == multiplicities.astype(int))
        assert multiplicities.sum() == 4
    assert not np.array_equal(network_weights(labels, 3, 0), network_weights(labels, 4, 0))
    for seed, draw in [(0, 0), (3, 5), (2**31, 1)]:
        assert int(kernels.bootstrap_state(seed, draw)) != int(kernels.stream_state(seed, draw))


def test_a_nonconcave_draw_is_refit_in_full():
    """A draw the kernel marks ``STATUS_NONCONCAVE`` is refit from the observed estimate on that draw's weights; other codes are reasons."""
    pairs, labels = _remated_cell(2, 200, "continuous", "binary")
    estimator = CELL_ESTIMATORS["continuous", "binary"].crude[0]
    observed = estimator.fn(pairs, np.ones(pairs.m.size), None)
    assert isinstance(observed, Fit)
    values = np.full(5, 0.1)
    status = np.array([0, kernels.STATUS_NONCONCAVE, kernels.STATUS_EMPTY_CATEGORY, kernels.STATUS_NONCONCAVE, 0])
    before = FIT_OUTCOMES["bootstrap_refit"]
    draws = _draw_estimates(estimator, pairs, labels, SEED, observed, values, status)
    assert FIT_OUTCOMES["bootstrap_refit"] - before == 2
    assert draws[0] == draws[4] == 0.1
    assert draws[2] == Undefined("empty_category")
    for d in (1, 3):
        expected = estimator.fn(pairs, network_weights(labels, SEED, d), observed.value)
        assert draws[d] == expected
        assert estimate_value(expected) != 0.1


def test_spearman_ci_needs_the_bootstrap():
    """Spearman has no sandwich: ``--bootstrap 0`` says so, ``--bootstrap N`` gives it the percentile CI."""
    df, traits = _frame(2, 120)
    without, with_bootstrap = (
        compute_assortative_mating(df, traits[:1], permutations=0, bootstrap=b, seed=0)["mate_correlation"][0]["crude"]
        for b in (0, 50)
    )
    assert without["spearman"]["ci"] is None
    assert without["spearman"]["ci_unavailable_reason"] == "bootstrap_not_requested"
    assert with_bootstrap["spearman"]["ci_method"] == "bootstrap"
    assert with_bootstrap["spearman"]["bootstrap"]["valid"] == 50
    assert with_bootstrap["spearman"]["se"] is None
    assert len(with_bootstrap["spearman"]["ci"]) == 2
    assert with_bootstrap["pearson"]["se"] == without["pearson"]["se"]


def _payload(threads: int) -> dict:
    import numba

    numba.set_num_threads(threads)
    df, traits = _frame(21, 300)
    return compute_assortative_mating(
        df, traits, permutations=0, bootstrap=150, seed=4, stratify_by="birth_year", birth_year_bin=10
    )


def test_thread_count_invariance():
    """The same seed gives identical bootstrap records on 1 thread and on 4 (a subprocess, since the suite pins numba to 1)."""
    one = _payload(1)
    assert one["settings"]["threads"] == 1
    four = four_thread_json("from test_assortative_bootstrap import _payload; print(json.dumps(_payload(4)))")
    assert (one["settings"].pop("threads"), four["settings"].pop("threads")) == (1, 4)
    assert json.loads(json.dumps(one)) == four
    with_ci = [
        record
        for cell in four["mate_correlation"]
        for form in ("crude", "stratified")
        for name, record in cell[form].items()
        if name != "table"
    ]
    assert len(with_ci) == 13
    assert all(record["ci_method"] == "bootstrap" and record["bootstrap"]["valid"] > 100 for record in with_ci)
    assert four["inference"]["bootstrap_method"] == INFERENCE["bootstrap_method"]
