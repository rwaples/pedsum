"""Repeated-dataset calibration of the assortative-mating inference (plan v3 phase 7): permutation size and CI coverage.

Each test simulates ``R`` independent Mating-Pair datasets with a known mate
correlation and counts rejections (``R_mf = 0``) or CI hits (``R_mf = 0.3``).
Monte Carlo tolerance is ``TOL_SD`` binomial standard deviations around the
nominal rate. Size is checked one-sided (a rate at or below nominal is valid
for discrete, tied and sequential p-values); coverage fails when too low and
is reported, not failed, when mildly high.

Run with ``-s`` to see the achieved rates with their binomial SDs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from pedsum.assortative_mating import Trait, bvn_cdf, compute_assortative_mating

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.slow

ALPHA = 0.05
NOMINAL_COVERAGE = 0.95
#: Binomial Monte Carlo tolerance in standard deviations of the achieved rate.
TOL_SD = 3.0
#: Coverage this high means the CI is far wider than it should be; above the "over" flag, it fails.
MAX_COVERAGE = 0.995
R_MF = 0.3
ORDINAL_CUTS = (-0.5, 0.7)
PRIMARIES = frozenset({"pearson", "tetrachoric", "biserial", "polychoric", "polyserial"})


@dataclass(frozen=True)
class Design:
    """One synthetic population: mothers mate once, fathers take ``mates_per_father`` mothers in a row.

    Every trait of a person is a function of one latent normal; the mother's
    latent is ``r_mf`` times her mate's plus independent noise, so the latent
    mate correlation is ``r_mf`` in every cell and pairs sharing a father are
    dependent (the Mate-Network clustering the sandwich must absorb).
    ``cohort_trend`` shifts both mates' latents by the father's birth decade,
    which stratifying by ``birth_year`` removes.
    """

    n_pairs: int
    mates_per_father: int
    r_mf: float = 0.0
    prevalence: float = 0.5
    cohort_trend: float = 0.0
    kinds: tuple[str, ...] = ("continuous", "binary")

    @property
    def n_fathers(self) -> int:
        """Distinct fathers; the last one may have fewer mates."""
        return -(-self.n_pairs // self.mates_per_father)

    def dataset(self, seed: int) -> tuple[pl.DataFrame, list[Trait]]:
        """One simulated pedigree (parents then one child per pair) and its traits."""
        rng = np.random.default_rng(seed)
        n_m, n_f = self.n_pairs, self.n_fathers
        father_of = np.arange(n_m) // self.mates_per_father
        decade_f = rng.integers(195, 198, n_f)
        latent_f = rng.normal(size=n_f)
        latent_m = self.r_mf * latent_f[father_of] + math.sqrt(1 - self.r_mf**2) * rng.normal(size=n_m)
        shift = self.cohort_trend * (decade_f - 196)
        latent = np.concatenate([latent_m + shift[father_of], latent_f + shift, np.full(n_m, np.nan)])

        n_parents = n_m + n_f
        df = pl.DataFrame(
            {
                "id": np.arange(1, n_parents + n_m + 1),
                "sex": np.concatenate([np.zeros(n_m), np.ones(n_f), np.zeros(n_m)]),
                "mother": np.concatenate([np.full(n_parents, -1), np.arange(1, n_m + 1)]),
                "father": np.concatenate([np.full(n_parents, -1), n_m + 1 + father_of]),
                "ped_depth": np.concatenate([np.zeros(n_parents), np.ones(n_m)]),
                "birth_year": np.concatenate([decade_f[father_of], decade_f, np.full(n_m, 200)]) * 10,
            },
            schema={
                "id": pl.Int64,
                "sex": pl.Int8,
                "mother": pl.Int64,
                "father": pl.Int64,
                "ped_depth": pl.Int32,
                "birth_year": pl.Int32,
            },
        )
        return df, [self.trait(kind, latent) for kind in self.kinds]

    @property
    def cut(self) -> float:
        """Latent threshold of the binary trait."""
        return float(norm.isf(self.prevalence))

    def trait(self, kind: str, latent: np.ndarray) -> Trait:
        """The trait of ``kind`` as a function of the per-row latent."""
        if kind == "continuous":
            return Trait("x", "continuous", "stated", None, latent)
        if kind == "binary":
            coded = (latent > self.cut).astype(np.float64)
            levels: tuple[str, ...] = ("0", "1")
        else:
            coded = np.digitize(latent, ORDINAL_CUTS).astype(np.float64)
            levels = ("0", "1", "2")
        return Trait(kind[0], kind, "inferred", levels, np.where(np.isnan(latent), np.nan, coded))

    def truth(self, estimator: str) -> float | None:
        """The population value each estimator targets; None when it has no closed form here (spearman)."""
        if estimator in PRIMARIES:
            return self.r_mf
        c, p = self.cut, self.prevalence
        if estimator == "point_biserial":
            return self.r_mf * norm.pdf(c) / math.sqrt(p * (1 - p))
        p11 = float(bvn_cdf(np.array([-c]), np.array([-c]), self.r_mf)[0])
        if estimator == "phi":
            return (p11 - p * p) / (p * (1 - p))
        if estimator == "odds_ratio":
            p10 = p - p11
            return p11 * (1 - 2 * p + p11) / (p10 * p10)
        return None


@dataclass
class Rate:
    """Hits out of ``n`` replicates with the binomial SD of the rate."""

    hits: int = 0
    n: int = 0

    @property
    def rate(self) -> float:
        """Achieved rate."""
        return self.hits / self.n

    @property
    def sd(self) -> float:
        """Binomial SD at the achieved rate."""
        return math.sqrt(self.rate * (1 - self.rate) / self.n)


def _binomial_sd(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def _cells(payload: dict) -> Iterator[tuple[str, str, str, dict]]:
    """``(cell, form, estimator, record)`` for every defined estimate in the payload."""
    for cell in payload["mate_correlation"]:
        name = cell["mother"] + cell["father"]
        for form in ("crude", "stratified"):
            for estimator, record in cell.get(form, {}).items():
                if isinstance(record, dict) and "se" in record:
                    yield name, form, estimator, record


def _replicates(design: Design, seed: int, n_reps: int, **kwargs) -> Iterator[dict]:
    for rep in range(n_reps):
        df, traits = design.dataset(seed + rep)
        yield compute_assortative_mating(df, traits, seed=seed + rep, stratify_by="birth_year", **kwargs)


def _table(title: str, nominal: float, rows: list[tuple[str, str, str, Rate, float, bool]]) -> None:
    print(f"\n{title}: nominal {nominal}, tolerance {TOL_SD} binomial SD")
    print(f"{'cell':<5}{'form':<12}{'estimator':<16}{'rate':>7}{'sd':>7}{'bound':>7}  ok")
    for cell, form, estimator, rate, bound, ok in rows:
        print(
            f"{cell:<5}{form:<12}{estimator:<16}{rate.rate:>7.3f}{rate.sd:>7.3f}{bound:>7.3f}  {'pass' if ok else 'FAIL'}"
        )


# ---------------------------------------------------------------------------
# 1. Size of the sequential score-statistic permutation p at alpha = 0.05 under R_mf = 0
# ---------------------------------------------------------------------------

SIZE_REPS = 500
SIZE_DESIGNS = {
    "no_remating": Design(600, 1),
    "heavy_remating": Design(600, 5),
    "sparse_binary": Design(1000, 3, prevalence=0.05),
}


def _rejections(design: Design, seed: int) -> dict[tuple[str, str, str], Rate]:
    rates: dict[tuple[str, str, str], Rate] = {}
    for payload in _replicates(design, seed, SIZE_REPS, permutations=999, bootstrap=0):
        for cell, form, estimator, record in _cells(payload):
            if estimator not in PRIMARIES:
                continue
            rate = rates.setdefault((cell, form, estimator), Rate())
            assert record["p_perm"] is not None, record["p_perm_unavailable_reason"]
            rate.n += 1
            rate.hits += record["p_perm"] < ALPHA
    return rates


def _check_size(title: str, rates: dict[tuple[str, str, str], Rate], forms: tuple[str, ...]) -> None:
    bound = ALPHA + TOL_SD * _binomial_sd(ALPHA, SIZE_REPS)
    rows = [(c, f, e, r, bound, r.rate <= bound) for (c, f, e), r in rates.items() if f in forms]
    _table(title, ALPHA, rows)
    assert all(ok for *_, ok in rows), [row[:3] for row in rows if not row[-1]]
    assert all(r.n == SIZE_REPS for *_, r, _b, _ok in rows)


@pytest.mark.parametrize("setting", sorted(SIZE_DESIGNS))
def test_permutation_size_under_the_null(setting):
    """Pearson, tetrachoric and biserial cells, crude and stratified, reject at most alpha + 3 SD."""
    _check_size(f"size {setting}", _rejections(SIZE_DESIGNS[setting], 100_000), ("crude", "stratified"))


def test_permutation_size_with_a_cohort_trend():
    """A birth-decade trend shared by both mates does not inflate the stratified test.

    The trend is real: the crude Pearson sandwich CI excludes zero in most
    replicates. The crude permutation p is not reported because its donors are
    drawn within father strata too (decision 17), so it is conditional on the
    cohort structure by construction.
    """
    design = Design(600, 4, cohort_trend=0.5)
    rates = _rejections(design, 200_000)
    _check_size("size cohort_trend", rates, ("stratified",))
    crude_excludes_zero = Rate()
    for payload in _replicates(design, 200_000, 100, permutations=0, bootstrap=0):
        (lo, hi) = payload["mate_correlation"][0]["crude"]["pearson"]["ci"]
        crude_excludes_zero.n += 1
        crude_excludes_zero.hits += lo > 0 or hi < 0
    print(f"positive control: crude pearson CI excludes 0 in {crude_excludes_zero.rate:.2f} of replicates")
    assert crude_excludes_zero.rate >= 0.5


# ---------------------------------------------------------------------------
# 2. Coverage of the nominal 95% CI at R_mf = 0.3
# ---------------------------------------------------------------------------

COVERAGE_REPS = 500
COVERAGE_DESIGNS = {
    "remating": Design(600, 4, r_mf=R_MF, kinds=("continuous", "binary", "ordinal")),
    "no_remating": Design(600, 1, r_mf=R_MF, kinds=("continuous", "binary", "ordinal")),
}


def _coverage(design: Design, seed: int, n_reps: int, **kwargs) -> dict[tuple[str, str, str], Rate]:
    rates: dict[tuple[str, str, str], Rate] = {}
    for payload in _replicates(design, seed, n_reps, permutations=0, **kwargs):
        for cell, form, estimator, record in _cells(payload):
            truth = design.truth(estimator)
            if truth is None:
                continue
            assert record["ci"] is not None, (cell, form, estimator, record["ci_unavailable_reason"])
            rate = rates.setdefault((cell, form, estimator), Rate())
            rate.n += 1
            rate.hits += record["ci"][0] <= truth <= record["ci"][1]
    return rates


def _check_coverage(title: str, rates: dict[tuple[str, str, str], Rate], n_reps: int) -> None:
    sd = _binomial_sd(NOMINAL_COVERAGE, n_reps)
    low, high = NOMINAL_COVERAGE - TOL_SD * sd, NOMINAL_COVERAGE + TOL_SD * sd
    rows = [(c, f, e, r, low, low <= r.rate <= MAX_COVERAGE) for (c, f, e), r in rates.items()]
    _table(title, NOMINAL_COVERAGE, rows)
    over = [(c, f, e) for c, f, e, r, *_ in rows if r.rate > high]
    if over:
        print(f"over-coverage above {high:.3f} (reported, not failed): {over}")
    assert all(ok for *_, ok in rows), [row[:3] for row in rows if not row[-1]]
    assert all(r.n == n_reps for *_, r, _b, _ok in rows)


@pytest.mark.parametrize("setting", sorted(COVERAGE_DESIGNS))
def test_sandwich_ci_coverage(setting):
    """Every sandwich CI (pearson, tetrachoric, biserial, polychoric, polyserial, phi, point-biserial, odds ratio) covers its truth."""
    rates = _coverage(COVERAGE_DESIGNS[setting], 300_000, COVERAGE_REPS, bootstrap=0)
    assert {e for _c, _f, e in rates} == PRIMARIES | {"phi", "point_biserial", "odds_ratio"}
    _check_coverage(f"sandwich coverage {setting}", rates, COVERAGE_REPS)


# ---------------------------------------------------------------------------
# 3. Opt-in one-step bootstrap coverage
# ---------------------------------------------------------------------------

BOOTSTRAP_REPS = 200
BOOTSTRAP_DRAWS = 399


def test_bootstrap_ci_coverage_with_remating():
    """The one-step Mate-Network percentile bootstrap covers the tetrachoric and Pearson truths."""
    design = Design(1000, 4, r_mf=R_MF)
    rates = _coverage(design, 400_000, BOOTSTRAP_REPS, bootstrap=BOOTSTRAP_DRAWS)
    checked = {k: r for k, r in rates.items() if k[2] in ("tetrachoric", "pearson")}
    _check_coverage("bootstrap coverage remating", checked, BOOTSTRAP_REPS)
