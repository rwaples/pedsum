"""Mate Correlation over Mating Pairs: trait typing, estimators, permutation null, Mate Network bootstrap."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import numba
import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.sparse.csgraph import connected_components
from scipy.special import ndtr, ndtri, owens_t

from pedsum import assortative_kernels as kernels
from pedsum.base import MIN_STRATUM_NETWORKS, PedigreeError
from pedsum.pedigree_ops import IdIndex, _group_mating_pairs, _parent_rows, unique_ints

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pedsum.base import TraitKind

StratifyBy = Literal["depth", "birth_year"]

#: More distinct numeric values than this makes an untyped trait continuous.
MAX_INFERRED_LEVELS = 20
CI_LEVEL = 0.95
#: A CI is published only when at least this share of bootstrap draws is valid.
MIN_VALID_DRAW_SHARE = 0.95

_BINARY_WORDS: tuple[tuple[str, str], ...] = (("false", "true"), ("no", "yes"))

INFERENCE = {
    "se_method": "cluster-robust sandwich over Mate Networks of the stacked two-step estimating equations "
    "(thresholds, stratum means and variances, then rho), G/(G-1) small-sample factor",
    "ci_scale": "Fisher z for correlations, log for the odds ratio",
    "bootstrap_unit": "mate_network",
    "bootstrap_assumption": "Mate Networks are independent. Dependence from remating is kept; "
    "dependence between networks through ancestry or siblings is not modelled.",
    "bootstrap_method": "one_step: each draw refits the first step (thresholds, stratum moments) and every "
    "closed-form estimator exactly on the resampled Mate Networks, and takes one Newton step for rho from "
    "the observed estimate with the draw-weighted score and the full-sample Hessian",
    "permutation_null": "father trait vectors exchangeable within blocks",
    "permutation_blocks": "father_stratum x father_missingness_pattern",
    "permutation_statistic": "score_at_zero: the score of the latent-correlation log-likelihood at rho = 0 with "
    "every margin refit on the permuted pairs (polychoric, tetrachoric, polyserial, biserial); "
    "pearson: Pearson r (continuous x continuous)",
    "permutation_stopping": "besag_clifford_closed",
    "permutation_stop_h": 20,
}

#: Besag & Clifford (1991) closed scheme: a cell's draws stop once ``SEQUENTIAL_H`` valid permutations are at
#: least as extreme as the observed statistic (p. 303 suggests 10 or 20). Exact under H0 at every size.
SEQUENTIAL_H = INFERENCE["permutation_stop_h"]
#: Draws in the first kernel launch. A later launch at least doubles the draws so far and runs on to where the
#: nearest open form would stop at its rate so far (``_Stopping.horizon``), so a strong-signal run takes two
#: launches and a null one stops after 64 or 128 draws. Each launch ends at a barrier with workers idle on its
#: last draws: 16 fixed launches of 64 cost 3-4% of a strong-signal kernel. The schedule follows the draws, never
#: the thread count, and the stopping decision scans in draw order, so results do not depend on it.
PERMUTATION_BATCH = 64

NOTES = [
    "Mating Pairs are observed only through offspring; partnerships without a recorded child are absent.",
    "Phenotypic correlation; tetrachoric/polychoric/polyserial assume bivariate-normal liability, "
    "the other estimators describe the observed values.",
    "Censored age-dependent diagnoses are not corrected; a mate who has not yet been diagnosed counts as unaffected.",
]


@dataclass(frozen=True)
class Trait:
    """One typed trait column, aligned to the pedigree rows.

    ``values`` is float64 per row with NaN for missing; a binary or ordinal
    trait is coded ``0..k-1`` in the order of ``levels``.
    """

    name: str
    kind: TraitKind
    type_source: Literal["inferred", "stated"]
    levels: tuple[str, ...] | None
    values: np.ndarray

    @property
    def n_missing(self) -> int:
        """Rows without a value."""
        return int(np.isnan(self.values).sum())


def _level_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def classify_trait(name: str, tokens: np.ndarray, stated: TraitKind | None = None) -> Trait:
    """Type the raw tokens of one trait column (``None`` = missing) and code its values.

    Two levels infer binary, more than ``MAX_INFERRED_LEVELS`` numeric values
    infer continuous, and anything between needs ``stated``. Binary levels are
    two numbers (the lower is the reference) or ``false/true`` / ``no/yes`` in
    any case; ordinal levels are numeric and ordered numerically. Raises
    ``PedigreeError`` on an all-missing, constant, non-finite, non-numeric or
    ambiguous column.
    """
    source: Literal["inferred", "stated"] = "inferred" if stated is None else "stated"
    column = pl.Series(name, tokens.tolist(), dtype=pl.String)
    present = column.is_not_null().to_numpy()
    if not present.any():
        raise PedigreeError(f"trait {name!r} is missing in every row")
    observed = column.drop_nulls()
    numeric = observed.cast(pl.Float64, strict=False)
    if numeric.is_null().any():
        words = sorted(set(observed.str.to_lowercase().to_list()))
        if stated in (None, "binary") and tuple(words) in _BINARY_WORDS:
            values = np.full(len(tokens), np.nan)
            values[present] = (observed.str.to_lowercase() == words[1]).cast(pl.Float64).to_numpy()
            return Trait(name, "binary", source, (words[0], words[1]), values)
        if len(words) == 1:
            raise PedigreeError(f"trait {name!r} is constant ({words[0]!r} in every non-missing row)")
        samples = observed.filter(numeric.is_null()).unique(maintain_order=True).head(3).to_list()
        raise PedigreeError(
            f"trait {name!r} must be numeric (binary traits may also use true/false or yes/no); got {samples}"
        )
    values = numeric.to_numpy()
    if not np.isfinite(values).all():
        samples = observed.filter(~numeric.is_finite()).unique(maintain_order=True).head(3).to_list()
        raise PedigreeError(f"trait {name!r} holds non-finite value(s) {samples}")
    n_levels = len(np.unique(values))
    if n_levels == 1:
        raise PedigreeError(f"trait {name!r} is constant ({_level_label(float(values[0]))} in every non-missing row)")
    kind = stated
    if kind is None:
        if n_levels == 2:
            kind = "binary"
        elif n_levels > MAX_INFERRED_LEVELS:
            kind = "continuous"
        else:
            raise PedigreeError(
                f"trait {name!r} has {n_levels} distinct numeric values; pass "
                f"--trait-type {name}=ordinal or --trait-type {name}=continuous"
            )
    if kind == "binary" and n_levels != 2:
        raise PedigreeError(f"trait {name!r} is stated binary but has {n_levels} distinct values")
    out = np.full(len(tokens), np.nan)
    if kind == "continuous":
        out[present] = values
        return Trait(name, kind, source, None, out)
    levels, codes = np.unique(values, return_inverse=True)
    out[present] = codes
    return Trait(name, kind, source, tuple(_level_label(float(v)) for v in levels), out)


# ---------------------------------------------------------------------------
# Estimators: pure functions on one cell's pair arrays and frequency weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Undefined:
    """An estimator with no value on this sample, and why."""

    reason: str


@dataclass(frozen=True)
class Fit:
    """A latent-correlation estimate; ``boundary`` is set when ρ̂ lands within ``BOUNDARY_MARGIN`` of a bound."""

    value: float
    boundary: bool


Estimate = float | Fit | Undefined


def estimate_value(estimate: float | Fit) -> float:
    """The number an estimate is written and compared by."""
    return estimate.value if isinstance(estimate, Fit) else estimate


def _estimate_fields(estimate: float | Fit) -> dict:
    return {"boundary": estimate.boundary} if isinstance(estimate, Fit) else {}


def shown_levels(codes: np.ndarray, stratum: np.ndarray, k: int) -> np.ndarray:
    """Per stratum, which of the ``k`` levels ``codes`` shows there: bool ``(n_strata, k)``."""
    present = np.zeros((int(stratum.max(initial=-1)) + 1, k), dtype=bool)
    present[stratum, codes.astype(np.int64)] = True
    return present


@dataclass(frozen=True)
class CellPairs:
    """One cell's analysed Mating Pairs: each pair's mother and father value and each mate's own stratum code.

    Stratum codes are dense (``0..k-1``); without stratification every code is 0.
    For a binary or ordinal side, ``m_levels``/``f_levels`` record the levels
    each stratum shows in the analysed sample (``shown_levels``). A resample
    keeps them, so an estimator can tell a level absent from a draw's margin
    from one the cell never had.
    """

    m: np.ndarray
    f: np.ndarray
    m_stratum: np.ndarray
    f_stratum: np.ndarray
    m_levels: np.ndarray | None = None
    f_levels: np.ndarray | None = None

    def take(self, idx: np.ndarray) -> CellPairs:
        """The pairs at ``idx``, repeats included."""
        return replace(self, m=self.m[idx], f=self.f[idx], m_stratum=self.m_stratum[idx], f_stratum=self.f_stratum[idx])

    def with_levels(self, k_m: int | None, k_f: int | None) -> CellPairs:
        """Record the levels shown per stratum for each discrete side (``k`` levels; ``None`` = continuous)."""
        return replace(
            self,
            m_levels=None if k_m is None else shown_levels(self.m, self.m_stratum, k_m),
            f_levels=None if k_f is None else shown_levels(self.f, self.f_stratum, k_f),
        )

    @property
    def n_strata(self) -> tuple[int, int]:
        """Dense stratum-code counts of the mother and father sides."""
        return int(self.m_stratum.max(initial=-1)) + 1, int(self.f_stratum.max(initial=-1)) + 1

    @cached_property
    def rank_orders(self) -> tuple[np.ndarray, np.ndarray]:
        """Sorting permutations of ``m`` and ``f``, computed once so every weighted Spearman draw reuses them."""
        return np.argsort(self.m), np.argsort(self.f)


def pooled(pairs: CellPairs) -> CellPairs:
    """The same pairs as one stratum, for a crude estimator."""
    zeros = np.zeros_like(pairs.m_stratum)
    return CellPairs(pairs.m, pairs.f, zeros, zeros, _pooled_levels(pairs.m_levels), _pooled_levels(pairs.f_levels))


def _pooled_levels(levels: np.ndarray | None) -> np.ndarray | None:
    return None if levels is None else _pooled_shown(levels)


def _pooled_shown(levels: np.ndarray) -> np.ndarray:
    """The levels shown in any stratum, as the one stratum of the pooled pairs: ``(1, k)``."""
    return levels.any(axis=0, keepdims=True)


def unit_weights(n: int) -> np.ndarray:
    """The weights of the observed sample: every pair once."""
    return np.ones(n)


def _weights(w: np.ndarray | None, n: int) -> np.ndarray:
    return unit_weights(n) if w is None else w


#: Why a kernel found no estimate on a sample (``kernels.STATUS_*``), as the records say it.
STATUS_REASONS = {
    kernels.STATUS_CONSTANT_MARGIN: "constant_margin",
    kernels.STATUS_EMPTY_CATEGORY: "empty_category",
    kernels.STATUS_DEGENERATE_STRATUM: "degenerate_stratum",
    kernels.STATUS_NO_PAIRS: "no_complete_pairs",
}


def pearson(m: np.ndarray, f: np.ndarray, w: np.ndarray | None = None) -> Estimate:
    """Pearson correlation of mother and father values, one observation per Mating Pair (``w`` times each)."""
    status, r = kernels.pearson(m, f, _weights(w, m.size))
    return Undefined(STATUS_REASONS[status]) if status else r


def _spearman(pairs: CellPairs, w: np.ndarray) -> Estimate:
    order_m, order_f = pairs.rank_orders
    return pearson(kernels.weighted_ranks(order_m, pairs.m, w), kernels.weighted_ranks(order_f, pairs.f, w), w)


def spearman(m: np.ndarray, f: np.ndarray, w: np.ndarray | None = None) -> Estimate:
    """Spearman rank correlation (average ranks for ties) of the multiset with pair ``i`` taken ``w[i]`` times."""
    zeros = np.zeros(m.size, dtype=np.int64)
    return _spearman(CellPairs(m, f, zeros, zeros), _weights(w, m.size))


def degenerate_strata(x: np.ndarray, code: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Per stratum code, whether the stratum is present in ``x`` and constant there (a single value is constant)."""
    n_codes = int(code.max(initial=-1)) + 1
    _, _, _, lo, hi = kernels.stratum_moments(x, code, _weights(w, x.size), n_codes)
    return lo == hi


def standardise(x: np.ndarray, code: np.ndarray, w: np.ndarray | None = None) -> np.ndarray | Undefined:
    """``x`` centred and scaled to unit SD (``ddof=0``) within each stratum of ``code``.

    Each entry is one Mating Pair, so the moments are pair-weighted. A
    degenerate stratum makes the result undefined, never merged into another.
    """
    moments = _standardisation(x, code, _weights(w, x.size))
    if isinstance(moments, Undefined):
        return moments
    mean, inv_sd = moments
    return (x - mean[code]) * inv_sd[code]


def _standardisation(x: np.ndarray, code: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray] | Undefined:
    """Per-stratum mean and ``1/SD`` of ``x``, or why the strata cannot be standardised."""
    n_codes = int(code.max(initial=-1)) + 1
    total, mean, var, lo, hi = kernels.stratum_moments(x, code, w, n_codes)
    if (lo == hi).any():
        return Undefined("degenerate_stratum")
    with np.errstate(divide="ignore"):
        inv_sd = np.where(total > 0, 1 / np.sqrt(var), 0.0)
    return mean, inv_sd


def stratified_pearson(pairs: CellPairs, w: np.ndarray | None = None) -> Estimate:
    """Pearson of mother and father values, each standardised within their own sex x stratum."""
    w = _weights(w, pairs.m.size)
    if pairs.m.size == 0 or not w.any():
        return Undefined("no_complete_pairs")
    zm = standardise(pairs.m, pairs.m_stratum, w)
    if isinstance(zm, Undefined):
        return zm
    zf = standardise(pairs.f, pairs.f_stratum, w)
    if isinstance(zf, Undefined):
        return zf
    return pearson(zm, zf, w)


# Latent (liability) correlations: two-step ML after Olsson (1979) and Olsson, Drasgow & Dorans (1982).
#: ρ is searched on (-LATENT_BOUND, LATENT_BOUND).
LATENT_BOUND = kernels.LATENT_BOUND
#: A fit within this distance of a bound is reported as ``boundary: true`` (decision 9).
BOUNDARY_MARGIN = 1e-3
#: So is a fit whose NLL the nearer bound matches to this relative tolerance: the likelihood
#: plateau reaches the bound and the optimiser's stopping point on it is arbitrary.
BOUNDARY_NLL_TOL = 1e-6
#: Newton stops when a step is smaller than this; it gives up after this many steps.
NEWTON_TOL = 1e-8
NEWTON_MAX_ITER = 12
#: Observed polyserial fits start at the eq 38 ad hoc estimate, clipped to this.
START_CLIP = 0.99
_TINY = 1e-300
#: How every latent fit ended: Newton converged, or fell back to Brent because the Hessian was not
#: positive, a step left the bracket, or the iterations ran out. Read by the CLI's debug log.
FIT_OUTCOMES: Counter[str] = Counter()


def _owens_t(h: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Owen's ``T(h, a)``, with ``T(h, ±inf) = ±Φ(-|h|) / 2``."""
    out = np.empty(h.shape)
    inf = np.isinf(a)
    out[inf] = np.sign(a[inf]) * ndtr(-np.abs(h[inf])) / 2
    out[~inf] = owens_t(h[~inf], a[~inf])
    return out


def _bvn_cdf_finite(h: np.ndarray, k: np.ndarray, rho: float) -> np.ndarray:
    """Φ2 at finite ``h``, ``k`` from Owen's T (Owen 1956, Ann. Math. Statist. 27:1075-1090, doi:10.1214/aoms/1177728074).

    ``Φ2(h, k, ρ) = (Φ(h) + Φ(k)) / 2 - T(h, a_h) - T(k, a_k) - β`` with
    ``a_h = (k - ρh) / (h√(1-ρ²))``, ``a_k`` by symmetry, and ``β = 1/2`` when
    ``hk < 0`` or ``hk = 0`` with ``h + k < 0``, else 0. At ``h = 0`` the limit of
    ``a_h`` is ``±inf`` (sign of ``k``), or ``-ρ/√(1-ρ²)`` when ``k = 0`` too, in
    which case ``a_k`` tends to ``+inf``. Agrees with ``scipy.stats.multivariate_normal.cdf``
    to 2e-15 on the grid in ``test_bvn_cdf_matches_scipy``.
    """
    s = math.sqrt((1 - rho) * (1 + rho))
    with np.errstate(divide="ignore", invalid="ignore"):
        a_h = np.where(h == 0, np.where(k == 0, -rho / s, np.sign(k) * np.inf), (k - rho * h) / (h * s))
        a_k = np.where(k == 0, np.where(h == 0, np.inf, np.sign(h) * np.inf), (h - rho * k) / (k * s))
    hk = h * k
    beta = np.where((hk < 0) | ((hk == 0) & (h + k < 0)), 0.5, 0.0)
    return 0.5 * (ndtr(h) + ndtr(k)) - _owens_t(h, a_h) - _owens_t(k, a_k) - beta


def bvn_cdf(h: np.ndarray, k: np.ndarray, rho: float) -> np.ndarray:
    """Standard bivariate normal lower-orthant CDF ``P(ξ < h, η < k)`` at correlation ``rho``, elementwise.

    Olsson (1979) eq 1. ``h`` and ``k`` broadcast and may hold ±inf; finite
    pairs go through ``_bvn_cdf_finite``. This is the one seam for Φ2.
    """
    h, k = np.broadcast_arrays(np.asarray(h, dtype=np.float64), np.asarray(k, dtype=np.float64))
    out = np.where(np.isposinf(h), ndtr(k), np.where(np.isposinf(k), ndtr(h), 0.0))
    finite = np.isfinite(h) & np.isfinite(k)
    if finite.any():
        out[finite] = _bvn_cdf_finite(h[finite], k[finite], rho)
    return out


def bvn_pdf_and_drho(h: np.ndarray, k: np.ndarray, rho: float) -> tuple[np.ndarray, np.ndarray]:
    """``φ2(h, k; ρ)`` and ``∂φ2/∂ρ``, elementwise, both 0 wherever an argument is infinite.

    ``∂φ2/∂ρ = φ2 · {hk(1−ρ²) − ρQ + ρ(1−ρ²)} / (1−ρ²)²`` with ``Q = h² − 2ρhk + k²``,
    the ρ-derivative of ``log φ2``. Olsson (1979) prints this in appendix A2 with the
    sign of ``ρQ`` and the factor of ``ρ(1−ρ²)`` wrong (checked against a finite
    difference in ``test_bvn_pdf_drho_matches_finite_difference``); the misprint
    reaches none of the paper's estimates.
    """
    finite = np.isfinite(h) & np.isfinite(k)
    hh, kk = np.where(finite, h, 0.0), np.where(finite, k, 0.0)
    q = (1 - rho) * (1 + rho)
    quad = hh * hh - 2 * rho * hh * kk + kk * kk
    pdf = np.exp(-quad / (2 * q)) / (2 * math.pi * math.sqrt(q))
    drho = pdf * (hh * kk * q - rho * quad + rho * q) / (q * q)
    return np.where(finite, pdf, 0.0), np.where(finite, drho, 0.0)


def thresholds(margin: np.ndarray) -> np.ndarray:
    """Per row of ``margin`` (strata × levels), ``Φ⁻¹`` of the cumulative proportions, framed by -inf and +inf.

    Olsson (1979) eqs 15-18; Olsson, Drasgow & Dorans (1982) eq 36 (F4). A row
    without counts gets all -inf and is never used.
    """
    return kernels.thresholds(np.asarray(margin, dtype=np.float64))


def _levels(pairs: CellPairs) -> tuple[np.ndarray, np.ndarray]:
    """Both sides' shown levels; a discrete × discrete cell always carries them (``with_levels``)."""
    assert pairs.m_levels is not None
    assert pairs.f_levels is not None
    return pairs.m_levels, pairs.f_levels


def count_table(pairs: CellPairs, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted pair counts by (mother stratum, father stratum, mother level, father level)."""
    m_levels, f_levels = _levels(pairs)
    n_m, n_f = pairs.n_strata
    return kernels.count_table(
        pairs.m_stratum,
        pairs.f_stratum,
        pairs.m,
        pairs.f,
        _weights(w, pairs.m.size),
        max(n_m, m_levels.shape[0]),
        max(n_f, f_levels.shape[0]),
        m_levels.shape[1],
        f_levels.shape[1],
    )


def _check_levels(margin: np.ndarray, levels: np.ndarray) -> Undefined | None:
    """Every level a stratum shows in the analysed sample must be in this margin too; pooled, two levels are needed.

    Levels are never merged inside a draw, so a missing one fails the draw.
    """
    populated = margin.sum(axis=1) > 0
    if (levels[populated] & (margin[populated] == 0)).any():
        return Undefined("empty_category")
    if (margin.sum(axis=0) > 0).sum() < 2:
        return Undefined("constant_margin")
    return None


def _maximise_rho(nll: Callable[[float], float]) -> Fit:
    """ρ̂ on (-LATENT_BOUND, LATENT_BOUND) by bounded Brent, with the ``boundary`` flag.

    The flag is set when ρ̂ is within ``BOUNDARY_MARGIN`` of a bound or when the
    nearer bound's NLL is within ``BOUNDARY_NLL_TOL`` (relative) of the optimum.
    """
    result = minimize_scalar(nll, bounds=(-LATENT_BOUND, LATENT_BOUND), method="bounded", options={"xatol": 1e-7})
    return _flag_boundary(nll, float(result.x), float(result.fun))


def _flag_boundary(nll: Callable[[float], float], rho: float, best: float) -> Fit:
    at_bound = abs(rho) >= LATENT_BOUND - BOUNDARY_MARGIN
    plateau = nll(math.copysign(LATENT_BOUND, rho)) - best <= BOUNDARY_NLL_TOL * max(1.0, abs(best))
    return Fit(rho, at_bound or plateau)


def _newton_rho(
    terms: Callable[[float], tuple[float, float, float]], nll: Callable[[float], float], start: float
) -> Fit:
    """ρ̂ by Newton on the analytic score from ``start``, with the same ``boundary`` flag as ``_maximise_rho``.

    ``terms(ρ)`` returns the NLL and its first two ρ-derivatives. Newton hands
    over to bounded Brent when the Hessian is not positive, when a step would
    leave (-LATENT_BOUND, LATENT_BOUND), or when ``NEWTON_MAX_ITER`` steps have
    not met ``NEWTON_TOL``, so every fit reaches the same optimum Brent would.
    """
    rho = min(max(start, -START_CLIP), START_CLIP)
    for _ in range(NEWTON_MAX_ITER):
        best, grad, hess = terms(rho)
        if not hess > 0:
            FIT_OUTCOMES["fallback_hessian"] += 1
            return _maximise_rho(nll)
        step = -grad / hess
        rho += step
        if not -LATENT_BOUND < rho < LATENT_BOUND:
            FIT_OUTCOMES["fallback_bracket"] += 1
            return _maximise_rho(nll)
        if abs(step) < NEWTON_TOL:
            FIT_OUTCOMES["newton"] += 1
            return _flag_boundary(nll, rho, best)
    FIT_OUTCOMES["fallback_iterations"] += 1
    return _maximise_rho(nll)


@dataclass(frozen=True)
class _Tables:
    """A polychoric cell's weighted count table with both sides' thresholds and the populated stratum combinations.

    ``counts`` is ``n[combos]`` (tables × mother levels × father levels) and
    ``h``/``k`` the matching corner grids (tables × (k_m+1) × (k_f+1)).
    """

    n: np.ndarray
    a: np.ndarray
    b: np.ndarray
    combos: np.ndarray
    counts: np.ndarray
    h: np.ndarray
    k: np.ndarray

    @property
    def observed(self) -> np.ndarray:
        return self.counts > 0


def _corner_differences(grid: np.ndarray) -> np.ndarray:
    """Cell masses from a corner grid: the two level axes are the last two."""
    return grid[..., 1:, 1:] - grid[..., :-1, 1:] - grid[..., 1:, :-1] + grid[..., :-1, :-1]


def _tables(pairs: CellPairs, w: np.ndarray | None) -> _Tables | Undefined:
    """The first step of the polychoric fit: weighted counts, per-stratum thresholds, and the corner grids."""
    m_levels, f_levels = _levels(pairs)
    n = count_table(pairs, w)
    if not n.any():
        return Undefined("no_complete_pairs")
    m_margin, f_margin = n.sum(axis=(1, 3)), n.sum(axis=(0, 2))
    for margin, levels in ((m_margin, m_levels), (f_margin, f_levels)):
        undefined = _check_levels(margin, levels)
        if undefined is not None:
            return undefined
    a, b = thresholds(m_margin), thresholds(f_margin)
    combos = np.argwhere(n.sum(axis=(2, 3)) > 0)
    h, k = np.broadcast_arrays(a[combos[:, 0]][:, :, None], b[combos[:, 1]][:, None, :])
    return _Tables(n, a, b, combos, n[combos[:, 0], combos[:, 1]], h, k)


def polychoric(pairs: CellPairs, w: np.ndarray | None = None, start: float | None = None) -> Estimate:
    """Two-step ML polychoric correlation of two binary or ordinal sides (tetrachoric when both are binary).

    Thresholds per sex × stratum from that stratum's pair-weighted margin
    (Olsson 1979 eqs 15-18), then one ρ maximising the log-likelihood (eq 3)
    with cell probabilities from the corner CDFs (eq 4), summed over every
    mother-stratum × father-stratum table, by Newton on the eq 9 score from
    ``start`` (0 when not given). Φ2 is evaluated on the corner grids only,
    never per pair.
    """
    tables = _tables(pairs, w)
    if isinstance(tables, Undefined):
        return tables
    h, k, observed = tables.h, tables.k, tables.observed
    weights = tables.counts[observed]

    def terms(rho: float) -> tuple[float, float, float]:
        pi = np.maximum(_corner_differences(bvn_cdf(h, k, rho))[observed], _TINY)
        pdf, drho = bvn_pdf_and_drho(h, k, rho)
        score = _corner_differences(pdf)[observed] / pi
        curvature = _corner_differences(drho)[observed] / pi - score * score
        return -float(np.dot(weights, np.log(pi))), -float(np.dot(weights, score)), -float(np.dot(weights, curvature))

    def nll(rho: float) -> float:
        pi = np.maximum(_corner_differences(bvn_cdf(h, k, rho))[observed], _TINY)
        return -float(np.dot(weights, np.log(pi)))

    return _newton_rho(terms, nll, 0.0 if start is None else start)


def _threshold_correction(a_rho_tau: np.ndarray, tau: np.ndarray, n_stratum: np.ndarray) -> np.ndarray:
    """``A_ρτ A_ττ⁻¹ ψ_τ`` per (stratum, level) for one discrete side: ``(n_strata, k)``.

    ``ψ_τj = 1[level ≤ j−1] − Φ(τ_j)`` and ``A_τjτj = −N_s φ(τ_j)``; a threshold
    at ±inf is not a parameter and contributes nothing.
    """
    k1 = tau.shape[1]
    finite = np.isfinite(tau[:, 1:-1])
    inner = np.where(finite, tau[:, 1:-1], 0.0)
    cdf = np.where(finite, ndtr(inner), 0.0)
    pdf = np.where(finite, np.exp(-0.5 * inner * inner) / math.sqrt(2 * math.pi), 1.0)
    levels = np.arange(k1 - 1)
    below = (levels[None, None, :] <= np.arange(1, k1 - 1)[None, :, None] - 1).astype(np.float64)
    psi = below - cdf[:, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(finite, -a_rho_tau[:, 1:-1] / (n_stratum[:, None] * pdf), 0.0)
    return np.einsum("sj,sjl->sl", scale, psi)


def _threshold_cross(
    d_cdf: np.ndarray, d_pdf: np.ndarray, pi: np.ndarray, score: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """``Σ_cells n ∂score/∂τ`` per table and threshold of the side on axis 1 of the grids: ``(tables, k+1)``.

    A cell in row ``r`` has upper threshold ``r+1`` and lower threshold ``r``;
    ``∂π/∂τ`` follows Olsson (1979) eq 10 with ``d_cdf = ∂Φ2/∂h`` (eq 12) and
    ``∂(∂π/∂ρ)/∂τ`` the same differences of ``d_pdf = ∂φ2/∂h``.
    """
    d_pi_upper = d_cdf[:, 1:, 1:] - d_cdf[:, 1:, :-1]
    d_pi_lower = d_cdf[:, :-1, :-1] - d_cdf[:, :-1, 1:]
    d_num_upper = d_pdf[:, 1:, 1:] - d_pdf[:, 1:, :-1]
    d_num_lower = d_pdf[:, :-1, :-1] - d_pdf[:, :-1, 1:]
    out = np.zeros((counts.shape[0], counts.shape[1] + 1))
    out[:, 1:] += (counts * (d_num_upper - score * d_pi_upper) / pi).sum(axis=2)
    out[:, :-1] += (counts * (d_num_lower - score * d_pi_lower) / pi).sum(axis=2)
    return out


def polychoric_influence(pairs: CellPairs, w: np.ndarray, rho: float) -> np.ndarray | Undefined:
    """Per-pair influence on the two-step polychoric ρ̂ at ``rho``, through the pair's cell in its stratum table.

    ``ψ_ρ`` is the eq 9 score of the pair's cell. The nuisance equations are the
    cumulative-margin thresholds (eqs 15-18); ``∂score/∂a`` uses eq 12 for
    ``∂Φ2/∂h`` and ``∂φ2/∂h = −φ2 (h − ρk)/(1−ρ²)``. Everything is computed on
    the corner grids and gathered per pair.
    """
    tables = _tables(pairs, w)
    if isinstance(tables, Undefined):
        return tables
    h, k, counts = tables.h, tables.k, tables.counts
    q = (1 - rho) * (1 + rho)
    pi = np.maximum(_corner_differences(bvn_cdf(h, k, rho)), _TINY)
    pdf, drho = bvn_pdf_and_drho(h, k, rho)
    score = _corner_differences(pdf) / pi
    a_rr = float(np.sum(counts * (_corner_differences(drho) / pi - score * score)))
    if not a_rr < 0:
        return Undefined("sandwich_undefined")
    finite_h, finite_k = np.isfinite(h), np.isfinite(k)
    hh, kk = np.where(finite_h, h, 0.0), np.where(finite_k, k, 0.0)
    phi_h = np.where(finite_h, np.exp(-0.5 * hh * hh) / math.sqrt(2 * math.pi), 0.0)
    phi_k = np.where(finite_k, np.exp(-0.5 * kk * kk) / math.sqrt(2 * math.pi), 0.0)
    d_cdf_h = phi_h * np.where(finite_k, ndtr((kk - rho * hh) / math.sqrt(q)), k > 0)
    d_cdf_k = phi_k * np.where(finite_h, ndtr((hh - rho * kk) / math.sqrt(q)), h > 0)
    d_pdf_h = -pdf * (hh - rho * kk) / q
    d_pdf_k = -pdf * (kk - rho * hh) / q
    swap = (0, 2, 1)
    cross_a = _threshold_cross(d_cdf_h, d_pdf_h, pi, score, counts)
    cross_b = _threshold_cross(
        d_cdf_k.transpose(swap),
        d_pdf_k.transpose(swap),
        pi.transpose(swap),
        score.transpose(swap),
        counts.transpose(swap),
    )
    a_rho_a = np.zeros(tables.a.shape)
    a_rho_b = np.zeros(tables.b.shape)
    np.add.at(a_rho_a, tables.combos[:, 0], cross_a)
    np.add.at(a_rho_b, tables.combos[:, 1], cross_b)
    correction_a = _threshold_correction(a_rho_a, tables.a, tables.n.sum(axis=(1, 2, 3)))
    correction_b = _threshold_correction(a_rho_b, tables.b, tables.n.sum(axis=(0, 2, 3)))
    influence = np.zeros(tables.n.shape)
    influence[tables.combos[:, 0], tables.combos[:, 1]] = score
    influence -= correction_a[:, None, :, None] + correction_b[None, :, None, :]
    influence /= -a_rr
    return kernels.gather_influence(influence, pairs.m_stratum, pairs.f_stratum, pairs.m, pairs.f, w)


@dataclass(frozen=True)
class _FirstStep:
    """The first step of a polyserial fit: ``x`` moments per stratum, ``y`` margin and thresholds per stratum."""

    mean: np.ndarray
    var: np.ndarray
    margin: np.ndarray
    tau: np.ndarray

    @property
    def inv_sd(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.where(self.var > 0, 1 / np.sqrt(self.var), 0.0)


def _first_step(
    x: np.ndarray, y: np.ndarray, x_stratum: np.ndarray, y_stratum: np.ndarray, y_levels: np.ndarray, w: np.ndarray
) -> _FirstStep | Undefined:
    n_strata, k = y_levels.shape
    total, mean, var, lo, hi = kernels.stratum_moments(x, x_stratum, w, int(x_stratum.max(initial=-1)) + 1)
    present = total > 0
    if not present.any():
        return Undefined("no_complete_pairs")
    if lo[present].min() == hi[present].max():
        return Undefined("constant_margin")
    if (lo == hi).any():
        return Undefined("degenerate_stratum")
    margin = kernels.margin(y, y_stratum, w, max(n_strata, int(y_stratum.max(initial=-1)) + 1), k)
    undefined = _check_levels(margin, y_levels)
    if undefined is not None:
        return undefined
    return _FirstStep(mean, var, margin, thresholds(margin))


def polyserial(
    x: np.ndarray,
    y: np.ndarray,
    x_stratum: np.ndarray,
    y_stratum: np.ndarray,
    y_levels: np.ndarray,
    w: np.ndarray | None = None,
    start: float | None = None,
) -> Estimate:
    """Two-step ML polyserial correlation of a continuous ``x`` and a binary or ordinal ``y`` (biserial when binary).

    ``x`` is standardised within its stratum with the 1/N variance and ``y``
    gets thresholds per stratum from its cumulative proportions (Olsson, Drasgow
    & Dorans 1982 eq 36); ρ then maximises the conditional term of eq 20 with
    the eq 19 probabilities, by Newton on the eq 26 score from ``start``, or
    from the eq 38 ad hoc estimate when no start is given.
    """
    w = _weights(w, x.size)
    first = _first_step(x, y, x_stratum, y_stratum, y_levels, w)
    if isinstance(first, Undefined):
        return first
    inv_sd = first.inv_sd
    args = (x, x_stratum, first.mean, inv_sd, y, y_stratum, first.tau, w)
    if start is None:
        start = _ad_hoc_polyserial((x - first.mean[x_stratum]) * inv_sd[x_stratum], y, w, first.margin)
    return _newton_rho(
        lambda rho: kernels.polyserial_terms(rho, *args), lambda rho: kernels.polyserial_nll(rho, *args), start
    )


def polyserial_influence(
    x: np.ndarray,
    y: np.ndarray,
    x_stratum: np.ndarray,
    y_stratum: np.ndarray,
    y_levels: np.ndarray,
    w: np.ndarray,
    rho: float,
) -> np.ndarray | Undefined:
    """Per-pair influence on the two-step polyserial ρ̂ at ``rho`` (``kernels.polyserial_influence``)."""
    first = _first_step(x, y, x_stratum, y_stratum, y_levels, w)
    if isinstance(first, Undefined):
        return first
    out, a_rr = kernels.polyserial_influence(rho, x, x_stratum, first.mean, first.var, y, y_stratum, first.tau, w)
    return out if a_rr < 0 else Undefined("sandwich_undefined")


def _ad_hoc_polyserial(z: np.ndarray, y: np.ndarray, w: np.ndarray, margin: np.ndarray) -> float:
    """Olsson, Drasgow & Dorans (1982) eq 38: ``r_xy s_y / Σ φ(τ̂_j)`` with pooled thresholds, as a Newton start."""
    _, r_xy = kernels.pearson(z, y, w)
    _, _, var, _, _ = kernels.stratum_moments(y, np.zeros(y.size, dtype=np.int64), w, 1)
    tau = thresholds(margin.sum(axis=0, keepdims=True))[0, 1:-1]
    return float(r_xy * math.sqrt(var[0]) / np.sum(np.exp(-0.5 * tau * tau) / math.sqrt(2 * math.pi)))


def _polyserial_mother_continuous(pairs: CellPairs, w: np.ndarray, start: float | None) -> Estimate:
    assert pairs.f_levels is not None
    return polyserial(pairs.m, pairs.f, pairs.m_stratum, pairs.f_stratum, pairs.f_levels, w, start)


def _polyserial_father_continuous(pairs: CellPairs, w: np.ndarray, start: float | None) -> Estimate:
    assert pairs.m_levels is not None
    return polyserial(pairs.f, pairs.m, pairs.f_stratum, pairs.m_stratum, pairs.m_levels, w, start)


def _polyserial_mother_continuous_influence(pairs: CellPairs, w: np.ndarray, rho: float) -> np.ndarray | Undefined:
    assert pairs.f_levels is not None
    return polyserial_influence(pairs.m, pairs.f, pairs.m_stratum, pairs.f_stratum, pairs.f_levels, w, rho)


def _polyserial_father_continuous_influence(pairs: CellPairs, w: np.ndarray, rho: float) -> np.ndarray | Undefined:
    assert pairs.m_levels is not None
    return polyserial_influence(pairs.f, pairs.m, pairs.f_stratum, pairs.m_stratum, pairs.m_levels, w, rho)


def table_2x2(pairs: CellPairs) -> list[list[int]]:
    """Pair counts of a binary × binary cell, rows in mother level order, columns in father level order."""
    return count_table(pooled(pairs))[0, 0].astype(np.int64).tolist()


def odds_ratio(pairs: CellPairs, w: np.ndarray | None = None) -> Estimate:
    """Cross-product ratio ``ad / (bc)`` of the 2×2 table; ``inf`` when ``bc = 0``.

    ``0 / 0`` needs a constant margin, which is undefined first.
    """
    table = count_table(pooled(pairs), w)[0, 0]
    if not table.any():
        return Undefined("no_complete_pairs")
    if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
        return Undefined("constant_margin")
    (a, b), (c, d) = table
    with np.errstate(divide="ignore"):
        return float(a * d / (b * c))


def odds_ratio_influence(pairs: CellPairs, w: np.ndarray, value: float) -> np.ndarray | Undefined:
    """Per-pair influence on ``log OR``: ``+1/n`` of its cell on the diagonal, ``−1/n`` off it (Woolf when i.i.d.).

    An odds ratio of 0 or ``inf`` (a zero cell) has an infinite log and no sandwich.
    """
    if not (math.isfinite(value) and value > 0):
        return Undefined("infinite_odds_ratio")
    table = count_table(pooled(pairs), w)[0, 0]
    sign = np.array([[1.0, -1.0], [-1.0, 1.0]])
    out = (sign / table)[pairs.m.astype(np.int64), pairs.f.astype(np.int64)]
    out[w == 0] = 0.0
    return out


def pearson_influence(pairs: CellPairs, w: np.ndarray, _value: float) -> np.ndarray:
    """Per-pair influence on the Pearson r of both sides standardised within their own strata (``kernels.pearson_influence``)."""
    n_m, n_f = pairs.n_strata
    _, mean_m, var_m, _, _ = kernels.stratum_moments(pairs.m, pairs.m_stratum, w, n_m)
    _, mean_f, var_f, _, _ = kernels.stratum_moments(pairs.f, pairs.f_stratum, w, n_f)
    return kernels.pearson_influence(
        pairs.m, pairs.f, pairs.m_stratum, pairs.f_stratum, mean_m, var_m, mean_f, var_f, w
    )


def _crude_pearson_influence(pairs: CellPairs, w: np.ndarray, value: float) -> np.ndarray:
    return pearson_influence(pooled(pairs), w, value)


#: An estimator sees the whole cell, the frequency weight of each pair (a bootstrap draw), and a start
#: for its optimiser (the observed estimate on a refit; ``None`` on the observed fit).
EstimatorFn = Callable[[CellPairs, np.ndarray, float | None], Estimate]
#: One cell kind's bootstrap: ``(values, status)``, each ``(n_draws, n_fitted)``, over the fitted
#: estimators in order (crude ones, then the stratified primary when fitted), from the observed
#: values (``starts``; NaN for an undefined estimate) and the pair weights of draws ``0..n_draws-1``
#: of ``seed`` over the Mate Network ``labels``. ``status`` is a ``kernels.STATUS_*`` code per draw.
DrawsFn = Callable[[CellPairs, np.ndarray, int, int, list[float]], tuple[np.ndarray, np.ndarray]]
#: The per-pair influence on a defined estimate (its value as the third argument): pair ``i``'s
#: contribution to ``estimate − truth``, on the ``ci_scale`` of its estimator. ``Undefined`` when
#: the sandwich does not exist for this estimate.
InfluenceFn = Callable[[CellPairs, np.ndarray, float], np.ndarray | Undefined]
CiScale = Literal["fisher_z", "log"]


@dataclass(frozen=True)
class Estimator:
    """A named cell estimator, the output key its value is written under, and its influence function.

    ``fn`` sees the whole cell, strata included, and a weight per pair: the
    observed fit, the oracles, and the full refit of a bootstrap draw whose
    one-step fails all call it. ``influence`` (``None``: no sandwich SE, as
    for Spearman) gives the per-pair influence the cluster-robust SE is built
    from; ``ci_scale`` is the scale of that SE and of the Wald interval.
    """

    name: str
    fn: EstimatorFn
    key: str
    influence: InfluenceFn | None = None
    ci_scale: CiScale = "fisher_z"


PEARSON = Estimator("pearson", lambda p, w, _: pearson(p.m, p.f, w), "r", _crude_pearson_influence)
SPEARMAN = Estimator("spearman", lambda p, w, _: _spearman(p, w), "r")
STRATIFIED_PEARSON = Estimator("pearson", lambda p, w, _: stratified_pearson(p, w), "r", pearson_influence)
#: Pearson with the binary side(s) coded 0/1.
PHI = Estimator("phi", lambda p, w, _: pearson(p.m, p.f, w), "r", _crude_pearson_influence)
POINT_BISERIAL = Estimator("point_biserial", lambda p, w, _: pearson(p.m, p.f, w), "r", _crude_pearson_influence)
ODDS_RATIO = Estimator("odds_ratio", lambda p, w, _: odds_ratio(p, w), "value", odds_ratio_influence, "log")


@dataclass(frozen=True)
class CellEstimators:
    """A cell's crude estimators, primary first, the stratified form of its primary estimator, its bootstrap and its table.

    The primary estimator is the one that gets a permutation p-value.
    """

    crude: tuple[Estimator, ...]
    stratified: Estimator
    draws: DrawsFn
    table: Callable[[CellPairs], list[list[int]]] | None = None

    def fitted(self, stratified: bool) -> tuple[Estimator, ...]:
        """The estimators a cell fits: the crude ones, plus the stratified primary under stratification."""
        return (*self.crude, self.stratified) if stratified else self.crude


# ---------------------------------------------------------------------------
# Sample structure and resampling
# ---------------------------------------------------------------------------


def mate_networks(mother_rows: np.ndarray, father_rows: np.ndarray) -> np.ndarray:
    """Mate Network label of each Mating Pair, numbered in order of each network's first pair.

    ``mother_rows``/``father_rows`` are pedigree rows, used directly as graph
    nodes; rows outside any pair are isolated nodes and never labelled.
    """
    n_pairs = len(mother_rows)
    if n_pairs == 0:
        return np.zeros(0, dtype=np.int64)
    n_nodes = int(max(mother_rows.max(), father_rows.max())) + 1
    graph = sp.csr_matrix((np.ones(n_pairs, dtype=np.int8), (mother_rows, father_rows)), shape=(n_nodes, n_nodes))
    n_components, component = connected_components(graph, directed=False)
    pair_component = component[mother_rows]
    first_pair = np.full(n_components, n_pairs)
    np.minimum.at(first_pair, pair_component, np.arange(n_pairs))
    opens_network = np.zeros(n_pairs, dtype=bool)
    opens_network[first_pair[first_pair < n_pairs]] = True
    return (np.cumsum(opens_network) - 1)[first_pair[pair_component]]


def network_weights(labels: np.ndarray, seed: int, draw: int) -> np.ndarray:
    """Pair weights of bootstrap draw ``draw`` of ``seed``: how often each pair's Mate Network was drawn, with replacement.

    The same weights the parallel draw kernels generate in place (``kernels.network_weights``).
    """
    w = np.empty(labels.size)
    kernels.network_weights(w, np.empty(int(labels.max(initial=-1)) + 1, dtype=np.int32), labels, seed, draw)
    return w


def bootstrap_networks(labels: np.ndarray, n_draws: int, seed: int) -> Iterator[np.ndarray]:
    """Yield the pair weights of draws ``0..n_draws-1`` of ``seed`` (``network_weights``)."""
    for d in range(n_draws):
        yield network_weights(labels, seed, d)


def _draw_scratch(labels: np.ndarray, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-worker scratch for the draw kernels: network multiplicities ``(W, G)`` and pair weights ``(W, n)``."""
    n_workers = min(numba.get_num_threads(), n_draws)
    return np.empty((n_workers, int(labels.max()) + 1), dtype=np.int32), np.empty((n_workers, labels.size))


def _continuous_draws(
    pairs: CellPairs, labels: np.ndarray, n_draws: int, seed: int, starts: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Pearson, Spearman and stratified Pearson on every weighted draw (``kernels.continuous_draws``)."""
    mult, w = _draw_scratch(labels, n_draws)
    order_m, order_f = pairs.rank_orders
    n_m, n_f = pairs.n_strata
    # The Spearman pass streams sorted copies: with one random access per pair instead of six it stays
    # linear once the cell outgrows the cache (measured 24x from 1e5 to 1e6 pairs before).
    inv_f = np.empty_like(order_f)
    inv_f[order_f] = np.arange(order_f.size)
    values, status = kernels.continuous_draws(
        seed,
        n_draws,
        labels,
        mult,
        w,
        np.empty_like(w),
        np.empty_like(w),
        pairs.m,
        pairs.f,
        pairs.m_stratum,
        pairs.f_stratum,
        n_m,
        n_f,
        pairs.m[order_m],
        pairs.f[order_f],
        labels[order_m],
        labels[order_f],
        inv_f[order_m],
        len(starts) == 3,
    )
    return values[:, : len(starts)], status[:, : len(starts)]


def _polyserial_hessian(
    rho: float, x: np.ndarray, y: np.ndarray, x_stratum: np.ndarray, y_stratum: np.ndarray, levels: np.ndarray
) -> float:
    """The full-sample ``d²NLL/dρ²`` of a polyserial form at ``rho`` (NaN when ``rho`` is)."""
    if math.isnan(rho):
        return math.nan
    first = _first_step(x, y, x_stratum, y_stratum, levels, unit_weights(x.size))
    assert not isinstance(first, Undefined)
    return kernels.polyserial_terms(
        rho, x, x_stratum, first.mean, first.inv_sd, y, y_stratum, first.tau, unit_weights(x.size)
    )[2]


def _polyserial_draws(x_is_mother: bool, point_biserial: bool) -> DrawsFn:
    """One-step polyserial draws (crude and stratified) plus the exact point-biserial when the discrete side is binary."""
    columns = [0, 1, 2] if point_biserial else [0, 2]

    def draws(
        pairs: CellPairs, labels: np.ndarray, n_draws: int, seed: int, starts: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        mult, w = _draw_scratch(labels, n_draws)
        if x_is_mother:
            x, y, x_stratum, y_stratum, levels = pairs.m, pairs.f, pairs.m_stratum, pairs.f_stratum, pairs.f_levels
        else:
            x, y, x_stratum, y_stratum, levels = pairs.f, pairs.m, pairs.f_stratum, pairs.m_stratum, pairs.m_levels
        assert levels is not None
        one_stratum = np.zeros_like(x_stratum)
        rho_crude, rho_strat = starts[0], starts[-1] if len(starts) == len(columns) else math.nan
        values, status = kernels.polyserial_draws(
            seed,
            n_draws,
            labels,
            mult,
            w,
            x,
            x_stratum,
            int(x_stratum.max(initial=-1)) + 1,
            y,
            y_stratum,
            levels,
            rho_crude,
            _polyserial_hessian(rho_crude, x, y, one_stratum, one_stratum, _pooled_shown(levels)),
            rho_strat,
            _polyserial_hessian(rho_strat, x, y, x_stratum, y_stratum, levels),
            point_biserial,
        )
        used = columns[: len(starts)]
        return values[:, used], status[:, used]

    return draws


def _levels_status(margin: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """``_check_levels`` per draw of ``margin`` ``(draws, strata, k)``: a status code per draw."""
    populated = margin.sum(axis=2) > 0
    empty = ((levels[None] & (margin == 0)) & populated[:, :, None]).any(axis=(1, 2))
    n_levels = (margin.sum(axis=1) > 0).sum(axis=1)
    return np.where(empty, kernels.STATUS_EMPTY_CATEGORY, np.where(n_levels < 2, kernels.STATUS_CONSTANT_MARGIN, 0))


#: Corner-grid elements a chunk of polychoric draws is evaluated over at once (bounds the temporaries).
_GRID_CHUNK = 2_000_000


def _polychoric_terms(
    tables: np.ndarray, m_levels: np.ndarray, f_levels: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per count table ``(tables, m strata, f strata, k_m, k_f)``: the NLL's ``(grad, hess, status)`` in ρ at ``rho``.

    Thresholds are refit per table from its margins (the first step of
    ``polychoric``), and the eq 9 score and its ρ-derivative are summed over
    the table. Φ2 is evaluated on the corner grids of a chunk of tables at once.
    """
    n_draws, n_m, n_f, k_m, k_f = tables.shape
    m_margin, f_margin = tables.sum(axis=(2, 4)), tables.sum(axis=(1, 3))
    status = np.where(tables.reshape(n_draws, -1).any(axis=1), 0, kernels.STATUS_NO_PAIRS)
    for side_margin, levels in ((m_margin, m_levels), (f_margin, f_levels)):
        side = _levels_status(side_margin, levels)
        status = np.where((status == 0) & (side != 0), side, status)
    a = thresholds(m_margin.reshape(-1, k_m)).reshape(n_draws, n_m, k_m + 1)
    b = thresholds(f_margin.reshape(-1, k_f)).reshape(n_draws, n_f, k_f + 1)
    grad, hess = np.full(n_draws, math.nan), np.full(n_draws, math.nan)
    chunk = max(1, _GRID_CHUNK // (n_m * n_f * (k_m + 1) * (k_f + 1)))
    for lo in range(0, n_draws, chunk):
        part = slice(lo, lo + chunk)
        h, k = a[part][:, :, None, :, None], b[part][:, None, :, None, :]
        n = tables[part]
        pi = np.maximum(_corner_differences(bvn_cdf(h, k, rho)), _TINY)
        pdf, drho = bvn_pdf_and_drho(h, k, rho)
        counted = n > 0
        score = np.where(counted, _corner_differences(pdf) / pi, 0.0)
        curvature = np.where(counted, _corner_differences(drho) / pi - score * score, 0.0)
        axes = (1, 2, 3, 4)
        grad[part], hess[part] = -(n * score).sum(axis=axes), -(n * curvature).sum(axis=axes)
    return grad, hess, status


def _polychoric_steps(
    tables: np.ndarray, observed: np.ndarray, m_levels: np.ndarray, f_levels: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """One-step polychoric draws ``(values, status)`` from per-draw count tables and the ``observed`` table.

    ``ρ_d = ρ̂ − S_w(ρ̂) / H(ρ̂)`` with the draw's score and the full-sample
    Hessian, clipped to the search interval; a Hessian that is not positive
    (a boundary fit) marks every draw ``STATUS_NONCONCAVE`` for the full refit.
    """
    _, hess, _ = _polychoric_terms(observed[None], m_levels, f_levels, rho)
    grad, _, status = _polychoric_terms(tables, m_levels, f_levels, rho)
    if not hess[0] > 0:
        return np.full(tables.shape[0], math.nan), np.where(status == 0, kernels.STATUS_NONCONCAVE, status)
    return np.clip(rho - grad / hess[0], -LATENT_BOUND, LATENT_BOUND), status


def _two_by_two_draws(table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Odds ratio and phi of every draw's pooled 2×2 ``table`` ``(draws, 2, 2)``: ``(odds, phi, status)``.

    Both are undefined on the same draws: no pairs, or a zero row or column
    (a constant side).
    """
    a, b, c, d = table[:, 0, 0], table[:, 0, 1], table[:, 1, 0], table[:, 1, 1]
    rows, cols = table.sum(axis=2), table.sum(axis=1)
    constant = (rows == 0).any(axis=1) | (cols == 0).any(axis=1)
    status = np.where(
        table.sum(axis=(1, 2)) == 0, kernels.STATUS_NO_PAIRS, np.where(constant, kernels.STATUS_CONSTANT_MARGIN, 0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        odds = a * d / (b * c)
        phi = np.clip((a * d - b * c) / np.sqrt(rows.prod(axis=1) * cols.prod(axis=1)), -1.0, 1.0)
    return odds, phi, status


#: Count-table elements of one chunk of bootstrap draws: bounds the draws' tables whatever the thread count.
_TABLE_CHUNK = 2_000_000


def _table_draws(two_by_two: bool) -> DrawsFn:
    """One-step polychoric draws (crude and stratified) from per-draw count tables, plus the exact odds ratio and phi of a 2×2 cell."""

    def draws(
        pairs: CellPairs, labels: np.ndarray, n_draws: int, seed: int, starts: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        mult, w = _draw_scratch(labels, n_draws)
        m_levels, f_levels = _levels(pairs)
        n_m, n_f = pairs.n_strata
        shape = (max(n_m, m_levels.shape[0]), max(n_f, f_levels.shape[0]), m_levels.shape[1], f_levels.shape[1])
        observed = count_table(pairs)
        stratified = len(starts) == 1 + 2 * two_by_two + 1
        values = np.full((n_draws, len(starts)), math.nan)
        status = np.zeros((n_draws, len(starts)), dtype=np.int64)
        chunk = max(1, _TABLE_CHUNK // math.prod(shape))
        for lo in range(0, n_draws, chunk):
            part = slice(lo, min(lo + chunk, n_draws))
            tables = kernels.table_draws(
                seed, part.start, part.stop, labels, mult, w, pairs.m_stratum, pairs.f_stratum, pairs.m, pairs.f, *shape
            )
            pooled_tables = tables.sum(axis=(1, 2), keepdims=True)
            values[part, 0], status[part, 0] = _polychoric_steps(
                pooled_tables,
                observed.sum(axis=(0, 1), keepdims=True),
                _pooled_shown(m_levels),
                _pooled_shown(f_levels),
                starts[0],
            )
            if two_by_two:
                values[part, 1], values[part, 2], status[part, 1] = _two_by_two_draws(pooled_tables[:, 0, 0])
                status[part, 2] = status[part, 1]
            if stratified:
                values[part, -1], status[part, -1] = _polychoric_steps(tables, observed, m_levels, f_levels, starts[-1])
        return values, status

    return draws


def _latent_cell(
    name: str, fn: EstimatorFn, influence: InfluenceFn, draws: DrawsFn, *secondary: Estimator, **kw
) -> CellEstimators:
    """A cell whose primary is the latent correlation ``fn``: crude on the pooled pairs, stratified as given."""
    crude = Estimator(name, lambda p, w, s: fn(pooled(p), w, s), "rho", lambda p, w, r: influence(pooled(p), w, r))
    return CellEstimators((crude, *secondary), Estimator(name, fn, "rho", influence), draws, **kw)


_MOTHER_CONTINUOUS = (_polyserial_mother_continuous, _polyserial_mother_continuous_influence)
_FATHER_CONTINUOUS = (_polyserial_father_continuous, _polyserial_father_continuous_influence)
_POLYCHORIC = (polychoric, polychoric_influence, _table_draws(two_by_two=False))

#: (mother trait kind, father trait kind) -> the cell's estimators. The Within-Person Cross-Trait
#: Correlation of (first kind, second kind) uses the same primary.
CELL_ESTIMATORS: dict[tuple[TraitKind, TraitKind], CellEstimators] = {
    ("continuous", "continuous"): CellEstimators((PEARSON, SPEARMAN), STRATIFIED_PEARSON, _continuous_draws),
    ("binary", "binary"): _latent_cell(
        "tetrachoric",
        polychoric,
        polychoric_influence,
        _table_draws(two_by_two=True),
        ODDS_RATIO,
        PHI,
        table=table_2x2,
    ),
    ("binary", "ordinal"): _latent_cell("polychoric", *_POLYCHORIC),
    ("ordinal", "binary"): _latent_cell("polychoric", *_POLYCHORIC),
    ("ordinal", "ordinal"): _latent_cell("polychoric", *_POLYCHORIC),
    ("continuous", "binary"): _latent_cell(
        "biserial", *_MOTHER_CONTINUOUS, _polyserial_draws(x_is_mother=True, point_biserial=True), POINT_BISERIAL
    ),
    ("binary", "continuous"): _latent_cell(
        "biserial", *_FATHER_CONTINUOUS, _polyserial_draws(x_is_mother=False, point_biserial=True), POINT_BISERIAL
    ),
    ("continuous", "ordinal"): _latent_cell(
        "polyserial", *_MOTHER_CONTINUOUS, _polyserial_draws(x_is_mother=True, point_biserial=False)
    ),
    ("ordinal", "continuous"): _latent_cell(
        "polyserial", *_FATHER_CONTINUOUS, _polyserial_draws(x_is_mother=False, point_biserial=False)
    ),
}


def _n_levels(trait: Trait) -> int | None:
    return None if trait.levels is None else len(trait.levels)


# ---------------------------------------------------------------------------
# Permutation blocks and strata
# ---------------------------------------------------------------------------


def permutation_blocks(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(by_block, block_start)``: the fathers listed block by block, and each block's offset in that list."""
    by_block = np.argsort(blocks, kind="stable")
    block_start = np.zeros(int(blocks.max(initial=-1)) + 2, dtype=np.int64)
    np.cumsum(np.bincount(blocks), out=block_start[1:])
    return by_block, block_start


def permutation_donors(blocks: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """Donor arrays ``(n_draws, n_fathers)``: in draw ``d``, distinct father ``i`` takes the trait vector of father ``donor[d, i]``.

    Each draw is a uniform permutation within every block of ``blocks`` (one
    label per distinct father), so a donor always shares the recipient's block.
    The same draws the permutation kernel generates in place (``kernels.shuffle_blocks``).
    """
    by_block, block_start = permutation_blocks(blocks)
    donors = np.empty((n_draws, len(blocks)), dtype=np.int64)
    for d in range(n_draws):
        kernels.shuffle_blocks(donors[d], by_block, block_start, seed, d)
    return donors


def father_blocks(stratum: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Permutation block of each distinct father: his stratum x which traits he has (``present``: traits x fathers).

    Blocks are numbered in (stratum, pattern) order; strata are known, so ``>= 0``.
    """
    pattern = (present * (1 << np.arange(len(present)))[:, None]).sum(axis=0)
    key = stratum << len(present) | pattern
    return np.unique(key, return_inverse=True)[1].ravel()


def strata(df: pl.DataFrame, by: StratifyBy, birth_year_bin: int) -> np.ndarray:
    """Stratum of each row: its Depth, or the first year of its birth-year bin; -1 when unknown."""
    if by == "depth":
        return df["ped_depth"].to_numpy().astype(np.int64)
    year = df["birth_year"].to_numpy().astype(np.int64)
    return np.where(year == -1, -1, year // birth_year_bin * birth_year_bin)


def stratum_networks(code: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per stratum code, how many distinct Mate Networks (``labels``) its pairs come from."""
    n_labels = int(labels.max(initial=-1)) + 1
    return np.bincount(unique_ints(code * n_labels + labels) // max(n_labels, 1))


def drop_thin_strata(
    pairs: CellPairs, mothers: np.ndarray, fathers: np.ndarray, min_networks: int
) -> tuple[np.ndarray, int, np.ndarray]:
    """The pairs kept once every small or degenerate sex x stratum is dropped: ``(keep, n_small, labels)``.

    ``keep`` indexes the kept pairs, ``n_small`` counts the pairs that left as
    small, and ``labels`` are the kept pairs' Mate Network labels.

    ``mothers``/``fathers`` are the rows of each pair's mates. A sex x stratum is
    small when its pairs come from fewer than ``min_networks`` Mate Networks of
    the pairs still kept, and degenerate when its values are constant. Dropping
    one stratum's pairs can make another small or degenerate, so both rules
    repeat until neither drops a pair. A pair counts as small when the small
    rule flagged it in the round it was dropped.
    """
    keep = np.arange(pairs.m.size)
    n_small = 0
    while True:
        labels = mate_networks(mothers[keep], fathers[keep])
        if not keep.size:
            break
        kept = pairs.take(keep)
        small = (stratum_networks(kept.m_stratum, labels)[kept.m_stratum] < min_networks) | (
            stratum_networks(kept.f_stratum, labels)[kept.f_stratum] < min_networks
        )
        bad = (
            small
            | degenerate_strata(kept.m, kept.m_stratum)[kept.m_stratum]
            | degenerate_strata(kept.f, kept.f_stratum)[kept.f_stratum]
        )
        if not bad.any():
            break
        n_small += int(small.sum())
        keep = keep[~bad]
    return keep, n_small, labels


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _failure_counts(draws: list[Estimate]) -> dict[str, int]:
    return dict(sorted(Counter(d.reason for d in draws if isinstance(d, Undefined)).items()))


def _valid(draws: list[Estimate]) -> np.ndarray:
    return np.array([estimate_value(d) for d in draws if not isinstance(d, Undefined)], dtype=np.float64)


def bootstrap_record(draws: list[Estimate], requested: int, n_networks: int) -> dict:
    """Percentile CI over the valid draws, with draw accounting and the reason a CI is withheld.

    The bounds are order statistics (``inverted_cdf``), so an odds ratio whose
    draws include ``inf`` gets an infinite bound, never NaN from interpolation.
    """
    valid = _valid(draws)
    failures = _failure_counts(draws)
    record = {
        "ci": None,
        "ci_method": None,
        "ci_unavailable_reason": None,
        "bootstrap": {
            "requested": requested,
            "valid": int(valid.size),
            "failed": len(draws) - int(valid.size),
            "failure_reasons": failures,
        },
    }
    if n_networks < 2:
        record["ci_unavailable_reason"] = "single_mate_network"
    elif valid.size < MIN_VALID_DRAW_SHARE * requested:
        record["ci_unavailable_reason"] = "too_many_failed_draws"
    else:
        tail = (1 - CI_LEVEL) / 2
        record["ci"] = [float(q) for q in np.quantile(valid, [tail, 1 - tail], method="inverted_cdf")]
        record["ci_method"] = "bootstrap"
    return record


def cluster_se(influence: np.ndarray, labels: np.ndarray) -> float:
    """``√(G/(G−1) · Σ_g (Σ_{i∈g} IF_i)²)`` over the ``G`` clusters of ``labels`` (dense ``0..G-1``).

    ``influence`` already carries the ``A⁻¹`` of the sandwich, so no further
    division by ``N``; with every pair its own cluster this is the i.i.d.
    sandwich times ``√(G/(G−1))``.
    """
    sums = np.bincount(labels, weights=influence)
    g = sums.size
    return math.sqrt(g / (g - 1) * float(np.dot(sums, sums)))


def sandwich_se(estimator: Estimator, pairs: CellPairs, estimate: float | Fit, labels: np.ndarray) -> float | Undefined:
    """The Mate-Network cluster-robust two-step sandwich SE of a defined estimate, or why there is none.

    Reasons: ``single_mate_network`` (``G < 2``), ``bootstrap_not_requested``
    (an estimator without an influence function), ``boundary`` (a latent fit at
    a bound, or a correlation of exactly ±1, where the sandwich is not valid),
    ``infinite_odds_ratio``, and ``sandwich_undefined`` (the ρ equation is not
    locally concave at ρ̂).
    """
    if int(labels.max(initial=-1)) + 1 < 2:
        return Undefined("single_mate_network")
    if estimator.influence is None:
        return Undefined("bootstrap_not_requested")
    value = estimate_value(estimate)
    if (isinstance(estimate, Fit) and estimate.boundary) or (estimator.ci_scale == "fisher_z" and abs(value) >= 1):
        return Undefined("boundary")
    influence = estimator.influence(pairs, unit_weights(pairs.m.size), value)
    if isinstance(influence, Undefined):
        return influence
    se = cluster_se(influence, labels)
    return se if math.isfinite(se) else Undefined("sandwich_undefined")


def wald_ci(value: float, se: float, scale: CiScale) -> list[float]:
    """The ``CI_LEVEL`` Wald interval on the Fisher-z scale (``tanh(atanh(v) ± z·se/(1−v²))``) or the log scale."""
    z = float(ndtri((1 + CI_LEVEL) / 2))
    if scale == "log":
        return [math.exp(math.log(value) - z * se), math.exp(math.log(value) + z * se)]
    centre, half = math.atanh(value), z * se / ((1 - value) * (1 + value))
    return [math.tanh(centre - half), math.tanh(centre + half)]


def inference_record(
    estimator: Estimator,
    estimate: float | Fit,
    se: float | Undefined,
    draws: list[Estimate],
    bootstrap: int,
    n_networks: int,
) -> dict:
    """``se``, ``ci``, ``ci_method`` and ``ci_unavailable_reason`` of one defined estimate.

    Without a bootstrap the CI is the Wald interval from the sandwich SE; with
    one, the percentile bootstrap CI (``bootstrap_record``) and the sandwich SE
    is still reported.
    """
    record: dict = {"se": se if isinstance(se, float) else None}
    if bootstrap > 0:
        return record | bootstrap_record(draws, bootstrap, n_networks)
    if isinstance(se, Undefined):
        return record | {"ci": None, "ci_method": None, "ci_unavailable_reason": se.reason}
    return record | {
        "ci": wald_ci(estimate_value(estimate), se, estimator.ci_scale),
        "ci_method": "sandwich",
        "ci_unavailable_reason": None,
    }


def _extreme(values: np.ndarray, observed: float) -> np.ndarray:
    """Which null statistics are at least as extreme as ``observed`` by magnitude, ties (to 1e-12 relative) included."""
    return np.abs(values) >= abs(observed) - 1e-12 * max(1.0, abs(observed))


def _draws_used(observed: float, draws: list[Estimate], h: int) -> int:
    """The draws through the ``h``-th valid exceedance of ``observed``, or all of them."""
    valid_at = [i for i, d in enumerate(draws) if not isinstance(d, Undefined)]
    hits = np.flatnonzero(_extreme(_valid(draws), observed))
    return valid_at[hits[h - 1]] + 1 if hits.size >= h else len(draws)


def permutation_record(
    observed: float,
    draws: list[Estimate],
    requested: int,
    seed: int,
    n_fixed: int,
    statistic: str,
    h: int = SEQUENTIAL_H,
) -> dict:
    """Two-sided sequential Monte Carlo p-value by magnitude, ties counted as extreme (Besag & Clifford 1991, closed scheme).

    ``draws`` are read in draw order and stop at the ``h``-th valid draw with
    ``|T*| >= |T_obs|``: then ``p = h / l`` with ``l`` the valid draws so far.
    Without ``h`` exceedances, ``p = (g + 1) / (B_valid + 1)`` over every draw.
    Failed draws count toward neither ``l`` nor ``B_valid``. ``statistic``
    names what ``observed`` and ``draws`` are (``PERMUTATION_STATISTICS``).
    """
    used = _draws_used(observed, draws, h)
    draws = draws[:used]
    valid = _valid(draws)
    record = {
        "p_perm": None,
        "p_perm_unavailable_reason": None,
        "permutation_statistic": statistic,
        "permutations": {
            "requested": requested,
            "valid": int(valid.size),
            "failed": len(draws) - int(valid.size),
            "failure_reasons": _failure_counts(draws),
            "seed": seed,
            "n_fixed_fathers": n_fixed,
            "stopped_early": used < requested,
            "draws_used": used,
            "sequential_h": h,
        },
    }
    if requested == 0:
        record["p_perm_unavailable_reason"] = "not_requested"
    elif not draws:
        record["p_perm_unavailable_reason"] = "no_informative_permutations"
    elif valid.size == 0:
        record["p_perm_unavailable_reason"] = "no_valid_permutations"
    else:
        g = int(_extreme(valid, observed).sum())
        record["p_perm"] = h / valid.size if g >= h else (g + 1) / (valid.size + 1)
    return record


def _network_summary(labels: np.ndarray) -> dict:
    sizes = np.bincount(labels)
    return {
        "n_mate_networks": len(sizes),
        "largest_mate_network_share": float(sizes.max() / labels.size) if labels.size else None,
    }


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


PERMUTATION_STATISTICS = ("score_at_zero", "pearson")


def latent_scores(values: np.ndarray, stratum: np.ndarray, k: int | None) -> np.ndarray | Undefined:
    """Per pair, the conditional mean of one side's latent variable given its value, with margins per stratum.

    A continuous side is standardised within its stratum (``Undefined`` when a
    stratum is degenerate); level ``c`` of a discrete side with ``k`` levels gets
    ``E[η | c]`` under its stratum's thresholds (``kernels.latent_means``).
    """
    if k is None:
        return standardise(values, stratum)
    n_strata = int(stratum.max(initial=-1)) + 1
    margin = kernels.margin(values, stratum, unit_weights(values.size), n_strata, k)
    return kernels.latent_means(margin)[stratum, values.astype(np.int64)]


@dataclass
class _PermutationTest:
    """A permutation test of one primary estimator's form (0 crude, 1 stratified), and the record its p-value joins."""

    record: dict
    form: int
    statistic: str


@dataclass
class _Cell:
    """One R_mf cell's permutation input: its pairs, its father trait, each pair's distinct father, and the mothers' latent scores.

    ``e_m`` holds the crude and the stratified per-pair mother score (``latent_scores``),
    fixed under the null; the father side is refit inside the kernel.
    """

    pairs: CellPairs
    trait: int
    pair_father: np.ndarray
    e_m: tuple[np.ndarray, np.ndarray]
    tests: list[_PermutationTest]


def _mother_scores(pairs: CellPairs, k_m: int | None) -> tuple[np.ndarray, np.ndarray]:
    """The crude and stratified mother scores of a cell; a form without a defined estimate gets zeros."""
    out = []
    for stratum in (np.zeros_like(pairs.m_stratum), pairs.m_stratum):
        scores = latent_scores(pairs.m, stratum, k_m)
        out.append(np.zeros(pairs.m.size) if isinstance(scores, Undefined) else scores)
    return out[0], out[1]


def _cell_fathers(cell: _Cell, rank: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, ...]:
    """One cell collapsed onto its distinct fathers, in row order (``rank``: father -> row).

    Returns ``(father_row, count, e1, e2, runs, run_stratum, sums)``: per
    distinct father his row, pair count and summed crude / stratified mother
    scores; the offsets of each stretch of fathers in one stratum and that
    stratum; and per stratum ``(shift, pairs, Σ e1, Σ e2)``, with the
    observed mean father value as the shift of a continuous trait. A father's
    pairs share his stratum, and permutation keeps him in it.
    """
    row = rank[cell.pair_father]
    pairs_per_row = np.bincount(row, minlength=rank.size)
    # int32 rows and counts, STREAM_DTYPE scores: the permutation kernel streams these arrays once per draw.
    father_row = np.flatnonzero(pairs_per_row).astype(np.int32)
    count = pairs_per_row[father_row].astype(np.int32)
    e1 = np.bincount(row, weights=cell.e_m[0], minlength=rank.size)[father_row].astype(rows.dtype)
    e2 = np.bincount(row, weights=cell.e_m[1], minlength=rank.size)[father_row].astype(rows.dtype)
    row_stratum = np.zeros(rank.size, dtype=np.int64)
    row_stratum[row] = cell.pairs.f_stratum
    stratum = row_stratum[father_row]
    runs = np.flatnonzero(np.diff(stratum, prepend=-1))
    n_strata = cell.pairs.n_strata[1]
    n_s = np.bincount(stratum, weights=count, minlength=n_strata)
    shift = np.zeros(n_strata)
    if cell.pairs.f_levels is None:
        total = np.bincount(stratum, weights=count * rows[father_row, cell.trait], minlength=n_strata)
        np.divide(total, n_s, out=shift, where=n_s > 0)
    sums = np.stack(
        [
            shift,
            n_s,
            np.bincount(stratum, weights=e1, minlength=n_strata),
            np.bincount(stratum, weights=e2, minlength=n_strata),
        ],
        axis=1,
    )
    return father_row, count, e1, e2, runs, stratum[runs], sums


@dataclass
class _Stopping:
    """The closed-scheme state of every cell and form: exceedances so far, draws consumed, and whether the form has stopped.

    A form stops at the draw that brings its count of valid draws with
    ``|T*| >= |T_obs|`` (``_extreme``, the rule of ``permutation_record``) to
    ``h``; the record then reads its first ``used`` draws. A form without a
    test is done from the start, so a cell runs only while a tested form is open.
    """

    observed: np.ndarray
    h: int
    exceedances: np.ndarray
    used: np.ndarray
    done: np.ndarray

    @classmethod
    def start(cls, observed: np.ndarray, tested: np.ndarray, h: int) -> _Stopping:
        zeros = np.zeros(observed.shape, dtype=np.int64)
        return cls(observed, h, zeros, zeros.copy(), ~tested)

    def active(self) -> np.ndarray:
        """The cells with an open form."""
        return np.flatnonzero(~self.done.all(axis=1))

    def horizon(self) -> int:
        """The draw by which the nearest open form reaches ``h`` exceedances at its rate so far (one per draw so far if it has none)."""
        open_ = ~self.done
        if not open_.any():
            return 0
        return int(np.ceil(self.h * self.used[open_] / np.maximum(self.exceedances[open_], 1)).min())

    def scan(self, null: np.ndarray, status: np.ndarray, first_draw: int) -> np.ndarray:
        """Advance every open form over one batch of draws ``first_draw + i`` (``null`` NaN where failed) and return the cells still open."""
        for c in self.active():
            for form in np.flatnonzero(~self.done[c]):
                hits = _extreme(null[:, c, form], self.observed[c, form]) & (status[:, c, form] == 0)
                total = self.exceedances[c, form] + np.cumsum(hits)
                reached = np.flatnonzero(total >= self.h)
                if reached.size:
                    self.used[c, form] = first_draw + reached[0] + 1
                    self.done[c, form] = True
                else:
                    self.used[c, form] = first_draw + null.shape[0]
                self.exceedances[c, form] = min(int(total[-1]), self.h)
        return self.active()


#: Storage of the arrays the permutation kernel streams once per draw (the fathers' trait values and the
#: cells' summed mother scores). The kernel accumulates in float64 whatever this is; float32 halves the
#: traffic of a memory-bound pass and moves a statistic by about 1e-7 relative. Never narrower than float32.
STREAM_DTYPE = np.float32


def _offsets(sizes: list[int]) -> np.ndarray:
    out = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=out[1:])
    return out


@dataclass(frozen=True)
class _PermutationInput:
    """The cells of a run packed for ``kernels.permutation_statistics``.

    Fathers are renumbered block by block (``order``: row -> father), so a
    permutation shuffles contiguous stretches of row indices and then gathers
    ``rows`` (one row of trait values per father, stored as ``dtype``). Each cell is collapsed onto
    its distinct fathers in row order (``_cell_fathers``) and the cells' arrays
    are concatenated with offsets. ``ss_m`` is ``Σ e_m²`` per cell and form, the fixed half of the
    statistic's normaliser.
    """

    order: np.ndarray
    block_start: np.ndarray
    rows: np.ndarray
    arrays: tuple
    ss_m: np.ndarray

    @classmethod
    def pack(
        cls, cells: list[_Cell], trait_values: np.ndarray, blocks: np.ndarray, dtype: type = STREAM_DTYPE
    ) -> _PermutationInput:
        order, block_start = permutation_blocks(blocks)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        rows = trait_values[:, order].T
        for trait in {c.trait for c in cells if c.pairs.f_levels is None}:
            # Each statistic is invariant to a positive affine map of a continuous trait. Standardising in
            # float64 first keeps its variation in float32 storage, which would round values near 2e7 to steps of 2.
            column = rows[:, trait]
            column -= np.nanmean(column)
            sd = np.nanstd(column)
            if sd > 0:
                column /= sd
        rows = np.ascontiguousarray(rows, dtype=dtype)
        per_cell = [_cell_fathers(c, rank, rows) for c in cells]
        father_start = _offsets([f[0].size for f in per_cell])
        runs = np.concatenate(
            [f[4] + start for f, start in zip(per_cell, father_start[:-1], strict=True)] + [father_start[-1:]]
        )
        k_f = np.array([0 if c.pairs.f_levels is None else c.pairs.f_levels.shape[1] for c in cells], dtype=np.int64)
        levels = [np.zeros(0, dtype=bool) if c.pairs.f_levels is None else c.pairs.f_levels.ravel() for c in cells]
        arrays = (
            np.array([c.trait for c in cells], dtype=np.int64),
            _offsets([f[4].size for f in per_cell]),
            *(np.concatenate([f[i] for f in per_cell]) for i in range(4)),
            runs,
            np.concatenate([f[5] for f in per_cell]),
            _offsets([f[6].shape[0] for f in per_cell]),
            np.concatenate([f[6] for f in per_cell]),
            np.array([c.pair_father.size for c in cells], dtype=np.int64),
            k_f,
            _offsets([lv.size for lv in levels]),
            np.concatenate(levels),
        )
        return cls(order, block_start, rows, arrays, np.array([[np.dot(e, e) for e in c.e_m] for c in cells]))

    def statistics(self, donors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(cross, ss_f, status)`` of explicit donor arrays (``permutation_donors``), each ``(draws, cells, 2)``."""
        rank = np.empty_like(self.order)
        rank[self.order] = np.arange(self.order.size)
        arrangements = np.ascontiguousarray(self.rows[rank[donors[:, self.order]]].transpose(0, 2, 1))
        return kernels.arrangement_statistics(arrangements, *self.arrays)

    def observed(self) -> np.ndarray:
        """The statistic of the unpermuted data, ``(cells, 2)``."""
        unpermuted = np.ascontiguousarray(self.rows.T[None])
        return _standardised(*kernels.arrangement_statistics(unpermuted, *self.arrays), self.ss_m)[0]

    def permuted(
        self, permutations: int, seed: int, stopping: _Stopping | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The statistic and status of the permutations, each ``(permutations, cells, 2)``.

        Draws run in batches from ``PERMUTATION_BATCH`` up, each at least
        doubling the draws so far and reaching ``stopping.horizon()``; after
        each, ``stopping`` retires the cells whose tested forms have all
        stopped, and the pass ends when none is open. A retired cell's later
        rows are not computed (they read as failed draws); ``stopping.used``
        says how many to read. Without ``stopping`` every draw of every cell
        is computed in one launch.
        """
        n_cells, n_fathers, n_traits = self.ss_m.shape[0], *self.rows.shape
        n_workers = min(numba.get_num_threads(), permutations)
        order = np.empty((n_workers, n_fathers), dtype=np.int32)
        arranged = np.empty((n_workers, n_traits, n_fathers), dtype=self.rows.dtype)
        cross = np.zeros((permutations, n_cells, 2))
        ss_f = np.zeros((permutations, n_cells, 2))
        status = np.full((permutations, n_cells, 2), kernels.STATUS_NO_PAIRS, dtype=np.int64)
        cell_trait = self.arrays[0]
        active = np.arange(n_cells)
        first = 0
        while first < permutations:
            if stopping is None:
                end = permutations
            else:
                end = min(max(2 * first, PERMUTATION_BATCH, stopping.horizon()), permutations)
            trait_needed = np.isin(np.arange(n_traits), cell_trait[active])
            kernels.permutation_statistics(
                seed,
                first,
                end,
                active,
                trait_needed,
                self.block_start,
                self.rows,
                order,
                arranged,
                cross,
                ss_f,
                status,
                *self.arrays,
            )
            if stopping is not None:
                batch = _standardised(cross[first:end], ss_f[first:end], status[first:end], self.ss_m)
                active = stopping.scan(batch, status[first:end], first)
                if active.size == 0:
                    break
            first = end
        return _standardised(cross, ss_f, status, self.ss_m), status


def _permutation_statistics(
    cells: list[_Cell], trait_values: np.ndarray, blocks: np.ndarray, permutations: int, seed: int
) -> tuple[np.ndarray, list[list[list[Estimate]]]]:
    """The observed statistic ``(cells, 2)`` and the null draws per cell and form, over ``permutations`` draws of ``seed``.

    The statistic is ``Σ e_m e_f / √(Σ e_m² Σ e_f²)``: the score at ρ = 0
    normalised by its information, which is the Pearson r of the two sides'
    latent scores and the Pearson r itself for two continuous sides.
    """
    packed = _PermutationInput.pack(cells, trait_values, blocks)
    observed = packed.observed()
    tested = np.array([[any(t.form == form for t in c.tests) for form in range(2)] for c in cells], dtype=bool)
    stopping = _Stopping.start(observed, tested, SEQUENTIAL_H)
    null, status = packed.permuted(permutations, seed, stopping)
    draws = [
        [_null_draws(null[:used, c, form], status[:used, c, form]) for form, used in enumerate(stopping.used[c])]
        for c in range(len(cells))
    ]
    return observed, draws


def _null_draws(null: np.ndarray, status: np.ndarray) -> list[Estimate]:
    """One form's permutation draws: the statistic, or why the permuted sample had none."""
    return [Undefined(STATUS_REASONS[int(st)]) if st else float(t) for t, st in zip(null, status, strict=True)]


def _standardised(cross: np.ndarray, ss_f: np.ndarray, status: np.ndarray, ss_m: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(status == 0, cross / np.sqrt(ss_m * ss_f), np.nan)


def _draw_estimates(
    estimator: Estimator,
    pairs: CellPairs,
    labels: np.ndarray,
    seed: int,
    observed: Estimate,
    values: np.ndarray,
    status: np.ndarray,
) -> list[Estimate]:
    """One estimator's bootstrap draws from the kernel's ``(values, status)`` columns.

    A draw whose one-step has no positive Hessian at ρ̂ is refit in full from
    the observed estimate (``FIT_OUTCOMES["bootstrap_refit"]``).
    """
    start = estimate_value(observed) if isinstance(observed, Fit) else None
    draws: list[Estimate] = []
    for d, (value, code) in enumerate(zip(values, status, strict=True)):
        if code == 0:
            draws.append(float(value))
        elif code == kernels.STATUS_NONCONCAVE:
            FIT_OUTCOMES["bootstrap_refit"] += 1
            draws.append(estimator.fn(pairs, network_weights(labels, seed, d), start))
        else:
            draws.append(Undefined(STATUS_REASONS[int(code)]))
    return draws


def fit_cell(
    cell: CellEstimators,
    pairs: CellPairs,
    labels: np.ndarray,
    bootstrap: int,
    seed: int,
    stratified: bool,
) -> tuple[list[Estimate], list[dict]]:
    """Observed estimates of one cell and their records: sandwich SE, and a CI from it or from a Mate Network bootstrap.

    The ``bootstrap`` draws (``cell.draws``) resample Mate Networks with
    replacement, shared by every estimator of the cell; a closed-form estimator
    is recomputed exactly on each draw and a latent correlation takes one
    Newton step from its observed estimate.
    """
    estimators = cell.fitted(stratified)
    observed = [e.fn(pairs, unit_weights(pairs.m.size), None) for e in estimators]
    n_networks = int(labels.max()) + 1 if labels.size else 0
    draws: list[list[Estimate]] = [[] for _ in estimators]
    if n_networks >= 2 and bootstrap > 0:
        starts = [math.nan if isinstance(obs, Undefined) else estimate_value(obs) for obs in observed]
        values, status = cell.draws(pairs, labels, bootstrap, seed, starts)
        draws = [
            _draw_estimates(e, pairs, labels, seed, obs, values[:, k], status[:, k])
            for k, (e, obs) in enumerate(zip(estimators, observed, strict=True))
        ]
    records = []
    for e, obs, d in zip(estimators, observed, draws, strict=True):
        if isinstance(obs, Undefined):
            records.append({e.key: None, "reason": obs.reason})
        else:
            se = sandwich_se(e, pairs, obs, labels)
            records.append(
                {
                    e.key: estimate_value(obs),
                    **_estimate_fields(obs),
                    **inference_record(e, obs, se, d, bootstrap, n_networks),
                }
            )
    return observed, records


def within_person(traits: list[Trait], ids: np.ndarray, mothers: np.ndarray, fathers: np.ndarray) -> dict:
    """The Within-Person Cross-Trait Correlation per sex, once per distinct individual.

    An individual counts when they have both traits and at least one of the
    analysed Mating Pairs (``mothers``/``fathers``: one entry per pair).
    """
    first, second = traits
    estimator = CELL_ESTIMATORS[first.kind, second.kind].crude[0]
    out = {}
    for sex, pair_rows in (("mothers", mothers), ("fathers", fathers)):
        rows = unique_ints(pair_rows)
        rows = rows[np.argsort(ids[rows])]
        x, y = first.values[rows], second.values[rows]
        both = ~np.isnan(x) & ~np.isnan(y)
        n = int(both.sum())
        one_stratum = np.zeros(n, dtype=np.int64)
        people = CellPairs(x[both], y[both], one_stratum, one_stratum).with_levels(_n_levels(first), _n_levels(second))
        r = estimator.fn(people, unit_weights(n), None)
        record: dict = {"estimator": estimator.name, "n": n}
        if isinstance(r, Undefined):
            record |= {estimator.key: None, "reason": r.reason}
        else:
            record |= {estimator.key: estimate_value(r), **_estimate_fields(r)}
        out[sex] = record
    return out


def _trait_record(trait: Trait) -> dict:
    record: dict = {"name": trait.name, "type": trait.kind, "type_source": trait.type_source}
    if trait.levels is not None:
        record["levels"] = list(trait.levels)
    n_missing = trait.n_missing
    record |= {"n_values": len(trait.values) - n_missing, "n_missing": n_missing}
    return record


def compute_assortative_mating(
    df: pl.DataFrame,
    traits: list[Trait],
    *,
    permutations: int,
    bootstrap: int,
    seed: int,
    stratify_by: StratifyBy | None = None,
    birth_year_bin: int = 10,
    min_stratum_networks: int = MIN_STRATUM_NETWORKS,
) -> dict:
    """The ``assortative_mating`` payload: Mate Correlation cells over the analysed Mating Pairs.

    With ``stratify_by``, pairs with a mate of unknown stratum are dropped
    before any cell, and each cell then drops the pairs in its strata that span
    fewer than ``min_stratum_networks`` Mate Networks or are degenerate, so a
    cell's crude and stratified estimates share one sample.
    ``seed`` keys every permutation and bootstrap draw.
    """
    ids = df["id"].to_numpy()
    id_index = IdIndex(ids)
    mother_rows, _ = _parent_rows(df["mother"].to_numpy(), id_index)
    father_rows, _ = _parent_rows(df["father"].to_numpy(), id_index)
    mating = _group_mating_pairs(df, mother_rows, father_rows)
    mothers, fathers = mating.mother_rows, mating.father_rows
    n_total = len(mothers)

    row_stratum = np.zeros(len(df), dtype=np.int64) if stratify_by is None else strata(df, stratify_by, birth_year_bin)
    stratum_code = np.searchsorted(np.unique(row_stratum), row_stratum)
    known = (row_stratum[mothers] != -1) & (row_stratum[fathers] != -1)
    mothers, fathers = mothers[known], fathers[known]

    # Distinct fathers in father-id order, so blocks and draws do not depend on row order.
    _, first, pair_father = np.unique(ids[fathers], return_index=True, return_inverse=True)
    father_of = fathers[first]
    blocks = father_blocks(row_stratum[father_of], np.stack([~np.isnan(t.values[father_of]) for t in traits]))
    fixed_father = np.bincount(blocks)[blocks] == 1

    labels = mate_networks(mothers, fathers)
    payload = {
        "traits": [_trait_record(t) for t in traits],
        "settings": {
            "permutations": permutations,
            "bootstrap": bootstrap,
            "seed": seed,
            "threads": numba.get_num_threads(),
            "ci_level": CI_LEVEL,
            "ci_method": "bootstrap" if bootstrap > 0 else "sandwich",
            "stratify_by": stratify_by,
            "birth_year_bin": birth_year_bin if stratify_by == "birth_year" else None,
            "min_stratum_networks": min_stratum_networks if stratify_by is not None else None,
        },
        "inference": dict(INFERENCE),
        "mating_pairs": {
            "n_total": n_total,
            "n_dropped": {"unknown_stratum": int(n_total - known.sum())},
            **_network_summary(labels),
            "n_mothers_multiple_mates": int((np.bincount(mothers) > 1).sum()),
            "n_fathers_multiple_mates": int((np.bincount(pair_father) > 1).sum()),
        },
    }

    records = []
    cells: list[_Cell] = []
    for mother_trait in traits:
        for j, father_trait in enumerate(traits):
            m_all = mother_trait.values[mothers]
            f_all = father_trait.values[fathers]
            m_missing, f_missing = np.isnan(m_all), np.isnan(f_all)
            pairs = np.flatnonzero(~m_missing & ~f_missing)
            cell = CellPairs(m_all[pairs], f_all[pairs], stratum_code[mothers[pairs]], stratum_code[fathers[pairs]])
            n_complete, n_small = len(pairs), 0
            if stratify_by is None:
                cell_labels = mate_networks(mothers[pairs], fathers[pairs])
            else:
                keep, n_small, cell_labels = drop_thin_strata(
                    cell, mothers[pairs], fathers[pairs], min_stratum_networks
                )
                pairs, cell = pairs[keep], cell.take(keep)
            cell = cell.with_levels(_n_levels(mother_trait), _n_levels(father_trait))
            record: dict = {
                "mother": mother_trait.name,
                "father": father_trait.name,
                "n": len(pairs),
                "n_dropped": {
                    "mother_missing": int((m_missing & ~f_missing).sum()),
                    "father_missing": int((~m_missing & f_missing).sum()),
                    "both_missing": int((m_missing & f_missing).sum()),
                    "small_stratum": n_small,
                    "degenerate_stratum": n_complete - len(pairs) - n_small,
                },
            }
            estimators = CELL_ESTIMATORS[mother_trait.kind, father_trait.kind]
            observed, fits = fit_cell(estimators, cell, cell_labels, bootstrap, seed, stratify_by is not None)
            crude: dict = {} if estimators.table is None else {"table": estimators.table(cell)}
            crude |= {e.name: fit for e, fit in zip(estimators.crude, fits[: len(estimators.crude)], strict=True)}
            record |= _network_summary(cell_labels) | {"crude": crude}
            primaries = [(crude[estimators.crude[0].name], observed[0])]
            if stratify_by is not None:
                stratified = fits[-1] | {
                    "n_strata_mothers": len(np.unique(cell.m_stratum)),
                    "n_strata_fathers": len(np.unique(cell.f_stratum)),
                }
                record["stratified"] = {estimators.stratified.name: stratified}
                primaries.append((stratified, observed[-1]))
            statistic = "pearson" if estimators.crude[0].name == "pearson" else "score_at_zero"
            tests = [
                _PermutationTest(fit, form, statistic)
                for form, (fit, obs) in enumerate(primaries)
                if not isinstance(obs, Undefined)
            ]
            mother_scores = _mother_scores(cell, _n_levels(mother_trait)) if tests else (np.zeros(0), np.zeros(0))
            records.append(record)
            cells.append(_Cell(cell, j, pair_father[pairs], mother_scores, tests))

    informative = [k for k, c in enumerate(cells) if c.tests and not fixed_father[c.pair_father].all()]
    observed_stats = np.full((len(cells), 2), math.nan)
    draws: list[list[list[Estimate]]] = [[[], []] for _ in cells]
    if informative and permutations > 0:
        trait_values = np.stack([t.values[father_of] for t in traits])
        stats, null = _permutation_statistics([cells[k] for k in informative], trait_values, blocks, permutations, seed)
        for k, stat, cell_null in zip(informative, stats, null, strict=True):
            observed_stats[k], draws[k] = stat, cell_null
    for cell, stat, cell_draws in zip(cells, observed_stats, draws, strict=True):
        n_fixed = int(fixed_father[unique_ints(cell.pair_father)].sum())
        for test in cell.tests:
            test.record |= permutation_record(
                float(stat[test.form]), cell_draws[test.form], permutations, seed, n_fixed, test.statistic
            )

    payload["mate_correlation"] = records
    if len(traits) == 2:
        payload["within_person"] = within_person(traits, ids, mothers, fathers)
    payload["notes"] = list(NOTES)
    return payload
