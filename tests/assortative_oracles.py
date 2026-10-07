"""The v2 NumPy estimators of ``pedsum.assortative_mating``, kept verbatim as oracles for the numba kernels.

Each function is the pre-kernel implementation: bounded Brent for ρ, ``ndtr``
over every pair, ``rankdata`` for Spearman, ``np.minimum.at`` for the
degenerate check. A resample is a ``take`` with repeats, never a weight, so an
oracle call on ``pairs.take(np.repeat(np.arange(n), w))`` is the reference for
a kernel call with weights ``w``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from scipy.stats import norm, rankdata

from pedsum.assortative_mating import BOUNDARY_MARGIN, BOUNDARY_NLL_TOL, LATENT_BOUND, Fit, Undefined, bvn_cdf, pooled

if TYPE_CHECKING:
    from collections.abc import Callable

    from pedsum.assortative_mating import CellPairs, Estimate

_TINY = 1e-300


def _check_margins(m: np.ndarray, f: np.ndarray) -> Undefined | None:
    if m.size == 0:
        return Undefined("no_complete_pairs")
    if m.min() == m.max() or f.min() == f.max():
        return Undefined("constant_margin")
    return None


def pearson(m: np.ndarray, f: np.ndarray) -> Estimate:
    """Pearson correlation of mother and father values, one observation per Mating Pair."""
    undefined = _check_margins(m, f)
    if undefined is not None:
        return undefined
    cm = m - m.mean()
    cf = f - f.mean()
    r = float(np.dot(cm, cf) / np.sqrt(np.dot(cm, cm) * np.dot(cf, cf)))
    return min(1.0, max(-1.0, r))


def spearman(m: np.ndarray, f: np.ndarray) -> Estimate:
    """Spearman rank correlation (average ranks for ties)."""
    undefined = _check_margins(m, f)
    if undefined is not None:
        return undefined
    return pearson(rankdata(m), rankdata(f))


def degenerate_strata(x: np.ndarray, code: np.ndarray) -> np.ndarray:
    """Per stratum code, whether the stratum is present in ``x`` and constant there (a single value is constant)."""
    lo = np.full(int(code.max(initial=-1)) + 1, np.inf)
    hi = np.full_like(lo, -np.inf)
    np.minimum.at(lo, code, x)
    np.maximum.at(hi, code, x)
    return lo == hi


def standardise(x: np.ndarray, code: np.ndarray) -> np.ndarray | Undefined:
    """``x`` centred and scaled to unit SD (``ddof=0``) within each stratum of ``code``.

    Each entry is one Mating Pair, so the moments are pair-weighted. A
    degenerate stratum makes the result undefined, never merged into another.
    """
    if degenerate_strata(x, code).any():
        return Undefined("degenerate_stratum")
    count = np.bincount(code)
    mean = np.bincount(code, weights=x) / np.maximum(count, 1)
    dev = x - mean[code]
    sd = np.sqrt(np.bincount(code, weights=dev * dev) / np.maximum(count, 1))
    return dev / sd[code]


def stratified_pearson(pairs: CellPairs) -> Estimate:
    """Pearson of mother and father values, each standardised within their own sex x stratum."""
    if pairs.m.size == 0:
        return Undefined("no_complete_pairs")
    zm = standardise(pairs.m, pairs.m_stratum)
    if isinstance(zm, Undefined):
        return zm
    zf = standardise(pairs.f, pairs.f_stratum)
    if isinstance(zf, Undefined):
        return zf
    return pearson(zm, zf)


def thresholds(margin: np.ndarray) -> np.ndarray:
    """Per row of ``margin`` (strata × levels), ``Φ⁻¹`` of the cumulative proportions, framed by -inf and +inf.

    Olsson (1979) eqs 15-18; Olsson, Drasgow & Dorans (1982) eq 36 (F4). A row
    without counts gets all -inf and is never used.
    """
    total = margin.sum(axis=1, keepdims=True)
    cum = np.cumsum(margin, axis=1) / np.where(total == 0, 1, total)
    edge = np.full((len(margin), 1), np.inf)
    return np.concatenate([-edge, norm.ppf(cum[:, :-1]), edge], axis=1)


def count_table(pairs: CellPairs) -> np.ndarray:
    """Pair counts by (mother stratum, father stratum, mother level, father level)."""
    m_levels, f_levels = pairs.m_levels, pairs.f_levels
    shape = (m_levels.shape[0], f_levels.shape[0], m_levels.shape[1], f_levels.shape[1])
    flat = np.ravel_multi_index(
        (pairs.m_stratum, pairs.f_stratum, pairs.m.astype(np.int64), pairs.f.astype(np.int64)), shape
    )
    return np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape)


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
    rho, best = float(result.x), float(result.fun)
    at_bound = abs(rho) >= LATENT_BOUND - BOUNDARY_MARGIN
    plateau = nll(math.copysign(LATENT_BOUND, rho)) - best <= BOUNDARY_NLL_TOL * max(1.0, abs(best))
    return Fit(rho, at_bound or plateau)


def polychoric(pairs: CellPairs) -> Estimate:
    """Two-step ML polychoric correlation of two binary or ordinal sides (tetrachoric when both are binary).

    Thresholds per sex × stratum from that stratum's pair-weighted margin
    (Olsson 1979 eqs 15-18), then one ρ maximising the log-likelihood (eq 3)
    with cell probabilities from the corner CDFs (eq 4), summed over every
    mother-stratum × father-stratum table. Φ2 is evaluated on the corner grids
    only, never per pair.
    """
    if pairs.m.size == 0:
        return Undefined("no_complete_pairs")
    m_levels, f_levels = pairs.m_levels, pairs.f_levels
    n = count_table(pairs)
    m_margin, f_margin = n.sum(axis=(1, 3)), n.sum(axis=(0, 2))
    for margin, levels in ((m_margin, m_levels), (f_margin, f_levels)):
        undefined = _check_levels(margin, levels)
        if undefined is not None:
            return undefined
    a, b = thresholds(m_margin), thresholds(f_margin)
    combos = np.argwhere(n.sum(axis=(2, 3)) > 0)
    counts = n[combos[:, 0], combos[:, 1]]
    observed = counts > 0
    h = a[combos[:, 0]][:, :, None]
    k = b[combos[:, 1]][:, None, :]

    def nll(rho: float) -> float:
        cdf = bvn_cdf(h, k, rho)
        pi = cdf[:, 1:, 1:] - cdf[:, :-1, 1:] - cdf[:, 1:, :-1] + cdf[:, :-1, :-1]
        return -float(np.sum(counts[observed] * np.log(np.maximum(pi[observed], _TINY))))

    return _maximise_rho(nll)


def polyserial(
    x: np.ndarray, y: np.ndarray, x_stratum: np.ndarray, y_stratum: np.ndarray, y_levels: np.ndarray
) -> Estimate:
    """Two-step ML polyserial correlation of a continuous ``x`` and a binary or ordinal ``y`` (biserial when binary).

    ``x`` is standardised within its stratum with the 1/N variance and ``y``
    gets thresholds per stratum from its cumulative proportions (Olsson, Drasgow
    & Dorans 1982 eq 36); ρ then maximises the conditional term of eq 20 with
    the eq 19 probabilities ``Φ(τ*_j) − Φ(τ*_{j−1})``, ``τ*_j = (τ_j − ρz)/√(1−ρ²)``.
    """
    if x.size == 0:
        return Undefined("no_complete_pairs")
    if x.min() == x.max():
        return Undefined("constant_margin")
    z = standardise(x, x_stratum)
    if isinstance(z, Undefined):
        return z
    n_strata, k = y_levels.shape
    codes = y.astype(np.int64)
    margin = np.bincount(y_stratum * k + codes, minlength=n_strata * k).reshape(n_strata, k)
    undefined = _check_levels(margin, y_levels)
    if undefined is not None:
        return undefined
    tau = thresholds(margin)
    upper, lower = tau[y_stratum, codes + 1], tau[y_stratum, codes]
    # Φ at an infinite threshold is exactly 0 or 1, so only the finite ones are evaluated.
    fin_u, fin_l = np.isfinite(upper), np.isfinite(lower)
    upper, z_u = upper[fin_u], z[fin_u]
    lower, z_l = lower[fin_l], z[fin_l]

    def nll(rho: float) -> float:
        scale = math.sqrt(1 - rho * rho)
        cdf_upper = np.ones(z.shape)
        cdf_upper[fin_u] = ndtr((upper - rho * z_u) / scale)
        cdf_lower = np.zeros(z.shape)
        cdf_lower[fin_l] = ndtr((lower - rho * z_l) / scale)
        return -float(np.sum(np.log(np.maximum(cdf_upper - cdf_lower, _TINY))))

    return _maximise_rho(nll)


def odds_ratio(pairs: CellPairs) -> Estimate:
    """Cross-product ratio ``ad / (bc)`` of the 2×2 table; ``inf`` when ``bc = 0``.

    ``0 / 0`` needs a constant margin, which is undefined first.
    """
    undefined = _check_margins(pairs.m, pairs.f)
    if undefined is not None:
        return undefined
    (a, b), (c, d) = count_table(pooled(pairs)).astype(np.float64)[0, 0]
    with np.errstate(divide="ignore"):
        return float(a * d / (b * c))


# ---------------------------------------------------------------------------
# Sandwich oracle: numerical Jacobian of the stacked estimating equations
# ---------------------------------------------------------------------------


def phi2(h: float, k: float, rho: float) -> float:
    """Φ2 by Olsson's identity ∂Φ2/∂ρ = φ2: Φ(h)Φ(k) plus the bivariate density integrated in ρ from 0."""
    if h == -np.inf or k == -np.inf:
        return 0.0
    if h == np.inf:
        return float(norm.cdf(k))
    if k == np.inf:
        return float(norm.cdf(h))

    def density(r):
        return np.exp(-(h * h - 2 * r * h * k + k * k) / (2 * (1 - r * r))) / (2 * np.pi * np.sqrt(1 - r * r))

    return float(norm.cdf(h) * norm.cdf(k) + quad(density, 0, rho, epsabs=1e-13, epsrel=1e-13)[0])


def _bvn_pdf(h: float, k: float, rho: float) -> float:
    if np.isinf(h) or np.isinf(k):
        return 0.0
    q = 1 - rho * rho
    return float(np.exp(-(h * h - 2 * rho * h * k + k * k) / (2 * q)) / (2 * np.pi * np.sqrt(q)))


def sandwich_se(
    psi: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, labels: np.ndarray, h: float = 1e-5
) -> float:
    """SE of the last parameter from ``A = ∂Σψ/∂θ`` by central differences and the cluster meat ``G/(G−1) Σ_g ψ_g ψ_gᵀ``.

    ``psi(theta)`` returns one row of estimating-equation values per pair; the
    parameter of interest is ``theta[-1]``. Nothing here knows the block
    structure, so the result checks the production influence functions from the
    outside.
    """
    n_p = theta.size
    jac = np.empty((n_p, n_p))
    for j in range(n_p):
        step = np.zeros(n_p)
        step[j] = h
        jac[:, j] = (psi(theta + step).sum(axis=0) - psi(theta - step).sum(axis=0)) / (2 * h)
    at = psi(theta)
    g = int(labels.max()) + 1
    sums = np.zeros((g, n_p))
    np.add.at(sums, labels, at)
    meat = g / (g - 1) * sums.T @ sums
    bread = np.linalg.inv(jac)
    cov = bread @ meat @ bread.T
    return float(np.sqrt(cov[-1, -1]))


def _moment_equations(x: np.ndarray, code: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """``(x − μ_s)`` and ``((x − μ_s)² − v_s)`` of each pair in its own stratum's columns: ``(n, 2S)``."""
    n_s = mean.size
    out = np.zeros((x.size, 2 * n_s))
    dev = x - mean[code]
    out[np.arange(x.size), code] = dev
    out[np.arange(x.size), n_s + code] = dev * dev - var[code]
    return out


def _moments(x: np.ndarray, code: np.ndarray, n_s: int) -> tuple[np.ndarray, np.ndarray]:
    count = np.bincount(code, minlength=n_s)
    mean = np.bincount(code, weights=x, minlength=n_s) / np.maximum(count, 1)
    var = np.bincount(code, weights=(x - mean[code]) ** 2, minlength=n_s) / np.maximum(count, 1)
    return mean, var


def pearson_equations(pairs: CellPairs) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """``θ = (μ_a, v_a, μ_b, v_b, ρ)`` over each side's strata; ``ψ_ρ = z_a z_b − ρ``."""
    n_a, n_b = pairs.n_strata
    mean_a, var_a = _moments(pairs.m, pairs.m_stratum, n_a)
    mean_b, var_b = _moments(pairs.f, pairs.f_stratum, n_b)
    z = (pairs.m - mean_a[pairs.m_stratum]) / np.sqrt(var_a[pairs.m_stratum])
    z *= (pairs.f - mean_b[pairs.f_stratum]) / np.sqrt(var_b[pairs.f_stratum])
    theta = np.concatenate([mean_a, var_a, mean_b, var_b, [z.mean()]])

    def psi(t: np.ndarray) -> np.ndarray:
        ma, va, mb, vb = t[:n_a], t[n_a : 2 * n_a], t[2 * n_a : 2 * n_a + n_b], t[2 * n_a + n_b : -1]
        za = (pairs.m - ma[pairs.m_stratum]) / np.sqrt(va[pairs.m_stratum])
        zb = (pairs.f - mb[pairs.f_stratum]) / np.sqrt(vb[pairs.f_stratum])
        return np.column_stack(
            [
                _moment_equations(pairs.m, pairs.m_stratum, ma, va),
                _moment_equations(pairs.f, pairs.f_stratum, mb, vb),
                za * zb - t[-1],
            ]
        )

    return psi, theta


def _finite_thresholds(codes: np.ndarray, stratum: np.ndarray, n_strata: int, k: int) -> list[tuple[int, int]]:
    """``(stratum, j)`` of every threshold strictly inside (0, 1) of the cumulative proportions, in parameter order."""
    margin = np.bincount(stratum * k + codes, minlength=n_strata * k).reshape(n_strata, k)
    cum = np.cumsum(margin, axis=1)[:, :-1]
    total = margin.sum(axis=1, keepdims=True)
    return [(s, j + 1) for s in range(n_strata) for j in range(k - 1) if 0 < cum[s, j] < total[s, 0]]


def _threshold_equations(
    codes: np.ndarray, stratum: np.ndarray, which: list[tuple[int, int]], values: np.ndarray
) -> np.ndarray:
    """``1[c ≤ j−1] − Φ(τ_sj)`` for each pair in stratum ``s``, one column per listed threshold."""
    out = np.zeros((codes.size, len(which)))
    for col, ((s, j), tau) in enumerate(zip(which, values, strict=True)):
        rows = stratum == s
        out[rows, col] = (codes[rows] <= j - 1) - norm.cdf(tau)
    return out


def _full_thresholds(which: list[tuple[int, int]], values: np.ndarray, n_strata: int, k: int) -> np.ndarray:
    """Threshold matrix ``(n_strata, k+1)`` framed by ±inf; an edge threshold not in ``which`` is ±inf."""
    out = np.full((n_strata, k + 1), np.nan)
    out[:, 0], out[:, k] = -np.inf, np.inf
    for (s, j), tau in zip(which, values, strict=True):
        out[s, j] = tau
    for s in range(n_strata):
        for j in range(1, k):
            if np.isnan(out[s, j]):
                out[s, j] = -np.inf if all(np.isnan(out[s, 1:j]) | np.isneginf(out[s, 1:j])) else np.inf
    return out


def polyserial_equations(
    x: np.ndarray, y: np.ndarray, x_stratum: np.ndarray, y_stratum: np.ndarray, k: int, rho: float
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """``θ = (μ, v, finite τ, ρ)``; ``ψ_ρ`` is ODD 1982 eq 26 per pair."""
    n_xs, n_ys = int(x_stratum.max()) + 1, int(y_stratum.max()) + 1
    codes = y.astype(np.int64)
    mean, var = _moments(x, x_stratum, n_xs)
    which = _finite_thresholds(codes, y_stratum, n_ys, k)
    tau_hat = thresholds(np.bincount(y_stratum * k + codes, minlength=n_ys * k).reshape(n_ys, k))
    theta = np.concatenate([mean, var, [tau_hat[s, j] for s, j in which], [rho]])

    def psi(t: np.ndarray) -> np.ndarray:
        mu, v, tau_values, r = t[:n_xs], t[n_xs : 2 * n_xs], t[2 * n_xs : -1], t[-1]
        tau = _full_thresholds(which, tau_values, n_ys, k)
        z = (x - mu[x_stratum]) / np.sqrt(v[x_stratum])
        s = np.sqrt(1 - r * r)
        upper, lower = tau[y_stratum, codes + 1], tau[y_stratum, codes]
        p = norm.cdf((upper - r * z) / s) - norm.cdf((lower - r * z) / s)
        with np.errstate(invalid="ignore"):
            g_u = np.where(np.isinf(upper), 0.0, norm.pdf((upper - r * z) / s) * (upper * r - z))
            g_l = np.where(np.isinf(lower), 0.0, norm.pdf((lower - r * z) / s) * (lower * r - z))
        score = (g_u - g_l) / (s**3 * p)
        return np.column_stack(
            [_moment_equations(x, x_stratum, mu, v), _threshold_equations(codes, y_stratum, which, tau_values), score]
        )

    return psi, theta


def polychoric_equations(pairs: CellPairs, rho: float) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """``θ = (finite a, finite b, ρ)``; ``ψ_ρ`` is the Olsson (1979) eq 9 score of each pair's cell, Φ2 by quadrature."""
    m, f = pairs.m.astype(np.int64), pairs.f.astype(np.int64)
    k_m, k_f = pairs.m_levels.shape[1], pairs.f_levels.shape[1]
    n_m, n_f = pairs.n_strata
    which_a = _finite_thresholds(m, pairs.m_stratum, n_m, k_m)
    which_b = _finite_thresholds(f, pairs.f_stratum, n_f, k_f)
    a_hat = thresholds(np.bincount(pairs.m_stratum * k_m + m, minlength=n_m * k_m).reshape(n_m, k_m))
    b_hat = thresholds(np.bincount(pairs.f_stratum * k_f + f, minlength=n_f * k_f).reshape(n_f, k_f))
    theta = np.concatenate([[a_hat[s, j] for s, j in which_a], [b_hat[s, j] for s, j in which_b], [rho]])
    cells = np.unique(np.column_stack([pairs.m_stratum, pairs.f_stratum, m, f]), axis=0)

    def psi(t: np.ndarray) -> np.ndarray:
        a = _full_thresholds(which_a, t[: len(which_a)], n_m, k_m)
        b = _full_thresholds(which_b, t[len(which_a) : -1], n_f, k_f)
        r = t[-1]
        score = {}
        for s, u, i, j in cells:
            corners = [(a[s, i + 1], b[u, j + 1]), (a[s, i], b[u, j + 1]), (a[s, i + 1], b[u, j]), (a[s, i], b[u, j])]
            signs = (1, -1, -1, 1)
            pi = sum(sg * phi2(h, k, r) for sg, (h, k) in zip(signs, corners, strict=True))
            d_pi = sum(sg * _bvn_pdf(h, k, r) for sg, (h, k) in zip(signs, corners, strict=True))
            score[s, u, i, j] = d_pi / pi
        rho_column = np.array(
            [score[s, u, i, j] for s, u, i, j in zip(pairs.m_stratum, pairs.f_stratum, m, f, strict=True)]
        )
        return np.column_stack(
            [
                _threshold_equations(m, pairs.m_stratum, which_a, t[: len(which_a)]),
                _threshold_equations(f, pairs.f_stratum, which_b, t[len(which_a) : -1]),
                rho_column,
            ]
        )

    return psi, theta


def odds_ratio_equations(pairs: CellPairs) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """``θ = (p_00, p_01, p_10, p_11, log OR)``; the last equation is ``log OR − log(p_00 p_11 / (p_01 p_10))``."""
    cell = (pairs.m.astype(np.int64) * 2 + pairs.f.astype(np.int64)).astype(np.int64)
    p = np.bincount(cell, minlength=4) / cell.size
    theta = np.concatenate([p, [np.log(p[0] * p[3] / (p[1] * p[2]))]])

    def psi(t: np.ndarray) -> np.ndarray:
        indicators = np.eye(4)[cell] - t[:4]
        last = np.full(cell.size, t[4] - np.log(t[0] * t[3] / (t[1] * t[2])))
        return np.column_stack([indicators, last])

    return psi, theta
