"""numba kernels behind the assortative-mating estimators: one pass over a cell's Mating Pairs each.

Every kernel takes frequency weights ``w`` (one float per pair; a pair with
``w = 0`` is absent), so a bootstrap draw is a weight vector over the cell's
pairs and no estimator ever copies values. The discrete sides arrive as the
float level codes ``0..k-1`` that ``Trait`` stores, cast inside the loops.
"""

from __future__ import annotations

import math

import numba
import numpy as np

_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_TINY = 1e-300

#: Status of one estimate on one sample: 0 defined, else why not (``STATUS_REASONS`` in the estimators module
#: names them). ``STATUS_NONCONCAVE`` is a one-step bootstrap draw whose Hessian at ρ̂ is not positive;
#: the caller refits that draw in full.
STATUS_CONSTANT_MARGIN = 1
STATUS_EMPTY_CATEGORY = 2
STATUS_DEGENERATE_STRATUM = 3
STATUS_NO_PAIRS = 4
STATUS_NONCONCAVE = 5


@numba.njit(cache=True)
def ndtr(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / _SQRT2)


# Wichura (1988) AS 241 PPND16 rational approximations, coefficients in ascending powers.
_PPND16_A = (
    3.3871328727963666080e0, 1.3314166789178437745e2, 1.9715909503065514427e3, 1.3731693765509461125e4,
    4.5921953931549871457e4, 6.7265770927008700853e4, 3.3430575583588128105e4, 2.5090809287301226727e3,
)  # fmt: skip
_PPND16_B = (
    1.0, 4.2313330701600911252e1, 6.8718700749205790830e2, 5.3941960214247511077e3,
    2.1213794301586595867e4, 3.9307895800092710610e4, 2.8729085735721942674e4, 5.2264952788528545610e3,
)  # fmt: skip
_PPND16_C = (
    1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0, 3.64784832476320460504e0,
    1.27045825245236838258e0, 2.41780725177450611770e-1, 2.27238449892691845833e-2, 7.74545014278341407640e-4,
)  # fmt: skip
_PPND16_D = (
    1.0, 2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1,
    1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4, 1.05075007164441684324e-9,
)  # fmt: skip
_PPND16_E = (
    6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0, 2.96560571828504891230e-1,
    2.65321895265761230930e-2, 1.24266094738807843860e-3, 2.71155556874348757815e-5, 2.01033439929228813265e-7,
)  # fmt: skip
_PPND16_F = (
    1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
    7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7, 2.04426310338993978564e-15,
)  # fmt: skip


@numba.njit(cache=True)
def _horner(coef: tuple, r: float) -> float:
    acc = 0.0
    for c in coef[::-1]:
        acc = acc * r + c
    return acc


@numba.njit(cache=True)
def _interval_mass(upper: float, lower: float) -> float:
    """``Φ(upper) − Φ(lower)`` without cancellation when both lie in the upper tail (then via ``1 − Φ``)."""
    if lower > 0.0:
        return 0.5 * (math.erfc(lower / _SQRT2) - math.erfc(upper / _SQRT2))
    return ndtr(upper) - ndtr(lower)


@numba.njit(cache=True)
def ndtri(p: float) -> float:
    """Standard normal quantile (Wichura 1988, algorithm AS 241, PPND16); ``±inf`` at ``p = 0, 1``.

    Agrees with ``scipy.special.ndtri`` to 1e-15 on the grid in
    ``test_ndtri_matches_scipy``.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * _horner(_PPND16_A, r) / _horner(_PPND16_B, r)
    r = math.sqrt(-math.log(p if q < 0.0 else 1.0 - p))
    if r <= 5.0:
        r -= 1.6
        value = _horner(_PPND16_C, r) / _horner(_PPND16_D, r)
    else:
        r -= 5.0
        value = _horner(_PPND16_E, r) / _horner(_PPND16_F, r)
    return -value if q < 0.0 else value


# ---------------------------------------------------------------------------
# Blocked reductions
# ---------------------------------------------------------------------------
#
# Every O(n) pass over a cell's pairs sums fixed blocks of ``_BLOCK`` pairs into
# per-block partials and adds the partials in block order, in a ``prange`` over
# the blocks (the point fits and the sandwich) or a plain ``range`` (the draw
# kernels, which already run one draw per thread). The blocking does not
# depend on the thread count, so a result is bit-identical under any
# ``--threads`` and the serial twin of a kernel equals its parallel form.

#: Pairs per reduction block.
_BLOCK = 1 << 14


@numba.njit(cache=True)
def _n_blocks(n: int) -> int:
    return max(1, (n + _BLOCK - 1) // _BLOCK)


@numba.njit(cache=True)
def _block_end(n: int, b: int) -> int:
    return min(n, (b + 1) * _BLOCK)


@numba.njit(cache=True)
def _block_sum(partials: np.ndarray) -> np.ndarray:
    """Sum of ``partials`` over its first axis, block by block."""
    out = np.zeros(partials.shape[1:])
    for b in range(partials.shape[0]):
        out += partials[b]
    return out


@numba.njit(cache=True)
def _moment_sums(
    x: np.ndarray,
    code: np.ndarray,
    w: np.ndarray,
    b: int,
    total: np.ndarray,
    s: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> None:
    for i in range(b * _BLOCK, _block_end(x.size, b)):
        if w[i] > 0.0:
            c = code[i]
            total[c] += w[i]
            s[c] += w[i] * x[i]
            lo[c] = min(lo[c], x[i])
            hi[c] = max(hi[c], x[i])


@numba.njit(cache=True)
def _variance_sums(x: np.ndarray, code: np.ndarray, w: np.ndarray, mean: np.ndarray, b: int, var: np.ndarray) -> None:
    for i in range(b * _BLOCK, _block_end(x.size, b)):
        if w[i] > 0.0:
            c = code[i]
            d = x[i] - mean[c]
            var[c] += w[i] * d * d


@numba.njit(cache=True)
def _moment_partials(n_blocks: int, n_codes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((n_blocks, n_codes)),
        np.zeros((n_blocks, n_codes)),
        np.full((n_blocks, n_codes), math.inf),
        np.full((n_blocks, n_codes), -math.inf),
    )


@numba.njit(cache=True)
def _moments_from_partials(
    total_b: np.ndarray, s_b: np.ndarray, lo_b: np.ndarray, hi_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = _block_sum(total_b)
    mean = _block_sum(s_b)
    lo = np.full(total.size, math.inf)
    hi = np.full(total.size, -math.inf)
    for b in range(lo_b.shape[0]):
        for c in range(total.size):
            lo[c] = min(lo[c], lo_b[b, c])
            hi[c] = max(hi[c], hi_b[b, c])
    for c in range(total.size):
        if total[c] > 0.0:
            mean[c] /= total[c]
    return total, mean, lo, hi


@numba.njit(cache=True)
def _divide_present(var: np.ndarray, total: np.ndarray) -> np.ndarray:
    for c in range(total.size):
        if total[c] > 0.0:
            var[c] /= total[c]
    return var


@numba.njit(parallel=True, cache=True)
def stratum_moments(
    x: np.ndarray, code: np.ndarray, w: np.ndarray, n_codes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per stratum ``code``: total weight, weighted mean, weighted ``1/N`` variance, min and max of ``x``.

    An absent stratum has weight 0, min ``+inf`` and max ``-inf``.
    """
    n_blocks = _n_blocks(x.size)
    total_b, s_b, lo_b, hi_b = _moment_partials(n_blocks, n_codes)
    for b in numba.prange(n_blocks):
        _moment_sums(x, code, w, b, total_b[b], s_b[b], lo_b[b], hi_b[b])
    total, mean, lo, hi = _moments_from_partials(total_b, s_b, lo_b, hi_b)
    var_b = np.zeros((n_blocks, n_codes))
    for b in numba.prange(n_blocks):
        _variance_sums(x, code, w, mean, b, var_b[b])
    return total, mean, _divide_present(_block_sum(var_b), total), lo, hi


@numba.njit(cache=True)
def stratum_moments_serial(
    x: np.ndarray, code: np.ndarray, w: np.ndarray, n_codes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``stratum_moments`` on the calling thread (for the draw kernels); the same numbers."""
    n_blocks = _n_blocks(x.size)
    total_b, s_b, lo_b, hi_b = _moment_partials(n_blocks, n_codes)
    for b in range(n_blocks):
        _moment_sums(x, code, w, b, total_b[b], s_b[b], lo_b[b], hi_b[b])
    total, mean, lo, hi = _moments_from_partials(total_b, s_b, lo_b, hi_b)
    var_b = np.zeros((n_blocks, n_codes))
    for b in range(n_blocks):
        _variance_sums(x, code, w, mean, b, var_b[b])
    return total, mean, _divide_present(_block_sum(var_b), total), lo, hi


@numba.njit(cache=True)
def _pearson_sums(a: np.ndarray, b: np.ndarray, w: np.ndarray, blk: int, out: np.ndarray) -> None:
    """Block ``blk`` of the first Pearson pass into ``out``: ``(Σw, Σwa, Σwb, min a, max a, min b, max b)``."""
    out[3] = out[5] = math.inf
    out[4] = out[6] = -math.inf
    for i in range(blk * _BLOCK, _block_end(a.size, blk)):
        if w[i] > 0.0:
            out[0] += w[i]
            out[1] += w[i] * a[i]
            out[2] += w[i] * b[i]
            out[3] = min(out[3], a[i])
            out[4] = max(out[4], a[i])
            out[5] = min(out[5], b[i])
            out[6] = max(out[6], b[i])


@numba.njit(cache=True)
def _pearson_cross_sums(
    a: np.ndarray, b: np.ndarray, w: np.ndarray, ma: float, mb: float, blk: int, out: np.ndarray
) -> None:
    """Block ``blk`` of the second Pearson pass into ``out``: ``(Σw da db, Σw da², Σw db²)``."""
    for i in range(blk * _BLOCK, _block_end(a.size, blk)):
        if w[i] > 0.0:
            da = a[i] - ma
            db = b[i] - mb
            out[0] += w[i] * da * db
            out[1] += w[i] * da * da
            out[2] += w[i] * db * db


@numba.njit(cache=True)
def _pearson_from_partials(sums: np.ndarray) -> tuple[int, float, float, float]:
    """``(status, total, mean a, mean b)`` from the first pass's block partials."""
    total = 0.0
    sa = 0.0
    sb = 0.0
    lo_a = math.inf
    hi_a = -math.inf
    lo_b = math.inf
    hi_b = -math.inf
    for blk in range(sums.shape[0]):
        total += sums[blk, 0]
        sa += sums[blk, 1]
        sb += sums[blk, 2]
        lo_a = min(lo_a, sums[blk, 3])
        hi_a = max(hi_a, sums[blk, 4])
        lo_b = min(lo_b, sums[blk, 5])
        hi_b = max(hi_b, sums[blk, 6])
    if total == 0.0:
        return STATUS_NO_PAIRS, 0.0, 0.0, 0.0
    if lo_a == hi_a or lo_b == hi_b:
        return STATUS_CONSTANT_MARGIN, 0.0, 0.0, 0.0
    return 0, total, sa / total, sb / total


@numba.njit(cache=True)
def _pearson_r(cross: np.ndarray) -> float:
    sab = 0.0
    saa = 0.0
    sbb = 0.0
    for blk in range(cross.shape[0]):
        sab += cross[blk, 0]
        saa += cross[blk, 1]
        sbb += cross[blk, 2]
    return min(1.0, max(-1.0, sab / math.sqrt(saa * sbb)))


@numba.njit(parallel=True, cache=True)
def pearson(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple[int, float]:
    """Weighted Pearson correlation: ``(status, r)``; ``STATUS_NO_PAIRS`` without weight, ``STATUS_CONSTANT_MARGIN`` for a constant side."""
    n_blocks = _n_blocks(a.size)
    sums = np.zeros((n_blocks, 7))
    for blk in numba.prange(n_blocks):
        _pearson_sums(a, b, w, blk, sums[blk])
    status, _, ma, mb = _pearson_from_partials(sums)
    if status:
        return status, 0.0
    cross = np.zeros((n_blocks, 3))
    for blk in numba.prange(n_blocks):
        _pearson_cross_sums(a, b, w, ma, mb, blk, cross[blk])
    return 0, _pearson_r(cross)


@numba.njit(cache=True)
def pearson_serial(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple[int, float]:
    """``pearson`` on the calling thread (for the draw kernels); the same numbers."""
    n_blocks = _n_blocks(a.size)
    sums = np.zeros((n_blocks, 7))
    for blk in range(n_blocks):
        _pearson_sums(a, b, w, blk, sums[blk])
    status, _, ma, mb = _pearson_from_partials(sums)
    if status:
        return status, 0.0
    cross = np.zeros((n_blocks, 3))
    for blk in range(n_blocks):
        _pearson_cross_sums(a, b, w, ma, mb, blk, cross[blk])
    return 0, _pearson_r(cross)


@numba.njit(cache=True)
def weighted_ranks(order: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Average ranks of the multiset in which pair ``i`` occurs ``w[i]`` times; ``order`` sorts ``v``.

    A tie group of total weight ``W`` after cumulative weight ``C`` holds
    positions ``C+1..C+W``, so every member gets ``C + (W + 1) / 2``. Pairs of
    weight 0 get rank 0 and never enter a weighted statistic.
    """
    rank_sorted = np.empty(v.size)
    sorted_ranks_into(rank_sorted, v[order], w[order])
    rank = np.empty(v.size)
    rank[order] = rank_sorted
    return rank


@numba.njit(cache=True)
def sorted_ranks_into(rank: np.ndarray, v_sorted: np.ndarray, w_sorted: np.ndarray) -> None:
    """``weighted_ranks`` of values already in sorted order, written in that order (``rank[t]`` is position ``t``'s)."""
    n = v_sorted.size
    rank[:] = 0.0
    cum = 0.0
    i = 0
    while i < n:
        value = v_sorted[i]
        j = i
        group = 0.0
        while j < n and v_sorted[j] == value:
            group += w_sorted[j]
            j += 1
        if group > 0.0:
            avg = cum + (group + 1.0) / 2.0
            for t in range(i, j):
                if w_sorted[t] > 0.0:
                    rank[t] = avg
            cum += group
        i = j


@numba.njit(parallel=True, cache=True)
def count_table(
    m_stratum: np.ndarray,
    f_stratum: np.ndarray,
    m: np.ndarray,
    f: np.ndarray,
    w: np.ndarray,
    n_m_strata: int,
    n_f_strata: int,
    k_m: int,
    k_f: int,
) -> np.ndarray:
    """Weighted pair counts by (mother stratum, father stratum, mother level, father level)."""
    n_blocks = _n_blocks(m.size)
    out = np.zeros((n_blocks, n_m_strata, n_f_strata, k_m, k_f))
    for b in numba.prange(n_blocks):
        for i in range(b * _BLOCK, _block_end(m.size, b)):
            out[b, m_stratum[i], f_stratum[i], int(m[i]), int(f[i])] += w[i]
    return _block_sum(out)


@numba.njit(cache=True)
def _margin_sums(code: np.ndarray, stratum: np.ndarray, w: np.ndarray, b: int, out: np.ndarray) -> None:
    for i in range(b * _BLOCK, _block_end(code.size, b)):
        out[stratum[i], int(code[i])] += w[i]


@numba.njit(parallel=True, cache=True)
def margin(code: np.ndarray, stratum: np.ndarray, w: np.ndarray, n_strata: int, k: int) -> np.ndarray:
    """Weighted level counts of one discrete side by (stratum, level)."""
    n_blocks = _n_blocks(code.size)
    out = np.zeros((n_blocks, n_strata, k))
    for b in numba.prange(n_blocks):
        _margin_sums(code, stratum, w, b, out[b])
    return _block_sum(out)


@numba.njit(cache=True)
def margin_serial(code: np.ndarray, stratum: np.ndarray, w: np.ndarray, n_strata: int, k: int) -> np.ndarray:
    """``margin`` on the calling thread (for the draw kernels); the same numbers."""
    n_blocks = _n_blocks(code.size)
    out = np.zeros((n_blocks, n_strata, k))
    for b in range(n_blocks):
        _margin_sums(code, stratum, w, b, out[b])
    return _block_sum(out)


@numba.njit(cache=True)
def thresholds(margin: np.ndarray) -> np.ndarray:
    """Per row of ``margin`` (strata × levels), ``Φ⁻¹`` of the cumulative proportions, framed by -inf and +inf.

    Olsson (1979) eqs 15-18; Olsson, Drasgow & Dorans (1982) eq 36 (F4). A row
    without counts gets all -inf and is never used.
    """
    n_strata, k = margin.shape
    out = np.empty((n_strata, k + 1))
    for s in range(n_strata):
        total = 0.0
        for j in range(k):
            total += margin[s, j]
        out[s, 0] = -math.inf
        out[s, k] = math.inf
        cum = 0.0
        for j in range(k - 1):
            cum += margin[s, j]
            out[s, j + 1] = ndtri(cum / total) if total > 0.0 else -math.inf
    return out


@numba.njit(parallel=True, cache=True)
def polyserial_nll(
    rho: float,
    x: np.ndarray,
    x_stratum: np.ndarray,
    mean: np.ndarray,
    inv_sd: np.ndarray,
    y: np.ndarray,
    y_stratum: np.ndarray,
    tau: np.ndarray,
    w: np.ndarray,
) -> float:
    """Negative conditional log-likelihood ``-Σ w log[Φ(τ*_u) − Φ(τ*_l)]`` (Olsson, Drasgow & Dorans 1982 eqs 19-20).

    ``τ* = (τ − ρz)/√(1−ρ²)`` with ``z`` standardised within ``x_stratum``;
    ``tau`` holds each ``y_stratum``'s thresholds framed by ±inf, where Φ is
    exactly 0 or 1 (``erfc(∓inf)`` is exact).
    """
    n_blocks = _n_blocks(x.size)
    partial = np.zeros((n_blocks, 3))
    for b in numba.prange(n_blocks):
        _polyserial_sums(rho, x, x_stratum, mean, inv_sd, y, y_stratum, tau, w, b, partial[b])
    return _block_sum(partial)[0]


@numba.njit(cache=True)
def _polyserial_sums(
    rho: float,
    x: np.ndarray,
    x_stratum: np.ndarray,
    mean: np.ndarray,
    inv_sd: np.ndarray,
    y: np.ndarray,
    y_stratum: np.ndarray,
    tau: np.ndarray,
    w: np.ndarray,
    b: int,
    out: np.ndarray,
) -> None:
    """Block ``b`` of ``polyserial_terms`` into ``out``: ``(nll, grad, hess)``."""
    s = math.sqrt((1.0 - rho) * (1.0 + rho))
    s3 = s * s * s
    s5 = s3 * s * s
    nll = 0.0
    grad = 0.0
    hess = 0.0
    for i in range(b * _BLOCK, _block_end(x.size, b)):
        if w[i] == 0.0:
            continue
        z = (x[i] - mean[x_stratum[i]]) * inv_sd[x_stratum[i]]
        c = int(y[i])
        t = tau[y_stratum[i], c + 1]
        ts_u = (t - rho * z) / s
        g_u = 0.0
        h_u = 0.0
        if not math.isinf(t):
            pdf = math.exp(-0.5 * ts_u * ts_u) / _SQRT_2PI
            d = (t * rho - z) / s3
            g_u = pdf * d
            h_u = pdf * (-ts_u * d * d + t / s3 + (t * rho - z) * 3.0 * rho / s5)
        t = tau[y_stratum[i], c]
        ts_l = (t - rho * z) / s
        g_l = 0.0
        h_l = 0.0
        if not math.isinf(t):
            pdf = math.exp(-0.5 * ts_l * ts_l) / _SQRT_2PI
            d = (t * rho - z) / s3
            g_l = pdf * d
            h_l = pdf * (-ts_l * d * d + t / s3 + (t * rho - z) * 3.0 * rho / s5)
        p = max(_interval_mass(ts_u, ts_l), _TINY)
        score = (g_u - g_l) / p
        nll -= w[i] * math.log(p)
        grad -= w[i] * score
        hess -= w[i] * ((h_u - h_l) / p - score * score)
    out[0] = nll
    out[1] = grad
    out[2] = hess


@numba.njit(parallel=True, cache=True)
def polyserial_terms(
    rho: float,
    x: np.ndarray,
    x_stratum: np.ndarray,
    mean: np.ndarray,
    inv_sd: np.ndarray,
    y: np.ndarray,
    y_stratum: np.ndarray,
    tau: np.ndarray,
    w: np.ndarray,
) -> tuple[float, float, float]:
    """``polyserial_nll`` with its first and second ρ-derivatives in the same pass.

    Per pair, ``∂P/∂ρ = φ(τ*_u) d_u − φ(τ*_l) d_l`` with ``d = (τρ − z)/(1−ρ²)^{3/2}``
    (eq 26), and ``∂²P/∂ρ² = Σ ±φ(τ*)(−τ* d² + τ/(1−ρ²)^{3/2} + 3ρ(τρ − z)/(1−ρ²)^{5/2})``.
    """
    n_blocks = _n_blocks(x.size)
    partial = np.zeros((n_blocks, 3))
    for b in numba.prange(n_blocks):
        _polyserial_sums(rho, x, x_stratum, mean, inv_sd, y, y_stratum, tau, w, b, partial[b])
    nll, grad, hess = _block_sum(partial)
    return nll, grad, hess


@numba.njit(cache=True)
def polyserial_terms_serial(
    rho: float,
    x: np.ndarray,
    x_stratum: np.ndarray,
    mean: np.ndarray,
    inv_sd: np.ndarray,
    y: np.ndarray,
    y_stratum: np.ndarray,
    tau: np.ndarray,
    w: np.ndarray,
) -> tuple[float, float, float]:
    """``polyserial_terms`` on the calling thread (for the draw kernels); the same numbers."""
    n_blocks = _n_blocks(x.size)
    partial = np.zeros((n_blocks, 3))
    for b in range(n_blocks):
        _polyserial_sums(rho, x, x_stratum, mean, inv_sd, y, y_stratum, tau, w, b, partial[b])
    nll, grad, hess = _block_sum(partial)
    return nll, grad, hess


# ---------------------------------------------------------------------------
# Per-pair influence on ρ̂ for the cluster-robust two-step sandwich
# ---------------------------------------------------------------------------
#
# With stacked estimating equations Σ_i ψ_i(θ, ρ) = 0 whose nuisance block
# does not involve ρ, the influence of pair i on ρ̂ is
#     IF_i = -A_ρρ⁻¹ (ψ_ρ,i − A_ρθ A_θθ⁻¹ ψ_θ,i),
# A = Σ_i ∂ψ_i/∂(θ, ρ). Every nuisance equation here is one stratum's mean,
# 1/N variance or one threshold, so A_θθ is diagonal and A_θθ⁻¹ ψ_θ,i is a
# per-pair scalar per parameter. Each kernel sums A_ρθ in a first loop and
# emits IF_i in a second; Σ IF_i = 0 and Var(ρ̂) is the clustered sum of squares.


@numba.njit(cache=True)
def _polyserial_pair(rho: float, z: float, tau_u: float, tau_l: float) -> tuple[float, float, float, float, float]:
    """One pair's eq 26 score and its derivatives in ``z``, ``τ_u``, ``τ_l`` and ρ.

    Returns ``(score, ∂score/∂z, ∂score/∂τ_u, ∂score/∂τ_l, ∂score/∂ρ)``. With
    ``A = (τ − ρz)/s``, ``s = √(1−ρ²)``, ``A_ρ = (ρτ − z)/s³`` and ``g = φ(A_u)A_ρu − φ(A_l)A_ρl``:
    ``score = g / p``, ``∂φ(A)/∂z = φ(A)·Aρ/s``, ``∂A_ρ/∂z = −1/s³``,
    ``∂φ(A)/∂τ = −φ(A)·A/s``, ``∂A_ρ/∂τ = ρ/s³``. An infinite threshold
    contributes nothing (φ = 0, Φ exact).
    """
    s = math.sqrt((1.0 - rho) * (1.0 + rho))
    s3 = s * s * s
    s5 = s3 * s * s
    a_u = (tau_u - rho * z) / s
    a_l = (tau_l - rho * z) / s
    pdf_u = g_u = h_u = gz_u = gt_u = 0.0
    if not math.isinf(tau_u):
        pdf_u = math.exp(-0.5 * a_u * a_u) / _SQRT_2PI
        d = (tau_u * rho - z) / s3
        g_u = pdf_u * d
        h_u = pdf_u * (-a_u * d * d + tau_u / s3 + (tau_u * rho - z) * 3.0 * rho / s5)
        gz_u = pdf_u * (a_u * rho / s * d - 1.0 / s3)
        gt_u = pdf_u * (-a_u / s * d + rho / s3)
    pdf_l = g_l = h_l = gz_l = gt_l = 0.0
    if not math.isinf(tau_l):
        pdf_l = math.exp(-0.5 * a_l * a_l) / _SQRT_2PI
        d = (tau_l * rho - z) / s3
        g_l = pdf_l * d
        h_l = pdf_l * (-a_l * d * d + tau_l / s3 + (tau_l * rho - z) * 3.0 * rho / s5)
        gz_l = pdf_l * (a_l * rho / s * d - 1.0 / s3)
        gt_l = pdf_l * (-a_l / s * d + rho / s3)
    p = max(_interval_mass(a_u, a_l), _TINY)
    score = (g_u - g_l) / p
    dp_dz = -(rho / s) * (pdf_u - pdf_l)
    ds_dz = (gz_u - gz_l) / p - score * dp_dz / p
    ds_dtu = gt_u / p - score * (pdf_u / s) / p
    ds_dtl = -gt_l / p + score * (pdf_l / s) / p
    ds_drho = (h_u - h_l) / p - score * score
    return score, ds_dz, ds_dtu, ds_dtl, ds_drho


@numba.njit(parallel=True, cache=True)
def polyserial_influence(
    rho: float,
    x: np.ndarray,
    x_stratum: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    y: np.ndarray,
    y_stratum: np.ndarray,
    tau: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Per-pair influence on the two-step polyserial ρ̂ and ``A_ρρ`` (the summed ∂score/∂ρ, negative at a maximum).

    Nuisance equations per ``x`` stratum ``s``: ``Σ w (x − μ_s) = 0`` and
    ``Σ w ((x − μ_s)² − v_s) = 0``; per ``y`` stratum ``t`` and finite threshold
    ``j``: ``Σ w (1[c ≤ j−1] − Φ(τ_tj)) = 0`` (ODD 1982 eq 36). ``∂score/∂μ_s =
    −∂score/∂z / √v_s`` and ``∂score/∂v_s = −z ∂score/∂z / (2 v_s)``. A pair of
    weight 0 has influence 0.
    """
    n_xs = mean.size
    n_ys, k1 = tau.shape
    inv_sd = np.zeros(n_xs)
    for s in range(n_xs):
        if var[s] > 0.0:
            inv_sd[s] = 1.0 / math.sqrt(var[s])
    cdf_tau = np.zeros((n_ys, k1))
    pdf_tau = np.zeros((n_ys, k1))
    for t in range(n_ys):
        for j in range(1, k1 - 1):
            if not math.isinf(tau[t, j]):
                cdf_tau[t, j] = ndtr(tau[t, j])
                pdf_tau[t, j] = math.exp(-0.5 * tau[t, j] * tau[t, j]) / _SQRT_2PI
    n_blocks = _n_blocks(x.size)
    n_x_b = np.zeros((n_blocks, n_xs))
    n_y_b = np.zeros((n_blocks, n_ys))
    a_mu_b = np.zeros((n_blocks, n_xs))
    a_var_b = np.zeros((n_blocks, n_xs))
    a_tau_b = np.zeros((n_blocks, n_ys, k1))
    a_rr_b = np.zeros((n_blocks, 1))
    for b in numba.prange(n_blocks):
        _polyserial_jacobian_sums(
            rho,
            x,
            x_stratum,
            mean,
            inv_sd,
            y,
            y_stratum,
            tau,
            w,
            b,
            n_x_b[b],
            n_y_b[b],
            a_mu_b[b],
            a_var_b[b],
            a_tau_b[b],
            a_rr_b[b],
        )
    n_x = _block_sum(n_x_b)
    n_y = _block_sum(n_y_b)
    a_mu = _block_sum(a_mu_b)
    a_var = _block_sum(a_var_b)
    a_tau = _block_sum(a_tau_b)
    a_rr = _block_sum(a_rr_b)[0]
    out = np.zeros(x.size)
    if not a_rr < 0.0:
        return out, a_rr
    for b in numba.prange(n_blocks):
        for i in range(b * _BLOCK, _block_end(x.size, b)):
            if w[i] == 0.0:
                continue
            s = x_stratum[i]
            t = y_stratum[i]
            c = int(y[i])
            dev = x[i] - mean[s]
            z = dev * inv_sd[s]
            score, _, _, _, _ = _polyserial_pair(rho, z, tau[t, c + 1], tau[t, c])
            correction = -(a_mu[s] * dev + a_var[s] * (dev * dev - var[s])) / n_x[s]
            for j in range(1, k1 - 1):
                if pdf_tau[t, j] > 0.0:
                    below = 1.0 if c <= j - 1 else 0.0
                    correction -= a_tau[t, j] * (below - cdf_tau[t, j]) / (n_y[t] * pdf_tau[t, j])
            out[i] = -(score - correction) / a_rr
    return out, a_rr


@numba.njit(cache=True)
def _polyserial_jacobian_sums(
    rho: float,
    x: np.ndarray,
    x_stratum: np.ndarray,
    mean: np.ndarray,
    inv_sd: np.ndarray,
    y: np.ndarray,
    y_stratum: np.ndarray,
    tau: np.ndarray,
    w: np.ndarray,
    b: int,
    n_x: np.ndarray,
    n_y: np.ndarray,
    a_mu: np.ndarray,
    a_var: np.ndarray,
    a_tau: np.ndarray,
    a_rr: np.ndarray,
) -> None:
    """Block ``b`` of the ``A_ρθ`` and ``A_ρρ`` sums of ``polyserial_influence``, into the given partials."""
    for i in range(b * _BLOCK, _block_end(x.size, b)):
        if w[i] == 0.0:
            continue
        s = x_stratum[i]
        t = y_stratum[i]
        c = int(y[i])
        z = (x[i] - mean[s]) * inv_sd[s]
        _, ds_dz, ds_dtu, ds_dtl, ds_drho = _polyserial_pair(rho, z, tau[t, c + 1], tau[t, c])
        n_x[s] += w[i]
        n_y[t] += w[i]
        a_mu[s] -= w[i] * ds_dz * inv_sd[s]
        a_var[s] -= w[i] * ds_dz * z * inv_sd[s] * inv_sd[s] / 2.0
        a_tau[t, c + 1] += w[i] * ds_dtu
        a_tau[t, c] += w[i] * ds_dtl
        a_rr[0] += w[i] * ds_drho


@numba.njit(parallel=True, cache=True)
def pearson_influence(
    a: np.ndarray,
    b: np.ndarray,
    a_stratum: np.ndarray,
    b_stratum: np.ndarray,
    mean_a: np.ndarray,
    var_a: np.ndarray,
    mean_b: np.ndarray,
    var_b: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Per-pair influence on the Pearson correlation of ``a`` and ``b``, each standardised within its own stratum.

    Estimating equations: each stratum's mean and 1/N variance, then
    ``Σ w (z_a z_b − ρ) = 0``, so ``A_ρρ = −N``, ``∂(z_a z_b)/∂μ_s = −z_b/√v_s``
    and ``∂(z_a z_b)/∂v_s = −z_a z_b/(2 v_s)``. With one stratum per side this is
    the classic ``(z_a z_b − ρ(z_a² + z_b²)/2) / N``.
    """
    n_a, n_b = mean_a.size, mean_b.size
    inv_sd_a = np.zeros(n_a)
    inv_sd_b = np.zeros(n_b)
    for s in range(n_a):
        if var_a[s] > 0.0:
            inv_sd_a[s] = 1.0 / math.sqrt(var_a[s])
    for t in range(n_b):
        if var_b[t] > 0.0:
            inv_sd_b[t] = 1.0 / math.sqrt(var_b[t])
    n_blocks = _n_blocks(a.size)
    scalars_b = np.zeros((n_blocks, 2))
    count_a_b = np.zeros((n_blocks, n_a))
    count_b_b = np.zeros((n_blocks, n_b))
    a_mu_a_b = np.zeros((n_blocks, n_a))
    a_var_a_b = np.zeros((n_blocks, n_a))
    a_mu_b_b = np.zeros((n_blocks, n_b))
    a_var_b_b = np.zeros((n_blocks, n_b))
    for blk in numba.prange(n_blocks):
        _pearson_jacobian_sums(
            a,
            b,
            a_stratum,
            b_stratum,
            mean_a,
            inv_sd_a,
            mean_b,
            inv_sd_b,
            w,
            blk,
            scalars_b[blk],
            count_a_b[blk],
            count_b_b[blk],
            a_mu_a_b[blk],
            a_var_a_b[blk],
            a_mu_b_b[blk],
            a_var_b_b[blk],
        )
    total, cross = _block_sum(scalars_b)
    count_a = _block_sum(count_a_b)
    count_b = _block_sum(count_b_b)
    a_mu_a = _block_sum(a_mu_a_b)
    a_var_a = _block_sum(a_var_a_b)
    a_mu_b = _block_sum(a_mu_b_b)
    a_var_b = _block_sum(a_var_b_b)
    rho = cross / total
    out = np.zeros(a.size)
    for blk in numba.prange(n_blocks):
        for i in range(blk * _BLOCK, _block_end(a.size, blk)):
            if w[i] == 0.0:
                continue
            s = a_stratum[i]
            t = b_stratum[i]
            dev_a = a[i] - mean_a[s]
            dev_b = b[i] - mean_b[t]
            psi = dev_a * inv_sd_a[s] * dev_b * inv_sd_b[t] - rho
            correction = -(a_mu_a[s] * dev_a + a_var_a[s] * (dev_a * dev_a - var_a[s])) / count_a[s]
            correction -= (a_mu_b[t] * dev_b + a_var_b[t] * (dev_b * dev_b - var_b[t])) / count_b[t]
            out[i] = (psi - correction) / total
    return out


@numba.njit(cache=True)
def _pearson_jacobian_sums(
    a: np.ndarray,
    b: np.ndarray,
    a_stratum: np.ndarray,
    b_stratum: np.ndarray,
    mean_a: np.ndarray,
    inv_sd_a: np.ndarray,
    mean_b: np.ndarray,
    inv_sd_b: np.ndarray,
    w: np.ndarray,
    blk: int,
    scalars: np.ndarray,
    count_a: np.ndarray,
    count_b: np.ndarray,
    a_mu_a: np.ndarray,
    a_var_a: np.ndarray,
    a_mu_b: np.ndarray,
    a_var_b: np.ndarray,
) -> None:
    """Block ``blk`` of the ``A_ρθ`` sums of ``pearson_influence``; ``scalars`` is ``(Σw, Σw z_a z_b)``."""
    for i in range(blk * _BLOCK, _block_end(a.size, blk)):
        if w[i] == 0.0:
            continue
        s = a_stratum[i]
        t = b_stratum[i]
        za = (a[i] - mean_a[s]) * inv_sd_a[s]
        zb = (b[i] - mean_b[t]) * inv_sd_b[t]
        scalars[0] += w[i]
        scalars[1] += w[i] * za * zb
        count_a[s] += w[i]
        count_b[t] += w[i]
        a_mu_a[s] -= w[i] * zb * inv_sd_a[s]
        a_var_a[s] -= w[i] * za * zb * inv_sd_a[s] * inv_sd_a[s] / 2.0
        a_mu_b[t] -= w[i] * za * inv_sd_b[t]
        a_var_b[t] -= w[i] * za * zb * inv_sd_b[t] * inv_sd_b[t] / 2.0


@numba.njit(parallel=True, cache=True)
def gather_influence(
    influence: np.ndarray, m_stratum: np.ndarray, f_stratum: np.ndarray, m: np.ndarray, f: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Each pair's entry of a per-(strata, levels) ``influence`` table; 0 for a pair of weight 0."""
    out = np.zeros(m.size)
    for i in numba.prange(m.size):
        if w[i] != 0.0:
            out[i] = influence[m_stratum[i], f_stratum[i], int(m[i]), int(f[i])]
    return out


# ---------------------------------------------------------------------------
# Score-statistic permutation test (plan v3, decision 17)
# ---------------------------------------------------------------------------
#
# At ρ = 0 the bivariate normal factorises, φ2(a, b) = φ(a)φ(b), and the ρ-score
# of every latent-correlation log-likelihood collapses to one sum over pairs,
#     U(0) = Σ_i e_m,i e_f,i,
# with e the conditional mean of a side's latent variable given its observed
# value: z itself for a continuous side (Olsson, Drasgow & Dorans 1982 eq 26 at
# ρ = 0) and (φ(τ_c) − φ(τ_{c+1})) / (Φ(τ_{c+1}) − Φ(τ_c)) for level c of a
# discrete side (Olsson 1979 eq 9 at ρ = 0). Continuous × continuous gives N·r.
# Mothers never move under the permutation null, so their e is computed once;
# the father side (margins, thresholds, moments) is refit in every draw.

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_TWO_POW_MINUS_53 = 2.0**-53


@numba.njit(cache=True)
def _mix64(z: np.uint64) -> np.uint64:
    """SplitMix64's output function, a bijection on 64-bit words."""
    z = (z ^ (z >> np.uint64(30))) * _MIX_1
    z = (z ^ (z >> np.uint64(27))) * _MIX_2
    return z ^ (z >> np.uint64(31))


@numba.njit(cache=True)
def stream_state(seed: int, draw: int) -> np.uint64:
    """The SplitMix64 state that starts draw ``draw`` of ``seed``: ``mix(mix(seed) + draw)``.

    Two draws start at two unrelated points of the Weyl sequence, so their
    streams overlap with probability about ``steps · draws / 2⁶⁴``.
    """
    return _mix64(_mix64(np.uint64(seed)) + np.uint64(draw))


@numba.njit(cache=True)
def next_below(state: np.uint64, n: int) -> tuple[np.uint64, int]:
    """Advance the SplitMix64 ``state`` and return it with a uniform integer in ``[0, n)``.

    The top 53 bits of the output scale a double in ``[0, 1)`` by ``n``; the bias
    of floor(u·n) against a uniform draw is at most ``n / 2⁵³``.
    """
    state += _GOLDEN
    u = float(_mix64(state) >> np.uint64(11)) * _TWO_POW_MINUS_53
    return state, min(int(u * n), n - 1)


@numba.njit(cache=True)
def shuffle_blocks(donor: np.ndarray, by_block: np.ndarray, block_start: np.ndarray, seed: int, draw: int) -> None:
    """Fill ``donor`` with draw ``draw`` of ``seed``: a uniform permutation of the fathers within each block.

    ``by_block`` lists the fathers block by block and ``block_start`` the
    offsets of each block in it (``n_blocks + 1`` entries). Fisher-Yates per
    block; a father alone in his block consumes no random number.
    """
    state = stream_state(seed, draw)
    for b in range(block_start.size - 1):
        lo, hi = block_start[b], block_start[b + 1]
        for t in range(lo, hi):
            donor[by_block[t]] = by_block[t]
        for t in range(hi - lo - 1, 0, -1):
            state, j = next_below(state, t + 1)
            a, c = by_block[lo + t], by_block[lo + j]
            donor[a], donor[c] = donor[c], donor[a]


@numba.njit(cache=True)
def latent_means(margin: np.ndarray) -> np.ndarray:
    """Per (stratum, level) of ``margin``: ``E[η | level]`` under the thresholds of that stratum's margin.

    ``(φ(τ_c) − φ(τ_{c+1})) / p_c`` with ``p_c`` the level's share of its
    stratum, which is ``Φ(τ_{c+1}) − Φ(τ_c)`` by construction of the thresholds;
    0 at an empty level.
    """
    n_strata, k = margin.shape
    tau = thresholds(margin)
    out = np.zeros((n_strata, k))
    for s in range(n_strata):
        total = 0.0
        for c in range(k):
            total += margin[s, c]
        for c in range(k):
            if margin[s, c] > 0.0:
                pdf_lo = math.exp(-0.5 * tau[s, c] * tau[s, c]) / _SQRT_2PI
                pdf_hi = math.exp(-0.5 * tau[s, c + 1] * tau[s, c + 1]) / _SQRT_2PI
                out[s, c] = (pdf_lo - pdf_hi) * total / margin[s, c]
    return out


@numba.njit(cache=True)
def _margin_status(margin: np.ndarray, levels: np.ndarray) -> int:
    """``_check_levels`` of the estimators: a shown level with no count fails before a constant margin does."""
    n_strata, k = margin.shape
    for s in range(n_strata):
        populated = False
        for c in range(k):
            populated = populated or margin[s, c] > 0.0
        if populated:
            for c in range(k):
                if levels[s, c] and margin[s, c] == 0.0:
                    return STATUS_EMPTY_CATEGORY
    n_levels = 0
    for c in range(k):
        for s in range(n_strata):
            if margin[s, c] > 0.0:
                n_levels += 1
                break
    return STATUS_CONSTANT_MARGIN if n_levels < 2 else 0


@numba.njit(cache=True)
def shuffle_index(order: np.ndarray, block_start: np.ndarray, seed: int, draw: int) -> None:
    """Apply draw ``draw`` of ``seed`` to ``order``, row indices of fathers listed block by block.

    ``shuffle_blocks`` with ``by_block`` the identity: starting from
    ``order[r] = r``, row ``r`` ends up taking the vector of row ``order[r]``,
    the donor ``shuffle_blocks`` gives father ``by_block[r]``. Shuffling 4-byte
    indices and gathering the rows afterwards keeps the random swaps in a
    quarter of the memory that swapping the rows themselves would touch.
    """
    state = stream_state(seed, draw)
    for b in range(block_start.size - 1):
        lo, hi = block_start[b], block_start[b + 1]
        for t in range(lo, hi):
            order[t] = t
        for t in range(hi - lo - 1, 0, -1):
            state, j = next_below(state, t + 1)
            order[lo + t], order[lo + j] = order[lo + j], order[lo + t]


@numba.njit(cache=True)
def _continuous_cell(
    values: np.ndarray,
    father: np.ndarray,
    count: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    run_start: np.ndarray,
    run_stratum: np.ndarray,
    first_run: int,
    end_run: int,
    sums: np.ndarray,
    n: int,
) -> tuple[float, float, float, float, int, int]:
    """Crude and stratified ``Σ e_m e_f`` with the fathers' values standardised on the permuted pairs.

    One pass over the cell's distinct fathers: father ``j`` stands for
    ``count[j]`` pairs whose mother scores sum to ``e1[j]`` (crude) and
    ``e2[j]`` (stratified). Sums are taken about the stratum's observed mean
    (``sums[:, 0]``) so the variances need no second pass; ``sums[:, 1:]`` are
    the stratum's pair count and mother-score sums. Returns
    ``(cross_crude, cross_strat, ss_f_crude, ss_f_strat, status_crude, status_strat)``;
    ``Σ e_f²`` is the pair count on both forms. A constant side is
    ``constant_margin``; a constant stratum is ``degenerate_stratum``.
    """
    n_strata = sums.shape[0]
    dev = np.zeros(n_strata)
    sq = np.zeros(n_strata)
    cross1 = np.zeros(n_strata)
    cross2 = np.zeros(n_strata)
    lo = np.full(n_strata, math.inf)
    hi = np.full(n_strata, -math.inf)
    for r in range(first_run, end_run):
        s = run_stratum[r]
        shift = sums[s, 0]
        d_sum = 0.0
        sq_sum = 0.0
        c1 = 0.0
        c2 = 0.0
        v_lo = math.inf
        v_hi = -math.inf
        for j in range(run_start[r], run_start[r + 1]):
            v = values[father[j]]
            d = v - shift
            d_sum += count[j] * d
            sq_sum += count[j] * d * d
            c1 += e1[j] * d
            c2 += e2[j] * d
            v_lo = min(v_lo, v)
            v_hi = max(v_hi, v)
        dev[s] += d_sum
        sq[s] += sq_sum
        cross1[s] += c1
        cross2[s] += c2
        lo[s] = min(lo[s], v_lo)
        hi[s] = max(hi[s], v_hi)
    if lo.min() == hi.max():
        return 0.0, 0.0, 0.0, 0.0, STATUS_CONSTANT_MARGIN, STATUS_CONSTANT_MARGIN
    pooled_mean = 0.0
    degenerate = False
    for s in range(n_strata):
        if sums[s, 1] > 0.0:
            pooled_mean += sums[s, 0] * sums[s, 1] + dev[s]
            degenerate = degenerate or lo[s] == hi[s]
    pooled_mean /= n
    ss_pooled = 0.0
    cross_pooled = 0.0
    cross_strat = 0.0
    for s in range(n_strata):
        if sums[s, 1] > 0.0:
            delta = sums[s, 0] - pooled_mean
            ss_pooled += sq[s] + delta * (2.0 * dev[s] + delta * sums[s, 1])
            cross_pooled += cross1[s] + delta * sums[s, 2]
            if not degenerate:
                mean_dev = dev[s] / sums[s, 1]
                ss_s = sq[s] - mean_dev * dev[s]
                cross_strat += (cross2[s] - mean_dev * sums[s, 3]) / math.sqrt(ss_s / sums[s, 1])
    cross_crude = cross_pooled / math.sqrt(ss_pooled / n)
    if degenerate:
        return cross_crude, 0.0, float(n), 0.0, 0, STATUS_DEGENERATE_STRATUM
    return cross_crude, cross_strat, float(n), float(n), 0, 0


@numba.njit(cache=True)
def _discrete_cell(
    values: np.ndarray,
    father: np.ndarray,
    count: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    run_start: np.ndarray,
    run_stratum: np.ndarray,
    first_run: int,
    end_run: int,
    levels: np.ndarray,
) -> tuple[float, float, float, float, int, int]:
    """Crude and stratified ``Σ e_m e_f`` with the fathers' thresholds refit on the permuted pairs' margins.

    One pass collects, per stratum and level, the pair count and the mother
    scores; the latent means of the refit margins then weight those sums.
    Returns ``(cross_crude, cross_strat, ss_f_crude, ss_f_strat, status_crude, status_strat)``
    with ``ss_f = Σ_pairs e_f²``.
    """
    n_strata, k = levels.shape
    margin = np.zeros((n_strata, k))
    score1 = np.zeros((n_strata, k))
    score2 = np.zeros((n_strata, k))
    for r in range(first_run, end_run):
        s = run_stratum[r]
        for j in range(run_start[r], run_start[r + 1]):
            c = int(values[father[j]])
            margin[s, c] += count[j]
            score1[s, c] += e1[j]
            score2[s, c] += e2[j]
    pooled = np.zeros((1, k))
    pooled_levels = np.zeros((1, k), dtype=np.bool_)
    for s in range(n_strata):
        for c in range(k):
            pooled[0, c] += margin[s, c]
            pooled_levels[0, c] = pooled_levels[0, c] or levels[s, c]
    status_crude = _margin_status(pooled, pooled_levels)
    status_strat = _margin_status(margin, levels)
    e_crude = latent_means(pooled)
    e_strat = latent_means(margin)
    ss_crude = 0.0
    ss_strat = 0.0
    cross_crude = 0.0
    cross_strat = 0.0
    for c in range(k):
        ss_crude += pooled[0, c] * e_crude[0, c] * e_crude[0, c]
        level_score = 0.0
        for s in range(n_strata):
            ss_strat += margin[s, c] * e_strat[s, c] * e_strat[s, c]
            level_score += score1[s, c]
            cross_strat += score2[s, c] * e_strat[s, c]
        cross_crude += level_score * e_crude[0, c]
    return cross_crude, cross_strat, ss_crude, ss_strat, status_crude, status_strat


@numba.njit(cache=True)
def row_statistics(
    arranged: np.ndarray,
    cells: np.ndarray,
    cell_trait: np.ndarray,
    cell_runs: np.ndarray,
    father: np.ndarray,
    count: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    run_start: np.ndarray,
    run_stratum: np.ndarray,
    strata_start: np.ndarray,
    sums: np.ndarray,
    n_pairs: np.ndarray,
    k_f: np.ndarray,
    levels_start: np.ndarray,
    f_levels: np.ndarray,
    cross: np.ndarray,
    ss_f: np.ndarray,
    status: np.ndarray,
) -> None:
    """The crude and stratified statistic of each cell in ``cells`` for one arrangement of the fathers' values, into ``cross[c, form]`` etc.

    ``arranged[t, r]`` is trait ``t`` of the father at row ``r`` (one contiguous
    column per trait, so a cell streams only its father trait). Cells not in
    ``cells`` are left as they are.

    Cell ``c`` owns runs ``cell_runs[c]:cell_runs[c + 1]`` (a run is a stretch of
    its distinct fathers in one stratum) and strata rows
    ``strata_start[c]:strata_start[c + 1]`` of ``sums``.
    """
    for i in range(cells.size):
        c = cells[i]
        s0, s1 = strata_start[c], strata_start[c + 1]
        if k_f[c] == 0:
            out = _continuous_cell(
                arranged[cell_trait[c]],
                father,
                count,
                e1,
                e2,
                run_start,
                run_stratum,
                cell_runs[c],
                cell_runs[c + 1],
                sums[s0:s1],
                n_pairs[c],
            )
        else:
            levels = f_levels[levels_start[c] : levels_start[c + 1]].reshape(s1 - s0, k_f[c])
            out = _discrete_cell(
                arranged[cell_trait[c]],
                father,
                count,
                e1,
                e2,
                run_start,
                run_stratum,
                cell_runs[c],
                cell_runs[c + 1],
                levels,
            )
        cross[c, 0], cross[c, 1], ss_f[c, 0], ss_f[c, 1], status[c, 0], status[c, 1] = out


@numba.njit(cache=True)
def arrangement_statistics(
    arrangements: np.ndarray,
    cell_trait: np.ndarray,
    cell_runs: np.ndarray,
    father: np.ndarray,
    count: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    run_start: np.ndarray,
    run_stratum: np.ndarray,
    strata_start: np.ndarray,
    sums: np.ndarray,
    n_pairs: np.ndarray,
    k_f: np.ndarray,
    levels_start: np.ndarray,
    f_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(cross, ss_f, status)``, each ``(draws, cells, 2)``, for given arrangements ``(draws, traits, fathers)``.

    The observed statistic is the unpermuted rows; a test can feed an exact
    enumeration. Column 0 is the crude form, column 1 the stratified one.
    """
    n_draws, n_cells = arrangements.shape[0], cell_trait.size
    cross = np.zeros((n_draws, n_cells, 2))
    ss_f = np.zeros((n_draws, n_cells, 2))
    status = np.zeros((n_draws, n_cells, 2), dtype=np.int64)
    cells = np.arange(n_cells)
    for d in range(n_draws):
        row_statistics(
            arrangements[d],
            cells,
            cell_trait,
            cell_runs,
            father,
            count,
            e1,
            e2,
            run_start,
            run_stratum,
            strata_start,
            sums,
            n_pairs,
            k_f,
            levels_start,
            f_levels,
            cross[d],
            ss_f[d],
            status[d],
        )
    return cross, ss_f, status


@numba.njit(parallel=True, cache=True)
def permutation_statistics(
    seed: int,
    first_draw: int,
    end_draw: int,
    cells: np.ndarray,
    trait_needed: np.ndarray,
    block_start: np.ndarray,
    rows: np.ndarray,
    order: np.ndarray,
    arranged: np.ndarray,
    cross: np.ndarray,
    ss_f: np.ndarray,
    status: np.ndarray,
    cell_trait: np.ndarray,
    cell_runs: np.ndarray,
    father: np.ndarray,
    count: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    run_start: np.ndarray,
    run_stratum: np.ndarray,
    strata_start: np.ndarray,
    sums: np.ndarray,
    n_pairs: np.ndarray,
    k_f: np.ndarray,
    levels_start: np.ndarray,
    f_levels: np.ndarray,
) -> None:
    """``row_statistics`` of ``cells`` for draws ``first_draw <= d < end_draw``: within-block permutations of ``seed`` of the fathers' ``rows``.

    Writes ``cross[d]``, ``ss_f[d]`` and ``status[d]``. Worker ``w`` takes
    draws ``first_draw + w, first_draw + w + W, …`` (``W = order.shape[0]``),
    shuffles ``order[w]`` and gathers the permuted rows of the traits in
    ``trait_needed`` into ``arranged[w]`` (traits x fathers). Draw ``d`` is
    ``shuffle_index(…, seed, d)`` whatever the worker count or the draw range,
    and every cell sees the same arrangement in a draw.
    """
    n_workers = order.shape[0]
    for w in numba.prange(n_workers):
        for d in range(first_draw + w, end_draw, n_workers):
            shuffle_index(order[w], block_start, seed, d)
            for r in range(rows.shape[0]):
                for q in range(rows.shape[1]):
                    if trait_needed[q]:
                        arranged[w, q, r] = rows[order[w, r], q]
            row_statistics(
                arranged[w],
                cells,
                cell_trait,
                cell_runs,
                father,
                count,
                e1,
                e2,
                run_start,
                run_stratum,
                strata_start,
                sums,
                n_pairs,
                k_f,
                levels_start,
                f_levels,
                cross[d],
                ss_f[d],
                status[d],
            )


# ---------------------------------------------------------------------------
# One-step Mate-Network bootstrap (plan v3, decision 16)
# ---------------------------------------------------------------------------
#
# A draw resamples the cell's G Mate Networks with replacement, and a pair's
# weight is the multiplicity of its network. Closed-form estimators are
# recomputed exactly on the weighted sample. A latent correlation refits its
# first step exactly (stratum moments, margins, thresholds) and takes one Newton
# step for ρ from the observed ρ̂ with the draw-weighted score and the
# full-sample Hessian at ρ̂:
#     ρ_d = ρ̂ − S_w(ρ̂) / H(ρ̂).
# The full-sample Hessian (the linearised bootstrap) tracks the full refit about
# four times more closely than the draw's own Hessian, whose draws shift with
# the score (development measurement over 4 seeds x 7 latent cells: worst CI
# bound difference 0.0006 vs 0.0024 at n = 5000, 0.0012 vs 0.0048 at 2500),
# and costs nothing per draw.
# Draws are keyed by (seed, draw) in a stream domain of their own, so they are
# independent of the permutations and identical for any thread count.

#: ρ is searched, and a one-step draw clipped, to (-LATENT_BOUND, LATENT_BOUND).
LATENT_BOUND = 0.9999
_BOOTSTRAP_DOMAIN = np.uint64(0xD1B54A32D192ED03)


@numba.njit(cache=True)
def bootstrap_state(seed: int, draw: int) -> np.uint64:
    """``stream_state`` of the bootstrap domain: the seed is XOR-ed with a constant first."""
    return _mix64(_mix64(np.uint64(seed) ^ _BOOTSTRAP_DOMAIN) + np.uint64(draw))


@numba.njit(cache=True)
def network_weights(w: np.ndarray, mult: np.ndarray, labels: np.ndarray, seed: int, draw: int) -> None:
    """Fill ``w`` with draw ``draw`` of ``seed``: ``G`` networks drawn with replacement, each pair weighted by its network's multiplicity.

    ``mult`` is scratch of size ``G`` (the networks' multiplicities, left filled).
    """
    g = mult.size
    mult[:] = 0
    state = bootstrap_state(seed, draw)
    for _ in range(g):
        state, j = next_below(state, g)
        mult[j] += 1
    for i in range(labels.size):
        w[i] = mult[labels[i]]


@numba.njit(cache=True)
def _any_degenerate(lo: np.ndarray, hi: np.ndarray) -> bool:
    """Whether some present stratum is constant (an absent one has ``lo = +inf``, ``hi = -inf``)."""
    return bool((lo == hi).any())


@numba.njit(cache=True)
def _continuous_status(total: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> int:
    """The polyserial first step's checks of the continuous side: no weight, constant, or a constant stratum."""
    present = False
    lo_min = math.inf
    hi_max = -math.inf
    for s in range(total.size):
        if total[s] > 0.0:
            present = True
            lo_min = min(lo_min, lo[s])
            hi_max = max(hi_max, hi[s])
    if not present:
        return STATUS_NO_PAIRS
    if lo_min == hi_max:
        return STATUS_CONSTANT_MARGIN
    if _any_degenerate(lo, hi):
        return STATUS_DEGENERATE_STRATUM
    return 0


@numba.njit(cache=True)
def _inverse_sd(var: np.ndarray) -> np.ndarray:
    out = np.zeros(var.size)
    for s in range(var.size):
        if var[s] > 0.0:
            out[s] = 1.0 / math.sqrt(var[s])
    return out


@numba.njit(cache=True)
def _clip_rho(rho: float) -> float:
    return min(max(rho, -LATENT_BOUND), LATENT_BOUND)


@numba.njit(cache=True)
def _polyserial_form(
    rho: float,
    hess: float,
    x: np.ndarray,
    x_stratum: np.ndarray,
    total: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    y: np.ndarray,
    y_stratum: np.ndarray,
    margin: np.ndarray,
    levels: np.ndarray,
    w: np.ndarray,
) -> tuple[int, float]:
    """One form's one-step polyserial draw, ``(status, ρ_d)``, from the draw's first step and the full-sample ``hess``.

    ``total``…``hi`` are the ``x`` moments per ``x`` stratum and ``margin`` the
    ``y`` counts per ``y`` stratum; the crude form passes the pooled values in
    every row. Checks run in the order of the estimator's first step; a
    ``hess`` that is not positive makes every draw ``STATUS_NONCONCAVE``.
    """
    status = _continuous_status(total, lo, hi)
    if status:
        return status, math.nan
    status = _margin_status(margin, levels)
    if status:
        return status, math.nan
    if not hess > 0.0:
        return STATUS_NONCONCAVE, math.nan
    _, grad, _ = polyserial_terms_serial(rho, x, x_stratum, mean, _inverse_sd(var), y, y_stratum, thresholds(margin), w)
    return 0, _clip_rho(rho - grad / hess)


@numba.njit(cache=True)
def _pooled_moments(
    total: np.ndarray, mean: np.ndarray, var: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[float, float, float, float, float]:
    """The moments of every stratum taken together: ``(total, mean, 1/N variance, min, max)``."""
    t = 0.0
    m = 0.0
    for s in range(total.size):
        t += total[s]
        m += total[s] * mean[s]
    if t > 0.0:
        m /= t
    v = 0.0
    for s in range(total.size):
        if total[s] > 0.0:
            d = mean[s] - m
            v += total[s] * (var[s] + d * d)
    if t > 0.0:
        v /= t
    return t, m, v, lo.min(), hi.max()


@numba.njit(parallel=True, cache=True)
def polyserial_draws(
    seed: int,
    n_draws: int,
    labels: np.ndarray,
    mult: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    x_stratum: np.ndarray,
    n_xs: int,
    y: np.ndarray,
    y_stratum: np.ndarray,
    y_levels: np.ndarray,
    rho_crude: float,
    hess_crude: float,
    rho_strat: float,
    hess_strat: float,
    point_biserial: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap draws of a continuous × discrete cell: ``(values, status)``, each ``(n_draws, 3)``.

    Column 0 is the crude one-step draw from ``rho_crude`` with the full-sample
    Hessian ``hess_crude``, column 2 the stratified one (a NaN ``rho`` skips the
    column), column 1 the weighted Pearson of ``x`` and ``y`` when
    ``point_biserial``. Worker ``k`` takes draws ``k, k + W, …`` with scratch
    ``mult[k]`` (``G``) and ``w[k]`` (pairs).
    """
    n_ys, k = y_levels.shape
    values = np.full((n_draws, 3), math.nan)
    status = np.zeros((n_draws, 3), dtype=np.int64)
    pooled_levels = np.zeros((n_ys, k), dtype=np.bool_)
    for c in range(k):
        shown = False
        for s in range(n_ys):
            shown = shown or y_levels[s, c]
        for s in range(n_ys):
            pooled_levels[s, c] = shown
    n_workers = mult.shape[0]
    for worker in numba.prange(n_workers):
        for d in range(worker, n_draws, n_workers):
            wd = w[worker]
            network_weights(wd, mult[worker], labels, seed, d)
            total, mean, var, lo, hi = stratum_moments_serial(x, x_stratum, wd, n_xs)
            margin_s = margin_serial(y, y_stratum, wd, n_ys, k)
            if not math.isnan(rho_crude):
                t, m, v, lo_all, hi_all = _pooled_moments(total, mean, var, lo, hi)
                pooled_margin = np.zeros((n_ys, k))
                for c in range(k):
                    count = 0.0
                    for s in range(n_ys):
                        count += margin_s[s, c]
                    for s in range(n_ys):
                        pooled_margin[s, c] = count
                status[d, 0], values[d, 0] = _polyserial_form(
                    rho_crude,
                    hess_crude,
                    x,
                    x_stratum,
                    np.full(n_xs, t),
                    np.full(n_xs, m),
                    np.full(n_xs, v),
                    np.full(n_xs, lo_all),
                    np.full(n_xs, hi_all),
                    y,
                    y_stratum,
                    pooled_margin,
                    pooled_levels,
                    wd,
                )
            if point_biserial:
                status[d, 1], values[d, 1] = pearson_serial(x, y, wd)
            if not math.isnan(rho_strat):
                status[d, 2], values[d, 2] = _polyserial_form(
                    rho_strat, hess_strat, x, x_stratum, total, mean, var, lo, hi, y, y_stratum, margin_s, y_levels, wd
                )
    return values, status


@numba.njit(cache=True)
def _spearman_sorted(
    m_sorted: np.ndarray, w_sorted: np.ndarray, rank_f: np.ndarray, f_pos: np.ndarray
) -> tuple[int, float]:
    """Weighted Pearson of the ranks of ``m`` and ``f``, walked in ``m``'s sorted order.

    ``m_sorted`` and ``w_sorted`` are the values and weights in that order,
    ``rank_f`` the father ranks in ``f``'s sorted order (``sorted_ranks_into``)
    and ``f_pos[q]`` the position there of the pair at ``m``-sorted position
    ``q``, the one random access per pair. Both rank sets have mean ``(T + 1) / 2``
    exactly (``T = Σw``), so the sums are centred there and lose nothing to
    cancellation.
    """
    t = 0.0
    for i in range(w_sorted.size):
        t += w_sorted[i]
    if t == 0.0:
        return STATUS_NO_PAIRS, 0.0
    centre = (t + 1.0) / 2.0
    sxy = 0.0
    sxx = 0.0
    syy = 0.0
    cum = 0.0
    n = m_sorted.size
    i = 0
    while i < n:
        value = m_sorted[i]
        j = i
        group = 0.0
        while j < n and m_sorted[j] == value:
            group += w_sorted[j]
            j += 1
        if group > 0.0:
            dx = cum + (group + 1.0) / 2.0 - centre
            for q in range(i, j):
                if w_sorted[q] > 0.0:
                    dy = rank_f[f_pos[q]] - centre
                    sxy += w_sorted[q] * dx * dy
                    sxx += w_sorted[q] * dx * dx
                    syy += w_sorted[q] * dy * dy
            cum += group
        i = j
    if sxx == 0.0 or syy == 0.0:
        return STATUS_CONSTANT_MARGIN, 0.0
    return 0, min(1.0, max(-1.0, sxy / math.sqrt(sxx * syy)))


@numba.njit(cache=True)
def _stratified_pearson(
    m: np.ndarray,
    f: np.ndarray,
    m_stratum: np.ndarray,
    f_stratum: np.ndarray,
    n_m: int,
    n_f: int,
    w: np.ndarray,
) -> tuple[int, float]:
    """Weighted Pearson of ``m`` and ``f`` each standardised within its own stratum, with the estimator's checks."""
    total_m, mean_m, var_m, lo_m, hi_m = stratum_moments_serial(m, m_stratum, w, n_m)
    if total_m.sum() == 0.0:
        return STATUS_NO_PAIRS, 0.0
    if _any_degenerate(lo_m, hi_m):
        return STATUS_DEGENERATE_STRATUM, 0.0
    _, mean_f, var_f, lo_f, hi_f = stratum_moments_serial(f, f_stratum, w, n_f)
    if _any_degenerate(lo_f, hi_f):
        return STATUS_DEGENERATE_STRATUM, 0.0
    inv_m = _inverse_sd(var_m)
    inv_f = _inverse_sd(var_f)
    sxy = 0.0
    sxx = 0.0
    syy = 0.0
    for i in range(m.size):
        if w[i] > 0.0:
            zm = (m[i] - mean_m[m_stratum[i]]) * inv_m[m_stratum[i]]
            zf = (f[i] - mean_f[f_stratum[i]]) * inv_f[f_stratum[i]]
            sxy += w[i] * zm * zf
            sxx += w[i] * zm * zm
            syy += w[i] * zf * zf
    return 0, min(1.0, max(-1.0, sxy / math.sqrt(sxx * syy)))


@numba.njit(parallel=True, cache=True)
def continuous_draws(
    seed: int,
    n_draws: int,
    labels: np.ndarray,
    mult: np.ndarray,
    w: np.ndarray,
    w_sorted: np.ndarray,
    rank_f: np.ndarray,
    m: np.ndarray,
    f: np.ndarray,
    m_stratum: np.ndarray,
    f_stratum: np.ndarray,
    n_m: int,
    n_f: int,
    m_sorted: np.ndarray,
    f_sorted: np.ndarray,
    labels_m_sorted: np.ndarray,
    labels_f_sorted: np.ndarray,
    f_pos: np.ndarray,
    stratified: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap draws of a continuous × continuous cell: ``(values, status)``, each ``(n_draws, 3)``.

    Columns: Pearson, Spearman, and (when ``stratified``) Pearson of the
    stratum-standardised sides, all exact on the weighted draw. The Spearman
    pass walks the pre-sorted copies ``m_sorted`` / ``f_sorted`` (with the
    network labels in the same orders) so that a draw streams them
    sequentially; ``f_pos`` maps ``m``-sorted to ``f``-sorted positions.
    ``w``, ``w_sorted`` and ``rank_f`` are per-worker scratch over the pairs.
    """
    values = np.full((n_draws, 3), math.nan)
    status = np.zeros((n_draws, 3), dtype=np.int64)
    n_workers = mult.shape[0]
    for worker in numba.prange(n_workers):
        for d in range(worker, n_draws, n_workers):
            wd = w[worker]
            ws = w_sorted[worker]
            network_weights(wd, mult[worker], labels, seed, d)
            status[d, 0], values[d, 0] = pearson_serial(m, f, wd)
            for i in range(ws.size):
                ws[i] = mult[worker, labels_f_sorted[i]]
            sorted_ranks_into(rank_f[worker], f_sorted, ws)
            for i in range(ws.size):
                ws[i] = mult[worker, labels_m_sorted[i]]
            status[d, 1], values[d, 1] = _spearman_sorted(m_sorted, ws, rank_f[worker], f_pos)
            if stratified:
                status[d, 2], values[d, 2] = _stratified_pearson(m, f, m_stratum, f_stratum, n_m, n_f, wd)
    return values, status


@numba.njit(parallel=True, cache=True)
def table_draws(
    seed: int,
    first_draw: int,
    end_draw: int,
    labels: np.ndarray,
    mult: np.ndarray,
    w: np.ndarray,
    m_stratum: np.ndarray,
    f_stratum: np.ndarray,
    m: np.ndarray,
    f: np.ndarray,
    n_m_strata: int,
    n_f_strata: int,
    k_m: int,
    k_f: int,
) -> np.ndarray:
    """The weighted count tables of draws ``first_draw <= d < end_draw`` of a discrete × discrete cell: ``(draws, m strata, f strata, k_m, k_f)``."""
    out = np.zeros((end_draw - first_draw, n_m_strata, n_f_strata, k_m, k_f))
    n_workers = mult.shape[0]
    for worker in numba.prange(n_workers):
        for d in range(first_draw + worker, end_draw, n_workers):
            wd = w[worker]
            network_weights(wd, mult[worker], labels, seed, d)
            for i in range(m.size):
                out[d - first_draw, m_stratum[i], f_stratum[i], int(m[i]), int(f[i])] += wd[i]
    return out
