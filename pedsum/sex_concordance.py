"""Offspring Sex Concordance under a pooled fixed-margin exchangeability null.

Asks whether resolved offspring sex is more or less concordant *within*
**Offspring Groups** than pooled exchangeability predicts, for three canonical
groupings (**Sibship**, **Maternal Offspring Group**, **Paternal Offspring
Group**). The statistic is the total number of concordant within-group
**Individual Pairs**::

    C = sum over groups g of (choose(M_g, 2) + choose(F_g, 2))

conditioned on the eligible group sizes and on the global male/female totals.
Exact conditional moments, with ``[x]_r`` a falling factorial, ``N = M + F``,
``P = sum_g choose(n_g, 2)``, ``S = sum_g 3*choose(n_g, 3)`` (unordered pairs
of concordance indicators sharing one offspring) and ``D = choose(P, 2) - S``
(disjoint indicator pairs)::

    q2  = ([M]_2 + [F]_2) / [N]_2
    q3  = ([M]_3 + [F]_3) / [N]_3
    q22 = ([M]_4 + [F]_4 + 2[M]_2[F]_2) / [N]_4
    E[C]   = P*q2
    Var(C) = P*q2*(1-q2) + 2*S*(q3-q2^2) + 2*D*(q22-q2^2)

Moments are combined as exact rationals (:mod:`fractions`) and converted to
float once, so the ``Var == 0`` degeneracy test is exact rather than a float64
near-miss. Exactness here is not a speed tradeoff — the integer path measures
*faster* than float64 for the per-group sums.

**Provenance is the headline eligibility axis.** ``pedsum.validate`` resolves
missing sex only for individuals used as a parent, so admitting imputed sex
makes eligibility conditional on having reproduced. When group-level selection
rates vary that is not absorbed by fixed-margin conditioning and the test
becomes wildly anti-conservative, with *identical* provenance counts to the
harmless uniform-rate case. The headline therefore uses ``sex_source ==
"input"`` only; all-resolved sex is reported as a sensitivity block.

The analytical p-value is an asymptotic normal approximation: sound at
conventional levels at every dominance level, but up to ~18x too liberal in the
far tail (and entirely in the *positive* tail — spurious over-concordance).
``max_group_pair_share`` predicts the effect; any claim at ``p < 0.01`` needs
permutations.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from pedsum.base import SEX_MALE, logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

# Grouping name → the parent column(s) that both define the group key and must
# be known for an offspring to belong to it. Order is the emit order.
GROUPINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sibship", ("mother", "father")),
    ("maternal_offspring_group", ("mother",)),
    ("paternal_offspring_group", ("father",)),
)

#: A group informs concordance only from this many eligible offspring up.
MIN_GROUP_SIZE = 2

#: ``sex_source`` values admitted by the headline analysis.
HEADLINE_SEX_SOURCES: tuple[str, ...] = ("input",)

#: ``sex_source`` values admitted by the all-resolved sensitivity analysis.
ALL_RESOLVED_SEX_SOURCES: tuple[str, ...] = (
    "input",
    "imputed_from_missing",
    "imputed_from_role",
)

#: Analytical p below this needs permutations before it may be claimed.
ANALYTICAL_P_WARN = 0.01

# Sampler size bounds, per backend. These are *different* limits with different
# causes, and neither is the ``P``-based overflow bound guarded in
# ``_pairs_and_triples`` — do not merge them.
#
# NumPy: ``Generator.multivariate_hypergeometric`` with the default "marginals"
# method documents ``sum(colors) < 10**9`` to avoid loss of precision.
_NUMPY_SAMPLER_MAX_N = 10**9
# Numba: its ``np.random.hypergeometric`` routes through float64 internally and
# starts returning zeros above the 2**53 exact-integer ceiling (measured: fine
# at 1e15, broken at 1e16). No 10**9 restriction.
_NUMBA_SAMPLER_MAX_N = 2**53

_SEX_SOURCE_ORDER: tuple[str, ...] = (
    "input",
    "imputed_from_missing",
    "imputed_from_role",
    "unresolved",
)
_SEX_SOURCE_CODE = {name: i for i, name in enumerate(_SEX_SOURCE_ORDER)}


# ---------------------------------------------------------------------------
# Exact integer / rational primitives
# ---------------------------------------------------------------------------


def _pairs_and_triples(sizes: np.ndarray, n_offspring: int) -> tuple[int, int]:
    """Return ``(P, S)`` for group ``sizes``, as Python ``int``.

    ``P = sum_g choose(n_g, 2)`` counts within-group **Individual Pairs**;
    ``S = sum_g 3*choose(n_g, 3) = sum_g n(n-1)(n-2)/2`` counts unordered pairs
    of concordance indicators sharing one offspring.

    ``>>1`` rather than ``//2``: integer *division* was the only slow part of
    the exact path, and the shift makes it faster than float64. Both shifted
    quantities are even by construction, and neither can go negative.

    ``P <= choose(N, 2)`` so it is int64-safe for any real pedigree. ``S`` is
    not: the vectorized path is taken only when the **pre-shift** intermediate
    ``n(n-1)(n-2) <= N * n_max^2`` fits int64. The guard carries no ``// 2`` —
    a halved guard admits a dominant group of ``n_max ~2.15M`` whose pre-shift
    product wraps int64 while the halved sum still looks safe. At ``N = 10M``
    the exact fallback takes over once ``n_max > ~0.96M``.

    Args:
        sizes: Eligible group sizes; integer dtype, each ``>= 0``.
        n_offspring: ``N``, the total number of eligible offspring.

    Returns:
        ``(P, S)`` as Python ``int`` — callers form ``choose(P, 2)`` from
        ``P``, which overflows int64 above ``P ~ 4.3e9``.
    """
    if sizes.size == 0:
        return 0, 0
    n = sizes.astype(np.int64, copy=False)
    a = n * (n - 1)  # always even
    n_pairs = int((a >> 1).sum())
    n_max = int(n.max())
    if n_offspring * n_max * n_max < 2**63:
        n_shared = int(((a * (n - 2)) >> 1).sum())
    else:
        n_shared = sum(int(x) * (int(x) - 1) * (int(x) - 2) // 2 for x in n)
    return n_pairs, n_shared


def _falling_factorial(x: int, r: int) -> int:
    """Return ``[x]_r = x(x-1)...(x-r+1)`` as an exact Python ``int``."""
    out = 1
    for k in range(r):
        out *= x - k
    return out


@dataclass(frozen=True)
class Moments:
    """Exact conditional moments of ``C`` under the fixed-margin null."""

    n_pairs: int
    """``P``: within-group Individual Pairs."""

    n_shared_index_pairs: int
    """``S``: unordered indicator pairs sharing one offspring."""

    n_disjoint_pairs: int
    """``D``: unordered indicator pairs over four distinct offspring."""

    expected: Fraction
    """``E[C]``, exact."""

    variance: Fraction
    """``Var(C)``, exact — so the zero-variance skip is a real equality."""


def analytical_moments(sizes: np.ndarray, n_male: int, n_female: int) -> Moments:
    """Exact conditional moments of ``C`` for group ``sizes`` and margins.

    Combines everything as :class:`fractions.Fraction`. In float64 a
    single-group pedigree at ``N = 10M`` returns ``Var = -9.997e9`` instead of
    ``0``, so a float guard never fires and ``z`` comes out NaN or garbage.

    Args:
        sizes: Eligible group sizes (each ``>= MIN_GROUP_SIZE``).
        n_male: ``M``, eligible males across all groups.
        n_female: ``F``, eligible females across all groups.

    Returns:
        The :class:`Moments` for this configuration.
    """
    n_total = n_male + n_female
    n_pairs, n_shared = _pairs_and_triples(sizes, n_total)
    # P and S are Python ints, so choose(P, 2) cannot wrap int64 — a single
    # sire with ~93,000 offspring already pushes P past 4.3e9, and uniform
    # groups of 900 at N=10M get there with no dominant individual at all.
    n_disjoint = n_pairs * (n_pairs - 1) // 2 - n_shared
    if n_total < 2 or n_pairs == 0:
        return Moments(n_pairs, n_shared, n_disjoint, Fraction(0), Fraction(0))

    q2 = Fraction(
        _falling_factorial(n_male, 2) + _falling_factorial(n_female, 2),
        _falling_factorial(n_total, 2),
    )
    variance = n_pairs * q2 * (1 - q2)
    if n_shared:  # S > 0 implies some group has n >= 3, so [N]_3 > 0
        q3 = Fraction(
            _falling_factorial(n_male, 3) + _falling_factorial(n_female, 3),
            _falling_factorial(n_total, 3),
        )
        variance += 2 * n_shared * (q3 - q2 * q2)
    if n_disjoint:  # D > 0 needs four distinct offspring, so [N]_4 > 0
        q22 = Fraction(
            _falling_factorial(n_male, 4)
            + _falling_factorial(n_female, 4)
            + 2 * _falling_factorial(n_male, 2) * _falling_factorial(n_female, 2),
            _falling_factorial(n_total, 4),
        )
        variance += 2 * n_disjoint * (q22 - q2 * q2)
    return Moments(n_pairs, n_shared, n_disjoint, n_pairs * q2, variance)


def concordant_pairs(male_counts: np.ndarray, sizes: np.ndarray) -> int:
    """Return ``C``, the observed concordant within-group Individual Pairs."""
    male = male_counts.astype(np.int64, copy=False)
    female = sizes.astype(np.int64, copy=False) - male
    return int(((male * (male - 1) + female * (female - 1)) >> 1).sum())


def _two_sided_normal_p(z: float) -> float:
    """Two-sided normal-approximation p-value for a z statistic."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment across the available grouping hypotheses.

    Valid under arbitrary dependence, and conservative here: Sibship pairs are
    a subset of both the maternal and paternal pair sets, so these are not
    three independent hypotheses.

    Args:
        pvalues: Raw p-values keyed by grouping name. Keys with a ``None``
            value are dropped from the family rather than counted in ``m``.

    Returns:
        Adjusted p-values, keyed as the input; monotone non-decreasing in the
        raw p ordering and clipped to ``1.0``.
    """
    items = sorted(((k, p) for k, p in pvalues.items() if p is not None), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


# ---------------------------------------------------------------------------
# Permutation samplers (NumPy fallback + optional compiled sampler)
# ---------------------------------------------------------------------------


def sample_concordance_numpy(
    sizes: np.ndarray,
    n_male: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Draw ``n_permutations`` values of ``C`` under the fixed-margin null.

    Production fallback *and* the oracle the compiled sampler is tested
    against. ``multivariate_hypergeometric(colors=sizes, nsample=M)`` is
    exactly the fixed-margin allocation: draw the ``M`` males from an urn whose
    colors are the groups, ``n_g`` balls each. Its default "marginals" method
    is the sequential conditional-hypergeometric chain, run in C.

    Draws are taken one at a time. Batching into ``B x G`` blocks buys nothing
    — the cost is real ``O(B*G)`` sampling work, not call overhead (measured
    0.97x at ``B=100, G=200k``) — while allocating 160 MB there.

    Args:
        sizes: Eligible group sizes.
        n_male: ``M``, held fixed across permutations.
        n_permutations: ``B``, the number of draws.
        seed: Seed for a fresh :class:`numpy.random.Generator`.

    Returns:
        ``(B,)`` int64 array of permuted ``C`` values.
    """
    rng = np.random.default_rng(seed)
    colors = sizes.astype(np.int64, copy=False)
    out = np.empty(n_permutations, dtype=np.int64)
    for b in range(n_permutations):
        male = rng.multivariate_hypergeometric(colors, n_male)
        female = colors - male
        out[b] = ((male * (male - 1) + female * (female - 1)) >> 1).sum()
    return out


def _sequential_concordance(
    sizes: np.ndarray,
    n_male: int,
    n_total: int,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Sequential conditional-hypergeometric sampler; the numba jit target.

    Walks the groups in canonical order, drawing each group's male count from
    the hypergeometric conditional on what the earlier groups took. Equivalent
    to permuting individual labels, in ``O(G)`` memory.

    Do **not** call this uncompiled outside tests: it seeds NumPy's legacy
    *global* RNG, which is what makes it jittable in the first place.
    """
    np.random.seed(seed)
    out = np.empty(n_permutations, dtype=np.int64)
    n_groups = sizes.shape[0]
    for b in range(n_permutations):
        remaining_male = n_male
        remaining_total = n_total
        concordant = 0
        for g in range(n_groups):
            size = sizes[g]
            male = np.random.hypergeometric(remaining_male, remaining_total - remaining_male, size)
            female = size - male
            concordant += (male * (male - 1) + female * (female - 1)) // 2
            remaining_male -= male
            remaining_total -= size
        out[b] = concordant
    return out


_NUMBA_SAMPLER: object | None = None
_NUMBA_PROBED = False


def load_numba_sampler():
    """Return the compiled sequential sampler, or ``None`` when unavailable.

    Numba is a **soft import**, deliberately absent from ``pyproject.toml``.
    It measures 1.8x over the NumPy sampler at ``G=200k`` and 2.8x at ``G=2M``
    (79.5 ms vs 224.2 ms per draw) — real, but not enough to justify pinning a
    compiled dependency, and constraining NumPy resolution for every pedsum
    user, for an opt-in path inside an opt-in feature. The NumPy sampler has to
    exist anyway as the test oracle, so the fallback is free, and numba arrives
    transitively via pedigree-graph for most installs.

    The probe result is memoised, so a missing numba costs one failed import.
    """
    global _NUMBA_SAMPLER, _NUMBA_PROBED  # noqa: PLW0603 - one-shot import probe cache
    if _NUMBA_PROBED:
        return _NUMBA_SAMPLER
    _NUMBA_PROBED = True
    try:
        import numba
    except ImportError:
        logger.debug("numba unavailable; sex-concordance permutations use the NumPy sampler")
        return None
    _NUMBA_SAMPLER = numba.njit(cache=True)(_sequential_concordance)
    return _NUMBA_SAMPLER


def _sampler_bound(backend: str) -> int:
    """Return the maximum eligible ``N`` the named backend can sample."""
    return _NUMBA_SAMPLER_MAX_N if backend == "numba" else _NUMPY_SAMPLER_MAX_N


def _run_permutations(
    sizes: np.ndarray,
    n_male: int,
    n_total: int,
    observed: int,
    expected: float,
    n_permutations: int,
    seed: int,
) -> dict:
    """Run the permutation calibration and return its metadata block.

    Two-sided by absolute deviation from ``E[C]``, matching the analytical
    test, with the finite-Monte-Carlo p-value ``(b+1)/(B+1)`` so it is never
    zero. Records the backend: with a soft-imported numba the same seed gives a
    different p depending on whether numba is installed, so the reproducibility
    guarantee is "given (seed, backend)", not seed alone. The difference is
    well inside Monte Carlo error (se ~= 0.007 at p ~= 0.05, B = 1000).
    """
    sampler = load_numba_sampler()
    backend = "numba" if sampler is not None else "numpy"
    block: dict = {
        "requested": int(n_permutations),
        "completed": 0,
        "seed": int(seed),
        "backend": backend,
        "p_raw": None,
        "skip_reason": None,
    }
    bound = _sampler_bound(backend)
    if n_total >= bound:
        block["skip_reason"] = f"sampler_size_limit_{backend}"
        logger.warning(
            "sex-concordance permutations skipped: %s eligible offspring reaches the %s sampler's "
            "documented bound (%s); analytical results are still reported",
            f"{n_total:,}",
            backend,
            f"{bound:,}",
        )
        return block

    if sampler is not None:
        draws = sampler(sizes.astype(np.int64, copy=False), int(n_male), int(n_total), int(n_permutations), int(seed))
    else:
        draws = sample_concordance_numpy(sizes, int(n_male), int(n_permutations), int(seed))

    # Relax the threshold by a relative epsilon so a draw that mirrors the
    # observation across E[C] counts as extreme even when float rounding puts
    # it a few ulps inside. Ties resolving *outward* inflates p, which is the
    # safe direction for a p-value.
    observed_deviation = abs(observed - expected) * (1.0 - 1e-12)
    n_extreme = int((np.abs(draws.astype(np.float64) - expected) >= observed_deviation).sum())
    block["completed"] = int(n_permutations)
    block["p_raw"] = (n_extreme + 1) / (n_permutations + 1)
    return block


# ---------------------------------------------------------------------------
# Group projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _GroupIndex:
    """Canonically sorted offspring groups for one grouping.

    Built once per grouping over every offspring whose required parent(s) are
    known, then reused for both the headline and the sensitivity provenance
    mask — the group key does not depend on which sexes are admitted.
    """

    group_sizes: np.ndarray
    """Total offspring per group, before any sex-provenance filtering."""

    group_starts: np.ndarray
    """Index of each group's first sorted offspring; the ``reduceat`` offsets."""

    group_of_row: np.ndarray
    """Group ordinal of each sorted offspring."""

    is_male: np.ndarray
    """Sorted male indicator (int64 0/1); meaningless where sex is unresolved."""

    sex_source_code: np.ndarray
    """Sorted ``sex_source`` codes, indices into ``_SEX_SOURCE_ORDER``."""


def _build_group_index(keys: tuple[np.ndarray, ...], is_male: np.ndarray, sex_source_code: np.ndarray) -> _GroupIndex:
    """Sort offspring into canonical group order and record group boundaries.

    Sorting by the group key makes results independent of input row order (any
    valid pedigree row order gives the same permutation stream), which is why
    the sampler can consume ``group_sizes`` positionally.
    """
    n_rows = keys[0].shape[0]
    if n_rows == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return _GroupIndex(empty_i, empty_i, empty_i, empty_i, np.empty(0, dtype=np.int8))
    # lexsort takes the primary key last, so reverse: (mother, father) sorts by
    # mother then father.
    order = np.lexsort(keys[::-1])
    boundary = np.zeros(n_rows, dtype=bool)
    boundary[0] = True
    for key in keys:
        sorted_key = key[order]
        boundary[1:] |= sorted_key[1:] != sorted_key[:-1]
    starts = np.flatnonzero(boundary)
    group_sizes = np.diff(np.append(starts, n_rows)).astype(np.int64)
    group_of_row = np.repeat(np.arange(starts.size, dtype=np.int64), group_sizes)
    return _GroupIndex(
        group_sizes=group_sizes,
        group_starts=starts.astype(np.int64),
        group_of_row=group_of_row,
        is_male=is_male[order].astype(np.int64),
        sex_source_code=sex_source_code[order],
    )


def _eligibility_counts(index: _GroupIndex, admitted: np.ndarray) -> dict:
    """Reduce an admitted-offspring mask to per-group counts and count metadata.

    ``admitted`` marks offspring whose sex is resolved *and* whose provenance
    the analysis admits. Eligible members of a group are retained even when
    other members are ineligible; a group informs concordance only from
    ``MIN_GROUP_SIZE`` admitted members up.
    """
    n_groups = index.group_sizes.size
    if n_groups == 0:
        zeros = np.empty(0, dtype=np.int64)
        return {
            "sizes": zeros,
            "male_counts": zeros,
            "eligible_row": np.empty(0, dtype=bool),
            "counts": {
                "n_groups_total": 0,
                "n_groups_eligible": 0,
                "n_groups_incomplete": 0,
                "n_groups_too_small": 0,
                "n_offspring_total": 0,
                "n_offspring_eligible": 0,
                "n_offspring_sex_excluded": 0,
                "n_offspring_in_small_groups": 0,
            },
        }
    admitted_i = admitted.astype(np.int64)
    admitted_per_group = np.add.reduceat(admitted_i, index.group_starts)
    male_per_group = np.add.reduceat(admitted_i * index.is_male, index.group_starts)
    keep = admitted_per_group >= MIN_GROUP_SIZE
    n_admitted = int(admitted_i.sum())
    n_eligible = int(admitted_per_group[keep].sum())
    n_offspring = int(index.group_sizes.sum())
    return {
        "sizes": admitted_per_group[keep],
        "male_counts": male_per_group[keep],
        "eligible_row": admitted & keep[index.group_of_row],
        "counts": {
            "n_groups_total": int(n_groups),
            "n_groups_eligible": int(keep.sum()),
            "n_groups_incomplete": int((keep & (admitted_per_group < index.group_sizes)).sum()),
            "n_groups_too_small": int((~keep).sum()),
            "n_offspring_total": n_offspring,
            "n_offspring_eligible": n_eligible,
            "n_offspring_sex_excluded": n_offspring - n_admitted,
            "n_offspring_in_small_groups": n_admitted - n_eligible,
        },
    }


# ---------------------------------------------------------------------------
# Descriptive breakdowns
# ---------------------------------------------------------------------------


def _proportion_distribution(values: np.ndarray) -> dict:
    """Unweighted distribution of per-group male proportions.

    Same key set as ``sections._numeric_distribution`` so the safe-attempt
    extrema drop recognises it. ``nz`` counts groups holding at least one male.
    """
    n = values.size
    if n == 0:
        return dict.fromkeys(("mean", "std", "min", "q1", "median", "q3", "max"), 0.0) | {"nz": 0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if n > 1 else 0.0,
        "min": float(values.min()),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
        "nz": int((values != 0).sum()),
    }


def _by_group_size(sizes: np.ndarray, male_counts: np.ndarray, q2: float) -> list[dict]:
    """Non-inferential per-group-size rows covering every observed size.

    ``expected_pair_concordance`` is the pooled null ``q2``, identical on every
    row — the rows exist so the reader can see *where* the excess sits, not so
    each size can be tested.
    """
    rows: list[dict] = []
    for size in np.unique(sizes):
        mask = sizes == size
        n_groups = int(mask.sum())
        male = male_counts[mask]
        female = int(size) - male
        observed = int(((male * (male - 1) + female * (female - 1)) >> 1).sum())
        total_pairs = n_groups * (int(size) * (int(size) - 1) // 2)
        observed_frac = observed / total_pairs
        rows.append(
            {
                "group_size": int(size),
                "n_groups": n_groups,
                "n_offspring": n_groups * int(size),
                "observed_pair_concordance": observed_frac,
                "expected_pair_concordance": q2,
                "excess_concordance": observed_frac - q2,
            },
        )
    return rows


# ---------------------------------------------------------------------------
# One analysis pass
# ---------------------------------------------------------------------------

_NO_INPUT_SEX_MESSAGE = (
    "no eligible offspring carry input sex; every resolved sex in this pedigree is "
    "imputed, and pedsum imputes sex only for individuals used as a parent — so "
    "eligibility would be conditional on having reproduced, which makes the "
    "statistic uninterpretable"
)

_SKIP_MESSAGES = {
    "no_input_sex": _NO_INPUT_SEX_MESSAGE,
    "fewer_than_two_eligible_groups": (
        f"fewer than two groups reached {MIN_GROUP_SIZE} eligible offspring; the fixed-margin null is degenerate"
    ),
    "single_sex": "every eligible offspring has the same sex; the fixed-margin null is degenerate",
    "zero_variance": "the exact conditional variance of C is zero; no inference is possible",
}


def _analyse(index: _GroupIndex, admitted: np.ndarray, *, require_input_sex: bool) -> dict:
    """Eligibility counts, the pair-concordance statistic, and analytical inference.

    Shared by the headline (``input`` provenance only) and the all-resolved
    sensitivity pass; the two differ only in ``admitted``.
    """
    projection = _eligibility_counts(index, admitted)
    sizes = projection["sizes"]
    male_counts = projection["male_counts"]
    counts = projection["counts"]

    by_source = dict.fromkeys(ALL_RESOLVED_SEX_SOURCES, 0)
    eligible_row = projection["eligible_row"]
    if eligible_row.size:
        codes = index.sex_source_code[eligible_row]
        for name in ALL_RESOLVED_SEX_SOURCES:
            by_source[name] = int((codes == _SEX_SOURCE_CODE[name]).sum())

    n_male = int(male_counts.sum())
    n_total = int(sizes.sum())
    n_female = n_total - n_male
    moments = analytical_moments(sizes, n_male, n_female)
    observed = concordant_pairs(male_counts, sizes)
    n_pairs = moments.n_pairs

    out: dict = {
        "computed": False,
        "skip_reason": None,
        "skip_message": None,
        **counts,
        "eligible_by_sex_source": by_source,
        "conditioning_male_fraction": (n_male / n_total) if n_total else None,
        "n_within_group_pairs": n_pairs,
        "max_group_pair_share": None,
        "observed_pair_concordance": (observed / n_pairs) if n_pairs else None,
        "expected_pair_concordance": None,
        "excess_concordance": None,
        "direction": None,
        "z": None,
        "p_analytical": None,
    }
    if sizes.size:
        n_max = int(sizes.max())
        out["max_group_pair_share"] = (n_max * (n_max - 1) // 2) / n_pairs if n_pairs else None

    # Offspring admitted by provenance, before the min-group-size cut — so a
    # pedigree whose only input-sex offspring sit in singleton groups reports
    # the group-size skip rather than the (wrong) provenance one.
    n_admitted = counts["n_offspring_eligible"] + counts["n_offspring_in_small_groups"]

    skip = None
    if require_input_sex and counts["n_offspring_total"] > 0 and n_admitted == 0:
        skip = "no_input_sex"
    elif counts["n_groups_eligible"] < 2:
        skip = "fewer_than_two_eligible_groups"
    elif n_male == 0 or n_female == 0:
        skip = "single_sex"
    elif moments.variance == 0:
        skip = "zero_variance"
    if skip is not None:
        out["skip_reason"] = skip
        out["skip_message"] = _SKIP_MESSAGES[skip]
        return out

    expected = moments.expected
    z = (observed - float(expected)) / math.sqrt(float(moments.variance))
    out.update(
        {
            "computed": True,
            "expected_pair_concordance": float(expected / n_pairs),
            # Exact comparison: the direction never hinges on float rounding.
            "excess_concordance": float((Fraction(observed) - expected) / n_pairs),
            "direction": (
                "over_concordant"
                if Fraction(observed) > expected
                else ("under_concordant" if Fraction(observed) < expected else "none")
            ),
            "z": z,
            "p_analytical": _two_sided_normal_p(z),
        },
    )
    out["_moments"] = moments
    out["_observed"] = observed
    out["_sizes"] = sizes
    out["_n_male"] = n_male
    out["_male_counts"] = male_counts
    return out


def _strip_internals(block: dict) -> dict:
    """Drop the underscore-prefixed carriers that never reach the report."""
    return {k: v for k, v in block.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@contextmanager
def _null_timer(label: str) -> Iterator[None]:
    """No-op stand-in for the CLI's ``_timed`` phase logger; ``label`` is ignored."""
    del label
    yield


def compute_offspring_sex_concordance(
    df: pd.DataFrame,
    *,
    permutations: int = 0,
    seed: int = 0,
    timer=None,
) -> dict:
    """Offspring Sex Concordance across the three canonical groupings.

    Reuses the validated ``sex`` / ``sex_source`` columns; never re-decodes or
    independently imputes sex. Allocation stays linear in rows and groups, and
    no per-individual column is added.

    Args:
        df: Validated pedigree frame carrying ``mother``, ``father``, ``sex``
            and ``sex_source`` (``-1`` is the missing-parent sentinel).
        permutations: ``B``, permutation draws for the headline analysis of
            each grouping. ``0`` runs the analytical test only.
        seed: Permutation seed; meaningful only when ``permutations > 0``.
        timer: Optional ``label -> context manager`` used to attribute phases
            (the CLI passes its ``_timed``).

    Returns:
        The ``offspring_sex_concordance`` report section: a ``null_model``
        block plus one block per grouping. Aggregate only — no per-parent,
        per-Mating-Pair or per-Sibship record is emitted.
    """
    timed = timer or _null_timer
    # ``sex_source`` already encodes resolution — validate.py stamps every
    # still-unknown row "unresolved" in its final pass — so admitting by
    # provenance admits only rows whose sex is resolved.
    is_male = (df["sex"].to_numpy() == SEX_MALE).astype(np.int64)
    source_raw = df["sex_source"].to_numpy()
    sex_source_code = np.full(len(df), _SEX_SOURCE_CODE["unresolved"], dtype=np.int8)
    for name, code in _SEX_SOURCE_CODE.items():
        sex_source_code[source_raw == name] = code

    headline_codes = np.array(sorted(_SEX_SOURCE_CODE[s] for s in HEADLINE_SEX_SOURCES), dtype=np.int8)
    all_resolved_codes = np.array(sorted(_SEX_SOURCE_CODE[s] for s in ALL_RESOLVED_SEX_SOURCES), dtype=np.int8)

    section: dict = {
        "null_model": {
            "method": "fixed_margin_exchangeability",
            "statistic": "within_group_pair_concordance",
            "sex_probability": "conditioned_margin",
            "min_group_size": MIN_GROUP_SIZE,
            "headline_eligibility": "sex_source_input",
            "sensitivity_eligibility": "all_resolved_sex",
            "multiplicity": "holm_across_groupings",
        },
    }

    blocks: dict[str, dict] = {}
    for name, parent_cols in GROUPINGS:
        with timed(f"sex concordance grouping ({name})"):
            parents = [df[col].to_numpy() for col in parent_cols]
            in_scope = np.ones(len(df), dtype=bool)
            for parent in parents:
                in_scope &= parent != -1
            index = _build_group_index(
                tuple(parent[in_scope] for parent in parents),
                is_male[in_scope],
                sex_source_code[in_scope],
            )
        with timed(f"sex concordance analytical ({name})"):
            headline = _analyse(
                index,
                np.isin(index.sex_source_code, headline_codes),
                require_input_sex=True,
            )
            headline["_sensitivity"] = _analyse(
                index,
                np.isin(index.sex_source_code, all_resolved_codes),
                require_input_sex=False,
            )
        blocks[name] = headline

    # Multiplicity across whichever groupings are computable.
    analytical_holm = holm_adjust({n: b["p_analytical"] for n, b in blocks.items() if b["computed"]})

    permutation_raw: dict[str, float] = {}
    if permutations > 0:
        for name, block in blocks.items():
            if not block["computed"]:
                continue
            with timed(f"sex concordance permutations ({name})"):
                perm = _run_permutations(
                    block["_sizes"],
                    block["_n_male"],
                    int(block["_sizes"].sum()),
                    block["_observed"],
                    float(block["_moments"].expected),
                    permutations,
                    seed,
                )
            block["_permutations"] = perm
            if perm["p_raw"] is not None:
                permutation_raw[name] = perm["p_raw"]
    permutation_holm = holm_adjust(permutation_raw)

    for name, block in blocks.items():
        sensitivity = _strip_internals(block.pop("_sensitivity"))
        perm = block.pop("_permutations", None)
        moments = block.pop("_moments", None)
        block.pop("_observed", None)
        male_counts = block.pop("_male_counts", None)
        sizes = block.pop("_sizes", None)
        block.pop("_n_male", None)

        block["p_analytical_holm"] = analytical_holm.get(name)
        if perm is not None and perm["p_raw"] is not None:
            perm["p_holm"] = permutation_holm.get(name)
        elif perm is not None:
            perm["p_holm"] = None
        block["permutations"] = perm
        use_perm = perm is not None and perm["p_raw"] is not None
        block["p_source"] = "permutation" if use_perm else ("analytical" if block["computed"] else None)
        block["p_holm"] = perm["p_holm"] if use_perm else block["p_analytical_holm"]
        block["all_resolved_excess_concordance"] = sensitivity["excess_concordance"]
        block["all_resolved"] = sensitivity
        if block["computed"] and sizes is not None and moments is not None:
            block["male_proportion_distribution"] = _proportion_distribution(male_counts / sizes)
            block["by_group_size"] = _by_group_size(
                sizes,
                male_counts,
                float(moments.expected / moments.n_pairs),
            )
        else:
            block["male_proportion_distribution"] = None
            block["by_group_size"] = []
        _warn_if_unpermuted_tail(name, block)
        section[name] = block
    return section


def _warn_if_unpermuted_tail(name: str, block: dict) -> None:
    """Warn when an unpermuted analytical p lands in the anti-conservative tail."""
    p = block.get("p_analytical")
    if not block.get("computed") or p is None or p >= ANALYTICAL_P_WARN:
        return
    if block.get("p_source") == "permutation":
        return
    logger.warning(
        "%s: analytical p = %.3g is below %.2f, where the normal approximation runs up to ~18x too "
        "liberal in the positive (over-concordant) tail; max_group_pair_share = %.3g. Re-run with "
        "--sex-concordance-permutations before claiming this.",
        name,
        p,
        ANALYTICAL_P_WARN,
        block.get("max_group_pair_share") or 0.0,
    )
