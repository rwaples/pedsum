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
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from pedsum.base import SEX_MALE, logger

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

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
        sizes: Eligible group sizes, in canonical group order.
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
    n_total = int(sizes.sum())
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


@dataclass(frozen=True)
class _Projection:
    """One provenance mask reduced to the groups it makes eligible."""

    sizes: np.ndarray
    """Admitted offspring per *eligible* group."""

    male_counts: np.ndarray
    """Admitted males per eligible group, aligned with ``sizes``."""

    eligible_row: np.ndarray
    """Sorted-row mask of the offspring the statistic actually uses."""

    counts: dict
    """Eligibility/exclusion metadata, emitted verbatim into the report block."""


_COUNT_KEYS: tuple[str, ...] = (
    "n_groups_total",
    "n_groups_eligible",
    "n_groups_incomplete",
    "n_groups_too_small",
    "n_offspring_total",
    "n_offspring_eligible",
    "n_offspring_sex_excluded",
    "n_offspring_in_small_groups",
)


def _project(index: _GroupIndex, admitted: np.ndarray) -> _Projection:
    """Reduce an admitted-offspring mask to eligible groups and count metadata.

    ``admitted`` marks offspring whose sex is resolved *and* whose provenance
    the analysis admits. Eligible members of a group are retained even when
    other members are ineligible; a group informs concordance only from
    ``MIN_GROUP_SIZE`` admitted members up.
    """
    n_groups = index.group_sizes.size
    if n_groups == 0:
        empty = np.empty(0, dtype=np.int64)
        return _Projection(empty, empty, np.empty(0, dtype=bool), dict.fromkeys(_COUNT_KEYS, 0))
    admitted_i = admitted.astype(np.int64)
    admitted_per_group = np.add.reduceat(admitted_i, index.group_starts)
    male_per_group = np.add.reduceat(admitted_i * index.is_male, index.group_starts)
    keep = admitted_per_group >= MIN_GROUP_SIZE
    n_admitted = int(admitted_i.sum())
    n_eligible = int(admitted_per_group[keep].sum())
    n_offspring = int(index.group_sizes.sum())
    return _Projection(
        sizes=admitted_per_group[keep],
        male_counts=male_per_group[keep],
        eligible_row=admitted & keep[index.group_of_row],
        counts={
            "n_groups_total": int(n_groups),
            "n_groups_eligible": int(keep.sum()),
            "n_groups_incomplete": int((keep & (admitted_per_group < index.group_sizes)).sum()),
            "n_groups_too_small": int((~keep).sum()),
            "n_offspring_total": n_offspring,
            "n_offspring_eligible": n_eligible,
            "n_offspring_sex_excluded": n_offspring - n_admitted,
            "n_offspring_in_small_groups": n_admitted - n_eligible,
        },
    )


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


def _direction(observed: int, expected: Fraction) -> str:
    """Classify the excess exactly, so the verdict never hinges on float rounding."""
    if Fraction(observed) > expected:
        return "over_concordant"
    if Fraction(observed) < expected:
        return "under_concordant"
    return "none"


@dataclass(frozen=True)
class _Analysis:
    """One analysis pass: the report block plus what permutation calibration needs.

    Keeping the arrays here rather than smuggling them through the block means
    the block is only ever the emitted payload.
    """

    block: dict
    sizes: np.ndarray
    male_counts: np.ndarray
    n_male: int
    observed: int
    moments: Moments


def _analyse(index: _GroupIndex, admitted: np.ndarray, *, require_input_sex: bool) -> _Analysis:
    """Eligibility counts, the pair-concordance statistic, and analytical inference.

    Shared by the headline (``input`` provenance only) and the all-resolved
    sensitivity pass; the two differ only in ``admitted``.
    """
    projection = _project(index, admitted)
    sizes = projection.sizes
    male_counts = projection.male_counts
    counts = projection.counts

    # One pass over the analysed rows, then read off the sources we report.
    # ``minlength`` is what makes the lookup safe: without it bincount sizes its
    # result to max(code) + 1, so a pedigree of purely input sex would return a
    # length-1 tally and IndexError on the imputed slots.
    tally = np.bincount(index.sex_source_code[projection.eligible_row], minlength=len(_SEX_SOURCE_ORDER))
    by_source = {name: int(tally[_SEX_SOURCE_CODE[name]]) for name in ALL_RESOLVED_SEX_SOURCES}

    n_male = int(male_counts.sum())
    n_total = int(sizes.sum())
    n_female = n_total - n_male
    moments = analytical_moments(sizes, n_male, n_female)
    observed = concordant_pairs(male_counts, sizes)
    n_pairs = moments.n_pairs

    # Every eligible group holds at least MIN_GROUP_SIZE offspring, so a
    # non-empty ``sizes`` always contributes at least one pair: ``n_pairs`` and
    # ``n_total`` are zero together, and one guard covers all three ratios.
    n_max = int(sizes.max()) if n_pairs else 0
    out: dict = {
        "computed": False,
        "skip_reason": None,
        "skip_message": None,
        **counts,
        "eligible_by_sex_source": by_source,
        "conditioning_male_fraction": (n_male / n_total) if n_pairs else None,
        "n_within_group_pairs": n_pairs,
        "max_group_pair_share": (n_max * (n_max - 1) // 2) / n_pairs if n_pairs else None,
        "observed_pair_concordance": (observed / n_pairs) if n_pairs else None,
        "expected_pair_concordance": None,
        "excess_concordance": None,
        "direction": None,
        "z": None,
        "p_analytical": None,
    }

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

    if skip is None:
        expected = moments.expected
        z = (observed - float(expected)) / math.sqrt(float(moments.variance))
        out.update(
            {
                "computed": True,
                "expected_pair_concordance": float(expected / n_pairs),
                "excess_concordance": float((Fraction(observed) - expected) / n_pairs),
                "direction": _direction(observed, expected),
                "z": z,
                "p_analytical": _two_sided_normal_p(z),
            },
        )
    else:
        out["skip_reason"] = skip
        out["skip_message"] = _SKIP_MESSAGES[skip]

    return _Analysis(out, sizes, male_counts, n_male, observed, moments)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _null_timer(label: str) -> AbstractContextManager[None]:
    """No-op stand-in for the CLI's ``_timed`` phase logger; ``label`` is ignored."""
    del label
    return nullcontext()


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

    headline: dict[str, _Analysis] = {}
    sensitivity: dict[str, dict] = {}
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
            headline[name] = _analyse(
                index,
                np.isin(index.sex_source_code, headline_codes),
                require_input_sex=True,
            )
            sensitivity[name] = _analyse(
                index,
                np.isin(index.sex_source_code, all_resolved_codes),
                require_input_sex=False,
            ).block

    # Multiplicity across whichever groupings are computable.
    computable = {name: a for name, a in headline.items() if a.block["computed"]}
    analytical_holm = holm_adjust({name: a.block["p_analytical"] for name, a in computable.items()})

    permutation_blocks: dict[str, dict] = {}
    if permutations > 0:
        for name, analysis in computable.items():
            with timed(f"sex concordance permutations ({name})"):
                permutation_blocks[name] = _run_permutations(
                    analysis.sizes,
                    analysis.n_male,
                    analysis.observed,
                    float(analysis.moments.expected),
                    permutations,
                    seed,
                )
    permutation_holm = holm_adjust(
        {name: p["p_raw"] for name, p in permutation_blocks.items() if p["p_raw"] is not None},
    )

    for name, analysis in headline.items():
        block = analysis.block
        block["p_analytical_holm"] = analytical_holm.get(name)

        permutation_block = permutation_blocks.get(name)
        use_permutation = permutation_block is not None and permutation_block["p_raw"] is not None
        if permutation_block is not None:
            permutation_block["p_holm"] = permutation_holm.get(name)
        block["permutations"] = permutation_block

        if use_permutation:
            block["p_source"] = "permutation"
            block["p_holm"] = permutation_block["p_holm"]
        elif block["computed"]:
            block["p_source"] = "analytical"
            block["p_holm"] = block["p_analytical_holm"]
        else:
            block["p_source"] = None
            block["p_holm"] = None

        block["all_resolved_excess_concordance"] = sensitivity[name]["excess_concordance"]
        block["all_resolved"] = sensitivity[name]

        if block["computed"]:
            block["male_proportion_distribution"] = _proportion_distribution(analysis.male_counts / analysis.sizes)
            block["by_group_size"] = _by_group_size(
                analysis.sizes,
                analysis.male_counts,
                float(analysis.moments.expected / analysis.moments.n_pairs),
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
