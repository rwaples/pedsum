"""Validation: a declarative Check registry driven by one dual-mode runner.

`load_and_validate` (summarize, fail-fast) and `validate_pedigree` (validate,
accumulating) share one `CHECKS` registry and one `_run_checks` runner — see
`docs/adr/0002-validation-check-registry.md`. Each Check carries its
prerequisites, label, and group; `_CHECK_ORDER` and `_CHECK_LABELS` are derived
from the registry. `_CHECK_GROUPS` is a curated *display* order (it intentionally
differs from execution order) and stays explicit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, PedigreeError, logger
from pedsum.checks import (
    CheckResult,
    Finding,
    _check_acyclic,
    _check_birth_year_range,
    _check_birth_year_topology,
    _check_duplicate_ids,
    _check_empty_pedigree,
    _check_negative_ids,
    _check_parent_refs_present,
    _check_parent_refs_sex_conflict,
    _check_parent_token_range,
    _check_parents_distinct,
    _check_self_loops,
    _check_sex_role_ambiguity,
    _check_sex_role_consistency,
    _check_unknown_sex,
    _summarize_findings,
)
from pedsum.parse import (
    _BIRTH_YEAR_DEFAULT_MIN,
    _as_birth_year_col,
    _as_int_col,
    _as_parent_int_col,
    _birth_year_default_max,
    _decode_sex,
    _format_id_sample,
    _maybe_warn_csv,
    _read_pedigree_table,
)
from pedsum.pedigree_ops import _build_children_csr, _compute_depth_unordered, _parent_rows

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Sex imputation
# ---------------------------------------------------------------------------


@dataclass
class _SexImputation:
    """Result of imputing unknown sex from mother / father role usage."""

    imputed_sex: np.ndarray
    n_imputed: int
    ambiguous_mask: np.ndarray
    original_unknown_mask: np.ndarray
    overridden_mask: np.ndarray
    sex_source: np.ndarray  # dtype=object; per-row category string


def _impute_sex_from_roles(
    sex: np.ndarray,
    ids: np.ndarray,
    mothers: np.ndarray,
    fathers: np.ndarray,
    *,
    override_asserted_sex: bool = True,
) -> _SexImputation:
    """Impute sex from mother / father role usage.

    Two passes over each row:

      1. Missing → role:
         - id used only as mother → impute F
         - id used only as father → impute M
         - id used as both        → leave -1, mark ambiguous (caller decides
                                    whether to raise / tolerate via
                                    ``--allow-missing-sex``)
         - id used as neither     → leave -1, orphan (caller decides whether
                                    to raise / tolerate via
                                    ``--allow-missing-sex``)

      2. Asserted → role (0.9 default; opt out via
         ``override_asserted_sex=False``):
         - asserted M used only as mother → override to F
         - asserted F used only as father → override to M
         - asserted used as both          → leave alone; ``_check_sex_role_
                                            consistency`` still hard-blocks
         - asserted used as neither       → leave alone

    Returns a `_SexImputation` whose `sex_source` is a per-row `dtype=object`
    array tagging each row's provenance: ``"input"`` / ``"imputed_from_missing"``
    / ``"imputed_from_role"`` / ``"unresolved"``.
    """
    original_unknown_mask = (sex == SEX_UNKNOWN).copy()
    imputed = sex.copy()
    sex_source = np.full(len(sex), "input", dtype=object)

    # The distinct parent IDs drive the as_mother / as_father role masks every
    # row needs. (The first-occurrence row of each ambiguous parent is computed
    # on demand in _check_sex_role_ambiguity, which only runs on the rare
    # ambiguous rows — so no id→row table is materialised here.)
    unique_mids = np.unique(mothers[mothers != -1])
    unique_fids = np.unique(fathers[fathers != -1])

    as_mother = np.isin(ids, unique_mids)
    as_father = np.isin(ids, unique_fids)

    # Pass 1: missing → role.
    impute_f = original_unknown_mask & as_mother & ~as_father
    impute_m = original_unknown_mask & as_father & ~as_mother
    ambiguous_mask = original_unknown_mask & as_mother & as_father
    imputed[impute_f] = SEX_FEMALE
    imputed[impute_m] = SEX_MALE
    imputed_from_missing = impute_f | impute_m
    sex_source[imputed_from_missing] = "imputed_from_missing"
    n_imputed = int(imputed_from_missing.sum())

    # Pass 2: asserted → role (when topology is unambiguous).
    override_to_f = (sex == SEX_MALE) & as_mother & ~as_father
    override_to_m = (sex == SEX_FEMALE) & as_father & ~as_mother
    if override_asserted_sex:
        imputed[override_to_f] = SEX_FEMALE
        imputed[override_to_m] = SEX_MALE
        overridden_mask = override_to_f | override_to_m
        sex_source[overridden_mask] = "imputed_from_role"
    else:
        overridden_mask = np.zeros(len(sex), dtype=bool)

    # Final pass: anything still SEX_UNKNOWN is unresolved.
    sex_source[imputed == SEX_UNKNOWN] = "unresolved"

    return _SexImputation(
        imputed_sex=imputed,
        n_imputed=n_imputed,
        ambiguous_mask=ambiguous_mask,
        original_unknown_mask=original_unknown_mask,
        overridden_mask=overridden_mask,
        sex_source=sex_source,
    )


# ---------------------------------------------------------------------------
# Check, CheckOutcome, ValidationContext
# ---------------------------------------------------------------------------


@dataclass
class CheckOutcome:
    """A single Check's verdict: status, any findings, and an optional annotation."""

    status: str  # "PASS" | "FAIL" | "SKIP"
    findings: list[Finding] = field(default_factory=list)
    count: int = 0
    skip_reason: str | None = None


_PASS = CheckOutcome("PASS")


def _from_findings(findings: list[Finding]) -> CheckOutcome:
    """PASS when no findings, else FAIL carrying them (count = len)."""
    if findings:
        return CheckOutcome("FAIL", findings, len(findings))
    return _PASS


def _try_parse(
    check_name: str,
    parse: Callable[[], np.ndarray],
) -> tuple[np.ndarray | None, CheckOutcome | None]:
    """Run a column parser, mapping a PedigreeError to a single-finding FAIL.

    Returns ``(value, None)`` on success or ``(None, outcome)`` on failure, so a
    parse Check reads ``value, fail = _try_parse(...); if fail: return fail``.
    """
    try:
        return parse(), None
    except PedigreeError as e:
        return None, CheckOutcome("FAIL", [Finding(check=check_name, detail=str(e))], 1)


@dataclass
class ValidationContext:
    """Mutable state threaded through every Check's ``run(ctx)``.

    Parse Checks populate the array fields (parsing *is* the Check); all other
    Checks read them. ``id_index`` and ``imputation`` are derived once on first
    access. Carries the CLI config a Check needs (column names, flags, bounds).
    """

    df_raw: pd.DataFrame
    id_col: str
    sex_col: str
    mother_col: str
    father_col: str
    birth_year_col: str | None
    sex_encoding: str
    zero_as_missing: bool
    allow_missing_sex: bool
    override_asserted_sex: bool
    birth_year_min: int
    birth_year_max: int | None
    # summarize requires a named --birth-year-col to exist (raises if absent);
    # validate treats a missing one as an inapplicable SKIP.
    require_birth_year_col: bool
    # --no-sex-check tolerates parent_refs_sex_conflict (SKIP); lives in the
    # context so the tolerance composes inside _run_checks (validate, the
    # drop-offending loop, and its self-verify) rather than as a cli post-filter.
    no_sex_check: bool = False
    # Populated by the parse Checks.
    ids: np.ndarray | None = None
    mothers: np.ndarray | None = None
    fathers: np.ndarray | None = None
    sex: np.ndarray | None = None
    birth_year: np.ndarray | None = None

    @cached_property
    def id_index(self) -> pd.Index:
        """Row lookup over the parsed IDs; built once (IDs are fixed after parse)."""
        return pd.Index(self.ids)

    @cached_property
    def imputation(self) -> _SexImputation:
        """Sex imputation from parent-role usage; computed once, shared by the sex checks."""
        return _impute_sex_from_roles(
            self.sex,
            self.ids,
            self.mothers,
            self.fathers,
            override_asserted_sex=self.override_asserted_sex,
        )

    def get_imputation(self) -> _SexImputation | None:
        """Return the imputation only if already computed (never triggers it)."""
        return self.__dict__.get("imputation")

    @property
    def by_max(self) -> int:
        """Resolved birth-year upper bound (default: current calendar year + 1)."""
        return self.birth_year_max if self.birth_year_max is not None else _birth_year_default_max()


@dataclass(frozen=True)
class Check:
    """One registry entry: a named integrity test with its prerequisites + display metadata."""

    name: str
    requires: tuple[str, ...]
    run: Callable[[ValidationContext], CheckOutcome]
    label: str
    group: str
    # Whether `validate --drop-offending` can resolve a FAIL by removing the
    # flagged individual(s). Per-individual checks are droppable; column/file-
    # level checks (missing column, parse failure, negative id, …) are not.
    droppable: bool = False


# ---------------------------------------------------------------------------
# Per-check run functions (ctx -> CheckOutcome). The _check_* helpers stay pure.
# ---------------------------------------------------------------------------


def _ck_required_columns(ctx: ValidationContext) -> CheckOutcome:
    # summarize (require_birth_year_col) treats a named --birth-year-col as
    # required; validate omits it and lets birth_year_dtype SKIP instead.
    needed = {ctx.id_col, ctx.sex_col, ctx.mother_col, ctx.father_col}
    if ctx.require_birth_year_col and ctx.birth_year_col is not None:
        needed.add(ctx.birth_year_col)
    miss = needed - set(ctx.df_raw.columns)
    if miss:
        msg = f"missing required columns: {sorted(miss)}; file has {list(ctx.df_raw.columns)}"
        return CheckOutcome("FAIL", [Finding(check="required_columns", detail=msg)], 1)
    return _PASS


def _ck_empty_pedigree(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(_check_empty_pedigree(len(ctx.df_raw)))


def _ck_id_dtype(ctx: ValidationContext) -> CheckOutcome:
    ctx.ids, fail = _try_parse("id_dtype", lambda: _as_int_col(ctx.df_raw[ctx.id_col], ctx.id_col))
    if fail:
        return fail
    logger.info("parsed %d ids; five random ids: %s", len(ctx.ids), _format_id_sample(ctx.ids))
    return _PASS


def _ck_mother_dtype(ctx: ValidationContext) -> CheckOutcome:
    ctx.mothers, fail = _try_parse(
        "mother_dtype",
        lambda: _as_parent_int_col(ctx.df_raw[ctx.mother_col], ctx.mother_col, ctx.zero_as_missing),
    )
    return fail or _PASS


def _ck_father_dtype(ctx: ValidationContext) -> CheckOutcome:
    ctx.fathers, fail = _try_parse(
        "father_dtype",
        lambda: _as_parent_int_col(ctx.df_raw[ctx.father_col], ctx.father_col, ctx.zero_as_missing),
    )
    return fail or _PASS


def _ck_sex_tokens(ctx: ValidationContext) -> CheckOutcome:
    ctx.sex, fail = _try_parse(
        "sex_tokens",
        lambda: _decode_sex(ctx.df_raw[ctx.sex_col], encoding=ctx.sex_encoding, zero_as_missing=ctx.zero_as_missing),
    )
    return fail or _PASS


def _ck_negative_ids(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(_check_negative_ids(ctx.ids))


def _ck_duplicate_ids(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(_check_duplicate_ids(ctx.ids))


def _ck_parent_token_range_mother(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(_check_parent_token_range(ctx.mothers, "mother"))


def _ck_parent_token_range_father(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(_check_parent_token_range(ctx.fathers, "father"))


def _ck_parent_refs_present_mother(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(_check_parent_refs_present(ctx.mothers, "mother", ctx.id_index))


def _ck_parent_refs_present_father(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(_check_parent_refs_present(ctx.fathers, "father", ctx.id_index))


def _ck_parent_refs_sex_conflict(ctx: ValidationContext) -> CheckOutcome:
    if ctx.no_sex_check:
        return CheckOutcome("SKIP", skip_reason="bypassed via --no-sex-check")
    return _from_findings(_check_parent_refs_sex_conflict(ctx.mothers, ctx.fathers, ctx.id_index))


def _ck_self_loops(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(_check_self_loops(ctx.ids, ctx.mothers, ctx.fathers))


def _ck_parents_distinct(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(_check_parents_distinct(ctx.ids, ctx.mothers, ctx.fathers))


def _ck_sex_role_ambiguity(ctx: ValidationContext) -> CheckOutcome:
    imp = ctx.imputation
    if ctx.allow_missing_sex:
        n = int(imp.ambiguous_mask.sum())
        return CheckOutcome("SKIP", skip_reason=f"bypassed via --allow-missing-sex ({n} tolerated)")
    return _from_findings(
        _check_sex_role_ambiguity(
            ctx.ids,
            imp.ambiguous_mask,
            ctx.mothers,
            ctx.fathers,
        )
    )


def _ck_sex_role_consistency(ctx: ValidationContext) -> CheckOutcome:
    imp = ctx.imputation
    # Always run the real check: the override clears asserted-vs-role
    # contradictions only for rows used in a SINGLE role; a row asserted M and
    # used as BOTH mother and father stays unoverridable and must still FAIL.
    findings = _check_sex_role_consistency(
        ctx.mothers,
        ctx.fathers,
        imp.imputed_sex,
        ctx.id_index,
        skip_mask=imp.original_unknown_mask,
    )
    if findings:
        return CheckOutcome("FAIL", findings, len(findings))
    # Clean: surface the override count as a PASS annotation
    # ("PASS (N overridden from role)") when the override fired.
    n = int(imp.overridden_mask.sum()) if ctx.override_asserted_sex else 0
    return CheckOutcome("PASS", count=n, skip_reason=(f"{n} overridden from role" if n > 0 else None))


def _ck_unknown_sex(ctx: ValidationContext) -> CheckOutcome:
    imp = ctx.imputation
    if ctx.allow_missing_sex:
        n = int(((imp.imputed_sex == SEX_UNKNOWN) & ~imp.ambiguous_mask).sum())
        return CheckOutcome("SKIP", skip_reason=f"bypassed via --allow-missing-sex ({n} tolerated)")
    return _from_findings(_check_unknown_sex(ctx.ids, imp.imputed_sex, imp.ambiguous_mask))


def _ck_acyclic(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(_check_acyclic(ctx.ids, ctx.mothers, ctx.fathers, ctx.id_index))


def _ck_birth_year_dtype(ctx: ValidationContext) -> CheckOutcome:
    if ctx.birth_year_col is None:
        return CheckOutcome("SKIP", skip_reason="no --birth-year-col specified")
    if ctx.birth_year_col not in ctx.df_raw.columns:
        return CheckOutcome("SKIP", skip_reason=f"column {ctx.birth_year_col!r} not present in file")
    ctx.birth_year, fail = _try_parse(
        "birth_year_dtype",
        lambda: _as_birth_year_col(ctx.df_raw[ctx.birth_year_col], ctx.birth_year_col),
    )
    return fail or _PASS


def _ck_birth_year_range(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(_check_birth_year_range(ctx.ids, ctx.birth_year, ctx.birth_year_min, ctx.by_max))


def _ck_birth_year_topology(ctx: ValidationContext) -> CheckOutcome:
    return _from_findings(
        _check_birth_year_topology(
            ctx.ids,
            ctx.mothers,
            ctx.fathers,
            ctx.birth_year,
            ctx.id_index,
        )
    )


# Prerequisite shorthands.
_IDS = ("id_dtype", "negative_ids", "duplicate_ids")
_IDS_PARENTS = (*_IDS, "mother_dtype", "father_dtype")

CHECKS: tuple[Check, ...] = (
    Check("required_columns", (), _ck_required_columns, "required columns present", "Columns & parsing"),
    Check("empty_pedigree", ("required_columns",), _ck_empty_pedigree, "pedigree is non-empty", "Columns & parsing"),
    Check("id_dtype", ("required_columns",), _ck_id_dtype, "id column parses as integer", "Columns & parsing"),
    Check(
        "mother_dtype", ("required_columns",), _ck_mother_dtype, "mother column parses as integer", "Columns & parsing"
    ),
    Check(
        "father_dtype", ("required_columns",), _ck_father_dtype, "father column parses as integer", "Columns & parsing"
    ),
    Check("sex_tokens", ("required_columns",), _ck_sex_tokens, "sex column tokens recognized", "Columns & parsing"),
    Check("negative_ids", ("id_dtype",), _ck_negative_ids, "no negative IDs", "IDs"),
    Check("duplicate_ids", ("id_dtype",), _ck_duplicate_ids, "no duplicate IDs", "IDs", droppable=True),
    Check(
        "parent_token_range_mother",
        ("mother_dtype",),
        _ck_parent_token_range_mother,
        "mother IDs in valid range",
        "Parent references",
    ),
    Check(
        "parent_token_range_father",
        ("father_dtype",),
        _ck_parent_token_range_father,
        "father IDs in valid range",
        "Parent references",
    ),
    Check(
        "parent_refs_present_mother",
        (*_IDS, "mother_dtype"),
        _ck_parent_refs_present_mother,
        "mother IDs present in pedigree",
        "Parent references",
    ),
    Check(
        "parent_refs_present_father",
        (*_IDS, "father_dtype"),
        _ck_parent_refs_present_father,
        "father IDs present in pedigree",
        "Parent references",
    ),
    Check(
        "parent_refs_sex_conflict",
        _IDS_PARENTS,
        _ck_parent_refs_sex_conflict,
        "no missing-parent sex conflicts",
        "Parent references",
        droppable=True,
    ),
    Check(
        "sex_role_ambiguity",
        (*_IDS_PARENTS, "sex_tokens"),
        _ck_sex_role_ambiguity,
        "no role-ambiguous unsexed individuals",
        "Parent references",
        droppable=True,
    ),
    Check("self_loops", _IDS_PARENTS, _ck_self_loops, "no self-loops", "Graph structure", droppable=True),
    Check(
        "parents_distinct",
        _IDS_PARENTS,
        _ck_parents_distinct,
        "mother and father distinct",
        "Parent references",
        droppable=True,
    ),
    Check(
        "sex_role_consistency",
        (*_IDS_PARENTS, "sex_tokens"),
        _ck_sex_role_consistency,
        "sex consistent with parent role",
        "Graph structure",
        droppable=True,
    ),
    Check(
        "unknown_sex",
        (*_IDS_PARENTS, "sex_tokens"),
        _ck_unknown_sex,
        "all individuals have resolved sex",
        "Graph structure",
        droppable=True,
    ),
    Check(
        "acyclic",
        ("parent_refs_present_mother", "parent_refs_present_father", "parents_distinct"),
        _ck_acyclic,
        "acyclic (no descent cycles)",
        "Graph structure",
        droppable=True,
    ),
    Check(
        "birth_year_dtype",
        ("required_columns",),
        _ck_birth_year_dtype,
        "birth_year column parses as numeric",
        "Birth years (optional)",
    ),
    Check(
        "birth_year_range",
        ("id_dtype", "birth_year_dtype"),
        _ck_birth_year_range,
        "birth years within sanity range",
        "Birth years (optional)",
        droppable=True,
    ),
    Check(
        "birth_year_topology",
        (*_IDS_PARENTS, "birth_year_dtype"),
        _ck_birth_year_topology,
        "child birth_year >= parent birth_year",
        "Birth years (optional)",
        droppable=True,
    ),
)

_CHECKS_BY_NAME: dict[str, Check] = {c.name: c for c in CHECKS}

# Derived from the registry (single source of truth).
_CHECK_ORDER: tuple[str, ...] = tuple(c.name for c in CHECKS)
_CHECK_LABELS: dict[str, str] = {c.name: c.label for c in CHECKS}

# Checks `validate --drop-offending` can resolve by removing the flagged
# individual(s). Derived from the registry so the set never drifts.
DROPPABLE_CHECKS: frozenset[str] = frozenset(c.name for c in CHECKS if c.droppable)

# Non-droppable checks that must still BLOCK even under --drop-offending: no
# row removal fixes a missing column / parse failure / negative id / etc.
# parent_refs_present_* is excluded — it is auto-fixed by founder synthesis,
# neither dropped nor blocking.
NON_REDUCIBLE_BLOCK_CHECKS: frozenset[str] = frozenset(
    c.name
    for c in CHECKS
    if not c.droppable and c.name not in ("parent_refs_present_mother", "parent_refs_present_father")
)

# Drop fraction (rows removed / input rows) above which --drop-offending warns.
DROP_FRACTION_WARN: float = 0.10

# Curated *display* grouping for the stderr summary. Order within a group
# intentionally differs from execution order (e.g. parents_distinct), so this
# stays an explicit constant rather than a derivation of _CHECK_ORDER.
_CHECK_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Columns & parsing",
        (
            "required_columns",
            "empty_pedigree",
            "id_dtype",
            "mother_dtype",
            "father_dtype",
            "sex_tokens",
        ),
    ),
    ("IDs", ("negative_ids", "duplicate_ids")),
    (
        "Parent references",
        (
            "parent_token_range_mother",
            "parent_token_range_father",
            "parent_refs_present_mother",
            "parent_refs_present_father",
            "parents_distinct",
            "parent_refs_sex_conflict",
            "sex_role_ambiguity",
        ),
    ),
    (
        "Graph structure",
        (
            "self_loops",
            "sex_role_consistency",
            "unknown_sex",
            "acyclic",
        ),
    ),
    (
        "Birth years (optional)",
        (
            "birth_year_dtype",
            "birth_year_range",
            "birth_year_topology",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_checks(
    ctx: ValidationContext,
    *,
    on_fail: str,
) -> tuple[dict[str, CheckResult], list[Finding]]:
    """Run every Check in registry order against ``ctx``.

    A Check whose prerequisite did not ``PASS`` is auto-``SKIP``ped: a FAILed
    prerequisite yields ``"<prereq> failed"``, a SKIPped one propagates its own
    reason (so ``required_columns failed`` / ``no --birth-year-col specified``
    flow transitively). ``on_fail="raise"`` raises on the first ``FAIL``
    (summarize); ``on_fail="accumulate"`` records it and continues (validate).
    """
    results: dict[str, CheckResult] = {}
    findings: list[Finding] = []
    for check in CHECKS:
        skip_reason = None
        for prereq in check.requires:
            r = results[prereq]
            if r.status != "PASS":
                skip_reason = r.skip_reason if r.status == "SKIP" else f"{prereq} failed"
                break
        if skip_reason is not None:
            results[check.name] = CheckResult(name=check.name, status="SKIP", skip_reason=skip_reason)
            continue
        outcome = check.run(ctx)
        if outcome.status == "FAIL":
            if on_fail == "raise":
                raise PedigreeError(_summarize_findings(outcome.findings))
            findings.extend(outcome.findings)
            results[check.name] = CheckResult(name=check.name, status="FAIL", count=outcome.count)
        else:
            results[check.name] = CheckResult(
                name=check.name,
                status=outcome.status,
                count=outcome.count,
                skip_reason=outcome.skip_reason,
            )
    return results, findings


def _build_context_from_df(
    df: pd.DataFrame,
    *,
    id_col: str,
    sex_col: str,
    mother_col: str,
    father_col: str,
    sex_encoding: str,
    zero_as_missing: bool,
    allow_missing_sex: bool,
    override_asserted_sex: bool,
    birth_year_col: str | None,
    birth_year_min: int,
    birth_year_max: int | None,
    require_birth_year_col: bool,
    no_sex_check: bool = False,
) -> ValidationContext:
    """Assemble a `ValidationContext` from an in-memory DataFrame (no file read).

    Used by `_build_context` (after reading the file) and by the
    `validate --drop-offending` loop, which rebuilds a context from the reduced
    pedigree each round so `_run_checks` re-runs the *same* registry — no second
    check sequence (ADR 0002).
    """
    return ValidationContext(
        df_raw=df,
        id_col=id_col,
        sex_col=sex_col,
        mother_col=mother_col,
        father_col=father_col,
        birth_year_col=birth_year_col,
        sex_encoding=sex_encoding,
        zero_as_missing=zero_as_missing,
        allow_missing_sex=allow_missing_sex,
        override_asserted_sex=override_asserted_sex,
        birth_year_min=birth_year_min,
        birth_year_max=birth_year_max,
        require_birth_year_col=require_birth_year_col,
        no_sex_check=no_sex_check,
    )


def _build_context(
    path: Path,
    *,
    id_col: str,
    sex_col: str,
    mother_col: str,
    father_col: str,
    sex_encoding: str,
    zero_as_missing: bool,
    allow_missing_sex: bool,
    override_asserted_sex: bool,
    birth_year_col: str | None,
    birth_year_min: int,
    birth_year_max: int | None,
    sep: str,
    require_birth_year_col: bool,
    log_read: bool,
    no_sex_check: bool = False,
) -> ValidationContext:
    """Read the table and assemble a `ValidationContext` (shared by both modes)."""
    t0 = time.perf_counter()
    df = _read_pedigree_table(path, sep=sep, dtype=str)
    if log_read:
        logger.info("read %d rows from %s in %.2fs", len(df), path, time.perf_counter() - t0)
    _maybe_warn_csv(df)
    return _build_context_from_df(
        df,
        id_col=id_col,
        sex_col=sex_col,
        mother_col=mother_col,
        father_col=father_col,
        sex_encoding=sex_encoding,
        zero_as_missing=zero_as_missing,
        allow_missing_sex=allow_missing_sex,
        override_asserted_sex=override_asserted_sex,
        birth_year_col=birth_year_col,
        birth_year_min=birth_year_min,
        birth_year_max=birth_year_max,
        require_birth_year_col=require_birth_year_col,
        no_sex_check=no_sex_check,
    )


# ---------------------------------------------------------------------------
# Public drivers
# ---------------------------------------------------------------------------


def load_and_validate(
    path: Path,
    id_col: str = "id",
    sex_col: str = "sex",
    mother_col: str = "mother",
    father_col: str = "father",
    sex_encoding: str = "auto",
    zero_as_missing: bool = False,
    allow_missing_sex: bool = False,
    override_asserted_sex: bool = True,
    birth_year_col: str | None = None,
    birth_year_min: int = _BIRTH_YEAR_DEFAULT_MIN,
    birth_year_max: int | None = None,
    sep: str = "auto",
    no_sex_check: bool = False,
) -> tuple[pd.DataFrame, sp.csr_matrix | None]:
    """Load TSV, run all QC fail-fast, return (df, children_csr).

    df has columns id, sex (int8), mother, father, sex_source, and
    ``birth_year`` (int32, sentinel -1) when ``birth_year_col`` is set and
    present. Missing parent encoded as -1. children_csr is the parent→child
    sparse matrix (None if there are no parent edges), shared across the
    generations / components / descendants / inbreeding passes. Raises
    ``PedigreeError`` on the first failing Check.

    Rows are topologically sorted (parents before children) so downstream
    ``PedigreeGraph`` construction holds its parents-precede-children invariant.
    ``ped_depth`` is populated by the caller from ``pg.generation`` before any
    summary function runs.
    """
    t0 = time.perf_counter()
    ctx = _build_context(
        path,
        id_col=id_col,
        sex_col=sex_col,
        mother_col=mother_col,
        father_col=father_col,
        sex_encoding=sex_encoding,
        zero_as_missing=zero_as_missing,
        allow_missing_sex=allow_missing_sex,
        override_asserted_sex=override_asserted_sex,
        birth_year_col=birth_year_col,
        birth_year_min=birth_year_min,
        birth_year_max=birth_year_max,
        sep=sep,
        require_birth_year_col=True,
        log_read=True,
        no_sex_check=no_sex_check,
    )
    _run_checks(ctx, on_fail="raise")

    imp = ctx.imputation
    sex = imp.imputed_sex
    n = len(ctx.ids)
    n_overridden = int(imp.overridden_mask.sum())
    n_unresolved = int((sex == SEX_UNKNOWN).sum())
    if imp.n_imputed > 0 or n_overridden > 0 or n_unresolved > 0:
        logger.info(
            "sex imputation: %d from missing, %d overridden from role, "
            "%d unresolved (sex_source column has per-row detail)",
            imp.n_imputed,
            n_overridden,
            n_unresolved,
        )
    if allow_missing_sex:
        n_ambiguous = int(imp.ambiguous_mask.sum())
        if n_ambiguous > 0:
            logger.info("kept %d row(s) with sex_role_ambiguity (--allow-missing-sex)", n_ambiguous)
        n_orphan = int(((sex == SEX_UNKNOWN) & ~imp.ambiguous_mask).sum())
        if n_orphan > 0:
            logger.info("kept %d row(s) with sex=%d (--allow-missing-sex)", n_orphan, SEX_UNKNOWN)

    # Map parent IDs to row indices, run a fixed-point depth sweep tolerant of
    # arbitrary input row order, and sort by depth (parents before children).
    ids, mothers, fathers = ctx.ids, ctx.mothers, ctx.fathers
    birth_year = ctx.birth_year
    id_index = ctx.id_index
    m_row, mask_m = _parent_rows(mothers, id_index)
    f_row, mask_f = _parent_rows(fathers, id_index)
    depth = _compute_depth_unordered(m_row, f_row, n)
    natural = np.arange(n)
    order = np.argsort(depth, kind="stable")
    reordered = not np.array_equal(order, natural)

    if reordered:
        n_oo = int((order != natural).sum())
        logger.info(
            "reordering %d/%d row(s) into topological order (parents before children)",
            n_oo,
            n,
        )
        out = pd.DataFrame(
            {
                "id": ids[order],
                "sex": sex[order],
                "mother": mothers[order],
                "father": fathers[order],
                "sex_source": imp.sex_source[order],
            },
        )
        if birth_year is not None:
            out["birth_year"] = birth_year[order]
        reordered_index = pd.Index(out["id"].to_numpy())
        m_row, mask_m = _parent_rows(out["mother"].to_numpy(), reordered_index)
        f_row, mask_f = _parent_rows(out["father"].to_numpy(), reordered_index)
    else:
        out = pd.DataFrame(
            {"id": ids, "sex": sex, "mother": mothers, "father": fathers, "sex_source": imp.sex_source},
        )
        if birth_year is not None:
            out["birth_year"] = birth_year

    children_csr = _build_children_csr(m_row, mask_m, f_row, mask_f, n)
    logger.info("validated %d rows in %.2fs", n, time.perf_counter() - t0)
    return out, children_csr


def validate_pedigree(
    path: Path,
    id_col: str = "id",
    sex_col: str = "sex",
    mother_col: str = "mother",
    father_col: str = "father",
    sex_encoding: str = "auto",
    zero_as_missing: bool = False,
    allow_missing_sex: bool = False,
    override_asserted_sex: bool = True,
    birth_year_col: str | None = None,
    birth_year_min: int = _BIRTH_YEAR_DEFAULT_MIN,
    birth_year_max: int | None = None,
    sep: str = "auto",
    no_sex_check: bool = False,
) -> tuple[int, list[CheckResult], list[Finding], ValidationContext]:
    """Run every Check accumulating.

    Returns ``(n_rows, results, findings, ctx)`` where ``results`` covers all
    checks in ``_CHECK_ORDER`` (each PASS / FAIL / SKIP), ``findings`` lists
    every per-individual finding, and ``ctx`` carries the coerced arrays and raw
    DataFrame for ``_run_validate``'s auto-fix.
    """
    ctx = _build_context(
        path,
        id_col=id_col,
        sex_col=sex_col,
        mother_col=mother_col,
        father_col=father_col,
        sex_encoding=sex_encoding,
        zero_as_missing=zero_as_missing,
        allow_missing_sex=allow_missing_sex,
        override_asserted_sex=override_asserted_sex,
        birth_year_col=birth_year_col,
        birth_year_min=birth_year_min,
        birth_year_max=birth_year_max,
        sep=sep,
        require_birth_year_col=False,
        log_read=False,
        no_sex_check=no_sex_check,
    )
    results, findings = _run_checks(ctx, on_fail="accumulate")
    return len(ctx.df_raw), list(results.values()), findings, ctx


# ---------------------------------------------------------------------------
# Reduction: validate --drop-offending (see docs/adr/0003)
# ---------------------------------------------------------------------------


@dataclass
class ReductionResult:
    """Outcome of `reduce_pedigree`: the reduced frame plus a drop record."""

    df_current: pd.DataFrame
    dropped: list[tuple[int, str, int]]  # one per distinct (id, check, round)
    n_rounds: int  # rounds that actually dropped something
    n_input_rows: int
    n_rows_removed: int
    n_distinct_dropped: int
    n_cleared_refs: int
    ctx_final: ValidationContext  # fixpoint context (arrays + imputation populated)


def _cycle_member_ids(ctx: ValidationContext) -> set[int]:
    """IDs in a true descent cycle (strongly-connected component, size >= 2).

    `_check_acyclic` flags every node it cannot topologically order — cycle
    members *and* their descendants. For reduction we drop only true cycle
    members, so descendants survive (as half-founders once the cycle edge is
    cleared) and re-validate the next round. Self-loops are size-1 SCCs here and
    are handled by the `self_loops` check.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    ids = ctx.ids
    n = len(ids)
    id_index = ctx.id_index
    rows: list[int] = []
    cols: list[int] = []
    for parents in (ctx.mothers, ctx.fathers):
        prow = id_index.get_indexer(parents)
        mask = (parents != -1) & (prow != -1)
        child_rows = np.where(mask)[0]
        rows.extend(child_rows.tolist())
        cols.extend(prow[mask].tolist())
    if not rows:
        return set()
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adj, directed=True, connection="strong")
    member_mask = (np.bincount(labels) >= 2)[labels]
    return {int(i) for i in ids[member_mask]}


def _round_drop_ids(findings: list[Finding], ctx: ValidationContext) -> set[int]:
    """IDs to drop this round.

    Every droppable Finding's id, except `acyclic` contributes only true cycle
    members (not the descendants it over-flags).
    """
    ids = {int(f.id) for f in findings if f.check != "acyclic" and f.id is not None}
    if any(f.check == "acyclic" for f in findings):
        ids |= _cycle_member_ids(ctx)
    return ids


def reduce_pedigree(ctx0: ValidationContext, *, rebuild_kwargs: dict) -> ReductionResult:
    """Iteratively drop offending individuals until the pedigree passes (fixpoint).

    Each round runs the registry (`_run_checks`, which re-imputes sex via a fresh
    context), collects droppable Findings, drops the flagged ids (deletes their
    rows and sets references to them to -1), and rebuilds the context from the
    reduced frame. The full `df_current` is carried so extra columns survive and
    the *original* declared sex tokens persist — a parent that loses its only
    role correctly becomes `unknown_sex` the next round. See docs/adr/0003.
    """
    mother_col = ctx0.mother_col
    father_col = ctx0.father_col
    df_cur = ctx0.df_raw
    n_input_rows = len(df_cur)
    ctx = ctx0
    dropped: list[tuple[int, str, int]] = []
    rows_removed = 0
    cleared_refs = 0
    drop_rounds = 0
    rnd = 0
    logger.info("--drop-offending: starting reduction loop over %d input row(s)", n_input_rows)
    while True:
        rnd += 1
        if rnd > n_input_rows + 1:
            raise PedigreeError("drop-offending failed to converge")
        _results, findings = _run_checks(ctx, on_fail="accumulate")
        droppable = [f for f in findings if f.check in DROPPABLE_CHECKS and f.id is not None]
        drop_ids = _round_drop_ids(droppable, ctx)
        if not drop_ids:
            break
        drop_rounds += 1
        # Deduped (id, check) reasons for this round — one finding may flag an
        # id twice (e.g. a self-loop via both parent columns).
        round_pairs = {(int(f.id), f.check) for f in droppable if int(f.id) in drop_ids}
        dropped.extend((fid, check, rnd) for fid, check in round_pairs)

        drop_arr = np.fromiter(drop_ids, dtype=np.int64, count=len(drop_ids))
        keep = ~np.isin(ctx.ids, drop_arr)
        surv_m = ctx.mothers[keep]
        surv_f = ctx.fathers[keep]
        m_hit = np.isin(surv_m, drop_arr)
        f_hit = np.isin(surv_f, drop_arr)
        n_round_rows = int((~keep).sum())
        n_round_cleared = int(m_hit.sum()) + int(f_hit.sum())
        rows_removed += n_round_rows
        cleared_refs += n_round_cleared

        by_check: dict[str, int] = {}
        for _fid, check in round_pairs:
            by_check[check] = by_check.get(check, 0) + 1
        logger.info(
            "drop-offending round %d: dropped %d individual(s) — %s; removed %d row(s), cleared %d reference(s)",
            rnd,
            len(drop_ids),
            ", ".join(f"{n} {check}" for check, n in sorted(by_check.items())),
            n_round_rows,
            n_round_cleared,
        )

        df_cur = df_cur[keep].reset_index(drop=True)
        df_cur.loc[m_hit, mother_col] = "-1"
        df_cur.loc[f_hit, father_col] = "-1"
        ctx = _build_context_from_df(df_cur, **rebuild_kwargs)
    return ReductionResult(
        df_current=df_cur,
        dropped=dropped,
        n_rounds=drop_rounds,
        n_input_rows=n_input_rows,
        n_rows_removed=rows_removed,
        n_distinct_dropped=len({fid for fid, _c, _r in dropped}),
        n_cleared_refs=cleared_refs,
        ctx_final=ctx,
    )
