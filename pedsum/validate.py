"""Validation: a declarative Check registry driven by one dual-mode runner.

`load_and_validate` (summarize, fail-fast) and `validate_pedigree` (validate,
accumulating) share one `CHECKS` registry and one `_run_checks` runner — see
`docs/adr/0002-validation-check-registry.md`. Each Check carries its
prerequisites and label; `_CHECK_ORDER` and `_CHECK_LABELS` are derived
from the registry. `_CHECK_GROUPS` is a curated *display* order (it intentionally
differs from execution order) and stays explicit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, PedigreeError, logger
from pedsum.checks import CHECKS, DROPPABLE_CHECKS, CheckResult, CheckStatus, _summarize_findings, blocking_failures
from pedsum.fixed_pedigree import _next_free_id, fix_pedigree, render_fixed_pedigree
from pedsum.parse import (
    _BIRTH_YEAR_DEFAULT_MIN,
    _birth_year_default_max,
    _maybe_warn_csv,
    _read_pedigree_table,
)
from pedsum.pedigree_ops import IdIndex, ParentRefs, _parent_rows, _structural_depth

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable
    from pathlib import Path

    from pedsum.checks import Finding
    from pedsum.fixed_pedigree import FixedPedigree
    from pedsum.ids import IdLabels


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
    mother_ids: np.ndarray,
    father_ids: np.ndarray,
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

    ``mother_ids`` / ``father_ids`` are the distinct ids referenced as mother
    / father (``ParentRefs.ids``).

    Returns a `_SexImputation` whose `sex_source` is a per-row `dtype=object`
    array tagging each row's provenance: ``"input"`` / ``"imputed_from_missing"``
    / ``"imputed_from_role"`` / ``"unresolved"``.
    """
    original_unknown_mask = (sex == SEX_UNKNOWN).copy()
    imputed = sex.copy()
    sex_source = np.full(len(sex), "input", dtype=object)

    # Membership by id, not by ParentRefs.rows: with duplicate ids every copy
    # of a referenced id takes the role, not just the first row.
    as_mother = np.isin(ids, mother_ids)
    as_father = np.isin(ids, father_ids)

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
# ValidationConfig, ValidationContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationConfig:
    """The column names, tolerances and birth-year bounds a validation run takes from the CLI."""

    id_col: str = "id"
    sex_col: str = "sex"
    mother_col: str = "mother"
    father_col: str = "father"
    birth_year_col: str | None = None
    birth_year_min: int = _BIRTH_YEAR_DEFAULT_MIN
    birth_year_max: int | None = None
    allow_missing_sex: bool = False
    override_asserted_sex: bool = True
    # --no-sex-check tolerates parent_refs_sex_conflict (SKIP). It lives here,
    # not as a CLI post-filter, so the tolerance composes wherever _run_checks
    # runs: validate, the drop-offending loop, and its self-verify.
    no_sex_check: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ValidationConfig:
        """The config of a parsed command line; only ``validate`` has ``--no-sex-check``."""
        return cls(
            id_col=args.id_col,
            sex_col=args.sex_col,
            mother_col=args.mother_col,
            father_col=args.father_col,
            birth_year_col=args.birth_year_col,
            birth_year_min=args.birth_year_min,
            birth_year_max=args.birth_year_max,
            allow_missing_sex=args.allow_missing_sex,
            override_asserted_sex=not args.no_override_asserted_sex,
            no_sex_check=getattr(args, "no_sex_check", False),
        )

    @property
    def by_max(self) -> int:
        """Resolved birth-year upper bound (default: current calendar year + 1)."""
        return self.birth_year_max if self.birth_year_max is not None else _birth_year_default_max()


@dataclass
class ValidationContext:
    """Mutable state threaded through every Check (``pedsum.checks``).

    Parse Checks populate the array fields (parsing *is* the Check); all other
    Checks read them. ``id_index`` and ``imputation`` are derived once on first
    access.
    """

    df_raw: pl.DataFrame
    config: ValidationConfig
    # summarize's run: raise on the first FAIL, and require a named
    # --birth-year-col to exist. Otherwise every FAIL is recorded and a missing
    # birth-year column is an inapplicable SKIP.
    fail_fast: bool = False
    # The --drop-offending rounds parse with the input's labels, so the ID
    # mode and string-mode codes stay fixed while rows are dropped.
    fixed_labels: IdLabels | None = None
    # Populated by the parse Checks.
    ids: np.ndarray | None = None
    mothers: np.ndarray | None = None
    fathers: np.ndarray | None = None
    id_labels: IdLabels | None = None
    sex: np.ndarray | None = None
    birth_year: np.ndarray | None = None

    @property
    def label(self) -> Callable[[int], str]:
        """Prints an id code as the input's ID; set by the id parse Check."""
        assert self.id_labels is not None
        return self.id_labels.label

    @cached_property
    def id_index(self) -> IdIndex:
        """Row lookup over the parsed IDs; built once (IDs are fixed after parse)."""
        return IdIndex(self.ids)

    @cached_property
    def mother_refs(self) -> ParentRefs:
        """Distinct mother ids and their rows; built once, shared by the parent and sex checks."""
        assert self.mothers is not None
        return ParentRefs.of(self.mothers, self.id_index)

    @cached_property
    def father_refs(self) -> ParentRefs:
        """Distinct father ids and their rows; built once, shared by the parent and sex checks."""
        assert self.fathers is not None
        return ParentRefs.of(self.fathers, self.id_index)

    @cached_property
    def depth(self) -> np.ndarray:
        """Topological depth per parsed row, -1 in or below a cycle; built once."""
        assert self.mothers is not None
        assert self.fathers is not None
        m_row, _ = _parent_rows(self.mothers, self.id_index)
        f_row, _ = _parent_rows(self.fathers, self.id_index)
        return _structural_depth(m_row, f_row)

    @cached_property
    def imputation(self) -> _SexImputation:
        """Sex imputation from parent-role usage; computed once, shared by the sex checks."""
        assert self.sex is not None
        assert self.ids is not None
        return _impute_sex_from_roles(
            self.sex,
            self.ids,
            self.mother_refs.ids,
            self.father_refs.ids,
            override_asserted_sex=self.config.override_asserted_sex,
        )

    def get_imputation(self) -> _SexImputation | None:
        """Return the imputation only if already computed (never triggers it)."""
        return self.__dict__.get("imputation")

    def require_id_parents(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the parsed ``(ids, mothers, fathers)``.

        Asserts the id/parent parse Checks populated them — guaranteed for any
        consuming Check by the ``requires`` gating in :func:`_run_checks`. One
        guarded accessor for the Checks that need the full pedigree.
        """
        assert self.ids is not None
        assert self.mothers is not None
        assert self.fathers is not None
        return self.ids, self.mothers, self.fathers


# Drop fraction (rows removed / input rows) above which --drop-offending warns.
DROP_FRACTION_WARN: float = 0.10

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_checks(ctx: ValidationContext) -> tuple[dict[str, CheckResult], list[Finding]]:
    """Run every Check in registry order against ``ctx``.

    A Check whose prerequisite did not ``PASS`` is auto-``SKIP``ped: a FAILed
    prerequisite yields ``"<prereq> failed"``, a SKIPped one propagates its own
    reason (so ``required_columns failed`` / ``no --birth-year-col specified``
    flow transitively). A ``fail_fast`` context raises on the first ``FAIL``
    (summarize); otherwise each is recorded and the run continues (validate).
    """
    results: dict[str, CheckResult] = {}
    findings: list[Finding] = []
    for check in CHECKS:
        skip_note = None
        for prereq in check.requires:
            r = results[prereq]
            if r.status != CheckStatus.PASS:
                skip_note = r.note if r.status == CheckStatus.SKIP else f"{prereq} failed"
                break
        result = check.run(ctx) if skip_note is None else CheckResult(CheckStatus.SKIP, note=skip_note)
        result.name = check.name
        for f in result.findings:
            f.check = check.name
        if result.status == CheckStatus.FAIL:
            if ctx.fail_fast:
                # The id parse Check runs before any Check that names an id.
                label = ctx.label if ctx.id_labels is not None else str
                raise PedigreeError(_summarize_findings(result.findings, label))
            findings.extend(result.findings)
        results[check.name] = result
    return results, findings


# ---------------------------------------------------------------------------
# Public drivers
# ---------------------------------------------------------------------------


def load_and_validate(path: Path, config: ValidationConfig, *, sep: str = "auto") -> tuple[pl.DataFrame, IdLabels]:
    """Load TSV, run all QC fail-fast, return the validated pedigree and its ID labels.

    The frame's id, mother and father columns hold int64 ID codes, which the
    labels print as the input's IDs (:mod:`pedsum.ids`); in integer mode a
    code is the ID. The frame has columns id, sex (int8), mother, father, sex_source,
    ``birth_year`` (int32, sentinel -1) when ``birth_year_col`` is set and
    present, and ``ped_depth`` (int32 topological depth, founders 0). Missing
    parent encoded as -1. Raises ``PedigreeError`` on the first failing Check.

    Rows are topologically sorted (parents before children). pedigree-graph
    accepts rows in any order, so the sort is pedsum's own output contract, not
    a construction precondition. The frame is the Fixed Pedigree's
    (:mod:`pedsum.fixed_pedigree`): the same rows, sex and order ``validate``
    writes for this input.
    """
    t0 = time.perf_counter()
    df = _read_pedigree_table(path, sep=sep, dtype=str)
    logger.info("read %d rows from %s in %.2fs", len(df), path, time.perf_counter() - t0)
    _maybe_warn_csv(df)
    ctx = ValidationContext(df, config, fail_fast=True)
    _run_checks(ctx)
    # Every Check passed, so no parent is missing and there is no founder to
    # synthesize: the frame is the input rows of the pedigree validate would write.
    fp = fix_pedigree(ctx, synthesize_founders=False)
    logger.info("validated %d rows in %.2fs", len(fp.ids), time.perf_counter() - t0)
    return fp.frame(), fp.labels


def validate_pedigree(
    path: Path, config: ValidationConfig, *, sep: str = "auto"
) -> tuple[int, list[CheckResult], list[Finding], ValidationContext]:
    """Run every Check accumulating.

    Returns ``(n_rows, results, findings, ctx)`` where ``results`` covers all
    checks in ``_CHECK_ORDER`` (each PASS / FAIL / SKIP), ``findings`` lists
    every per-individual finding, and ``ctx`` carries the coerced arrays and raw
    DataFrame for ``_run_validate``'s auto-fix.
    """
    df = _read_pedigree_table(path, sep=sep, dtype=str)
    _maybe_warn_csv(df)
    ctx = ValidationContext(df, config)
    results, findings = _run_checks(ctx)
    return len(ctx.df_raw), list(results.values()), findings, ctx


# ---------------------------------------------------------------------------
# Reduction: validate --drop-offending (see docs/adr/0003)
# ---------------------------------------------------------------------------


@dataclass
class ReductionResult:
    """Outcome of `reduce_pedigree`: the reduced frame plus a drop record."""

    df_current: pl.DataFrame
    dropped: list[tuple[str, str, int]]  # one per distinct (id, check, round), the id as the input wrote it
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

    ids, mothers, fathers = ctx.require_id_parents()
    n = len(ids)
    id_index = ctx.id_index
    rows: list[int] = []
    cols: list[int] = []
    for parents in (mothers, fathers):
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


def reduce_pedigree(ctx0: ValidationContext) -> ReductionResult:
    """Iteratively drop offending individuals until the pedigree passes (fixpoint).

    Each round runs the registry (`_run_checks`, which re-imputes sex via a fresh
    context), collects droppable Findings, drops the flagged ids (deletes their
    rows and sets references to them to -1), and rebuilds the context from the
    reduced frame. The full `df_current` is carried so extra columns survive and
    the *original* declared sex tokens persist — a parent that loses its only
    role correctly becomes `unknown_sex` the next round. See docs/adr/0003.
    """
    mother_col = ctx0.config.mother_col
    father_col = ctx0.config.father_col
    df_cur = ctx0.df_raw
    n_input_rows = len(df_cur)
    ctx = ctx0
    ctx0.fixed_labels = ctx0.id_labels
    dropped: list[tuple[str, str, int]] = []
    rows_removed = 0
    cleared_refs = 0
    drop_rounds = 0
    rnd = 0
    logger.info("--drop-offending: starting reduction loop over %d input row(s)", n_input_rows)
    while True:
        rnd += 1
        if rnd > n_input_rows + 1:
            raise PedigreeError("drop-offending failed to converge")
        _results, findings = _run_checks(ctx)
        droppable = [f for f in findings if f.check in DROPPABLE_CHECKS and f.id is not None]
        drop_ids = _round_drop_ids(droppable, ctx)
        if not drop_ids:
            break
        drop_rounds += 1
        # Deduped (id, check) reasons for this round — one finding may flag an
        # id twice (e.g. a self-loop via both parent columns). Sorted, because
        # str hashing varies per process and the manifest keeps this order.
        round_pairs = sorted({(f.id, f.check) for f in droppable if f.id is not None and f.id in drop_ids})
        # The manifest writes IDs as the input wrote them.
        dropped.extend((ctx.label(fid), check, rnd) for fid, check in round_pairs)

        drop_arr = np.fromiter(drop_ids, dtype=np.int64, count=len(drop_ids))
        ids, mothers, fathers = ctx.require_id_parents()
        keep = ~np.isin(ids, drop_arr)
        surv_m = mothers[keep]
        surv_f = fathers[keep]
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

        df_cur = (
            df_cur.filter(pl.Series(keep))
            .with_columns(pl.Series("__m_hit", m_hit), pl.Series("__f_hit", f_hit))
            .with_columns(
                pl.when(pl.col("__m_hit")).then(pl.lit("-1")).otherwise(pl.col(mother_col)).alias(mother_col),
                pl.when(pl.col("__f_hit")).then(pl.lit("-1")).otherwise(pl.col(father_col)).alias(father_col),
            )
            .drop("__m_hit", "__f_hit")
        )
        ctx = ValidationContext(df_cur, ctx0.config, fixed_labels=ctx0.id_labels)
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


@dataclass(frozen=True)
class ValidateOutcome:
    """What ``validate`` made of a validated input: BLOCKED, or a Fixed Pedigree written."""

    exit_code: int
    # Set when BLOCKED: the reason after "BLOCKED — " and one line per cause.
    # Nothing was written.
    blocked: str | None = None
    block_lines: list[str] = field(default_factory=list)
    fixed: FixedPedigree | None = None
    # --drop-offending only: the reduction when it removed anyone, else
    # nothing_to_drop (the input already passed).
    reduction: ReductionResult | None = None
    nothing_to_drop: bool = False


def fix_validated(
    ctx: ValidationContext,
    results: list[CheckResult],
    *,
    drop_offending: bool,
    fill_half_founders: bool,
    write_pedigree: Callable[[pl.DataFrame, FixedPedigree], Path],
    write_manifest: Callable[[list[tuple[str, str, int]]], None],
) -> ValidateOutcome:
    """Decide what ``validate`` writes for ``ctx`` and its ``results``, and write it.

    BLOCKs per each Check's Fail Policy (:func:`blocking_failures`). Otherwise
    builds the Fixed Pedigree, after reducing it to a passing one under
    ``--drop-offending`` (ADR 0003), renders it, and hands the frame to
    ``write_pedigree``, which returns the path written. Under
    ``--drop-offending`` the removal manifest goes to ``write_manifest``
    first, so it is on disk even if the self-verify fails: the Reduced
    Pedigree is re-validated from the written file under the same flags, and
    a FAIL there is a bug and raises ``PedigreeError``.

    Exit codes: 2 when BLOCKED. Otherwise plain ``validate`` exits 1 when any
    Check FAILed, and ``--drop-offending`` exits 1 when it dropped anyone; else 0.
    """
    by_check = {r.name: r for r in results}
    blocks = blocking_failures(by_check, drop_offending=drop_offending)
    if blocks:
        reason = (
            "--drop-offending cannot fix these by removing individuals:"
            if drop_offending
            else "fix the following before re-running:"
        )
        return ValidateOutcome(exit_code=2, blocked=reason, block_lines=[msg for _name, msg in blocks])

    if not drop_offending:
        fp, _path = _write_fixed(ctx, fill_half_founders=fill_half_founders, next_id=None, write=write_pedigree)
        failed = any(r.status == CheckStatus.FAIL for r in results)
        return ValidateOutcome(exit_code=1 if failed else 0, fixed=fp)

    if not any(by_check[n].status == CheckStatus.FAIL for n in DROPPABLE_CHECKS if n in by_check):
        write_manifest([])
        fp, _path = _write_fixed(ctx, fill_half_founders=fill_half_founders, next_id=None, write=write_pedigree)
        return ValidateOutcome(exit_code=0, fixed=fp, nothing_to_drop=True)

    reduction = reduce_pedigree(ctx)
    if len(reduction.df_current) == 0:
        return ValidateOutcome(exit_code=2, blocked="--drop-offending removed every individual (empty pedigree)")
    write_manifest(reduction.dropped)
    # Number phantoms past the input's IDs so none reuses a dropped ID.
    next_id = _next_free_id(*ctx.require_id_parents())
    fp, path = _write_fixed(
        reduction.ctx_final, fill_half_founders=fill_half_founders, next_id=next_id, write=write_pedigree
    )

    # Self-verify the written file under the invoked flags. It is always
    # tab-separated whatever the input's separator, so sniff it.
    _n, vresults, _vf, _vctx = validate_pedigree(path, ctx.config, sep="auto")
    still_failing = sorted(r.name for r in vresults if r.status == CheckStatus.FAIL)
    if still_failing:
        raise PedigreeError(f"--drop-offending self-verify failed; reduced pedigree still FAILs: {still_failing}")
    return ValidateOutcome(exit_code=1, fixed=fp, reduction=reduction)


def _write_fixed(
    ctx: ValidationContext,
    *,
    fill_half_founders: bool,
    next_id: int | None,
    write: Callable[[pl.DataFrame, FixedPedigree], Path],
) -> tuple[FixedPedigree, Path]:
    fp = fix_pedigree(ctx, fill_half_founders=fill_half_founders, next_id=next_id)
    frame = render_fixed_pedigree(
        fp,
        ctx.df_raw,
        id_col=ctx.config.id_col,
        sex_col=ctx.config.sex_col,
        mother_col=ctx.config.mother_col,
        father_col=ctx.config.father_col,
    )
    return fp, write(frame, fp)
