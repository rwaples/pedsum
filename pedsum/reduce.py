"""What ``validate`` writes: BLOCKED, the Fixed Pedigree, or under ``--drop-offending`` the Reduced Pedigree.

``fix_validated`` applies each Check's Fail Policy (ADR 0005); ``reduce_pedigree``
is the ``--drop-offending`` reduction (ADR 0003).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from pedsum.base import PedigreeError, logger
from pedsum.checks import DROPPABLE_CHECKS, CheckStatus, blocking_failures
from pedsum.fixed_pedigree import fix_pedigree, next_free_id, render_fixed_pedigree
from pedsum.validate import ValidationContext, run_checks, validate_pedigree

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pedsum.checks import CheckResult, Finding
    from pedsum.fixed_pedigree import FixedPedigree

# Drop fraction (rows removed / input rows) above which --drop-offending warns.
DROP_FRACTION_WARN: float = 0.10


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

    `checks._acyclic` flags every node it cannot topologically order — cycle
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

    Each round runs the registry (`run_checks`, which re-imputes sex via a fresh
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
        _results, findings = run_checks(ctx)
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
    next_id = next_free_id(*ctx.require_id_parents())
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
        missing_parent_tokens=ctx.config.missing_parent_tokens,
    )
    return fp, write(frame, fp)
