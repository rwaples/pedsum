"""``validate``: every Check accumulating, ``validate.log``, then BLOCKED or the Fixed or Reduced Pedigree."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from pedsum.base import (
    PedigreeError,
    logger,
)
from pedsum.cli.common import commit_thread_budget
from pedsum.ids import PHANTOM_PREFIX
from pedsum.reduce import DROP_FRACTION_WARN, fix_validated
from pedsum.report import (
    _format_check_summary,
    _prepare_out_dir,
    _to_csv_gz,
    _write_dropped_manifest,
    _write_validate_log,
)
from pedsum.validate import ValidationConfig, validate_pedigree

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    import polars as pl

    from pedsum.fixed_pedigree import FixedPedigree
    from pedsum.reduce import ReductionResult


def _report_reduction(result: ReductionResult) -> None:
    """Write the ``--drop-offending`` reduction summary to stderr."""
    pct = 100.0 * result.n_rows_removed / result.n_input_rows
    sys.stderr.write(
        f"--drop-offending: dropped {result.n_distinct_dropped} individual(s) / "
        f"{result.n_rows_removed} row(s) of {result.n_input_rows} ({pct:.1f}%) over "
        f"{result.n_rounds} round(s); cleared {result.n_cleared_refs} reference(s)\n"
    )
    if result.n_rows_removed / result.n_input_rows > DROP_FRACTION_WARN:
        logger.warning(
            "--drop-offending removed %.1f%% of rows; relatedness / Ne / founder counts reflect the reduced set",
            pct,
        )


def _remove_stale_validate_outputs(out_dir: Path) -> None:
    """Delete a previous run's ``validate.tsv.gz`` / ``validate.dropped.tsv`` from ``out_dir``.

    This run writes whichever of them applies; a BLOCKED run writes neither,
    so a leftover file would otherwise pass for this input's result.
    """
    for name in ("validate.tsv.gz", "validate.dropped.tsv"):
        path = out_dir / name
        if path.is_file():
            path.unlink()
            logger.info("removed %s from a previous run", path)


def run_validate(args: argparse.Namespace, cmd: str) -> int:
    """``validate``: run every Check, write ``validate.log``, then BLOCK or write ``validate.tsv.gz``."""
    commit_thread_budget(args)
    if _prepare_out_dir(args.out_dir) != 0:
        return 1
    _remove_stale_validate_outputs(args.out_dir)
    try:
        n_total, results, findings, ctx = validate_pedigree(
            args.in_path, ValidationConfig.from_args(args), sep=args.sep
        )
    except PedigreeError as e:
        logger.error("validation could not run: %s", e)
        return 2
    except (FileNotFoundError, OSError) as e:
        logger.error("file error: %s", e)
        return 2

    if args.fill_half_founders and ctx.id_labels is not None and (clashes := ctx.id_labels.phantom_clashes()):
        logger.error(
            "--fill-half-founders names phantom parents %s1, %s2, ...; the input already has IDs with that prefix: %s",
            PHANTOM_PREFIX,
            PHANTOM_PREFIX,
            clashes,
        )
        return 2

    sys.stderr.write(_format_check_summary(args.in_path, n_total, results))

    out_dir = args.out_dir
    log_path = out_dir / "validate.log"
    _write_validate_log(findings, log_path, ctx.id_labels)
    sys.stderr.write(f"wrote {log_path} ({len(findings)} finding(s))\n")

    def write_pedigree(frame: pl.DataFrame, fp: FixedPedigree) -> Path:
        out_path = out_dir / "validate.tsv.gz"
        _to_csv_gz(frame, out_path)
        sys.stderr.write(f"wrote {out_path} ({len(frame):,} rows; {len(fp.founders)} founder(s) added)\n")
        return out_path

    def write_manifest(dropped: list[tuple[str, str, int]]) -> None:
        manifest_path = out_dir / "validate.dropped.tsv"
        _write_dropped_manifest(dropped, manifest_path)
        if dropped:
            n_ids = len({fid for fid, _check, _round in dropped})
            sys.stderr.write(f"wrote {manifest_path} ({n_ids} id(s) dropped)\n")

    outcome = fix_validated(
        ctx,
        results,
        drop_offending=args.drop_offending,
        fill_half_founders=args.fill_half_founders,
        write_pedigree=write_pedigree,
        write_manifest=write_manifest,
    )
    if outcome.blocked is not None:
        sys.stderr.write(f"\nBLOCKED — {outcome.blocked}\n")
        for line in outcome.block_lines:
            sys.stderr.write(f"  - {line}\n")
    elif outcome.reduction is not None:
        _report_reduction(outcome.reduction)
    elif outcome.nothing_to_drop:
        sys.stderr.write("--drop-offending: nothing to drop; pedigree already passes\n")
    return outcome.exit_code
