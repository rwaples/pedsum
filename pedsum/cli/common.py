"""Runner plumbing shared by the subcommands: logging, the thread budget, and loading a validated pedigree."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from pedigree_graph import configure_threads

from pedsum.base import (
    PedigreeError,
    logger,
)
from pedsum.progress import timed
from pedsum.report import (
    write_validate_log,
)
from pedsum.validate import ValidationConfig, load_and_validate, validate_pedigree

if TYPE_CHECKING:
    import argparse

    import polars as pl

    from pedsum.ids import IdLabels


def commit_thread_budget(args: argparse.Namespace) -> None:
    """Hand ``--threads`` to pedigree-graph before its first computation.

    The library commits one budget per process and refuses a later change,
    so this runs before any graph is built. Its results do not depend on the
    value; only wall time does.
    """
    threads = getattr(args, "threads", None) or 1
    try:
        configure_threads(threads)
    except RuntimeError:
        # Already committed by an earlier call in this process (the test
        # suite drives several runs in-process); the budget stands.
        logger.debug("pedigree-graph thread budget already committed; --threads=%s ignored", threads)


def init_logging(verbose: bool, quiet: bool) -> None:
    """Log to stderr at INFO, or DEBUG under ``--verbose``, or WARNING under ``--quiet``."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
        force=True,
    )


def write_validation_failure_log(args: argparse.Namespace, config: ValidationConfig) -> None:
    """Persist a full ``validate.log`` after summarize's fail-fast validation bailed.

    ``load_and_validate`` raises on the *first* failing check carrying only a
    short sample summary, so the user otherwise gets a one-line console error
    and no record. Re-run the accumulating validator to capture every finding
    and write the same ``validate.log`` the validate subcommand produces — the
    extra pass only happens on the (already-failing) error path. Best-effort:
    a secondary failure here is logged but must not mask the original error.
    """
    log_path = args.out_dir / "validate.log"
    try:
        _, _, findings, ctx = validate_pedigree(args.in_path, config, sep=args.sep, require_birth_year_col=True)
        write_validate_log(findings, log_path, ctx.id_labels)
    except (PedigreeError, FileNotFoundError, OSError) as e:
        logger.error("could not write %s: %s", log_path, e)
        return
    logger.error("wrote %s — fix the reported issue(s) and re-run", log_path)


def load_or_fail(args: argparse.Namespace) -> tuple[pl.DataFrame, IdLabels] | int:
    """Load and validate ``--in`` fail-fast, or log why not and return the exit code.

    A failed Check returns 1, after writing the full ``validate.log``; an
    unreadable file returns 2. The ``load+validate`` phase label lets the
    benchmark profiler attribute the read and validation peak.
    """
    config = ValidationConfig.from_args(args)
    try:
        with timed("load+validate"):
            return load_and_validate(args.in_path, config, sep=args.sep)
    except PedigreeError as e:
        logger.error("validation failed: %s", e)
        write_validation_failure_log(args, config)
        return 1
    except (FileNotFoundError, OSError) as e:
        logger.error("file error: %s", e)
        return 2
