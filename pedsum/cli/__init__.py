"""The ``pedsum`` command: ``main`` runs the runner the parsed subcommand names."""

from __future__ import annotations

import shlex
import sys
import traceback
from typing import TYPE_CHECKING

from pedsum import safe_log
from pedsum.base import logger
from pedsum.cli.common import init_logging
from pedsum.cli.parser import parse_args
from pedsum.memory import MemoryWatchdog, resolve_limit
from pedsum.progress import current_profile_phase
from pedsum.report import id_log_name

if TYPE_CHECKING:
    import argparse


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    args = parse_args(argv)
    init_logging(args.verbose, args.quiet)
    cmd = shlex.join(["pedsum", *(sys.argv[1:] if argv is None else argv)])
    if not getattr(args, "safe_attempt", False):
        return _run(args, cmd)
    with safe_log.scrubbing():
        try:
            return _run(args, cmd)
        except KeyboardInterrupt, SystemExit:
            raise
        except BaseException as e:
            # An uncaught exception's message can quote a row, and the interpreter would print it.  A Rust panic
            # in pedigree-graph arrives as PyO3's PanicException, a BaseException.
            logger.error("stopped on an unexpected %s: %s", safe_log.Public(type(e).__name__), traceback.format_exc())
            return 1
        finally:
            if safe_log.anything_withheld() and not safe_log.id_log_is_open():
                logger.info(
                    "safe-attempt: this log withheld values; --id-log writes those lines in full to %s, "
                    "a file that must never be transferred",
                    args.out_dir / id_log_name(args.subcommand),
                )


def _run(args: argparse.Namespace, cmd: str) -> int:
    with MemoryWatchdog(resolve_limit(args.max_memory), phase=current_profile_phase) as watchdog:
        return args.run(args, cmd, watchdog)
