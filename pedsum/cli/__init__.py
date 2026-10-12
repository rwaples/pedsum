"""The ``pedsum`` command: ``main`` dispatches each subcommand to its runner."""

from __future__ import annotations

import shlex
import sys

from pedsum.cli.assortative_mating import run_assortative_mating
from pedsum.cli.common import init_logging
from pedsum.cli.effective_size import run_effective_size
from pedsum.cli.epimight_input import run_epimight_input
from pedsum.cli.parser import parse_args
from pedsum.cli.summarize import run_summarize
from pedsum.cli.validate import run_validate
from pedsum.memory import MemoryWatchdog, resolve_limit
from pedsum.progress import current_profile_phase


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    args = parse_args(argv)
    init_logging(args.verbose, args.quiet)
    cmd = shlex.join(["pedsum", *(sys.argv[1:] if argv is None else argv)])
    with MemoryWatchdog(resolve_limit(args.max_memory), phase=current_profile_phase) as watchdog:
        if args.subcommand == "summarize":
            return run_summarize(args, cmd)
        if args.subcommand == "effective-size":
            return run_effective_size(args, cmd, watchdog)
        if args.subcommand == "validate":
            return run_validate(args, cmd)
        if args.subcommand == "epimight-input":
            return run_epimight_input(args)
        if args.subcommand == "assortative-mating":
            return run_assortative_mating(args, cmd)
    return 1
