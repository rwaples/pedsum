"""The ``pedsum`` command: ``main`` runs the runner the parsed subcommand names."""

from __future__ import annotations

import shlex
import sys

from pedsum.cli.common import init_logging
from pedsum.cli.parser import parse_args
from pedsum.memory import MemoryWatchdog, resolve_limit
from pedsum.progress import current_profile_phase


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    args = parse_args(argv)
    init_logging(args.verbose, args.quiet)
    cmd = shlex.join(["pedsum", *(sys.argv[1:] if argv is None else argv)])
    with MemoryWatchdog(resolve_limit(args.max_memory), phase=current_profile_phase) as watchdog:
        return args.run(args, cmd, watchdog)
