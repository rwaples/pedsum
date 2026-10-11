#!/usr/bin/env python3
"""Pedigree summary CLI — thin entry point for an uninstalled checkout.

The implementation lives in the :mod:`pedsum` package, and an install
provides the ``pedsum`` command (and ``python -m pedsum``). This module stays
a runnable script (``python pedigree_summary.py summarize|validate ...``) and
re-exports the symbols the test suite imports directly. See ``pedsum/`` for
the modules and ``DESIGN.md`` for the maintainer map.
"""

from __future__ import annotations

import sys

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, PedigreeError
from pedsum.checks import CheckResult, Finding, _summarize_findings
from pedsum.cli import main
from pedsum.fixed_pedigree import _build_added_founders
from pedsum.pairs import _build_pedigree_graph
from pedsum.parse import _decode_sex
from pedsum.report import (
    _apply_safe_attempt,
    _build_individual_data,
    _format_check_summary,
)
from pedsum.schema import KNOWN_YAML_DROPS, _split_summary
from pedsum.sections import (
    compute_aggregate_sections,
    compute_founder_summary,
    compute_size_structure,
)
from pedsum.validate import _CHECK_ORDER, load_and_validate, validate_pedigree

__all__ = [
    "KNOWN_YAML_DROPS",
    "SEX_FEMALE",
    "SEX_MALE",
    "SEX_UNKNOWN",
    "_CHECK_ORDER",
    "CheckResult",
    "Finding",
    "PedigreeError",
    "_apply_safe_attempt",
    "_build_added_founders",
    "_build_individual_data",
    "_build_pedigree_graph",
    "_decode_sex",
    "_format_check_summary",
    "_split_summary",
    "_summarize_findings",
    "compute_aggregate_sections",
    "compute_founder_summary",
    "compute_size_structure",
    "load_and_validate",
    "main",
    "validate_pedigree",
]


if __name__ == "__main__":
    sys.exit(main())
