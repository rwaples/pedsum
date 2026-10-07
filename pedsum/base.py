"""Shared constants, the package logger, and the PedigreeError type."""

from __future__ import annotations

import logging
from typing import Literal

#: Warn before F when rows times the largest ancestor set a row could have
#: passes this. Each row's Meuwissen-Luo walk visits its whole ancestor set,
#: at most ``min(2 ** (depth + 1) - 2, rows)`` rows, so **Depth** multiplies
#: the cost. 1M rows at depth 11 come to about 4.1e9 and took 29 s.
_F_WALK_WARN_VISITS = 4_000_000_000

VERSION = "0.14.0"

SEX_FEMALE = 0

SEX_MALE = 1

SEX_UNKNOWN = -1

INBRED_TOL = 1e-9

TraitKind = Literal["continuous", "binary", "ordinal"]
TRAIT_KINDS: tuple[TraitKind, ...] = ("continuous", "binary", "ordinal")

#: A sex x stratum whose pairs span fewer Mate Networks than this is dropped from its assortative-mating cell.
MIN_STRATUM_NETWORKS = 10

logger = logging.getLogger("pedigree_summary")


class PedigreeError(Exception):
    """Raised on any input validation failure."""
