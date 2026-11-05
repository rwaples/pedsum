"""Shared constants, the package logger, and the PedigreeError type."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Literal

#: Warn before F when rows times the largest ancestor set a row could have
#: passes this. Each row's Meuwissen-Luo walk visits its whole ancestor set,
#: at most ``min(2 ** (depth + 1) - 2, rows)`` rows, so **Depth** multiplies
#: the cost. 1M rows at depth 11 come to about 4.1e9 and took 29 s.
F_WALK_WARN_VISITS = 4_000_000_000

VERSION = "0.16.1"

SEX_FEMALE = 0

SEX_MALE = 1

SEX_UNKNOWN = -1

INBRED_TOL = 1e-9

TraitKind = Literal["continuous", "binary", "ordinal"]
TRAIT_KINDS: tuple[TraitKind, ...] = ("continuous", "binary", "ordinal")

#: A sex x stratum whose pairs span fewer Mate Networks than this is dropped from its assortative-mating cell.
MIN_STRATUM_NETWORKS = 10

logger = logging.getLogger("pedigree_summary")


class SexSource(StrEnum):
    """Where a row's resolved sex came from: the ``sex_source`` column, in report order."""

    INPUT = "input"
    IMPUTED_FROM_MISSING = "imputed_from_missing"
    IMPUTED_FROM_ROLE = "imputed_from_role"
    UNRESOLVED = "unresolved"


class ParentRole(StrEnum):
    """A parent column's role; its value is the role's name in Findings and the cleared-links manifest."""

    MOTHER = "mother"
    FATHER = "father"

    @property
    def expected_sex(self) -> int:
        """The sex code a parent in this role has."""
        return SEX_FEMALE if self is ParentRole.MOTHER else SEX_MALE

    @property
    def sex_letter(self) -> str:
        """``expected_sex`` as the input writes it."""
        return "F" if self is ParentRole.MOTHER else "M"

    @property
    def other(self) -> ParentRole:
        """The other parent role."""
        return ParentRole.FATHER if self is ParentRole.MOTHER else ParentRole.MOTHER


#: Both parent roles, mother first: the order every per-role Check reports in.
PARENT_ROLES = (ParentRole.MOTHER, ParentRole.FATHER)


class PedigreeError(Exception):
    """Raised on any input validation failure."""
