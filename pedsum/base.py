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

VERSION = "0.18.0"

SEX_FEMALE = 0

SEX_MALE = 1

SEX_UNKNOWN = -1

INBRED_TOL = 1e-9

#: Estimators that need every row's sex resolved (pedigree-graph raises
#: ``MissingMetadataError`` for them otherwise). The others run on a pedigree
#: with unknown sex, so ``effective-size`` refuses only when one is selected.
SEX_DEPENDENT_ESTIMATORS = frozenset({"ne_sex_ratio", "ne_variance_family_size", "ne_hill_overlapping"})

TraitKind = Literal["continuous", "binary", "ordinal"]
TRAIT_KINDS: tuple[TraitKind, ...] = ("continuous", "binary", "ordinal")

#: A sex x stratum whose pairs span fewer Mate Networks than this is dropped from its assortative-mating cell.
MIN_STRATUM_NETWORKS = 10

#: Full sibs of one Multiple-Birth Set with adjacent birth dates more than this many days apart were born apart.
BIRTHS_APART_DAYS = 7

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
    def sex_name(self) -> str:
        """``expected_sex`` as a word."""
        return "female" if self is ParentRole.MOTHER else "male"

    @property
    def other(self) -> ParentRole:
        """The other parent role."""
        return ParentRole.FATHER if self is ParentRole.MOTHER else ParentRole.MOTHER


#: Both parent roles, mother first: the order every per-role Check reports in.
PARENT_ROLES = (ParentRole.MOTHER, ParentRole.FATHER)


class PedigreeError(Exception):
    """Raised on any input validation failure.

    ``public`` is what a ``--safe-attempt`` log prints in place of the
    message: check and column names, no ID and no data value. True says the
    message itself is that. Without it the message is withheld there
    (:mod:`pedsum.safe_log`).
    """

    def __init__(self, message: str, *, public: str | bool = False) -> None:
        """Keep ``message`` as the error text and resolve ``public``."""
        super().__init__(message)
        self.public: str | None = message if public is True else (public or None)
