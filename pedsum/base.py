"""Shared constants, the package logger, and the PedigreeError type."""

from __future__ import annotations

import logging

_F_KERNEL_WARN_THRESHOLD = 1_000_000

VERSION = "0.14.0"

SEX_FEMALE = 0

SEX_MALE = 1

SEX_UNKNOWN = -1

INBRED_TOL = 1e-9

#: Ne estimators that run pedigree-graph's kinship DP, opt-in together via
#: ``--ne-coancestry``. On the 783K-row horse pedigree the DP passes 12 GiB.
KINSHIP_DP_ESTIMATORS = ("ne_coancestry", "ne_group_coancestry")

logger = logging.getLogger("pedigree_summary")


class PedigreeError(Exception):
    """Raised on any input validation failure."""
