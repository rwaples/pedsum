"""Shared constants, the package logger, and the PedigreeError type."""

from __future__ import annotations

import logging

_F_KERNEL_WARN_THRESHOLD = 1_000_000

VERSION = "0.13.0"

SEX_FEMALE = 0

SEX_MALE = 1

SEX_UNKNOWN = -1

INBRED_TOL = 1e-9

logger = logging.getLogger("pedigree_summary")


class PedigreeError(Exception):
    """Raised on any input validation failure."""
