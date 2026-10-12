"""Helpers shared by the Offspring Sex Concordance test files."""

from __future__ import annotations

import numpy as np
import polars as pl

GROUPINGS = ("sibship", "maternal_offspring_group", "paternal_offspring_group")


def offspring_frame(mothers, fathers, sexes, sources=None) -> pl.DataFrame:
    """Minimal validated-frame stand-in: the four columns the analysis reads."""
    n = len(mothers)
    return pl.DataFrame(
        {
            "mother": np.asarray(mothers, dtype=np.int64),
            "father": np.asarray(fathers, dtype=np.int64),
            "sex": np.asarray(sexes, dtype=np.int8),
            "sex_source": np.asarray(["input"] * n if sources is None else sources, dtype=object),
        },
    )


def uniform_groups(n_groups: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (mothers, fathers) for ``n_groups`` disjoint groups of ``size``."""
    mothers = np.repeat(np.arange(1, n_groups + 1) * 10, size)
    return mothers, mothers + 1
