"""Property-based tests for the inbreeding histogram in ``pedsum.sections``.

``_build_inbreeding_summary`` bins per-individual F into a PMF that must sum to 1.
The ``@example`` near ``INBRED_TOL`` pins the boundary regression: before the
``edges`` fix, F in ``(0, 1e-9]`` was counted in both the zero bucket and the
first range bucket, so the histogram summed above 1.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from pedsum.base import INBRED_TOL
from pedsum.sections import _build_inbreeding_summary

_F_VALUES = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@settings(deadline=None)
@given(values=st.lists(_F_VALUES, max_size=60))
@example(values=[5e-10])
@example(values=[5e-10, 0.03, 0.2, 0.9, 0.0])
def test_inbreeding_histogram_partitions(values: list[float]) -> None:
    """The histogram sums to 1, frac/n_inbred are consistent, and mean/max match the data."""
    arr = np.array(values, dtype=float)
    out = _build_inbreeding_summary(arr)
    n = len(arr)

    if n == 0:
        assert out["n_inbred"] == 0
        assert out["frac_inbred"] == 0.0
        assert sum(out["hist"].values()) == 0.0
        return

    assert sum(out["hist"].values()) == pytest.approx(1.0)
    assert all(0.0 <= bucket <= 1.0 for bucket in out["hist"].values())
    assert 0.0 <= out["frac_inbred"] <= 1.0
    assert out["n_inbred"] == int((arr > INBRED_TOL).sum())
    assert out["frac_inbred"] == pytest.approx(out["n_inbred"] / n)
    assert out["mean_F"] == pytest.approx(float(arr.mean()))
    assert out["max_F"] == pytest.approx(float(arr.max()))
