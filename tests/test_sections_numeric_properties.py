"""Property-based tests for numeric summaries in ``pedsum.sections``.

``_numeric_distribution`` must produce ordered quantiles with the mean inside the
support and be permutation-invariant; ``_effective_count_from_weights`` must obey
its closed form ``1 / sum(p^2)`` and the bound ``0 <= Ne <= n_positive``. Inputs
are bounded, finite numeric arrays so NumPy reductions cannot overflow.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pedsum.sections import _effective_count_from_weights, _numeric_distribution

_FINITE_FLOATS = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
_NONNEG_FLOATS = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


def _assert_distribution_invariants(d: dict, n: int) -> None:
    """Assert the ordering, support, and count invariants of a distribution dict."""
    if n == 0:
        assert d["mean"] == 0.0
        assert d["std"] == 0.0
        assert d["nz"] == 0
        return
    assert d["min"] <= d["q1"] <= d["median"] <= d["q3"] <= d["max"]
    assert d["min"] - 1e-6 <= d["mean"] <= d["max"] + 1e-6
    assert d["std"] >= 0.0
    assert 0 <= d["nz"] <= n


@settings(deadline=None)
@given(vals=st.lists(_FINITE_FLOATS, max_size=50))
def test_numeric_distribution_float(vals: list[float]) -> None:
    """Float distributions are ordered, bounded, and invariant to input order."""
    arr = np.array(vals, dtype=float)
    d = _numeric_distribution(arr)
    _assert_distribution_invariants(d, len(vals))

    reversed_d = _numeric_distribution(arr[::-1])
    for key, value in d.items():
        assert reversed_d[key] == pytest.approx(value)


@settings(deadline=None)
@given(vals=st.lists(st.integers(min_value=-10_000, max_value=10_000), max_size=50))
def test_numeric_distribution_int(vals: list[int]) -> None:
    """Integer distributions keep the quantile ordering despite truncating casts."""
    arr = np.array(vals, dtype=np.int64)
    _assert_distribution_invariants(_numeric_distribution(arr), len(vals))


@settings(deadline=None)
@given(weights=st.lists(_NONNEG_FLOATS, max_size=50))
def test_effective_count_bounds(weights: list[float]) -> None:
    """Ne is in ``[0, n_positive]`` and is exactly 0 when no weight is positive."""
    arr = np.array(weights, dtype=float)
    ne = _effective_count_from_weights(arr)
    n_positive = int((arr > 0).sum())
    assert ne >= 0.0
    assert ne <= n_positive + 1e-6
    if n_positive == 0:
        assert ne == 0.0


@settings(deadline=None)
@given(k=st.integers(min_value=1, max_value=50), weight=st.floats(min_value=1e-3, max_value=1e3))
def test_effective_count_equal_weights(k: int, weight: float) -> None:
    """Equal positive weights give an effective count equal to the number of weights."""
    arr = np.full(k, weight, dtype=float)
    assert _effective_count_from_weights(arr) == pytest.approx(k)


@settings(deadline=None)
@given(k=st.integers(min_value=1, max_value=20), positive=st.floats(min_value=1e-3, max_value=1e6))
def test_effective_count_single_dominant(k: int, positive: float) -> None:
    """One positive weight among zeros gives an effective count of exactly 1."""
    arr = np.zeros(k, dtype=float)
    arr[0] = positive
    assert _effective_count_from_weights(arr) == pytest.approx(1.0)
