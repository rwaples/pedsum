"""Property tests for numeric input parsing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pedsum.base import PedigreeError
from pedsum.parse import _PARENT_MISSING_TOKENS, _as_birth_year_col, _as_parent_int_col, _replace_missing_with

_MISSING_TEXT_TOKENS = tuple(sorted(t for t in _PARENT_MISSING_TOKENS if t))
_INVALID_NUMERIC_TOKENS = ("abc", "12x", "1,2", "--", "M", "year2020")


@st.composite
def missing_token_variants(draw: st.DrawFn) -> str:
    """Generate case and whitespace variants of recognized missing tokens."""
    token = draw(st.sampled_from(_MISSING_TEXT_TOKENS))
    case_mode = draw(st.sampled_from(("lower", "upper", "title")))
    if case_mode == "lower":
        token = token.lower()
    elif case_mode == "upper":
        token = token.upper()
    else:
        token = token.title()
    left = draw(st.text(alphabet=" \t", max_size=3))
    right = draw(st.text(alphabet=" \t", max_size=3))
    return f"{left}{token}{right}"


@given(st.lists(missing_token_variants() | st.none(), min_size=1, max_size=50))
def test_missing_parent_tokens_parse_to_sentinel(tokens: list[str | None]) -> None:
    """Recognized parent missing tokens and null cells parse to ``-1``."""
    parsed = _as_parent_int_col(pd.Series(tokens, dtype=object), "mother")
    assert parsed.dtype == np.int64
    assert parsed.tolist() == [-1] * len(tokens)


@given(st.lists(missing_token_variants() | st.none(), min_size=1, max_size=50))
def test_replace_missing_with_uses_requested_sentinel(tokens: list[str | None]) -> None:
    """Missing-token normalization uses the caller-provided sentinel exactly."""
    sentinel = "<missing>"
    cleaned = _replace_missing_with(pd.Series(tokens, dtype=object), _PARENT_MISSING_TOKENS, sentinel)
    assert cleaned.tolist() == [sentinel] * len(tokens)


@given(st.lists(st.integers(min_value=0, max_value=1_000_000), min_size=1, max_size=100))
def test_parent_int_parser_preserves_integer_tokens(values: list[int]) -> None:
    """Valid integer parent IDs round-trip when zero is not treated as missing."""
    tokens = [f"  {value}  " for value in values]
    parsed = _as_parent_int_col(pd.Series(tokens), "father", zero_as_missing=False)
    assert parsed.tolist() == values


@given(st.lists(st.integers(min_value=0, max_value=1_000_000), min_size=1, max_size=100))
def test_parent_int_parser_can_treat_zero_as_missing(values: list[int]) -> None:
    """PLINK-style zero-as-missing maps only literal zero to ``-1``."""
    tokens = [str(value) for value in values]
    parsed = _as_parent_int_col(pd.Series(tokens), "father", zero_as_missing=True)
    expected = [-1 if value == 0 else value for value in values]
    assert parsed.tolist() == expected


@given(
    st.lists(
        st.one_of(
            st.integers(min_value=-1, max_value=3000).map(str),
            st.floats(min_value=-1, max_value=3000, allow_nan=False, allow_infinity=False).map(str),
        ),
        min_size=1,
        max_size=100,
    )
)
def test_birth_year_parser_matches_int32_cast(tokens: list[str]) -> None:
    """Birth-year parsing accepts numeric tokens and follows NumPy int32 casting."""
    series = pd.Series(tokens)
    parsed = _as_birth_year_col(series, "birth_year")
    expected = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64).astype(np.int32)
    assert parsed.dtype == np.int32
    assert np.array_equal(parsed, expected)


@given(st.sampled_from(_INVALID_NUMERIC_TOKENS))
def test_parent_int_parser_rejects_non_numeric_tokens(token: str) -> None:
    """Parent-ID parsing rejects non-numeric non-missing tokens."""
    with pytest.raises(PedigreeError):
        _as_parent_int_col(pd.Series([token]), "mother")


@given(st.sampled_from(_INVALID_NUMERIC_TOKENS))
def test_birth_year_parser_rejects_non_numeric_tokens(token: str) -> None:
    """Birth-year parsing rejects non-numeric non-missing tokens."""
    with pytest.raises(PedigreeError):
        _as_birth_year_col(pd.Series([token]), "birth_year")
