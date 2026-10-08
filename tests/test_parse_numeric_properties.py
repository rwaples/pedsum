"""Property tests for ID and numeric input parsing."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pedsum.base import PedigreeError
from pedsum.ids import _plain_ints, parse_ids
from pedsum.parse import _PARENT_MISSING_TOKENS, _as_birth_year_col, _replace_missing_with

_MISSING_TEXT_TOKENS = tuple(sorted(t for t in _PARENT_MISSING_TOKENS if t))
_INVALID_NUMERIC_TOKENS = ("abc", "12x", "1,2", "--", "M", "year2020")


def _parse_parents(tokens: list, *, zero_as_missing: bool = False) -> np.ndarray:
    """Mother codes for ``tokens``, beside integer ids 1..n and no fathers."""
    n = len(tokens)
    ids = pl.Series([str(i) for i in range(1, n + 1)])
    _, mothers, _, _ = parse_ids(
        ids, pl.Series(tokens, dtype=pl.String), pl.Series(["-1"] * n), zero_as_missing=zero_as_missing
    )
    return mothers


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
    parsed = _parse_parents(tokens)
    assert parsed.dtype == np.int64
    assert parsed.tolist() == [-1] * len(tokens)


@given(st.lists(missing_token_variants() | st.none(), min_size=1, max_size=50))
def test_replace_missing_with_uses_requested_sentinel(tokens: list[str | None]) -> None:
    """Missing-token normalization uses the caller-provided sentinel exactly."""
    sentinel = "<missing>"
    cleaned = _replace_missing_with(pl.Series(tokens, dtype=pl.String), _PARENT_MISSING_TOKENS, sentinel)
    assert cleaned.to_list() == [sentinel] * len(tokens)


@given(st.lists(st.integers(min_value=0, max_value=1_000_000), min_size=1, max_size=100))
def test_parent_int_parser_preserves_integer_tokens(values: list[int]) -> None:
    """Valid integer parent IDs round-trip when zero is not treated as missing."""
    tokens = [f"  {value}  " for value in values]
    parsed = _parse_parents(tokens)
    assert parsed.tolist() == values


@given(st.lists(st.integers(min_value=0, max_value=1_000_000), min_size=1, max_size=100))
def test_parent_int_parser_can_treat_zero_as_missing(values: list[int]) -> None:
    """PLINK-style zero-as-missing maps only literal zero to ``-1``."""
    tokens = [str(value) for value in values]
    parsed = _parse_parents(tokens, zero_as_missing=True)
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
    series = pl.Series(tokens, dtype=pl.String)
    parsed = _as_birth_year_col(series, "birth_year")
    expected = np.array([float(t) for t in tokens], dtype=np.float64).astype(np.int32)
    assert parsed.dtype == np.int32
    assert np.array_equal(parsed, expected)


_ID_TOKENS = st.one_of(
    st.integers(min_value=-3, max_value=30).map(str),
    st.sampled_from(["001", "01", "1.0", "2.9", "+5", "-0", "A1", "a1", "x", "1e3"]),
    st.text(alphabet="abc019_-.", min_size=1, max_size=4).map(str.strip).filter(bool),
).filter(lambda t: t.upper() not in _PARENT_MISSING_TOKENS)  # a missing token is not an id


@given(
    st.lists(_ID_TOKENS, min_size=1, max_size=40),
    st.lists(st.one_of(_ID_TOKENS, st.just("NA"), st.just("-1")), min_size=1, max_size=40),
)
def test_ids_print_back_as_written(id_tokens: list[str], parent_tokens: list[str]) -> None:
    """Every ID prints back as its token, and two tokens share a code only if they are the same string."""
    n = len(id_tokens)
    parents = (parent_tokens * n)[:n]
    ids, mothers, fathers, labels = parse_ids(
        pl.Series(id_tokens), pl.Series(parents), pl.Series(list(reversed(parents)))
    )
    assert labels.strings(ids).to_list() == id_tokens
    expected_parents = ["-1" if t.upper() in _PARENT_MISSING_TOKENS or t == "-1" else t for t in parents]
    assert labels.strings(mothers).to_list() == expected_parents
    assert labels.strings(fathers).to_list() == list(reversed(expected_parents))
    tokens = id_tokens + expected_parents
    codes = np.concatenate([ids, mothers])
    for t, c in zip(tokens, codes, strict=True):
        assert (codes == c).sum() == tokens.count(t) or t == "-1"


@given(st.lists(st.integers(min_value=-5, max_value=10**12), min_size=1, max_size=50))
def test_plain_integer_ids_are_their_own_codes(values: list[int]) -> None:
    """When every token is a plain integer, the codes are the integers (integer mode)."""
    tokens = pl.Series([str(v) for v in values])
    ids, _, _, labels = parse_ids(tokens, pl.Series(["-1"] * len(values)), pl.Series(["-1"] * len(values)))
    assert not labels.is_string
    assert ids.tolist() == values


@given(st.sampled_from(_INVALID_NUMERIC_TOKENS))
def test_non_integer_parent_token_is_a_string_id(token: str) -> None:
    """A non-integer parent token is an ID, read in string mode, not a parse error."""
    ids, mothers, _, labels = parse_ids(pl.Series(["1"]), pl.Series([token]), pl.Series(["-1"]))
    assert labels.is_string
    assert labels.label(mothers[0]) == token
    assert labels.label(ids[0]) == "1"


def test_missing_id_is_an_error() -> None:
    """A row without an id cannot be parsed."""
    with pytest.raises(PedigreeError, match="missing value"):
        parse_ids(pl.Series(["1", None]), pl.Series(["-1", "-1"]), pl.Series(["-1", "-1"]))


@given(st.sampled_from(_INVALID_NUMERIC_TOKENS))
def test_birth_year_parser_rejects_non_numeric_tokens(token: str) -> None:
    """Birth-year parsing rejects non-numeric non-missing tokens."""
    with pytest.raises(PedigreeError):
        _as_birth_year_col(pl.Series([token], dtype=pl.String), "birth_year")


def _python_plain(token: str) -> bool:
    try:
        value = int(token)
    except ValueError:
        return False
    return str(value) == token and -(2**63) <= value < 2**63


@given(
    st.lists(
        st.one_of(
            st.integers(min_value=-(2**64), max_value=2**64).map(str),
            st.text(alphabet="+-0123456789.e", min_size=1, max_size=22),
        ),
        min_size=1,
        max_size=50,
    )
)
def test_plain_int_check_matches_python(tokens: list[str]) -> None:
    """A token is a plain integer exactly when Python's int prints it back unchanged and it fits int64."""
    assert _plain_ints(pl.Series(tokens, dtype=pl.String))[1].tolist() == [_python_plain(t) for t in tokens]
