"""Property-based round-trip tests for sex decoding in ``pedsum.parse``.

``_decode_sex`` maps free-form tokens to ``{SEX_FEMALE, SEX_MALE, SEX_UNKNOWN}``
under a chosen encoding. The round-trip strategy renders known sexes to valid
tokens (with mixed case), decodes them, and checks the originals come back. This
exercises the encoding branches and missing-token handling far past the example
suite.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN
from pedsum.parse import _SEX_MISSING_TOKENS, _decode_sex

_MISSING_TOKENS = sorted(_SEX_MISSING_TOKENS)
_MALE_TOKENS = ["1", "M", "m", "Male", "MALE", "male"]
_FEMALE_DEFAULT_TOKENS = ["0", "F", "f", "Female", "FEMALE", "female"]
_FEMALE_PLINK_TOKENS = ["2", "F", "f", "Female", "FEMALE", "female"]


@st.composite
def _sex_series(draw: st.DrawFn) -> tuple[str, list[str], list[int]]:
    """Render a list of known sexes to valid tokens for a randomly chosen encoding."""
    encoding = draw(st.sampled_from(["default", "plink"]))
    female_tokens = _FEMALE_DEFAULT_TOKENS if encoding == "default" else _FEMALE_PLINK_TOKENS
    n = draw(st.integers(min_value=0, max_value=30))
    tokens: list[str] = []
    expected: list[int] = []
    for _ in range(n):
        sex = draw(st.sampled_from([SEX_FEMALE, SEX_MALE, SEX_UNKNOWN]))
        if sex == SEX_UNKNOWN:
            tokens.append(draw(st.sampled_from(_MISSING_TOKENS)))
        elif sex == SEX_FEMALE:
            tokens.append(draw(st.sampled_from(female_tokens)))
        else:
            tokens.append(draw(st.sampled_from(_MALE_TOKENS)))
        expected.append(sex)
    return encoding, tokens, expected


@settings(deadline=None)
@given(case=_sex_series())
def test_decode_sex_round_trip(case: tuple) -> None:
    """Decoding valid tokens recovers the original sexes; output is int8 in {-1, 0, 1}."""
    encoding, tokens, expected = case
    out = _decode_sex(pl.Series(tokens, dtype=pl.String), encoding=encoding)
    assert out.dtype == np.int8
    assert set(np.unique(out)).issubset({-1, 0, 1})
    np.testing.assert_array_equal(out, np.array(expected, dtype=np.int8))


@settings(deadline=None)
@given(
    tokens=st.lists(st.sampled_from(_MISSING_TOKENS), max_size=20),
    encoding=st.sampled_from(["default", "plink", "auto"]),
)
def test_missing_tokens_decode_to_unknown(tokens: list[str], encoding: str) -> None:
    """Every missing token decodes to ``SEX_UNKNOWN`` under any encoding."""
    out = _decode_sex(pl.Series(tokens, dtype=pl.String), encoding=encoding)
    assert (out == SEX_UNKNOWN).all()


def test_decode_sex_case_insensitive() -> None:
    """Word tokens decode the same regardless of case."""
    out = _decode_sex(pl.Series(["M", "m", "male", "MALE"], dtype=pl.String), encoding="default")
    assert (out == SEX_MALE).all()
