"""Property-based round-trip tests for sex decoding in ``pedsum.parse``.

``_decode_sex`` maps free-form tokens to ``{SEX_FEMALE, SEX_MALE, SEX_UNKNOWN}``
(PLINK's 1=male, 2=female, 0=unknown, plus words). The round-trip strategy
renders known sexes to valid tokens (with mixed case), decodes them, and checks
the originals come back. This exercises the missing-token handling far past the
example suite.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN
from pedsum.parse import _SEX_MISSING_TOKENS, _decode_sex, plink_sex

_MISSING_TOKENS = sorted(_SEX_MISSING_TOKENS | {"0"})
_MALE_TOKENS = ["1", "M", "m", "Male", "MALE", "male"]
_FEMALE_TOKENS = ["2", "F", "f", "Female", "FEMALE", "female"]


@st.composite
def _sex_series(draw: st.DrawFn) -> tuple[list[str], list[int]]:
    """Render a list of known sexes to valid tokens."""
    n = draw(st.integers(min_value=0, max_value=30))
    tokens: list[str] = []
    expected: list[int] = []
    for _ in range(n):
        sex = draw(st.sampled_from([SEX_FEMALE, SEX_MALE, SEX_UNKNOWN]))
        if sex == SEX_UNKNOWN:
            tokens.append(draw(st.sampled_from(_MISSING_TOKENS)))
        elif sex == SEX_FEMALE:
            tokens.append(draw(st.sampled_from(_FEMALE_TOKENS)))
        else:
            tokens.append(draw(st.sampled_from(_MALE_TOKENS)))
        expected.append(sex)
    return tokens, expected


@settings(deadline=None)
@given(case=_sex_series())
def test_decode_sex_round_trip(case: tuple) -> None:
    """Decoding valid tokens recovers the original sexes; output is int8 in {-1, 0, 1}."""
    tokens, expected = case
    out = _decode_sex(pl.Series(tokens, dtype=pl.String))
    assert out.dtype == np.int8
    assert set(np.unique(out)).issubset({-1, 0, 1})
    np.testing.assert_array_equal(out, np.array(expected, dtype=np.int8))


@settings(deadline=None)
@given(tokens=st.lists(st.sampled_from(_MISSING_TOKENS), max_size=20))
def test_missing_tokens_decode_to_unknown(tokens: list[str]) -> None:
    """Every missing token, PLINK's 0 among them, decodes to ``SEX_UNKNOWN``."""
    out = _decode_sex(pl.Series(tokens, dtype=pl.String))
    assert (out == SEX_UNKNOWN).all()


@settings(deadline=None)
@given(sex=st.lists(st.sampled_from([SEX_FEMALE, SEX_MALE, SEX_UNKNOWN]), max_size=30))
def test_plink_sex_round_trips_through_decode(sex: list[int]) -> None:
    """What pedsum writes decodes back to the same internal codes."""
    arr = np.array(sex, dtype=np.int8)
    written = plink_sex(arr).astype(str)
    np.testing.assert_array_equal(_decode_sex(pl.Series(written, dtype=pl.String)), arr)


def test_decode_sex_case_insensitive() -> None:
    """Word tokens decode the same regardless of case."""
    out = _decode_sex(pl.Series(["M", "m", "male", "MALE"], dtype=pl.String))
    assert (out == SEX_MALE).all()
