"""ID tokens: parse the id and parent columns to int64 codes, and print codes back as the input's IDs.

pedsum treats an ID as an opaque string: two tokens are the same ID only if
they are the same string, so ``001`` and ``1`` are different IDs. Every check
runs on int64 codes. When every token is a plain integer (it prints back
exactly as written), the code is the integer itself and the run is the integer
path pedsum has always had. Any other token puts the whole file in string
mode, where codes number the distinct tokens in sorted order and
:class:`IdLabels` prints them back. See issue #16.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import polars as pl

from pedsum.base import PedigreeError, logger
from pedsum.parse import _PARENT_MISSING_TOKENS

#: Prefix of the IDs that ``--fill-half-founders`` gives phantom parents in
#: string mode. validate refuses the fill when an input ID starts with it.
PHANTOM_PREFIX = "_pedsum_phantom_"


#: 10**1 .. 10**19: an unsigned value below 10**k has at most k digits.
_POW10 = np.array([10**k for k in range(1, 20)], dtype=np.uint64)


def _printed_len(values: np.ndarray) -> np.ndarray:
    """Length of each int64 value as Python prints it (sign included)."""
    u = values.view(np.uint64)
    magnitude = np.where(values < 0, ~u + np.uint64(1), u)  # two's complement, exact for INT64_MIN
    return np.searchsorted(_POW10, magnitude, side="right") + 1 + (values < 0)


def _plain_ints(tokens: pl.Series, lengths: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(values, plain)``: tokens cast to int64 (0 where the cast fails), and which are plain integers.

    A plain integer is written exactly as int64 prints it: no ``001``, ``+5``, ``-0`` or ``1.0``.

    The cast accepts ``[+-]?digits``, so a token that casts prints back as
    written exactly when its length is its value's printed length; comparing
    lengths skips casting the values back to strings.
    """
    ints = tokens.cast(pl.Int64, strict=False)
    cast_ok = ints.is_not_null().to_numpy()
    values = ints.fill_null(0).to_numpy(writable=True)
    if lengths is None:
        lengths = tokens.str.len_bytes().fill_null(0).to_numpy()
    plain = cast_ok & (lengths == _printed_len(values))
    return values, plain


@dataclass(frozen=True)
class IdLabels:
    """Prints int64 ID codes as the IDs the input wrote.

    Integer mode (``tokens is None``): a code is the ID. String mode: code ``c``
    is ``tokens[c]``, the distinct tokens in sorted order. Codes from
    ``phantom_from`` on are phantom parents, ``PHANTOM_PREFIX`` + 1, 2, ...
    Code ``-1`` (no parent) prints as ``-1`` in both modes.
    """

    tokens: pl.Series | None = None
    phantom_from: int | None = None

    @property
    def is_string(self) -> bool:
        """Whether the IDs are in string mode."""
        return self.tokens is not None

    @cached_property
    def _token_list(self) -> list[str]:
        assert self.tokens is not None
        return self.tokens.to_list()

    def label(self, code: int | np.integer) -> str:
        """One code as its ID."""
        c = int(code)
        if self.tokens is None or c == -1:
            return str(c)
        if self.phantom_from is not None and c >= self.phantom_from:
            return f"{PHANTOM_PREFIX}{c - self.phantom_from + 1}"
        return self._token_list[c]

    def strings(self, codes: np.ndarray) -> pl.Series:
        """Codes as ID strings, in both modes."""
        if self.tokens is None:
            return pl.Series(codes).cast(pl.String)
        n = len(self.tokens)
        known = (codes >= 0) & (codes < n)
        out = self.tokens.gather(np.where(known, codes, 0))
        missing = np.flatnonzero(codes == -1)
        if len(missing):
            out = out.scatter(missing, "-1")
        phantoms = np.flatnonzero(codes >= n)
        if len(phantoms):
            out = out.scatter(phantoms, [self.label(c) for c in codes[phantoms]])
        return out

    @cached_property
    def _token_ints(self) -> np.ndarray:
        assert self.tokens is not None
        values, plain = _plain_ints(self.tokens)
        values[~plain] = 0
        return values

    def int_values(self, codes: np.ndarray) -> np.ndarray:
        """Each code's ID as an integer where it is written as one, else 0; ``-1`` stays ``-1``.

        Feeds the negative-ID checks, which judge integer-form tokens the same in both modes.
        """
        if self.tokens is None:
            return codes
        known = (codes >= 0) & (codes < len(self.tokens))
        return np.where(known, self._token_ints[np.where(known, codes, 0)], np.where(codes == -1, -1, 0))

    def codes(self, ids: pl.Series) -> np.ndarray:
        """The codes of the IDs in ``ids``, a column of the input this labels was parsed from.

        Raises ``PedigreeError`` on an ID the input does not have.
        """
        if ids.dtype == pl.String:
            ids = ids.str.strip_chars()
        if self.tokens is None:
            return ids.cast(pl.Int64).to_numpy()
        ids = ids.cast(pl.String)
        codes = self.tokens.search_sorted(ids).cast(pl.Int64).to_numpy()
        found = self.tokens.gather(np.minimum(codes, len(self.tokens) - 1)) == ids
        if not found.all():
            raise PedigreeError(f"internal: ID(s) {ids.filter(~found).head(3).to_list()} are not in the parsed input")
        return codes

    def relabel(self, frame: pl.DataFrame, *columns: str) -> pl.DataFrame:
        """``frame`` with the code ``columns`` printed as IDs; unchanged in integer mode."""
        if self.tokens is None:
            return frame
        return frame.with_columns(
            self.strings(frame[c].cast(pl.Int64).to_numpy()).alias(c) for c in columns if c in frame.columns
        )

    def phantom_clashes(self) -> list[str]:
        """Input IDs that start with ``PHANTOM_PREFIX`` (at most three), which a phantom could take."""
        if self.tokens is None:
            return []
        return self.tokens.filter(self.tokens.str.starts_with(PHANTOM_PREFIX)).head(3).to_list()

    def with_phantoms(self, first_code: int) -> IdLabels:
        """These labels with phantom parents numbered from ``first_code``.

        Integer mode needs no change: the phantom codes are new integers. The
        caller refuses a string-mode file with :meth:`phantom_clashes`.
        """
        if self.tokens is None:
            return self
        logger.info("validate: phantom parents get IDs %s1, %s2, ...", PHANTOM_PREFIX, PHANTOM_PREFIX)
        return IdLabels(self.tokens, first_code)


@dataclass
class _Column:
    """One ID column, stripped once.

    ``missing`` marks missing tokens (null, ``NA``, ``.``, ... and ``0`` under
    ``zero_as_missing``). A scan with ints also has ``values``, each token as
    int64 (0 where it is not a plain integer), and ``strings``, the indexes of
    tokens that are neither: the string IDs that put a file in string mode.
    """

    tokens: pl.Series
    missing: np.ndarray
    values: np.ndarray | None = None
    strings: np.ndarray | None = None


#: Longest missing-parent token, in bytes; longer tokens skip the lookup.
_MISSING_MAX_LEN = max(len(t) for t in _PARENT_MISSING_TOKENS)


def _scan(series: pl.Series, *, ints: bool, zero_as_missing: bool = False) -> _Column:
    """Strip a column and find its missing tokens, plus its integers when ``ints``.

    Only short tokens can be missing tokens, and with ``ints`` only the short
    ones that are not plain integers, so few tokens are upper-cased and looked up.
    """
    tokens = series.cast(pl.String).str.strip_chars()
    lengths = tokens.str.len_bytes().fill_null(0).to_numpy()
    values = nonplain = None
    if ints:
        values, plain = _plain_ints(tokens, lengths)
        nonplain = np.flatnonzero(~plain)
        rows = nonplain[lengths[nonplain] <= _MISSING_MAX_LEN]
    else:
        rows = np.flatnonzero(lengths <= _MISSING_MAX_LEN)
    sub = tokens.gather(rows)
    sub_missing = (sub.is_null() | sub.str.to_uppercase().is_in(sorted(_PARENT_MISSING_TOKENS))).fill_null(False)
    missing = np.zeros(len(tokens), dtype=bool)
    missing[rows[sub_missing.to_numpy()]] = True
    if zero_as_missing:
        missing |= (tokens == "0").fill_null(False).to_numpy()
    if nonplain is None:
        return _Column(tokens, missing)
    return _Column(tokens, missing, values, nonplain[~missing[nonplain]])


def parse_ids(
    id_series: pl.Series,
    mother_series: pl.Series,
    father_series: pl.Series,
    *,
    zero_as_missing: bool = False,
    labels: IdLabels | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, IdLabels]:
    """Parse the id and parent columns to int64 codes; missing parents are ``-1``.

    Raises ``PedigreeError`` when an id is missing or is a missing-parent
    token (``.``, ``?``, ...), which no parent column could name. Parent
    columns cannot fail: a missing token is ``-1``, and any other token is an ID.

    ``labels`` from an earlier parse of a superset of these rows (the
    ``--drop-offending`` rounds) keeps its mode and, in string mode, its codes.
    """
    ids = _scan(id_series, ints=True)
    n_missing = int(ids.missing.sum())
    if n_missing:
        raise PedigreeError(
            f"column {id_series.name!r} has {n_missing} missing value(s) (blank, NA, '.', '?', ...); "
            "every row needs an id"
        )
    assert ids.values is not None
    assert ids.strings is not None
    fixed = labels.tokens if labels is not None else None
    # The parents' integers matter only while integer mode is still possible.
    ints = fixed is None and len(ids.strings) == 0
    mothers = _scan(mother_series, ints=ints, zero_as_missing=zero_as_missing)
    fathers = _scan(father_series, ints=ints, zero_as_missing=zero_as_missing)
    columns = (ids, mothers, fathers)

    m_values, f_values = mothers.values, fathers.values
    if (
        m_values is not None
        and f_values is not None
        and not any(len(c.strings) for c in columns if c.strings is not None)
    ):
        m_values[mothers.missing] = -1
        f_values[fathers.missing] = -1
        return ids.values, m_values, f_values, IdLabels()

    # String mode. A parent token "-1" is the no-parent sentinel, not an ID.
    for c in (mothers, fathers):
        c.missing |= (c.tokens == "-1").fill_null(False).to_numpy()
    if fixed is not None:
        tokens = fixed
    else:
        named = [c.tokens.filter(pl.Series(~c.missing)) for c in columns]
        tokens = pl.concat(named).unique().sort()
        odd = pl.concat([c.tokens.gather(c.strings[:5]) for c in columns if c.strings is not None])
        odd = odd.unique().sort().head(5)
        logger.warning(
            "IDs read as strings, since some tokens are not plain integers (e.g. %s); "
            "two tokens are the same ID only if they are the same string, so 001 and 1 differ",
            odd.to_list(),
        )

    def codes(c: _Column) -> np.ndarray:
        out = tokens.search_sorted(c.tokens.fill_null("")).cast(pl.Int64).to_numpy(writable=True)
        out[c.missing] = -1
        return out

    return codes(ids), codes(mothers), codes(fathers), IdLabels(tokens)
