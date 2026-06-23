"""Input parsing: delimiter sniffing, column coercion, sex decoding."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, PedigreeError, logger

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TextIO

_PARENT_MISSING_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "NA",
        "NAN",
        "N/A",
        ".",
        "?",
        "NONE",
        "NULL",
    }
)

_SEX_MISSING_TOKENS: frozenset[str] = _PARENT_MISSING_TOKENS | frozenset(
    {
        "-1",
        "U",
        "UNKNOWN",
    }
)


def _detect_sex_encoding(
    upper_non_missing: pd.Series,
    zero_as_missing: bool,
) -> tuple[str, str]:
    """Resolve the sex-column encoding from the observed tokens.

    Returns ``(encoding, ambiguity_class)`` where encoding is ``"default"``
    (0=F, 1=M) or ``"plink"`` (1=M, 2=F), and ambiguity_class is one of
    ``"confident"``, ``"word_only"``, or ``"ones_only"``.
    """
    numeric = {t for t in upper_non_missing.unique() if t.isdigit() or t.lstrip("-").isdigit()}
    if "2" in numeric:
        return "plink", "confident"
    if "0" in numeric and not zero_as_missing:
        return "default", "confident"
    if "0" in numeric and zero_as_missing:
        return "plink", "confident"
    if not numeric:
        return "default", "word_only"
    return "default", "ones_only"


def _format_id_sample(ids: np.ndarray, k: int = 5) -> str:
    """Deterministic random sample of ``k`` IDs as a comma-separated string.

    Logged so collaborators can eyeball whether the id column was parsed
    correctly (right column, not coerced to junk). Seed is fixed so reruns
    show the same sample.
    """
    n = len(ids)
    if n == 0:
        return ""
    sample_size = min(k, n)
    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(n, size=sample_size, replace=False))
    return ", ".join(str(int(x)) for x in ids[indices])


def _decode_sex(
    series: pd.Series,
    *,
    encoding: str = "auto",
    zero_as_missing: bool = False,
) -> np.ndarray:
    """Parse a sex column to int8 with ``SEX_UNKNOWN`` (-1) for missing.

    Accepts M/F (any case), Male/Female, and numeric tokens whose meaning
    depends on the resolved encoding:

    - ``encoding="default"`` (pedsum default): ``0=female, 1=male``.
    - ``encoding="plink"`` (PLINK fam convention): ``1=male, 2=female``,
      with ``0`` always treated as missing (PLINK fam spec).
    - ``encoding="auto"``: detect from the observed tokens (presence of
      ``"2"`` → plink; presence of ``"0"`` → default unless
      ``zero_as_missing=True``, which flips to plink).

    Missing tokens (``""``, ``NA``, ``NaN``, ``N/A``, ``.``, ``?``, ``None``,
    ``null``, ``-1``, ``U``, ``Unknown``, case-insensitive) decode to
    ``SEX_UNKNOWN``. Unrecognized non-missing tokens raise ``PedigreeError``.

    Returns an ``int8`` array; rows whose token was missing carry
    ``SEX_UNKNOWN`` (-1) and must be resolved by the caller.
    """
    # Cast through ``fillna("")`` so NaN cells (pandas' default na_value for
    # StringDtype) collapse into the missing-token set rather than leaking
    # into the unique-token scan.
    str_vals = series.fillna("").astype(str).str.strip()
    upper = str_vals.str.upper()
    missing_mask = upper.isin(_SEX_MISSING_TOKENS)
    non_missing = upper[~missing_mask]

    if encoding == "auto":
        resolved, ambiguity = _detect_sex_encoding(non_missing, zero_as_missing)
        if ambiguity == "ones_only":
            logger.warning(
                "sex auto-detect: only '1' tokens present; defaulting to "
                "0=female, 1=male — pass --sex-encoding=plink if your file "
                "uses 1=male, 2=female",
            )
    elif encoding in ("default", "plink"):
        resolved = encoding
    else:
        raise PedigreeError(
            f"unknown sex encoding {encoding!r}; expected 'auto', 'default', or 'plink'",
        )

    # Under PLINK, "0" is always missing (fam-file spec); fold it into the
    # missing mask before decoding numeric tokens.
    if resolved == "plink":
        missing_mask = missing_mask | (str_vals == "0")

    out = np.full(len(str_vals), SEX_UNKNOWN, dtype=np.int8)
    female_words = (upper == "F") | (upper == "FEMALE")
    male_words = (upper == "M") | (upper == "MALE")
    if resolved == "plink":
        out[female_words | (str_vals == "2")] = SEX_FEMALE
        out[male_words | (str_vals == "1")] = SEX_MALE
        allowed = "M/F (any case), Male/Female, or 1/2 (1=male, 2=female; PLINK convention)."
    else:
        out[female_words | (str_vals == "0")] = SEX_FEMALE
        out[male_words | (str_vals == "1")] = SEX_MALE
        allowed = (
            "M/F (any case), Male/Female, or 0/1 (0=female, 1=male). "
            "Pass --sex-encoding=plink if your file uses 1=male, 2=female."
        )

    bad = (out == SEX_UNKNOWN) & ~missing_mask
    if bad.any():
        bad_rows = np.where(bad)[0][:5]
        bad_vals = str_vals.iloc[bad_rows].tolist()
        raise PedigreeError(
            f"sex column has {int(bad.sum())} invalid value(s); "
            f"first offending rows {bad_rows.tolist()} -> {bad_vals}. "
            f"Allowed: {allowed}",
        )
    # Surface the literal tokens that mapped to each sex (case preserved) so
    # collaborators can verify sex was handled correctly without re-reading
    # the file. ADR-0001 collaborator-facing transparency.
    tokens_female = sorted(set(str_vals[out == SEX_FEMALE].tolist()))
    tokens_male = sorted(set(str_vals[out == SEX_MALE].tolist()))
    logger.info(
        "sex parsed: encoding=%s, female={%s} (n=%d), male={%s} (n=%d), unknown=%d",
        resolved,
        ", ".join(tokens_female),
        int((out == SEX_FEMALE).sum()),
        ", ".join(tokens_male),
        int((out == SEX_MALE).sum()),
        int((out == SEX_UNKNOWN).sum()),
    )
    return out


def _as_int_col(series: pd.Series, name: str) -> np.ndarray:
    try:
        return pd.to_numeric(series, errors="raise").astype(np.int64).to_numpy()
    except (ValueError, TypeError) as e:
        raise PedigreeError(f"column {name!r} must be integer-valued; failed to parse: {e}") from None


def _replace_missing_with(
    series: pd.Series,
    missing_tokens: frozenset[str],
    sentinel: str,
) -> pd.Series:
    """Normalize ``series`` to stripped strings with ``missing_tokens`` → ``sentinel``.

    Used by the parent-ID and birth-year parsers to fold every recognized
    missing token (NA, blank, NaN, etc.) into a single sentinel string before
    handing off to ``pd.to_numeric``.
    """
    filled = series.where(series.notna(), sentinel)
    str_vals = filled.astype(str).str.strip()
    missing_mask = str_vals.str.upper().isin(missing_tokens)
    return str_vals.where(~missing_mask, sentinel)


def _maybe_warn_csv(df: pd.DataFrame) -> None:
    """Raise a clear error when the file looks like CSV but was read as TSV.

    Detected by a single column whose name contains commas (TSV reader
    treats the entire comma-joined header as one column). Only fires
    when the user has pinned ``--sep tab`` (or any non-comma separator)
    and the file is actually comma-separated — the default ``--sep
    auto`` sniffs the right delimiter up front.
    """
    if len(df.columns) == 1 and "," in str(df.columns[0]):
        raise PedigreeError(
            f"input appears to be CSV (single column {df.columns[0]!r}); "
            "this script defaulted to a non-comma separator. Re-run "
            "with --sep auto (the default) or --sep comma."
        )


_SEP_CHOICES = ("auto", "tab", "comma", "semicolon", "pipe", "whitespace")

_SEP_MAP = {
    "tab": "\t",
    "comma": ",",
    "semicolon": ";",
    "pipe": "|",
    "whitespace": r"\s+",
}

_SEP_HUMAN = {v: k for k, v in _SEP_MAP.items()}


def _open_text_for_sniff(path: Path) -> TextIO:
    """Open ``path`` as text for delimiter sniffing; transparent to gzip."""
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _sniff_delimiter(path: Path) -> str:
    r"""Return the most likely column delimiter for ``path``.

    Counts ``\t`` / ``,`` / ``;`` / ``|`` on the first non-empty line.
    If none appear and the line splits into >=2 whitespace-separated
    tokens, returns ``r'\s+'`` (PLINK fam-style). Otherwise falls back
    to ``\t`` and lets downstream validation surface the column
    mismatch.
    """
    with _open_text_for_sniff(path) as fh:
        first = ""
        for line in fh:
            stripped = line.strip()
            if stripped:
                first = stripped
                break
    if not first:
        raise PedigreeError(f"input file {path} is empty or contains only blank lines")
    best = max(("\t", ",", ";", "|"), key=first.count)
    best_count = first.count(best)
    if best_count > 0:
        return best
    if len(first.split()) >= 2:
        return r"\s+"
    return "\t"


def _read_pedigree_table(
    path: Path,
    sep: str = "auto",
    *,
    dtype: object | None = None,
) -> pd.DataFrame:
    r"""Read a pedigree table, sniffing the delimiter when ``sep == 'auto'``.

    ``sep`` accepts the argparse keywords in ``_SEP_CHOICES`` or a
    literal delimiter (``"\t"``, ``","``, ...). When auto-sniff resolves
    to anything other than tab, an INFO log records the chosen
    delimiter so the routing is visible in the run log.
    """
    if not path.exists():
        raise PedigreeError(f"input file not found: {path}")
    if sep == "auto":
        chosen = _sniff_delimiter(path)
        if chosen != "\t":
            logger.info("input: sniffed %s-separated", _SEP_HUMAN[chosen])
    else:
        chosen = _SEP_MAP.get(sep, sep)
    engine = "python" if chosen == r"\s+" else None
    return pd.read_csv(path, sep=chosen, dtype=dtype, engine=engine)  # ty: ignore[no-matching-overload]


def _as_parent_int_col(
    series: pd.Series,
    name: str,
    zero_as_missing: bool = False,
) -> np.ndarray:
    """Parse a parent-ID column, with NA-like tokens (and optionally 0) → -1.

    Recognised missing tokens (case-insensitive): empty string, NA, NaN,
    N/A, ".", "?", None, null. With ``zero_as_missing=True``, the literal
    integer 0 is also remapped to -1 (PLINK fam convention).
    """
    cleaned = _replace_missing_with(series, _PARENT_MISSING_TOKENS, "-1")
    try:
        arr = pd.to_numeric(cleaned, errors="raise").astype(np.int64).to_numpy(copy=True)
    except (ValueError, TypeError) as e:
        raise PedigreeError(
            f"column {name!r} must be integer-valued (with -1, NA, blank, or empty for unknown); failed to parse: {e}"
        ) from None
    if zero_as_missing:
        arr[arr == 0] = -1
    return arr


def _as_birth_year_col(series: pd.Series, name: str) -> np.ndarray:
    """Parse a birth-year column to int32 with sentinel -1 for unknown.

    Accepts integer- or float-valued tokens (``"1988"``, ``"1988.0"``) and
    the same missing tokens as parent IDs (empty/NA/NaN/N/A/./?/None/null).
    Float values are truncated to int (a birth year is by definition a
    whole calendar year). Sentinel encoding matches
    ``pedigree_graph.PedigreeGraph.birth_year``.
    """
    cleaned = _replace_missing_with(series, _PARENT_MISSING_TOKENS, "-1")
    try:
        as_float = pd.to_numeric(cleaned, errors="raise").to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise PedigreeError(
            f"birth-year column {name!r} must be numeric (integer or float "
            f"calendar year, with -1/NA/blank for unknown); failed to parse: {e}"
        ) from None
    return as_float.astype(np.int32)


_BIRTH_YEAR_DEFAULT_MIN = 1800


def _birth_year_default_max() -> int:
    """Default upper bound for birth-year sanity (current calendar year + 1)."""
    return datetime.now(tz=UTC).year + 1
