"""Input parsing: delimiter sniffing, column coercion, sex decoding."""

from __future__ import annotations

import gzip
import io
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, PedigreeError, logger
from pedsum.pedigree_ops import IdIndex

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
    upper_non_missing: list[str],
    zero_as_missing: bool,
) -> tuple[str, str]:
    """Resolve the sex-column encoding from the observed tokens.

    Returns ``(encoding, ambiguity_class)`` where encoding is ``"default"``
    (0=F, 1=M) or ``"plink"`` (1=M, 2=F), and ambiguity_class is one of
    ``"confident"``, ``"word_only"``, or ``"ones_only"``.
    """
    numeric = {t for t in upper_non_missing if t.isdigit() or t.lstrip("-").isdigit()}
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
    series: pl.Series,
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
    # Null cells (polars' representation for empty fields) collapse into the
    # missing-token set rather than leaking into the unique-token scan.
    str_vals = series.cast(pl.String).fill_null("").str.strip_chars()
    upper = str_vals.str.to_uppercase()
    missing_mask = upper.is_in(sorted(_SEX_MISSING_TOKENS))

    if encoding == "auto":
        resolved, ambiguity = _detect_sex_encoding(upper.filter(~missing_mask).unique().to_list(), zero_as_missing)
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
    female_token, male_token = ("2", "1") if resolved == "plink" else ("0", "1")
    if resolved == "plink":
        missing_mask = missing_mask | (str_vals == "0")
        allowed = "M/F (any case), Male/Female, or 1/2 (1=male, 2=female; PLINK convention)."
    else:
        allowed = (
            "M/F (any case), Male/Female, or 0/1 (0=female, 1=male). "
            "Pass --sex-encoding=plink if your file uses 1=male, 2=female."
        )
    is_female = (upper.is_in(["F", "FEMALE"]) | (str_vals == female_token)).to_numpy()
    is_male = (upper.is_in(["M", "MALE"]) | (str_vals == male_token)).to_numpy()
    out = np.full(len(str_vals), SEX_UNKNOWN, dtype=np.int8)
    out[is_female] = SEX_FEMALE
    out[is_male] = SEX_MALE

    bad = (out == SEX_UNKNOWN) & ~missing_mask.to_numpy()
    if bad.any():
        bad_rows = np.where(bad)[0][:5]
        bad_vals = str_vals.gather(bad_rows).to_list()
        raise PedigreeError(
            f"sex column has {int(bad.sum())} invalid value(s); "
            f"first offending rows {bad_rows.tolist()} -> {bad_vals}. "
            f"Allowed: {allowed}",
        )
    # Surface the literal tokens that mapped to each sex (case preserved) so
    # collaborators can verify sex was handled correctly without re-reading
    # the file. ADR-0001 collaborator-facing transparency.
    tokens_female = sorted(str_vals.filter(out == SEX_FEMALE).unique().to_list())
    tokens_male = sorted(str_vals.filter(out == SEX_MALE).unique().to_list())
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


def _parse_int_tokens(cleaned: pl.Series) -> np.ndarray:
    """Parse stripped string tokens to int64, tolerating float-form integers.

    Integer-form tokens (``"7"``) parse directly; float-form tokens
    (``"7.0"``, ``"7.5"``) fall back through float and truncate toward zero
    (matching the historical ``pd.to_numeric(...).astype(int64)`` behavior).
    Raises ``ValueError`` naming a sample of unparseable tokens.
    """
    as_int = cleaned.cast(pl.Int64, strict=False)
    bad = as_int.is_null()
    if not bad.any():
        # writable=True: callers (zero-as-missing remap) mutate the result.
        return as_int.to_numpy(writable=True)
    as_float = cleaned.cast(pl.Float64, strict=False)
    still_bad = as_float.is_null() | as_float.is_nan()
    if still_bad.any():
        samples = cleaned.filter(still_bad).head(3).to_list()
        raise ValueError(f"unable to parse value(s) {samples} as numeric")
    out = as_int.to_numpy(writable=True)
    bad_np = bad.to_numpy()
    out[bad_np] = as_float.to_numpy()[bad_np].astype(np.int64)
    return out


def _as_int_col(series: pl.Series, name: str) -> np.ndarray:
    try:
        cleaned = series.cast(pl.String).str.strip_chars()
        if cleaned.is_null().any():
            raise ValueError("column contains missing values")
        return _parse_int_tokens(cleaned)
    except (ValueError, TypeError, pl.exceptions.PolarsError) as e:
        raise PedigreeError(f"column {name!r} must be integer-valued; failed to parse: {e}") from None


def _replace_missing_with(
    series: pl.Series,
    missing_tokens: frozenset[str],
    sentinel: str,
) -> pl.Series:
    """Normalize ``series`` to stripped strings with ``missing_tokens`` → ``sentinel``.

    Used by the parent-ID and birth-year parsers to fold every recognized
    missing token (NA, blank, null, etc.) into a single sentinel string
    before handing off to the numeric parser.
    """
    str_vals = series.cast(pl.String).fill_null(sentinel).str.strip_chars()
    missing_mask = str_vals.str.to_uppercase().is_in(list(missing_tokens))
    return pl.select(pl.when(missing_mask).then(pl.lit(sentinel)).otherwise(str_vals).alias(series.name)).to_series()


def _maybe_warn_csv(df: pl.DataFrame) -> None:
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

# The reader's missing-token set, applied on both the all-string and the
# dtype-inferring read. Kept byte-compatible with the historical pandas reader
# (``pandas._libs.parsers.STR_NA_VALUES``, which applied even under
# ``dtype=str``): every one of these renders as an empty field when the frame
# is written back out (e.g. ``validate.tsv.gz`` echoing the input pedigree).
# polars nulls only empty fields by default, so the set is passed explicitly.
_READER_NULL_TOKENS = (
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
)

#: ``null_values`` argument form (polars wants a list).
_NULL_TOKENS = list(_READER_NULL_TOKENS)


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


def _read_whitespace_table(path: Path, *, as_str: bool, columns: list[str] | None) -> pl.DataFrame:
    r"""Read a whitespace-delimited (PLINK fam-style) table.

    polars has no regex separator, so ``\s+`` inputs are split manually.
    Rows whose token count differs from the header raise ``PedigreeError``.
    """
    with _open_text_for_sniff(path) as fh:
        lines = [line.strip() for line in fh]
    rows = [line.split() for line in lines if line]
    if not rows:
        raise PedigreeError(f"input file {path} is empty or contains only blank lines")
    header, data = rows[0], rows[1:]
    n_cols = len(header)
    for i, row in enumerate(data):
        if len(row) != n_cols:
            raise PedigreeError(
                f"whitespace-separated input {path}: row {i + 1} has {len(row)} field(s), expected {n_cols}"
            )
    # Re-serialize as an in-memory TSV (tokens are whitespace-split, so they
    # cannot contain tabs) and reuse the delimiter-file reader for identical
    # string / inference semantics.
    buf = io.BytesIO("\n".join("\t".join(row) for row in rows).encode("utf-8"))
    if as_str:
        return pl.read_csv(buf, separator="\t", infer_schema=False, null_values=_NULL_TOKENS, columns=columns)
    return pl.read_csv(buf, separator="\t", null_values=_NULL_TOKENS, infer_schema_length=None, columns=columns)


def _read_pedigree_table(
    path: Path,
    sep: str = "auto",
    *,
    dtype: object | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    r"""Read a pedigree table, sniffing the delimiter when ``sep == 'auto'``.

    ``sep`` accepts the argparse keywords in ``_SEP_CHOICES`` or a
    literal delimiter (``"\t"``, ``","``, ...). When auto-sniff resolves
    to anything other than tab, an INFO log records the chosen
    delimiter so the routing is visible in the run log.

    ``dtype=str`` reads every column as string (no inference), the mode
    used by validation; the default infers dtypes (the annotated re-read).
    ``columns`` reads only those columns; an absent one raises polars'
    ``ColumnNotFoundError``.
    """
    if not path.exists():
        raise PedigreeError(f"input file not found: {path}")
    if sep == "auto":
        chosen = _sniff_delimiter(path)
        if chosen != "\t":
            logger.info("input: sniffed %s-separated", _SEP_HUMAN[chosen])
    else:
        chosen = _SEP_MAP.get(sep, sep)
    as_str = dtype is str
    if chosen == r"\s+":
        return _read_whitespace_table(path, as_str=as_str, columns=columns)
    if as_str:
        return pl.read_csv(path, separator=chosen, infer_schema=False, null_values=_NULL_TOKENS, columns=columns)
    return pl.read_csv(path, separator=chosen, null_values=_NULL_TOKENS, infer_schema_length=None, columns=columns)


def _as_parent_int_col(
    series: pl.Series,
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
        arr = _parse_int_tokens(cleaned)
    except (ValueError, TypeError, pl.exceptions.PolarsError) as e:
        raise PedigreeError(
            f"column {name!r} must be integer-valued (with -1, NA, blank, or empty for unknown); failed to parse: {e}"
        ) from None
    if zero_as_missing:
        arr[arr == 0] = -1
    return arr


def _as_birth_year_col(series: pl.Series, name: str) -> np.ndarray:
    """Parse a birth-year column to int32 with sentinel -1 for unknown.

    Accepts integer- or float-valued tokens (``"1988"``, ``"1988.0"``) and
    the same missing tokens as parent IDs (empty/NA/NaN/N/A/./?/None/null).
    Float values are truncated to int (a birth year is by definition a
    whole calendar year). Sentinel encoding matches
    ``pedigree_graph.PedigreeGraph.birth_year``.
    """
    cleaned = _replace_missing_with(series, _PARENT_MISSING_TOKENS, "-1")
    as_float = cleaned.cast(pl.Float64, strict=False)
    bad = as_float.is_null() | as_float.is_nan()
    if bad.any():
        samples = cleaned.filter(bad).head(3).to_list()
        raise PedigreeError(
            f"birth-year column {name!r} must be numeric (integer or float "
            f"calendar year, with -1/NA/blank for unknown); failed to parse: "
            f"unable to parse value(s) {samples} as numeric"
        )
    return as_float.to_numpy().astype(np.int32)


_BIRTH_YEAR_DEFAULT_MIN = 1800


def _birth_year_default_max() -> int:
    """Default upper bound for birth-year sanity (current calendar year + 1)."""
    return datetime.now(tz=UTC).year + 1


_REFERENCE_TRUE = frozenset({"1", "true", "t", "yes", "y"})
_REFERENCE_FALSE = frozenset({"0", "false", "f", "no", "n"})


def read_reference_mask(path: Path, sep: str, id_col: str, column: str, ids: np.ndarray) -> np.ndarray:
    """Boolean reference-subpopulation mask from ``column`` of the input, aligned to ``ids``.

    True tokens are ``1/true/t/yes/y`` and false tokens ``0/false/f/no/n``
    (any case); missing tokens are false. Rows are matched by id, because
    loading may reorder them, and ids absent from the file (founders that
    ``validate`` added) are false.
    """
    raw = _read_pedigree_table(path, sep=sep, dtype=str)
    if column not in raw.columns:
        raise PedigreeError(f"reference column {column!r} not in input; file has {list(raw.columns)}")
    tokens = raw[column].str.strip_chars().str.to_lowercase()
    missing = tokens.is_null() | tokens.is_in([t.lower() for t in _READER_NULL_TOKENS])
    is_true = tokens.is_in(sorted(_REFERENCE_TRUE)) & ~missing
    bad = ~(is_true | tokens.is_in(sorted(_REFERENCE_FALSE)) | missing)
    if bad.any():
        samples = raw[column].filter(bad).head(3).to_list()
        raise PedigreeError(f"reference column {column!r} must hold 1/0 or true/false; got {samples}")
    raw_ids = _as_int_col(raw[id_col], id_col)
    return np.isin(ids, raw_ids[is_true.to_numpy()])


def read_trait_columns(
    path: Path,
    sep: str,
    id_col: str,
    columns: list[str],
    ids: np.ndarray,
    missing: list[str] | tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Raw trait tokens from ``columns`` of the input, aligned to ``ids``, ``None`` where missing.

    Missing is a reader null token (exact), a parent missing token after strip
    in any case, or one of ``missing`` after strip. Rows are matched by id,
    because loading may reorder them, and ids absent from the file (founders
    that ``validate`` added) are missing.
    """
    try:
        raw = _read_pedigree_table(path, sep=sep, dtype=str, columns=list(dict.fromkeys([id_col, *columns])))
    except pl.exceptions.ColumnNotFoundError:
        header = _read_pedigree_table(path, sep=sep, dtype=str).columns
        absent = [column for column in columns if column not in header]
        if not absent:
            raise
        raise PedigreeError(f"trait column(s) {absent} not in input; file has {header}") from None
    raw_ids = _as_int_col(raw[id_col], id_col)
    extra = sorted({token.strip() for token in missing})
    upper_missing = sorted(_PARENT_MISSING_TOKENS)

    def masked(column: str) -> pl.Expr:
        tokens = pl.col(column).str.strip_chars()
        is_missing = tokens.is_null() | tokens.str.to_uppercase().is_in(upper_missing) | tokens.is_in(extra)
        return pl.when(is_missing).then(None).otherwise(tokens).alias(column)

    frame = raw.select(masked(column) for column in dict.fromkeys(columns))
    if not np.array_equal(raw_ids, ids):
        # A null gather index (an id absent from the file) yields a null token.
        rows = pl.Series(IdIndex(raw_ids).get_indexer(ids)).replace(-1, None)
        frame = frame.select(pl.all().gather(rows))
    return {column: frame[column].to_numpy() for column in columns}
