#!/usr/bin/env python3
"""Pedigree summary CLI.

Reads a TSV pedigree (id, sex, mother, father), validates it, and writes
machine-readable summaries covering size, structure, family-size
distribution, relationship-pair counts, and per-individual inbreeding.

Outputs of ``summarize`` (with ``--out BASENAME``):
    BASENAME.summary.yaml             combined pedigree+individual summary
    BASENAME.summary.pedigree.tsv     long-form pedigree-level summary
    BASENAME.summary.individual.tsv   long-form per-individual distribution
    BASENAME.annotated.tsv.gz         input pedigree + per-individual cols
                                      (suppressed under ``--safe-attempt``)

Outputs of ``validate``:
    BASENAME.validate.log             per-finding TSV
    BASENAME.validate.tsv.gz          fixed pedigree (omitted on hard block)

Single-file CLI on top of numpy, scipy, pandas, pyyaml, numba, and the
``pedigree_graph`` package (which provides the matrix-engine
relationship enumeration through degree 5). The BFS engine in
Section 3b lives only here.

Counting semantics: relationship pairs (FS / MHS / PHS / GP / Av / 1C
and all named codes through degree 5) are unique unordered pairs.
``PO`` is the synthetic sum ``MO + FO``. The per-individual
``n_descendants`` column is a path count and overcounts in inbred
pedigrees.

``--safe-attempt`` applies best-effort GDPR-style redaction (skips the
annotated TSV, drops min/max from distributions, nulls counts and
strata below cell-size 5). Not a safe-harbor guarantee.

Sex encoding: by default 0=female, 1=male; M/F (any case) and
Male/Female are also accepted. Pass ``--plink-sex`` for PLINK's
1=male, 2=female encoding. Missing parents: ``-1`` (default), or any
of ``NA``, ``NaN``, ``N/A``, ``.``, ``?``, blank, ``None``, ``null``
case-insensitively. Pass ``--zero-as-missing`` to treat ``0`` as
missing (PLINK fam convention).

Usage:
    python pedigree_summary.py summarize --in PED.tsv --out BASENAME [options]
    python pedigree_summary.py validate  --in PED.tsv --out BASENAME [options]
"""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import yaml
from pedigree_graph import REL_REGISTRY, PedigreeGraph
from pedigree_graph.experimental import count_pairs_bfs

_BFS_AUTO_THRESHOLD = 5_000_000

VERSION = "0.3"
SEX_FEMALE = 0
SEX_MALE = 1
INBRED_TOL = 1e-9

logger = logging.getLogger("pedigree_summary")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PedigreeError(Exception):
    """Raised on any input validation failure."""


@dataclass
class Finding:
    """A single per-individual validation finding."""

    check: str
    id: int | None = None
    row: int | None = None
    detail: str = ""


@dataclass
class CheckResult:
    """Per-check status report (PASS / FAIL / SKIP) with finding count."""

    name: str
    status: str
    count: int = 0
    skip_reason: str | None = None


_CHECK_ORDER: tuple[str, ...] = (
    "required_columns",
    "empty_pedigree",
    "id_dtype",
    "mother_dtype",
    "father_dtype",
    "sex_tokens",
    "negative_ids",
    "duplicate_ids",
    "parent_token_range_mother",
    "parent_token_range_father",
    "parent_refs_present_mother",
    "parent_refs_present_father",
    "parent_refs_sex_conflict",
    "self_loops",
    "parents_distinct",
    "sex_role_consistency",
    "topological_row_order",
    "acyclic",
)


def _id_list(ids, max_show: int = 5) -> str:
    ids = list(ids)
    if len(ids) <= max_show:
        return ", ".join(str(i) for i in ids)
    return ", ".join(str(i) for i in ids[:max_show]) + f", ... ({len(ids)} total)"


def _parent_rows(parents: np.ndarray, id_index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Map parent IDs to row indices; -1 for missing. Returns (row_index, present_mask)."""
    out = np.full(len(parents), -1, dtype=np.int64)
    mask = parents != -1
    if mask.any():
        out[mask] = id_index.get_indexer(parents[mask])
    return out, mask


def _full_sib_groups(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, np.ndarray]:
    """Per-row full-sib counts plus underlying mating-pair group sizes.

    Returns (fs_count, fs_groups, both_present) where fs_count[i] is the
    number of full sibs of row i (0 if either parent is unknown), fs_groups
    is the groupby-size series indexed by (mother, father), and both_present
    is the boolean row mask for rows with both parents known.
    """
    n = len(df)
    fs_count = np.zeros(n, dtype=np.int64)
    both_present = ((df["mother"] != -1) & (df["father"] != -1)).to_numpy()
    if not both_present.any():
        return fs_count, pd.Series(dtype=np.int64), both_present
    children = df.loc[both_present]
    fs_groups = children.groupby(["mother", "father"]).size()
    idx = pd.MultiIndex.from_arrays(
        [children["mother"].to_numpy(), children["father"].to_numpy()],
        names=["mother", "father"],
    )
    fs_count[np.where(both_present)[0]] = fs_groups.reindex(idx).to_numpy() - 1
    return fs_count, fs_groups, both_present


def _grandparent_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (mm, mf, fm, ff) arrays of grandparent IDs (-1 for unknown)."""
    parent_lookup = df.set_index("id")[["mother", "father"]]
    return tuple(
        df[outer].map(parent_lookup[inner]).fillna(-1).astype(np.int64).to_numpy()
        for outer, inner in (
            ("mother", "mother"),
            ("mother", "father"),
            ("father", "mother"),
            ("father", "father"),
        )
    )


def _build_children_csr(
    m_row: np.ndarray,
    mask_m: np.ndarray,
    f_row: np.ndarray,
    mask_f: np.ndarray,
    n: int,
) -> sp.csr_matrix | None:
    """Build parent→child CSR from parent-row arrays. None if no edges."""
    parent_rows = np.concatenate([m_row[mask_m], f_row[mask_f]])
    if len(parent_rows) == 0:
        return None
    child_rows = np.concatenate([np.where(mask_m)[0], np.where(mask_f)[0]])
    return sp.csr_matrix(
        (np.ones(len(parent_rows), dtype=np.int8), (parent_rows, child_rows)),
        shape=(n, n),
    )


# ---------------------------------------------------------------------------
# Per-check Issue helpers (used by both load_and_validate and validate_pedigree)
# ---------------------------------------------------------------------------


def _check_negative_ids(ids: np.ndarray) -> list[Finding]:
    """Detect negative IDs (id < 0). Returns one Finding per offending row."""
    return [
        Finding(check="negative_ids", id=int(ids[i]), row=int(i),
                detail=f"id={int(ids[i])} is negative; must be >= 0")
        for i in np.where(ids < 0)[0]
    ]


def _check_duplicate_ids(ids: np.ndarray) -> list[Finding]:
    """Detect duplicate IDs. Returns one Finding per duplicated ID value."""
    sort_order = np.argsort(ids, kind="stable")
    sorted_ids = ids[sort_order]
    dup_mask = sorted_ids[1:] == sorted_ids[:-1]
    if not dup_mask.any():
        return []
    dup_ids = np.unique(sorted_ids[1:][dup_mask])
    return [
        Finding(check="duplicate_ids", id=int(d), detail=f"id={int(d)} appears more than once")
        for d in dup_ids
    ]


def _check_parent_token_range(arr: np.ndarray, role: str) -> list[Finding]:
    """Detect parent column values < -1; one Finding per offending row."""
    return [
        Finding(check=f"parent_token_range_{role}", row=int(i),
                detail=f"row {int(i)}: {role} value {int(arr[i])} < -1; missing parent must be -1")
        for i in np.where(arr < -1)[0]
    ]


def _check_parent_refs_present(arr: np.ndarray, role: str, id_index: pd.Index) -> list[Finding]:
    """Detect parent IDs that don't have their own row; one Finding per missing ID."""
    present = arr != -1
    if not present.any():
        return []
    unique_parents = np.unique(arr[present])
    mapped = id_index.get_indexer(unique_parents)
    missing = unique_parents[mapped == -1]
    if len(missing) == 0:
        return []
    findings = []
    zero_in_id = id_index.get_indexer([0])[0] != -1
    for mid in missing:
        n_refs = int((arr == mid).sum())
        detail = f"id={int(mid)} referenced as {role} {n_refs} time(s) but no row has this id"
        if int(mid) == 0 and not zero_in_id:
            detail += " (if file uses 0/NA/blank as missing token, convert to -1)"
        findings.append(Finding(check=f"parent_refs_present_{role}", id=int(mid), detail=detail))
    return findings


def _check_empty_pedigree(n: int) -> list[Finding]:
    """Detect empty pedigrees (n == 0). Returns a single Finding when empty."""
    if n > 0:
        return []
    return [Finding(check="empty_pedigree", detail="pedigree has 0 rows")]


def _check_parents_distinct(
    ids: np.ndarray, mothers: np.ndarray, fathers: np.ndarray,
) -> list[Finding]:
    """Detect rows where mother == father (and both != -1); one Finding per row."""
    bad = (mothers != -1) & (fathers != -1) & (mothers == fathers)
    return [
        Finding(check="parents_distinct", id=int(ids[i]), row=int(i),
                detail=f"row {int(i)}: id={int(ids[i])} has mother == father == {int(mothers[i])}")
        for i in np.where(bad)[0]
    ]


def _check_self_loops(ids: np.ndarray, mothers: np.ndarray, fathers: np.ndarray) -> list[Finding]:
    """Detect rows where id == mother or id == father; one Finding per offending row."""
    findings = [
        Finding(check="self_loops", id=int(ids[i]), row=int(i),
                detail=f"row {int(i)}: id={int(ids[i])} listed as own mother")
        for i in np.where(mothers == ids)[0]
    ]
    findings.extend(
        Finding(check="self_loops", id=int(ids[i]), row=int(i),
                detail=f"row {int(i)}: id={int(ids[i])} listed as own father")
        for i in np.where(fathers == ids)[0]
    )
    return findings


def _check_parent_refs_sex_conflict(
    mothers: np.ndarray,
    fathers: np.ndarray,
    id_index: pd.Index,
) -> list[Finding]:
    """Detect missing parent IDs referenced as both mother AND father; one Finding per ID."""
    moms = np.unique(mothers[mothers != -1])
    dads = np.unique(fathers[fathers != -1])
    moms_missing = moms[id_index.get_indexer(moms) == -1]
    dads_missing = dads[id_index.get_indexer(dads) == -1]
    conflicts = np.intersect1d(moms_missing, dads_missing)
    return [
        Finding(check="parent_refs_sex_conflict", id=int(c),
                detail=f"id={int(c)} referenced as both mother and father (sex ambiguous)")
        for c in conflicts
    ]


def _check_topological_row_order(
    ids: np.ndarray,
    mothers: np.ndarray,
    fathers: np.ndarray,
    id_index: pd.Index,
) -> list[Finding]:
    """Detect rows whose parent appears in a later row; one Finding per offending row."""
    n = len(ids)
    m_row, _ = _parent_rows(mothers, id_index)
    f_row, _ = _parent_rows(fathers, id_index)
    own = np.arange(n, dtype=np.int64)
    bad_m = (m_row != -1) & (m_row >= own)
    bad_f = (f_row != -1) & (f_row >= own)
    findings = []
    for i in np.where(bad_m | bad_f)[0]:
        roles = []
        if bad_m[i]:
            roles.append(f"mother (row {int(m_row[i])})")
        if bad_f[i]:
            roles.append(f"father (row {int(f_row[i])})")
        findings.append(Finding(
            check="topological_row_order", id=int(ids[i]), row=int(i),
            detail=f"row {int(i)} (id={int(ids[i])}) appears before its " + " and ".join(roles),
        ))
    return findings


def _check_sex_role_consistency(
    mothers: np.ndarray,
    fathers: np.ndarray,
    sex: np.ndarray,
    id_index: pd.Index,
) -> list[Finding]:
    """Detect IDs used as mother but sex != female, or as father but sex != male."""
    used_as_mother = np.unique(mothers[mothers != -1])
    used_as_father = np.unique(fathers[fathers != -1])
    rows_um = id_index.get_indexer(used_as_mother)
    rows_uf = id_index.get_indexer(used_as_father)
    findings = [
        Finding(check="sex_role_consistency", id=int(mid),
                detail=f"id={int(mid)} used as mother but sex != female")
        for mid in used_as_mother[(rows_um != -1) & (sex[rows_um] != SEX_FEMALE)]
    ]
    findings.extend(
        Finding(check="sex_role_consistency", id=int(fid),
                detail=f"id={int(fid)} used as father but sex != male")
        for fid in used_as_father[(rows_uf != -1) & (sex[rows_uf] != SEX_MALE)]
    )
    return findings


def _summarize_findings(findings: list[Finding]) -> str:
    """Build a human-readable error message from a list of findings."""
    if not findings:
        return ""
    check = findings[0].check
    n = len(findings)
    samples = []
    for f in findings[:5]:
        if f.id is not None and f.row is not None:
            samples.append(f"row {f.row} (id={f.id})")
        elif f.id is not None:
            samples.append(f"id={f.id}")
        elif f.row is not None:
            samples.append(f"row {f.row}")
    sample_str = ", ".join(samples) if samples else findings[0].detail
    extra = f" (and {n - 5} more)" if n > 5 else ""
    return f"{check}: {n} finding(s) — {sample_str}{extra}"


def _check_acyclic(ids: np.ndarray, mothers: np.ndarray, fathers: np.ndarray,
                   id_index: pd.Index) -> list[Finding]:
    """Detect IDs in a cycle via Kahn's; one Finding per node that couldn't be resolved."""
    n = len(ids)
    m_row, mask_m = _parent_rows(mothers, id_index)
    f_row, mask_f = _parent_rows(fathers, id_index)
    children = _build_children_csr(m_row, mask_m, f_row, mask_f, n)
    indeg = mask_m.astype(np.int32) + mask_f.astype(np.int32)
    frontier = np.where(indeg == 0)[0]
    while len(frontier) > 0 and children is not None:
        sub = children[frontier]
        kids = sub.indices
        if len(kids) == 0:
            break
        np.subtract.at(indeg, kids, 1)
        unique_kids = np.unique(kids)
        frontier = unique_kids[indeg[unique_kids] == 0]
    return [
        Finding(check="acyclic", id=int(ids[i]), row=int(i),
                detail=f"id={int(ids[i])} could not be topologically ordered (in a cycle)")
        for i in np.where(indeg > 0)[0]
    ]


# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------


_PARENT_MISSING_TOKENS: frozenset[str] = frozenset({
    "", "NA", "NAN", "N/A", ".", "?", "NONE", "NULL",
})


def _decode_sex(series: pd.Series, plink: bool = False) -> np.ndarray:
    """Parse a sex column. Accepts M/F (any case) and Male/Female by default.

    Numeric encoding: 0=female, 1=male (matches the script's internal
    convention). When ``plink=True``, numeric encoding is 1=male, 2=female
    (PLINK convention).
    """
    str_vals = series.astype(str).str.strip()
    upper = str_vals.str.upper()
    out = np.full(len(str_vals), -1, dtype=np.int8)
    female_words = (upper == "F") | (upper == "FEMALE")
    male_words = (upper == "M") | (upper == "MALE")
    if plink:
        out[female_words | (str_vals == "2")] = SEX_FEMALE
        out[male_words | (str_vals == "1")] = SEX_MALE
        allowed = "M/F (any case), Male/Female, or 1/2 (1=male, 2=female; PLINK convention)."
    else:
        out[female_words | (str_vals == "0")] = SEX_FEMALE
        out[male_words | (str_vals == "1")] = SEX_MALE
        allowed = (
            "M/F (any case), Male/Female, or 0/1 (0=female, 1=male). "
            "Pass --plink-sex if your file uses 1=male, 2=female."
        )
    bad = out == -1
    if bad.any():
        bad_rows = np.where(bad)[0][:5]
        bad_vals = str_vals.iloc[bad_rows].tolist()
        raise PedigreeError(
            f"sex column has {int(bad.sum())} invalid value(s); "
            f"first offending rows {bad_rows.tolist()} -> {bad_vals}. "
            f"Allowed: {allowed}"
        )
    return out


def _as_int_col(series: pd.Series, name: str) -> np.ndarray:
    try:
        return pd.to_numeric(series, errors="raise").astype(np.int64).to_numpy()
    except (ValueError, TypeError) as e:
        raise PedigreeError(f"column {name!r} must be integer-valued; failed to parse: {e}") from None


def _maybe_warn_csv(df: pd.DataFrame) -> None:
    """Raise a clear error when the file looks like CSV but was read as TSV.

    Detected by a single column whose name contains commas (TSV reader
    treats the entire comma-joined header as one column).
    """
    if len(df.columns) == 1 and "," in str(df.columns[0]):
        raise PedigreeError(
            f"input appears to be CSV (single column {df.columns[0]!r}); "
            "this script reads TSV (tab-separated). Convert with: "
            "tr ',' '\\t' < input.csv > input.tsv"
        )


def _as_parent_int_col(
    series: pd.Series, name: str, zero_as_missing: bool = False,
) -> np.ndarray:
    """Parse a parent-ID column, with NA-like tokens (and optionally 0) → -1.

    Recognised missing tokens (case-insensitive): empty string, NA, NaN,
    N/A, ".", "?", None, null. With ``zero_as_missing=True``, the literal
    integer 0 is also remapped to -1 (PLINK fam convention).
    """
    filled = series.where(series.notna(), "-1")
    str_vals = filled.astype(str).str.strip()
    missing_mask = str_vals.str.upper().isin(_PARENT_MISSING_TOKENS)
    cleaned = str_vals.where(~missing_mask, "-1")
    try:
        arr = pd.to_numeric(cleaned, errors="raise").astype(np.int64).to_numpy(copy=True)
    except (ValueError, TypeError) as e:
        raise PedigreeError(
            f"column {name!r} must be integer-valued (with -1, NA, blank, "
            f"or empty for unknown); failed to parse: {e}"
        ) from None
    if zero_as_missing:
        arr[arr == 0] = -1
    return arr


def load_and_validate(
    path: Path,
    id_col: str = "id",
    sex_col: str = "sex",
    mother_col: str = "mother",
    father_col: str = "father",
    plink_sex: bool = False,
    zero_as_missing: bool = False,
) -> tuple[pd.DataFrame, sp.csr_matrix | None]:
    """Load TSV, run all QC, return (df, children_csr).

    df has columns id, sex (int8), mother, father, generation (int32).
    Missing parent encoded as -1. children_csr is the parent→child sparse
    matrix (None if there are no parent edges) and is shared across the
    generations / components / descendants / inbreeding passes.
    """
    if not path.exists():
        raise PedigreeError(f"input file not found: {path}")

    t0 = time.perf_counter()
    df = pd.read_csv(path, sep="\t", dtype=str)
    logger.info("read %d rows from %s in %.2fs", len(df), path, time.perf_counter() - t0)
    _maybe_warn_csv(df)

    needed = {id_col, sex_col, mother_col, father_col}
    miss_cols = needed - set(df.columns)
    if miss_cols:
        raise PedigreeError(f"missing required columns: {sorted(miss_cols)}; file has {list(df.columns)}")

    def _raise_first(findings: list[Finding]) -> None:
        if findings:
            raise PedigreeError(_summarize_findings(findings))

    _raise_first(_check_empty_pedigree(len(df)))

    ids = _as_int_col(df[id_col], id_col)
    mothers = _as_parent_int_col(df[mother_col], mother_col, zero_as_missing)
    fathers = _as_parent_int_col(df[father_col], father_col, zero_as_missing)
    sex = _decode_sex(df[sex_col], plink=plink_sex)

    n = len(ids)

    _raise_first(_check_negative_ids(ids))
    _raise_first(_check_duplicate_ids(ids))
    _raise_first(_check_parent_token_range(mothers, "mother"))
    _raise_first(_check_parent_token_range(fathers, "father"))

    id_index = pd.Index(ids)
    for findings in (
        _check_parent_refs_present(mothers, "mother", id_index),
        _check_parent_refs_present(fathers, "father", id_index),
        _check_self_loops(ids, mothers, fathers),
        _check_parents_distinct(ids, mothers, fathers),
        _check_sex_role_consistency(mothers, fathers, sex, id_index),
    ):
        if findings:
            raise PedigreeError(_summarize_findings(findings))

    out = pd.DataFrame(
        {"id": ids, "sex": sex, "mother": mothers, "father": fathers},
    )
    m_row, mask_m = _parent_rows(mothers, id_index)
    f_row, mask_f = _parent_rows(fathers, id_index)
    children_csr = _build_children_csr(m_row, mask_m, f_row, mask_f, n)
    out["ped_depth"] = compute_generations(out, mask_m, mask_f, children_csr)

    logger.info("validated %d rows in %.2fs", n, time.perf_counter() - t0)
    return out, children_csr


# ---------------------------------------------------------------------------
# Topological generation (Kahn) — also detects cycles
# ---------------------------------------------------------------------------


def compute_generations(
    df: pd.DataFrame,
    mask_m: np.ndarray,
    mask_f: np.ndarray,
    children: sp.csr_matrix | None,
) -> np.ndarray:
    """Topological depth per row via Kahn's algorithm; raises on cycles."""
    n = len(df)
    indeg = mask_m.astype(np.int32) + mask_f.astype(np.int32)
    gen = np.zeros(n, dtype=np.int32)

    frontier = np.where(indeg == 0)[0]
    processed = len(frontier)
    while len(frontier) > 0 and children is not None:
        sub = children[frontier]
        kids = sub.indices
        if len(kids) == 0:
            break
        parents_per_edge = np.repeat(frontier, np.diff(sub.indptr))
        np.maximum.at(gen, kids, gen[parents_per_edge] + 1)
        np.subtract.at(indeg, kids, 1)
        unique_kids = np.unique(kids)
        frontier = unique_kids[indeg[unique_kids] == 0]
        processed += len(frontier)

    if processed != n:
        unresolved = np.where(indeg > 0)[0]
        sample_ids = df["id"].iloc[unresolved[:5]].tolist()
        raise PedigreeError(
            f"pedigree contains a cycle: {len(unresolved)} individual(s) could not "
            f"be topologically ordered (e.g. ids {sample_ids})"
        )

    return gen


# ---------------------------------------------------------------------------
# Validate (accumulating mode)
# ---------------------------------------------------------------------------


def validate_pedigree(
    path: Path,
    id_col: str = "id",
    sex_col: str = "sex",
    mother_col: str = "mother",
    father_col: str = "father",
    plink_sex: bool = False,
    zero_as_missing: bool = False,
) -> tuple[int, list[CheckResult], list[Finding], dict]:
    """Run every integrity check accumulating.

    Returns ``(n_rows, results, findings, ctx)`` where ``results`` covers all
    checks in ``_CHECK_ORDER`` (each PASS / FAIL / SKIP), ``findings`` lists
    every per-individual finding, and ``ctx`` carries the coerced arrays and
    raw DataFrame for downstream auto-fix.
    """
    if not path.exists():
        raise PedigreeError(f"input file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str)
    _maybe_warn_csv(df)
    n = len(df)
    findings: list[Finding] = []
    results: dict[str, CheckResult] = {
        name: CheckResult(name=name, status="SKIP") for name in _CHECK_ORDER
    }
    ctx: dict = {"ids": None, "mothers": None, "fathers": None, "sex": None, "df_raw": df}

    def _record(name: str, fs: list[Finding]) -> None:
        findings.extend(fs)
        results[name] = CheckResult(name=name, status="FAIL" if fs else "PASS", count=len(fs))

    def _skip(name: str, reason: str) -> None:
        results[name] = CheckResult(name=name, status="SKIP", skip_reason=reason)

    needed = {id_col, sex_col, mother_col, father_col}
    miss_cols = needed - set(df.columns)
    if miss_cols:
        msg = f"missing required columns: {sorted(miss_cols)}; file has {list(df.columns)}"
        findings.append(Finding(check="required_columns", detail=msg))
        results["required_columns"] = CheckResult(name="required_columns", status="FAIL", count=1)
        for name in _CHECK_ORDER:
            if name != "required_columns" and results[name].status == "SKIP":
                results[name].skip_reason = "required_columns failed"
        return n, list(results.values()), findings, ctx
    results["required_columns"] = CheckResult(name="required_columns", status="PASS")

    empty_findings = _check_empty_pedigree(n)
    findings.extend(empty_findings)
    results["empty_pedigree"] = CheckResult(
        name="empty_pedigree",
        status="FAIL" if empty_findings else "PASS",
        count=len(empty_findings),
    )

    def _coerce_int(col_name: str, label: str, parser=_as_int_col) -> np.ndarray | None:
        try:
            arr = parser(df[col_name], col_name)
            results[label] = CheckResult(name=label, status="PASS")
        except PedigreeError as e:
            findings.append(Finding(check=label, detail=str(e)))
            results[label] = CheckResult(name=label, status="FAIL", count=1)
            return None
        return arr

    def _parent_parser(s: pd.Series, name: str) -> np.ndarray:
        return _as_parent_int_col(s, name, zero_as_missing=zero_as_missing)

    ids = _coerce_int(id_col, "id_dtype")
    mothers = _coerce_int(mother_col, "mother_dtype", _parent_parser)
    fathers = _coerce_int(father_col, "father_dtype", _parent_parser)
    sex: np.ndarray | None = None
    try:
        sex = _decode_sex(df[sex_col], plink=plink_sex)
        results["sex_tokens"] = CheckResult(name="sex_tokens", status="PASS")
    except PedigreeError as e:
        findings.append(Finding(check="sex_tokens", detail=str(e)))
        results["sex_tokens"] = CheckResult(name="sex_tokens", status="FAIL", count=1)

    if ids is not None:
        _record("negative_ids", _check_negative_ids(ids))
        _record("duplicate_ids", _check_duplicate_ids(ids))
    else:
        _skip("negative_ids", "id_dtype failed")
        _skip("duplicate_ids", "id_dtype failed")

    if mothers is not None:
        _record("parent_token_range_mother", _check_parent_token_range(mothers, "mother"))
    else:
        _skip("parent_token_range_mother", "mother_dtype failed")
    if fathers is not None:
        _record("parent_token_range_father", _check_parent_token_range(fathers, "father"))
    else:
        _skip("parent_token_range_father", "father_dtype failed")

    blocking = {"id_dtype", "negative_ids", "duplicate_ids"}
    can_index = ids is not None and not any(
        results[name].status == "FAIL" for name in blocking
    )
    if can_index:
        id_index = pd.Index(ids)
        if mothers is not None:
            _record("parent_refs_present_mother",
                    _check_parent_refs_present(mothers, "mother", id_index))
        else:
            _skip("parent_refs_present_mother", "mother_dtype failed")
        if fathers is not None:
            _record("parent_refs_present_father",
                    _check_parent_refs_present(fathers, "father", id_index))
        else:
            _skip("parent_refs_present_father", "father_dtype failed")
        if mothers is not None and fathers is not None:
            _record("parent_refs_sex_conflict",
                    _check_parent_refs_sex_conflict(mothers, fathers, id_index))
            _record("self_loops", _check_self_loops(ids, mothers, fathers))
            _record("parents_distinct", _check_parents_distinct(ids, mothers, fathers))
            if sex is not None:
                _record("sex_role_consistency",
                        _check_sex_role_consistency(mothers, fathers, sex, id_index))
            else:
                _skip("sex_role_consistency", "sex_tokens failed")
            _record("topological_row_order",
                    _check_topological_row_order(ids, mothers, fathers, id_index))
            blocks_acyclic = []
            if results["parent_refs_present_mother"].status == "FAIL":
                blocks_acyclic.append("missing mother references")
            if results["parent_refs_present_father"].status == "FAIL":
                blocks_acyclic.append("missing father references")
            if results["parents_distinct"].status == "FAIL":
                blocks_acyclic.append("mother == father rows")
            if not blocks_acyclic:
                _record("acyclic", _check_acyclic(ids, mothers, fathers, id_index))
            else:
                _skip("acyclic", "; ".join(blocks_acyclic))
        else:
            _skip("parent_refs_sex_conflict", "mother_dtype or father_dtype failed")
            _skip("self_loops", "mother_dtype or father_dtype failed")
            _skip("parents_distinct", "mother_dtype or father_dtype failed")
            _skip("sex_role_consistency", "mother_dtype or father_dtype failed")
            _skip("topological_row_order", "mother_dtype or father_dtype failed")
            _skip("acyclic", "mother_dtype or father_dtype failed")
    else:
        for name in (
            "parent_refs_present_mother", "parent_refs_present_father",
            "parent_refs_sex_conflict", "self_loops", "parents_distinct",
            "sex_role_consistency", "topological_row_order", "acyclic",
        ):
            _skip(name, "id_dtype/negative_ids/duplicate_ids failed")

    ctx["ids"] = ids
    ctx["mothers"] = mothers
    ctx["fathers"] = fathers
    ctx["sex"] = sex
    return n, list(results.values()), findings, ctx


# ---------------------------------------------------------------------------
# Section 1: size & structure
# ---------------------------------------------------------------------------


def compute_size_structure(
    df: pd.DataFrame,
    children_csr: sp.csr_matrix | None,
) -> tuple[dict, np.ndarray]:
    """Counts, sex breakdown, generation depth, connected components.

    Returns (summary_dict, component_labels) where component_labels[i] is the
    connected-component id for row i.
    """
    n = len(df)
    has_mom = (df["mother"].to_numpy() != -1)
    has_dad = (df["father"].to_numpy() != -1)
    has_parent = has_mom | has_dad
    n_founders = int((~has_parent).sum())
    n_nonfounders = int(has_parent.sum())

    sex = df["sex"].to_numpy()
    n_male = int((sex == SEX_MALE).sum())
    n_female = int((sex == SEX_FEMALE).sum())

    gen = df["ped_depth"].to_numpy()
    max_depth = int(gen.max()) if n > 0 else 0
    gen_counts = np.bincount(gen).tolist() if n > 0 else []

    if children_csr is not None:
        n_components, comp_labels = csgraph.connected_components(children_csr, directed=False)
    else:
        n_components = n
        comp_labels = np.arange(n, dtype=np.int32)

    comp_sizes = np.bincount(comp_labels) if n > 0 else np.array([], dtype=np.int64)
    sorted_sizes = np.sort(comp_sizes)[::-1]

    summary = {
        "n_total": n,
        "n_founders": n_founders,
        "n_nonfounders": n_nonfounders,
        "n_male": n_male,
        "n_female": n_female,
        "max_depth": max_depth,
        "gen_counts": gen_counts,
        "n_components": int(n_components),
        "largest_component": int(sorted_sizes[0]) if len(sorted_sizes) else 0,
        "next_components": sorted_sizes[1:6].tolist(),
    }
    return summary, comp_labels


# ---------------------------------------------------------------------------
# Section 2: family sizes (ported from simace.analysis.stats.pedigree)
# ---------------------------------------------------------------------------


def _offspring_dist(counts: np.ndarray, n: int) -> dict:
    if n == 0:
        return dict.fromkeys(("0", "1", "2", "3", "4+"), 0.0)
    out = {"0": float((counts == 0).sum()) / n}
    for k in (1, 2, 3):
        out[str(k)] = float((counts == k).sum()) / n
    out["4+"] = float((counts >= 4).sum()) / n
    return out


def compute_family_sizes(df: pd.DataFrame) -> dict:
    """Sibship size distribution and offspring/mate counts (per simace conv.)."""
    both_present = (df["mother"] != -1) & (df["father"] != -1)
    children = df.loc[both_present]
    if len(children) == 0:
        return {"empty": True}

    family_sizes = children.groupby(["mother", "father"]).size()
    families_with_sibs = family_sizes[family_sizes >= 2].index
    has_sib = children.set_index(["mother", "father"]).index.isin(families_with_sibs)
    frac_with_full_sib = float(has_sib.sum()) / len(children)

    n_fam = len(family_sizes)
    size_dist = {str(k): float((family_sizes == k).sum()) / n_fam for k in (1, 2, 3)}
    size_dist["4+"] = float((family_sizes >= 4).sum()) / n_fam

    n_total = len(df)
    ids = df["id"].to_numpy()
    id_index = pd.Index(ids)
    m_rows = id_index.get_indexer(children["mother"].to_numpy())
    f_rows = id_index.get_indexer(children["father"].to_numpy())
    counts = np.bincount(m_rows, minlength=n_total) + np.bincount(f_rows, minlength=n_total)

    person_dist = _offspring_dist(counts, n_total)
    sex = df["sex"].to_numpy()
    male_mask = sex == SEX_MALE
    female_mask = sex == SEX_FEMALE
    person_dist_male = _offspring_dist(counts[male_mask], int(male_mask.sum()))
    person_dist_female = _offspring_dist(counts[female_mask], int(female_mask.sum()))

    mates_female = children.groupby("mother")["father"].nunique()
    mates_male = children.groupby("father")["mother"].nunique()

    return {
        "empty": False,
        "n_families": int(n_fam),
        "mean": float(family_sizes.mean()),
        "median": float(family_sizes.median()),
        "q1": float(family_sizes.quantile(0.25)),
        "q3": float(family_sizes.quantile(0.75)),
        "frac_with_full_sib": frac_with_full_sib,
        "size_dist": size_dist,
        "person_dist": person_dist,
        "person_dist_male": person_dist_male,
        "person_dist_female": person_dist_female,
        "mates_female_mean": float(mates_female.mean()) if len(mates_female) else 0.0,
        "mates_male_mean": float(mates_male.mean()) if len(mates_male) else 0.0,
    }


# ---------------------------------------------------------------------------
# Section 3: relationship pair count summary
# ---------------------------------------------------------------------------


def _augment_pair_counts(named: dict[str, int]) -> dict:
    """Add ``PO`` (= MO + FO) and ``by_degree`` aggregates to a named-codes dict.

    Both engine wrappers (``_count_pairs_matrix`` and ``_count_pairs_bfs``)
    share this so the YAML output schema is identical regardless of engine.
    """
    out = {code: int(count) for code, count in named.items()}
    out["PO"] = int(named.get("MO", 0) + named.get("FO", 0))
    by_degree = dict.fromkeys(range(6), 0)
    for code, count in named.items():
        by_degree[REL_REGISTRY[code].degree] += int(count)
    out["by_degree"] = by_degree
    return out


def count_relationship_pairs(
    df: pd.DataFrame,
    engine: str = "auto",
    threshold: int = _BFS_AUTO_THRESHOLD,
) -> dict:
    """Dispatch to the matrix or BFS relationship-pair enumerator.

    Returns a dict with the 23 named relationship codes through degree
    5 plus ``PO`` (= MO + FO, synthetic alias), ``by_degree`` (a
    map keyed 0..5 summing named counts by kinship degree), and
    ``_engine`` (the engine that produced the counts; consumed by
    ``_build_pedigree_data`` and emitted as ``pairs_engine`` in the
    summary).

    Assumes every non-``-1`` parent ID appears in ``df['id']``; this is
    enforced by ``load_and_validate``'s ``parent_refs_present_*`` checks
    and lets the internal ID-compaction ``reindex`` skip NaN handling.
    """
    chosen = _select_engine(len(df), engine, threshold)
    logger.info("relationship engine: %s (n=%d)", chosen, len(df))
    if chosen == "bfs":
        out = _count_pairs_bfs(df)
    else:
        out = _count_pairs_matrix(df)
    out["_engine"] = chosen
    return out


def _select_engine(n: int, requested: str, threshold: int) -> str:
    """Resolve ``--engine`` choice given pedigree size."""
    if requested == "matrix":
        return "matrix"
    if requested == "bfs" or n >= threshold:
        chosen = "bfs"
    else:
        chosen = "matrix"
    if chosen == "bfs":
        logger.warning(
            "BFS engine is experimental — perf claims unverified, may be removed",
        )
    return chosen


def _count_pairs_matrix(df: pd.DataFrame) -> dict:
    """Sparse matrix-power enumerator (default engine for n < threshold).

    Delegates relationship enumeration to ``pedigree_graph.PedigreeGraph``;
    this wrapper compacts IDs to ``0..n-1`` first because ``PedigreeGraph``
    allocates an ``id_to_row`` table sized to ``max(id)+1``.
    """
    pg = _build_compacted_graph(df)
    named = pg.count_pairs(max_degree=5)
    return _augment_pair_counts(named)


def _count_pairs_bfs(df: pd.DataFrame) -> dict:
    """BFS / boolean-matmul / numba enumerator (experimental).

    Thin wrapper around :func:`pedigree_graph.experimental.count_pairs_bfs`.
    See that function's docstring for the inbred-pedigree caveat
    (distinct-shared-ancestor counting vs path-multiplicity).
    """
    pg = _build_compacted_graph(df)
    named = count_pairs_bfs(pg, max_degree=5)
    return _augment_pair_counts(named)


def _build_compacted_graph(df: pd.DataFrame) -> PedigreeGraph:
    """Compact arbitrary IDs to ``0..n-1`` and build a ``PedigreeGraph``."""
    ids = df["id"].to_numpy()
    n = len(ids)
    new_ids = np.arange(n, dtype=np.int64)
    id_to_compact = pd.Series(new_ids, index=ids)

    def _remap(parents: np.ndarray) -> np.ndarray:
        return np.where(
            parents == -1, -1, id_to_compact.reindex(parents).to_numpy(),
        ).astype(np.int64)

    return PedigreeGraph.from_arrays(
        new_ids,
        _remap(df["mother"].to_numpy()),
        _remap(df["father"].to_numpy()),
    )


# ---------------------------------------------------------------------------
# Section 4: inbreeding (recursive memoized kinship)
# ---------------------------------------------------------------------------


def _merge_sorted_rows(
    a_cols: np.ndarray,
    a_vals: np.ndarray,
    b_cols: np.ndarray,
    b_vals: np.ndarray,
    scale_a: float = 1.0,
    scale_b: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two sorted (cols, vals) sparse rows; sums values at matching cols.

    Optional scale_a / scale_b are applied to the inputs during the merge.
    Vectorized two-pointer alternatives microbench slower than this
    argsort-based form for the ~100-entry rows we hit at scale.
    """
    if len(a_cols) == 0:
        return b_cols, (b_vals * scale_b) if scale_b != 1.0 else b_vals
    if len(b_cols) == 0:
        return a_cols, (a_vals * scale_a) if scale_a != 1.0 else a_vals
    all_cols = np.concatenate([a_cols, b_cols])
    all_vals = np.concatenate([a_vals * scale_a, b_vals * scale_b])
    order = np.argsort(all_cols, kind="stable")
    sorted_cols = all_cols[order]
    sorted_vals = all_vals[order]
    boundaries = np.concatenate([[0], np.where(np.diff(sorted_cols) != 0)[0] + 1])
    merged_cols = sorted_cols[boundaries].astype(np.int32)
    merged_vals = np.add.reduceat(sorted_vals, boundaries)
    return merged_cols, merged_vals


def compute_inbreeding(
    df: pd.DataFrame,
    id_index: pd.Index,
    children_csr: sp.csr_matrix | None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Per-individual inbreeding F via Henderson-Quaas L D L^T decomposition.

    Stores L[i, *] as a sorted (cols, vals) pair per row; non-zeros are
    only at k = ancestors of i (genuinely sparse, no sibling fill-in).
    F[i] = sum_k L[i, k]^2 * D[k] - 1, where D[k] is the Mendelian-sampling
    scaling derived from F[sire_k], F[dam_k] (already known in topological
    order). L rows are dropped as soon as their last child has consumed
    them, capping peak memory near max(remaining ancestor frontier).

    Returns (summary_dict, F_vector, n_ancestors) where n_ancestors[i] is the
    number of strict ancestors of row i (used by per-individual reports).
    """
    n = len(df)
    sire, _ = _parent_rows(df["father"].to_numpy(), id_index)
    dam, _ = _parent_rows(df["mother"].to_numpy(), id_index)

    n_children = (
        np.diff(children_csr.indptr).astype(np.int64) if children_csr is not None
        else np.zeros(n, dtype=np.int64)
    )

    gen = df["ped_depth"].to_numpy()
    order = np.argsort(gen, kind="stable")

    F = np.zeros(n, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)
    L_cols: dict[int, np.ndarray] = {}
    L_vals: dict[int, np.ndarray] = {}
    n_ancestors = np.zeros(n, dtype=np.int32)
    empty_c = np.array([], dtype=np.int32)
    empty_v = np.array([], dtype=np.float64)
    peak_entries = 0
    live_entries = 0

    def _drop(parent: int) -> None:
        nonlocal live_entries
        cols = L_cols.pop(parent)
        L_vals.pop(parent)
        n_ancestors[parent] = len(cols) - 1
        live_entries -= len(cols)

    for i in order:
        s = sire[i]
        d = dam[i]

        if s == -1 and d == -1:
            D[i] = 1.0
        elif s == -1:
            D[i] = 0.75 - 0.25 * F[d]
        elif d == -1:
            D[i] = 0.75 - 0.25 * F[s]
        else:
            D[i] = 0.5 - 0.25 * (F[s] + F[d])

        s_cols = L_cols.get(s, empty_c) if s != -1 else empty_c
        d_cols = L_cols.get(d, empty_c) if d != -1 else empty_c
        s_vals = L_vals.get(s, empty_v) if s != -1 else empty_v
        d_vals = L_vals.get(d, empty_v) if d != -1 else empty_v

        merged_cols, merged_vals = _merge_sorted_rows(
            s_cols, s_vals, d_cols, d_vals, scale_a=0.5, scale_b=0.5
        )

        new_cols = np.empty(len(merged_cols) + 1, dtype=np.int32)
        new_vals = np.empty(len(merged_vals) + 1, dtype=np.float64)
        new_cols[:-1] = merged_cols
        new_cols[-1] = i
        new_vals[:-1] = merged_vals
        new_vals[-1] = 1.0

        L_cols[i] = new_cols
        L_vals[i] = new_vals
        live_entries += len(new_cols)
        if live_entries > peak_entries:
            peak_entries = live_entries

        F[i] = float(np.sum(new_vals * new_vals * D[new_cols]) - 1.0)

        if s != -1:
            n_children[s] -= 1
            if n_children[s] == 0:
                _drop(s)
        if d != -1:
            n_children[d] -= 1
            if d != s and n_children[d] == 0:
                _drop(d)

    for i in list(L_cols.keys()):
        _drop(i)

    inbred = F > INBRED_TOL
    n_inbred = int(inbred.sum())
    edges = [0.0, 0.0625, 0.125, 0.25, 1.0]
    hist = {}
    hist["0"] = float((F <= INBRED_TOL).sum()) / n if n else 0.0
    for lo, hi in pairwise(edges):
        label = f"<{hi:g}"
        hist[label] = float(((lo < F) & (hi >= F)).sum()) / n if n else 0.0

    summary = {
        "n_inbred": n_inbred,
        "frac_inbred": n_inbred / n if n else 0.0,
        "mean_F": float(F.mean()) if n else 0.0,
        "max_F": float(F.max()) if n else 0.0,
        "hist": hist,
        "memo_size": peak_entries,
    }
    return summary, F, n_ancestors


# ---------------------------------------------------------------------------
# Section 5: per-individual data
# ---------------------------------------------------------------------------


def compute_descendants(df: pd.DataFrame, children_csr: sp.csr_matrix | None) -> np.ndarray:
    """Per-individual descendant count via reverse-topological scalar sum.

    Path-count semantics: ``n_desc[v]`` counts (v, w) walks down the DAG, not
    unique descendants. Identical to unique counts in non-inbred pedigrees;
    over-counts by the inbreeding rate where marriage loops give a
    descendant multiple ancestor paths to v. Matches the convention used
    for GP / Av / 1C in :func:`count_relationship_pairs`.
    """
    n = len(df)
    if n == 0 or children_csr is None:
        return np.zeros(n, dtype=np.int32)

    gen = df["ped_depth"].to_numpy()
    rev_order = np.argsort(-gen, kind="stable")
    indptr = children_csr.indptr
    indices = children_csr.indices

    n_desc = np.zeros(n, dtype=np.int64)
    for i in rev_order:
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue
        kids = indices[start:end]
        n_desc[i] = (end - start) + int(n_desc[kids].sum())

    return n_desc.astype(np.int32)


def build_individual_df(
    df: pd.DataFrame,
    id_index: pd.Index,
    F: np.ndarray,
    n_ancestors: np.ndarray,
    n_descendants: np.ndarray,
    component_labels: np.ndarray,
) -> pd.DataFrame:
    """Assemble per-individual table with the maximal column set."""
    n = len(df)
    ids_arr = df["id"].to_numpy()
    mothers = df["mother"].to_numpy()
    fathers = df["father"].to_numpy()
    sex = df["sex"].to_numpy()
    gen = df["ped_depth"].to_numpy()

    m_row, has_mom = _parent_rows(mothers, id_index)
    f_row, has_dad = _parent_rows(fathers, id_index)
    is_founder = ~(has_mom | has_dad)

    fs_count64, _, both_present = _full_sib_groups(df)
    fs_count = fs_count64.astype(np.int32)
    rows_m = np.where(has_mom)[0]
    rows_f = np.where(has_dad)[0]
    rows_bp = np.where(both_present)[0]

    n_mhs = np.zeros(n, dtype=np.int32)
    if rows_m.size:
        mom_groups = df.loc[has_mom].groupby("mother").size()
        share_mom = mom_groups.reindex(mothers[rows_m]).to_numpy().astype(np.int64) - 1
        n_mhs[rows_m] = share_mom.astype(np.int32) - fs_count[rows_m]

    n_phs = np.zeros(n, dtype=np.int32)
    if rows_f.size:
        dad_groups = df.loc[has_dad].groupby("father").size()
        share_dad = dad_groups.reindex(fathers[rows_f]).to_numpy().astype(np.int64) - 1
        n_phs[rows_f] = share_dad.astype(np.int32) - fs_count[rows_f]

    n_off = (
        np.bincount(m_row[has_mom], minlength=n).astype(np.int32)
        + np.bincount(f_row[has_dad], minlength=n).astype(np.int32)
    )

    n_mates = np.zeros(n, dtype=np.int32)
    if rows_bp.size:
        children = df.loc[both_present]
        mates_per_mother = children.groupby("mother")["father"].nunique()
        mates_per_father = children.groupby("father")["mother"].nunique()
        mom_rows = id_index.get_indexer(mates_per_mother.index.to_numpy())
        n_mates[mom_rows] = mates_per_mother.to_numpy().astype(np.int32)
        dad_rows = id_index.get_indexer(mates_per_father.index.to_numpy())
        n_mates[dad_rows] += mates_per_father.to_numpy().astype(np.int32)

    mm, mf, fm, ff = _grandparent_arrays(df)
    n_gp = (
        (mm != -1).astype(np.int32)
        + (mf != -1).astype(np.int32)
        + (fm != -1).astype(np.int32)
        + (ff != -1).astype(np.int32)
    )

    js = np.tile(np.arange(n, dtype=np.int64), 4)
    gps = np.concatenate([mm, mf, fm, ff])
    keep = gps != -1
    if keep.any():
        gp_rows = id_index.get_indexer(gps[keep])
        unique_pairs = np.unique(np.column_stack([js[keep], gp_rows]), axis=0)
        n_gc = np.bincount(unique_pairs[:, 1], minlength=n).astype(np.int32)
    else:
        n_gc = np.zeros(n, dtype=np.int32)

    mother_fs = np.zeros(n, dtype=np.int32)
    if rows_m.size:
        mother_fs[rows_m] = fs_count[m_row[rows_m]]
    father_fs = np.zeros(n, dtype=np.int32)
    if rows_f.size:
        father_fs[rows_f] = fs_count[f_row[rows_f]]
    n_ua = mother_fs + father_fs

    ua_offspring_sum = np.zeros(n, dtype=np.int32)
    if rows_bp.size:
        children = df.loc[both_present]
        noff_for_bp = n_off[rows_bp]
        mating_total = children.assign(noff=noff_for_bp).groupby(["mother", "father"])["noff"].sum()
        idx = pd.MultiIndex.from_arrays(
            [children["mother"].to_numpy(), children["father"].to_numpy()],
            names=["mother", "father"],
        )
        per_child_total = mating_total.reindex(idx).to_numpy().astype(np.int64)
        ua_offspring_sum[rows_bp] = (per_child_total - noff_for_bp).astype(np.int32)

    n_fc = np.zeros(n, dtype=np.int32)
    if rows_m.size:
        n_fc[rows_m] += ua_offspring_sum[m_row[rows_m]]
    if rows_f.size:
        n_fc[rows_f] += ua_offspring_sum[f_row[rows_f]]

    return pd.DataFrame(
        {
            "id": ids_arr,
            "sex": sex.astype(np.int8),
            "mother": mothers,
            "father": fathers,
            "ped_depth": gen.astype(np.int32),
            "is_founder": is_founder,
            "F": F.astype(np.float64),
            "n_full_sibs": fs_count,
            "n_mat_half_sibs": n_mhs,
            "n_pat_half_sibs": n_phs,
            "n_offspring": n_off,
            "n_mates": n_mates,
            "component_id": component_labels.astype(np.int32),
            "n_grandparents": n_gp,
            "n_grandchildren": n_gc,
            "n_uncles_aunts": n_ua,
            "n_first_cousins": n_fc,
            "n_ancestors": n_ancestors,
            "n_descendants": n_descendants.astype(np.int32),
        }
    )


# ---------------------------------------------------------------------------
# Section 6: structured (YAML / long-form TSV) outputs
# ---------------------------------------------------------------------------


_NUMERIC_COLS = (
    "F",
    "n_full_sibs",
    "n_mat_half_sibs",
    "n_pat_half_sibs",
    "n_offspring",
    "n_mates",
    "n_grandparents",
    "n_grandchildren",
    "n_uncles_aunts",
    "n_first_cousins",
    "n_ancestors",
    "n_descendants",
)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_pedigree_data(
    path: Path,
    cmd: str,
    size: dict,
    family: dict,
    pairs: dict,
    inbreeding: dict | None,
) -> dict:
    """Canonical nested dict for the pedigree-level report; safe to YAML-dump."""
    family_section: dict | None
    if family.get("empty"):
        family_section = None
    else:
        family_section = {
            "n_families": int(family["n_families"]),
            "mean": float(family["mean"]),
            "median": float(family["median"]),
            "q1": float(family["q1"]),
            "q3": float(family["q3"]),
            "frac_with_full_sib": float(family["frac_with_full_sib"]),
            "size_dist": {str(k): float(v) for k, v in family["size_dist"].items()},
            "person_dist": {str(k): float(v) for k, v in family["person_dist"].items()},
            "person_dist_male": {str(k): float(v) for k, v in family["person_dist_male"].items()},
            "person_dist_female": {str(k): float(v) for k, v in family["person_dist_female"].items()},
            "mates_female_mean": float(family["mates_female_mean"]),
            "mates_male_mean": float(family["mates_male_mean"]),
        }
    inb_section: dict | None
    if inbreeding is None:
        inb_section = None
    else:
        inb_section = {
            "n_inbred": int(inbreeding["n_inbred"]),
            "frac_inbred": float(inbreeding["frac_inbred"]),
            "mean_F": float(inbreeding["mean_F"]),
            "max_F": float(inbreeding["max_F"]),
            "hist": {str(k): float(v) for k, v in inbreeding["hist"].items()},
            "memo_size": int(inbreeding["memo_size"]),
        }
    return {
        "input": str(path),
        "command": cmd,
        "version": VERSION,
        "generated_at": _now_iso(),
        "n_total": int(size["n_total"]),
        "max_degree_enumerated": 5,
        "size_structure": {
            "n_founders": int(size["n_founders"]),
            "n_nonfounders": int(size["n_nonfounders"]),
            "n_male": int(size["n_male"]),
            "n_female": int(size["n_female"]),
            "max_depth": int(size["max_depth"]),
            "gen_counts": [int(x) for x in size["gen_counts"]],
            "n_components": int(size["n_components"]),
            "largest_component": int(size["largest_component"]),
            "next_components": [int(x) for x in size["next_components"]],
        },
        "family_size": family_section,
        "pairs_engine": str(pairs.get("_engine", "matrix")),
        "pairs": {
            k: ({str(deg): int(c) for deg, c in v.items()} if k == "by_degree" else int(v))
            for k, v in pairs.items()
            if k != "_engine"
        },
        "inbreeding": inb_section,
    }


def _build_individual_data(idf: pd.DataFrame, path: Path, cmd: str) -> dict:
    """Canonical nested dict for the per-individual distribution report."""
    n = len(idf)
    distributions: dict[str, dict] = {}
    for col in _NUMERIC_COLS:
        if col not in idf.columns:
            continue
        c = idf[col]
        is_float = pd.api.types.is_float_dtype(c)
        cast = float if is_float else int
        distributions[col] = {
            "mean": float(c.mean()) if n else 0.0,
            "std": float(c.std()) if n > 1 else 0.0,
            "min": cast(c.min()) if n else 0,
            "q1": cast(c.quantile(0.25)) if n else 0,
            "median": cast(c.median()) if n else 0,
            "q3": cast(c.quantile(0.75)) if n else 0,
            "max": cast(c.max()) if n else 0,
            "nz": int((c != 0).sum()),
        }

    out: dict = {
        "input": str(path),
        "command": cmd,
        "version": VERSION,
        "generated_at": _now_iso(),
        "n_total": n,
        "distributions": distributions,
    }

    if "is_founder" in idf.columns:
        n_f = int(idf["is_founder"].sum())
        out["founders"] = {"n": n_f, "frac": (n_f / n) if n else 0.0}
    if "sex" in idf.columns:
        out["sex_breakdown"] = {
            "male": int((idf["sex"] == SEX_MALE).sum()),
            "female": int((idf["sex"] == SEX_FEMALE).sum()),
        }
    if "ped_depth" in idf.columns and "F" in idf.columns:
        gens = []
        for g, sub in idf.groupby("ped_depth"):
            gens.append({
                "gen": int(g),
                "n": len(sub),
                "mean_F": float(sub["F"].mean()),
                "max_F": float(sub["F"].max()),
                "n_inbred": int((sub["F"] > INBRED_TOL).sum()),
            })
        out["f_by_generation"] = gens
    if "component_id" in idf.columns:
        sizes = idf.groupby("component_id").size()
        components: dict = {
            "n": int(sizes.size),
            "largest": int(sizes.max()) if sizes.size else 0,
            "median_size": int(sizes.median()) if sizes.size else 0,
        }
        if sizes.size > 1:
            singletons = int((sizes == 1).sum())
            components["singletons"] = singletons
            components["singletons_frac"] = singletons / int(sizes.size)
        out["components"] = components
    if "n_offspring" in idf.columns and "sex" in idf.columns:
        ob: dict = {}
        for label, code in (("female", SEX_FEMALE), ("male", SEX_MALE)):
            sub = idf.loc[idf["sex"] == code, "n_offspring"]
            if len(sub) > 0:
                no_kids = int((sub == 0).sum())
                ob[label] = {
                    "n": len(sub),
                    "mean": float(sub.mean()),
                    "max": int(sub.max()),
                    "no_children": no_kids,
                    "no_children_frac": no_kids / len(sub),
                }
        if ob:
            out["offspring_by_sex"] = ob
    return out


_SUMMARY_META_KEYS = ("input", "command", "version", "generated_at", "n_total")

SAFE_MIN_CELL = 5


def _apply_safe_attempt(ped_data: dict, ind_data: dict, min_cell: int = SAFE_MIN_CELL) -> None:
    """Best-effort small-cell redaction (in place). Not a safe-harbor.

    - Pedigree-level: nulls ``pairs`` codes and ``inbreeding`` fields below
      ``min_cell``; suppresses ``hist`` buckets whose implied count
      (``frac × n_total``) is below the threshold; drops ``next_components``
      entries below it; nulls positional ``gen_counts`` entries below it.
    - Individual-level: drops ``min``/``max`` from every distribution, nulls
      ``nz`` below threshold, nulls ``f_by_generation`` rows with ``n <
      min_cell``, nulls component / sex / offspring strata with ``n <
      min_cell``.
    """
    n_total = int(ped_data.get("n_total", 0))

    sizes = ped_data.get("size_structure", {})
    sizes["next_components"] = [
        s for s in sizes.get("next_components", []) if s >= min_cell
    ]
    sizes["gen_counts"] = [
        (g if g >= min_cell else None) for g in sizes.get("gen_counts", [])
    ]
    if 0 < int(sizes.get("largest_component", 0)) < min_cell:
        sizes["largest_component"] = None

    fam = ped_data.get("family_size")
    if fam is not None and 0 < int(fam.get("n_families", 0)) < min_cell:
        for k in (
            "mean", "median", "q1", "q3", "frac_with_full_sib",
            "mates_female_mean", "mates_male_mean",
        ):
            if k in fam:
                fam[k] = None
        for k in ("size_dist", "person_dist", "person_dist_male", "person_dist_female"):
            if k in fam and isinstance(fam[k], dict):
                fam[k] = dict.fromkeys(fam[k])

    pairs = ped_data.get("pairs", {})
    for code, count in list(pairs.items()):
        if code == "by_degree":
            for d, c in pairs[code].items():
                if 0 < c < min_cell:
                    pairs[code][d] = None
        elif isinstance(count, int) and 0 < count < min_cell:
            pairs[code] = None

    inb = ped_data.get("inbreeding")
    if inb is not None:
        if 0 < int(inb.get("n_inbred", 0)) < min_cell:
            inb["n_inbred"] = None
            inb["frac_inbred"] = None
            inb["mean_F"] = None
            inb["max_F"] = None
        for bucket, frac in list(inb.get("hist", {}).items()):
            if frac is None:
                continue
            if 0 < frac * n_total < min_cell:
                inb["hist"][bucket] = None

    for col, dist in ind_data.get("distributions", {}).items():
        dist.pop("min", None)
        dist.pop("max", None)
        if 0 < int(dist.get("nz", 0)) < min_cell:
            dist["nz"] = None
        _ = col  # silence unused-loop-var

    founders = ind_data.get("founders")
    if founders is not None and 0 < int(founders.get("n", 0)) < min_cell:
        founders["n"] = None
        founders["frac"] = None

    sx = ind_data.get("sex_breakdown")
    if sx is not None:
        for k in ("male", "female"):
            if 0 < int(sx.get(k, 0)) < min_cell:
                sx[k] = None

    for row in ind_data.get("f_by_generation", []):
        if int(row.get("n", 0)) < min_cell:
            row["mean_F"] = None
            row["max_F"] = None
            row["n_inbred"] = None

    comps = ind_data.get("components")
    if comps is not None:
        for k in ("largest", "median_size", "singletons"):
            if 0 < int(comps.get(k, 0) or 0) < min_cell:
                comps[k] = None
        if comps.get("singletons") is None and "singletons_frac" in comps:
            comps["singletons_frac"] = None

    obs = ind_data.get("offspring_by_sex", {})
    for stats in obs.values():
        if int(stats.get("n", 0)) < min_cell:
            stats["mean"] = None
            stats["max"] = None
            stats["no_children"] = None
            stats["no_children_frac"] = None
        elif 0 < int(stats.get("no_children", 0)) < min_cell:
            stats["no_children"] = None
            stats["no_children_frac"] = None


def _build_summary_data(ped_data: dict, ind_data: dict) -> dict:
    """Merge per-section dicts into one summary; shared meta lifted to the top."""
    out = {k: ped_data[k] for k in _SUMMARY_META_KEYS}
    out["pedigree"] = {
        k: v for k, v in ped_data.items() if k not in _SUMMARY_META_KEYS
    }
    out["individual"] = {
        k: v for k, v in ind_data.items() if k not in _SUMMARY_META_KEYS
    }
    return out


def _flatten_long(obj, prefix: tuple = ()):
    """Yield (section, key, subkey, value) rows from a nested dict / list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten_long(v, (*prefix, str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten_long(v, (*prefix, str(i)))
    else:
        if not prefix:
            return
        section = "meta" if len(prefix) == 1 else prefix[0]
        key = prefix[0] if len(prefix) == 1 else prefix[1]
        subkey = ".".join(prefix[2:]) if len(prefix) > 2 else ""
        yield section, key, subkey, obj


def _round_floats(obj: object, ndigits: int = 4) -> object:
    """Recursively round floats in nested dicts/lists to ``ndigits``."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(x, ndigits) for x in obj]
    return obj


def _write_yaml(data: dict, path: Path) -> None:
    """Write data as YAML to path (creates parent dirs); floats rounded to 4dp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(_round_floats(data), fh, sort_keys=False, default_flow_style=False)


def _write_long_tsv(data: dict, path: Path) -> None:
    """Write data flattened to a long-form TSV; floats rounded to 4dp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(_flatten_long(_round_floats(data)))
    df = pd.DataFrame(rows, columns=["section", "key", "subkey", "value"])
    df.to_csv(path, sep="\t", index=False)


def _at(base: Path, suffix: str) -> Path:
    """Append ``suffix`` to ``base`` (handles paths without extensions)."""
    return base.with_name(base.name + suffix)


def _to_csv_gz(df: pd.DataFrame, out_path: Path) -> None:
    """Write ``df`` as gzipped TSV; uses pigz when available, else gzip level 1.

    Output is standard .gz either way.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pigz = shutil.which("pigz")
    if pigz is not None:
        with out_path.open("wb") as fh_out, subprocess.Popen(
            [pigz, "-1", "-p", "4", "-c"], stdin=subprocess.PIPE, stdout=fh_out,
        ) as proc:
            df.to_csv(proc.stdin, sep="\t", index=False)
            proc.stdin.close()
            if proc.wait() != 0:
                raise PedigreeError(f"pigz exited with status {proc.returncode}")
        return
    with gzip.open(out_path, "wb", compresslevel=1) as fh:
        df.to_csv(fh, sep="\t", index=False)


def _write_annotated_tsv(
    in_path: Path,
    args: argparse.Namespace,
    idf: pd.DataFrame,
    out_path: Path,
) -> None:
    """Re-read input pedigree, append derived columns, write annotated tsv.gz.

    Output preserves input columns under canonical names (id/sex/mother/
    father; user-supplied names are renamed). Sex is the validated int
    encoding (0=female, 1=male). All derived per-individual columns are
    appended. Row order matches input. Gzipped tab-separated.
    """
    raw = pd.read_csv(in_path, sep="\t")
    rename_map = {
        args.id_col: "id",
        args.sex_col: "sex",
        args.mother_col: "mother",
        args.father_col: "father",
    }
    rename_map = {k: v for k, v in rename_map.items() if k != v}
    if rename_map:
        raw = raw.rename(columns=rename_map)

    raw_ids = pd.to_numeric(raw["id"], errors="raise").astype(np.int64).to_numpy()
    if not np.array_equal(raw_ids, idf["id"].to_numpy()):
        raise PedigreeError(
            "internal: row order mismatch between input and individual table"
        )

    canonical = ("id", "sex", "mother", "father")
    extras = raw.drop(columns=[c for c in canonical if c in raw.columns]).reset_index(drop=True)

    derived_cols = set(idf.columns) - set(canonical)
    collisions = [c for c in extras.columns if c in derived_cols]
    if collisions:
        new_names = {c: f"{c}_input" for c in collisions}
        extras = extras.rename(columns=new_names)
        logger.warning(
            "input columns %s collide with derived columns; preserved as %s",
            collisions, [new_names[c] for c in collisions],
        )

    annotated = pd.concat([idf.reset_index(drop=True), extras], axis=1)
    _to_csv_gz(annotated, out_path)


# ---------------------------------------------------------------------------
# Section 9: validate output (stderr summary, log file, fixed-pedigree tsv.gz)
# ---------------------------------------------------------------------------


def _format_check_summary(path: Path, n_total: int, results: list[CheckResult]) -> str:
    """Multi-line stderr summary: header + per-check PASS/FAIL/SKIP + result line."""
    width = max(len(r.name) for r in results) + 2
    lines = [f"pedigree_summary.py: validating {path} (N={n_total:,})"]
    for r in results:
        line = f"  {r.name:<{width}} {r.status}"
        if r.status == "FAIL":
            line += f" ({r.count})"
        elif r.status == "SKIP" and r.skip_reason:
            line += f" ({r.skip_reason})"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _write_validate_log(findings: list[Finding], out_path: Path) -> None:
    """Tab-separated log: one row per finding (check / id / row / detail)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "check": f.check,
            "id": "" if f.id is None else f.id,
            "row": "" if f.row is None else f.row,
            "detail": f.detail,
        }
        for f in findings
    ]
    df = pd.DataFrame(rows, columns=["check", "id", "row", "detail"])
    df.to_csv(out_path, sep="\t", index=False)


def _build_added_founders(
    mothers: np.ndarray,
    fathers: np.ndarray,
    id_index: pd.Index,
    no_sex_check: bool,
) -> list[dict]:
    """Synthesize founder rows for missing parent IDs, sorted by ID."""
    moms_ref = np.unique(mothers[mothers != -1])
    dads_ref = np.unique(fathers[fathers != -1])
    moms_missing = set(moms_ref[id_index.get_indexer(moms_ref) == -1].tolist())
    dads_missing = set(dads_ref[id_index.get_indexer(dads_ref) == -1].tolist())
    conflicts = moms_missing & dads_missing
    moms_only = moms_missing - conflicts
    dads_only = dads_missing - conflicts

    def _rows_listing(rows: np.ndarray) -> str:
        if len(rows) <= 5:
            return f"row(s) {rows.tolist()}"
        return f"row(s) {rows[:5].tolist()} (and {len(rows) - 5} more)"

    out: list[dict] = []
    for mid in sorted(moms_only):
        rows = np.where(mothers == mid)[0]
        out.append({"id": int(mid), "sex": "F",
                    "reason": f"referenced as mother in {_rows_listing(rows)}"})
    for did in sorted(dads_only):
        rows = np.where(fathers == did)[0]
        out.append({"id": int(did), "sex": "M",
                    "reason": f"referenced as father in {_rows_listing(rows)}"})
    if no_sex_check:
        for cid in sorted(conflicts):
            rows_m = np.where(mothers == cid)[0]
            rows_f = np.where(fathers == cid)[0]
            out.append({"id": int(cid), "sex": "F",
                        "reason": (
                            f"--no-sex-check; conflicting roles "
                            f"(mother {_rows_listing(rows_m)}, father {_rows_listing(rows_f)})"
                        )})
    out.sort(key=lambda x: x["id"])
    return out


def _write_validate_tsv_gz(
    df_raw: pd.DataFrame,
    added_founders: list[dict],
    id_col: str,
    sex_col: str,
    mother_col: str,
    father_col: str,
    out_path: Path,
) -> None:
    """Write input pedigree (gzipped TSV), with new founder rows prepended at top."""
    if added_founders:
        new_rows = pd.DataFrame(
            {col: [""] * len(added_founders) for col in df_raw.columns}
        )
        new_rows[id_col] = [str(f["id"]) for f in added_founders]
        new_rows[sex_col] = [f["sex"] for f in added_founders]
        new_rows[mother_col] = ["-1"] * len(added_founders)
        new_rows[father_col] = ["-1"] * len(added_founders)
        out_df = pd.concat([new_rows, df_raw.reset_index(drop=True)], ignore_index=True)
    else:
        out_df = df_raw
    _to_csv_gz(out_df, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _FullHelpParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help (not just usage) on parse errors."""

    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        sys.stderr.write(f"\nerror: {message}\n")
        sys.exit(2)


def _add_logging_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "-v", "--verbose", action="store_true",
        help="DEBUG-level logging to stderr (default: INFO)",
    )
    g.add_argument(
        "-q", "--quiet", action="store_true",
        help="WARNING-level logging only (suppress per-section timings)",
    )


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--engine", choices=("auto", "matrix", "bfs"), default="auto",
        help="relationship-pair enumeration engine. ``auto`` picks "
        "``bfs`` when n is at or above the threshold; otherwise ``matrix`` "
        "(default: %(default)s). The ``bfs`` engine is experimental — see "
        "the README's 'Choosing an engine' section.",
    )
    p.add_argument(
        "--bfs-threshold", type=int, default=_BFS_AUTO_THRESHOLD,
        metavar="N",
        help="auto-select threshold for the bfs engine (default: %(default)s)",
    )


def _add_format_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--plink-sex", action="store_true",
        help="interpret the sex column with the PLINK convention (1=male, "
        "2=female) instead of the default 0=female, 1=male",
    )
    p.add_argument(
        "--zero-as-missing", action="store_true",
        help="treat 0 in mother/father columns as missing (PLINK fam "
        "convention). NA/blank/N/A/./? are always treated as missing.",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _FullHelpParser(
        prog="pedigree_summary.py",
        description=(
            "Pedigree summary CLI. Self-contained: depends only on "
            "numpy, scipy, pandas, pyyaml."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {VERSION}",
    )
    sub = parser.add_subparsers(dest="subcommand", parser_class=_FullHelpParser)

    p_sum = sub.add_parser("summarize", help="summarise a pedigree (TSV input)")
    p_sum.add_argument(
        "--in", dest="in_path", required=True, type=Path,
        help="input pedigree (.tsv or .tsv.gz)",
    )
    p_sum.add_argument(
        "--out", dest="out_basename", required=True, type=Path, metavar="BASENAME",
        help="output basename (no extension); writes BASENAME.summary.yaml "
        "(combined pedigree+individual summary), BASENAME.summary.pedigree.tsv, "
        "BASENAME.summary.individual.tsv, and BASENAME.annotated.tsv.gz "
        "(input pedigree + per-individual columns; suppressed under "
        "--safe-attempt)",
    )
    p_sum.add_argument(
        "--id-col", default="id", metavar="NAME",
        help="column name for individual ID (int) (default: %(default)s)",
    )
    p_sum.add_argument(
        "--sex-col", default="sex", metavar="NAME",
        help="column name for sex; accepts M/F (any case), Male/Female, or "
        "0/1 (default: %(default)s; 0=female, 1=male). See --plink-sex.",
    )
    p_sum.add_argument(
        "--mother-col", default="mother", metavar="NAME",
        help="column name for mother ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p_sum.add_argument(
        "--father-col", default="father", metavar="NAME",
        help="column name for father ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p_sum.add_argument(
        "--inbreeding",
        action="store_true",
        help="compute inbreeding F (off by default; ~5 min on 10M rows). "
        "When off, F and n_ancestors in the per-individual table are zero-filled.",
    )
    p_sum.add_argument(
        "--safe-attempt",
        action="store_true",
        help="best-effort GDPR-style redaction: skip the per-individual "
        "annotated TSV, drop min/max from distributions, and null any "
        "count or stratum below cell-size 5. Not a safe-harbor guarantee.",
    )
    _add_engine_args(p_sum)
    _add_format_args(p_sum)
    _add_logging_args(p_sum)

    p_val = sub.add_parser("validate", help="run all integrity checks accumulating; report issues")
    p_val.add_argument("--in", dest="in_path", required=True, type=Path, help="input pedigree TSV")
    p_val.add_argument(
        "--out", dest="out_basename", required=True, type=Path, metavar="BASENAME",
        help="output basename (no extension); writes BASENAME.validate.log "
        "(per-finding TSV) and BASENAME.validate.tsv.gz (the pedigree with any "
        "auto-fixes applied; not written if a block is detected)",
    )
    p_val.add_argument(
        "--id-col", default="id", metavar="NAME",
        help="column name for individual ID (int) (default: %(default)s)",
    )
    p_val.add_argument(
        "--sex-col", default="sex", metavar="NAME",
        help="column name for sex; accepts M/F or 0/1 with 0=female, 1=male "
        "(default: %(default)s)",
    )
    p_val.add_argument(
        "--mother-col", default="mother", metavar="NAME",
        help="column name for mother ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p_val.add_argument(
        "--father-col", default="father", metavar="NAME",
        help="column name for father ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p_val.add_argument(
        "--no-sex-check", action="store_true",
        help="bypass the sex-conflict check on missing parents; auto-added "
        "founders default to sex=F when the role is ambiguous (default: off)",
    )
    _add_format_args(p_val)
    _add_logging_args(p_val)

    args = parser.parse_args(argv)
    if args.subcommand is None:
        parser.print_help(sys.stderr)
        sys.exit(0)
    return args


def _init_logging(verbose: bool, quiet: bool) -> None:
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


def _run_summarize(args: argparse.Namespace, cmd: str) -> int:
    try:
        df, children_csr = load_and_validate(
            args.in_path,
            id_col=args.id_col,
            sex_col=args.sex_col,
            mother_col=args.mother_col,
            father_col=args.father_col,
            plink_sex=args.plink_sex,
            zero_as_missing=args.zero_as_missing,
        )
    except PedigreeError as e:
        logger.error("validation failed: %s", e)
        return 1
    except (FileNotFoundError, OSError) as e:
        logger.error("file error: %s", e)
        return 2

    id_index = pd.Index(df["id"].to_numpy())

    t0 = time.perf_counter()
    size, comp_labels = compute_size_structure(df, children_csr)
    logger.info("size+structure in %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    family = compute_family_sizes(df)
    logger.info("family sizes in %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    pairs = count_relationship_pairs(
        df, engine=args.engine, threshold=args.bfs_threshold,
    )
    logger.info("relationship pairs in %.2fs", time.perf_counter() - t0)

    n_indiv = len(df)
    if args.inbreeding:
        t0 = time.perf_counter()
        inb_summary, F_vec, n_anc = compute_inbreeding(df, id_index, children_csr)
        logger.info(
            "inbreeding in %.2fs (peak L entries: %d)",
            time.perf_counter() - t0,
            inb_summary["memo_size"],
        )
    else:
        logger.info("inbreeding: skipped (pass --inbreeding to enable)")
        inb_summary: dict | None = None
        F_vec = np.zeros(n_indiv, dtype=np.float64)
        n_anc = np.zeros(n_indiv, dtype=np.int32)

    base = args.out_basename
    base.parent.mkdir(parents=True, exist_ok=True)

    ped_data = _build_pedigree_data(
        args.in_path, cmd, size, family, pairs, inb_summary,
    )

    t0 = time.perf_counter()
    n_desc = compute_descendants(df, children_csr)
    logger.info("descendants in %.2fs", time.perf_counter() - t0)
    t0 = time.perf_counter()
    idf = build_individual_df(df, id_index, F_vec, n_anc, n_desc, comp_labels)
    logger.info("individual table built in %.2fs", time.perf_counter() - t0)

    ind_data = _build_individual_data(idf, args.in_path, cmd)

    if args.safe_attempt:
        _apply_safe_attempt(ped_data, ind_data)
        logger.info("safe-attempt redaction applied (min cell = %d)", SAFE_MIN_CELL)

    _write_long_tsv(ped_data, _at(base, ".summary.pedigree.tsv"))
    _write_long_tsv(ind_data, _at(base, ".summary.individual.tsv"))

    summary_data = _build_summary_data(ped_data, ind_data)
    _write_yaml(summary_data, _at(base, ".summary.yaml"))
    logger.info(
        "wrote %s.summary.yaml + %s.summary.{pedigree,individual}.tsv", base, base,
    )

    if args.safe_attempt:
        logger.info("safe-attempt: skipped %s.annotated.tsv.gz (per-individual)", base)
    else:
        t0 = time.perf_counter()
        _write_annotated_tsv(args.in_path, args, idf, _at(base, ".annotated.tsv.gz"))
        logger.info("wrote annotated pedigree to %s.annotated.tsv.gz in %.2fs", base, time.perf_counter() - t0)

    return 0


def _run_validate(args: argparse.Namespace, cmd: str) -> int:
    try:
        n_total, results, findings, ctx = validate_pedigree(
            args.in_path,
            id_col=args.id_col,
            sex_col=args.sex_col,
            mother_col=args.mother_col,
            father_col=args.father_col,
            plink_sex=args.plink_sex,
            zero_as_missing=args.zero_as_missing,
        )
    except PedigreeError as e:
        logger.error("validation could not run: %s", e)
        return 2
    except (FileNotFoundError, OSError) as e:
        logger.error("file error: %s", e)
        return 2

    by_check = {r.name: r for r in results}

    if args.no_sex_check:
        findings = [f for f in findings if f.check != "parent_refs_sex_conflict"]
        by_check["parent_refs_sex_conflict"] = CheckResult(
            name="parent_refs_sex_conflict", status="SKIP",
            skip_reason="bypassed via --no-sex-check",
        )
        results = [by_check[name] for name in _CHECK_ORDER]

    blocks: list[str] = []
    if by_check["duplicate_ids"].status == "FAIL":
        blocks.append("duplicate IDs detected")
    if by_check["acyclic"].status == "FAIL":
        blocks.append("cycle detected")
    if by_check["parents_distinct"].status == "FAIL":
        blocks.append("rows with mother == father (cannot disambiguate)")
    if by_check["parent_refs_sex_conflict"].status == "FAIL":
        blocks.append("sex conflict on missing parent(s); pass --no-sex-check to default to sex=F")

    sys.stderr.write(_format_check_summary(args.in_path, n_total, results))

    base = args.out_basename
    base.parent.mkdir(parents=True, exist_ok=True)
    log_path = _at(base, ".validate.log")
    _write_validate_log(findings, log_path)
    sys.stderr.write(f"wrote {log_path} ({len(findings)} finding(s))\n")

    if blocks:
        sys.stderr.write("\nBLOCKED — fix the following before re-running:\n")
        for b in blocks:
            sys.stderr.write(f"  - {b}\n")
        return 2

    added_founders: list[dict] = []
    if ctx["ids"] is not None and ctx["mothers"] is not None and ctx["fathers"] is not None:
        id_index = pd.Index(ctx["ids"])
        added_founders = _build_added_founders(
            ctx["mothers"], ctx["fathers"], id_index, args.no_sex_check,
        )

    out_path = _at(base, ".validate.tsv.gz")
    _write_validate_tsv_gz(
        ctx["df_raw"], added_founders,
        args.id_col, args.sex_col, args.mother_col, args.father_col,
        out_path,
    )
    n_total_out = n_total + len(added_founders)
    sys.stderr.write(
        f"wrote {out_path} ({n_total_out:,} rows; {len(added_founders)} founder(s) added)\n"
    )

    return 0 if not findings else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    args = _parse_args(argv)
    _init_logging(args.verbose, args.quiet)
    cmd = " ".join(sys.argv)
    if args.subcommand == "summarize":
        return _run_summarize(args, cmd)
    if args.subcommand == "validate":
        return _run_validate(args, cmd)
    return 1


if __name__ == "__main__":
    sys.exit(main())
