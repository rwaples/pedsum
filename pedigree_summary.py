#!/usr/bin/env python3
"""Pedigree summary CLI.

Reads a TSV pedigree (id, sex, mother, father), validates it, and writes
machine-readable summaries covering size, structure, family-size
distribution, relationship-pair counts, per-individual inbreeding (F),
and pedigree-based effective population size (Ne).

Outputs of ``summarize`` (with ``--out DIR``):
    DIR/summary.yaml              slim categorised summary (~500 lines)
    DIR/summary.extra.yaml        per-generation / per-cohort arrays and
                                  full per-individual quantiles
    DIR/annotated.tsv.gz          input pedigree + per-individual cols
                                  (suppressed under ``--safe-attempt``)
    DIR/summary.pedigree.tsv      long-form pedigree-level summary
                                  (only with ``--tsv``)
    DIR/summary.individual.tsv    long-form per-individual distribution
                                  (only with ``--tsv``)

Outputs of ``validate`` (with ``--out DIR``):
    DIR/validate.log              per-finding TSV
    DIR/validate.tsv.gz           fixed pedigree (omitted on hard block)

Pedsum 0.7 makes F and the seven cheap Ne estimators on by default;
pass ``--no-inbreeding`` or ``--no-effective-size`` to skip. The
coancestry-rate Ne_C estimator remains opt-in via ``--ne-coancestry``
because its kinship DP can blow up RAM on very large pedigrees.

Single-file CLI on top of numpy, scipy, pandas, pyyaml, and the
``pedigree_graph`` package (which provides the matrix-engine
relationship enumeration through degree 5 plus the streaming-scalar
pair counter that is the default for the 23 named codes).

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
case-insensitively.

Usage:
    python pedigree_summary.py summarize --in PED.tsv --out DIR [options]
    python pedigree_summary.py validate  --in PED.tsv --out DIR [options]
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
_F_KERNEL_WARN_THRESHOLD = 1_000_000

VERSION = "0.10.0"
SEX_FEMALE = 0
SEX_MALE = 1
SEX_UNKNOWN = -1
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
    "sex_role_ambiguity",
    "self_loops",
    "parents_distinct",
    "sex_role_consistency",
    "unknown_sex",
    "acyclic",
    "birth_year_dtype",
    "birth_year_range",
    "birth_year_topology",
)


_CHECK_LABELS: dict[str, str] = {
    "required_columns":            "required columns present",
    "id_dtype":                    "id column parses as integer",
    "mother_dtype":                "mother column parses as integer",
    "father_dtype":                "father column parses as integer",
    "sex_tokens":                  "sex column tokens recognized",
    "negative_ids":                "no negative IDs",
    "duplicate_ids":               "no duplicate IDs",
    "parent_token_range_mother":   "mother IDs in valid range",
    "parent_token_range_father":   "father IDs in valid range",
    "parent_refs_present_mother":  "mother IDs present in pedigree",
    "parent_refs_present_father":  "father IDs present in pedigree",
    "parents_distinct":            "mother and father distinct",
    "parent_refs_sex_conflict":    "no missing-parent sex conflicts",
    "sex_role_ambiguity":          "no role-ambiguous unsexed individuals",
    "self_loops":                  "no self-loops",
    "sex_role_consistency":        "sex consistent with parent role",
    "unknown_sex":                 "all individuals have resolved sex",
    "acyclic":                     "acyclic (no descent cycles)",
    "empty_pedigree":              "pedigree is non-empty",
    "birth_year_dtype":            "birth_year column parses as numeric",
    "birth_year_range":            "birth years within sanity range",
    "birth_year_topology":         "child birth_year >= parent birth_year",
}


_CHECK_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Columns & parsing", (
        "required_columns", "empty_pedigree",
        "id_dtype", "mother_dtype", "father_dtype", "sex_tokens",
    )),
    ("IDs", ("negative_ids", "duplicate_ids")),
    ("Parent references", (
        "parent_token_range_mother", "parent_token_range_father",
        "parent_refs_present_mother", "parent_refs_present_father",
        "parents_distinct",
        "parent_refs_sex_conflict", "sex_role_ambiguity",
    )),
    ("Graph structure", (
        "self_loops", "sex_role_consistency", "unknown_sex", "acyclic",
    )),
    ("Birth years (optional)", (
        "birth_year_dtype", "birth_year_range", "birth_year_topology",
    )),
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


def _check_sex_role_consistency(
    mothers: np.ndarray,
    fathers: np.ndarray,
    sex: np.ndarray,
    id_index: pd.Index,
    *,
    skip_mask: np.ndarray | None = None,
) -> list[Finding]:
    """Detect IDs used as mother but sex != female, or as father but sex != male.

    When ``skip_mask`` is supplied, rows whose mask entry is True are excluded
    from both role tests. This is used to bypass rows whose sex started as
    unknown (already covered by ``sex_role_ambiguity`` and ``unknown_sex``).
    """
    used_as_mother = np.unique(mothers[mothers != -1])
    used_as_father = np.unique(fathers[fathers != -1])
    rows_um = id_index.get_indexer(used_as_mother)
    rows_uf = id_index.get_indexer(used_as_father)
    if skip_mask is not None:
        keep_m = (rows_um != -1) & ~skip_mask[np.where(rows_um != -1, rows_um, 0)]
        keep_f = (rows_uf != -1) & ~skip_mask[np.where(rows_uf != -1, rows_uf, 0)]
    else:
        keep_m = rows_um != -1
        keep_f = rows_uf != -1
    findings = [
        Finding(check="sex_role_consistency", id=int(mid),
                detail=f"id={int(mid)} used as mother but sex != female")
        for mid in used_as_mother[keep_m & (sex[rows_um] != SEX_FEMALE)]
    ]
    findings.extend(
        Finding(check="sex_role_consistency", id=int(fid),
                detail=f"id={int(fid)} used as father but sex != male")
        for fid in used_as_father[keep_f & (sex[rows_uf] != SEX_MALE)]
    )
    return findings


@dataclass
class _SexImputation:
    """Result of imputing unknown sex from mother / father role usage."""

    imputed_sex: np.ndarray
    n_imputed: int
    ambiguous_mask: np.ndarray
    original_unknown_mask: np.ndarray
    mother_first_row: dict[int, int]
    father_first_row: dict[int, int]
    overridden_mask: np.ndarray
    sex_source: np.ndarray  # dtype=object; per-row category string


def _impute_sex_from_roles(
    sex: np.ndarray,
    ids: np.ndarray,
    mothers: np.ndarray,
    fathers: np.ndarray,
    *,
    override_asserted_sex: bool = True,
) -> _SexImputation:
    """Impute sex from mother / father role usage.

    Two passes over each row:

      1. Missing → role:
         - id used only as mother → impute F
         - id used only as father → impute M
         - id used as both        → leave -1, mark ambiguous (caller decides
                                    whether to raise / tolerate via
                                    ``--allow-missing-sex``)
         - id used as neither     → leave -1, orphan (caller decides whether
                                    to raise / tolerate via
                                    ``--allow-missing-sex``)

      2. Asserted → role (0.9 default; opt out via
         ``override_asserted_sex=False``):
         - asserted M used only as mother → override to F
         - asserted F used only as father → override to M
         - asserted used as both          → leave alone; ``_check_sex_role_
                                            consistency`` still hard-blocks
         - asserted used as neither       → leave alone

    Returns a `_SexImputation` whose `sex_source` is a per-row `dtype=object`
    array tagging each row's provenance: ``"input"`` / ``"imputed_from_missing"``
    / ``"imputed_from_role"`` / ``"unresolved"``.
    """
    original_unknown_mask = (sex == SEX_UNKNOWN).copy()
    imputed = sex.copy()
    sex_source = np.full(len(sex), "input", dtype=object)

    # first_row dicts are only consumed to render row-X / row-Y detail in
    # sex_role_ambiguity findings.
    m_idx = np.where(mothers != -1)[0]
    f_idx = np.where(fathers != -1)[0]
    unique_mids, first_m = np.unique(mothers[m_idx], return_index=True)
    unique_fids, first_f = np.unique(fathers[f_idx], return_index=True)
    mother_first_row = {int(mid): int(m_idx[i]) for mid, i in zip(unique_mids, first_m, strict=True)}
    father_first_row = {int(fid): int(f_idx[i]) for fid, i in zip(unique_fids, first_f, strict=True)}

    as_mother = np.isin(ids, unique_mids)
    as_father = np.isin(ids, unique_fids)

    # Pass 1: missing → role.
    impute_f = original_unknown_mask & as_mother & ~as_father
    impute_m = original_unknown_mask & as_father & ~as_mother
    ambiguous_mask = original_unknown_mask & as_mother & as_father
    imputed[impute_f] = SEX_FEMALE
    imputed[impute_m] = SEX_MALE
    imputed_from_missing = impute_f | impute_m
    sex_source[imputed_from_missing] = "imputed_from_missing"
    n_imputed = int(imputed_from_missing.sum())

    # Pass 2: asserted → role (when topology is unambiguous).
    override_to_f = (sex == SEX_MALE) & as_mother & ~as_father
    override_to_m = (sex == SEX_FEMALE) & as_father & ~as_mother
    if override_asserted_sex:
        imputed[override_to_f] = SEX_FEMALE
        imputed[override_to_m] = SEX_MALE
        overridden_mask = override_to_f | override_to_m
        sex_source[overridden_mask] = "imputed_from_role"
    else:
        overridden_mask = np.zeros(len(sex), dtype=bool)

    # Final pass: anything still SEX_UNKNOWN is unresolved.
    sex_source[imputed == SEX_UNKNOWN] = "unresolved"

    return _SexImputation(
        imputed_sex=imputed,
        n_imputed=n_imputed,
        ambiguous_mask=ambiguous_mask,
        original_unknown_mask=original_unknown_mask,
        mother_first_row=mother_first_row,
        father_first_row=father_first_row,
        overridden_mask=overridden_mask,
        sex_source=sex_source,
    )


def _check_sex_role_ambiguity(
    ids: np.ndarray,
    ambiguous_mask: np.ndarray,
    mother_first_row: dict[int, int],
    father_first_row: dict[int, int],
) -> list[Finding]:
    """Detect unsexed rows that are referenced as BOTH mother and father.

    A hard block by default — this is a data contradiction, not just missing
    information. Pedsum 0.8 added ``--allow-missing-sex`` which downgrades
    this to a SKIP with a tolerated count; the fixed validate-output writes
    these rows' sex as ``-1`` so the on-disk pedigree is self-consistent.
    """
    findings: list[Finding] = []
    for row_idx in np.where(ambiguous_mask)[0]:
        rid = int(ids[row_idx])
        mrow = mother_first_row.get(rid, -1)
        frow = father_first_row.get(rid, -1)
        findings.append(Finding(
            check="sex_role_ambiguity",
            id=rid,
            row=int(row_idx),
            detail=(
                f"id={rid} has unknown sex AND is referenced as both "
                f"mother (row {mrow}) and father (row {frow}); "
                "sex cannot be imputed"
            ),
        ))
    return findings


def _check_unknown_sex(
    ids: np.ndarray,
    sex: np.ndarray,
    ambiguous_mask: np.ndarray,
) -> list[Finding]:
    """Detect rows that still carry SEX_UNKNOWN after imputation.

    Excludes rows already reported by ``sex_role_ambiguity``.
    """
    orphan_mask = (sex == SEX_UNKNOWN) & ~ambiguous_mask
    return [
        Finding(
            check="unknown_sex",
            id=int(ids[i]),
            row=int(i),
            detail=(
                f"id={int(ids[i])} has unknown sex and is not referenced as a "
                "parent; cannot be imputed"
            ),
        )
        for i in np.where(orphan_mask)[0]
    ]


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


def _compute_depth_unordered(
    mother_rows: np.ndarray, father_rows: np.ndarray, n: int,
) -> np.ndarray:
    """Per-row topological depth, tolerant of any input row order.

    Vectorized fixed-point sweep: founders have depth 0, other rows have
    ``max(parent_depth) + 1``.  Iterates until depths stabilise — unlike
    a single Kahn pass, this does not require parents to already precede
    children in row index.  Returns ``np.int32``.  Raises ``PedigreeError``
    when no progress can be made (true cycle).
    """
    depth = np.full(n, -1, dtype=np.int32)
    depth[(mother_rows < 0) & (father_rows < 0)] = 0
    while True:
        todo = depth < 0
        if not todo.any():
            return depth
        m_safe = np.maximum(mother_rows, 0)
        f_safe = np.maximum(father_rows, 0)
        m_resolved = (mother_rows < 0) | (depth[m_safe] >= 0)
        f_resolved = (father_rows < 0) | (depth[f_safe] >= 0)
        eligible = todo & m_resolved & f_resolved
        if not eligible.any():
            unresolved = np.where(todo)[0][:5].tolist()
            raise PedigreeError(
                f"pedigree contains a cycle: {int(todo.sum())} individual(s) "
                f"could not be topologically ordered (e.g. rows {unresolved})",
            )
        md = np.where(mother_rows >= 0, depth[m_safe], 0)
        fd = np.where(father_rows >= 0, depth[f_safe], 0)
        depth[eligible] = np.maximum(md[eligible], fd[eligible]) + 1


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

_SEX_MISSING_TOKENS: frozenset[str] = _PARENT_MISSING_TOKENS | frozenset({
    "-1", "U", "UNKNOWN",
})


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
    series: pd.Series, missing_tokens: frozenset[str], sentinel: str,
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


def _open_text_for_sniff(path: Path):
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
    if first.count(best) > 0:
        return best
    if len(first.split()) >= 2:
        return r"\s+"
    return "\t"


def _read_pedigree_table(
    path: Path, sep: str = "auto", *, dtype: object | None = None,
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
    return pd.read_csv(path, sep=chosen, dtype=dtype, engine=engine)


def _as_parent_int_col(
    series: pd.Series, name: str, zero_as_missing: bool = False,
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
            f"column {name!r} must be integer-valued (with -1, NA, blank, "
            f"or empty for unknown); failed to parse: {e}"
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


def _check_birth_year_range(
    ids: np.ndarray,
    birth_year: np.ndarray,
    year_min: int,
    year_max: int,
) -> list[Finding]:
    """Detect known birth years outside ``[year_min, year_max]``.

    Sentinel ``-1`` (unknown) is skipped. One Finding per offending row.
    """
    known = birth_year != -1
    bad = known & ((birth_year < year_min) | (birth_year > year_max))
    return [
        Finding(
            check="birth_year_range",
            id=int(ids[i]),
            row=int(i),
            detail=(
                f"id={int(ids[i])} birth_year={int(birth_year[i])} is outside "
                f"the sanity range [{year_min}, {year_max}]"
            ),
        )
        for i in np.where(bad)[0]
    ]


def _check_birth_year_topology(
    ids: np.ndarray,
    mothers: np.ndarray,
    fathers: np.ndarray,
    birth_year: np.ndarray,
    id_index: pd.Index,
) -> list[Finding]:
    """Detect parent-child edges with ``child.birth_year < parent.birth_year``.

    Edges where either endpoint has ``birth_year == -1`` are skipped. One
    Finding per offending edge.
    """
    findings: list[Finding] = []
    for role, parents in (("mother", mothers), ("father", fathers)):
        parent_rows = id_index.get_indexer(parents)
        edges = (parents != -1) & (parent_rows != -1)
        if not edges.any():
            continue
        child_rows = np.where(edges)[0]
        prows = parent_rows[child_rows]
        child_by = birth_year[child_rows]
        parent_by = birth_year[prows]
        both_known = (child_by != -1) & (parent_by != -1)
        bad = both_known & (child_by < parent_by)
        for i in np.where(bad)[0]:
            row_idx = int(child_rows[i])
            findings.append(Finding(
                check="birth_year_topology",
                id=int(ids[row_idx]),
                row=row_idx,
                detail=(
                    f"id={int(ids[row_idx])} birth_year={int(child_by[i])} < "
                    f"{role} (id={int(parents[row_idx])}) "
                    f"birth_year={int(parent_by[i])}"
                ),
            ))
    return findings


def load_and_validate(
    path: Path,
    id_col: str = "id",
    sex_col: str = "sex",
    mother_col: str = "mother",
    father_col: str = "father",
    sex_encoding: str = "auto",
    zero_as_missing: bool = False,
    allow_missing_sex: bool = False,
    override_asserted_sex: bool = True,
    birth_year_col: str | None = None,
    birth_year_min: int = _BIRTH_YEAR_DEFAULT_MIN,
    birth_year_max: int | None = None,
    sep: str = "auto",
) -> tuple[pd.DataFrame, sp.csr_matrix | None]:
    """Load TSV, run all QC, return (df, children_csr).

    df has columns id, sex (int8), mother, father, generation (int32),
    and ``birth_year`` (int32, sentinel -1) when ``birth_year_col`` is
    set. Missing parent encoded as -1. children_csr is the parent→child
    sparse matrix (None if there are no parent edges) and is shared
    across the generations / components / descendants / inbreeding
    passes.
    """
    t0 = time.perf_counter()
    df = _read_pedigree_table(path, sep=sep, dtype=str)
    logger.info("read %d rows from %s in %.2fs", len(df), path, time.perf_counter() - t0)
    _maybe_warn_csv(df)

    needed = {id_col, sex_col, mother_col, father_col}
    if birth_year_col is not None:
        needed.add(birth_year_col)
    miss_cols = needed - set(df.columns)
    if miss_cols:
        raise PedigreeError(f"missing required columns: {sorted(miss_cols)}; file has {list(df.columns)}")

    def _raise_first(findings: list[Finding]) -> None:
        if findings:
            raise PedigreeError(_summarize_findings(findings))

    _raise_first(_check_empty_pedigree(len(df)))

    ids = _as_int_col(df[id_col], id_col)
    logger.info(
        "parsed %d ids; five random ids: %s", len(ids), _format_id_sample(ids),
    )
    mothers = _as_parent_int_col(df[mother_col], mother_col, zero_as_missing)
    fathers = _as_parent_int_col(df[father_col], father_col, zero_as_missing)
    sex = _decode_sex(df[sex_col], encoding=sex_encoding, zero_as_missing=zero_as_missing)
    birth_year = (
        _as_birth_year_col(df[birth_year_col], birth_year_col)
        if birth_year_col is not None
        else None
    )

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
    ):
        if findings:
            raise PedigreeError(_summarize_findings(findings))

    # Surface as PedigreeError here so summarize exits cleanly instead of
    # crashing later inside PedigreeGraph.from_arrays with a ValueError.
    if birth_year is not None:
        by_max = birth_year_max if birth_year_max is not None else _birth_year_default_max()
        _raise_first(_check_birth_year_range(ids, birth_year, birth_year_min, by_max))
        _raise_first(_check_birth_year_topology(ids, mothers, fathers, birth_year, id_index))

    imp = _impute_sex_from_roles(
        sex, ids, mothers, fathers,
        override_asserted_sex=override_asserted_sex,
    )
    sex = imp.imputed_sex
    n_overridden = int(imp.overridden_mask.sum())
    n_unresolved = int((sex == SEX_UNKNOWN).sum())
    if imp.n_imputed > 0 or n_overridden > 0 or n_unresolved > 0:
        logger.info(
            "sex imputation: %d from missing, %d overridden from role, "
            "%d unresolved (sex_source column has per-row detail)",
            imp.n_imputed, n_overridden, n_unresolved,
        )
    if not allow_missing_sex:
        _raise_first(_check_sex_role_ambiguity(
            ids, imp.ambiguous_mask, imp.mother_first_row, imp.father_first_row,
        ))
        _raise_first(_check_unknown_sex(ids, sex, imp.ambiguous_mask))
    else:
        n_ambiguous = int(imp.ambiguous_mask.sum())
        if n_ambiguous > 0:
            logger.info(
                "kept %d row(s) with sex_role_ambiguity (--allow-missing-sex)",
                n_ambiguous,
            )
        n_orphan = int(((sex == SEX_UNKNOWN) & ~imp.ambiguous_mask).sum())
        if n_orphan > 0:
            logger.info(
                "kept %d row(s) with sex=%d (--allow-missing-sex)",
                n_orphan, SEX_UNKNOWN,
            )

    role_consistency = _check_sex_role_consistency(
        mothers, fathers, sex, id_index, skip_mask=imp.original_unknown_mask,
    )
    if role_consistency:
        raise PedigreeError(_summarize_findings(role_consistency))

    # Map parent IDs to row indices, then run a fixed-point depth sweep
    # that is tolerant of arbitrary input row order (real-world TSV /
    # PLINK fam files commonly aren't topologically ordered).  Sort the
    # df by depth so PedigreeGraph's parents-precede-children invariant
    # holds for downstream construction.  Detects cycles by failure to
    # converge.
    m_row, mask_m = _parent_rows(mothers, id_index)
    f_row, mask_f = _parent_rows(fathers, id_index)
    depth = _compute_depth_unordered(m_row, f_row, n)
    natural = np.arange(n)
    order = np.argsort(depth, kind="stable")
    reordered = not np.array_equal(order, natural)

    if reordered:
        n_oo = int((order != natural).sum())
        logger.info(
            "reordering %d/%d row(s) into topological order (parents before children)",
            n_oo, n,
        )
        out = pd.DataFrame(
            {"id": ids[order], "sex": sex[order],
             "mother": mothers[order], "father": fathers[order],
             "sex_source": imp.sex_source[order]},
        )
        if birth_year is not None:
            out["birth_year"] = birth_year[order]
        id_index = pd.Index(out["id"].to_numpy())
        m_row, mask_m = _parent_rows(out["mother"].to_numpy(), id_index)
        f_row, mask_f = _parent_rows(out["father"].to_numpy(), id_index)
    else:
        out = pd.DataFrame(
            {"id": ids, "sex": sex, "mother": mothers, "father": fathers,
             "sex_source": imp.sex_source},
        )
        if birth_year is not None:
            out["birth_year"] = birth_year

    children_csr = _build_children_csr(m_row, mask_m, f_row, mask_f, n)
    # Note: ``ped_depth`` is populated by the caller (``_run_summarize``)
    # immediately after ``PedigreeGraph`` construction, using
    # ``pg.generation``.  Every downstream summary reads ``ped_depth``,
    # so callers building a df from ``load_and_validate`` must set the
    # column before invoking any summary function.

    logger.info("validated %d rows in %.2fs", n, time.perf_counter() - t0)
    return out, children_csr


# ---------------------------------------------------------------------------
# Validate (accumulating mode)
# ---------------------------------------------------------------------------


def validate_pedigree(
    path: Path,
    id_col: str = "id",
    sex_col: str = "sex",
    mother_col: str = "mother",
    father_col: str = "father",
    sex_encoding: str = "auto",
    zero_as_missing: bool = False,
    allow_missing_sex: bool = False,
    override_asserted_sex: bool = True,
    birth_year_col: str | None = None,
    birth_year_min: int = _BIRTH_YEAR_DEFAULT_MIN,
    birth_year_max: int | None = None,
    sep: str = "auto",
) -> tuple[int, list[CheckResult], list[Finding], dict]:
    """Run every integrity check accumulating.

    Returns ``(n_rows, results, findings, ctx)`` where ``results`` covers all
    checks in ``_CHECK_ORDER`` (each PASS / FAIL / SKIP), ``findings`` lists
    every per-individual finding, and ``ctx`` carries the coerced arrays and
    raw DataFrame for downstream auto-fix.
    """
    df = _read_pedigree_table(path, sep=sep, dtype=str)
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

    def _skip_many(names: tuple[str, ...], reason: str) -> None:
        for name in names:
            _skip(name, reason)

    needed = {id_col, sex_col, mother_col, father_col}
    miss_cols = needed - set(df.columns)
    if miss_cols:
        msg = f"missing required columns: {sorted(miss_cols)}; file has {list(df.columns)}"
        findings.append(Finding(check="required_columns", detail=msg))
        results["required_columns"] = CheckResult(name="required_columns", status="FAIL", count=1)
        for name, result in results.items():
            if name != "required_columns" and result.status == "SKIP":
                result.skip_reason = "required_columns failed"
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
    if ids is not None:
        logger.info(
            "parsed %d ids; five random ids: %s", len(ids), _format_id_sample(ids),
        )
    mothers = _coerce_int(mother_col, "mother_dtype", _parent_parser)
    fathers = _coerce_int(father_col, "father_dtype", _parent_parser)
    sex: np.ndarray | None = None
    try:
        sex = _decode_sex(df[sex_col], encoding=sex_encoding, zero_as_missing=zero_as_missing)
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
                imp = _impute_sex_from_roles(
                    sex, ids, mothers, fathers,
                    override_asserted_sex=override_asserted_sex,
                )
                imputed_sex = imp.imputed_sex
                n_ambiguous = int(imp.ambiguous_mask.sum())
                if allow_missing_sex:
                    _skip(
                        "sex_role_ambiguity",
                        f"bypassed via --allow-missing-sex ({n_ambiguous} tolerated)",
                    )
                else:
                    _record("sex_role_ambiguity", _check_sex_role_ambiguity(
                        ids, imp.ambiguous_mask, imp.mother_first_row, imp.father_first_row,
                    ))
                if override_asserted_sex:
                    # The override has already cleared assertions that contradict
                    # role-implied sex; the consistency check would find zero
                    # findings. Surface the override count directly so users see
                    # "PASS (N overridden from role)" in the grouped summary.
                    n_overridden = int(imp.overridden_mask.sum())
                    results["sex_role_consistency"] = CheckResult(
                        name="sex_role_consistency", status="PASS",
                        count=n_overridden,
                        skip_reason=(
                            f"{n_overridden} overridden from role"
                            if n_overridden > 0 else None
                        ),
                    )
                else:
                    _record("sex_role_consistency", _check_sex_role_consistency(
                        mothers, fathers, imputed_sex, id_index,
                        skip_mask=imp.original_unknown_mask,
                    ))
                n_orphan = int(((imputed_sex == SEX_UNKNOWN) & ~imp.ambiguous_mask).sum())
                if allow_missing_sex:
                    _skip(
                        "unknown_sex",
                        f"bypassed via --allow-missing-sex ({n_orphan} tolerated)",
                    )
                else:
                    _record("unknown_sex", _check_unknown_sex(
                        ids, imputed_sex, imp.ambiguous_mask,
                    ))
                ctx["sex_imputation"] = {
                    "imputed_sex": imputed_sex,
                    "n_imputed": imp.n_imputed,
                    "ambiguous_mask": imp.ambiguous_mask,
                    "original_unknown_mask": imp.original_unknown_mask,
                    "sex_source": imp.sex_source,
                    "overridden_mask": imp.overridden_mask,
                }
            else:
                _skip("sex_role_ambiguity", "sex_tokens failed")
                _skip("sex_role_consistency", "sex_tokens failed")
                _skip("unknown_sex", "sex_tokens failed")
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
            _skip("sex_role_ambiguity", "mother_dtype or father_dtype failed")
            _skip("self_loops", "mother_dtype or father_dtype failed")
            _skip("parents_distinct", "mother_dtype or father_dtype failed")
            _skip("sex_role_consistency", "mother_dtype or father_dtype failed")
            _skip("unknown_sex", "mother_dtype or father_dtype failed")
            _skip("acyclic", "mother_dtype or father_dtype failed")
    else:
        _skip_many(
            (
                "parent_refs_present_mother", "parent_refs_present_father",
                "parent_refs_sex_conflict", "sex_role_ambiguity",
                "self_loops", "parents_distinct",
                "sex_role_consistency", "unknown_sex", "acyclic",
            ),
            "id_dtype/negative_ids/duplicate_ids failed",
        )

    birth_year_checks = ("birth_year_dtype", "birth_year_range", "birth_year_topology")
    if birth_year_col is None:
        _skip_many(birth_year_checks, "no --birth-year-col specified")
    elif birth_year_col not in df.columns:
        _skip_many(birth_year_checks, f"column {birth_year_col!r} not present in file")
    else:
        by_max = birth_year_max if birth_year_max is not None else _birth_year_default_max()
        try:
            birth_year = _as_birth_year_col(df[birth_year_col], birth_year_col)
            results["birth_year_dtype"] = CheckResult(name="birth_year_dtype", status="PASS")
        except PedigreeError as e:
            findings.append(Finding(check="birth_year_dtype", detail=str(e)))
            results["birth_year_dtype"] = CheckResult(
                name="birth_year_dtype", status="FAIL", count=1,
            )
            birth_year = None
        if birth_year is not None and ids is not None:
            _record("birth_year_range", _check_birth_year_range(
                ids, birth_year, birth_year_min, by_max,
            ))
            if mothers is not None and fathers is not None and can_index:
                # id_index is already built above when can_index is True.
                _record("birth_year_topology", _check_birth_year_topology(
                    ids, mothers, fathers, birth_year, id_index,
                ))
            else:
                _skip(
                    "birth_year_topology",
                    "parent columns or ID index unavailable",
                )
        else:
            _skip("birth_year_range", "birth_year_dtype failed")
            _skip("birth_year_topology", "birth_year_dtype failed")

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
    has_both_parents = has_mom & has_dad
    n_founders = int((~has_parent).sum())
    n_nonfounders = int(has_parent.sum())
    n_mother_links = int(has_mom.sum())
    n_father_links = int(has_dad.sum())

    sex = df["sex"].to_numpy()
    n_male = int((sex == SEX_MALE).sum())
    n_female = int((sex == SEX_FEMALE).sum())
    n_unknown = int((sex == SEX_UNKNOWN).sum())

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
        "founder_frac": n_founders / n if n else 0.0,
        "n_nonfounders": n_nonfounders,
        "nonfounder_frac": n_nonfounders / n if n else 0.0,
        "n_male": n_male,
        "n_female": n_female,
        "n_unknown_sex": n_unknown,
        "n_mother_links": n_mother_links,
        "n_father_links": n_father_links,
        "n_parent_child_edges": n_mother_links + n_father_links,
        "n_with_both_parents": int(has_both_parents.sum()),
        "n_with_mother_only": int((has_mom & ~has_dad).sum()),
        "n_with_father_only": int((~has_mom & has_dad).sum()),
        "n_half_founders": int((has_mom ^ has_dad).sum()),
        "max_depth": max_depth,
        "mean_depth": float(gen.mean()) if n else 0.0,
        "median_depth": float(np.median(gen)) if n else 0.0,
        "depth_counts": gen_counts,
        "n_components": int(n_components),
        "largest_component": int(sorted_sizes[0]) if len(sorted_sizes) else 0,
        "largest_component_frac": (int(sorted_sizes[0]) / n) if len(sorted_sizes) and n else 0.0,
        "next_components": sorted_sizes[1:6].tolist(),
    }
    return summary, comp_labels


def _numeric_distribution(values: pd.Series | np.ndarray) -> dict:
    """Compact distribution summary for pedigree-level aggregate sections."""
    c = pd.Series(values)
    n = len(c)
    is_float = pd.api.types.is_float_dtype(c)
    cast = float if is_float else int
    return {
        "mean": float(c.mean()) if n else 0.0,
        "std": float(c.std()) if n > 1 else 0.0,
        "min": cast(c.min()) if n else 0,
        "q1": cast(c.quantile(0.25)) if n else 0,
        "median": cast(c.median()) if n else 0,
        "q3": cast(c.quantile(0.75)) if n else 0,
        "max": cast(c.max()) if n else 0,
        "nz": int((c != 0).sum()) if n else 0,
    }


def _effective_count_from_weights(weights: np.ndarray) -> float:
    """Return 1 / sum(p_i^2), or 0 when no positive weights are present."""
    positive = weights[weights > 0].astype(np.float64)
    total = float(positive.sum())
    if total == 0.0:
        return 0.0
    p = positive / total
    return float(1.0 / np.sum(p * p))


def compute_mating_pair_summary(df: pd.DataFrame) -> dict | None:
    """Aggregate per-Mating-Pair statistics: count, children-per-pair, effective pairs.

    Per-individual mate counts live in :func:`compute_aggregate_sections` under
    ``reproduction:`` (over **all** individuals, zero-included). This section is
    reserved for per-Mating-Pair quantities only.
    """
    both_present = (df["mother"] != -1) & (df["father"] != -1)
    children = df.loc[both_present]
    if len(children) == 0:
        return None

    pair_sizes = children.groupby(["mother", "father"]).size()

    return {
        "n_pairs": len(pair_sizes),
        "n_pairs_with_multiple_children": int((pair_sizes >= 2).sum()),
        "frac_pairs_with_multiple_children": float((pair_sizes >= 2).sum()) / len(pair_sizes),
        "children_per_pair": _numeric_distribution(pair_sizes.to_numpy()),
        "effective_pairs_by_children": _effective_count_from_weights(pair_sizes.to_numpy()),
    }


def _offspring_dist(counts: np.ndarray, n: int) -> dict:
    if n == 0:
        return dict.fromkeys(("0", "1", "2", "3", "4+"), 0.0)
    out = {"0": float((counts == 0).sum()) / n}
    for k in (1, 2, 3):
        out[str(k)] = float((counts == k).sum()) / n
    out["4+"] = float((counts >= 4).sum()) / n
    return out


def compute_founder_summary(
    idf: pd.DataFrame,
    max_lineage_cells: int = 5_000_000,
) -> tuple[dict, np.ndarray]:
    """Founder-contribution-by-depth using unique **Founder Ancestor** sets.

    Returns a ``(summary, n_founder_ancestors)`` tuple. The second element
    is the per-individual count of distinct **Founder Ancestors** (zero
    for **Founders** themselves; zeros when the section is skipped).

    Bounded: carrying founder sets per row can become large on very large
    pedigrees, so the section reports ``computed: false`` instead of
    risking a memory blow-up.
    """
    n = len(idf)
    founders = idf["is_founder"].to_numpy(dtype=bool)
    founder_rows = np.where(founders)[0]
    n_founders = len(founder_rows)
    zeros = np.zeros(n, dtype=np.int32)
    if n == 0 or n_founders == 0:
        return {"computed": True, "by_depth": [], "bottleneck": None}, zeros
    if n * n_founders > max_lineage_cells:
        skip = {
            "computed": False,
            "skip_reason": (
                f"n_individuals * n_founders = {n * n_founders} exceeds "
                f"max_lineage_cells={max_lineage_cells}"
            ),
        }
        return skip, zeros

    row_to_founder = {int(row): i for i, row in enumerate(founder_rows)}
    id_index = pd.Index(idf["id"].to_numpy())
    mothers = idf["mother"].to_numpy()
    fathers = idf["father"].to_numpy()
    m_row, has_mom = _parent_rows(mothers, id_index)
    f_row, has_dad = _parent_rows(fathers, id_index)
    order = np.argsort(idf["ped_depth"].to_numpy(), kind="stable")

    founder_sets: list[set[int]] = [set() for _ in range(n)]
    for i in order:
        i_int = int(i)
        if founders[i_int]:
            founder_sets[i_int] = {row_to_founder[i_int]}
            continue
        s: set[int] = set()
        if has_mom[i_int]:
            s.update(founder_sets[int(m_row[i_int])])
        if has_dad[i_int]:
            s.update(founder_sets[int(f_row[i_int])])
        founder_sets[i_int] = s

    n_founder_ancestors = np.array(
        [len(fs) for fs in founder_sets], dtype=np.int32,
    )

    by_depth = []
    for depth, sub in idf.groupby("ped_depth", sort=True):
        rows = sub.index.to_numpy()
        active: set[int] = set()
        counts = np.zeros(n_founders, dtype=np.int64)
        line_counts = np.zeros(len(rows), dtype=np.int32)
        for pos, row in enumerate(rows):
            fs = founder_sets[int(row)]
            line_counts[pos] = len(fs)
            if not fs:
                continue
            active.update(fs)
            counts[list(fs)] += 1
        active_counts = counts[counts > 0]
        by_depth.append({
            "depth": int(depth),
            "n": len(rows),
            "active_founders": len(active),
            "active_founder_frac": len(active) / n_founders,
            "effective_founders_by_descendants": _effective_count_from_weights(active_counts),
            "founder_ancestors": _numeric_distribution(line_counts),
        })

    nonempty = [row for row in by_depth if row["n"] > 0]
    if nonempty:
        min_active = min(row["active_founders"] for row in nonempty)
        min_eff = min(row["effective_founders_by_descendants"] for row in nonempty)
        bottleneck = {
            "min_active_founders": int(min_active),
            "min_active_founder_frac": min_active / n_founders,
            "min_active_depths": [
                int(row["depth"]) for row in nonempty if row["active_founders"] == min_active
            ],
            "min_effective_founders_by_descendants": float(min_eff),
            "min_effective_depths": [
                int(row["depth"])
                for row in nonempty
                if row["effective_founders_by_descendants"] == min_eff
            ],
        }
    else:
        bottleneck = None

    summary = {"computed": True, "by_depth": by_depth, "bottleneck": bottleneck}
    return summary, n_founder_ancestors


def compute_aggregate_sections(
    idf: pd.DataFrame,
    founder_summary: dict,
    include_inbreeding: bool,
) -> dict:
    """Pedigree-level aggregate sections derived from the individual table.

    ``founder_summary`` is the dict returned by :func:`compute_founder_summary`;
    it is computed beforehand so that its per-individual
    ``n_founder_ancestors`` vector can be added to ``idf`` before this
    function is called.
    """
    n = len(idf)
    if n == 0:
        return {
            "reproduction": {},
            "genealogy": {},
            "founder_contribution": {},
            "founder_summary": founder_summary,
            "components": {},
            "sex_summary": {},
            "depth_summary": [],
        }

    reproductive = idf["n_offspring"] > 0
    no_children = ~reproductive
    founders = idf["is_founder"].astype(bool)
    descendant_path_counts = idf.loc[founders, "n_descendant_paths"].to_numpy()

    n_offspring_arr = idf["n_offspring"].to_numpy()
    n_mates_arr = idf["n_mates"].to_numpy()
    sex_arr = idf["sex"].to_numpy()
    male_mask = sex_arr == SEX_MALE
    female_mask = sex_arr == SEX_FEMALE
    n_male = int(male_mask.sum())
    n_female = int(female_mask.sum())

    # frac_with_full_sib: fraction of individuals WITH BOTH PARENTS present
    # who share their (mother, father) with at least one other individual.
    both_parents = (idf["mother"] != -1) & (idf["father"] != -1)
    n_both = int(both_parents.sum())
    if n_both:
        frac_with_full_sib = float(
            (idf.loc[both_parents, "n_full_sibs"] >= 1).sum()
        ) / n_both
    else:
        frac_with_full_sib = 0.0

    # Per-Individual reproductive output: offspring counts, mate counts,
    # reproductive/terminal classification. Distributions follow the
    # CONTEXT.md naming convention: <noun>_count for summary stats;
    # <noun>_count_hist for binned PMF; _male / _female stratify either.
    reproduction = {
        "n_reproductive": int(reproductive.sum()),
        "frac_reproductive": float(reproductive.sum()) / n,
        "n_terminal": int(no_children.sum()),
        "frac_terminal": float(no_children.sum()) / n,
        "frac_with_full_sib": frac_with_full_sib,
        "offspring_count": _numeric_distribution(n_offspring_arr),
        "offspring_count_hist": _offspring_dist(n_offspring_arr, n),
        "offspring_count_hist_male": _offspring_dist(n_offspring_arr[male_mask], n_male),
        "offspring_count_hist_female": _offspring_dist(n_offspring_arr[female_mask], n_female),
        # mate_count_* are over ALL males / ALL females (zero-included), a
        # behavior change from 0.9's `female_mate_count` / `male_mate_count`
        # which only summed over parents-with-children. See CHANGELOG 0.10.0.
        "mate_count": _numeric_distribution(n_mates_arr),
        "mate_count_male": _numeric_distribution(n_mates_arr[male_mask]) if n_male else None,
        "mate_count_female": _numeric_distribution(n_mates_arr[female_mask]) if n_female else None,
    }

    # Per-individual ancestry. Asymmetric semantics today (paths for
    # descendants, distinct for ancestors) — see CONTEXT.md.
    genealogy = {
        "descendant_paths": _numeric_distribution(idf["n_descendant_paths"]),
    }
    if include_inbreeding:
        genealogy["distinct_ancestors"] = _numeric_distribution(idf["n_distinct_ancestors"])
    else:
        genealogy["distinct_ancestors"] = None

    n_founders = int(founders.sum())
    founders_with_desc = int((descendant_path_counts > 0).sum()) if n_founders else 0
    founder_contribution = {
        "n_founders_with_descendants": founders_with_desc,
        "n_founders_without_descendants": n_founders - founders_with_desc,
        "frac_founders_with_descendants": (founders_with_desc / n_founders) if n_founders else 0.0,
        "descendant_paths_per_founder": _numeric_distribution(descendant_path_counts),
        "effective_founders_by_descendant_paths": _effective_count_from_weights(descendant_path_counts),
    }
    sizes = idf.groupby("component_id").size()
    component_dist = _numeric_distribution(sizes.to_numpy())
    singletons = int((sizes == 1).sum())
    components = {
        "singletons": singletons,
        "singletons_frac": singletons / int(sizes.size) if sizes.size else 0.0,
        "size_dist": {
            "1": float((sizes == 1).sum()) / len(sizes) if len(sizes) else 0.0,
            "2": float((sizes == 2).sum()) / len(sizes) if len(sizes) else 0.0,
            "3-9": float(((sizes >= 3) & (sizes <= 9)).sum()) / len(sizes) if len(sizes) else 0.0,
            "10-99": float(((sizes >= 10) & (sizes <= 99)).sum()) / len(sizes) if len(sizes) else 0.0,
            "100+": float((sizes >= 100).sum()) / len(sizes) if len(sizes) else 0.0,
        },
        "component_size": component_dist,
    }

    sex_summary = {}
    for label, code in (("female", SEX_FEMALE), ("male", SEX_MALE)):
        sub = idf.loc[idf["sex"] == code]
        if len(sub) == 0:
            continue
        sx_reproductive = sub["n_offspring"] > 0
        row = {
            "n": len(sub),
            "n_founders": int(sub["is_founder"].sum()),
            "n_reproductive": int(sx_reproductive.sum()),
            "frac_reproductive": float(sx_reproductive.sum()) / len(sub),
            "n_terminal": int((~sx_reproductive).sum()),
            "offspring_count": _numeric_distribution(sub["n_offspring"]),
            "mate_count": _numeric_distribution(sub["n_mates"]),
            "depth": _numeric_distribution(sub["ped_depth"]),
        }
        if include_inbreeding:
            row["F"] = _numeric_distribution(sub["F"])
            row["n_inbred"] = int((sub["F"] > INBRED_TOL).sum())
        sex_summary[label] = row

    depth_summary = []
    for depth, sub in idf.groupby("ped_depth", sort=True):
        d_reproductive = sub["n_offspring"] > 0
        row = {
            "depth": int(depth),
            "n": len(sub),
            "n_male": int((sub["sex"] == SEX_MALE).sum()),
            "n_female": int((sub["sex"] == SEX_FEMALE).sum()),
            "n_founders": int(sub["is_founder"].sum()),
            "n_reproductive": int(d_reproductive.sum()),
            "frac_reproductive": float(d_reproductive.sum()) / len(sub),
            "n_terminal": int((~d_reproductive).sum()),
            "offspring_count": _numeric_distribution(sub["n_offspring"]),
            "offspring_count_hist": _offspring_dist(sub["n_offspring"].to_numpy(), len(sub)),
            "mate_count": _numeric_distribution(sub["n_mates"]),
            "mean_distinct_ancestors": None,
            "mean_descendant_paths": float(sub["n_descendant_paths"].mean()),
        }
        if include_inbreeding:
            row["mean_distinct_ancestors"] = float(sub["n_distinct_ancestors"].mean())
            row["mean_F"] = float(sub["F"].mean())
            row["max_F"] = float(sub["F"].max())
            row["n_inbred"] = int((sub["F"] > INBRED_TOL).sum())
        depth_summary.append(row)

    return {
        "reproduction": reproduction,
        "genealogy": genealogy,
        "founder_contribution": founder_contribution,
        "founder_summary": founder_summary,
        "components": components,
        "sex_summary": sex_summary,
        "depth_summary": depth_summary,
    }


# ---------------------------------------------------------------------------
# Section 2: family sizes (ported from simace.analysis.stats.pedigree)
# ---------------------------------------------------------------------------


def compute_sibship_sizes(df: pd.DataFrame) -> dict:
    """Per-**Sibship** size statistics.

    A **Sibship** is the offspring set of one **Mating Pair**. This function
    only emits per-Sibship aggregates; per-individual offspring counts
    (binned and sex-stratified) live in :func:`compute_aggregate_sections`
    under ``reproduction:``.
    """
    both_present = (df["mother"] != -1) & (df["father"] != -1)
    children = df.loc[both_present]
    if len(children) == 0:
        return {"empty": True}

    sibship_sizes = children.groupby(["mother", "father"]).size()
    n_sib = len(sibship_sizes)
    size_dist = {str(k): float((sibship_sizes == k).sum()) / n_sib for k in (1, 2, 3)}
    size_dist["4+"] = float((sibship_sizes >= 4).sum()) / n_sib

    return {
        "empty": False,
        "n_sibships": int(n_sib),
        "mean": float(sibship_sizes.mean()),
        "median": float(sibship_sizes.median()),
        "q1": float(sibship_sizes.quantile(0.25)),
        "q3": float(sibship_sizes.quantile(0.75)),
        "size_dist": size_dist,
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
    include_pair_lists: bool = False,
    pg: PedigreeGraph | None = None,
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

    When ``pg`` is supplied the wrapper reuses it instead of building a
    fresh compacted PedigreeGraph; saves one compaction pass when the
    caller already needed a graph for other primitives (F, lineage,
    effective size).
    """
    chosen = _select_engine(len(df), engine, threshold)
    logger.info("relationship engine: %s (n=%d)", chosen, len(df))
    if chosen == "bfs":
        out = _count_pairs_bfs(df, pg=pg)
    else:
        out = (
            _count_pairs_matrix_with_lists(df, pg=pg)
            if include_pair_lists
            else _count_pairs_matrix(df, pg=pg)
        )
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


def _count_pairs_matrix(df: pd.DataFrame, pg: PedigreeGraph | None = None) -> dict:
    """Sparse matrix-power enumerator (default engine for n < threshold).

    Delegates relationship enumeration to ``pedigree_graph.PedigreeGraph``;
    when ``pg`` is None this wrapper compacts IDs to ``0..n-1`` first
    because ``PedigreeGraph`` allocates an ``id_to_row`` table sized to
    ``max(id)+1``.
    """
    if pg is None:
        pg = _build_pedigree_graph(df)
    named = pg.count_pairs(max_degree=5)
    return _augment_pair_counts(named)


def _count_pairs_matrix_with_lists(df: pd.DataFrame, pg: PedigreeGraph | None = None) -> dict:
    """Sparse matrix enumerator that retains pair lists for richer summaries."""
    if pg is None:
        pg = _build_pedigree_graph(df)
    pair_lists = pg.extract_pairs(max_degree=5)
    named = {code: len(a) for code, (a, _) in pair_lists.items()}
    out = _augment_pair_counts(named)
    out["_pair_lists"] = pair_lists
    return out


def _count_pairs_bfs(df: pd.DataFrame, pg: PedigreeGraph | None = None) -> dict:
    """BFS / boolean-matmul / numba enumerator (experimental).

    Thin wrapper around :func:`pedigree_graph.experimental.count_pairs_bfs`.
    See that function's docstring for the inbred-pedigree caveat
    (distinct-shared-ancestor counting vs path-multiplicity).
    """
    if pg is None:
        pg = _build_pedigree_graph(df)
    named = count_pairs_bfs(pg, max_degree=5)
    return _augment_pair_counts(named)


def compute_relationship_summary(
    df: pd.DataFrame,
    pair_lists: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> dict:
    """Density and per-individual relationship-burden summaries.

    Pair-list-derived metrics are exact for the matrix engine. The BFS engine
    currently returns aggregate counts only, so these fields are unavailable
    there instead of being approximated from non-unique relationship counts.
    """
    n = len(df)
    n_possible = n * (n - 1) // 2
    if pair_lists is None:
        return {
            "computed": False,
            "skip_reason": "relationship pair lists are only available from the matrix engine",
            "n_possible_pairs": int(n_possible),
        }
    if n == 0:
        return {
            "computed": True,
            "n_possible_pairs": 0,
            "n_related_pairs": 0,
            "n_unrelated_pairs": 0,
            "related_pair_density": 0.0,
            "related_pairs_by_closest_degree": {str(d): 0 for d in range(1, 6)},
            "closest_relationship_per_individual": {"none": 0, **{str(d): 0 for d in range(1, 6)}},
            "relatives_by_degree": {str(d): _numeric_distribution(np.array([], dtype=np.int64)) for d in range(1, 6)},
            "relatives_total": _numeric_distribution(np.array([], dtype=np.int64)),
            "related_pair_density_by_generation": [],
        }

    keys_parts = []
    degree_parts = []
    for code, (a_raw, b_raw) in pair_lists.items():
        if code not in REL_REGISTRY:
            continue
        a = np.asarray(a_raw, dtype=np.int64)
        b = np.asarray(b_raw, dtype=np.int64)
        if len(a) == 0:
            continue
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        keep = lo != hi
        if not keep.any():
            continue
        keys_parts.append(lo[keep] * n + hi[keep])
        degree_parts.append(
            np.full(int(keep.sum()), REL_REGISTRY[code].degree, dtype=np.int8),
        )

    if not keys_parts:
        closest_degree = np.zeros(n, dtype=np.int8)
        return {
            "computed": True,
            "n_possible_pairs": int(n_possible),
            "n_related_pairs": 0,
            "n_unrelated_pairs": int(n_possible),
            "related_pair_density": 0.0,
            "related_pairs_by_closest_degree": {str(d): 0 for d in range(1, 6)},
            "closest_relationship_per_individual": {
                "none": int((closest_degree == 0).sum()),
                **{str(d): 0 for d in range(1, 6)},
            },
            "relatives_by_degree": {
                str(d): _numeric_distribution(np.zeros(n, dtype=np.int64))
                for d in range(1, 6)
            },
            "relatives_total": _numeric_distribution(np.zeros(n, dtype=np.int64)),
            "related_pair_density_by_generation": [],
        }

    keys = np.concatenate(keys_parts)
    degrees = np.concatenate(degree_parts)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    degrees = degrees[order]
    starts = np.concatenate(([0], np.where(np.diff(keys) != 0)[0] + 1))
    unique_keys = keys[starts]
    min_degrees = np.minimum.reduceat(degrees, starts)

    lo = unique_keys // n
    hi = unique_keys % n
    n_related = len(unique_keys)

    counts_by_degree = {}
    total_relatives = np.zeros(n, dtype=np.int64)
    closest_degree = np.zeros(n, dtype=np.int8)
    for degree in range(1, 6):
        mask = min_degrees == degree
        degree_counts = (
            np.bincount(lo[mask], minlength=n)
            + np.bincount(hi[mask], minlength=n)
        ).astype(np.int64)
        counts_by_degree[str(degree)] = _numeric_distribution(degree_counts)
        total_relatives += degree_counts

    for degree in range(5, 0, -1):
        has_degree = (
            np.bincount(lo[min_degrees == degree], minlength=n)
            + np.bincount(hi[min_degrees == degree], minlength=n)
        ) > 0
        closest_degree[has_degree] = degree

    related_by_closest_degree = {
        str(degree): int((min_degrees == degree).sum()) for degree in range(1, 6)
    }
    closest_dist = {"none": int((closest_degree == 0).sum())}
    closest_dist.update({
        str(degree): int((closest_degree == degree).sum()) for degree in range(1, 6)
    })

    depth = df["ped_depth"].to_numpy()
    depth_rows = []
    for d in range(int(depth.max()) + 1 if n else 0):
        n_d = int((depth == d).sum())
        possible = n_d * (n_d - 1) // 2
        if possible:
            related = int(((depth[lo] == d) & (depth[hi] == d)).sum())
            density = related / possible
        else:
            related = 0
            density = 0.0
        depth_rows.append({
            "depth": int(d),
            "n": n_d,
            "n_individual_pairs": int(possible),
            "n_related_pairs": related,
            "n_unrelated_pairs": int(possible - related),
            "related_pair_density": float(density),
        })

    return {
        "computed": True,
        "max_degree": 5,
        "n_individual_pairs": int(n_possible),
        "n_related_pairs": n_related,
        "n_unrelated_pairs": int(n_possible - n_related),
        "related_pair_density": (n_related / n_possible) if n_possible else 0.0,
        "related_pairs_by_closest_degree": related_by_closest_degree,
        "closest_relationship_per_individual": closest_dist,
        "relatives_by_degree": counts_by_degree,
        "relatives_total": _numeric_distribution(total_relatives),
        "related_pair_density_by_depth": depth_rows,
    }


def _build_pedigree_graph(df: pd.DataFrame) -> PedigreeGraph:
    """Compact arbitrary IDs to ``0..n-1`` and build a full ``PedigreeGraph``.

    Threads ``sex`` through to ``PedigreeGraph.from_arrays`` so downstream
    sex-aware estimators (Ne_sr, the sex-decomposed Ne_V quadrants, sex-
    stratified relationship-pair extraction) receive correct sex data
    rather than the silent zeros that the bare-arrays construction path
    would supply. When the df carries a ``birth_year`` column (populated
    by ``load_and_validate`` under ``--birth-year-col``), the array is
    threaded through as well so the Hill overlapping-generation estimator
    can build its cohort window.

    Generation is derived inside ``from_arrays`` from the (already-remapped)
    parent arrays via a fixed-point sweep — same semantics as pedsum's
    historical Kahn pass.  ``twin`` defaults to ``-1`` because pedsum's
    input format does not carry twin annotations.

    The compaction is necessary because ``PedigreeGraph`` allocates an
    ``id_to_row`` table sized to ``max(id) + 1``; passing original IDs
    on a sparse pedigree would inflate memory by orders of magnitude.

    Assumes the input has already passed ``load_and_validate``, which
    sorts rows into topological order; ``PedigreeGraph`` requires
    parents to precede children in row order.
    """
    ids = df["id"].to_numpy()
    n = len(ids)
    new_ids = np.arange(n, dtype=np.int64)
    id_to_compact = pd.Series(new_ids, index=ids)

    def _remap(parents: np.ndarray) -> np.ndarray:
        return np.where(
            parents == -1, -1, id_to_compact.reindex(parents).to_numpy(),
        ).astype(np.int64)

    birth_year = (
        df["birth_year"].to_numpy().astype(np.int32)
        if "birth_year" in df.columns
        else None
    )
    return PedigreeGraph.from_arrays(
        ids=new_ids,
        mothers=_remap(df["mother"].to_numpy()),
        fathers=_remap(df["father"].to_numpy()),
        sex=df["sex"].to_numpy().astype(np.int8),
        birth_year=birth_year,
    )


# ---------------------------------------------------------------------------
# Section 4: inbreeding (summary helper)
# ---------------------------------------------------------------------------


def compute_effective_size(
    pg: PedigreeGraph,
    *,
    ne_coancestry: bool = False,
    n_threads: int = 1,
) -> dict:
    """Run the eight pedigree-based Ne estimators via ``pedigree_graph``.

    Thin wrapper around ``pedigree_graph.compute_all_ne``: builds the
    founder-contribution structures once, dispatches every estimator,
    and serialises each result dataclass to a YAML-ready dict via its
    own ``.to_dict()`` method.

    When ``ne_coancestry`` is False (the default), the coancestry-rate
    Ne_C estimator is skipped — its kinship DP can dominate memory on
    very large pedigrees.  The ``ne_coancestry`` slot in the returned
    dict will then carry ``ne=None`` and NaN per-gen arrays.
    """
    from pedigree_graph import compute_all_ne

    raw = compute_all_ne(
        pg,
        skip_ne_coancestry=not ne_coancestry,
        n_threads=n_threads,
    )
    return {
        name: _normalise_effective_size_keys(result.to_dict())
        for name, result in raw.items()
    }


def _normalise_effective_size_keys(d: dict) -> dict:
    """Rename upstream ``n_generations_used`` → ``n_depths_used`` on the way out.

    pedigree-graph emits ``n_generations_used`` in ``ne_inbreeding`` and
    ``ne_caballero_toro``; pedsum's CONTEXT.md treats "generation" as a
    temporal/birth-cohort term, distinct from topological **Depth**.
    This shim renames the key in pedsum's output without touching the
    upstream package. Delete this helper when pedigree-graph itself
    adopts the depth-based name.
    """
    if "n_generations_used" in d:
        d = {("n_depths_used" if k == "n_generations_used" else k): v
             for k, v in d.items()}
    return d


def _build_inbreeding_summary(F: np.ndarray) -> dict:
    """Aggregate a per-individual F vector into the YAML-shaped summary.

    F itself is computed by ``pedigree_graph.PedigreeGraph.compute_inbreeding()``
    (Meuwissen-Luo); pedsum no longer owns an F implementation.  This
    helper produces only the histogram / aggregate fields previously
    returned by the deleted ``compute_inbreeding`` function, and drops
    the ``memo_size`` diagnostic (which described the deleted algorithm
    and has no analogue in the upstream implementation).
    """
    n = len(F)
    inbred = F > INBRED_TOL
    n_inbred = int(inbred.sum())
    edges = [0.0, 0.0625, 0.125, 0.25, 1.0]
    hist: dict[str, float] = {}
    hist["0"] = float((F <= INBRED_TOL).sum()) / n if n else 0.0
    for lo, hi in pairwise(edges):
        label = f"<{hi:g}"
        hist[label] = float(((lo < F) & (hi >= F)).sum()) / n if n else 0.0
    return {
        "n_inbred": n_inbred,
        "frac_inbred": n_inbred / n if n else 0.0,
        "mean_F": float(F.mean()) if n else 0.0,
        "max_F": float(F.max()) if n else 0.0,
        "hist": hist,
    }


# ---------------------------------------------------------------------------
# Section 5: per-individual data
# ---------------------------------------------------------------------------


def build_individual_df(
    df: pd.DataFrame,
    id_index: pd.Index,
    F: np.ndarray,
    n_distinct_ancestors: np.ndarray,
    n_descendant_paths: np.ndarray,
    component_labels: np.ndarray,
    sex_source: np.ndarray,
) -> pd.DataFrame:
    """Assemble per-individual table with the maximal column set.

    ``n_founder_ancestors`` is added by the caller after
    :func:`compute_founder_summary` runs against this table.
    """
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
            "sex_source": sex_source,
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
            "n_distinct_ancestors": n_distinct_ancestors,
            "n_descendant_paths": n_descendant_paths.astype(np.int32),
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
    "n_founder_ancestors",
    "n_distinct_ancestors",
    "n_descendant_paths",
)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Categorised summary YAML schema (Axis 1, one nesting level)
# ---------------------------------------------------------------------------
# Two output files split a single pre-categorised pedigree dict:
#   <base>.summary.yaml         — slim, headline content
#   <base>.summary.extra.yaml   — residue (omitted keys / arrays), see split contract
# The flat dict that feeds the TSV writers is untouched — TSV output is
# YAML-reorg-invariant. Drops, renames, and category structure live ONLY
# here in this schema; section builders stay split-unaware.


@dataclass(frozen=True)
class SectionSpec:
    """One leaf section under a category."""
    name: str
    slim_keys: tuple[str, ...] | None = None
    list_of_dict_slim_keys: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CategorySpec:
    """One category bucket grouping related sections."""
    name: str
    sections: tuple[SectionSpec, ...]


# 23 named relationship codes from REL_REGISTRY plus PO and engine.
_PAIRS_SLIM_KEYS: tuple[str, ...] = (
    "MZ", "MO", "FO", "FS", "MHS", "PHS", "GP", "Av", "GGP", "HAv", "GAv",
    "1C", "GGGP", "HGAv", "GGAv", "H1C", "1C1R", "G3GP", "HGGAv", "G3Av",
    "H1C1R", "1C2R", "2C", "PO", "engine",
)


SUMMARY_SCHEMA: tuple[CategorySpec, ...] = (
    CategorySpec("structure", (
        SectionSpec("size_structure"),
        SectionSpec("components"),
        SectionSpec("max_degree_enumerated"),
    )),
    CategorySpec("demography", (
        SectionSpec("sibship_size"),
        SectionSpec("mating_pairs"),
    )),
    CategorySpec("individuals", (
        SectionSpec("reproduction"),
        SectionSpec("genealogy"),
    )),
    CategorySpec("founders", (
        SectionSpec("founder_contribution"),
        SectionSpec("founder_summary"),
    )),
    CategorySpec("relatedness", (
        SectionSpec("relationship_pairs", slim_keys=_PAIRS_SLIM_KEYS),
        SectionSpec("relationship_summary"),
        SectionSpec("inbreeding"),
    )),
    CategorySpec("popgen", (
        SectionSpec("effective_size"),  # per-estimator special handling
    )),
    CategorySpec("strata", (
        SectionSpec("sex_summary"),     # per-stratum special handling
        SectionSpec("depth_summary", list_of_dict_slim_keys=("depth", "n")),
    )),
)


# Per-stratum slim keys for sex_summary (the scalar integers only).
_SEX_SUMMARY_SLIM_KEYS: tuple[str, ...] = (
    "n", "n_founders", "n_reproductive", "n_inbred",
)

# Per-individual distribution split: only headline columns in slim, and within
# each column only mean+median; extra carries all columns × all quantile keys.
INDIVIDUAL_SLIM_COLS: tuple[str, ...] = ("n_offspring", "n_mates", "F", "n_distinct_ancestors")
INDIVIDUAL_SLIM_DIST_KEYS: tuple[str, ...] = ("mean", "median")

# YAML-only drops (paths within the categorised dict). Constructed in the
# flat dict and visible in the TSV; excluded from both slim and extra YAML.
KNOWN_YAML_DROPS: frozenset[str] = frozenset({
    "pedigree.relatedness.relationship_pairs.by_degree",
    "individual.distributions.F.max",
})

# Schema renames from flat → categorised path. Used by the totality test so
# renamed fields aren't flagged as missing.
RENAMES: dict[str, str] = {
    "pedigree.pairs_engine": "pedigree.relatedness.relationship_pairs.engine",
}


def _categorise_pedigree(flat_ped: dict) -> dict:
    """Wrap the flat pedigree dict into the category structure.

    Pure function. Reads ``SUMMARY_SCHEMA`` to bucket sections by
    category, drops empty categories, and folds the sibling
    ``pairs_engine`` into ``relationship_pairs.engine`` so it lives
    inside the relatedness.relationship_pairs subtree (matches
    ``RENAMES``).

    Does NOT apply slim/extra splitting — that is ``_split_summary``'s
    job. Does NOT apply ``KNOWN_YAML_DROPS`` — splitter handles them.
    """
    sections = dict(flat_ped)
    pairs = sections.get("relationship_pairs")
    pairs_engine = sections.pop("pairs_engine", None)
    if pairs is not None and pairs_engine is not None:
        # Non-destructive: build a new dict so the caller's flat payload
        # (which the TSV writer also reads) is not mutated.
        sections["relationship_pairs"] = {**pairs, "engine": pairs_engine}

    nested: dict = {}
    for cat in SUMMARY_SCHEMA:
        cat_dict: dict = {}
        for sec in cat.sections:
            value = sections.get(sec.name)
            if value is None:
                continue
            if isinstance(value, dict) and not value:
                continue
            if isinstance(value, list) and not value:
                continue
            cat_dict[sec.name] = value
        if cat_dict:
            nested[cat.name] = cat_dict
    return nested


# Per-estimator routing for effective_size. Detection is name-based (not
# value-based) so a field that *normally* carries an array still routes to
# extra when upstream emits ``None`` (e.g. Hill arrays when birth_year is
# absent and the estimator collapses).
_EFFECTIVE_SIZE_ARRAY_SUFFIXES: tuple[str, ...] = (
    "_per_gen", "_per_transition", "_per_cohort",
)
_EFFECTIVE_SIZE_ARRAY_NAMES: frozenset[str] = frozenset({
    "cohort_years", "age_table",
    "v_mm", "v_mf", "v_fm", "v_ff", "cov_m", "cov_f",
})


def _is_effective_size_array_key(key: str) -> bool:
    """Return True if ``key`` names an array field inside an Ne estimator."""
    if key in _EFFECTIVE_SIZE_ARRAY_NAMES:
        return True
    return any(key.endswith(suf) for suf in _EFFECTIVE_SIZE_ARRAY_SUFFIXES)


def _split_effective_size(es_dict: dict) -> tuple[dict, dict]:
    """Per-estimator scalar/array split for ``popgen.effective_size``.

    For each estimator: scalars and small dicts like ``cohort_window``
    stay in slim; per-generation / per-cohort / per-transition arrays
    plus ``age_table`` go to extra. Routing is name-based, so
    placeholder ``None`` values for unpopulated arrays still land in
    extra. ``ne_coancestry`` with ``ne is None`` gets a slim-only
    ``{ne: null}`` stub and no extra entry.
    """
    slim: dict = {}
    extra: dict = {}
    for est_name, est_value in es_dict.items():
        if not isinstance(est_value, dict):
            slim[est_name] = est_value
            continue
        if est_name == "ne_coancestry" and est_value.get("ne") is None:
            slim[est_name] = {"ne": None}
            continue
        est_slim: dict = {}
        est_extra: dict = {}
        for k, v in est_value.items():
            if _is_effective_size_array_key(k):
                est_extra[k] = v
            else:
                est_slim[k] = v
        if est_slim:
            slim[est_name] = est_slim
        if est_extra:
            extra[est_name] = est_extra
    return slim, extra


def _split_sex_summary(sex_dict: dict) -> tuple[dict, dict]:
    """Per-stratum split for ``strata.sex_summary``: scalars in slim, dists in extra."""
    slim: dict = {}
    extra: dict = {}
    for stratum_name, stratum_value in sex_dict.items():
        if not isinstance(stratum_value, dict):
            slim[stratum_name] = stratum_value
            continue
        s_slim = {k: stratum_value[k] for k in _SEX_SUMMARY_SLIM_KEYS if k in stratum_value}
        s_extra = {k: v for k, v in stratum_value.items() if k not in _SEX_SUMMARY_SLIM_KEYS}
        if s_slim:
            slim[stratum_name] = s_slim
        if s_extra:
            extra[stratum_name] = s_extra
    return slim, extra


def _split_section(value, spec: SectionSpec) -> tuple[object, object | None]:
    """Split a single section value into (slim, extra) per its SectionSpec."""
    if not isinstance(value, (dict, list)):
        return value, None  # scalar section → slim only
    if isinstance(value, list):
        if spec.list_of_dict_slim_keys is None:
            return value, None
        slim_rows: list = []
        extra_rows: list = []
        any_extra = False
        for row in value:
            if not isinstance(row, dict):
                slim_rows.append(row)
                extra_rows.append(None)
                continue
            keep = spec.list_of_dict_slim_keys
            slim_rows.append({k: row[k] for k in keep if k in row})
            row_extra = {k: v for k, v in row.items() if k not in keep}
            extra_rows.append(row_extra if row_extra else {})
            if row_extra:
                any_extra = True
        return slim_rows, (extra_rows if any_extra else None)
    # Dict section.
    if spec.name == "effective_size":
        s, e = _split_effective_size(value)
        return s, (e if e else None)
    if spec.name == "sex_summary":
        s, e = _split_sex_summary(value)
        return s, (e if e else None)
    if spec.slim_keys is None:
        return value, None
    slim_dict = {k: value[k] for k in spec.slim_keys if k in value}
    extra_dict = {k: v for k, v in value.items() if k not in spec.slim_keys}
    return slim_dict, (extra_dict if extra_dict else None)


def _drop_dotted_path(d: dict, parts: tuple[str, ...]) -> None:
    """Delete ``d[parts[0]][parts[1]]...`` if present. In-place; safe on missing keys."""
    if not parts or not isinstance(d, dict):
        return
    if len(parts) == 1:
        d.pop(parts[0], None)
        return
    sub = d.get(parts[0])
    if isinstance(sub, dict):
        _drop_dotted_path(sub, parts[1:])


def _split_summary(nested_ped: dict) -> tuple[dict, dict]:
    """Split a categorised pedigree dict into (slim, extra) per ``SUMMARY_SCHEMA``.

    Implements the split contract: every leaf key in ``nested_ped``
    appears in exactly one of (slim, extra, ``KNOWN_YAML_DROPS``). Empty
    categories are omitted from both files.
    """
    spec_by_name = {sec.name: sec for cat in SUMMARY_SCHEMA for sec in cat.sections}
    slim: dict = {}
    extra: dict = {}
    for cat_name, cat_dict in nested_ped.items():
        slim_cat: dict = {}
        extra_cat: dict = {}
        for sec_name, value in cat_dict.items():
            spec = spec_by_name.get(sec_name)
            if spec is None:
                slim_cat[sec_name] = value
                continue
            slim_val, extra_val = _split_section(value, spec)
            if slim_val not in (None, {}, []):
                slim_cat[sec_name] = slim_val
            if extra_val not in (None, {}, []):
                extra_cat[sec_name] = extra_val
        if slim_cat:
            slim[cat_name] = slim_cat
        if extra_cat:
            extra[cat_name] = extra_cat
    for drop_path in KNOWN_YAML_DROPS:
        if not drop_path.startswith("pedigree."):
            continue
        parts = tuple(drop_path.split(".")[1:])
        _drop_dotted_path(slim, parts)
        _drop_dotted_path(extra, parts)
    return slim, extra


def _split_individual_distributions(
    dists: dict,
) -> tuple[dict, dict]:
    """Split per-individual distributions into (slim, extra) residues.

    For each column:

    * If the column is in ``INDIVIDUAL_SLIM_COLS``, slim keeps the
      ``INDIVIDUAL_SLIM_DIST_KEYS`` (mean/median); extra carries the
      remaining quantile keys (min/std/q1/q3/max/nz).
    * Otherwise the full distribution dict goes to extra; slim has no
      entry for that column.

    Applies the ``individual.distributions.F.max`` YAML-only drop (drops
    from extra; slim never carried ``max`` for ``F`` anyway). Under this
    contract, no leaf path appears in both slim and extra.
    """
    slim_d: dict = {}
    extra_d: dict = {}
    for col, dist in dists.items():
        if not isinstance(dist, dict):
            slim_d[col] = dist
            continue
        if col in INDIVIDUAL_SLIM_COLS:
            slim_d[col] = {k: dist[k] for k in INDIVIDUAL_SLIM_DIST_KEYS if k in dist}
            residue = {k: v for k, v in dist.items() if k not in INDIVIDUAL_SLIM_DIST_KEYS}
            if residue:
                extra_d[col] = residue
        else:
            extra_d[col] = dict(dist)
    if "F" in extra_d and isinstance(extra_d["F"], dict):
        extra_d["F"].pop("max", None)  # KNOWN_YAML_DROPS
    return slim_d, extra_d


def _deep_merge_summary(slim, extra):
    """Merge a (slim, extra) pair back into one structure for ``--single-file``.

    Three cases: dict (deep-union keys), list-of-dict (zip-by-index,
    deep-union per entry), and scalars (slim wins — extra should not
    carry scalars under the split contract).
    """
    if extra is None:
        return slim
    if slim is None:
        return extra
    if isinstance(slim, dict) and isinstance(extra, dict):
        out: dict = dict(slim)
        for k, v in extra.items():
            if k in out:
                out[k] = _deep_merge_summary(out[k], v)
            else:
                out[k] = v
        return out
    if isinstance(slim, list) and isinstance(extra, list):
        if len(slim) != len(extra):
            return slim  # length mismatch: take slim (defensive; should not happen)
        return [_deep_merge_summary(a, b) for a, b in zip(slim, extra, strict=False)]
    return slim


def _build_pedigree_data(
    path: Path,
    cmd: str,
    size: dict,
    sibships: dict,
    pairs: dict,
    inbreeding: dict | None,
    mating_pairs: dict | None,
    relationship_summary: dict | None,
    aggregates: dict | None = None,
) -> tuple[dict, dict]:
    """Build the pedigree-level report payloads.

    Returns ``(tsv_payload, yaml_extras)``.

    * ``tsv_payload`` is the dict that gets flattened into
      ``summary.pedigree.tsv`` by ``_write_long_tsv``.  It contains every
      section that should appear as long-form (key, subkey, value) rows.
    * ``yaml_extras`` contains deep structures (e.g. ``effective_size``)
      that should appear in ``summary.yaml`` under ``pedigree:`` but
      should NOT be flattened to TSV.  Currently always returned empty
      from this helper — populated by the caller after Ne computation.
    """
    sibship_section: dict | None
    if sibships.get("empty"):
        sibship_section = None
    else:
        sibship_section = {
            "n_sibships": int(sibships["n_sibships"]),
            "mean": float(sibships["mean"]),
            "median": float(sibships["median"]),
            "q1": float(sibships["q1"]),
            "q3": float(sibships["q3"]),
            "size_dist": {str(k): float(v) for k, v in sibships["size_dist"].items()},
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
        }
    tsv_payload = {
        "input": str(path),
        "command": cmd,
        "version": VERSION,
        "generated_at": _now_iso(),
        "n_total": int(size["n_total"]),
        "max_degree_enumerated": 5,
        "size_structure": {
            "n_founders": int(size["n_founders"]),
            "founder_frac": float(size["founder_frac"]),
            "n_nonfounders": int(size["n_nonfounders"]),
            "nonfounder_frac": float(size["nonfounder_frac"]),
            "n_male": int(size["n_male"]),
            "n_female": int(size["n_female"]),
            "n_mother_links": int(size["n_mother_links"]),
            "n_father_links": int(size["n_father_links"]),
            "n_parent_child_edges": int(size["n_parent_child_edges"]),
            "n_with_both_parents": int(size["n_with_both_parents"]),
            "n_with_mother_only": int(size["n_with_mother_only"]),
            "n_with_father_only": int(size["n_with_father_only"]),
            "n_half_founders": int(size["n_half_founders"]),
            "max_depth": int(size["max_depth"]),
            "mean_depth": float(size["mean_depth"]),
            "median_depth": float(size["median_depth"]),
            "depth_counts": [int(x) for x in size["depth_counts"]],
            "n_components": int(size["n_components"]),
            "largest_component": int(size["largest_component"]),
            "largest_component_frac": float(size["largest_component_frac"]),
            "next_components": [int(x) for x in size["next_components"]],
        },
        "sibship_size": sibship_section,
        "mating_pairs": mating_pairs,
        "relationship_summary": relationship_summary,
        "reproduction": (aggregates or {}).get("reproduction", {}),
        "genealogy": (aggregates or {}).get("genealogy", {}),
        "founder_contribution": (aggregates or {}).get("founder_contribution", {}),
        "founder_summary": (aggregates or {}).get("founder_summary", {}),
        "components": (aggregates or {}).get("components", {}),
        "sex_summary": (aggregates or {}).get("sex_summary", {}),
        "depth_summary": (aggregates or {}).get("depth_summary", []),
        "pairs_engine": str(pairs.get("_engine", "matrix")),
        "relationship_pairs": {
            k: ({str(deg): int(c) for deg, c in v.items()} if k == "by_degree" else int(v))
            for k, v in pairs.items()
            if not k.startswith("_")
        },
        "inbreeding": inb_section,
    }
    yaml_extras: dict = {}
    return tsv_payload, yaml_extras


def _build_individual_data(
    idf: pd.DataFrame,
    path: Path,
    cmd: str,
    include_inbreeding: bool,
) -> dict:
    """Canonical nested dict for the per-individual distribution report."""
    n = len(idf)
    distributions: dict[str, dict] = {}
    for col in _NUMERIC_COLS:
        if col in {"F", "n_distinct_ancestors"} and not include_inbreeding:
            continue
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
    return out


_SUMMARY_META_KEYS = ("input", "command", "version", "generated_at", "n_total")

SAFE_MIN_CELL = 5


def _drop_distribution_extrema(obj: object) -> None:
    """Remove min/max from nested distribution dicts for safe-attempt output."""
    if isinstance(obj, dict):
        if {"mean", "q1", "median", "q3", "min", "max"}.issubset(obj.keys()):
            obj.pop("min", None)
            obj.pop("max", None)
        for v in obj.values():
            _drop_distribution_extrema(v)
    elif isinstance(obj, list):
        for v in obj:
            _drop_distribution_extrema(v)


def _apply_safe_attempt(ped_data: dict, ind_data: dict, min_cell: int = SAFE_MIN_CELL) -> None:
    """Best-effort small-cell redaction (in place). Not a safe-harbor.

    - Pedigree-level: nulls ``relationship_pairs`` codes and ``inbreeding``
      fields below ``min_cell``; suppresses ``hist`` buckets whose implied
      count (``frac × n_total``) is below the threshold; drops
      ``next_components`` entries below it; nulls positional ``depth_counts``
      entries below it.
    - Individual-level: drops ``min``/``max`` from every distribution, nulls
      ``nz`` below threshold.
    """
    n_total = int(ped_data.get("n_total", 0))

    sizes = ped_data.get("size_structure", {})
    sizes["next_components"] = [
        s for s in sizes.get("next_components", []) if s >= min_cell
    ]
    sizes["depth_counts"] = [
        (g if g >= min_cell else None) for g in sizes.get("depth_counts", [])
    ]
    if 0 < int(sizes.get("largest_component", 0)) < min_cell:
        sizes["largest_component"] = None

    _drop_distribution_extrema(ped_data)

    sibship = ped_data.get("sibship_size")
    mating = ped_data.get("mating_pairs")
    n_pairs = int(mating.get("n_pairs", 0)) if mating is not None else 0
    if sibship is not None and 0 < n_pairs < min_cell:
        for k in ("size_dist",):
            if k in sibship and isinstance(sibship[k], dict):
                sibship[k] = dict.fromkeys(sibship[k])

    if mating is not None:
        if 0 < int(mating.get("n_pairs", 0)) < min_cell:
            for k in list(mating):
                if k != "n_pairs":
                    mating[k] = None
        for k in ("n_pairs_with_multiple_children",):
            if 0 < int(mating.get(k, 0) or 0) < min_cell:
                mating[k] = None

    rel_summary = ped_data.get("relationship_summary") or {}
    for k in ("n_related_pairs", "n_unrelated_pairs"):
        if 0 < int(rel_summary.get(k, 0) or 0) < min_cell:
            rel_summary[k] = None
    for section in ("related_pairs_by_closest_degree", "closest_relationship_per_individual"):
        vals = rel_summary.get(section)
        if isinstance(vals, dict):
            for k, v in list(vals.items()):
                if isinstance(v, int) and 0 < v < min_cell:
                    vals[k] = None
    for row in rel_summary.get("related_pair_density_by_depth", []):
        if int(row.get("n", 0)) < min_cell:
            for k in list(row):
                if k not in ("depth", "n"):
                    row[k] = None
        else:
            for k in ("n_individual_pairs", "n_related_pairs", "n_unrelated_pairs"):
                if 0 < int(row.get(k, 0) or 0) < min_cell:
                    row[k] = None

    reproduction = ped_data.get("reproduction", {})
    for k in ("n_reproductive", "n_terminal"):
        if 0 < int(reproduction.get(k, 0) or 0) < min_cell:
            reproduction[k] = None

    founder = ped_data.get("founder_contribution", {})
    for k in ("n_founders_with_descendants", "n_founders_without_descendants"):
        if 0 < int(founder.get(k, 0) or 0) < min_cell:
            founder[k] = None

    founder_summary = ped_data.get("founder_summary", {})
    for row in founder_summary.get("by_depth", []):
        if int(row.get("n", 0)) < min_cell:
            for k in list(row):
                if k not in ("depth", "n"):
                    row[k] = None
        else:
            for k in ("active_founders",):
                if 0 < int(row.get(k, 0) or 0) < min_cell:
                    row[k] = None
    bottleneck = founder_summary.get("bottleneck")
    if isinstance(bottleneck, dict):
        for k in ("min_active_founders",):
            if 0 < int(bottleneck.get(k, 0) or 0) < min_cell:
                bottleneck[k] = None

    comps_full = ped_data.get("components", {})
    for k in ("singletons",):
        if 0 < int(comps_full.get(k, 0) or 0) < min_cell:
            comps_full[k] = None

    sex_summary = ped_data.get("sex_summary", {})
    for stats in sex_summary.values():
        if int(stats.get("n", 0)) < min_cell:
            for k in list(stats):
                if k != "n":
                    stats[k] = None
        else:
            for k in ("n_founders", "n_reproductive", "n_terminal", "n_inbred"):
                if 0 < int(stats.get(k, 0) or 0) < min_cell:
                    stats[k] = None

    for row in ped_data.get("depth_summary", []):
        if int(row.get("n", 0)) < min_cell:
            for k in list(row):
                if k not in ("depth", "n"):
                    row[k] = None
        else:
            for k in ("n_male", "n_female", "n_founders", "n_reproductive", "n_terminal", "n_inbred"):
                if 0 < int(row.get(k, 0) or 0) < min_cell:
                    row[k] = None

    pairs = ped_data.get("relationship_pairs", {})
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


def _build_summary_data(
    ped_data: dict, ind_data: dict, *, yaml_extras: dict | None = None,
) -> tuple[dict, dict]:
    """Build the (slim, extra) categorised YAML payloads from flat dicts.

    Pipeline: strip meta → drop ``effective_size_scalars`` (TSV-only)
    → splice ``yaml_extras`` (carries ``effective_size``) → categorise
    → split per ``SUMMARY_SCHEMA``. The same meta block sits at the top
    of both files so each is self-identifying. Per-individual
    ``distributions`` gets its own slim/extra split via
    ``_split_individual_distributions``.

    ``ped_data`` is left untouched (the TSV writers read it directly).
    """
    meta = {k: ped_data[k] for k in _SUMMARY_META_KEYS}

    flat_ped = {k: v for k, v in ped_data.items() if k not in _SUMMARY_META_KEYS}
    # ``effective_size_scalars`` is the TSV's separate scalar projection
    # (built in ``_run_summarize``); it never belonged in YAML. Drop it
    # before categorisation so it doesn't leak into the slim or extra
    # YAML files.
    flat_ped.pop("effective_size_scalars", None)
    if yaml_extras:
        flat_ped.update(yaml_extras)

    nested_ped = _categorise_pedigree(flat_ped)
    slim_ped, extra_ped = _split_summary(nested_ped)

    ind_payload = {k: v for k, v in ind_data.items() if k not in _SUMMARY_META_KEYS}
    dists = ind_payload.get("distributions", {})
    slim_dists, extra_dists = _split_individual_distributions(dists)
    slim_ind: dict = {k: v for k, v in ind_payload.items() if k != "distributions"}
    extra_ind: dict = {k: v for k, v in ind_payload.items() if k != "distributions"}
    if slim_dists:
        slim_ind["distributions"] = slim_dists
    if extra_dists:
        extra_ind["distributions"] = extra_dists

    slim_yaml = {**meta, "pedigree": slim_ped, "individual": slim_ind}
    extra_yaml = {**meta, "pedigree": extra_ped, "individual": extra_ind}
    return slim_yaml, extra_yaml


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


def _prepare_out_dir(path: Path) -> int:
    """Create ``path`` as a directory if needed; refuse if it is a file.

    Returns 0 on success, 1 if the path exists as a non-directory.
    """
    if path.exists() and not path.is_dir():
        logger.error(
            "--out %s exists and is not a directory; pass a directory path",
            path,
        )
        return 1
    path.mkdir(parents=True, exist_ok=True)
    return 0


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
    raw = _read_pedigree_table(in_path, sep=getattr(args, "sep", "auto"))
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
    idf_ids = idf["id"].to_numpy()
    if not np.array_equal(raw_ids, idf_ids):
        # ``load_and_validate`` may have reordered rows into topological
        # order when the input was not already sorted parents-before-
        # children.  Realign the raw read to ``idf`` order by ID.
        raw_id_to_row = pd.Series(np.arange(len(raw_ids), dtype=np.int64), index=raw_ids)
        perm = raw_id_to_row.reindex(idf_ids).to_numpy()
        if np.isnan(perm).any() or len(perm) != len(idf_ids):
            # Truly mismatched (rows added or dropped between input and
            # idf) — not a benign reorder.
            raise PedigreeError(
                "internal: row order mismatch between input and individual table"
            )
        raw = raw.iloc[perm.astype(np.int64)].reset_index(drop=True)

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
    """Render the validate summary as grouped sections with friendly labels.

    The internal check names remain in ``.validate.log``; only the on-screen
    rendering changes here. Unknown check names (e.g. after a partial rename)
    are skipped defensively rather than crashing the formatter.
    """
    by_check = {r.name: r for r in results}
    width = max(len(label) for label in _CHECK_LABELS.values()) + 1
    lines = [f"pedigree_summary.py: validating {path} (N={n_total:,})"]
    total_findings = 0
    for group_name, check_names in _CHECK_GROUPS:
        lines.append("")
        lines.append(group_name)
        for name in check_names:
            r = by_check.get(name)
            if r is None:
                continue  # defensive: tolerate missing check names
            label = _CHECK_LABELS.get(name, name)
            line = f"  {label} {'.' * (width - len(label))} {r.status}"
            if r.status == "FAIL":
                line += f" ({r.count})"
                total_findings += r.count
            elif r.skip_reason and r.status in ("SKIP", "PASS"):
                line += f" ({r.skip_reason})"
            lines.append(line)
    lines.append("")
    lines.append(f"result: {total_findings} finding(s)")
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
    """ArgumentParser that prints full help (not just usage) on parse errors.

    Disables prefix-matching abbreviation so deleted long-options cannot be
    silently resurrected via partial-match.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

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


def _add_format_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--sep", choices=_SEP_CHOICES, default="auto",
        help="input column delimiter. 'auto' (default) sniffs the first "
        "non-empty line for tab/comma/semicolon/pipe; falls back to "
        "whitespace (PLINK fam-style) when none are present. Pass an "
        "explicit choice to opt out of sniffing.",
    )
    p.add_argument(
        "--sex-encoding", choices=("auto", "default", "plink"), default="auto",
        help="how to decode the sex column: 'default' = 0=female, 1=male "
        "(pedsum default); 'plink' = 1=male, 2=female, 0=unknown (PLINK fam "
        "convention); 'auto' (default) detects from the observed tokens.",
    )
    p.add_argument(
        "--plink-sex", action="store_const", dest="sex_encoding", const="plink",
        help="legacy alias for --sex-encoding=plink (PLINK convention: 1=male, 2=female)",
    )
    p.add_argument(
        "--allow-missing-sex", action="store_true",
        help="tolerate rows whose sex is missing after imputation — either "
        "because the row is unsexed and not used as a parent (orphan), OR "
        "because it is used as BOTH mother and father with unknown sex "
        "(role-ambiguous). Such rows are auto-fixed to sex=-1 in the "
        "validate-fixed output. Without this flag, either case hard-blocks. "
        "Incompatible with --effective-size / --inbreeding in summarize "
        "(sex-stratified estimators require resolved sex).",
    )
    p.add_argument(
        "--no-override-asserted-sex", action="store_true",
        help="disable the 0.9 default of overriding asserted sex when topology "
        "unambiguously implies the opposite (asserted M used only as mother "
        "-> F; asserted F used only as father -> M). The existing "
        "missing->F/M imputation is unaffected. Restores 0.8's hard-block on "
        "sex/role contradictions via the sex_role_consistency check.",
    )


def _positive_int(v: str) -> int:
    """Argparse type guard for ints >= 1."""
    try:
        iv = int(v)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {v!r}") from exc
    if iv < 1:
        raise argparse.ArgumentTypeError(f"expected integer >= 1, got {iv}")
    return iv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _FullHelpParser(
        prog="pedigree_summary.py",
        description=(
            "Pedigree summary CLI. Depends on numpy, scipy, pandas, pyyaml, "
            "and pedigree-graph."
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
        "--out", dest="out_dir", required=True, type=Path, metavar="DIR",
        help="output directory (created if needed). Always writes "
        "summary.yaml (slim categorised summary), summary.extra.yaml "
        "(per-generation / per-cohort / per-transition arrays and full "
        "per-individual quantiles), and annotated.tsv.gz (input pedigree "
        "+ per-individual columns; suppressed under --safe-attempt). "
        "Pass --tsv to also write summary.pedigree.tsv and "
        "summary.individual.tsv.",
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
        "--birth-year-col", default=None, metavar="NAME",
        help="optional column name for birth year (integer or float "
        "calendar year; -1/NA/blank for unknown). When set, pedsum threads "
        "the column through to PedigreeGraph so the Hill overlapping-"
        "generation Ne estimator (Ne_H) can build its cohort window; "
        "without it Ne_H collapses to Ne_V.",
    )
    p_sum.add_argument(
        "--birth-year-min", type=int, default=_BIRTH_YEAR_DEFAULT_MIN, metavar="YEAR",
        help="inclusive lower bound for birth_year sanity check (default: %(default)s). "
        "No-op without --birth-year-col.",
    )
    p_sum.add_argument(
        "--birth-year-max", type=int, default=None, metavar="YEAR",
        help="inclusive upper bound for birth_year sanity check "
        "(default: current calendar year + 1). No-op without --birth-year-col.",
    )
    p_sum.add_argument(
        "--inbreeding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute per-individual F and the inbreeding summary section "
        "(default: on; pass --no-inbreeding to skip). F is the most expensive "
        "single computation in pedsum (~minutes on 10M-row pedigrees); pedsum "
        "logs an INFO line above N=1,000,000 so naive runs cannot silently "
        "hang. When `--effective-size` is also on, F is shared with the Ne "
        "pipeline (computed once via pedigree-graph's Meuwissen-Luo kernel). "
        "When off, F and n_ancestors in the per-individual table are "
        "zero-filled.",
    )
    p_sum.add_argument(
        "--effective-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute seven pedigree-based effective population size "
        "estimators (Ne_I, Ne_V, Ne_sr, Ne_iDeltaF, Ne_LTC, Ne_H, Ne_CT) via "
        "pedigree-graph.compute_all_ne (default: on; pass "
        "--no-effective-size to skip). The eighth estimator (Ne_C, coancestry "
        "rate) is opt-in via `--ne-coancestry` because its kinship DP can "
        "blow up RAM on very large pedigrees.",
    )
    p_sum.add_argument(
        "--ne-coancestry",
        action="store_true",
        help="include the coancestry-rate Ne_C estimator alongside the other "
        "seven. Off by default because the kinship DP can blow up RAM on very "
        "large pedigrees (>~500K rows). No-op without `--effective-size`.",
    )
    p_sum.add_argument(
        "--ne-threads",
        type=_positive_int, default=1, metavar="N",
        help="number of worker threads for independent Ne estimator dispatch "
        "(default: %(default)s; serial). No-op without `--effective-size`.",
    )
    p_sum.add_argument(
        "--per-individual-pairs",
        action="store_true",
        help="opt into per-individual relationship-burden summary. "
        "Requires the matrix engine to materialise full pair lists "
        "(OOMs on pair-dense pedigrees with N > ~500K). When unset "
        "(default), pedsum uses count_pairs_streaming for the 23 pair "
        "counts in O(N) memory; the burden summary is left as a stub. "
        "See the README for the streaming-vs-matrix precision contract.",
    )
    p_sum.add_argument(
        "--tsv",
        action="store_true",
        help="additionally write the long-form TSV summaries "
        "(summary.pedigree.tsv + summary.individual.tsv) inside --out. "
        "Off by default; collaborators typically need only the YAML.",
    )
    p_sum.add_argument(
        "--safe-attempt",
        action="store_true",
        help="best-effort GDPR-style redaction: skip the per-individual "
        "annotated TSV, drop min/max from distributions, and null any "
        "count or stratum below cell-size 5. Not a safe-harbor guarantee.",
    )
    _add_format_args(p_sum)
    _add_logging_args(p_sum)

    p_val = sub.add_parser("validate", help="run all integrity checks accumulating; report issues")
    p_val.add_argument("--in", dest="in_path", required=True, type=Path, help="input pedigree TSV")
    p_val.add_argument(
        "--out", dest="out_dir", required=True, type=Path, metavar="DIR",
        help="output directory (created if needed); writes validate.log "
        "(per-finding TSV) and validate.tsv.gz (the pedigree with any "
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
    p_val.add_argument(
        "--birth-year-col", default=None, metavar="NAME",
        help="optional column name for birth year (integer or float calendar "
        "year; -1/NA/blank for unknown). When set, validate runs three checks: "
        "birth_year_dtype (numeric parsing), birth_year_range (within "
        "[--birth-year-min, --birth-year-max]), and birth_year_topology "
        "(child birth_year >= parent birth_year).",
    )
    p_val.add_argument(
        "--birth-year-min", type=int, default=_BIRTH_YEAR_DEFAULT_MIN, metavar="YEAR",
        help="inclusive lower bound for birth_year_range check (default: %(default)s).",
    )
    p_val.add_argument(
        "--birth-year-max", type=int, default=None, metavar="YEAR",
        help="inclusive upper bound for birth_year_range check "
        "(default: current calendar year + 1).",
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
    if _prepare_out_dir(args.out_dir) != 0:
        return 1
    try:
        df, children_csr = load_and_validate(
            args.in_path,
            id_col=args.id_col,
            sex_col=args.sex_col,
            mother_col=args.mother_col,
            father_col=args.father_col,
            sex_encoding=args.sex_encoding,
            zero_as_missing=False,
            allow_missing_sex=args.allow_missing_sex,
            override_asserted_sex=not args.no_override_asserted_sex,
            birth_year_col=args.birth_year_col,
            birth_year_min=args.birth_year_min,
            birth_year_max=args.birth_year_max,
            sep=args.sep,
        )
    except PedigreeError as e:
        logger.error("validation failed: %s", e)
        return 1
    except (FileNotFoundError, OSError) as e:
        logger.error("file error: %s", e)
        return 2

    # Sex-stratified Ne / F kernel cannot honour SEX_UNKNOWN rows; refuse
    # the combination cleanly rather than producing silently-miscounted
    # output.
    if (df["sex"].to_numpy() == SEX_UNKNOWN).any() and (args.effective_size or args.inbreeding):
        logger.error(
            "sex-stratified Ne / F kernel requires resolved sex for every row; "
            "remove --allow-missing-sex, supply sex for the offending rows, or "
            "pass --no-effective-size / --no-inbreeding",
        )
        return 1

    # Flag-combination validation (must happen before any heavy work).
    # --effective-size is on by default; the warning fires only when the user
    # explicitly passed --no-effective-size alongside --ne-coancestry or a
    # non-default --ne-threads.
    if not args.effective_size and (args.ne_coancestry or args.ne_threads != 1):
        logger.warning(
            "--ne-coancestry / --ne-threads have no effect under --no-effective-size",
        )

    # Build the PedigreeGraph once and reuse for every primitive that
    # needs it (relationship pairs, F, lineage counts, effective size).
    # ``ped_depth`` MUST be populated from ``pg.generation`` before any
    # summary function runs — six callers read it.
    t0 = time.perf_counter()
    pg = _build_pedigree_graph(df)
    df["ped_depth"] = np.asarray(pg.generation, dtype=np.int32)
    logger.info("built PedigreeGraph in %.2fs", time.perf_counter() - t0)

    id_index = pd.Index(df["id"].to_numpy())

    t0 = time.perf_counter()
    size, comp_labels = compute_size_structure(df, children_csr)
    logger.info("size+structure in %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    sibships = compute_sibship_sizes(df)
    logger.info("sibship sizes in %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    mating_pairs = compute_mating_pair_summary(df)
    logger.info("mating-pair summary in %.2fs", time.perf_counter() - t0)

    n_indiv = len(df)
    if args.per_individual_pairs:
        t0 = time.perf_counter()
        pairs = count_relationship_pairs(
            df, include_pair_lists=True, pg=pg,
        )
        logger.info("relationship pairs in %.2fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        relationship_summary = compute_relationship_summary(df, pairs.get("_pair_lists"))
        logger.info("relationship burden summary in %.2fs", time.perf_counter() - t0)
    else:
        t0 = time.perf_counter()
        streamed_counts = pg.count_pairs_streaming(max_degree=5, scope="full")
        logger.info(
            "relationship pair counts (count_pairs_streaming) in %.2fs",
            time.perf_counter() - t0,
        )
        pairs = _augment_pair_counts(streamed_counts)
        pairs["_engine"] = "streaming_scalar"
        relationship_summary = {
            "computed": False,
            "skip_reason": (
                "per-individual relationship burden requires full pair-list "
                "enumeration; pass --per-individual-pairs to compute via the "
                "matrix engine"
            ),
            "n_individual_pairs": int(n_indiv * (n_indiv - 1) // 2),
        }

    if args.inbreeding:
        if n_indiv > _F_KERNEL_WARN_THRESHOLD:
            logger.info(
                "computing F on N=%s rows; may take several minutes — pass "
                "--no-inbreeding to skip",
                f"{n_indiv:,}",
            )
        t0 = time.perf_counter()
        F_vec = pg.compute_inbreeding()
        n_anc = pg.compute_n_ancestors()
        inb_summary: dict | None = _build_inbreeding_summary(F_vec)
        logger.info("inbreeding (F + n_ancestors) in %.2fs", time.perf_counter() - t0)
    else:
        logger.info("inbreeding: skipped (--no-inbreeding)")
        inb_summary = None
        F_vec = np.zeros(n_indiv, dtype=np.float64)
        n_anc = np.zeros(n_indiv, dtype=np.int32)

    t0 = time.perf_counter()
    n_desc = pg.compute_n_descendants()
    logger.info("descendants in %.2fs", time.perf_counter() - t0)

    effective_size: dict | None = None
    if args.effective_size:
        t0 = time.perf_counter()
        effective_size = compute_effective_size(
            pg, ne_coancestry=args.ne_coancestry, n_threads=args.ne_threads,
        )
        logger.info(
            "effective size (%d estimators) in %.2fs",
            8 if args.ne_coancestry else 7,
            time.perf_counter() - t0,
        )

    out_dir = args.out_dir

    t0 = time.perf_counter()
    sex_source = df["sex_source"].to_numpy()
    idf = build_individual_df(
        df, id_index, F_vec, n_anc, n_desc, comp_labels, sex_source,
    )
    founder_summary, n_founder_anc = compute_founder_summary(idf)
    idf["n_founder_ancestors"] = n_founder_anc
    logger.info("individual table built in %.2fs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    aggregates = compute_aggregate_sections(
        idf, founder_summary=founder_summary, include_inbreeding=args.inbreeding,
    )
    logger.info("aggregate pedigree sections in %.2fs", time.perf_counter() - t0)

    tsv_payload, yaml_extras = _build_pedigree_data(
        args.in_path, cmd, size, sibships, pairs, inb_summary, mating_pairs,
        relationship_summary, aggregates,
    )
    if effective_size is not None:
        tsv_payload["effective_size_scalars"] = {
            name: result["ne"] for name, result in effective_size.items()
        }
        yaml_extras["effective_size"] = effective_size

    ind_data = _build_individual_data(
        idf, args.in_path, cmd, include_inbreeding=args.inbreeding,
    )

    if args.safe_attempt:
        _apply_safe_attempt(tsv_payload, ind_data)
        logger.info("safe-attempt redaction applied (min cell = %d)", SAFE_MIN_CELL)

    slim_yaml, extra_yaml = _build_summary_data(
        tsv_payload, ind_data, yaml_extras=yaml_extras,
    )
    _write_yaml(slim_yaml, out_dir / "summary.yaml")
    _write_yaml(extra_yaml, out_dir / "summary.extra.yaml")
    logger.info(
        "wrote %s/{summary.yaml, summary.extra.yaml}", out_dir,
    )

    if args.tsv:
        _write_long_tsv(tsv_payload, out_dir / "summary.pedigree.tsv")
        _write_long_tsv(ind_data, out_dir / "summary.individual.tsv")
        logger.info(
            "wrote %s/{summary.pedigree.tsv, summary.individual.tsv}", out_dir,
        )

    if args.safe_attempt:
        logger.info(
            "safe-attempt: skipped %s/annotated.tsv.gz (per-individual)", out_dir,
        )
    else:
        t0 = time.perf_counter()
        _write_annotated_tsv(args.in_path, args, idf, out_dir / "annotated.tsv.gz")
        logger.info(
            "wrote %s/annotated.tsv.gz in %.2fs",
            out_dir, time.perf_counter() - t0,
        )

    return 0


def _run_validate(args: argparse.Namespace, cmd: str) -> int:
    if _prepare_out_dir(args.out_dir) != 0:
        return 1
    try:
        n_total, results, findings, ctx = validate_pedigree(
            args.in_path,
            id_col=args.id_col,
            sex_col=args.sex_col,
            mother_col=args.mother_col,
            father_col=args.father_col,
            sex_encoding=args.sex_encoding,
            zero_as_missing=False,
            allow_missing_sex=args.allow_missing_sex,
            override_asserted_sex=not args.no_override_asserted_sex,
            birth_year_col=args.birth_year_col,
            birth_year_min=args.birth_year_min,
            birth_year_max=args.birth_year_max,
            sep=args.sep,
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
    if by_check["sex_role_ambiguity"].status == "FAIL":
        blocks.append(
            "present individual(s) with unknown sex used as BOTH mother and father "
            "(sex cannot be imputed); pass --allow-missing-sex to tolerate"
        )
    if by_check["unknown_sex"].status == "FAIL":
        blocks.append("rows with unresolved sex; pass --allow-missing-sex to tolerate")

    sys.stderr.write(_format_check_summary(args.in_path, n_total, results))

    out_dir = args.out_dir
    log_path = out_dir / "validate.log"
    _write_validate_log(findings, log_path)
    sys.stderr.write(f"wrote {log_path} ({len(findings)} finding(s))\n")

    if blocks:
        sys.stderr.write("\nBLOCKED — fix the following before re-running:\n")
        for b in blocks:
            sys.stderr.write(f"  - {b}\n")
        return 2

    added_founders: list[dict] = []
    df_out = ctx["df_raw"]
    if ctx["ids"] is not None and ctx["mothers"] is not None and ctx["fathers"] is not None:
        id_index = pd.Index(ctx["ids"])
        added_founders = _build_added_founders(
            ctx["mothers"], ctx["fathers"], id_index, args.no_sex_check,
        )
        # Fold sex imputation into the fixed output so the user's "fixed"
        # file reflects the auto-fix instead of the original blanks.
        sex_imp = ctx.get("sex_imputation")
        if sex_imp is not None:
            df_out = df_out.copy()
            imputed = sex_imp["imputed_sex"]
            original_unknown = sex_imp["original_unknown_mask"]
            overridden = sex_imp.get(
                "overridden_mask", np.zeros(len(imputed), dtype=bool),
            )
            # Rewrite the sex column wherever pedsum changed it: missing→F/M
            # imputation (0.8) and asserted→role overrides (0.9). Rows still
            # SEX_UNKNOWN after both passes — orphan or role-ambiguous —
            # normalise to "-1" so the fixed TSV is self-consistent.
            sex_col_values = df_out[args.sex_col].astype(object).copy()
            modified = original_unknown | overridden
            unresolved_mask = imputed == SEX_UNKNOWN
            sex_col_values.loc[modified & (imputed == SEX_FEMALE)] = "F"
            sex_col_values.loc[modified & (imputed == SEX_MALE)] = "M"
            sex_col_values.loc[unresolved_mask] = "-1"
            df_out[args.sex_col] = sex_col_values
            if sex_imp["n_imputed"] > 0:
                logger.info(
                    "validate: imputed sex for %d row(s) from parent role",
                    int(sex_imp["n_imputed"]),
                )
            n_normalised = int(unresolved_mask.sum())
            if n_normalised > 0:
                logger.info(
                    "validate: normalised %d unresolved-sex row(s) to -1 in fixed output",
                    n_normalised,
                )
            # 0.9 per-row audit: stamp sex_source onto the fixed output BEFORE
            # the topological reorder below, so pandas reorders the column
            # along with the rest.
            df_out["sex_source"] = sex_imp["sex_source"]
        # Reorder so the fixed file is parents-before-children and feeds
        # back into pedsum without further auto-fixes.
        m_row, _ = _parent_rows(ctx["mothers"], id_index)
        f_row, _ = _parent_rows(ctx["fathers"], id_index)
        try:
            depth = _compute_depth_unordered(m_row, f_row, len(ctx["ids"]))
        except PedigreeError:
            depth = None  # acyclic FAIL already surfaced; skip reorder
        if depth is not None:
            order = np.argsort(depth, kind="stable")
            if not np.array_equal(order, np.arange(len(order))):
                logger.info(
                    "validate: reordering %d row(s) into topological order",
                    int((order != np.arange(len(order))).sum()),
                )
                df_out = df_out.iloc[order].reset_index(drop=True)

    out_path = out_dir / "validate.tsv.gz"
    _write_validate_tsv_gz(
        df_out, added_founders,
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
