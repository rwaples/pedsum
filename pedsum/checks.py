"""Per-check validation finding producers and check metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN
from pedsum.pedigree_ops import _build_children_csr, _parent_rows

if TYPE_CHECKING:
    from pedsum.pedigree_ops import IdIndex


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


# The check registry (names, ordering, labels, grouping) lives in
# ``pedsum.validate`` alongside the runner. This module is the pure
# library of finding producers consumed by that registry.


def _rows_phrase(rows: np.ndarray, limit: int = 5) -> str:
    """Render referencing rows for a finding detail, capped at ``limit``.

    Lets ID-level checks (sex_role_consistency, parent_refs_sex_conflict) pin
    the offending lines in their detail. Wording matches ``_rows_listing`` in
    ``report.py``.
    """
    listed = rows[:limit].tolist()
    suffix = f" (and {len(rows) - limit} more)" if len(rows) > limit else ""
    return f"row(s) {listed}{suffix}"


def _check_negative_ids(ids: np.ndarray) -> list[Finding]:
    """Detect negative IDs (id < 0). Returns one Finding per offending row."""
    return [
        Finding(check="negative_ids", id=int(ids[i]), row=int(i), detail=f"id={int(ids[i])} is negative; must be >= 0")
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
    return [Finding(check="duplicate_ids", id=int(d), detail=f"id={int(d)} appears more than once") for d in dup_ids]


def _check_parent_token_range(arr: np.ndarray, role: str) -> list[Finding]:
    """Detect parent column values < -1; one Finding per offending row."""
    return [
        Finding(
            check=f"parent_token_range_{role}",
            row=int(i),
            detail=f"row {int(i)}: {role} value {int(arr[i])} < -1; missing parent must be -1",
        )
        for i in np.where(arr < -1)[0]
    ]


def _check_parent_refs_present(arr: np.ndarray, role: str, id_index: IdIndex) -> list[Finding]:
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
    ids: np.ndarray,
    mothers: np.ndarray,
    fathers: np.ndarray,
) -> list[Finding]:
    """Detect rows where mother == father (and both != -1); one Finding per row."""
    bad = (mothers != -1) & (fathers != -1) & (mothers == fathers)
    return [
        Finding(
            check="parents_distinct",
            id=int(ids[i]),
            row=int(i),
            detail=f"row {int(i)}: id={int(ids[i])} has mother == father == {int(mothers[i])}",
        )
        for i in np.where(bad)[0]
    ]


def _check_self_loops(ids: np.ndarray, mothers: np.ndarray, fathers: np.ndarray) -> list[Finding]:
    """Detect rows where id == mother or id == father; one Finding per offending row."""
    findings = [
        Finding(
            check="self_loops",
            id=int(ids[i]),
            row=int(i),
            detail=f"row {int(i)}: id={int(ids[i])} listed as own mother",
        )
        for i in np.where(mothers == ids)[0]
    ]
    findings.extend(
        Finding(
            check="self_loops",
            id=int(ids[i]),
            row=int(i),
            detail=f"row {int(i)}: id={int(ids[i])} listed as own father",
        )
        for i in np.where(fathers == ids)[0]
    )
    return findings


def _check_parent_refs_sex_conflict(
    mothers: np.ndarray,
    fathers: np.ndarray,
    id_index: IdIndex,
) -> list[Finding]:
    """Detect missing parent IDs referenced as both mother AND father; one Finding per ID."""
    moms = np.unique(mothers[mothers != -1])
    dads = np.unique(fathers[fathers != -1])
    moms_missing = moms[id_index.get_indexer(moms) == -1]
    dads_missing = dads[id_index.get_indexer(dads) == -1]
    conflicts = np.intersect1d(moms_missing, dads_missing)
    return [
        Finding(
            check="parent_refs_sex_conflict",
            id=int(c),
            detail=(
                f"id={int(c)} referenced as both mother ({_rows_phrase(np.where(mothers == c)[0])}) "
                f"and father ({_rows_phrase(np.where(fathers == c)[0])}) (sex ambiguous)"
            ),
        )
        for c in conflicts
    ]


def _sex_role_findings(
    role: str,
    parents: np.ndarray,
    ids: np.ndarray,
    own_rows: np.ndarray,
    sex: np.ndarray,
    expected_sex: int,
) -> list[Finding]:
    """Build sex_role_consistency Findings for ids used as ``role`` with the wrong sex.

    ``row`` is the individual's own row (where its sex is declared); the detail
    names the row(s) that reference the id in this role so the log pins the
    offending lines.
    """
    expected = "female" if expected_sex == SEX_FEMALE else "male"
    bad = sex[own_rows] != expected_sex
    return [
        Finding(
            check="sex_role_consistency",
            id=int(pid),
            row=int(own_row),
            detail=f"id={int(pid)} used as {role} in {_rows_phrase(np.where(parents == pid)[0])} but sex != {expected}",
        )
        for pid, own_row in zip(ids[bad], own_rows[bad], strict=True)
    ]


def _check_sex_role_consistency(
    mothers: np.ndarray,
    fathers: np.ndarray,
    sex: np.ndarray,
    id_index: IdIndex,
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

    keep_m = rows_um != -1
    keep_f = rows_uf != -1
    if skip_mask is not None:
        present_m = keep_m.copy()
        present_f = keep_f.copy()
        keep_m[present_m] = ~skip_mask[rows_um[present_m]]
        keep_f[present_f] = ~skip_mask[rows_uf[present_f]]

    mother_ids = used_as_mother[keep_m]
    mother_rows = rows_um[keep_m]
    father_ids = used_as_father[keep_f]
    father_rows = rows_uf[keep_f]

    findings = _sex_role_findings("mother", mothers, mother_ids, mother_rows, sex, SEX_FEMALE)
    findings.extend(_sex_role_findings("father", fathers, father_ids, father_rows, sex, SEX_MALE))
    return findings


def _check_sex_role_ambiguity(
    ids: np.ndarray,
    ambiguous_mask: np.ndarray,
    mothers: np.ndarray,
    fathers: np.ndarray,
) -> list[Finding]:
    """Detect unsexed rows that are referenced as BOTH mother and father.

    A hard block by default — this is a data contradiction, not just missing
    information. Pedsum 0.8 added ``--allow-missing-sex`` which downgrades
    this to a SKIP with a tolerated count; the fixed validate-output writes
    these rows' sex as ``-1`` so the on-disk pedigree is self-consistent.

    The first row referencing each ambiguous id as mother / father is found on
    demand here (rather than precomputed) because ``ambiguous_mask`` is empty on
    every well-formed pedigree — this loop almost never runs.
    """
    findings: list[Finding] = []
    for row_idx in np.where(ambiguous_mask)[0]:
        rid = int(ids[row_idx])
        m_rows = np.where(mothers == rid)[0]
        f_rows = np.where(fathers == rid)[0]
        mrow = int(m_rows[0]) if m_rows.size else -1
        frow = int(f_rows[0]) if f_rows.size else -1
        findings.append(
            Finding(
                check="sex_role_ambiguity",
                id=rid,
                row=int(row_idx),
                detail=(
                    f"id={rid} has unknown sex AND is referenced as both "
                    f"mother (row {mrow}) and father (row {frow}); "
                    "sex cannot be imputed"
                ),
            )
        )
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
            detail=(f"id={int(ids[i])} has unknown sex and is not referenced as a parent; cannot be imputed"),
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


def _check_acyclic(ids: np.ndarray, mothers: np.ndarray, fathers: np.ndarray, id_index: IdIndex) -> list[Finding]:
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
        Finding(
            check="acyclic",
            id=int(ids[i]),
            row=int(i),
            detail=f"id={int(ids[i])} could not be topologically ordered (in a cycle)",
        )
        for i in np.where(indeg > 0)[0]
    ]


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
                f"id={int(ids[i])} birth_year={int(birth_year[i])} is outside the sanity range [{year_min}, {year_max}]"
            ),
        )
        for i in np.where(bad)[0]
    ]


def _check_birth_year_topology(
    ids: np.ndarray,
    mothers: np.ndarray,
    fathers: np.ndarray,
    birth_year: np.ndarray,
    id_index: IdIndex,
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
            findings.append(
                Finding(
                    check="birth_year_topology",
                    id=int(ids[row_idx]),
                    row=row_idx,
                    detail=(
                        f"id={int(ids[row_idx])} birth_year={int(child_by[i])} < "
                        f"{role} (id={int(parents[row_idx])}) "
                        f"birth_year={int(parent_by[i])}"
                    ),
                )
            )
    return findings
