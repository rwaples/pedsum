"""The Check registry: each integrity Check is one ``(ctx) -> CheckResult`` function.

``pedsum.validate`` runs ``CHECKS`` in order (ADR 0002). A Check reads the
parsed arrays and the config off the ``ValidationContext``; the parse Checks
(``id_dtype``, ``sex_tokens``, ``birth_year_dtype``) also populate them. The
runner stamps each result's ``name`` and each Finding's ``check``, so a Check
returns a fresh ``CheckResult`` and leaves both unset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, PedigreeError, logger
from pedsum.ids import parse_ids
from pedsum.parse import (
    ZYGOSITY_MZ,
    ZYGOSITY_UNKNOWN,
    as_birth_date_col,
    as_birth_year_col,
    as_group_codes,
    as_zygosity_col,
    decode_sex,
    format_id_sample,
)
from pedsum.pedigree_ops import cycle_components, rows_of_parents

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pedsum.validate import ValidationContext

    #: Prints an id code as the input's ID (``IdLabels.label``).
    Label = Callable[[int], str]


@dataclass
class Finding:
    """A single per-individual validation finding.

    ``id`` is the int64 ID code; ``detail`` names IDs as the input wrote them.
    ``n_refs`` is set only by ``parent_refs_present_*``: how many rows name the
    missing id in that parent column. ``check`` is stamped by the runner.
    """

    check: str = ""
    id: int | None = None
    row: int | None = None
    detail: str = ""
    n_refs: int | None = None


class CheckStatus(StrEnum):
    """A Check's verdict."""

    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    """One Check's verdict and its Findings; a FAIL has at least one.

    ``note`` says why a Check was SKIPped, or annotates a PASS. ``name`` is
    stamped by the runner.
    """

    status: CheckStatus
    findings: list[Finding] = field(default_factory=list)
    note: str | None = None
    name: str = ""

    @property
    def count(self) -> int:
        """Number of Findings."""
        return len(self.findings)


def _verdict(findings: list[Finding]) -> CheckResult:
    """PASS when there are no Findings, else FAIL carrying them."""
    return CheckResult(CheckStatus.FAIL if findings else CheckStatus.PASS, findings)


def _failed(detail: str) -> CheckResult:
    """A FAIL with one Finding that names no individual."""
    return CheckResult(CheckStatus.FAIL, [Finding(detail=detail)])


def rows_phrase(rows: np.ndarray, limit: int = 5) -> str:
    """Render referencing rows for a finding detail or a founder reason, capped at ``limit``."""
    listed = rows[:limit].tolist()
    suffix = f" (and {len(rows) - limit} more)" if len(rows) > limit else ""
    return f"row(s) {listed}{suffix}"


def summarize_findings(findings: list[Finding], label: Label) -> str:
    """Build a human-readable error message from a list of findings."""
    if not findings:
        return ""
    check = findings[0].check
    n = len(findings)
    samples = []
    for f in findings[:5]:
        if f.id is not None and f.row is not None:
            samples.append(f"row {f.row} (id={label(f.id)})")
        elif f.id is not None:
            samples.append(f"id={label(f.id)}")
        elif f.row is not None:
            samples.append(f"row {f.row}")
    sample_str = ", ".join(samples) if samples else findings[0].detail
    extra = f" (and {n - 5} more)" if n > 5 else ""
    return f"{check}: {n} finding(s) — {sample_str}{extra}"


# ---------------------------------------------------------------------------
# Columns & parsing
# ---------------------------------------------------------------------------


def _required_columns(ctx: ValidationContext) -> CheckResult:
    # summarize treats a named --birth-year-col as required; validate omits it
    # and lets birth_year_dtype SKIP instead. --twin-max-days needs the dates.
    config = ctx.config
    needed = {config.id_col, config.sex_col, config.mother_col, config.father_col}
    if ctx.require_birth_year_col and config.birth_year_col is not None:
        needed.add(config.birth_year_col)
    if config.twin_max_days is not None and config.birth_date_col is not None:
        needed.add(config.birth_date_col)
    needed |= {c for c in (config.multiple_birth_col, config.zygosity_col) if c is not None}
    miss = needed - set(ctx.df_raw.columns)
    if miss:
        return _failed(f"missing required columns: {sorted(miss)}; file has {list(ctx.df_raw.columns)}")
    return _verdict([])


def _empty_pedigree(ctx: ValidationContext) -> CheckResult:
    return _verdict([] if len(ctx.df_raw) else [Finding(detail="pedigree has 0 rows")])


def _id_dtype(ctx: ValidationContext) -> CheckResult:
    df, config = ctx.df_raw, ctx.config
    try:
        ctx.ids, ctx.mothers, ctx.fathers, ctx.id_labels = parse_ids(
            df[config.id_col],
            df[config.mother_col],
            df[config.father_col],
            labels=ctx.fixed_labels,
            missing_parent_tokens=config.missing_parent_tokens,
        )
    except PedigreeError as e:
        return _failed(str(e))
    logger.info("parsed %d ids; five random ids: %s", len(ctx.ids), format_id_sample(ctx.ids, ctx.label))
    return _verdict([])


def _warn_zero_female_coding(ctx: ValidationContext) -> None:
    """Warn when mothers are coded 0 and no row is 2: likely 0=female, the coding pedsum wrote before 0.15.0.

    pedsum's own output never writes a mother as 0 (she is imputed female),
    so its unknown-sex rows do not trigger this.
    """
    if ctx.ids is None or ctx.mothers is None:
        return
    tokens = ctx.df_raw[ctx.config.sex_col].cast(pl.String).str.strip_chars()
    zero = (tokens == "0").fill_null(False).to_numpy()
    if not zero.any() or (tokens == "2").any():
        return
    n = int((zero & np.isin(ctx.ids, ctx.mothers[ctx.mothers != -1])).sum())
    if n:
        logger.warning(
            "sex column codes %d mother(s) as 0 and has no 2: pedsum reads 0 as unknown "
            "(PLINK: 1=male, 2=female, 0=unknown). Recode a file that uses 0=female, 1=male, "
            "including validate and summarize output from pedsum before 0.15.0.",
            n,
        )


def _sex_tokens(ctx: ValidationContext) -> CheckResult:
    try:
        ctx.sex = decode_sex(ctx.df_raw[ctx.config.sex_col])
    except PedigreeError as e:
        return _failed(str(e))
    _warn_zero_female_coding(ctx)
    return _verdict([])


# ---------------------------------------------------------------------------
# IDs
# ---------------------------------------------------------------------------


def _negative_ids(ctx: ValidationContext) -> CheckResult:
    """One Finding per row whose ID is written as a negative integer."""
    ids = ctx.ids
    assert ids is not None
    assert ctx.id_labels is not None
    values = ctx.id_labels.int_values(ids)
    return _verdict(
        [
            Finding(id=int(ids[i]), row=ctx.row(i), detail=f"id={ctx.label(ids[i])} is negative; must be >= 0")
            for i in np.where(values < 0)[0]
        ]
    )


def _duplicate_rows(ctx: ValidationContext) -> CheckResult:
    """One Finding per row that repeats an earlier row; the copies are removed from ``ctx``.

    Rows are compared as read (ADR 0006): the ID, parents, sex, birth year and
    birth date after parsing, so ``NA`` and ``.`` are the same missing parent and ``M``
    and ``1`` the same sex, and every other column as text. Only rows whose
    ID is on more than one row are compared.
    """
    ids, mothers, fathers = ctx.require_id_parents()
    assert ctx.sex is not None
    if not (ctx.sorted_ids[1:] == ctx.sorted_ids[:-1]).any():
        return _verdict([])
    order = np.argsort(ids, kind="stable")
    same = ids[order][1:] == ids[order][:-1]
    repeated = np.zeros(len(ids), dtype=bool)
    repeated[order[1:][same]] = True
    repeated[order[:-1][same]] = True
    rows = np.flatnonzero(repeated)
    config = ctx.config
    parsed = {config.id_col, config.sex_col, config.mother_col, config.father_col}
    keys = {"__id": ids[rows], "__mother": mothers[rows], "__father": fathers[rows], "__sex": ctx.sex[rows]}
    if ctx.birth_year is not None and config.birth_year_col is not None:
        parsed.add(config.birth_year_col)
        keys["__birth_year"] = ctx.birth_year[rows]
    if ctx.birth_date is not None and config.birth_date_col is not None:
        parsed.add(config.birth_date_col)
        keys["__birth_date"] = ctx.birth_date[rows].astype(np.int64)
    # Positional names, so no input column can collide with the key columns.
    other = ctx.df_raw[rows].select(
        pl.col(c).alias(f"__col{i}") for i, c in enumerate(ctx.df_raw.columns) if c not in parsed
    )
    frame = pl.DataFrame(keys).hstack(other).with_columns(__row=pl.Series(rows))
    key_cols = [c for c in frame.columns if c != "__row"]
    first = frame.select(pl.col("__row").min().over(key_cols)).to_series().to_numpy()
    copies = first != rows
    if not copies.any():
        return _verdict([])
    findings = [
        Finding(
            id=int(ids[r]),
            row=ctx.row(r),
            detail=f"row {ctx.row(r)} repeats row {ctx.row(f)} (id={ctx.label(ids[r])}); the copy is removed",
        )
        for r, f in zip(rows[copies], first[copies], strict=True)
    ]
    keep = np.ones(len(ids), dtype=bool)
    keep[rows[copies]] = False
    ctx.keep_rows(keep)
    # The kept ids, still sorted: duplicate_ids reads them without sorting again.
    ctx.__dict__["sorted_ids"] = ids[order][keep[order]]
    return _verdict(findings)


def _duplicate_ids(ctx: ValidationContext) -> CheckResult:
    """One Finding per ID on rows that differ (exact copies are ``duplicate_rows``')."""
    sorted_ids = ctx.sorted_ids
    dup_mask = sorted_ids[1:] == sorted_ids[:-1]
    if not dup_mask.any():
        return _verdict([])
    dup_ids = np.unique(sorted_ids[1:][dup_mask])
    return _verdict([Finding(id=int(d), detail=f"id={ctx.label(d)} is on rows that differ") for d in dup_ids])


# ---------------------------------------------------------------------------
# Parent references
# ---------------------------------------------------------------------------


def _parent_token_range(ctx: ValidationContext, role: str) -> CheckResult:
    """One Finding per row whose ``role`` value is below -1."""
    parents = ctx.mothers if role == "mother" else ctx.fathers
    assert parents is not None
    assert ctx.id_labels is not None
    values = ctx.id_labels.int_values(parents)
    return _verdict(
        [
            Finding(
                row=ctx.row(i),
                detail=(
                    f"row {ctx.row(i)}: {role} value {int(values[i])} < -1; a missing parent is -1, blank or NA "
                    f"(or pass --missing-parent-token {int(values[i])})"
                ),
            )
            for i in np.where(values < -1)[0]
        ]
    )


def _parent_refs_present(ctx: ValidationContext, role: str) -> CheckResult:
    """One Finding per ``role`` ID that has no row of its own."""
    refs = ctx.mother_refs if role == "mother" else ctx.father_refs
    no_row = refs.rows == -1
    missing, counts = refs.ids[no_row], refs.n_children[no_row]
    findings = []
    for mid, n in zip(missing, counts, strict=True):
        n_refs = int(n)
        name = ctx.label(mid)
        detail = f"id={name} referenced as {role} {n_refs} time(s) but no row has this id"
        if name == "0":
            detail += " (if the file writes a missing parent as 0, pass --missing-parent-token 0)"
        findings.append(Finding(id=int(mid), detail=detail, n_refs=n_refs))
    return _verdict(findings)


def _parent_child_count(ctx: ValidationContext) -> CheckResult:
    """One Finding per parent with more children than its role's limit (pedsum#27).

    A placeholder for an unknown parent, such as ``9999``, becomes one parent
    of everyone it stands in for, and so a full or half sib of strangers.
    The limits are per role, as a real father can have more children than a
    real mother; 0 turns a role off.
    """
    config = ctx.config
    if not config.max_children_mother and not config.max_children_father:
        return CheckResult(CheckStatus.SKIP, note="no --max-children-mother or --max-children-father")
    findings = []
    for role, refs, limit in (
        ("mother", ctx.mother_refs, config.max_children_mother),
        ("father", ctx.father_refs, config.max_children_father),
    ):
        if not limit:
            continue
        for k in np.flatnonzero(refs.n_children > limit):
            pid, row, name = int(refs.ids[k]), int(refs.rows[k]), ctx.label(refs.ids[k])
            # A missing-parent token cannot also be an id with a row.
            remedy = (
                f"it has no row, so validate adds it as a founder. If it stands for an unknown {role}, "
                f"pass --missing-parent-token {name}"
                if row == -1
                else f"if it stands for an unknown {role}, delete its row and pass --missing-parent-token {name}"
            )
            findings.append(
                Finding(
                    id=pid,
                    row=None if row == -1 else ctx.row(row),
                    detail=(
                        f"id={name} is the {role} of {int(refs.n_children[k])} children "
                        f"(more than --max-children-{role} {limit}); {remedy}"
                    ),
                )
            )
    return _verdict(findings)


def _zero_parent_present(ctx: ValidationContext) -> CheckResult:
    """One Finding when a parent ID written ``0`` has no row of its own, in either role.

    PLINK and most R cleaners write a missing parent as ``0``; pedsum reads it
    as an ID unless ``--missing-parent-token 0`` says otherwise (ADR 0006).
    """
    _, mothers, fathers = ctx.require_id_parents()
    assert ctx.id_labels is not None
    zero = ctx.id_labels.code_of("0")
    if zero is None or ctx.id_index.get_indexer([zero])[0] != -1:
        return _verdict([])
    roles = [
        f"{role} in {rows_phrase(ctx.rows(rows))}"
        for role, parents in (("mother", mothers), ("father", fathers))
        if len(rows := np.flatnonzero(parents == zero))
    ]
    if not roles:
        return _verdict([])
    return _verdict(
        [
            Finding(
                id=zero,
                detail=(
                    f"id=0 referenced as {' and as '.join(roles)} but no row has this id; "
                    "if the file writes a missing parent as 0, pass --missing-parent-token 0"
                ),
            )
        ]
    )


def _cycle_breaks(ctx: ValidationContext) -> CheckResult:
    """One Finding per parent link ``--break-cycles`` clears; the links are cleared in ``ctx``.

    It clears every self-link, and within each descent cycle every link whose
    parent is born after the child (ADR 0006). A cycle with no such link is
    left for ``acyclic``. Clearing a link inside a cycle cannot put another
    such link into a new cycle, so one pass suffices.
    """
    if not ctx.config.break_cycles:
        return CheckResult(CheckStatus.SKIP, note="no --break-cycles")
    ids, mothers, fathers = ctx.require_id_parents()
    birth_year = ctx.birth_year
    assert birth_year is not None
    m_row, _ = rows_of_parents(mothers, ctx.id_index)
    f_row, _ = rows_of_parents(fathers, ctx.id_index)
    component = cycle_components(m_row, f_row)
    findings: list[Finding] = []
    for role, parents, parent_rows in (("mother", mothers, m_row), ("father", fathers, f_row)):
        own = parents == ids
        known = parent_rows >= 0
        prow = np.where(known, parent_rows, 0)
        child_by, parent_by = birth_year, birth_year[prow]
        late = (
            known
            & ~own
            & (component >= 0)
            & (component == component[prow])
            & (child_by != -1)
            & (parent_by != -1)
            & (parent_by > child_by)
        )
        for i in np.flatnonzero(own | late):
            child, parent = ctx.label(ids[i]), ctx.label(parents[i])
            if own[i]:
                reason = f"listed as own {role}"
            else:
                reason = f"in a cycle, {role} born {int(parent_by[i])} after the child (born {int(child_by[i])})"
            findings.append(
                Finding(id=int(ids[i]), row=ctx.row(i), detail=f"id={child}: cleared {role} id={parent}, {reason}")
            )
            ctx.cleared_links.append((child, role, parent, reason))
        if (own | late).any():
            ctx.clear_parents(role, np.flatnonzero(own | late))
    return _verdict(findings)


def _parent_refs_sex_conflict(ctx: ValidationContext) -> CheckResult:
    """One Finding per missing parent ID referenced as both mother and father."""
    if ctx.config.no_sex_check:
        return CheckResult(CheckStatus.SKIP, note="bypassed via --no-sex-check")
    _, mothers, fathers = ctx.require_id_parents()
    moms_missing = ctx.mother_refs.ids[ctx.mother_refs.rows == -1]
    dads_missing = ctx.father_refs.ids[ctx.father_refs.rows == -1]
    return _verdict(
        [
            Finding(
                id=int(c),
                detail=(
                    f"id={ctx.label(c)} referenced as both mother ({rows_phrase(ctx.rows(np.where(mothers == c)[0]))}) "
                    f"and father ({rows_phrase(ctx.rows(np.where(fathers == c)[0]))}) (sex ambiguous)"
                ),
            )
            for c in np.intersect1d(moms_missing, dads_missing)
        ]
    )


def _sex_role_ambiguity(ctx: ValidationContext) -> CheckResult:
    """One Finding per unsexed row referenced as BOTH mother and father.

    A hard block by default: a data contradiction, not just missing
    information. ``--allow-missing-sex`` downgrades it to a SKIP with a
    tolerated count, and the fixed validate output writes these rows' sex as
    unknown so the on-disk pedigree is self-consistent.

    The first row referencing each ambiguous id as mother / father is found on
    demand here (rather than precomputed) because ``ambiguous_mask`` is empty on
    every well-formed pedigree — this loop almost never runs.
    """
    imp = ctx.imputation
    if ctx.config.allow_missing_sex:
        n = int(imp.ambiguous_mask.sum())
        return CheckResult(CheckStatus.SKIP, note=f"bypassed via --allow-missing-sex ({n} tolerated)")
    ids, mothers, fathers = ctx.require_id_parents()
    findings: list[Finding] = []
    for row_idx in np.where(imp.ambiguous_mask)[0]:
        rid = int(ids[row_idx])
        m_rows = np.where(mothers == rid)[0]
        f_rows = np.where(fathers == rid)[0]
        mrow = ctx.row(m_rows[0]) if m_rows.size else -1
        frow = ctx.row(f_rows[0]) if f_rows.size else -1
        findings.append(
            Finding(
                id=rid,
                row=ctx.row(row_idx),
                detail=(
                    f"id={ctx.label(rid)} has unknown sex AND is referenced as both "
                    f"mother (row {mrow}) and father (row {frow}); "
                    "sex cannot be imputed"
                ),
            )
        )
    return _verdict(findings)


def _parents_distinct(ctx: ValidationContext) -> CheckResult:
    """One Finding per row whose mother and father are the same known ID."""
    ids, mothers, fathers = ctx.require_id_parents()
    bad = (mothers != -1) & (fathers != -1) & (mothers == fathers)
    return _verdict(
        [
            Finding(
                id=int(ids[i]),
                row=ctx.row(i),
                detail=f"row {ctx.row(i)}: id={ctx.label(ids[i])} has mother == father == {ctx.label(mothers[i])}",
            )
            for i in np.where(bad)[0]
        ]
    )


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


def _self_loops(ctx: ValidationContext) -> CheckResult:
    """One Finding per row listed as its own mother, then per row listed as its own father."""
    ids, mothers, fathers = ctx.require_id_parents()
    return _verdict(
        [
            Finding(
                id=int(ids[i]), row=ctx.row(i), detail=f"row {ctx.row(i)}: id={ctx.label(ids[i])} listed as own {role}"
            )
            for role, parents in (("mother", mothers), ("father", fathers))
            for i in np.where(parents == ids)[0]
        ]
    )


def _sex_role_findings(
    role: str,
    parents: np.ndarray,
    ids: np.ndarray,
    own_rows: np.ndarray,
    sex: np.ndarray,
    expected_sex: int,
    label: Label,
    input_rows: Callable[[np.ndarray], np.ndarray],
) -> list[Finding]:
    """sex_role_consistency Findings for ids used as ``role`` with the wrong sex.

    ``row`` is the individual's own row (where its sex is declared); the detail
    names the row(s) that reference the id in this role so the log pins the
    offending lines.
    """
    expected = "female" if expected_sex == SEX_FEMALE else "male"
    bad = sex[own_rows] != expected_sex
    return [
        Finding(
            id=int(pid),
            row=int(own_row),
            detail=(
                f"id={label(pid)} used as {role} in {rows_phrase(input_rows(np.where(parents == pid)[0]))} "
                f"but sex != {expected}"
            ),
        )
        for pid, own_row in zip(ids[bad], input_rows(own_rows[bad]), strict=True)
    ]


def _sex_role_override(ctx: ValidationContext) -> CheckResult:
    """One Finding per recorded sex overridden from the only parent role the individual has.

    The override is a repair (ADR 0001 0.9, ADR 0006): an asserted M used only
    as a mother becomes F, an asserted F used only as a father becomes M.
    ``--no-override-asserted-sex`` turns it off, and ``sex_role_consistency``
    then flags the contradiction instead.
    """
    if not ctx.config.override_asserted_sex:
        return CheckResult(CheckStatus.SKIP, note="off (--no-override-asserted-sex)")
    imp = ctx.imputation
    ids, mothers, fathers = ctx.require_id_parents()
    assert ctx.sex is not None
    findings = []
    overridden = np.flatnonzero(imp.overridden_mask)
    for role, parents, recorded, now in (("mother", mothers, "M", "F"), ("father", fathers, "F", "M")):
        own = overridden[ctx.sex[overridden] == (SEX_MALE if role == "mother" else SEX_FEMALE)]
        for i, naming in zip(own, _rows_naming(parents, ids[own]), strict=True):
            findings.append(
                Finding(
                    id=int(ids[i]),
                    row=ctx.row(i),
                    detail=(
                        f"id={ctx.label(ids[i])} recorded {recorded} but used only as {role} in "
                        f"{rows_phrase(ctx.rows(naming))}; sex set to {now}"
                    ),
                )
            )
    findings.sort(key=lambda f: f.row or 0)
    return _verdict(findings)


def _rows_naming(parents: np.ndarray, pids: np.ndarray) -> list[np.ndarray]:
    """For each id in ``pids``, the rows whose parent column names it, ascending; one sort in all."""
    if len(pids) == 0:
        return []
    order = np.argsort(parents, kind="stable")
    ordered = parents[order]
    lo = np.searchsorted(ordered, pids, side="left")
    hi = np.searchsorted(ordered, pids, side="right")
    return [order[a:b] for a, b in zip(lo, hi, strict=True)]


def _sex_role_consistency(ctx: ValidationContext) -> CheckResult:
    """IDs used as mother with resolved sex != female, or as father with sex != male.

    Always runs: the override clears asserted-vs-role contradictions only for
    rows used in a SINGLE role; a row asserted M and used as BOTH mother and
    father stays unoverridable and must still FAIL. Rows whose sex started
    unknown are left to ``sex_role_ambiguity``.
    """
    imp = ctx.imputation
    _, mothers, fathers = ctx.require_id_parents()
    findings: list[Finding] = []
    for role, parents, refs, expected_sex in (
        ("mother", mothers, ctx.mother_refs, SEX_FEMALE),
        ("father", fathers, ctx.father_refs, SEX_MALE),
    ):
        present = refs.rows != -1
        keep = present.copy()
        keep[present] = ~imp.original_unknown_mask[refs.rows[present]]
        findings += _sex_role_findings(
            role, parents, refs.ids[keep], refs.rows[keep], imp.imputed_sex, expected_sex, ctx.label, ctx.rows
        )
    return _verdict(findings)


def _acyclic(ctx: ValidationContext) -> CheckResult:
    """One Finding per row in or below a cycle (``depth == -1``; see ``structural_depth``)."""
    ids, _, _ = ctx.require_id_parents()
    return _verdict(
        [
            Finding(
                id=int(ids[i]),
                row=ctx.row(i),
                detail=f"id={ctx.label(ids[i])} could not be topologically ordered (in a cycle)",
            )
            for i in np.where(ctx.depth < 0)[0]
        ]
    )


# ---------------------------------------------------------------------------
# Birth years (optional)
# ---------------------------------------------------------------------------


def _birth_year_dtype(ctx: ValidationContext) -> CheckResult:
    column = ctx.config.birth_year_col
    if column is None:
        return CheckResult(CheckStatus.SKIP, note="no --birth-year-col specified")
    if column not in ctx.df_raw.columns:
        return CheckResult(CheckStatus.SKIP, note=f"column {column!r} not present in file")
    try:
        ctx.birth_year = as_birth_year_col(ctx.df_raw[column], column)
    except PedigreeError as e:
        return _failed(str(e))
    return _verdict([])


def _birth_year_range(ctx: ValidationContext) -> CheckResult:
    """One Finding per known birth year outside ``[--birth-year-min, --birth-year-max]``; ``-1`` is unknown."""
    ids, birth_year = ctx.ids, ctx.birth_year
    assert ids is not None
    assert birth_year is not None
    year_min, year_max = ctx.config.birth_year_min, ctx.config.by_max
    bad = (birth_year != -1) & ((birth_year < year_min) | (birth_year > year_max))
    return _verdict(
        [
            Finding(
                id=int(ids[i]),
                row=ctx.row(i),
                detail=(
                    f"id={ctx.label(ids[i])} birth_year={int(birth_year[i])} is outside the sanity range "
                    f"[{year_min}, {year_max}]"
                ),
            )
            for i in np.where(bad)[0]
        ]
    )


def _parent_ages(
    ctx: ValidationContext,
    checked: Callable[[np.ndarray, np.ndarray], np.ndarray],
    age: Callable[[np.ndarray, np.ndarray], np.ndarray],
    shown: Label,
) -> CheckResult:
    """One Finding per parent-child edge whose parent was younger than ``--min-parent-age`` at the birth.

    ``checked(child_rows, parent_rows)`` picks the edges this Check owns, and
    ``age`` gives the parent's whole years at each of their births, negative
    for a child born before its parent. The default age 0 flags only such a
    child. ``shown(row)`` names a row's birth year or date in the detail.
    """
    min_age = ctx.config.min_parent_age
    ids, mothers, fathers = ctx.require_id_parents()
    findings: list[Finding] = []
    for role, parents in (("mother", mothers), ("father", fathers)):
        parent_rows = ctx.id_index.get_indexer(parents)
        child_rows = np.flatnonzero((parents != -1) & (parent_rows != -1))
        child_rows = child_rows[checked(child_rows, parent_rows[child_rows])]
        ages = age(child_rows, parent_rows[child_rows])
        for i in np.flatnonzero(ages < min_age):
            row_idx = int(child_rows[i])
            parent_idx = int(parent_rows[row_idx])
            child = f"id={ctx.label(ids[row_idx])} {shown(row_idx)}"
            parent = f"{role} (id={ctx.label(parents[row_idx])}) {shown(parent_idx)}"
            detail = f"{child} < {parent}"
            if ages[i] >= 0:
                detail = f"{child} is {int(ages[i])} year(s) after {parent}; --min-parent-age is {min_age}"
            findings.append(Finding(id=int(ids[row_idx]), row=ctx.row(row_idx), detail=detail))
    if findings or not min_age:
        return _verdict(findings)
    return CheckResult(CheckStatus.PASS, note=f"--min-parent-age {min_age}")


def _birth_year_topology(ctx: ValidationContext) -> CheckResult:
    """``--min-parent-age`` on the birth years of each edge whose two birth dates are not both known.

    Edges with an unknown birth year are skipped; ``birth_date_topology``
    checks the edges with both dates known.
    """
    birth_year, birth_date = ctx.birth_year, ctx.birth_date
    assert birth_year is not None

    def checked(child: np.ndarray, parent: np.ndarray) -> np.ndarray:
        both = (birth_year[child] != -1) & (birth_year[parent] != -1)
        if birth_date is None:
            return both
        return both & (np.isnat(birth_date[child]) | np.isnat(birth_date[parent]))

    return _parent_ages(
        ctx,
        checked,
        lambda child, parent: birth_year[child] - birth_year[parent],
        lambda i: f"birth_year={int(birth_year[i])}",
    )


# ---------------------------------------------------------------------------
# Birth dates (optional)
# ---------------------------------------------------------------------------


def _birth_date_dtype(ctx: ValidationContext) -> CheckResult:
    column = ctx.config.birth_date_col
    if column is None:
        return CheckResult(CheckStatus.SKIP, note="no --birth-date-col specified")
    if column not in ctx.df_raw.columns:
        return CheckResult(CheckStatus.SKIP, note=f"column {column!r} not present in file")
    try:
        ctx.birth_date = as_birth_date_col(ctx.df_raw[column], column)
    except PedigreeError as e:
        return _failed(str(e))
    return _verdict([])


def _date_years(birth_date: np.ndarray) -> np.ndarray:
    """The calendar year of each date, ``-1`` for an unknown (``NaT``) one."""
    years = birth_date.astype("datetime64[Y]").astype(np.int64) + 1970
    return np.where(np.isnat(birth_date), -1, years)


def _whole_years(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Completed years from each known ``start`` date to its known ``end``, negative when ``end`` is earlier."""
    months = [d.astype("datetime64[M]") for d in (start, end)]
    # Month (0-11) and day of month as one sortable number per date.
    month_day = [
        (m.astype(np.int64) % 12) * 32 + (d - m.astype("datetime64[D]")).astype(np.int64)
        for d, m in zip((start, end), months, strict=True)
    ]
    years = _date_years(end) - _date_years(start)
    return years - (month_day[1] < month_day[0])


def _birth_date_range(ctx: ValidationContext) -> CheckResult:
    """One Finding per known birth date whose year is outside ``[--birth-year-min, --birth-year-max]``."""
    ids, birth_date = ctx.ids, ctx.birth_date
    assert ids is not None
    assert birth_date is not None
    year_min, year_max = ctx.config.birth_year_min, ctx.config.by_max
    date_year = _date_years(birth_date)
    bad = (date_year != -1) & ((date_year < year_min) | (date_year > year_max))
    return _verdict(
        [
            Finding(
                id=int(ids[i]),
                row=ctx.row(i),
                detail=(
                    f"id={ctx.label(ids[i])} birth_date={birth_date[i]} is outside the sanity range "
                    f"[{year_min}, {year_max}]"
                ),
            )
            for i in np.flatnonzero(bad)
        ]
    )


def _birth_date_year_consistency(ctx: ValidationContext) -> CheckResult:
    """One Finding per row whose known birth date falls outside its known birth year."""
    ids, birth_year, birth_date = ctx.ids, ctx.birth_year, ctx.birth_date
    assert ids is not None
    assert birth_year is not None
    assert birth_date is not None
    date_year = _date_years(birth_date)
    bad = (date_year != -1) & (birth_year != -1) & (date_year != birth_year)
    return _verdict(
        [
            Finding(
                id=int(ids[i]),
                row=ctx.row(i),
                detail=f"id={ctx.label(ids[i])} birth_date={birth_date[i]} but birth_year={int(birth_year[i])}",
            )
            for i in np.flatnonzero(bad)
        ]
    )


def _birth_date_topology(ctx: ValidationContext) -> CheckResult:
    """``--min-parent-age`` on the birth dates of each edge whose two dates are known, to the day."""
    birth_date = ctx.birth_date
    assert birth_date is not None
    return _parent_ages(
        ctx,
        lambda child, parent: ~np.isnat(birth_date[child]) & ~np.isnat(birth_date[parent]),
        lambda child, parent: _whole_years(birth_date[parent], birth_date[child]),
        lambda i: f"birth_date={birth_date[i]}",
    )


# ---------------------------------------------------------------------------
# Multiple births (optional)
# ---------------------------------------------------------------------------


def _zygosity_tokens(ctx: ValidationContext) -> CheckResult:
    column = ctx.config.zygosity_col
    if column is None:
        return CheckResult(CheckStatus.SKIP, note="no --zygosity-col specified")
    try:
        ctx.zygosity = as_zygosity_col(ctx.df_raw[column], column)
    except PedigreeError as e:
        return _failed(str(e))
    return _verdict([])


def _multiple_birth_sets(ctx: ValidationContext) -> CheckResult:
    """One Finding per set member whose set cell ``validate`` clears.

    A member stays when both its parents are known and shared with at least
    one other member, and its set holds no other such group of full sibs
    (which of the two deliveries the id means is then ambiguous). Sets
    ``ctx.multiple_birth`` to the set codes left and
    ``ctx.cleared_multiple_birth`` to the rows cleared.
    """
    column = ctx.config.multiple_birth_col
    if column is None:
        return CheckResult(CheckStatus.SKIP, note="no --multiple-birth-col specified")
    ids, mothers, fathers = ctx.require_id_parents()
    sets = as_group_codes(ctx.df_raw[column])
    rows = np.flatnonzero(sets >= 0)
    members = pl.DataFrame({"row": rows, "set": sets[rows], "m": mothers[rows], "f": fathers[rows]}).with_columns(
        known=(pl.col("m") != -1) & (pl.col("f") != -1),
        n_set=pl.len().over("set"),
        n_sibs=pl.len().over("set", "m", "f"),
    )
    members = members.with_columns(full_sibs=pl.col("known") & (pl.col("n_sibs") >= 2))
    n_sibships = members.filter("full_sibs").group_by("set").agg(n_sibships=pl.struct("m", "f").n_unique())
    members = members.join(n_sibships, on="set", how="left").with_columns(pl.col("n_sibships").fill_null(0))
    reason = (
        pl.when(pl.col("n_set") == 1)
        .then(pl.lit("it is the only member of its set"))
        .when(~pl.col("known"))
        .then(pl.lit("a parent is unknown"))
        .when(pl.col("n_sibs") < 2)
        .then(pl.lit("no other member of its set has the same mother and father"))
        .when(pl.col("n_sibships") > 1)
        .then(pl.lit("its set holds more than one group of full sibs"))
    )
    cleared = members.with_columns(reason=reason).filter(pl.col("reason").is_not_null())
    tokens = ctx.df_raw[column].cast(pl.String).str.strip_chars()
    mask = np.zeros(len(sets), dtype=bool)
    mask[cleared["row"].to_numpy()] = True
    ctx.multiple_birth = np.where(mask, -1, sets)
    ctx.cleared_multiple_birth = mask
    return _verdict(
        [
            Finding(
                id=int(ids[i]),
                row=ctx.row(i),
                detail=f"id={ctx.label(ids[i])} multiple-birth set {tokens[i]!r} cleared: {why}",
            )
            for i, why in cleared.select("row", "reason").sort("row").iter_rows()
        ]
    )


def _mz_links(ctx: ValidationContext) -> CheckResult:
    """One Finding per zygosity mark ``validate`` clears; the rest link MZ co-twins.

    The two MZ members of a set become co-twins (``ctx.co_twin``, as ID
    codes). A mark is cleared when its row is in no set, when it is the only
    MZ in its set, when its set has three or more (pedigree-graph links MZ
    co-twins in pairs), or when the two co-twins have known, different sex.
    ``ctx.cleared_zygosity`` records the rows cleared.
    """
    ids = ctx.ids
    sets, zygosity = ctx.multiple_birth, ctx.zygosity
    assert ids is not None
    # --zygosity-col needs --multiple-birth-col, so multiple_birth_sets ran.
    assert sets is not None
    assert zygosity is not None
    sex = ctx.imputation.imputed_sex
    reasons: dict[int, str] = {}
    for i in np.flatnonzero((zygosity != ZYGOSITY_UNKNOWN) & (sets < 0)):
        reasons[int(i)] = "it is in no multiple-birth set"
    co_twin = np.full(len(ids), -1, dtype=np.int64)
    mz = np.flatnonzero((zygosity == ZYGOSITY_MZ) & (sets >= 0))
    mz = mz[np.argsort(sets[mz], kind="stable")]
    _, starts, counts = np.unique(sets[mz], return_index=True, return_counts=True)
    for start, count in zip(starts.tolist(), counts.tolist(), strict=True):
        group = mz[start : start + count]
        if count == 1:
            reasons[int(group[0])] = "it is the only MZ member of its set"
        elif count > 2:
            for i in group:
                reasons[int(i)] = f"its set has {count} MZ members; MZ co-twins are linked in pairs"
        else:
            a, b = int(group[0]), int(group[1])
            if SEX_UNKNOWN not in (sex[a], sex[b]) and sex[a] != sex[b]:
                reasons[a] = f"its MZ co-twin id={ctx.label(ids[b])} has the opposite sex"
                reasons[b] = f"its MZ co-twin id={ctx.label(ids[a])} has the opposite sex"
            else:
                co_twin[a], co_twin[b] = ids[b], ids[a]
    mask = np.zeros(len(ids), dtype=bool)
    mask[list(reasons)] = True
    ctx.co_twin = co_twin
    ctx.cleared_zygosity = mask
    return _verdict(
        [
            Finding(id=int(ids[i]), row=ctx.row(i), detail=f"id={ctx.label(ids[i])} zygosity cleared: {why}")
            for i, why in sorted(reasons.items())
        ]
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class FailPolicy(Enum):
    """What a Check's ``FAIL`` does to the pedigree ``validate`` writes (ADR 0005).

    The policy decides both modes: whether plain ``validate`` BLOCKs or writes
    the pedigree as-is, and whether ``--drop-offending`` can resolve the FAIL
    by removing the flagged individual(s).
    """

    # No row removal fixes it (missing column, parse failure, negative id):
    # BLOCKs in both modes.
    BLOCK = "block"
    # BLOCKs plain validate; --drop-offending removes the flagged ids.
    BLOCK_OR_DROP = "block_or_drop"
    # Plain validate writes the pedigree with the contradiction kept;
    # --drop-offending removes the flagged ids.
    KEEP_OR_DROP = "keep_or_drop"
    # Repaired in the Fixed Pedigree (a founder synthesized, a duplicate row
    # collapsed, a cycle link cleared, a sex overridden from role); never
    # BLOCKs and is never dropped.
    AUTOFIX = "autofix"


@dataclass(frozen=True)
class Check:
    """One registry entry: a named integrity test, its prerequisites, display label and Fail Policy."""

    name: str
    requires: tuple[str, ...]
    run: Callable[[ValidationContext], CheckResult]
    label: str
    fail_policy: FailPolicy
    # The BLOCKED line plain validate prints for this Check, with any flag that
    # tolerates it. Defaults to the Check name.
    block_message: str | None = None


# Prerequisite shorthand: id_dtype parses the id and both parent columns.
_IDS = ("id_dtype", "negative_ids", "duplicate_ids")

CHECKS: tuple[Check, ...] = (
    Check(
        "required_columns",
        (),
        _required_columns,
        "required columns present",
        FailPolicy.BLOCK,
    ),
    Check(
        "empty_pedigree",
        ("required_columns",),
        _empty_pedigree,
        "pedigree is non-empty",
        FailPolicy.BLOCK,
    ),
    Check(
        "id_dtype",
        ("required_columns",),
        _id_dtype,
        "every row has an id",
        FailPolicy.BLOCK,
    ),
    Check(
        "sex_tokens",
        ("required_columns",),
        _sex_tokens,
        "sex column tokens recognized",
        FailPolicy.BLOCK,
    ),
    Check(
        "birth_year_dtype",
        ("required_columns",),
        _birth_year_dtype,
        "birth_year column parses as numeric",
        FailPolicy.BLOCK,
    ),
    Check(
        "birth_date_dtype",
        ("required_columns",),
        _birth_date_dtype,
        "birth_date column parses as ISO dates",
        FailPolicy.BLOCK,
    ),
    Check(
        "zygosity_tokens",
        ("required_columns",),
        _zygosity_tokens,
        "zygosity column holds MZ / DZ",
        FailPolicy.BLOCK,
    ),
    Check(
        # Before every Check that reads the parsed arrays: it removes the copies.
        "duplicate_rows",
        ("id_dtype", "sex_tokens"),
        _duplicate_rows,
        "no duplicate rows",
        FailPolicy.AUTOFIX,
    ),
    Check(
        "negative_ids",
        ("id_dtype",),
        _negative_ids,
        "no negative IDs",
        FailPolicy.BLOCK,
    ),
    Check(
        "duplicate_ids",
        ("id_dtype",),
        _duplicate_ids,
        "no duplicate IDs",
        FailPolicy.BLOCK_OR_DROP,
        block_message="duplicate IDs detected",
    ),
    Check(
        "parent_token_range_mother",
        ("id_dtype",),
        partial(_parent_token_range, role="mother"),
        "mother IDs in valid range",
        FailPolicy.BLOCK,
    ),
    Check(
        "parent_token_range_father",
        ("id_dtype",),
        partial(_parent_token_range, role="father"),
        "father IDs in valid range",
        FailPolicy.BLOCK,
    ),
    Check(
        "zero_parent_present",
        _IDS,
        _zero_parent_present,
        "parent ID 0 has a row",
        FailPolicy.BLOCK,
        block_message=(
            "parent ID 0 has no row; if the file writes a missing parent as 0, pass --missing-parent-token 0"
        ),
    ),
    Check(
        # Before every Check that reads the parent refs, depth or sex
        # imputation: it clears parent links.
        "cycle_breaks",
        (*_IDS, "birth_year_dtype"),
        _cycle_breaks,
        "no cycle links to clear",
        FailPolicy.AUTOFIX,
    ),
    Check(
        "parent_refs_present_mother",
        _IDS,
        partial(_parent_refs_present, role="mother"),
        "mother IDs present in pedigree",
        FailPolicy.AUTOFIX,
    ),
    Check(
        "parent_refs_present_father",
        _IDS,
        partial(_parent_refs_present, role="father"),
        "father IDs present in pedigree",
        FailPolicy.AUTOFIX,
    ),
    Check(
        # Kept in plain validate; --drop-offending drops the parent, which
        # clears its children's links and keeps the children.
        "parent_child_count",
        _IDS,
        _parent_child_count,
        "children per parent within limits",
        FailPolicy.KEEP_OR_DROP,
    ),
    Check(
        "parent_refs_sex_conflict",
        _IDS,
        _parent_refs_sex_conflict,
        "no missing-parent sex conflicts",
        FailPolicy.BLOCK_OR_DROP,
        block_message="sex conflict on missing parent(s); pass --no-sex-check to default to sex=F",
    ),
    Check(
        "sex_role_ambiguity",
        (*_IDS, "sex_tokens"),
        _sex_role_ambiguity,
        "no role-ambiguous unsexed individuals",
        FailPolicy.BLOCK_OR_DROP,
        block_message=(
            "present individual(s) with unknown sex used as BOTH mother and father "
            "(sex cannot be imputed); pass --allow-missing-sex to tolerate"
        ),
    ),
    Check(
        "self_loops",
        _IDS,
        _self_loops,
        "no self-loops",
        FailPolicy.KEEP_OR_DROP,
    ),
    Check(
        "parents_distinct",
        _IDS,
        _parents_distinct,
        "mother and father distinct",
        FailPolicy.BLOCK_OR_DROP,
        block_message="rows with mother == father (cannot disambiguate)",
    ),
    Check(
        "sex_role_override",
        (*_IDS, "sex_tokens"),
        _sex_role_override,
        "no recorded sex overridden by role",
        FailPolicy.AUTOFIX,
    ),
    Check(
        "sex_role_consistency",
        (*_IDS, "sex_tokens"),
        _sex_role_consistency,
        "sex consistent with parent role",
        FailPolicy.KEEP_OR_DROP,
    ),
    Check(
        "acyclic",
        # Not parent_refs_present_*: a parent with no row counts as a founder
        # for depth, so a cycle is found (and BLOCKs) even when a parent is
        # missing (pedsum#22).
        (*_IDS, "parents_distinct"),
        _acyclic,
        "acyclic (no descent cycles)",
        FailPolicy.BLOCK_OR_DROP,
        block_message="cycle detected",
    ),
    Check(
        "birth_year_range",
        ("id_dtype", "birth_year_dtype"),
        _birth_year_range,
        "birth years within sanity range",
        FailPolicy.KEEP_OR_DROP,
    ),
    Check(
        "birth_year_topology",
        (*_IDS, "birth_year_dtype"),
        _birth_year_topology,
        "parent age at birth >= min parent age",
        FailPolicy.KEEP_OR_DROP,
    ),
    Check(
        "birth_date_range",
        ("id_dtype", "birth_date_dtype"),
        _birth_date_range,
        "birth dates within sanity range",
        FailPolicy.KEEP_OR_DROP,
    ),
    Check(
        "birth_date_year_consistency",
        ("id_dtype", "birth_date_dtype", "birth_year_dtype"),
        _birth_date_year_consistency,
        "birth dates fall in their birth year",
        FailPolicy.KEEP_OR_DROP,
    ),
    Check(
        "birth_date_topology",
        (*_IDS, "birth_date_dtype"),
        _birth_date_topology,
        "parent age at birth, from birth dates",
        FailPolicy.KEEP_OR_DROP,
    ),
    Check(
        # After cycle_breaks: it reads the parents left.
        "multiple_birth_sets",
        _IDS,
        _multiple_birth_sets,
        "multiple-birth sets are full sibs",
        FailPolicy.AUTOFIX,
    ),
    Check(
        # Not after multiple_birth_sets: its FAIL is an autofix that still
        # leaves the set codes, so the links form in the same pass.
        "mz_links",
        (*_IDS, "sex_tokens", "zygosity_tokens"),
        _mz_links,
        "MZ marks form co-twin pairs",
        FailPolicy.AUTOFIX,
    ),
)

# Derived from the registry (single source of truth).
_CHECK_ORDER: tuple[str, ...] = tuple(c.name for c in CHECKS)
CHECK_LABELS: dict[str, str] = {c.name: c.label for c in CHECKS}

# Checks `validate --drop-offending` can resolve by removing the flagged
# individual(s). Derived from the registry so the set never drifts.
DROPPABLE_CHECKS: frozenset[str] = frozenset(
    c.name for c in CHECKS if c.fail_policy in (FailPolicy.BLOCK_OR_DROP, FailPolicy.KEEP_OR_DROP)
)


def blocking_failures(results: Mapping[str, CheckResult], *, drop_offending: bool) -> list[tuple[str, str]]:
    """The FAILed Checks that stop ``validate`` writing a pedigree, as ``(name, message)``.

    In registry order. Under ``--drop-offending`` only ``FailPolicy.BLOCK``
    Checks block (the rest are removed); in plain ``validate``
    ``BLOCK_OR_DROP`` Checks block too. The message is the Check's
    ``block_message``, else its name.
    """
    blocking = {FailPolicy.BLOCK} if drop_offending else {FailPolicy.BLOCK, FailPolicy.BLOCK_OR_DROP}
    return [
        (c.name, c.block_message or c.name)
        for c in CHECKS
        if c.fail_policy in blocking and c.name in results and results[c.name].status == CheckStatus.FAIL
    ]


# Curated *display* grouping for the stderr summary. Order within a group
# intentionally differs from execution order (e.g. parents_distinct), so this
# stays an explicit constant rather than a derivation of _CHECK_ORDER.
CHECK_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Columns & parsing",
        (
            "required_columns",
            "empty_pedigree",
            "id_dtype",
            "sex_tokens",
        ),
    ),
    ("IDs", ("negative_ids", "duplicate_rows", "duplicate_ids")),
    (
        "Parent references",
        (
            "parent_token_range_mother",
            "parent_token_range_father",
            "zero_parent_present",
            "parent_refs_present_mother",
            "parent_refs_present_father",
            "parent_child_count",
            "parents_distinct",
            "parent_refs_sex_conflict",
            "sex_role_ambiguity",
        ),
    ),
    (
        "Graph structure",
        (
            "cycle_breaks",
            "self_loops",
            "sex_role_override",
            "sex_role_consistency",
            "acyclic",
        ),
    ),
    (
        "Birth years (optional)",
        (
            "birth_year_dtype",
            "birth_year_range",
            "birth_year_topology",
        ),
    ),
    (
        "Birth dates (optional)",
        (
            "birth_date_dtype",
            "birth_date_range",
            "birth_date_year_consistency",
            "birth_date_topology",
        ),
    ),
    (
        "Multiple births (optional)",
        (
            "multiple_birth_sets",
            "zygosity_tokens",
            "mz_links",
        ),
    ),
)
