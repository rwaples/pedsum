"""Validation: one dual-mode runner over the Check registry in ``pedsum.checks``.

`load_and_validate` (summarize, fail-fast) and `validate_pedigree` (validate,
accumulating) share the `CHECKS` registry and the `run_checks` runner — see
`docs/adr/0002-validation-check-registry.md`. This module also holds the
validation config and context and sex imputation; `pedsum.reduce` decides what
`validate` writes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, PedigreeError, SexSource, logger
from pedsum.checks import CHECKS, CheckResult, CheckStatus, summarize_findings
from pedsum.fixed_pedigree import fix_pedigree
from pedsum.parse import (
    BIRTH_YEAR_DEFAULT_MIN,
    birth_year_default_max,
    maybe_warn_csv,
    read_pedigree_table,
)
from pedsum.pedigree_ops import IdIndex, ParentRefs, rows_of_parents, structural_depth

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable
    from pathlib import Path

    from pedsum.checks import Finding
    from pedsum.ids import IdLabels


# ---------------------------------------------------------------------------
# Sex imputation
# ---------------------------------------------------------------------------


@dataclass
class _SexImputation:
    """Result of imputing unknown sex from mother / father role usage."""

    imputed_sex: np.ndarray
    n_imputed: int
    ambiguous_mask: np.ndarray
    original_unknown_mask: np.ndarray
    overridden_mask: np.ndarray
    sex_source: np.ndarray  # dtype=object; per-row category string


def _impute_sex_from_roles(
    sex: np.ndarray,
    ids: np.ndarray,
    mother_ids: np.ndarray,
    father_ids: np.ndarray,
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
         - id used as neither     → leave -1; kept unsexed (ADR 0006)

      2. Asserted → role (0.9 default; opt out via
         ``override_asserted_sex=False``):
         - asserted M used only as mother → override to F
         - asserted F used only as father → override to M
         - asserted used as both          → leave alone; ``sex_role_consistency``
                                            flags it
         - asserted used as neither       → leave alone

    ``mother_ids`` / ``father_ids`` are the distinct ids referenced as mother
    / father (``ParentRefs.ids``).

    Returns a `_SexImputation` whose `sex_source` is a per-row `dtype=object`
    array tagging each row's provenance with a ``SexSource`` value (plain
    strings, which polars converts fastest).
    """
    original_unknown_mask = (sex == SEX_UNKNOWN).copy()
    imputed = sex.copy()
    sex_source = np.full(len(sex), SexSource.INPUT.value, dtype=object)

    # Membership by id, not by ParentRefs.rows: with duplicate ids every copy
    # of a referenced id takes the role, not just the first row.
    as_mother = np.isin(ids, mother_ids)
    as_father = np.isin(ids, father_ids)

    # Pass 1: missing → role.
    impute_f = original_unknown_mask & as_mother & ~as_father
    impute_m = original_unknown_mask & as_father & ~as_mother
    ambiguous_mask = original_unknown_mask & as_mother & as_father
    imputed[impute_f] = SEX_FEMALE
    imputed[impute_m] = SEX_MALE
    imputed_from_missing = impute_f | impute_m
    sex_source[imputed_from_missing] = SexSource.IMPUTED_FROM_MISSING.value
    n_imputed = int(imputed_from_missing.sum())

    # Pass 2: asserted → role (when topology is unambiguous).
    override_to_f = (sex == SEX_MALE) & as_mother & ~as_father
    override_to_m = (sex == SEX_FEMALE) & as_father & ~as_mother
    if override_asserted_sex:
        imputed[override_to_f] = SEX_FEMALE
        imputed[override_to_m] = SEX_MALE
        overridden_mask = override_to_f | override_to_m
        sex_source[overridden_mask] = SexSource.IMPUTED_FROM_ROLE.value
    else:
        overridden_mask = np.zeros(len(sex), dtype=bool)

    # Final pass: anything still SEX_UNKNOWN is unresolved.
    sex_source[imputed == SEX_UNKNOWN] = SexSource.UNRESOLVED.value

    return _SexImputation(
        imputed_sex=imputed,
        n_imputed=n_imputed,
        ambiguous_mask=ambiguous_mask,
        original_unknown_mask=original_unknown_mask,
        overridden_mask=overridden_mask,
        sex_source=sex_source,
    )


# ---------------------------------------------------------------------------
# ValidationConfig, ValidationContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationConfig:
    """The column names, tolerances and birth-year bounds a validation run takes from the CLI."""

    id_col: str = "id"
    sex_col: str = "sex"
    mother_col: str = "mother"
    father_col: str = "father"
    birth_year_col: str | None = None
    birth_year_min: int = BIRTH_YEAR_DEFAULT_MIN
    birth_year_max: int | None = None
    # --min-parent-age: birth_year_topology wants child - parent >= this.
    min_parent_age: int = 0
    # --missing-parent-token: more tokens that mean a missing parent in the
    # mother and father columns (ADR 0006).
    missing_parent_tokens: tuple[str, ...] = ()
    allow_missing_sex: bool = False
    override_asserted_sex: bool = True
    # --break-cycles: clear self-links and, within a cycle, links to a parent
    # born after the child (cycle_breaks, ADR 0006).
    break_cycles: bool = False
    # --no-sex-check tolerates parent_refs_sex_conflict (SKIP). It lives here,
    # not as a CLI post-filter, so the tolerance composes wherever run_checks
    # runs: validate, the drop-offending loop, and its self-verify.
    no_sex_check: bool = False
    # --birth-date-col: an ISO birth date per row (birth_date_dtype).
    birth_date_col: str | None = None
    # --twin-max-days: full sibs born at most this many days apart form a
    # multiple-birth set, written by the Fixed Pedigree. None leaves it off.
    twin_max_days: int | None = None
    # --multiple-birth-col / --zygosity-col: given sets, and MZ / DZ marks
    # within them (multiple_birth_sets, zygosity_tokens, mz_links).
    multiple_birth_col: str | None = None
    zygosity_col: str | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ValidationConfig:
        """The config of a parsed command line; only ``validate`` has ``--no-sex-check``."""
        return cls(
            id_col=args.id_col,
            sex_col=args.sex_col,
            mother_col=args.mother_col,
            father_col=args.father_col,
            birth_year_col=args.birth_year_col,
            birth_year_min=args.birth_year_min,
            birth_year_max=args.birth_year_max,
            min_parent_age=args.min_parent_age,
            # Cells are matched after stripping, so the tokens are too.
            missing_parent_tokens=tuple(t.strip() for t in args.missing_parent_token),
            allow_missing_sex=args.allow_missing_sex,
            override_asserted_sex=not args.no_override_asserted_sex,
            no_sex_check=getattr(args, "no_sex_check", False),
            break_cycles=getattr(args, "break_cycles", False),
            birth_date_col=getattr(args, "birth_date_col", None),
            twin_max_days=getattr(args, "twin_max_days", None),
            multiple_birth_col=getattr(args, "multiple_birth_col", None),
            zygosity_col=getattr(args, "zygosity_col", None),
        )

    @property
    def by_max(self) -> int:
        """Resolved birth-year upper bound (default: current calendar year + 1)."""
        return self.birth_year_max if self.birth_year_max is not None else birth_year_default_max()


@dataclass
class ValidationContext:
    """Mutable state threaded through every Check (``pedsum.checks``).

    Parse Checks populate the array fields (parsing *is* the Check); all other
    Checks read them. ``id_index`` and ``imputation`` are derived once on first
    access.
    """

    df_raw: pl.DataFrame
    config: ValidationConfig
    # summarize's run raises on the first FAIL; otherwise every FAIL is recorded.
    fail_fast: bool = False
    # summarize and its validate.log rerun require a named --birth-year-col to
    # exist; validate treats a missing one as an inapplicable SKIP.
    require_birth_year_col: bool = False
    # The --drop-offending rounds parse with the input's labels, so the ID
    # mode and string-mode codes stay fixed while rows are dropped.
    fixed_labels: IdLabels | None = None
    # Populated by the parse Checks.
    ids: np.ndarray | None = None
    mothers: np.ndarray | None = None
    fathers: np.ndarray | None = None
    id_labels: IdLabels | None = None
    sex: np.ndarray | None = None
    birth_year: np.ndarray | None = None
    birth_date: np.ndarray | None = None  # datetime64[D], NaT unknown
    zygosity: np.ndarray | None = None  # int8 ZYGOSITY_* codes
    # Set by multiple_birth_sets: each row's set code (-1 none) after the
    # members it clears, and which rows' set cells it cleared.
    multiple_birth: np.ndarray | None = None
    cleared_multiple_birth: np.ndarray | None = None
    # Set by mz_links: each row's MZ co-twin ID code (-1 none), and which
    # rows' zygosity cells it cleared.
    co_twin: np.ndarray | None = None
    cleared_zygosity: np.ndarray | None = None
    # The parent links cycle_breaks cleared, as (id, role, parent, reason)
    # with IDs as the input wrote them: validate's cleared-links manifest.
    cleared_links: list[tuple[str, str, str, str]] = field(default_factory=list)
    # Rows read from the input, before duplicate_rows collapsed any copies.
    n_input_rows: int = field(init=False)
    # Set when duplicate_rows collapsed copies: each remaining row's position
    # in the input, so Findings name the rows the user can find in the file.
    input_rows: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Count the input's rows before any Check changes ``df_raw``."""
        self.n_input_rows = len(self.df_raw)

    def row(self, i: int | np.integer) -> int:
        """Row ``i`` of the parsed arrays as its row in the input."""
        return int(i) if self.input_rows is None else int(self.input_rows[i])

    def rows(self, idx: np.ndarray) -> np.ndarray:
        """Rows ``idx`` of the parsed arrays as their rows in the input."""
        return idx if self.input_rows is None else self.input_rows[idx]

    def clear_parents(self, role: str, rows: np.ndarray) -> None:
        """Make the ``role`` parent of ``rows`` missing, in the parsed arrays and the raw frame.

        Like :meth:`keep_rows`, runs before the derived lookups are read.
        """
        column = self.config.mother_col if role == "mother" else self.config.father_col
        _, mothers, fathers = self.require_id_parents()
        parents = (mothers if role == "mother" else fathers).copy()
        parents[rows] = -1
        if role == "mother":
            self.mothers = parents
        else:
            self.fathers = parents
        cleared = np.zeros(len(parents), dtype=bool)
        cleared[rows] = True
        self.df_raw = self.df_raw.with_columns(
            pl.when(pl.Series(cleared)).then(None).otherwise(pl.col(column)).alias(column)
        )
        for name in ("mother_refs", "father_refs", "depth", "imputation"):
            self.__dict__.pop(name, None)

    def keep_rows(self, keep: np.ndarray) -> None:
        """Keep only the rows where ``keep`` is True, in the raw frame and every parsed array.

        Runs before any Check reads the derived lookups (``id_index``, the parent
        refs, ``depth``, ``imputation``); any already built is dropped.
        """
        self.df_raw = self.df_raw.filter(pl.Series(keep))
        for name in ("ids", "mothers", "fathers", "sex", "birth_year", "birth_date", "zygosity"):
            values = getattr(self, name)
            if values is not None:
                setattr(self, name, values[keep])
        kept = np.flatnonzero(keep)
        self.input_rows = kept if self.input_rows is None else self.input_rows[kept]
        for name in ("sorted_ids", "id_index", "mother_refs", "father_refs", "depth", "imputation"):
            self.__dict__.pop(name, None)

    @property
    def label(self) -> Callable[[int], str]:
        """Prints an id code as the input's ID; set by the id parse Check."""
        assert self.id_labels is not None
        return self.id_labels.label

    @cached_property
    def sorted_ids(self) -> np.ndarray:
        """The parsed IDs sorted; one sort shared by the duplicate Checks."""
        assert self.ids is not None
        return np.sort(self.ids)

    @cached_property
    def id_index(self) -> IdIndex:
        """Row lookup over the parsed IDs; built once (IDs are fixed after parse)."""
        return IdIndex(self.ids)

    @cached_property
    def mother_refs(self) -> ParentRefs:
        """Distinct mother ids and their rows; built once, shared by the parent and sex checks."""
        assert self.mothers is not None
        return ParentRefs.of(self.mothers, self.id_index)

    @cached_property
    def father_refs(self) -> ParentRefs:
        """Distinct father ids and their rows; built once, shared by the parent and sex checks."""
        assert self.fathers is not None
        return ParentRefs.of(self.fathers, self.id_index)

    @cached_property
    def depth(self) -> np.ndarray:
        """Topological depth per parsed row, -1 in or below a cycle; built once."""
        assert self.mothers is not None
        assert self.fathers is not None
        m_row, _ = rows_of_parents(self.mothers, self.id_index)
        f_row, _ = rows_of_parents(self.fathers, self.id_index)
        return structural_depth(m_row, f_row)

    @cached_property
    def imputation(self) -> _SexImputation:
        """Sex imputation from parent-role usage; computed once, shared by the sex checks."""
        assert self.sex is not None
        assert self.ids is not None
        return _impute_sex_from_roles(
            self.sex,
            self.ids,
            self.mother_refs.ids,
            self.father_refs.ids,
            override_asserted_sex=self.config.override_asserted_sex,
        )

    def require_id_parents(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the parsed ``(ids, mothers, fathers)``.

        Asserts the id/parent parse Checks populated them — guaranteed for any
        consuming Check by the ``requires`` gating in :func:`run_checks`. One
        guarded accessor for the Checks that need the full pedigree.
        """
        assert self.ids is not None
        assert self.mothers is not None
        assert self.fathers is not None
        return self.ids, self.mothers, self.fathers


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_checks(ctx: ValidationContext) -> tuple[dict[str, CheckResult], list[Finding]]:
    """Run every Check in registry order against ``ctx``.

    A Check whose prerequisite did not ``PASS`` is auto-``SKIP``ped: a FAILed
    prerequisite yields ``"<prereq> failed"``, a SKIPped one propagates its own
    reason (so ``required_columns failed`` / ``no --birth-year-col specified``
    flow transitively). A ``fail_fast`` context raises on the first ``FAIL``
    (summarize); otherwise each is recorded and the run continues (validate).
    """
    results: dict[str, CheckResult] = {}
    findings: list[Finding] = []
    for check in CHECKS:
        skip_note = None
        for prereq in check.requires:
            r = results[prereq]
            if r.status != CheckStatus.PASS:
                skip_note = r.note if r.status == CheckStatus.SKIP else f"{prereq} failed"
                break
        result = check.run(ctx) if skip_note is None else CheckResult(CheckStatus.SKIP, note=skip_note)
        result.name = check.name
        for f in result.findings:
            f.check = check.name
        if result.status == CheckStatus.FAIL:
            if ctx.fail_fast:
                # The id parse Check runs before any Check that names an id.
                label = ctx.label if ctx.id_labels is not None else str
                raise PedigreeError(summarize_findings(result.findings, label))
            findings.extend(result.findings)
        results[check.name] = result
    return results, findings


# ---------------------------------------------------------------------------
# Public drivers
# ---------------------------------------------------------------------------


def load_and_validate(path: Path, config: ValidationConfig, *, sep: str = "auto") -> tuple[pl.DataFrame, IdLabels]:
    """Load TSV, run all QC fail-fast, return the validated pedigree and its ID labels.

    The frame's id, mother and father columns hold int64 ID codes, which the
    labels print as the input's IDs (:mod:`pedsum.ids`); in integer mode a
    code is the ID. The frame has columns id, sex (int8), mother, father, sex_source,
    ``birth_year`` (int32, sentinel -1) when ``birth_year_col`` is set and
    present, and ``ped_depth`` (int32 topological depth, founders 0). Missing
    parent encoded as -1. Raises ``PedigreeError`` on the first failing Check.

    Rows are topologically sorted (parents before children). pedigree-graph
    accepts rows in any order, so the sort is pedsum's own output contract, not
    a construction precondition. The frame is the Fixed Pedigree's
    (:mod:`pedsum.fixed_pedigree`): the same rows, sex and order ``validate``
    writes for this input.
    """
    t0 = time.perf_counter()
    df = read_pedigree_table(path, sep=sep, dtype=str)
    logger.info("read %d rows from %s in %.2fs", len(df), path, time.perf_counter() - t0)
    maybe_warn_csv(df)
    ctx = ValidationContext(df, config, fail_fast=True, require_birth_year_col=True)
    run_checks(ctx)
    # Every Check passed, so no parent is missing and there is no founder to
    # synthesize: the frame is the input rows of the pedigree validate would write.
    fp = fix_pedigree(ctx, synthesize_founders=False)
    logger.info("validated %d rows in %.2fs", len(fp.ids), time.perf_counter() - t0)
    return fp.frame(), fp.labels


def validate_pedigree(
    path: Path, config: ValidationConfig, *, sep: str = "auto", require_birth_year_col: bool = False
) -> tuple[int, list[CheckResult], list[Finding], ValidationContext]:
    """Run every Check accumulating.

    Returns ``(n_rows, results, findings, ctx)`` where ``results`` covers every
    Check in ``CHECKS`` order (each PASS / FAIL / SKIP), ``findings`` lists
    every per-individual finding, and ``ctx`` carries the coerced arrays and raw
    DataFrame for :func:`pedsum.reduce.fix_validated`. ``require_birth_year_col``
    makes a missing ``birth_year_col`` a ``required_columns`` FAIL, as in
    :func:`load_and_validate`.
    """
    df = read_pedigree_table(path, sep=sep, dtype=str)
    maybe_warn_csv(df)
    ctx = ValidationContext(df, config, require_birth_year_col=require_birth_year_col)
    results, findings = run_checks(ctx)
    return ctx.n_input_rows, list(results.values()), findings, ctx
