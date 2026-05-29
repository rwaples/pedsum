"""Load + validate drivers (fail-fast and accumulating) and sex imputation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pedsum.base import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, PedigreeError, logger
from pedsum.checks import (
    _CHECK_ORDER,
    CheckResult,
    Finding,
    _check_acyclic,
    _check_birth_year_range,
    _check_birth_year_topology,
    _check_duplicate_ids,
    _check_empty_pedigree,
    _check_negative_ids,
    _check_parent_refs_present,
    _check_parent_refs_sex_conflict,
    _check_parent_token_range,
    _check_parents_distinct,
    _check_self_loops,
    _check_sex_role_ambiguity,
    _check_sex_role_consistency,
    _check_unknown_sex,
    _summarize_findings,
)
from pedsum.parse import (
    _BIRTH_YEAR_DEFAULT_MIN,
    _as_birth_year_col,
    _as_int_col,
    _as_parent_int_col,
    _birth_year_default_max,
    _decode_sex,
    _format_id_sample,
    _maybe_warn_csv,
    _read_pedigree_table,
)
from pedsum.pedigree_ops import _build_children_csr, _compute_depth_unordered, _parent_rows

if TYPE_CHECKING:
    from pathlib import Path

    import scipy.sparse as sp


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
