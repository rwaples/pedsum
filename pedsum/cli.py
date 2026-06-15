"""Argument parsing and the summarize / validate command runners."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pedsum.base import _F_KERNEL_WARN_THRESHOLD, SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, VERSION, PedigreeError, logger
from pedsum.pairs import _augment_pair_counts, _build_pedigree_graph, _count_pairs_matrix_with_lists
from pedsum.parse import _BIRTH_YEAR_DEFAULT_MIN, _SEP_CHOICES
from pedsum.pedigree_ops import _compute_depth_unordered, _parent_rows
from pedsum.report import (
    SAFE_MIN_CELL,
    _apply_safe_attempt,
    _build_added_founders,
    _build_individual_data,
    _build_pedigree_data,
    _build_summary_data,
    _format_check_summary,
    _prepare_out_dir,
    _write_annotated_tsv,
    _write_dropped_manifest,
    _write_long_tsv,
    _write_validate_log,
    _write_validate_tsv_gz,
    _write_yaml,
)
from pedsum.sections import (
    _build_inbreeding_summary,
    build_individual_df,
    compute_aggregate_sections,
    compute_effective_size,
    compute_founder_summary,
    compute_mating_pair_summary,
    compute_relationship_summary,
    compute_sibship_sizes,
    compute_size_structure,
)
from pedsum.validate import (
    DROP_FRACTION_WARN,
    DROPPABLE_CHECKS,
    NON_REDUCIBLE_BLOCK_CHECKS,
    load_and_validate,
    reduce_pedigree,
    validate_pedigree,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pedsum.validate import ValidationContext


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
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG-level logging to stderr (default: INFO)",
    )
    g.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="WARNING-level logging only (suppress per-section timings)",
    )


def _add_format_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--sep",
        choices=_SEP_CHOICES,
        default="auto",
        help="input column delimiter. 'auto' (default) sniffs the first "
        "non-empty line for tab/comma/semicolon/pipe; falls back to "
        "whitespace (PLINK fam-style) when none are present. Pass an "
        "explicit choice to opt out of sniffing.",
    )
    p.add_argument(
        "--sex-encoding",
        choices=("auto", "default", "plink"),
        default="auto",
        help="how to decode the sex column: 'default' = 0=female, 1=male "
        "(pedsum default); 'plink' = 1=male, 2=female, 0=unknown (PLINK fam "
        "convention); 'auto' (default) detects from the observed tokens.",
    )
    p.add_argument(
        "--plink-sex",
        action="store_const",
        dest="sex_encoding",
        const="plink",
        help="legacy alias for --sex-encoding=plink (PLINK convention: 1=male, 2=female)",
    )
    p.add_argument(
        "--allow-missing-sex",
        action="store_true",
        help="tolerate rows whose sex is missing after imputation — either "
        "because the row is unsexed and not used as a parent (orphan), OR "
        "because it is used as BOTH mother and father with unknown sex "
        "(role-ambiguous). Such rows are auto-fixed to sex=-1 in the "
        "validate-fixed output. Without this flag, either case hard-blocks. "
        "Incompatible with --effective-size / --inbreeding in summarize "
        "(sex-stratified estimators require resolved sex).",
    )
    p.add_argument(
        "--no-override-asserted-sex",
        action="store_true",
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
        description=("Pedigree summary CLI. Depends on numpy, scipy, pandas, pyyaml, and pedigree-graph."),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    sub = parser.add_subparsers(dest="subcommand", parser_class=_FullHelpParser)

    p_sum = sub.add_parser("summarize", help="summarise a pedigree (TSV input)")
    p_sum.add_argument(
        "--in",
        dest="in_path",
        required=True,
        type=Path,
        help="input pedigree (.tsv or .tsv.gz)",
    )
    p_sum.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="output directory (created if needed). Always writes "
        "summary.yaml (slim categorised summary), summary.extra.yaml "
        "(per-generation / per-cohort / per-transition arrays and full "
        "per-individual quantiles), and annotated.tsv.gz (input pedigree "
        "+ per-individual columns; suppressed under --safe-attempt). "
        "Pass --tsv to also write summary.pedigree.tsv and "
        "summary.individual.tsv.",
    )
    p_sum.add_argument(
        "--id-col",
        default="id",
        metavar="NAME",
        help="column name for individual ID (int) (default: %(default)s)",
    )
    p_sum.add_argument(
        "--sex-col",
        default="sex",
        metavar="NAME",
        help="column name for sex; accepts M/F (any case), Male/Female, or "
        "0/1 (default: %(default)s; 0=female, 1=male). See --plink-sex.",
    )
    p_sum.add_argument(
        "--mother-col",
        default="mother",
        metavar="NAME",
        help="column name for mother ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p_sum.add_argument(
        "--father-col",
        default="father",
        metavar="NAME",
        help="column name for father ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p_sum.add_argument(
        "--birth-year-col",
        default=None,
        metavar="NAME",
        help="optional column name for birth year (integer or float "
        "calendar year; -1/NA/blank for unknown). When set, pedsum threads "
        "the column through to PedigreeGraph so the Hill overlapping-"
        "generation Ne estimator (Ne_H) can build its cohort window; "
        "without it Ne_H collapses to Ne_V.",
    )
    p_sum.add_argument(
        "--birth-year-min",
        type=int,
        default=_BIRTH_YEAR_DEFAULT_MIN,
        metavar="YEAR",
        help="inclusive lower bound for birth_year sanity check (default: %(default)s). "
        "No-op without --birth-year-col.",
    )
    p_sum.add_argument(
        "--birth-year-max",
        type=int,
        default=None,
        metavar="YEAR",
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
        type=_positive_int,
        default=1,
        metavar="N",
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
        "--out",
        dest="out_dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="output directory (created if needed); writes validate.log "
        "(per-finding TSV) and validate.tsv.gz (the pedigree with any "
        "auto-fixes applied; not written if a block is detected). With "
        "--drop-offending also writes validate.dropped.tsv (the removal manifest)",
    )
    p_val.add_argument(
        "--id-col",
        default="id",
        metavar="NAME",
        help="column name for individual ID (int) (default: %(default)s)",
    )
    p_val.add_argument(
        "--sex-col",
        default="sex",
        metavar="NAME",
        help="column name for sex; accepts M/F or 0/1 with 0=female, 1=male (default: %(default)s)",
    )
    p_val.add_argument(
        "--mother-col",
        default="mother",
        metavar="NAME",
        help="column name for mother ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p_val.add_argument(
        "--father-col",
        default="father",
        metavar="NAME",
        help="column name for father ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p_val.add_argument(
        "--no-sex-check",
        action="store_true",
        help="bypass the sex-conflict check on missing parents; auto-added "
        "founders default to sex=F when the role is ambiguous (default: off)",
    )
    p_val.add_argument(
        "--drop-offending",
        action="store_true",
        help="produce a Reduced Pedigree: iteratively remove every individual "
        "named in a droppable check finding (clearing references to it) until "
        "the pedigree passes under the invoked flags. Writes validate.dropped.tsv "
        "and exits 1 whenever anything was dropped. Column/parse-level failures "
        "still BLOCK. WARNING: changes relatedness/Ne/founder counts (default: off)",
    )
    p_val.add_argument(
        "--birth-year-col",
        default=None,
        metavar="NAME",
        help="optional column name for birth year (integer or float calendar "
        "year; -1/NA/blank for unknown). When set, validate runs three checks: "
        "birth_year_dtype (numeric parsing), birth_year_range (within "
        "[--birth-year-min, --birth-year-max]), and birth_year_topology "
        "(child birth_year >= parent birth_year).",
    )
    p_val.add_argument(
        "--birth-year-min",
        type=int,
        default=_BIRTH_YEAR_DEFAULT_MIN,
        metavar="YEAR",
        help="inclusive lower bound for birth_year_range check (default: %(default)s).",
    )
    p_val.add_argument(
        "--birth-year-max",
        type=int,
        default=None,
        metavar="YEAR",
        help="inclusive upper bound for birth_year_range check (default: current calendar year + 1).",
    )
    _add_format_args(p_val)
    _add_logging_args(p_val)

    args = parser.parse_args(argv)
    if args.subcommand is None:
        parser.print_help(sys.stderr)
        sys.exit(0)
    return args


def _init_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


# Module-level stack of the active ``_timed`` labels, innermost last. Touched
# only by the benchmark RSS profiler (``benchmarks/profile_memory.py``) so it
# can attribute each sample to the phase running when it was taken; empty and
# inert in normal runs.
_PROFILE_PHASE_STACK: list[str] = []


def _current_profile_phase() -> str | None:
    """Return the innermost active ``_timed`` label, or None outside any block.

    Exposed for the benchmark RSS profiler so it can attribute sampled RSS to a
    phase without depending on the private stack variable's name.
    """
    return _PROFILE_PHASE_STACK[-1] if _PROFILE_PHASE_STACK else None


@contextmanager
def _timed(label: str) -> Iterator[None]:
    """Log ``"<label> in <elapsed>s"`` at INFO around the wrapped block.

    Also pushes ``label`` onto a module-level phase stack for the block's
    duration so the benchmark profiler can attribute RSS samples to it. The
    push/pop is exception-safe (``finally``); the INFO log stays *after* the
    block so — exactly as before — it fires only on normal exit, never when the
    wrapped block raises.
    """
    t0 = time.perf_counter()
    _PROFILE_PHASE_STACK.append(label)
    try:
        yield
    finally:
        _PROFILE_PHASE_STACK.pop()
    logger.info("%s in %.2fs", label, time.perf_counter() - t0)


def _validation_kwargs(args: argparse.Namespace) -> dict:
    """Shared validation config pulled off ``args``.

    Both validation drivers — summarize's fail-fast ``load_and_validate`` and
    validate's accumulating ``validate_pedigree`` (also re-run on summarize's
    failure path to write the log) — take the same column-name / encoding /
    tolerance options. Building them in one place keeps the call sites from
    drifting.
    """
    return {
        "id_col": args.id_col,
        "sex_col": args.sex_col,
        "mother_col": args.mother_col,
        "father_col": args.father_col,
        "sex_encoding": args.sex_encoding,
        "zero_as_missing": False,
        "allow_missing_sex": args.allow_missing_sex,
        "override_asserted_sex": not args.no_override_asserted_sex,
        "birth_year_col": args.birth_year_col,
        "birth_year_min": args.birth_year_min,
        "birth_year_max": args.birth_year_max,
        "sep": args.sep,
        # validate-only; summarize args lack it. The tolerance lives in the
        # registry (ctx.no_sex_check) so it composes everywhere _run_checks runs.
        "no_sex_check": getattr(args, "no_sex_check", False),
    }


def _reduction_rebuild_kwargs(args: argparse.Namespace) -> dict:
    """Context-rebuild kwargs for the ``--drop-offending`` loop / self-verify.

    Exactly ``_validation_kwargs`` minus the file-only ``sep``, plus
    ``require_birth_year_col=False`` — what ``_build_context_from_df`` needs to
    rebuild a context from the reduced pedigree each round under the invoked
    flags (so the tolerance flags compose).
    """
    kwargs = _validation_kwargs(args)
    del kwargs["sep"]
    kwargs["require_birth_year_col"] = False
    return kwargs


def _write_validation_failure_log(args: argparse.Namespace) -> None:
    """Persist a full ``validate.log`` after summarize's fail-fast validation bailed.

    ``load_and_validate`` raises on the *first* failing check carrying only a
    short sample summary, so the user otherwise gets a one-line console error
    and no record. Re-run the accumulating validator to capture every finding
    and write the same ``validate.log`` the validate subcommand produces — the
    extra pass only happens on the (already-failing) error path. Best-effort:
    a secondary failure here is logged but must not mask the original error.
    """
    log_path = args.out_dir / "validate.log"
    try:
        _, _, findings, _ = validate_pedigree(args.in_path, **_validation_kwargs(args))
        _write_validate_log(findings, log_path)
    except (PedigreeError, FileNotFoundError, OSError) as e:
        logger.error("could not write %s: %s", log_path, e)
        return
    logger.error("wrote %s — fix the reported issue(s) and re-run", log_path)


def _run_summarize(args: argparse.Namespace, cmd: str) -> int:
    if _prepare_out_dir(args.out_dir) != 0:
        return 1
    try:
        # Wrapped so the read/validation peak is attributed to a phase by the
        # benchmark profiler; load_and_validate already logs its own timing.
        with _timed("load+validate"):
            df, children_csr = load_and_validate(args.in_path, **_validation_kwargs(args))
    except PedigreeError as e:
        logger.error("validation failed: %s", e)
        _write_validation_failure_log(args)
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
    with _timed("built PedigreeGraph"):
        pg = _build_pedigree_graph(df)
        df["ped_depth"] = np.asarray(pg.generation, dtype=np.int32)

    id_index = pd.Index(df["id"].to_numpy())

    with _timed("size+structure"):
        size, comp_labels = compute_size_structure(df, children_csr)

    with _timed("sibship sizes"):
        sibships = compute_sibship_sizes(df)

    with _timed("mating-pair summary"):
        mating_pairs = compute_mating_pair_summary(df)

    n_indiv = len(df)
    if args.per_individual_pairs:
        with _timed("relationship pairs"):
            pairs = _count_pairs_matrix_with_lists(df, pg=pg)
            pairs["_engine"] = "matrix"

        with _timed("relationship burden summary"):
            relationship_summary = compute_relationship_summary(df, pairs.get("_pair_lists"))
        # The materialised pair lists are consumed only by the burden summary
        # above; on a pair-dense pedigree they dominate the post-extraction
        # resident floor that persists through inbreeding / Ne / write. Drop
        # them now — nothing downstream reads pairs["_pair_lists"] (report.py
        # touches only the scalar counts and by_degree). Leaves pairs["_engine"].
        pairs.pop("_pair_lists", None)
    else:
        with _timed("relationship pair counts (count_pairs_streaming)"):
            streamed_counts = pg.count_pairs_streaming(max_degree=5, scope="full")
        # count_pairs_streaming builds the transient adjacency powers (_A … _A5)
        # and now releases them on exit, mirroring extract_pairs (pedigree-graph#4),
        # so they no longer stay resident through the inbreeding / Ne /
        # individual-table / write phases where the streaming run peaks.  We used
        # to reach into the private pg._release_pair_matrices() here; the upstream
        # symmetry fix made that unnecessary.
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
                "computing F on N=%s rows; may take several minutes — pass --no-inbreeding to skip",
                f"{n_indiv:,}",
            )
        with _timed("inbreeding (F + n_ancestors)"):
            F_vec = pg.compute_inbreeding()
            n_anc = pg.compute_n_ancestors()
            inb_summary: dict | None = _build_inbreeding_summary(F_vec)
    else:
        logger.info("inbreeding: skipped (--no-inbreeding)")
        inb_summary = None
        F_vec = np.zeros(n_indiv, dtype=np.float64)
        n_anc = np.zeros(n_indiv, dtype=np.int32)

    with _timed("descendants"):
        n_desc = pg.compute_n_descendants()

    effective_size: dict | None = None
    if args.effective_size:
        n_estimators = 8 if args.ne_coancestry else 7
        with _timed(f"effective size ({n_estimators} estimators)"):
            effective_size = compute_effective_size(
                pg,
                ne_coancestry=args.ne_coancestry,
                n_threads=args.ne_threads,
            )

    out_dir = args.out_dir

    with _timed("individual table built"):
        sex_source = df["sex_source"].to_numpy()
        idf = build_individual_df(
            df,
            id_index,
            F_vec,
            n_anc,
            n_desc,
            comp_labels,
            sex_source,
        )
        founder_summary, n_founder_anc = compute_founder_summary(idf)
        idf["n_founder_ancestors"] = n_founder_anc

    with _timed("aggregate pedigree sections"):
        aggregates = compute_aggregate_sections(
            idf,
            founder_summary=founder_summary,
            include_inbreeding=args.inbreeding,
        )

    tsv_payload, yaml_extras = _build_pedigree_data(
        args.in_path,
        cmd,
        size,
        sibships,
        pairs,
        inb_summary,
        mating_pairs,
        relationship_summary,
        aggregates,
    )
    if effective_size is not None:
        tsv_payload["effective_size_scalars"] = {name: result["ne"] for name, result in effective_size.items()}
        yaml_extras["effective_size"] = effective_size

    ind_data = _build_individual_data(
        idf,
        args.in_path,
        cmd,
        include_inbreeding=args.inbreeding,
    )

    if args.safe_attempt:
        _apply_safe_attempt(tsv_payload, ind_data)
        logger.info("safe-attempt redaction applied (min cell = %d)", SAFE_MIN_CELL)

    slim_yaml, extra_yaml = _build_summary_data(
        tsv_payload,
        ind_data,
        yaml_extras=yaml_extras,
    )
    _write_yaml(slim_yaml, out_dir / "summary.yaml")
    _write_yaml(extra_yaml, out_dir / "summary.extra.yaml")
    logger.info(
        "wrote %s/{summary.yaml, summary.extra.yaml}",
        out_dir,
    )

    if args.tsv:
        _write_long_tsv(tsv_payload, out_dir / "summary.pedigree.tsv")
        _write_long_tsv(ind_data, out_dir / "summary.individual.tsv")
        logger.info(
            "wrote %s/{summary.pedigree.tsv, summary.individual.tsv}",
            out_dir,
        )

    if args.safe_attempt:
        logger.info(
            "safe-attempt: skipped %s/annotated.tsv.gz (per-individual)",
            out_dir,
        )
    else:
        # Stable phase label (no per-run path) so the benchmark profiler can
        # aggregate this phase across repeats; out_dir is already in the
        # summary.yaml log line above.
        with _timed("wrote annotated.tsv.gz"):
            _write_annotated_tsv(args.in_path, args, idf, out_dir / "annotated.tsv.gz")

    return 0


def _write_fixed_pedigree(ctx: ValidationContext, args: argparse.Namespace, out_dir: Path) -> int:
    """Write ``validate.tsv.gz`` from ``ctx``; return total rows written.

    Synthesizes founder rows for missing parents, folds sex imputation into the
    sex column, topo-reorders (parents before children), and writes the gzipped
    TSV. Shared by the normal validate path and ``--drop-offending`` (which
    passes the reduced context).
    """
    n_total = len(ctx.df_raw)
    added_founders: list[dict] = []
    df_out = ctx.df_raw
    if ctx.ids is not None and ctx.mothers is not None and ctx.fathers is not None:
        id_index = pd.Index(ctx.ids)
        added_founders = _build_added_founders(ctx.mothers, ctx.fathers, id_index, args.no_sex_check)
        # Fold sex imputation into the fixed output so the user's "fixed"
        # file reflects the auto-fix instead of the original blanks.
        sex_imp = ctx.get_imputation()
        if sex_imp is not None:
            df_out = df_out.copy()
            imputed = sex_imp.imputed_sex
            original_unknown = sex_imp.original_unknown_mask
            overridden = sex_imp.overridden_mask
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
            if sex_imp.n_imputed > 0:
                logger.info("validate: imputed sex for %d row(s) from parent role", int(sex_imp.n_imputed))
            n_normalised = int(unresolved_mask.sum())
            if n_normalised > 0:
                logger.info("validate: normalised %d unresolved-sex row(s) to -1 in fixed output", n_normalised)
            # Stamp sex_source BEFORE the topological reorder so pandas reorders
            # the column along with the rest.
            df_out["sex_source"] = sex_imp.sex_source
        # Reorder so the fixed file is parents-before-children and feeds back
        # into pedsum without further auto-fixes.
        m_row, _ = _parent_rows(ctx.mothers, id_index)
        f_row, _ = _parent_rows(ctx.fathers, id_index)
        try:
            depth = _compute_depth_unordered(m_row, f_row, len(ctx.ids))
        except PedigreeError:
            depth = None  # acyclic FAIL already surfaced; skip reorder
        if depth is not None:
            order = np.argsort(depth, kind="stable")
            natural = np.arange(len(order))
            if not np.array_equal(order, natural):
                logger.info("validate: reordering %d row(s) into topological order", int((order != natural).sum()))
                df_out = df_out.iloc[order].reset_index(drop=True)

    out_path = out_dir / "validate.tsv.gz"
    _write_validate_tsv_gz(
        df_out, added_founders, args.id_col, args.sex_col, args.mother_col, args.father_col, out_path
    )
    n_total_out = n_total + len(added_founders)
    sys.stderr.write(f"wrote {out_path} ({n_total_out:,} rows; {len(added_founders)} founder(s) added)\n")
    return n_total_out


def _run_validate_drop(args: argparse.Namespace, by_check: dict, out_dir: Path, ctx: ValidationContext) -> int:
    """``--drop-offending``: reduce the pedigree to a passing one (or BLOCK).

    Column/parse-level failures still BLOCK (no row removal fixes them).
    Otherwise iterate to a fixpoint, write the removal manifest + reduced
    pedigree, self-verify the result passes under the invoked flags, and exit 1
    if anything was dropped (0 if nothing needed dropping).
    """
    manifest_path = out_dir / "validate.dropped.tsv"
    blocking = sorted(n for n in NON_REDUCIBLE_BLOCK_CHECKS if n in by_check and by_check[n].status == "FAIL")
    if blocking:
        sys.stderr.write("\nBLOCKED — --drop-offending cannot fix these by removing individuals:\n")
        for n in blocking:
            sys.stderr.write(f"  - {n}\n")
        return 2

    if not any(by_check[n].status == "FAIL" for n in DROPPABLE_CHECKS if n in by_check):
        _write_dropped_manifest([], manifest_path)
        _write_fixed_pedigree(ctx, args, out_dir)
        sys.stderr.write("--drop-offending: nothing to drop; pedigree already passes\n")
        return 0

    result = reduce_pedigree(ctx, rebuild_kwargs=_reduction_rebuild_kwargs(args))
    if len(result.df_current) == 0:
        sys.stderr.write("\nBLOCKED — --drop-offending removed every individual (empty pedigree)\n")
        return 2

    _write_dropped_manifest(result.dropped, manifest_path)
    sys.stderr.write(f"wrote {manifest_path} ({result.n_distinct_dropped} id(s) dropped)\n")
    _write_fixed_pedigree(result.ctx_final, args, out_dir)

    # Self-verify the written artifact passes under the invoked flags. The
    # output is always tab-separated regardless of the input --sep, so sniff it.
    out_path = out_dir / "validate.tsv.gz"
    _n, vresults, _vf, _vctx = validate_pedigree(out_path, **{**_validation_kwargs(args), "sep": "auto"})
    failed = sorted(r.name for r in vresults if r.status == "FAIL")
    if failed:
        raise PedigreeError(f"--drop-offending self-verify failed; reduced pedigree still FAILs: {failed}")

    pct = 100.0 * result.n_rows_removed / result.n_input_rows
    sys.stderr.write(
        f"--drop-offending: dropped {result.n_distinct_dropped} individual(s) / "
        f"{result.n_rows_removed} row(s) of {result.n_input_rows} ({pct:.1f}%) over "
        f"{result.n_rounds} round(s); cleared {result.n_cleared_refs} reference(s)\n"
    )
    if result.n_rows_removed / result.n_input_rows > DROP_FRACTION_WARN:
        logger.warning(
            "--drop-offending removed %.1f%% of rows; relatedness / Ne / founder counts reflect the reduced set",
            pct,
        )
    return 1


def _run_validate(args: argparse.Namespace, cmd: str) -> int:
    if _prepare_out_dir(args.out_dir) != 0:
        return 1
    try:
        n_total, results, findings, ctx = validate_pedigree(args.in_path, **_validation_kwargs(args))
    except PedigreeError as e:
        logger.error("validation could not run: %s", e)
        return 2
    except (FileNotFoundError, OSError) as e:
        logger.error("file error: %s", e)
        return 2

    by_check = {r.name: r for r in results}

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

    if args.drop_offending:
        return _run_validate_drop(args, by_check, out_dir, ctx)

    if blocks:
        sys.stderr.write("\nBLOCKED — fix the following before re-running:\n")
        for b in blocks:
            sys.stderr.write(f"  - {b}\n")
        return 2

    _write_fixed_pedigree(ctx, args, out_dir)
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
