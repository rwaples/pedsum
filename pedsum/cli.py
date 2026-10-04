"""Argument parsing and the summarize / validate command runners."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import polars as pl
from pedigree_graph import ResourceError, configure_threads
from pedigree_graph.effective_size import ALL_EFFECTIVE_SIZE_ESTIMATORS

from pedsum.base import (
    _F_WALK_WARN_VISITS,
    SEX_UNKNOWN,
    VERSION,
    PedigreeError,
    logger,
)
from pedsum.epimight import (
    BASE_YEAR,
    EPIMIGHT_RELATIONSHIP_ORDER,
    PLACEHOLDER_COLUMNS,
    build_epimight_skeleton,
    iter_relative_pairs,
    relationship_diagnostics,
    validate_relationship_codes,
)
from pedsum.memory import GiB, MemoryWatchdog, parse_size, resolve_limit
from pedsum.pairs import _augment_pair_counts, _build_pedigree_graph
from pedsum.parse import _BIRTH_YEAR_DEFAULT_MIN, _SEP_CHOICES, read_reference_mask
from pedsum.pedigree_ops import _group_mating_pairs
from pedsum.progress import relationship_progress
from pedsum.report import (
    SAFE_MIN_CELL,
    _apply_safe_attempt,
    _build_added_founders,
    _build_effective_size_data,
    _build_individual_data,
    _build_pedigree_data,
    _build_phantom_parents,
    _build_summary_data,
    _format_check_summary,
    _next_free_id,
    _prepare_out_dir,
    _write_annotated_tsv,
    _write_dropped_manifest,
    _write_long_tsv,
    _write_validate_log,
    _write_validate_tsv_gz,
    _write_yaml,
    atomic_output,
)
from pedsum.sections import (
    _build_inbreeding_summary,
    build_individual_df,
    compute_aggregate_sections,
    compute_effective_size,
    compute_founder_summary,
    compute_individual_delta_f_for_reference,
    compute_mating_pair_summary,
    compute_relationship_summary_from_burden,
    compute_sibship_sizes,
    compute_size_structure,
    equivalent_complete_generations,
)
from pedsum.sex_concordance import compute_offspring_sex_concordance
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

    def error(self, message: str) -> NoReturn:
        self.print_help(sys.stderr)
        sys.stderr.write(f"\nerror: {message}\n")
        sys.exit(2)


class _RemovedForEffectiveSize(argparse.Action):
    """Exit 2 on a summarize flag that 0.15.0 removed when Ne moved to ``effective-size`` (ADR 0004)."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> NoReturn:
        """Name the flag and the command that replaced it."""
        parser.exit(
            2,
            f"{option_string} was removed in 0.15.0; effective population size moved to its own command:\n"
            "  pedsum effective-size --in X --out Y [--estimators all]\n",
        )


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


def _add_threads_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--threads",
        type=_positive_int,
        default=1,
        help="worker threads for the pedigree-graph engine (default 1). "
        "Counts and pair lists are identical under any value; only wall "
        "time changes. Set once per process, before the first computation.",
    )


def _add_memory_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--max-memory",
        type=parse_size,
        default=None,
        metavar="SIZE",
        help="stop with exit code 3 once pedsum's resident memory passes SIZE "
        "(e.g. 500M, 12G; 0 turns the limit off). Default: 80%% of the memory "
        "available at start, the smallest of the host's MemAvailable and the "
        "headroom of every enclosing cgroup. Best effort: memory is sampled "
        "once a second, so a fast allocation can still reach the kernel's OOM "
        "killer.",
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
        "effective-size refuses such rows (its sex-stratified estimators "
        "need resolved sex).",
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


def _add_input_args(p: argparse.ArgumentParser, *, out_help: str, birth_year_help: str) -> None:
    """``--in``, ``--out`` and the column / birth-year options of a command that summarises one pedigree."""
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        type=Path,
        help="input pedigree (.tsv or .tsv.gz)",
    )
    p.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        type=Path,
        metavar="DIR",
        help=out_help,
    )
    p.add_argument(
        "--id-col",
        default="id",
        metavar="NAME",
        help="column name for individual ID (int) (default: %(default)s)",
    )
    p.add_argument(
        "--sex-col",
        default="sex",
        metavar="NAME",
        help="column name for sex; accepts M/F (any case), Male/Female, or "
        "0/1 (default: %(default)s; 0=female, 1=male). See --plink-sex.",
    )
    p.add_argument(
        "--mother-col",
        default="mother",
        metavar="NAME",
        help="column name for mother ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p.add_argument(
        "--father-col",
        default="father",
        metavar="NAME",
        help="column name for father ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p.add_argument("--birth-year-col", default=None, metavar="NAME", help=birth_year_help)
    p.add_argument(
        "--birth-year-min",
        type=int,
        default=_BIRTH_YEAR_DEFAULT_MIN,
        metavar="YEAR",
        help="inclusive lower bound for birth_year sanity check (default: %(default)s). "
        "No-op without --birth-year-col.",
    )
    p.add_argument(
        "--birth-year-max",
        type=int,
        default=None,
        metavar="YEAR",
        help="inclusive upper bound for birth_year sanity check "
        "(default: current calendar year + 1). No-op without --birth-year-col.",
    )


def _estimator_list(v: str) -> list[str]:
    """Argparse type for ``--estimators``: a comma list of estimator names, or ``all``."""
    names = [name.strip() for name in v.split(",") if name.strip()]
    unknown = [name for name in names if name != "all" and name not in ALL_EFFECTIVE_SIZE_ESTIMATORS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(
            f"unknown estimator(s) {', '.join(unknown) or repr(v)}; "
            f"choose from {', '.join(ALL_EFFECTIVE_SIZE_ESTIMATORS)}, or all"
        )
    return list(ALL_EFFECTIVE_SIZE_ESTIMATORS) if "all" in names else names


def _positive_int(v: str) -> int:
    """Argparse type guard for ints >= 1."""
    n = _nonnegative_int(v)
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {v!r}")
    return n


def _nonnegative_int(v: str) -> int:
    """Argparse type guard for ints >= 0."""
    try:
        iv = int(v)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {v!r}") from exc
    if iv < 0:
        raise argparse.ArgumentTypeError(f"expected integer >= 0, got {iv}")
    return iv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _FullHelpParser(
        prog="pedigree_summary.py",
        description=("Pedigree summary CLI. Depends on numpy, scipy, polars, pyyaml, and pedigree-graph."),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    sub = parser.add_subparsers(dest="subcommand", parser_class=_FullHelpParser)

    p_sum = sub.add_parser("summarize", help="summarise a pedigree (TSV input)")
    _add_input_args(
        p_sum,
        out_help="output directory (created if needed). Always writes "
        "summary.yaml (slim categorised summary), summary.extra.yaml "
        "(per-generation / per-cohort / per-transition arrays and full "
        "per-individual quantiles), and annotated.tsv.gz (input pedigree "
        "+ per-individual columns; suppressed under --safe-attempt). "
        "Pass --tsv to also write summary.pedigree.tsv and "
        "summary.individual.tsv.",
        birth_year_help="optional column name for birth year (integer or float "
        "calendar year; -1/NA/blank for unknown). When set, summarize "
        "validates it: numeric, within --birth-year-min/--birth-year-max, and "
        "no child born before a parent. annotated.tsv.gz copies the column "
        "like any other input column, with or without this flag.",
    )
    p_sum.add_argument(
        "--inbreeding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute per-individual F and the inbreeding summary section "
        "(default: on; pass --no-inbreeding to skip). F is the most expensive "
        "single computation in pedsum: its time grows with rows and steeply with "
        "depth, and pedsum logs a WARNING first when the pedigree is large or "
        "deep enough for that to take a while. When off, F and n_ancestors in "
        "the per-individual table are zero-filled.",
    )
    p_sum.add_argument(
        "--effective-size",
        "--no-effective-size",
        "--ne-coancestry",
        action=_RemovedForEffectiveSize,
        nargs=0,
        help=argparse.SUPPRESS,
    )
    p_sum.add_argument(
        "--max-degree",
        type=int,
        choices=range(1, 6),
        default=5,
        metavar="{1..5}",
        help="count relationship pairs up to this degree (default: %(default)s). "
        "Codes past it, and their by_degree entries, are null: not counted, "
        "not zero. Cost climbs steeply with degree; on a 783K-row pedigree at "
        "--threads 10, degree 2 took 1.4 s, 3 took 221 s, and 5 took 1,833 s.",
    )
    p_sum.add_argument(
        "--per-individual-burden",
        "--per-individual-pairs",
        dest="per_individual_pairs",
        action="store_true",
        help="compute the per-individual relationship-burden summary with "
        "O(N) native arrays and no pair lists. The historical "
        "--per-individual-pairs spelling is also accepted. Unset "
        "(default), the burden summary is a stub.",
    )
    p_sum.add_argument(
        "--sex-concordance",
        action="store_true",
        help="compute Offspring Sex Concordance: whether resolved offspring "
        "sex is more or less concordant within Sibships / Maternal Offspring "
        "Groups / Paternal Offspring Groups than a pooled fixed-margin "
        "exchangeability null predicts. Off by default. The headline uses "
        "input sex only (imputed sex is conditional on having reproduced, "
        "which breaks calibration); all-resolved sex is reported alongside as "
        "a sensitivity. Adds `demography.offspring_sex_concordance` to the "
        "summary YAML.",
    )
    p_sum.add_argument(
        "--sex-concordance-permutations",
        type=_nonnegative_int,
        default=0,
        metavar="N",
        help="calibrate the sex-concordance p-values against N fixed-margin "
        "permutations of the headline analysis (default: %(default)s — "
        "analytical only). Implies `--sex-concordance`. The analytical "
        "p-value is asymptotic and screening-only: any claim at p < 0.01 "
        "needs permutations. 1,000 is a reasonable starting point; cost "
        "scales linearly in N and in the number of eligible groups (~80ms "
        "per draw at 2M groups with numba, ~224ms with the NumPy fallback).",
    )
    p_sum.add_argument(
        "--sex-concordance-seed",
        type=int,
        default=None,
        metavar="INT",
        help="seed for the sex-concordance permutation sampler (default: 0). "
        "No effect without `--sex-concordance-permutations`. Results are "
        "reproducible given (seed, backend) — the backend is recorded in the "
        "output because numba is a soft import.",
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
    _add_threads_args(p_sum)
    _add_memory_args(p_sum)
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
        "--fill-half-founders",
        action="store_true",
        help="give each half-founder (exactly one known parent) its own new "
        "founder in the missing slot: female for a missing mother, male for a "
        "missing father, with IDs above every existing ID. Kinship and F among the "
        "input individuals are unchanged; founder contributions then sum to 1, "
        "which ne_long_term_contributions requires. WARNING: adds one founder "
        "per half-founder, so founder counts and founder-based statistics "
        "change (default: off)",
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
    _add_threads_args(p_val)
    _add_memory_args(p_val)
    _add_logging_args(p_val)

    p_epi = sub.add_parser(
        "epimight-input",
        help="emit the structural skeleton of an EPIMIGHT long-form input (TSV; --parquet for the native format)",
    )
    p_epi.add_argument("--in", dest="in_path", required=True, type=Path, help="input pedigree (.tsv or .tsv.gz)")
    p_epi.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="output directory (created if needed); writes pipeline_input.tsv "
        "(and pipeline_input.parquet under --parquet). Structural columns "
        "(person_id, relationship_kind, relatives, born_at_year) are computed; "
        "phenotype columns (failure_status, failure_time, relatives_diagnosed, "
        "dead_at_year) are emitted as empty placeholders to fill downstream.",
    )
    p_epi.add_argument(
        "--id-col",
        default="id",
        metavar="NAME",
        help="column name for individual ID (int) (default: %(default)s)",
    )
    p_epi.add_argument(
        "--sex-col",
        default="sex",
        metavar="NAME",
        help="column name for sex (validated but unused by relationship extraction) (default: %(default)s)",
    )
    p_epi.add_argument(
        "--mother-col",
        default="mother",
        metavar="NAME",
        help="column name for mother ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p_epi.add_argument(
        "--father-col",
        default="father",
        metavar="NAME",
        help="column name for father ID; -1/NA/blank for unknown (default: %(default)s)",
    )
    p_epi.add_argument(
        "--birth-year-col",
        default=None,
        metavar="NAME",
        help="optional birth-year column; when set it becomes born_at_year "
        "(unknown -> empty), else born_at_year = --base-year + generation.",
    )
    p_epi.add_argument(
        "--birth-year-min",
        type=int,
        default=_BIRTH_YEAR_DEFAULT_MIN,
        metavar="YEAR",
        help="inclusive lower bound for birth_year sanity check (default: %(default)s). No-op without --birth-year-col.",
    )
    p_epi.add_argument(
        "--birth-year-max",
        type=int,
        default=None,
        metavar="YEAR",
        help="inclusive upper bound for birth_year sanity check (default: current calendar year + 1). "
        "No-op without --birth-year-col.",
    )
    p_epi.add_argument(
        "--rels",
        default=",".join(EPIMIGHT_RELATIONSHIP_ORDER),
        metavar="CODES",
        help="comma-separated EPIMIGHT relationship codes to emit, in order (default: %(default)s).",
    )
    p_epi.add_argument(
        "--disorder",
        default="trait1",
        metavar="NAME",
        help="disorder label for the single emitted block (a pedigree carries no "
        "trait; EPIMIGHT long form is one block per disorder) (default: %(default)s).",
    )
    p_epi.add_argument(
        "--base-year",
        type=int,
        default=BASE_YEAR,
        metavar="YEAR",
        help="calendar offset for the derived born_at_year = base-year + generation "
        "(default: %(default)s). No-op when --birth-year-col is set.",
    )
    p_epi.add_argument(
        "--drop-founders",
        action="store_true",
        help="drop founder-generation rows (off by default; the fitACE emitter "
        "drops them because a founder's degenerate full-sib stratum breaks h2 "
        "estimation — opt in when the output feeds estimation).",
    )
    p_epi.add_argument(
        "--pairs",
        action="store_true",
        help="also write relative_pairs.tsv — the list of relative pairs backing "
        "the skeleton's counts (one row per pair per relationship_kind: id1, id2, "
        "relationship_kind, kinship). The kinship column is the nominal coefficient "
        "looked up by relationship_kind (not computed from the pedigree). "
        "Materialises every pair, so it can be large on pair-dense pedigrees "
        "(cousins scale ~quadratically).",
    )
    p_epi.add_argument(
        "--exact-kinship",
        action="store_true",
        help="add a kinship_exact column to relative_pairs with the exact pedigree "
        "kinship (inbreeding-, MZ-, and multi-path-aware), which can exceed the "
        "nominal value. Runs the kinship recurrence over every pair; cost scales "
        "with pair count × pedigree depth. pedigree-graph computes the recurrence "
        "in float32, so the emitted float64 column carries float32-origin values. "
        "No-op without --pairs.",
    )
    p_epi.add_argument(
        "--parquet",
        action="store_true",
        help="also write the parquet form of each emitted table (pipeline_input "
        "and, under --pairs, relative_pairs), the format EPIMIGHT's R Pipeline "
        "reads natively.",
    )
    _add_format_args(p_epi)
    _add_threads_args(p_epi)
    _add_memory_args(p_epi)
    _add_logging_args(p_epi)

    p_es = sub.add_parser(
        "effective-size",
        help="estimate effective population size (Ne) with up to eight pedigree-based estimators",
    )
    _add_input_args(
        p_es,
        out_help="output directory (created if needed); writes effective_size.yaml "
        "with every estimator's scalar and per-depth arrays. After a stop at "
        "the memory limit (exit 3) the file holds the estimators that finished, "
        "with status: stopped_memory_limit.",
        birth_year_help="optional column name for birth year (integer or float "
        "calendar year; -1/NA/blank for unknown). The Hill overlapping-"
        "generation estimator (ne_hill_overlapping) builds its cohort window "
        "from it; without it ne_hill_overlapping collapses to "
        "ne_variance_family_size.",
    )
    p_es.add_argument(
        "--estimators",
        type=_estimator_list,
        action="extend",
        default=None,
        metavar="NAME[,NAME...]",
        help="estimators to run, as a comma list or repeated flags; 'all' "
        f"selects every one. Choices: {', '.join(ALL_EFFECTIVE_SIZE_ESTIMATORS)}. "
        "Default: all. Every estimator appears in the output; one not selected "
        "reports reason: not_requested.",
    )
    p_es.add_argument(
        "--reference-col",
        default=None,
        metavar="NAME",
        help="column marking the reference subpopulation (1/0 or true/false; "
        "missing counts as 0) over which ne_individual_delta_f averages the "
        "individual increase in inbreeding (Gutiérrez et al. 2008). Default: "
        "the last observed depth. The record gains reference_column.",
    )
    _add_format_args(p_es)
    _add_threads_args(p_es)
    _add_memory_args(p_es)
    _add_logging_args(p_es)

    args = parser.parse_args(argv)
    if args.subcommand is None:
        parser.print_help(sys.stderr)
        sys.exit(0)
    # Asking for permutations is asking for the analysis. There is no
    # --no-sex-concordance; absence of the flag is the off state.
    if getattr(args, "sex_concordance_permutations", 0) > 0:
        args.sex_concordance = True
    if args.subcommand == "summarize" and args.per_individual_pairs and args.max_degree < 5:
        p_sum.error("--per-individual-burden always counts degrees 1-5; drop --max-degree")
    if args.subcommand == "effective-size":
        selected = set(args.estimators or ALL_EFFECTIVE_SIZE_ESTIMATORS)
        args.estimators = [name for name in ALL_EFFECTIVE_SIZE_ESTIMATORS if name in selected]
    return args


def _commit_thread_budget(args: argparse.Namespace) -> None:
    """Hand ``--threads`` to pedigree-graph before its first computation.

    The library commits one budget per process and refuses a later change,
    so this runs before any graph is built. Its results do not depend on the
    value; only wall time does.
    """
    threads = getattr(args, "threads", 1)
    try:
        configure_threads(threads)
    except RuntimeError:
        # Already committed by an earlier call in this process (the test
        # suite drives several runs in-process); the budget stands.
        logger.debug("pedigree-graph thread budget already committed; --threads=%s ignored", threads)


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
        force=True,
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
    # One slice, not a check then an index: the memory watchdog calls this
    # from its own thread while the main thread pops labels.
    (label,) = _PROFILE_PHASE_STACK[-1:] or (None,)
    return label


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
    _commit_thread_budget(args)
    try:
        # Wrapped so the read/validation peak is attributed to a phase by the
        # benchmark profiler; load_and_validate already logs its own timing.
        with _timed("load+validate"):
            df = load_and_validate(args.in_path, **_validation_kwargs(args))
    except PedigreeError as e:
        logger.error("validation failed: %s", e)
        _write_validation_failure_log(args)
        return 1
    except (FileNotFoundError, OSError) as e:
        logger.error("file error: %s", e)
        return 2

    if args.sex_concordance_seed is not None and args.sex_concordance_permutations == 0:
        logger.warning(
            "--sex-concordance-seed has no effect without --sex-concordance-permutations",
        )

    # Build the PedigreeGraph once and reuse for every primitive that
    # needs it (relationship pairs, F, lineage counts). Its graph rows are
    # df's rows, so its parent rows index df directly.
    with _timed("built PedigreeGraph"):
        pg = _build_pedigree_graph(df)
    mother_rows, father_rows = pg.mother_rows, pg.father_rows
    n_indiv = len(df)
    max_depth = int(df["ped_depth"].max()) if n_indiv else 0

    # One cheap pass, but path counts can double with each level of a deep
    # pedigree and overflow int64; counting first fails before the expensive
    # phases rather than after them.
    try:
        with _timed("descendants"):
            n_desc = pg.descendant_path_counts()
    except ResourceError as e:
        if e.code != "arithmetic_overflow":
            raise
        logger.error(
            "descendant path counts overflow int64 at max depth %d, so summarize cannot "
            "report n_descendant_paths for this pedigree; see README 'Deep pedigrees'",
            max_depth,
        )
        return 1

    with _timed("size+structure"):
        size, comp_labels = compute_size_structure(df, mother_rows, father_rows)

    with _timed("mating pairs grouped"):
        mating = _group_mating_pairs(df, mother_rows, father_rows)

    with _timed("sibship sizes"):
        sibships = compute_sibship_sizes(mating.sizes)

    with _timed("mating-pair summary"):
        mating_pairs = compute_mating_pair_summary(mating.sizes)

    # Opt-in: no grouping, no moments, no sampler unless --sex-concordance was
    # typed. Runs on the validated final sex, so it must follow load_and_validate.
    sex_concordance: dict | None = None
    if args.sex_concordance:
        sex_concordance = compute_offspring_sex_concordance(
            df,
            permutations=args.sex_concordance_permutations,
            seed=args.sex_concordance_seed or 0,
            timer=_timed,
        )

    if args.per_individual_pairs:
        with _timed("relationship burden"):
            with relationship_progress("relationship_burden") as progress:
                burden = pg.relationship_burden(progress=progress)
            pairs = _augment_pair_counts(burden.category_counts)
            pairs["_engine"] = "rust_streaming_burden"
            relationship_summary = compute_relationship_summary_from_burden(df, burden)
    else:
        with (
            _timed("relationship pair counts (relationship_counts)"),
            relationship_progress("relationship_counts") as progress,
        ):
            counts = pg.relationship_counts(max_degree=args.max_degree, progress=progress)
        # The Rust row-streaming engine counts every pair under its closest
        # category without materialising pair lists, so the 23 counts are
        # exact and peak memory stays O(N) (pedigree-graph ADR 0010).
        pairs = _augment_pair_counts(counts)
        pairs["_engine"] = "rust_streaming"
        relationship_summary = {
            "computed": False,
            "skip_reason": "per-individual relationship burden is opt-in; pass --per-individual-burden to compute it",
            "n_individual_pairs": int(n_indiv * (n_indiv - 1) // 2),
        }

    if args.inbreeding:
        if n_indiv * min(2 ** (max_depth + 1) - 2, n_indiv) > _F_WALK_WARN_VISITS:
            logger.warning(
                "computing F on %s rows at max depth %d; its time grows with rows and steeply "
                "with depth (README 'Deep pedigrees') — pass --no-inbreeding to skip",
                f"{n_indiv:,}",
                max_depth,
            )
        with _timed("inbreeding (F + n_ancestors)"):
            F_vec = pg.inbreeding()
            n_anc = pg.distinct_ancestor_counts()
            inb_summary: dict | None = _build_inbreeding_summary(F_vec)
    else:
        logger.info("inbreeding: skipped (--no-inbreeding)")
        inb_summary = None
        F_vec = np.zeros(n_indiv, dtype=np.float64)
        n_anc = np.zeros(n_indiv, dtype=np.int32)

    out_dir = args.out_dir

    with _timed("individual table built"):
        sex_source = df["sex_source"].to_numpy()
        idf = build_individual_df(
            df,
            mother_rows,
            father_rows,
            mating,
            F_vec,
            n_anc,
            n_desc,
            comp_labels,
            sex_source,
        )
        founder_summary, n_founder_anc = compute_founder_summary(idf, mother_rows, father_rows)
        idf = idf.with_columns(
            pl.Series("n_founder_ancestors", n_founder_anc),
            pl.Series("ecg", equivalent_complete_generations(df, mother_rows, father_rows)),
        )

    with _timed("aggregate pedigree sections"):
        aggregates = compute_aggregate_sections(
            idf,
            founder_summary=founder_summary,
            include_inbreeding=args.inbreeding,
        )

    tsv_payload = _build_pedigree_data(
        args.in_path,
        cmd,
        size,
        sibships,
        pairs,
        inb_summary,
        mating_pairs,
        relationship_summary,
        aggregates,
        sex_concordance,
        max_degree=args.max_degree,
    )

    ind_data = _build_individual_data(
        idf,
        args.in_path,
        cmd,
        include_inbreeding=args.inbreeding,
    )

    if args.safe_attempt:
        _apply_safe_attempt(tsv_payload, ind_data)
        logger.info("safe-attempt redaction applied (min cell = %d)", SAFE_MIN_CELL)

    slim_yaml, extra_yaml = _build_summary_data(tsv_payload, ind_data)
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


#: Estimators that need every row's sex resolved (pedigree-graph raises
#: ``MissingMetadataError`` for them otherwise). The others run on a pedigree
#: with unknown sex, so ``effective-size`` refuses only when one is selected.
_SEX_DEPENDENT_ESTIMATORS = frozenset({"ne_sex_ratio", "ne_variance_family_size", "ne_hill_overlapping"})


def _run_effective_size(args: argparse.Namespace, cmd: str, watchdog: MemoryWatchdog) -> int:
    """``effective-size``: run the selected Ne estimators and write ``effective_size.yaml``.

    A stop at the memory limit still publishes ``effective_size.yaml``, with
    the estimators that were running marked ``memory_limit`` (ADR 0004 §4).
    """
    if _prepare_out_dir(args.out_dir) != 0:
        return 1
    _commit_thread_budget(args)

    requested = args.estimators
    out_path = args.out_dir / "effective_size.yaml"
    # Registered before the input is read, so a stop at any point replaces an
    # effective_size.yaml left by an earlier run. Until the graph is built,
    # n_total is unknown and no estimator has run.
    n_total: int | None = None
    results: dict = {name: {"ne": None, "reason": "not_requested"} for name in ALL_EFFECTIVE_SIZE_ESTIMATORS}

    def publish_partial(rss: int, limit: int) -> None:
        stopped = {
            "ne": None,
            "reason": "memory_limit",
            "rss_gib": round(rss / GiB, 2),
            "limit_gib": round(limit / GiB, 2),
        }
        # One copy per estimator: a shared dict would be dumped as a YAML alias.
        partial = {
            name: dict(stopped) if name in requested and record.get("reason") == "not_requested" else record
            for name, record in results.items()
        }
        data = _build_effective_size_data(args.in_path, cmd, n_total, "stopped_memory_limit", requested, partial)
        _write_yaml(data, out_path)
        logger.error("wrote %s with the estimators that finished", out_path)

    watchdog.on_breach(publish_partial)
    try:
        with _timed("load+validate"):
            df = load_and_validate(args.in_path, **_validation_kwargs(args))
    except PedigreeError as e:
        logger.error("validation failed: %s", e)
        _write_validation_failure_log(args)
        return 1
    except (FileNotFoundError, OSError) as e:
        logger.error("file error: %s", e)
        return 2

    needs_sex = [name for name in requested if name in _SEX_DEPENDENT_ESTIMATORS]
    if needs_sex and (df["sex"].to_numpy() == SEX_UNKNOWN).any():
        logger.error(
            "effective size needs resolved sex for every row to run %s; remove "
            "--allow-missing-sex, supply sex for the offending rows, or select "
            "only estimators that do not use sex",
            ", ".join(needs_sex),
        )
        return 1

    reference_rows = None
    if args.reference_col is not None:
        try:
            mask = read_reference_mask(args.in_path, args.sep, args.id_col, args.reference_col, df["id"].to_numpy())
        except PedigreeError as e:
            logger.error("%s", e)
            return 1
        reference_rows = np.flatnonzero(mask)
        if reference_rows.size == 0:
            logger.error("reference column %r marks no row", args.reference_col)
            return 1
        if "ne_individual_delta_f" not in requested:
            logger.warning("--reference-col only affects ne_individual_delta_f, which was not requested")

    with _timed("built PedigreeGraph"):
        pg = _build_pedigree_graph(df)
    n_total = len(df)
    with _timed(f"effective size ({', '.join(requested)})"):
        computed = compute_effective_size(pg, requested)
        if reference_rows is not None and "ne_individual_delta_f" in requested:
            computed["ne_individual_delta_f"] = compute_individual_delta_f_for_reference(
                pg, reference_rows, args.reference_col
            )
        # One rebinding, so the breach callback sees every result or none.
        results = computed

    delta_f = results["ne_individual_delta_f"]
    if delta_f.get("reason") == "no_estimate" and any(ne is not None for ne in delta_f["ne_per_gen"]):
        if "reference_column" in delta_f:
            reference = f"the reference column {delta_f['reference_column']!r}"
        elif delta_f["reference_generation"] is not None:
            reference = f"the last observed depth (depth {delta_f['reference_generation']})"
        else:
            reference = "the last observed depth"
        logger.warning(
            "ne_individual_delta_f has no estimate (%s) over %s with n_reference=%d, though ne_per_gen has "
            "one at another depth; pass --reference-col NAME to choose a different reference subpopulation",
            delta_f["code"],
            reference,
            delta_f["n_reference"],
        )

    with watchdog.disarm() as owns_output:
        if owns_output:
            data = _build_effective_size_data(args.in_path, cmd, len(df), "complete", requested, results)
            _write_yaml(data, out_path)
            logger.info("wrote %s", out_path)
    return 0


def _write_fixed_pedigree(
    ctx: ValidationContext, args: argparse.Namespace, out_dir: Path, *, next_id: int | None = None
) -> int:
    """Write ``validate.tsv.gz`` from ``ctx``; return total rows written.

    Synthesizes founder rows for missing parents (and, with
    ``--fill-half-founders``, phantom parents numbered from ``next_id``), folds
    sex imputation into the sex column, topo-reorders (parents before
    children), and writes the gzipped TSV. Shared by the normal validate path
    and ``--drop-offending`` (which passes the reduced context, and the input's
    ``next_id`` so no phantom reuses a dropped ID).
    """
    n_total = len(ctx.df_raw)
    added_founders: list[dict] = []
    df_out = ctx.df_raw
    if ctx.ids is not None and ctx.mothers is not None and ctx.fathers is not None:
        added_founders = _build_added_founders(ctx.mothers, ctx.fathers, ctx.id_index, args.no_sex_check)
        # Write every row's resolved sex (after imputation and role overrides)
        # in pedsum's canonical encoding, 0=female, 1=male, -1=unknown,
        # whatever encoding the input used, so the column holds one encoding.
        # sex_source records which rows pedsum changed.
        sex_imp = ctx.get_imputation()
        if sex_imp is not None:
            if sex_imp.n_imputed > 0:
                logger.info("validate: imputed sex for %d row(s) from parent role", int(sex_imp.n_imputed))
            n_unresolved = int((sex_imp.imputed_sex == SEX_UNKNOWN).sum())
            if n_unresolved > 0:
                logger.info("validate: wrote %d unresolved-sex row(s) as -1 in fixed output", n_unresolved)
            # Stamp sex and sex_source BEFORE the topological reorder so they
            # are reordered along with the rest.
            df_out = df_out.with_columns(
                pl.Series(args.sex_col, sex_imp.imputed_sex).cast(pl.String),
                pl.Series("sex_source", sex_imp.sex_source.astype(str)),
            )
        if args.fill_half_founders:
            if next_id is None:
                next_id = _next_free_id(ctx.ids, ctx.mothers, ctx.fathers)
            mothers, fathers, phantoms = _build_phantom_parents(ctx.mothers, ctx.fathers, next_id)
            filled_m = mothers != ctx.mothers
            filled_f = fathers != ctx.fathers
            df_out = df_out.with_columns(
                pl.when(pl.Series(filled_m))
                .then(pl.Series(mothers.astype(str)))
                .otherwise(pl.col(args.mother_col).cast(pl.String))
                .alias(args.mother_col),
                pl.when(pl.Series(filled_f))
                .then(pl.Series(fathers.astype(str)))
                .otherwise(pl.col(args.father_col).cast(pl.String))
                .alias(args.father_col),
            )
            added_founders += phantoms
            logger.info("validate: added %d phantom parent(s) for half-founders", len(phantoms))
        # Reorder so the fixed file is parents-before-children and feeds back
        # into pedsum without further auto-fixes.
        depth = ctx.depth
        if (depth >= 0).all():  # else the acyclic FAIL already surfaced; skip reorder
            order = np.argsort(depth, kind="stable")
            natural = np.arange(len(order))
            if not np.array_equal(order, natural):
                logger.info("validate: reordering %d row(s) into topological order", int((order != natural).sum()))
                df_out = df_out[order]

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
    _write_fixed_pedigree(result.ctx_final, args, out_dir, next_id=_next_free_id(*ctx.require_id_parents()))

    # Self-verify the written artifact passes under the invoked flags. The
    # output is always tab-separated regardless of the input --sep, so sniff it,
    # and always writes sex as 0=female, 1=male whatever the input encoding.
    out_path = out_dir / "validate.tsv.gz"
    _n, vresults, _vf, _vctx = validate_pedigree(
        out_path,
        **{**_validation_kwargs(args), "sep": "auto", "sex_encoding": "default"},  # ty: ignore[invalid-argument-type]
    )
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
    _commit_thread_budget(args)
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


def _write_epimight_table(frame: pl.DataFrame, out_dir: Path, stem: str, *, parquet: bool) -> int:
    """Write ``frame`` to ``<out_dir>/<stem>.tsv`` (and ``.parquet`` if requested).

    Returns 0 on success. (polars writes parquet natively, so ``--parquet``
    no longer needs pyarrow.)
    """
    tsv_path = out_dir / f"{stem}.tsv"
    with atomic_output(tsv_path) as tmp:
        frame.write_csv(tmp, separator="\t")
    logger.info("wrote %s (%d rows)", tsv_path, len(frame))
    if parquet:
        parquet_path = out_dir / f"{stem}.parquet"
        with atomic_output(parquet_path) as tmp:
            frame.write_parquet(tmp)
        logger.info("wrote %s", parquet_path)
    return 0


def _write_relative_pairs(frames: Iterator[pl.DataFrame], out_dir: Path, *, parquet: bool) -> None:
    """Append each kind's pairs to ``relative_pairs.tsv`` (and ``.parquet``) as they come.

    Only one kind's frame is alive at a time. The parquet is written as one
    temporary file per kind, then streamed into a single file.
    """
    tsv_path = out_dir / "relative_pairs.tsv"
    parquet_path = out_dir / "relative_pairs.parquet"
    n_rows = 0
    with (
        atomic_output(tsv_path) as tsv_tmp,
        tsv_tmp.open("wb") as tsv,
        tempfile.TemporaryDirectory(dir=out_dir) as tmp,
    ):
        parts: list[Path] = []
        for i, frame in enumerate(frames):
            frame.write_csv(tsv, separator="\t", include_header=i == 0)
            if parquet:
                parts.append(Path(tmp) / f"{i}.parquet")
                frame.write_parquet(parts[-1])
            n_rows += len(frame)
            del frame
        if parquet:
            with atomic_output(parquet_path) as parquet_tmp:
                pl.scan_parquet(parts).sink_parquet(parquet_tmp)
    logger.info("wrote %s (%d rows)", tsv_path, n_rows)
    if parquet:
        logger.info("wrote %s", parquet_path)


def _run_epimight_input(args: argparse.Namespace) -> int:
    _commit_thread_budget(args)
    """``epimight-input``: emit the EPIMIGHT long-form skeleton from a pedigree."""
    if _prepare_out_dir(args.out_dir) != 0:
        return 1
    try:
        rels = validate_relationship_codes([r.strip() for r in args.rels.split(",") if r.strip()])
    except ValueError as e:
        logger.error("%s", e)
        return 2

    try:
        with _timed("load+validate"):
            df = load_and_validate(args.in_path, **_validation_kwargs(args))
    except PedigreeError as e:
        logger.error("validation failed: %s", e)
        _write_validation_failure_log(args)
        return 1
    except (FileNotFoundError, OSError) as e:
        logger.error("file error: %s", e)
        return 2

    with _timed("built PedigreeGraph"):
        pg = _build_pedigree_graph(df)

    # The skeleton counts relatives without a pair list. --pairs extracts the
    # list for its own export, after the skeleton's counts are freed.
    with _timed("epimight skeleton"):
        frame = build_epimight_skeleton(
            df,
            pg,
            rels=rels,
            disorder=args.disorder,
            base_year=args.base_year,
            drop_founders=args.drop_founders,
        )

    out_dir = args.out_dir
    rc = _write_epimight_table(frame, out_dir, "pipeline_input", parquet=args.parquet)
    if rc != 0:
        return rc
    # Summarize the skeleton now and free it, so it is not resident under --pairs.
    diagnostics = relationship_diagnostics(frame, rels)
    del frame

    if args.exact_kinship and not args.pairs:
        logger.warning("--exact-kinship has no effect without --pairs")

    if args.pairs:
        with _timed("relative pairs"):
            _write_relative_pairs(
                iter_relative_pairs(df, pg, rels=rels, exact_kinship=args.exact_kinship),
                out_dir,
                parquet=args.parquet,
            )

    logger.info(
        "structural columns computed; placeholder column(s) left empty: %s",
        ", ".join(PLACEHOLDER_COLUMNS),
    )
    for code, n_with, mean_rel in diagnostics:
        logger.info(
            "  %s %3s: %d person(s) with relatives, mean relatives %.3f",
            args.disorder,
            code,
            n_with,
            mean_rel,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns process exit code."""
    args = _parse_args(argv)
    _init_logging(args.verbose, args.quiet)
    cmd = " ".join(sys.argv)
    with MemoryWatchdog(resolve_limit(args.max_memory), phase=_current_profile_phase) as watchdog:
        if args.subcommand == "summarize":
            return _run_summarize(args, cmd)
        if args.subcommand == "effective-size":
            return _run_effective_size(args, cmd, watchdog)
        if args.subcommand == "validate":
            return _run_validate(args, cmd)
        if args.subcommand == "epimight-input":
            return _run_epimight_input(args)
    return 1
