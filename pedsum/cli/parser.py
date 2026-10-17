"""Argument parsing for every ``pedsum`` subcommand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from pedigree_graph.effective_size import ALL_EFFECTIVE_SIZE_ESTIMATORS

from pedsum.base import (
    MIN_STRATUM_NETWORKS,
    TRAIT_KINDS,
    VERSION,
)
from pedsum.epimight import (
    BASE_YEAR,
    EPIMIGHT_RELATIONSHIP_ORDER,
)
from pedsum.memory import parse_size
from pedsum.parse import BIRTH_YEAR_DEFAULT_MIN, SEP_CHOICES


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


class _RemovedSexEncoding(argparse.Action):
    """Exit 2 on a sex-encoding flag that 0.15.0 removed when PLINK coding became the only numeric one."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> NoReturn:
        """Name the flag and the one coding pedsum now reads."""
        parser.exit(
            2,
            f"{option_string} was removed in 0.15.0; pedsum reads sex as M/F, Male/Female, or PLINK's "
            "1=male, 2=female, 0=unknown. Recode a 0=female, 1=male column first.\n",
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


def _add_threads_args(p: argparse.ArgumentParser, *, default_cores: bool = False) -> None:
    if default_cores:
        default = None
        purpose = (
            "threads for the pg-phenotype computation (default: the physical cores the process may run on, "
            "capped by NUMBA_NUM_THREADS; the pedigree-graph engine keeps 1 unless this is given)"
        )
    else:
        default = 1
        purpose = "worker threads for the pedigree-graph engine (default 1)"
    p.add_argument(
        "--threads",
        type=_positive_int,
        default=default,
        help=f"{purpose}. Results are identical under any value; only wall "
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
        choices=SEP_CHOICES,
        default="auto",
        help="input column delimiter. 'auto' (default) sniffs the first "
        "non-empty line for tab/comma/semicolon/pipe; falls back to "
        "whitespace (PLINK fam-style) when none are present. Pass an "
        "explicit choice to opt out of sniffing.",
    )
    p.add_argument("--sex-encoding", action=_RemovedSexEncoding, nargs="?", help=argparse.SUPPRESS)
    p.add_argument("--plink-sex", action=_RemovedSexEncoding, nargs=0, help=argparse.SUPPRESS)
    p.add_argument(
        "--allow-missing-sex",
        action="store_true",
        help="tolerate an individual with unknown sex used as BOTH mother and father "
        "(sex_role_ambiguity), which otherwise blocks; validate writes its sex as 0. "
        "An unsexed individual who is never a parent needs no flag: it is kept with "
        "sex 0 and a WARNING. effective-size refuses its sex-dependent estimators "
        "while any row is unsexed.",
    )
    p.add_argument(
        "--no-override-asserted-sex",
        action="store_true",
        help="keep a recorded sex that contradicts the only parent role an individual has, "
        "instead of overriding it (asserted M used only as mother -> F; asserted F used only "
        "as father -> M; each override is a sex_role_override finding). The contradiction is "
        "then a sex_role_consistency finding: validate writes the pedigree with it and exits 1, "
        "and --drop-offending removes the individual. Filling a missing sex from role is "
        "unaffected.",
    )


def _add_input_args(
    p: argparse.ArgumentParser, *, out_help: str, birth_year_help: str, sex_col_help: str | None = None
) -> None:
    """``--in``, ``--out`` and the column / birth-year options every subcommand reads its pedigree with."""
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
        help="column name for individual ID; integers or strings (default: %(default)s)",
    )
    p.add_argument(
        "--sex-col",
        default="sex",
        metavar="NAME",
        help=sex_col_help
        or "column name for sex; accepts M/F (any case), Male/Female, or "
        "PLINK's 1=male, 2=female, 0=unknown (default: %(default)s).",
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
    p.add_argument(
        "--missing-parent-token",
        action="append",
        default=[],
        metavar="TOKEN",
        help="a token that means a missing parent in the mother and father columns, besides -1, "
        "blank, NA, '.', '?' and the like; repeatable. Matched exactly after stripping whitespace. "
        "pedsum reads 0 as an ID, so a file that writes a missing parent as 0 (PLINK .fam, most R "
        "packages) needs --missing-parent-token 0. A token that is also the ID of a row is an "
        "error. validate writes such parents blank, so its output needs no flag.",
    )
    p.add_argument("--birth-year-col", default=None, metavar="NAME", help=birth_year_help)
    p.add_argument(
        "--birth-year-min",
        type=int,
        default=BIRTH_YEAR_DEFAULT_MIN,
        metavar="YEAR",
        help="inclusive lower bound for the birth_year sanity check, and for the year of a "
        "validate --birth-date-col date (default: %(default)s). No-op without either column.",
    )
    p.add_argument(
        "--birth-year-max",
        type=int,
        default=None,
        metavar="YEAR",
        help="inclusive upper bound for the birth_year sanity check, and for the year of a "
        "validate --birth-date-col date (default: current calendar year + 1). No-op without "
        "either column.",
    )
    p.add_argument(
        "--min-parent-age",
        type=_nonnegative_int,
        default=0,
        metavar="YEARS",
        help="flag a parent fewer than YEARS years older than its child, comparing birth years "
        "(birth_year_topology: child.birth_year - parent.birth_year >= YEARS). The default 0 "
        "flags only a child born before a parent. No-op without --birth-year-col.",
    )
    p.add_argument(
        "--multiple-birth-col",
        default=None,
        metavar="NAME",
        help="optional column holding a multiple-birth set id shared by everyone born in one "
        "delivery (blank/NA for none), such as validate --twin-max-days's multiple_birth_set or a "
        "register's own id. A member whose parents are not both known and shared with another "
        "member, or who is alone in a set, is a multiple_birth_sets finding; validate writes that "
        "cell blank. summarize reports the sets under demography.multiple_births (default: off)",
    )
    p.add_argument(
        "--zygosity-col",
        default=None,
        metavar="NAME",
        help="optional column holding MZ or DZ (blank/NA for unknown) per multiple-birth set "
        "member; needs --multiple-birth-col. The two MZ members of a set are linked as MZ "
        "co-twins in the PedigreeGraph, so they count as MZ, not FS, pairs. A mark that cannot "
        "form such a link (outside a set, the only MZ in its set, three or more MZ in a set, or "
        "MZ co-twins of opposite sex) is an mz_links finding; validate writes that cell blank "
        "(default: off)",
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


def _trait_type(v: str) -> tuple[str, str]:
    """Argparse type for ``--trait-type``: ``COL=TYPE`` with TYPE one of the trait kinds."""
    column, sep, kind = v.rpartition("=")
    if not sep or not column or kind not in TRAIT_KINDS:
        raise argparse.ArgumentTypeError(f"expected COL={'|'.join(TRAIT_KINDS)}, got {v!r}")
    return column, kind


def _int64(v: str) -> int:
    """Argparse type guard for the int64 range pg-phenotype takes a seed in."""
    try:
        n = int(v)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {v!r}") from exc
    if not -(2**63) <= n < 2**63:
        raise argparse.ArgumentTypeError(f"expected an integer from -2^63 to 2^63-1, got {v!r}")
    return n


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a ``pedsum`` command line; no subcommand prints the help and exits 0."""
    parser = _FullHelpParser(
        prog="pedsum",
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
        "deep enough for that to take a while. When off, annotated.tsv.gz has no "
        "F or n_distinct_ancestors column.",
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
        "per draw at 2M groups).",
    )
    p_sum.add_argument(
        "--sex-concordance-seed",
        type=int,
        default=None,
        metavar="INT",
        help="seed for the sex-concordance permutation sampler (default: 0). "
        "No effect without `--sex-concordance-permutations`. The same seed "
        "reproduces the same p-values.",
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
    _add_input_args(
        p_val,
        out_help="output directory (created if needed); writes validate.log "
        "(per-finding TSV) and validate.tsv.gz (the pedigree with any "
        "auto-fixes applied; not written if a block is detected). With "
        "--drop-offending also writes validate.dropped.tsv (the removal manifest), and with "
        "--break-cycles validate.cleared_links.tsv (the links it cleared)",
        birth_year_help="optional column name for birth year (integer or float calendar "
        "year; -1/NA/blank for unknown). When set, validate runs three checks: "
        "birth_year_dtype (numeric parsing), birth_year_range (within "
        "[--birth-year-min, --birth-year-max]), and birth_year_topology "
        "(child birth_year - parent birth_year >= --min-parent-age, default 0).",
    )
    p_val.add_argument(
        "--birth-date-col",
        default=None,
        metavar="NAME",
        help="optional column name for birth date (ISO YYYY-MM-DD; NA/blank for unknown). "
        "When set, validate runs birth_date_dtype (date parsing; an unparsable date blocks), "
        "birth_date_range (the year within [--birth-year-min, --birth-year-max]) and, with "
        "--birth-year-col, birth_date_year_consistency (the date falls in the birth year).",
    )
    p_val.add_argument(
        "--twin-max-days",
        type=_nonnegative_int,
        default=None,
        metavar="N",
        help="annotate multiple births: full sibs (same mother and father) born at most N days "
        "apart form a set, and validate.tsv.gz gets a multiple_birth_set column holding each "
        "member's set number (blank for none). Triplets are one set. Needs --birth-date-col, "
        "whose column must then be present. Zygosity is unknown, so a set is not an MZ "
        "annotation. Without this flag an input multiple_birth_set column is dropped "
        "(default: off)",
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
        "--break-cycles",
        action="store_true",
        help="clear the parent links the data show to be wrong instead of blocking on them: "
        "every self-link, and within a descent cycle every link to a parent born after the "
        "child. Needs --birth-year-col. Individuals stay; each cleared link is a cycle_breaks "
        "finding and a row of validate.cleared_links.tsv (id, role, parent, reason). A cycle "
        "with no such link still BLOCKs, or loses its members under --drop-offending "
        "(default: off)",
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
    _add_format_args(p_val)
    _add_threads_args(p_val)
    _add_memory_args(p_val)
    _add_logging_args(p_val)

    p_epi = sub.add_parser(
        "epimight-input",
        help="emit the structural skeleton of an EPIMIGHT long-form input (TSV; --parquet for the native format)",
    )
    _add_input_args(
        p_epi,
        out_help="output directory (created if needed); writes pipeline_input.tsv "
        "(and pipeline_input.parquet under --parquet). Structural columns "
        "(person_id, relationship_kind, relatives, born_at_year) are computed; "
        "phenotype columns (failure_status, failure_time, relatives_diagnosed, "
        "dead_at_year) are emitted as empty placeholders to fill downstream.",
        birth_year_help="optional birth-year column; when set it becomes born_at_year "
        "(unknown -> empty), else born_at_year = --base-year + generation.",
        sex_col_help="column name for sex (validated but unused by relationship extraction) (default: %(default)s)",
    )
    p_epi.add_argument(
        "--rels",
        default=",".join(EPIMIGHT_RELATIONSHIP_ORDER),
        metavar="CODES",
        help="comma-separated EPIMIGHT relationship codes to emit, in order (default: %(default)s). "
        "MZ is opt-in: it holds the co-twins --zygosity-col links, whom FS then leaves out.",
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

    p_am = sub.add_parser(
        "assortative-mating",
        help="estimate the Mate Correlation of one or two trait columns over Mating Pairs",
    )
    _add_input_args(
        p_am,
        out_help="output directory (created if needed); writes assortative_mating.yaml.",
        birth_year_help="optional column name for birth year (integer or float "
        "calendar year; -1/NA/blank for unknown). Required by --stratify-by birth_year.",
    )
    p_am.add_argument(
        "--trait",
        nargs="+",
        required=True,
        metavar="COL",
        help="one or two trait columns. With two, every mother trait x father trait cell is reported.",
    )
    p_am.add_argument(
        "--trait-type",
        type=_trait_type,
        action="append",
        default=[],
        metavar="COL=TYPE",
        help=f"state a trait's type ({', '.join(TRAIT_KINDS)}); repeatable. Unstated, two levels "
        "are binary, more than 20 numeric values continuous, and 3-20 values an error. Ordinal "
        "must always be stated.",
    )
    p_am.add_argument(
        "--trait-missing",
        action="append",
        default=[],
        metavar="TOK",
        help="an extra trait token that means missing, matched after stripping whitespace; "
        "repeatable. Blank, NA, NaN, null, '.' and '?' are always missing.",
    )
    p_am.add_argument(
        "--stratify-by",
        choices=("depth", "birth_year"),
        default=None,
        help="stratify by each individual's Depth or birth-year bin. Mating Pairs with a mate of "
        "unknown stratum are dropped from every estimate.",
    )
    p_am.add_argument(
        "--birth-year-bin",
        type=_positive_int,
        default=10,
        metavar="YEARS",
        help="birth-year bin width for --stratify-by birth_year (default: %(default)s).",
    )
    p_am.add_argument(
        "--min-stratum-networks",
        type=_positive_int,
        default=MIN_STRATUM_NETWORKS,
        metavar="N",
        help="with --stratify-by, a sex x stratum is kept only when the cell's pairs in it span at least N "
        "Mate Networks; smaller strata are dropped from both the crude and the stratified estimates of "
        "that cell (default: %(default)s).",
    )
    p_am.add_argument(
        "--permutations",
        type=_nonnegative_int,
        default=999,
        metavar="N",
        help="father permutations for each cell's p-value (default: %(default)s; 0 turns them off).",
    )
    p_am.add_argument(
        "--bootstrap",
        type=_nonnegative_int,
        default=0,
        metavar="N",
        help="Mate Network bootstrap draws. 0 (default) gives each estimate a Wald CI from its cluster-robust "
        "sandwich SE over Mate Networks; N > 0 replaces the CI with the percentile CI of N one-step draws "
        "(the sandwich SE is still reported).",
    )
    p_am.add_argument(
        "--seed",
        type=_int64,
        default=0,
        metavar="INT",
        help="seed for the permutations and the bootstrap draws, an integer from -2^63 to 2^63-1 "
        "(default: %(default)s).",
    )
    _add_format_args(p_am)
    _add_threads_args(p_am, default_cores=True)
    _add_memory_args(p_am)
    _add_logging_args(p_am)

    args = parser.parse_args(argv)
    if args.subcommand is None:
        parser.print_help(sys.stderr)
        sys.exit(0)
    # Asking for permutations is asking for the analysis. There is no
    # --no-sex-concordance; absence of the flag is the off state.
    if getattr(args, "sex_concordance_permutations", 0) > 0:
        args.sex_concordance = True
    if args.subcommand == "validate" and args.break_cycles and args.birth_year_col is None:
        p_val.error("--break-cycles needs --birth-year-col: it clears links to a parent born after the child")
    if args.subcommand == "validate" and args.twin_max_days is not None and args.birth_date_col is None:
        p_val.error("--twin-max-days needs --birth-date-col: twins are full sibs born at most N days apart")
    if args.subcommand == "validate" and args.twin_max_days is not None and args.multiple_birth_col is not None:
        p_val.error("--twin-max-days and --multiple-birth-col both define the multiple-birth sets; pass one")
    if getattr(args, "zygosity_col", None) is not None and args.multiple_birth_col is None:
        parser.error("--zygosity-col needs --multiple-birth-col: zygosity is read within multiple-birth sets")
    if args.subcommand == "summarize" and args.per_individual_pairs and args.max_degree < 5:
        p_sum.error("--per-individual-burden always counts degrees 1-5; drop --max-degree")
    if args.subcommand == "assortative-mating":
        _check_assortative_mating_args(p_am, args)
    if args.subcommand == "effective-size":
        selected = set(args.estimators or ALL_EFFECTIVE_SIZE_ESTIMATORS)
        args.estimators = [name for name in ALL_EFFECTIVE_SIZE_ESTIMATORS if name in selected]
    return args


def _check_assortative_mating_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Usage errors (exit 2) of ``assortative-mating``; turns ``--trait-type`` into a column -> type map."""
    if len(args.trait) > 2:
        parser.error(f"--trait takes one or two columns, got {len(args.trait)}")
    if len(set(args.trait)) != len(args.trait):
        parser.error(f"--trait names the same column twice: {' '.join(args.trait)}")
    stated: dict[str, str] = {}
    for column, kind in args.trait_type:
        if column not in args.trait:
            parser.error(f"--trait-type names {column!r}, which is not a --trait column")
        if column in stated:
            parser.error(f"--trait-type names {column!r} twice")
        stated[column] = kind
    args.trait_type = stated
    if args.stratify_by == "birth_year" and args.birth_year_col is None:
        parser.error("--stratify-by birth_year needs --birth-year-col")
