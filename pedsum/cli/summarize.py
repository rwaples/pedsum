"""``summarize``: the pedigree summary YAMLs, the long TSVs and ``annotated.tsv.gz``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from pedigree_graph import ResourceError

from pedsum.base import (
    F_WALK_WARN_VISITS,
    SexSource,
    logger,
)
from pedsum.cli.common import commit_thread_budget, load_or_fail
from pedsum.pairs import augment_pair_counts, build_pedigree_graph
from pedsum.parse import read_aligned_columns
from pedsum.pedigree_ops import group_mating_pairs
from pedsum.progress import relationship_progress, timed
from pedsum.redaction import SAFE_MIN_CELL
from pedsum.report import (
    apply_safe_attempt,
    as_plain,
    build_individual_data,
    build_pedigree_data,
    build_summary_data,
    prepare_out_dir,
    write_annotated_tsv,
    write_long_tsv,
    write_yaml,
)
from pedsum.sections import (
    build_inbreeding_summary,
    build_individual_df,
    compute_aggregate_sections,
    compute_founder_summary,
    compute_mating_pair_summary,
    compute_relationship_summary_from_burden,
    compute_sibship_sizes,
    compute_size_structure,
    equivalent_complete_generations,
)
from pedsum.sex_concordance import compute_offspring_sex_concordance
from pedsum.validate import ValidationConfig

if TYPE_CHECKING:
    import argparse

    from pedsum.ids import IdLabels


def _with_input_sex_source(
    df: pl.DataFrame, args: argparse.Namespace, labels: IdLabels
) -> tuple[pl.DataFrame, dict[str, int] | None]:
    """``df`` with the input's own ``sex_source`` column, and its counts; ``(df, None)`` without one.

    A Fixed Pedigree (``validate.tsv.gz``) records where each sex came from;
    loading it again finds every sex already resolved, so the provenance comes
    from the column (ADR 0006). A row the column leaves blank (a founder
    ``validate`` added) keeps the value loading derived. A column with values
    pedsum does not write is ignored with a WARNING.
    """
    try:
        raw = read_aligned_columns(
            args.in_path, args.sep, args.id_col, df["id"].to_numpy(), labels, columns=["sex_source"]
        )
    except pl.exceptions.ColumnNotFoundError:
        return df, None
    known = [s.value for s in SexSource]
    tokens = raw["sex_source"].str.strip_chars()
    unknown = tokens.filter(tokens.is_not_null() & ~tokens.is_in(known)).unique().sort()
    if len(unknown):
        logger.warning(
            "input column sex_source has values pedsum does not write (%s); ignored", unknown.head(3).to_list()
        )
        return df, None
    sex_source = pl.select(pl.coalesce(tokens, df["sex_source"])).to_series()
    logger.info("sex_source: read from the input's sex_source column")
    return df.with_columns(sex_source.alias("sex_source")), {v: int((sex_source == v).sum()) for v in known}


def run_summarize(args: argparse.Namespace, cmd: str) -> int:
    """``summarize``: write the summary YAMLs, ``annotated.tsv.gz`` and, under ``--tsv``, the long TSVs."""
    if prepare_out_dir(args.out_dir) != 0:
        return 1
    commit_thread_budget(args)
    loaded = load_or_fail(args)
    if isinstance(loaded, int):
        return loaded
    df, labels = loaded
    df, sex_source_counts = _with_input_sex_source(df, args, labels)

    if args.sex_concordance_seed is not None and args.sex_concordance_permutations == 0:
        logger.warning(
            "--sex-concordance-seed has no effect without --sex-concordance-permutations",
        )

    # Build the PedigreeGraph once and reuse for every primitive that
    # needs it (relationship pairs, F, lineage counts). Its graph rows are
    # df's rows, so its parent rows index df directly.
    with timed("built PedigreeGraph"):
        pg = build_pedigree_graph(df)
    mother_rows, father_rows = pg.mother_rows, pg.father_rows
    n_indiv = len(df)
    max_depth = int(df["ped_depth"].to_numpy().max()) if n_indiv else 0

    # One cheap pass, but path counts can double with each level of a deep
    # pedigree and overflow int64; counting first fails before the expensive
    # phases rather than after them.
    try:
        with timed("descendants"):
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

    with timed("size+structure"):
        size, comp_labels = compute_size_structure(df, mother_rows, father_rows)
    size["sex_source"] = sex_source_counts

    with timed("mating pairs grouped"):
        mating = group_mating_pairs(df, mother_rows, father_rows)

    with timed("sibship sizes"):
        sibships = compute_sibship_sizes(mating.sizes)

    with timed("mating-pair summary"):
        mating_pairs = compute_mating_pair_summary(mating.sizes)

    # Opt-in: no grouping, no moments, no sampler unless --sex-concordance was
    # typed. Runs on the validated final sex, so it must follow load_and_validate.
    sex_concordance: dict | None = None
    if args.sex_concordance:
        sex_concordance = compute_offspring_sex_concordance(
            df,
            permutations=args.sex_concordance_permutations,
            seed=args.sex_concordance_seed or 0,
        )

    if args.per_individual_pairs:
        with timed("relationship burden"):
            with relationship_progress("relationship_burden") as progress:
                burden = pg.relationship_burden(progress=progress)
            pairs = augment_pair_counts(burden.category_counts)
            pairs["_engine"] = "rust_streaming_burden"
            relationship_summary = compute_relationship_summary_from_burden(df, burden)
    else:
        with (
            timed("relationship pair counts (relationship_counts)"),
            relationship_progress("relationship_counts") as progress,
        ):
            counts = pg.relationship_counts(max_degree=args.max_degree, progress=progress)
        # The Rust row-streaming engine counts every pair under its closest
        # category without materialising pair lists, so the 23 counts are
        # exact and peak memory stays O(N) (pedigree-graph ADR 0010).
        pairs = augment_pair_counts(counts)
        pairs["_engine"] = "rust_streaming"
        relationship_summary = {
            "computed": False,
            "skip_reason": "per-individual relationship burden is opt-in; pass --per-individual-burden to compute it",
            "n_individual_pairs": int(n_indiv * (n_indiv - 1) // 2),
        }

    if args.inbreeding:
        if n_indiv * min(2 ** (max_depth + 1) - 2, n_indiv) > F_WALK_WARN_VISITS:
            logger.warning(
                "computing F on %s rows at max depth %d; its time grows with rows and steeply "
                "with depth (README 'Deep pedigrees') — pass --no-inbreeding to skip",
                f"{n_indiv:,}",
                max_depth,
            )
        with timed("inbreeding (F + n_ancestors)"):
            F_vec = pg.inbreeding()
            n_anc = pg.distinct_ancestor_counts()
            inb_summary: dict | None = build_inbreeding_summary(F_vec)
    else:
        logger.info("inbreeding: skipped (--no-inbreeding)")
        inb_summary = None
        F_vec = n_anc = None

    out_dir = args.out_dir

    with timed("individual table built"):
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

    with timed("aggregate pedigree sections"):
        aggregates = compute_aggregate_sections(idf, founder_summary=founder_summary)

    tsv_payload = build_pedigree_data(
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

    ind_data = build_individual_data(idf, args.in_path, cmd)

    if args.safe_attempt:
        apply_safe_attempt(tsv_payload, ind_data)
        logger.info("safe-attempt redaction applied (min cell = %d)", SAFE_MIN_CELL)
    # The writers below walk plain dicts and lists.
    tsv_payload = as_plain(tsv_payload)
    ind_data = as_plain(ind_data)

    slim_yaml, extra_yaml = build_summary_data(tsv_payload, ind_data)
    write_yaml(slim_yaml, out_dir / "summary.yaml")
    write_yaml(extra_yaml, out_dir / "summary.extra.yaml")
    logger.info(
        "wrote %s/{summary.yaml, summary.extra.yaml}",
        out_dir,
    )

    if args.tsv:
        write_long_tsv(tsv_payload, out_dir / "summary.pedigree.tsv")
        write_long_tsv(ind_data, out_dir / "summary.individual.tsv")
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
        with timed("wrote annotated.tsv.gz"):
            write_annotated_tsv(
                args.in_path, ValidationConfig.from_args(args), args.sep, idf, out_dir / "annotated.tsv.gz", labels
            )

    return 0
