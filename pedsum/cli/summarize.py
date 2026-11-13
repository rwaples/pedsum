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
from pedsum.cli.common import build_graph_or_fail, commit_thread_budget, load_or_fail, prepare_outputs
from pedsum.multiple_births import compute_multiple_births
from pedsum.pairs import augment_pair_counts
from pedsum.parse import read_header
from pedsum.pedigree_ops import group_mating_pairs
from pedsum.progress import relationship_progress, timed
from pedsum.report import (
    as_plain,
    build_individual_data,
    build_pedigree_data,
    build_summary_data,
    safe_meta_block,
    write_annotated_tsv,
    write_long_tsv,
    write_yaml,
)
from pedsum.safe_log import Public, detail
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

    from pedsum.memory import MemoryWatchdog


def run_summarize(args: argparse.Namespace, cmd: str, _watchdog: MemoryWatchdog) -> int:
    """``summarize``: write the summary YAMLs, ``annotated.tsv.gz`` and, under ``--tsv``, the long TSVs.

    A ``--safe-attempt`` run writes the safe names of the YAMLs and of the pedigree TSV, and neither the
    individual TSV nor ``annotated.tsv.gz``.
    """
    names = ["summary.yaml", "summary.extra.yaml"]
    if args.tsv:
        names.append("summary.pedigree.tsv")
        if not args.safe_attempt:
            names.append("summary.individual.tsv")
    paths = prepare_outputs(args, *names)
    if isinstance(paths, int):
        return paths
    commit_thread_budget(args)
    loaded = load_or_fail(args)
    if isinstance(loaded, int):
        return loaded
    df, labels = loaded

    if args.sex_concordance_seed is not None and args.sex_concordance_permutations == 0:
        logger.warning(
            "--sex-concordance-seed has no effect without --sex-concordance-permutations",
        )

    # Build the PedigreeGraph once and reuse for every primitive that
    # needs it (relationship pairs, F, lineage counts). Its graph rows are
    # df's rows, so its parent rows index df directly.
    pg = build_graph_or_fail(df, labels)
    if isinstance(pg, int):
        return pg
    mother_rows, father_rows = pg.mother_rows, pg.father_rows
    n_indiv = len(df)
    max_depth = int(df["ped_depth"].to_numpy().max()) if n_indiv else 0

    # One cheap pass, but path counts can double with each level of a deep
    # pedigree and overflow int64; counting first fails before the expensive
    # phases rather than after them. No safe output reads them.
    n_desc = None
    if not args.safe_attempt:
        try:
            with timed("descendants"):
                n_desc = pg.descendant_path_counts()
        except ResourceError as e:
            if e.code != "arithmetic_overflow":
                raise
            logger.error(
                "descendant path counts overflow int64 at max depth %s, so summarize cannot "
                "report n_descendant_paths for this pedigree; see README 'Deep pedigrees'",
                detail(max_depth),
            )
            return 1

    with timed("size+structure"):
        size, comp_labels = compute_size_structure(df, mother_rows, father_rows)
    # A Fixed Pedigree records where each sex came from; loading kept it (ADR 0006).
    if "sex_source" in read_header(args.in_path, args.sep):
        sex_source = df["sex_source"]
        size["sex_source"] = {s.value: int((sex_source == s.value).sum()) for s in SexSource}

    with timed("mating pairs grouped"):
        mating = group_mating_pairs(df, mother_rows, father_rows)

    # Opt-in: no grouping, no moments, no sampler unless --sex-concordance was
    # typed. Runs on the validated final sex, so it must follow load_and_validate.
    sex_concordance: dict | None = None
    if args.sex_concordance:
        sex_concordance = compute_offspring_sex_concordance(
            df,
            permutations=args.sex_concordance_permutations,
            seed=args.sex_concordance_seed or 0,
        )

    if args.inbreeding:
        if n_indiv * min(2 ** (max_depth + 1) - 2, n_indiv) > F_WALK_WARN_VISITS:
            logger.warning(
                "computing F on %s rows at max depth %s; its time grows with rows and steeply "
                "with depth (README 'Deep pedigrees') — pass --no-inbreeding to skip",
                detail(f"{n_indiv:,}"),
                detail(max_depth),
            )
        with timed("inbreeding (F + n_ancestors)"):
            F_vec = pg.inbreeding()
            n_anc = None if args.safe_attempt else pg.distinct_ancestor_counts()
    else:
        logger.info("inbreeding: skipped (--no-inbreeding)")
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
        if not args.safe_attempt:
            founder_summary, n_founder_anc = compute_founder_summary(idf, mother_rows, father_rows)
            idf = idf.with_columns(
                pl.Series("n_founder_ancestors", n_founder_anc),
                pl.Series("ecg", equivalent_complete_generations(df, mother_rows, father_rows)),
            )

    if args.safe_attempt:
        # Imported here so an ordinary run does not pay for loading the release list.
        from pedsum import safe_summary

        # Nothing below reads the report payload: every safe output is laid out from the release.
        # So a safe run computes none of its sections, and counts the pairs in the pass that gives their support.
        with timed("safe-attempt pair counts and support"):
            pairs = safe_summary.pair_counts(
                pg, 5 if args.per_individual_pairs else args.max_degree, mother_rows, father_rows
            )
        source = safe_summary.build_source(
            meta=safe_meta_block(args.in_path, cmd, n_indiv),
            max_degree=args.max_degree,
            df=df,
            idf=idf,
            mating=mating,
            n_components=size["n_components"],
            sex_source=size.get("sex_source"),
            pairs=pairs,
            sex_concordance=sex_concordance,
        )
        outputs = safe_summary.safe_outputs(source, args.min_cell)
        slim_yaml, extra_yaml, pedigree_tsv = outputs.slim, outputs.extra, outputs.pedigree_tsv
        individual_tsv = None
        logger.info("safe-attempt: release list applied (min cell = %d)", Public(args.min_cell))
    else:
        with timed("sibship sizes"):
            sibships = compute_sibship_sizes(mating.sizes)

        with timed("mating-pair summary"):
            mating_pairs = compute_mating_pair_summary(mating.sizes)

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

        # --multiple-birth-col: load_and_validate carried the sets in the frame.
        multiple_births = compute_multiple_births(df, pairs) if "multiple_birth" in df.columns else None

        inb_summary = None if F_vec is None else build_inbreeding_summary(F_vec)

        with timed("aggregate pedigree sections"):
            aggregates = compute_aggregate_sections(idf, founder_summary=founder_summary)

        # The writers below walk plain dicts and lists.
        pedigree_tsv = as_plain(
            build_pedigree_data(
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
                multiple_births=multiple_births,
            )
        )
        individual_tsv = as_plain(build_individual_data(idf, args.in_path, cmd))
        slim_yaml, extra_yaml = build_summary_data(pedigree_tsv, individual_tsv)

    write_yaml(slim_yaml, paths["summary.yaml"])
    write_yaml(extra_yaml, paths["summary.extra.yaml"])
    if args.tsv:
        write_long_tsv(pedigree_tsv, paths["summary.pedigree.tsv"])
        if individual_tsv is not None:
            write_long_tsv(individual_tsv, paths["summary.individual.tsv"])
    logger.info("wrote %s/{%s}", out_dir, Public(", ".join(path.name for path in paths.values())))

    if args.safe_attempt:
        logger.info("safe-attempt: annotated.tsv.gz is not written (per-individual)")
    else:
        # Stable phase label (no per-run path) so the benchmark profiler can
        # aggregate this phase across repeats; out_dir is already in the
        # summary.yaml log line above.
        with timed("wrote annotated.tsv.gz"):
            write_annotated_tsv(
                args.in_path, ValidationConfig.from_args(args), args.sep, idf, out_dir / "annotated.tsv.gz", labels
            )

    return 0
