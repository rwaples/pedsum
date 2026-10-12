"""``epimight-input``: the EPIMIGHT long-form skeleton and, with ``--pairs``, the relative pairs."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from pedsum.base import (
    logger,
)
from pedsum.cli.common import commit_thread_budget, load_or_fail
from pedsum.epimight import (
    PLACEHOLDER_COLUMNS,
    build_epimight_skeleton,
    iter_relative_pairs,
    relationship_diagnostics,
    validate_relationship_codes,
)
from pedsum.pairs import build_pedigree_graph
from pedsum.progress import timed
from pedsum.report import (
    atomic_output,
    prepare_out_dir,
)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator


def _write_epimight_table(frame: pl.DataFrame, out_dir: Path, stem: str, *, parquet: bool) -> None:
    """Write ``frame`` to ``<out_dir>/<stem>.tsv`` (and ``.parquet`` if requested)."""
    tsv_path = out_dir / f"{stem}.tsv"
    with atomic_output(tsv_path) as tmp:
        frame.write_csv(tmp, separator="\t")
    logger.info("wrote %s (%d rows)", tsv_path, len(frame))
    if parquet:
        parquet_path = out_dir / f"{stem}.parquet"
        with atomic_output(parquet_path) as tmp:
            frame.write_parquet(tmp)
        logger.info("wrote %s", parquet_path)


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


def run_epimight_input(args: argparse.Namespace) -> int:
    """``epimight-input``: emit the EPIMIGHT long-form skeleton from a pedigree."""
    commit_thread_budget(args)
    if prepare_out_dir(args.out_dir) != 0:
        return 1
    try:
        rels = validate_relationship_codes([r.strip() for r in args.rels.split(",") if r.strip()])
    except ValueError as e:
        logger.error("%s", e)
        return 2

    loaded = load_or_fail(args)
    if isinstance(loaded, int):
        return loaded
    df, labels = loaded

    with timed("built PedigreeGraph"):
        pg = build_pedigree_graph(df)

    # The skeleton counts relatives without a pair list. --pairs extracts the
    # list for its own export, after the skeleton's counts are freed.
    with timed("epimight skeleton"):
        # person_id comes from df's id column: relabel it once, at N rows,
        # not the 8N-row skeleton.
        frame = build_epimight_skeleton(
            labels.relabel(df, "id"),
            pg,
            rels=rels,
            disorder=args.disorder,
            base_year=args.base_year,
            drop_founders=args.drop_founders,
        )

    out_dir = args.out_dir
    _write_epimight_table(frame, out_dir, "pipeline_input", parquet=args.parquet)
    # Summarize the skeleton now and free it, so it is not resident under --pairs.
    diagnostics = relationship_diagnostics(frame, rels)
    del frame

    if args.exact_kinship and not args.pairs:
        logger.warning("--exact-kinship has no effect without --pairs")

    if args.pairs:
        with timed("relative pairs"):
            _write_relative_pairs(
                (
                    labels.relabel(pairs, "id1", "id2")
                    for pairs in iter_relative_pairs(df, pg, rels=rels, exact_kinship=args.exact_kinship)
                ),
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
