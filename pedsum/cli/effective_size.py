"""``effective-size``: the Ne estimators and ``effective_size.yaml``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pedigree_graph.effective_size import ALL_EFFECTIVE_SIZE_ESTIMATORS

from pedsum.base import (
    SEX_UNKNOWN,
    PedigreeError,
    logger,
)
from pedsum.cli.common import commit_thread_budget, load_or_fail
from pedsum.memory import GiB
from pedsum.pairs import build_pedigree_graph
from pedsum.parse import read_reference_mask
from pedsum.progress import timed
from pedsum.report import (
    build_effective_size_data,
    prepare_out_dir,
    write_yaml,
)
from pedsum.sections import (
    compute_effective_size,
    compute_individual_delta_f_for_reference,
)

if TYPE_CHECKING:
    import argparse

    from pedsum.memory import MemoryWatchdog

#: Estimators that need every row's sex resolved (pedigree-graph raises
#: ``MissingMetadataError`` for them otherwise). The others run on a pedigree
#: with unknown sex, so ``effective-size`` refuses only when one is selected.
_SEX_DEPENDENT_ESTIMATORS = frozenset({"ne_sex_ratio", "ne_variance_family_size", "ne_hill_overlapping"})


def run_effective_size(args: argparse.Namespace, cmd: str, watchdog: MemoryWatchdog) -> int:
    """``effective-size``: run the selected Ne estimators and write ``effective_size.yaml``.

    A stop at the memory limit still publishes ``effective_size.yaml``, with
    the estimators that were running marked ``memory_limit`` (ADR 0004 §4).
    """
    if prepare_out_dir(args.out_dir) != 0:
        return 1
    commit_thread_budget(args)

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
        data = build_effective_size_data(args.in_path, cmd, n_total, "stopped_memory_limit", requested, partial)
        write_yaml(data, out_path)
        logger.error("wrote %s with the estimators that finished", out_path)

    watchdog.on_breach(publish_partial)
    loaded = load_or_fail(args)
    if isinstance(loaded, int):
        return loaded
    df, labels = loaded

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
            mask = read_reference_mask(
                args.in_path, args.sep, args.id_col, args.reference_col, df["id"].to_numpy(), labels
            )
        except PedigreeError as e:
            logger.error("%s", e)
            return 1
        reference_rows = np.flatnonzero(mask)
        if reference_rows.size == 0:
            logger.error("reference column %r marks no row", args.reference_col)
            return 1
        if "ne_individual_delta_f" not in requested:
            logger.warning("--reference-col only affects ne_individual_delta_f, which was not requested")

    with timed("built PedigreeGraph"):
        pg = build_pedigree_graph(df)
    n_total = len(df)
    with timed(f"effective size ({', '.join(requested)})"):
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
            data = build_effective_size_data(args.in_path, cmd, len(df), "complete", requested, results)
            write_yaml(data, out_path)
            logger.info("wrote %s", out_path)
    return 0
