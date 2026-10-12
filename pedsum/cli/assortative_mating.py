"""``assortative-mating``: the Mate Correlation and ``assortative_mating.yaml``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from pedsum.base import (
    PedigreeError,
    logger,
)
from pedsum.cli.common import commit_thread_budget, load_or_fail
from pedsum.parse import read_trait_columns
from pedsum.progress import timed
from pedsum.report import (
    ASSORTATIVE_MATING_FIGURES,
    _build_assortative_mating_data,
    _prepare_out_dir,
    _write_yaml,
)

if TYPE_CHECKING:
    import argparse


def _parse_cpu_list(text: str) -> set[int]:
    """CPU numbers of a sysfs list such as ``0,6`` or ``0-3,8``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def physical_cores(sysfs: Path = Path("/sys/devices/system/cpu")) -> int:
    """Physical cores among the CPUs this process may run on: its affinity set, with SMT siblings counted once.

    A CPU's siblings come from ``topology/core_cpus_list`` (``thread_siblings_list``
    on older kernels); a CPU without either counts as a core of its own. Without
    ``sched_getaffinity`` (not Linux) the answer is ``os.cpu_count()``.
    """
    try:
        cpus = os.sched_getaffinity(0)
    except AttributeError:
        return os.cpu_count() or 1
    cores: set[frozenset[int]] = set()
    for cpu in cpus:
        topology = sysfs / f"cpu{cpu}" / "topology"
        for name in ("core_cpus_list", "thread_siblings_list"):
            try:
                cores.add(frozenset(_parse_cpu_list((topology / name).read_text())))
                break
            except OSError, ValueError:
                continue
        else:
            cores.add(frozenset([cpu]))
    return max(1, len(cores))


def _set_assortative_threads(threads: int) -> None:
    """Hand ``--threads`` to pg-phenotype, capped as numba caps it.

    numba caps a process at ``NUMBA_NUM_THREADS`` (the test task pins it to 1)
    and refuses more; pg-phenotype gets the count numba keeps.  Like
    pedigree-graph, pg-phenotype commits one budget per process.
    """
    import numba
    import pg_phenotype

    try:
        numba.set_num_threads(threads)
    except ValueError:
        logger.debug("numba allows fewer than --threads=%d thread(s) in this process; keeping its maximum", threads)
    try:
        pg_phenotype.configure_threads(numba.get_num_threads())
    except RuntimeError:
        logger.debug("pg-phenotype thread budget already committed; --threads=%s ignored", threads)


def run_assortative_mating(args: argparse.Namespace, cmd: str) -> int:
    """``assortative-mating``: type the trait columns and write ``assortative_mating.yaml``."""
    from pedsum.assortative_mating import classify_trait, compute_assortative_mating

    if _prepare_out_dir(args.out_dir) != 0:
        return 1
    commit_thread_budget(args)
    _set_assortative_threads(physical_cores() if args.threads is None else args.threads)
    loaded = load_or_fail(args)
    if isinstance(loaded, int):
        return loaded
    df, labels = loaded

    try:
        tokens = read_trait_columns(
            args.in_path, args.sep, args.id_col, args.trait, df["id"].to_numpy(), labels, args.trait_missing
        )
        traits = [classify_trait(name, tokens[name], args.trait_type.get(name)) for name in args.trait]
    except PedigreeError as e:
        logger.error("%s", e)
        return 1
    del tokens  # per-row Python str objects: ~2 GiB at 10^7 pairs, unused after typing
    for trait in traits:
        logger.info(
            "trait %s: %s (%s), %d missing of %d",
            trait.name,
            trait.kind,
            trait.type_source,
            trait.n_missing,
            len(df),
        )

    try:
        with timed("assortative mating"):
            payload = compute_assortative_mating(
                df,
                traits,
                permutations=args.permutations,
                bootstrap=args.bootstrap,
                seed=args.seed,
                stratify_by=args.stratify_by,
                birth_year_bin=args.birth_year_bin,
                min_stratum_networks=args.min_stratum_networks,
            )
    except PedigreeError as e:
        logger.error("%s", e)
        return 1
    out_path = args.out_dir / "assortative_mating.yaml"
    data = _build_assortative_mating_data(args.in_path, cmd, len(df), payload)
    _write_yaml(data, out_path, figures=ASSORTATIVE_MATING_FIGURES)
    logger.info("wrote %s", out_path)
    return 0
