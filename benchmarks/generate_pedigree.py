#!/usr/bin/env python3
"""Deterministic synthetic pedigree generator for the pedsum memory benchmarks.

The benchmark inputs are **not** checked in — they are regenerated from a seed so
the baselines in ``benchmarks/memory_results.md`` stay reproducible without
committing multi-hundred-MB TSVs. Re-run the command recorded in the results
file to recreate any input exactly.

Model (a representative, *non*-pathological pedigree):

* ``--generations`` discrete generations of roughly equal size; founders
  (generation 0) have ``mother == father == -1``.
* Each later generation forms ``n_children // --avg-sibship`` couples by
  sampling mothers and fathers *with replacement* from the previous
  generation, then assigns each child to a couple. Sampling with replacement
  lets a parent appear in several couples, producing realistic half-sibs as
  well as full-sibs, cousins, and the multi-hop lineal chains (grandparent …
  great³-grandparent) that exercise ``_A`` … ``_A5`` in the streaming counter.
* Sibship sizes vary around ``--avg-sibship`` because children are assigned to
  couples by a uniform draw; no single giant sibling-heavy family forms (the
  pathological case the plan warns against), as long as the couple pool is
  large.

``pg.generation`` for this structure equals the generation index, so the
observed max depth is ``--generations - 1``; pick ``--generations >= 6`` to
exercise the degree-5 adjacency power ``_A5``.

A ``<out>.meta.json`` sidecar records rows / generations / observed max depth /
extra-column count / seed so the profiler can report them without recomputing.

Usage::

    python benchmarks/generate_pedigree.py --rows 1000000 --generations 8 \
        --seed 0 --out /tmp/ped_narrow.tsv
    python benchmarks/generate_pedigree.py --rows 1000000 --generations 8 \
        --seed 0 --extra-cols 8 --out /tmp/ped_wide.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

# Span kept narrow so even the deepest generation stays inside the default
# birth-year sanity range ([1800, current calendar year + 1]); the benchmark
# can then run default `summarize` without a non-default --birth-year-max.
_BASE_YEAR = 1850
_GENERATION_INTERVAL = 20


def generate_pedigree(
    rows: int,
    generations: int,
    seed: int,
    avg_sibship: float = 2.5,
    extra_cols: int = 0,
) -> tuple[pl.DataFrame, int]:
    """Build a deterministic synthetic pedigree DataFrame.

    Returns ``(df, max_depth)`` where ``df`` has columns ``id, sex, mother,
    father, birth_year`` (plus ``extra_cols`` dummy string columns ``pad_*``)
    and ``max_depth`` is the largest generation index actually produced.
    """
    if generations < 1:
        raise ValueError("generations must be >= 1")
    rng = np.random.default_rng(seed)
    per_gen = max(2, rows // generations)

    ids: list[np.ndarray] = []
    sexes: list[np.ndarray] = []
    mothers: list[np.ndarray] = []
    fathers: list[np.ndarray] = []
    gen_index: list[np.ndarray] = []

    next_id = 1
    prev_ids: np.ndarray | None = None
    prev_sex: np.ndarray | None = None
    max_depth = 0

    for g in range(generations):
        remaining = rows - (next_id - 1)
        if remaining <= 0:
            break
        n = per_gen if g < generations - 1 else remaining
        n = min(n, remaining)
        if n <= 0:
            break

        cur_ids = np.arange(next_id, next_id + n, dtype=np.int64)
        cur_sex = rng.integers(0, 2, size=n)  # 0 = female, 1 = male

        if g == 0 or prev_ids is None:
            cur_mothers = np.full(n, -1, dtype=np.int64)
            cur_fathers = np.full(n, -1, dtype=np.int64)
        else:
            females_prev = prev_ids[prev_sex == 0]
            males_prev = prev_ids[prev_sex == 1]
            if len(females_prev) == 0 or len(males_prev) == 0:
                # Degenerate previous generation (all one sex) — treat as
                # founders rather than crash; keeps the generator robust.
                cur_mothers = np.full(n, -1, dtype=np.int64)
                cur_fathers = np.full(n, -1, dtype=np.int64)
            else:
                n_couples = max(1, int(n // avg_sibship))
                couple_mothers = rng.choice(females_prev, size=n_couples, replace=True)
                couple_fathers = rng.choice(males_prev, size=n_couples, replace=True)
                couple_of_child = rng.integers(0, n_couples, size=n)
                cur_mothers = couple_mothers[couple_of_child]
                cur_fathers = couple_fathers[couple_of_child]
                max_depth = g

        ids.append(cur_ids)
        sexes.append(cur_sex)
        mothers.append(cur_mothers)
        fathers.append(cur_fathers)
        gen_index.append(np.full(n, g, dtype=np.int64))

        prev_ids = cur_ids
        prev_sex = cur_sex
        next_id += n

    id_arr = np.concatenate(ids)
    sex_arr = np.concatenate(sexes)
    mother_arr = np.concatenate(mothers)
    father_arr = np.concatenate(fathers)
    gen_arr = np.concatenate(gen_index)

    df = pl.DataFrame(
        {
            "id": id_arr,
            "sex": np.where(sex_arr == 0, "F", "M"),
            "mother": mother_arr,
            "father": father_arr,
            "birth_year": _BASE_YEAR + gen_arr * _GENERATION_INTERVAL,
        }
    )
    if extra_cols:
        # Dummy wide payload: string columns to grow df_raw, deterministic but
        # with enough cardinality to behave like real free-text annotation.
        df = df.with_columns(pl.Series(f"pad_{k}", "v" + ((id_arr + k) % 1000).astype(str)) for k in range(extra_cols))

    return df, int(max_depth)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, required=True, help="target row count")
    p.add_argument("--generations", type=int, default=8, help="number of generations (max depth = generations - 1)")
    p.add_argument("--seed", type=int, default=0, help="numpy RNG seed (default: 0)")
    p.add_argument("--avg-sibship", type=float, default=2.5, help="mean children per couple (default: 2.5)")
    p.add_argument("--extra-cols", type=int, default=0, help="dummy string columns to append (wide variant)")
    p.add_argument("--out", type=Path, required=True, help="output TSV path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate a pedigree TSV and a ``.meta.json`` sidecar; return exit code."""
    args = _parse_args(argv)
    df, max_depth = generate_pedigree(
        rows=args.rows,
        generations=args.generations,
        seed=args.seed,
        avg_sibship=args.avg_sibship,
        extra_cols=args.extra_cols,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.out, separator="\t")
    meta = {
        "rows": len(df),
        "generations": int(args.generations),
        "max_depth": int(max_depth),
        "extra_cols": int(args.extra_cols),
        "seed": int(args.seed),
        "avg_sibship": float(args.avg_sibship),
        "columns": list(df.columns),
    }
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {args.out} ({len(df):,} rows, max_depth={max_depth}, extra_cols={args.extra_cols})")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
