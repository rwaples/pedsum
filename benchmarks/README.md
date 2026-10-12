# pedsum memory benchmarks

Tooling for measuring pedsum's wall time and peak RSS. Optimizations and
refactors are gated on before/after medians (`profile_memory.py --compare`);
`memory_results.md` records the memory-reduction work on `pedsum summarize`.

## What's here

- `generate_pedigree.py` — deterministic synthetic pedigree generator
  (seed + parameters). **Generated inputs are not checked in**; regenerate them
  from the commands below so baselines stay reproducible without committing
  large TSVs.
- `profile_memory.py` — wall-time and peak-RSS profiler for any subcommand.
  Each repeat runs in a child process; peak RSS is `wait4`'s `ru_maxrss`, and
  a sampler in the child attributes approximate per-phase peaks via
  `pedsum.progress.current_profile_phase()`. `--compare BASELINE.json` exits 1
  when the exit codes differ from the baseline's, or when the median command
  time (the child's `main()`, without interpreter startup and imports) or the
  median peak RSS exceeds 1.05x the baseline's.
- `bench_pytest_workers.sh` — repeatable pytest-xdist worker/thread sweep. It
  compares the serial suite with pinned and unpinned parallel cells and samples
  aggregate RSS across the whole pytest process tree.
- `bench_pedsum.sh` with `simace_scenarios.yaml` — times `pedsum summarize` on a
  pedigree simACE simulates (`simace run <scenario> --until simulate`); see the
  script header.
- `memory_results.md` — the recorded baseline / before-after table.

The `assortative-mating` subcommand has its own tooling:

- `generate_assortative_mating.py` writes a synthetic pedigree with a
  known Mate Correlation (`--pairs N --seed S`, `--r-mf` for the mother ×
  father block), with remating and birth-decade strata.
- `bench_assortative_mating.py` runs the CLI repeatedly under
  `/usr/bin/time` and records the median wall time and peak RSS per point.
  `results/assortative_mating_cost.md` holds the measurements and the
  README "Cost" table.
- `validate_assortative_mating.py` cross-checks the command against
  simACE on simACE's own pedigrees. It is manual and needs a simACE
  checkout with built replicates. It writes
  `results/assortative_mating_simace.md` and exits 1 when a check fails.

Run the pytest sweep from the repo root:

```bash
bash benchmarks/bench_pytest_workers.sh
```

Set `SWEEPS`, `CELLS`, or pass pytest arguments to narrow a diagnostic run:

```bash
SWEEPS=1 CELLS="n8t1" bash benchmarks/bench_pytest_workers.sh tests/test_pairs_properties.py
```

## Regenerating the benchmark inputs

Run from the repo root. The four cells are {narrow, wide} × {streaming,
matrix}. Streaming uses ~1M rows; the matrix (`--per-individual-pairs`) cells
use a smaller size that completes without OOM (pair-list materialization is the
memory-dangerous behavior being measured). `--generations 8` gives observed max
depth 7, which exercises `_A` … `_A5` in the streaming counter.

```bash
# Streaming cells (~1M rows)
python benchmarks/generate_pedigree.py --rows 1000000 --generations 8 --seed 0 \
    --out /tmp/ped_1m_narrow.tsv
python benchmarks/generate_pedigree.py --rows 1000000 --generations 8 --seed 0 \
    --extra-cols 8 --out /tmp/ped_1m_wide.tsv

# Matrix cells (smaller; see memory_results.md for the size actually used)
python benchmarks/generate_pedigree.py --rows 200000 --generations 8 --seed 0 \
    --out /tmp/ped_200k_narrow.tsv
python benchmarks/generate_pedigree.py --rows 200000 --generations 8 --seed 0 \
    --extra-cols 8 --out /tmp/ped_200k_wide.tsv
```

Each run also writes a `<out>.meta.json` sidecar (rows / max depth /
extra-column count / seed) that the profiler reads for its metadata line.

## Capturing a baseline / before-after

```bash
# Before the change: 1 warm-up + 5 measured runs.
python benchmarks/profile_memory.py --label validate/1m --repeats 5 \
    --json-out /tmp/validate_before.json \
    -- validate --in /tmp/ped_1m_narrow.tsv --out /tmp/prof_out --birth-year-col birth_year

# After the change: the same command gated against the baseline (exit 1 on a regression).
python benchmarks/profile_memory.py --label validate/1m --repeats 5 \
    --compare /tmp/validate_before.json \
    -- validate --in /tmp/ped_1m_narrow.tsv --out /tmp/prof_out --birth-year-col birth_year
```

Everything after `--` is a pedsum argv, so any subcommand works.
`summarize` at `--max-degree 5` takes tens of minutes on 1M rows; gate
it at `--max-degree 2` unless the change touches pair counting.

Since 0.15.0 `summarize` computes no Ne, so the peaks and runtimes in
`memory_results.md`, recorded before then, are not comparable to a current run.

## Repeats / interpretation

- 1 warm-up + 5 measured repeats. A single-run comparison is not evidence.
- The profiler reports medians and the observed range; RSS varies with
  allocator/cache state, so compare medians. Check the CPU clock and load
  before measuring.
- Per-phase peaks attribute RSS to the active `timed` block. Samples taken
  outside any block (e.g. YAML serialization) land under `(unprofiled)`. This
  intentionally prioritizes phase attribution over CLI startup/import RSS.
