# pedsum memory benchmarks

Tooling for the *measure-first* memory-reduction work on `pedsum summarize`
(see the plan in `i-want-to-make-structured-narwhal.md`). No optimization is
merged before its before/after peak-RSS delta is recorded in
`memory_results.md`.

## What's here

- `generate_pedigree.py` — deterministic synthetic pedigree generator
  (seed + parameters). **Generated inputs are not checked in**; regenerate them
  from the commands below so baselines stay reproducible without committing
  large TSVs.
- `profile_memory.py` — in-process peak-RSS profiler. Runs
  `pedsum.cli._run_summarize` (reusing `_parse_args`) while a background thread
  samples `/proc/self/statm`, attributing per-phase peaks via
  `pedsum.cli._current_profile_phase()`.
- `memory_results.md` — the recorded baseline / before-after table.

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
# Streaming, narrow
python benchmarks/profile_memory.py --label narrow/streaming --repeats 3 --warmup 1 \
    --json-out benchmarks/results/narrow_streaming.json \
    -- summarize --in /tmp/ped_1m_narrow.tsv --out /tmp/prof_out --birth-year-col birth_year

# Matrix, narrow
python benchmarks/profile_memory.py --label narrow/matrix --repeats 3 --warmup 1 \
    --json-out benchmarks/results/narrow_matrix.json \
    -- summarize --in /tmp/ped_200k_narrow.tsv --out /tmp/prof_out \
       --birth-year-col birth_year --per-individual-pairs
```

Diagnostic rows that isolate pair-count RSS (later F/Ne peaks can otherwise
hide whether releasing pair matrices lowered the overall peak):

```bash
python benchmarks/profile_memory.py --label narrow/streaming/no-fne --repeats 2 --warmup 1 \
    -- summarize --in /tmp/ped_1m_narrow.tsv --out /tmp/prof_out --birth-year-col birth_year \
       --no-inbreeding --no-effective-size
```

For expensive ~1M-row cells use `--repeats 2`; for smaller inputs `--repeats 3`.
Pass `--baseline <prior.json>` to flag (not block) >20% runtime regressions.

## Repeats / interpretation

- 1 warm-up + 3 measured repeats (small/medium) or ≥2 (expensive ~1M-row).
- The profiler reports median peak RSS and the observed range; RSS varies with
  allocator/cache state, so compare medians.
- Per-phase peaks attribute RSS to the active `_timed` block. Samples taken
  outside any block (e.g. YAML serialization) land under `(unprofiled)`. This
  intentionally prioritizes phase attribution over CLI startup/import RSS.
