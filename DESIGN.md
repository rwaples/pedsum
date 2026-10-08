# DESIGN.md — pedsum internals (maintainer notes)

Pointers to where implementation details live. The README is the
collaborator-facing surface; this file is for maintainers.

## Package layout

The implementation is the `pedsum/` package; `pedigree_summary.py` is a thin
runnable shim (`python pedigree_summary.py …`) that re-exports the symbols the
test suite imports directly. Modules, in dependency order (each imports only
from those above it):

| Module | Responsibility |
|---|---|
| `pedsum/base.py` | shared constants, the `pedigree_summary` logger, `PedigreeError` |
| `pedsum/progress.py` | `relationship_progress`: the terminal progress bar for pedigree-graph's long relationship calls |
| `pedsum/pedigree_ops.py` | low-level array helpers (parent rows, parent references, sib groups, topological depth, sort-based distinct integers) |
| `pedsum/parse.py` | delimiter sniffing, column coercion, sex decoding, trait-column reading |
| `pedsum/checks.py` | per-check finding producers + check metadata (`_CHECK_*`) |
| `pedsum/validate.py` | `load_and_validate` (fail-fast) / `validate_pedigree` (accumulating) + sex imputation |
| `pedsum/pairs.py` | relationship-pair enumeration + `PedigreeGraph` construction |
| `pedsum/sections.py` | per-section summary computations |
| `pedsum/sex_concordance.py` | Offspring Sex Concordance: group projection, exact conditional moments, Holm, permutation samplers |
| `pedsum/assortative_mating.py` | Mate Correlation: trait typing, row strata, the call to pg-phenotype's `mate_correlation` and the YAML payload built from its result |
| `pedsum/schema.py` | categorised YAML schema + slim/extra split machinery |
| `pedsum/memory.py` | the best-effort memory limit: `MemoryWatchdog`, default-limit discovery, `--max-memory` parsing |
| `pedsum/report.py` | report payload builders, safe-attempt redaction, output writers, `atomic_output` |
| `pedsum/cli.py` | argparse, the `summarize` / `validate` / `effective-size` / `epimight-input` / `assortative-mating` runners, `main` |

## CLI design rationale

See [docs/adr/0001-collaborator-cli-redesign.md](docs/adr/0001-collaborator-cli-redesign.md).

## Output schema

Output schema follows [CONTEXT.md](CONTEXT.md) as of 0.10. The glossary fixes section names, key names, and the convention for distribution naming (`<noun>_count` / `<noun>_count_hist` / `<noun>_count_<sex>`). When extending the output, add the term to CONTEXT.md first, then emit keys that match it.

## Flag and behavior version history

See [CHANGELOG.md](CHANGELOG.md). Notable:
- `assortative-mating` reports the Mate Correlation (0.15; pedsum#13).
- Effective population size moves from `summarize` to `effective-size`,
  and every command runs under a memory limit (0.15; see
  [docs/adr/0004](docs/adr/0004-effective-size-subcommand-and-memory-limit.md)).
- `validate --drop-offending` emits a **Reduced Pedigree** (0.11; see
  [docs/adr/0003](docs/adr/0003-drop-offending-reduction.md)). `--no-sex-check`
  now lives in the registry (`ValidationContext.no_sex_check`) so the tolerance
  composes inside `_run_checks` rather than as a cli post-filter.
- `--allow-unknown-sex` → `--allow-missing-sex` (0.8)
- `ped_depth` sourced from `PedigreeGraph.depth` (0.4; the attribute was
  named `generation` before pedigree-graph 0.8)

## Engine selection & semantics

Pedsum uses two pair-counting paths, picked by `--per-individual-burden`
(alias `--per-individual-pairs`; no engine auto-tiering — per ADR 0001, the
matrix/BFS dispatch was removed). Both delegate to `pedigree-graph` and run
its Rust row-streaming engine, so neither builds a pair list:

- Default: `PedigreeGraph.relationship_counts(max_degree=N)`, with `N`
  from `--max-degree` (default 5; `_engine` reported as `rust_streaming`). The engine classifies every pair one row
  at a time, so the 23 counts are exact in O(N) memory; aggregate counts
  only.
- `--per-individual-burden`: `PedigreeGraph.relationship_burden()` (`_engine`
  reported as `rust_streaming_burden`). Besides the 23 category counts it
  returns each person's relative counts by degree (1-5) and the related-pair
  count within each depth, in O(N) output storage.
  `compute_relationship_summary_from_burden` (`pedsum/sections.py`) turns
  those arrays into the per-individual relationship-burden summary. It reports the same 23 counts as the default.
  Its parity oracle, a fold over explicit pair lists, lives in
  `tests/relationship_summary_oracle.py`.
- Both paths assign each pair its single closest relationship category, so
  the 23 counts partition the related pairs. A pair that is both
  parent-offspring and half sib (parent-offspring incest) counts only as
  parent-offspring. The YAML `pairs_engine` field records which path
  produced each summary.
- The experimental BFS enumerator (matrix counts *paths* / multiplicity;
  BFS counts *distinct shared ancestors*, disagreeing on `1C1R`, `H1C1R`,
  `1C2R`, `2C`) is no longer reachable from pedsum. It remains available
  to direct callers via `pedigree_graph.experimental.count_pairs_bfs`.
  Open upstream issues:
  [pedigree-graph#2](https://github.com/rwaples/pedigree-graph/issues/2),
  [pedigree-graph#3](https://github.com/rwaples/pedigree-graph/issues/3).

## Performance thresholds

- F kernel (Meuwissen-Luo): logs a WARNING when rows times the largest
  possible ancestor set, `min(2^(depth+1) - 2, rows)`, passes 4e9, so
  large or deep runs don't silently hang. See `_F_WALK_WARN_VISITS`
  (`pedsum/base.py`) and its use in `_run_summarize` (`pedsum/cli.py`).
- Descendant path counts run right after the graph build: they cost one
  pass, and an int64 overflow on a deep pedigree then exits 1 before any
  expensive phase.
- `--per-individual-burden`: O(N) output storage like the default; it
  builds no pair list.
- `epimight-input --pairs`: the one path that still materialises pairs,
  because `relative_pairs.tsv` lists them. It requests only the categories
  the `--rels` kinds read, with `execution="memory"`, and writes one kind
  at a time: beyond the engine's pair blocks it holds one kind's frame,
  and it never sorts the whole list.
- `effective-size`: all eight estimators run in memory linear in N since
  pedigree-graph 0.12.1, which took the kinship DP out of Ne_C and Ne_GC
  (its issue #38; Ne_GC had passed 12 GiB within 100 s on the 783K-row
  horse pedigree). The default selection runs all eight there in 1.11 s.
- `summarize --max-degree`: `relationship_counts` wall time on the horse
  pedigree at 10 threads was 0.61 s at degree 1, 1.36 s at 2, 221 s at 3,
  632 s at 4 and 1,833 s at 5. Codes past the cutoff are null.
- Memory limit (ADR 0004 §3): every command runs under `MemoryWatchdog`,
  which samples `/proc/self/statm` once a second and exits 3 past the
  limit (default 80% of the smallest of `MemAvailable` and each enclosing
  cgroup's headroom). It runs breach callbacks first; `effective-size`
  uses one to publish its finished estimators. Final publication and the
  breach share a lock through `MemoryWatchdog.disarm()`. Every writer
  publishes through `atomic_output`, so a stop leaves at most a
  `.<name>.partial-<pid>` file, never a truncated output.

## Assortative mating

`assortative-mating` reports the Mate Correlation over Mating Pairs; the
README covers its use and output. The computation is pg-phenotype's
`pg_phenotype.assortative.mate_correlation`. The method was designed here
in pedsum#13 and ported to pg-phenotype 0.1 unchanged. Its
[design notes](https://github.com/rwaples/pg-phenotype/blob/v0.1.0/docs/assortative-mating-design.md)
give the reasons for every choice (the Mate Network as cluster, the
two-step sandwich, the one-step bootstrap, the permutation null and its
sequential stopping, the threshold fits, thin strata, thread
independence, calibration), and pg-phenotype's
[ADR 0005](https://github.com/rwaples/pg-phenotype/blob/v0.1.0/docs/adr/0005-assortative-mating-reproduces-pedsums-numerics.md)
records that the port reproduces pedsum's numbers to the bit. The one
exception, an SE that pedsum withheld on empty-cell rounding noise, is in
the CHANGELOG.

pedsum keeps the shell. `pedsum/assortative_mating.py`:

- types each trait column (`classify_trait`) and codes binary and
  ordinal levels `0..k-1`;
- labels each row's stratum (`strata`: its Depth, or the first year of
  its birth-year bin, `-1` unknown), passed to pg-phenotype as NaN;
- maps the result onto the YAML payload: the trait records, `settings`,
  the `inference` block (pg-phenotype's `MateCorrelation.method`, written
  unchanged so the description cannot drift from the code), the counts,
  the cells, the Within-Person Cross-Trait Correlation, then `NOTES`.

**Threads.** `cli._set_assortative_threads` hands `--threads` to numba,
which caps it at `NUMBA_NUM_THREADS`, then passes numba's count to
`pg_phenotype.configure_threads`. pg-phenotype commits one budget per
process, as pedigree-graph does, so a later run in the same process (the
test suite drives several) keeps the first budget, and `settings.threads`
reports the budget the run used.

**Refusals.** pg-phenotype's errors become `PedigreeError`. The CLI never
reaches them: `classify_trait` refuses an all-missing or constant column,
and the parser takes one or two traits. A direct caller of
`compute_assortative_mating` can, with a constant trait or three traits.

**Tests.** `tests/test_assortative_mating_payload.py` pins the YAML of six
small pedigrees byte for byte. `tests/test_assortative_calibration.py`
(marked `slow`) checks size and coverage through
`compute_assortative_mating`; a three-trait design runs as its three
trait pairs. pg-phenotype carries the estimator, sandwich, bootstrap and
permutation tests, and its golden gate against pedsum#13.

## Upstream integration

Both pair-counting engines delegate to `pedigree-graph`; bug fixes
propagate on `pip install -U`. Sparse/non-contiguous IDs are still
compacted internally to dense `0..n-1`, though since pedigree-graph 0.8
that is for pedsum's own convenience — a compacted id and the graph row
it names are the same number — rather than a memory necessity: the
library indexes its own ids densely and accepts rows in any order.

## Stance: opt-outs vs auto-tiering

Size-tiered behaviors stay as flags the user types (`--no-inbreeding`,
`--max-degree`, `--per-individual-pairs`, `effective-size --estimators`),
not as auto-tiered defaults. The memory limit is not one of these: it
never changes a result, only stops a run that would otherwise be
OOM-killed ([ADR 0004](docs/adr/0004-effective-size-subcommand-and-memory-limit.md)).
The rest of the reasoning is informally documented here pending a future
ADR.
