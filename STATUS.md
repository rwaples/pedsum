# pedsum status — post-port

## Resolved in pedsum 0.8 (unified missing-sex tolerance)

Continues the [ADR-0001](docs/adr/0001-collaborator-cli-redesign.md)
collaborator-friendly principle that drove 0.7.

- **Renamed `--allow-unknown-sex` → `--allow-missing-sex`.** Hard break,
  no alias.
- **Broadened scope.** The renamed flag now tolerates BOTH orphan
  unsexed rows and role-ambiguous rows (unsexed individuals used as
  both mother and father). 0.7 hard-blocked role-ambiguity
  unconditionally; 0.8 makes it an opt-in tolerance like orphan rows.
- **Fixed-TSV auto-fix.** `validate.tsv.gz` now writes `"-1"` for every
  row whose final sex is `SEX_UNKNOWN`, regardless of provenance. The
  rewrite no longer requires `n_imputed > 0`, so orphan-only pedigrees
  also get the normalisation.
- The `_run_validate` block-list messages and the sex-stratified
  `_run_summarize` refusal now name `--allow-missing-sex`.

## Resolved in pedsum 0.7 (collaborator-friendly CLI redesign)

Breaking. Full rationale and rejected alternatives in
[`docs/adr/0001-collaborator-cli-redesign.md`](docs/adr/0001-collaborator-cli-redesign.md);
sequenced changes and per-test acceptance in
[`PLAN.md`](PLAN.md); migration table in
[`CHANGELOG.md`](CHANGELOG.md).

- **`--out` is now a directory.** Files inside get fixed names
  (`summary.yaml`, `summary.extra.yaml`, `annotated.tsv.gz`, and with
  `--tsv` the two long-form TSVs). Passing a path that already exists
  as a regular file exits rc=1 cleanly.

- **`--inbreeding` / `--effective-size` flipped to opt-out.** Bare
  `summarize` now produces F and the seven cheap Ne estimators by
  default; `--no-inbreeding` / `--no-effective-size` skip them.
  `--ne-coancestry` stays opt-in. On N > 1,000,000 pedigrees pedsum
  logs a one-line cost-estimate before the F kernel runs.

- **`--burden` renamed `--per-individual-pairs`.** The CLI flag is
  the only thing renamed; the output schema's `relationship_burden`
  section keeps that name.

- **Flags deleted:** `--engine`, `--bfs-threshold`,
  `--zero-as-missing`, `--single-file`. argparse runs with
  `allow_abbrev=False` so a deleted long-option cannot be silently
  abbreviated into a surviving one.

- `_BFS_AUTO_THRESHOLD` is no longer reachable from the CLI but
  remains as an internal default in `count_relationship_pairs`. The
  BFS engine itself stays available via
  `pedigree_graph.experimental.count_pairs_bfs`.

## Resolved in pedsum 0.6.x (post-0.6 follow-ups)

- **Birth-year validation in `validate`.** `--birth-year-col NAME` is
  now accepted by the validate subcommand and runs three checks:
  `birth_year_dtype` (numeric parsing), `birth_year_range` (within
  `[--birth-year-min, --birth-year-max]`; defaults `1800` and the
  current calendar year + 1), and `birth_year_topology` (child's
  birth_year >= each known parent's birth_year). Findings appear in
  `.validate.log` like any other check. summarize now also surfaces
  range / topology violations as a clean `PedigreeError` (return
  code 1) instead of letting `PedigreeGraph` raise a `ValueError`
  mid-pipeline with a traceback.

## Resolved in pedsum 0.6 (collaborator-friendly polish)

- **Sex-encoding auto-detection.** `--sex-encoding=auto` (the new default)
  picks `default` (0=F, 1=M) or `plink` (1=M, 2=F, 0=unknown) from the
  observed tokens. `--plink-sex` remains as a legacy alias. Word-only
  columns get no warning (encoding moot); only-`1` numeric columns get a
  WARNING since the encoding is genuinely ambiguous. Under `plink`, sex=0
  is unconditionally treated as missing (PLINK fam spec) without needing
  `--zero-as-missing`.

- **Missing-sex imputation.** Missing-sex tokens (the parent-missing set
  plus `-1`, `U`, `Unknown`) decode to a sentinel and are then imputed
  from parent role: F if the row is used as a mother, M if used as a
  father. Two new checks surface the residual cases:
  `sex_role_ambiguity` (present individual with unknown sex referenced as
  BOTH mother and father — always a hard block) and `unknown_sex` (orphan
  unsexed row — hard block unless `--allow-unknown-sex` is passed, in
  which case it shows as SKIP with a tolerated-count note, mirroring the
  existing `--no-sex-check` precedent). The imputed sex is folded into
  `.validate.tsv.gz` so the fixed output reflects the auto-fix.

- **Row-order check dropped, reorder silent.** The `topological_row_order`
  check is gone. Both `summarize` and `validate` reorder rows into
  topological order silently (single INFO log line with the moved count).
  The `.validate.tsv.gz` output is always parents-before-children.

- **Grouped validation summary.** The stderr summary now prints four
  named sections ("Columns & parsing", "IDs", "Parent references",
  "Graph structure") with human-friendly dot-padded labels. Internal
  check names are preserved in `.validate.log` so parsing contracts are
  unchanged.

- **`--effective-size` / `--inbreeding` refuse `sex=-1`.** When
  `--allow-unknown-sex` leaves rows with sentinel sex AND the user asks
  for sex-stratified Ne or the F kernel, summarize exits 1 with a clear
  message rather than silently miscounting.



The BFS / boolean-matmul / numba relationship-pair engine that used to
live inline in `pedigree_summary.py` (under the misleading name
`_count_pairs_graphtool` / `--engine=graph-tool`) has been moved to the
`pedigree-graph` package as a public-but-experimental submodule:
`pedigree_graph.experimental.count_pairs_bfs`.

Pedsum is now a thin wrapper over both engines via
`pedigree_graph.PedigreeGraph.count_pairs` (matrix) and
`pedigree_graph.experimental.count_pairs_bfs` (bfs).

## Resolved in the port (no longer pedsum's problem)

- Engine renamed `graph-tool` → `bfs` everywhere (CLI flag, dispatcher,
  YAML `pairs_engine` field, README, environment.yml).
- Dead `import graph_tool.all` probe + `_GRAPH_TOOL_AVAILABLE` gate
  removed.
- Hostile module-load `numba.set_num_threads(8)` global call removed.
- BFS-only helpers (`_dedup_pairs`, `_pairs_from_groups`, `_po_pairs`,
  `_lineal_pairs_lo_hi`, `_subtract_pair_keys`, `_run_per_anchor`,
  `_enumerate_pairs_kernel`) deleted from pedsum.
- Latent bug in `_count_after_subtract` (a dead list-comprehension that
  silently dropped accumulated subtract keys on a base-id bump) fixed
  during the port; covered by a unit test in
  `pedigree-graph/tests/test_experimental.py::test_count_after_subtract_handles_subtract_with_larger_hi`.
- BFS engine now reports `MZ` correctly (delegating to
  `PedigreeGraph._mz_twin_pairs()`) for callers who supply twin
  data. **Pedsum's input format has no twin column**, so pedsum users
  still see `MZ=0` from both engines — this is by design, not a
  regression.

## Open follow-ups (re-homed against `pedigree-graph`)

These were carried forward from the BFS engine's original home (this
repo) and now belong to the package:

1. **Confirm numba kernel actually parallelizes at scale.** STATUS at
   the time of port: kernel was killed at 2:49 on a 2M-row pedigree
   (`d00`) with only ~102% CPU, suggesting `prange` was not engaging.
   The instrumented `[bfs]` log lines (`[bfs] _enumerate_shared(a,b)
   X.XXs (kernel=Y.YYs ...)`) make this easy to verify on a clean
   run. Should the kernel actually parallelize, the BFS engine becomes
   a viable scalability story; if not, the package may delete it.
2. **Test scaling on a 10M+ pedigree** (the original target where the
   matrix engine is expected to OOM). Until this is run, the BFS
   engine has not been demonstrated to beat the matrix engine at any
   pedigree size we care about.
3. **`int8` overflow risk in P_k boolean matmul.** Pedsum uses
   `np.int8` for `P_k.data`. Theoretically vulnerable to silent
   path-count overflow under extreme consanguinity (>127 distinct
   paths to a single (i, X) pair before the `M.data[:] = 1` clamp).
   Empirically not seen on tested pedigrees. Switch to `int32` if it
   ever bites.

These items live in this STATUS.md only as a paper trail — open the
matching issues / TODOs in the `pedigree-graph` repo when you act on
them.

## Resolved in pedsum 0.4 (pedigree-graph v0.5.0 consolidation)

- Pedsum's recursive memoized kinship `compute_inbreeding` (and the
  helper `_merge_sorted_rows`) deleted; F is now sourced from
  `pedigree_graph.PedigreeGraph.compute_inbreeding()` (Meuwissen-Luo).
- Pedsum's `compute_descendants` (path-count BFS) and
  `compute_generations` (Kahn) deleted; both moved upstream to
  `PedigreeGraph.compute_n_descendants()` and `PedigreeGraph.generation`
  respectively.
- The `n_ancestors` column is now sourced from
  `PedigreeGraph.compute_n_ancestors()` (distinct-count, sparse
  transitive closure).
- `PedigreeGraph` is built once in `_run_summarize` and threaded through
  every primitive that needs it (relationship pairs, F, lineage
  counts, Ne), rather than each engine constructing its own.
- `--effective-size`, `--ne-coancestry`, `--ne-threads N` flags added
  for pedigree-based Ne estimation.

## Open follow-up against pedigree-graph

1. **`PedigreeGraph.compute_n_ancestors` scalability.** The current
   implementation is a sparse boolean transitive closure of the parent
   graph (`_lineage_kernel._compute_n_ancestors`).  Memory scales with
   `sum_i n_ancestors[i]`, so very deep / very wide pedigrees may hit
   RAM limits — at N=100K, G=10 with random mating it runs in 2.2 s and
   peak RSS ~0.5 GB; extrapolating to N=10M with saturated ancestry
   exceeds commodity hardware.  A retirement-style DP (analogous to the
   F kernel's row-retirement) would bound peak memory to the live
   frontier; deferred until a user hits the wall.

2. **Relationship-pair counting** — the matrix and BFS engines
   materialise pair arrays before counting, so peak memory scales
   with `O(answer size)` not `O(N)`.  Stallion-heavy livestock
   pedigrees (e.g., one sire with 2,500 offspring → ~156 M paternal
   half-sib pairs) OOM at 30 GB.

   **Pedsum 0.5 (2026-05-15) flipped the default to streaming.**
   By default `summarize` uses
   `pedigree_graph.PedigreeGraph.count_pairs_streaming` (scalar,
   O(N) memory) to populate the 23 pair counts.  Bit-identical to
   the matrix engine for the 10 simple codes; ~1% scalar
   approximation on the 13 cousin / collateral codes when the
   pedigree has inbreeding, twins, or shallow depth.  Per-individual
   relationship-burden summary stays a stub.

   The opt-in `--burden` flag restores the matrix / BFS engine
   path with full pair-list output (and the per-individual burden
   summary).  OOMs on pair-dense pedigrees; use only when burden
   is required.

   See [`../pedigree-graph/LIMITATIONS.md`](../pedigree-graph/LIMITATIONS.md)
   for the precision contract and what would unlock exact
   matrix-engine semantics at scale.
