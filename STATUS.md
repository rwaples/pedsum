# pedsum status — post-port

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
