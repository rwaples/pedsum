# Changelog

## 0.12.0 — 2026-06-26 — epimight-input subcommand

### Features

- **`epimight-input`** emits the *structural skeleton* of an EPIMIGHT long-form input from a pedigree — one row per `person × disorder × relationship_kind` over the eight EPIMIGHT relationship codes (`PO, FS, HS, mHS, pHS, Av, 1G, 1C`). The columns a pedigree determines are computed (`person_id`, `relationship_kind`, `relatives` via the pedigree-graph pair extractor, `born_at_year` from a birth-year column or `--base-year + generation`); the columns that need phenotype/affection/demography (`failure_status`, `failure_time`, `relatives_diagnosed`, `dead_at_year`) are emitted as empty `<NA>` placeholders (schema-correct nullable integers) for filling downstream. Reuses `summarize`/`validate`'s loader (delimiter sniffing, sex decode, QC, topological sort), so the same text pedigrees work. Writes `pipeline_input.tsv` by default; `--parquet` additionally writes the parquet EPIMIGHT's R Pipeline reads natively (requires `pyarrow`, imported lazily). `--rels` selects/orders the kinds, `--disorder` labels the single emitted block, `--drop-founders` matches the fitACE emitter's founder drop. The relationship grouping mirrors `fitace.relationships` (pedsum cannot import the private `fitace`); keep them in sync.
- **`epimight-input --pairs`** additionally writes `relative_pairs.tsv` — the list of relative pairs backing the skeleton's counts: one row per pair per `relationship_kind` with columns `id1, id2, relationship_kind, kinship`. Directional kinds (`PO`, `Av`, `1G`) put the younger member in `id1`; symmetric kinds are canonicalized `id1 < id2`. Shares the single `extract_pairs()` with the skeleton, honors the same `--rels`, and is written as parquet too under `--parquet`. Because the EPIMIGHT codes overlap, a maternal half-sib pair is listed under both `HS` and `mHS` and MZ twins appear as `FS`, so `kinship` is the nominal per-kind coefficient. Materialises every pair, so it can be large on pair-dense pedigrees.

## 0.11.2 — 2026-06-15 — summarize regenerates re-ingested provenance columns

### Fixes

- **`summarize` no longer warns when re-ingesting its own provenance columns.** A `validate.tsv.gz` (including one produced by `validate --drop-offending`) carries pedsum's own `sex_source` column; summarizing it re-derived `sex_source` and the collision guard preserved the stale copy as `sex_source_input` plus a `WARNING` — so the canonical clean-then-summarize pipeline always emitted spurious noise and a redundant column. pedsum-reserved provenance columns (currently `sex_source`) are now regenerated: the stale input copy is dropped with an `INFO` line. Genuine user-column collisions are unchanged (still preserved as `<name>_input` with a `WARNING`).

## 0.11.1 — 2026-06-15 — per-round drop-offending logging

### Features

- **`validate --drop-offending` logs each reduction round at INFO** — `drop-offending round K: dropped N individual(s) — <by-check breakdown>; removed R row(s), cleared P reference(s)` — so the changes made each iteration are visible while it converges. The full per-id detail remains in `validate.dropped.tsv`.

## 0.11.0 — 2026-06-15 — validate --drop-offending (Reduced Pedigree)

### Features

- **`validate --drop-offending`** emits a **Reduced Pedigree** — it iteratively removes the individuals named in droppable check findings (clearing references to them) until the pedigree passes under the invoked flags, after all existing sex imputation and founder synthesis. Removal is clear-refs (a dropped parent's children become **half-founders**), not cascade; the loop re-imputes each round so a parent orphaned by a drop is re-evaluated. Writes a `validate.dropped.tsv` manifest (`id`, `check`, `round`), logs `dropped N individual(s) / R row(s) of M (X%) over K round(s); cleared P reference(s)`, warns when >10% of rows are removed (relatedness / Ne / founder counts reflect the reduced set), and exits 1 whenever anything was dropped. Column/parse-level failures (missing column, non-integer id, negative id, …) still `BLOCK`. `acyclic` drops only true cycle members (descendants survive). See [docs/adr/0003](docs/adr/0003-drop-offending-reduction.md).

### Internal

- `--no-sex-check` now SKIPs `parent_refs_sex_conflict` inside the check registry (`ValidationContext.no_sex_check`) instead of a `validate`-side post-filter, so the tolerance composes everywhere `_run_checks` runs (validate, the drop loop, and its self-verify). No change to observable `--no-sex-check` behavior.

## 0.10.2 — 2026-06-12 — validate.log pins the offending rows

### Fixes

- **`validate.log` now names the offending rows for the role-conflict checks.** `sex_role_consistency` and `parent_refs_sex_conflict` previously recorded only the id (the `row` column was blank), so a pedigree where an individual is used as both a mother and a father told you the id but not which lines to fix. The findings now carry the id's own row in the `row` column (where it has one) and list the referencing rows in the detail — e.g. `id=3 used as father in row(s) [4, 5] but sex != male`, and `id=99 referenced as both mother (row(s) [2, 4]) and father (row(s) [3])`. This matches what `sex_role_ambiguity` already did.

## 0.10.1 — 2026-06-12 — summarize writes validate.log on validation failure

### Fixes

- **`summarize` now writes `validate.log` when validation hard-fails.** Previously a fail-fast validation error — e.g. an individual used as a mother in one row and a father in another — printed a single truncated console line and left the output directory empty, leaving the user no record to act on. summarize now re-runs the accumulating validator on the failure path and writes the same `validate.log` the `validate` subcommand produces, capturing every finding across all failing checks (not just the first). Clean pedigrees are unaffected (no log is written on success).

## 0.10.0 — 2026-06-12 — output schema renamed to match CONTEXT.md glossary

**Breaking** — `summary.yaml`, `summary.extra.yaml`, `summary.pedigree.tsv` and `annotated.tsv.gz` all rename sections, keys, and per-individual columns to match the canonical terminology fixed in [CONTEXT.md](CONTEXT.md). The flat dict that feeds both YAML and long-form TSV is rewritten, so TSV column headers change too.

### Behavior change (separate from rename)

- `reproduction.mate_count_male` / `reproduction.mate_count_female` are summary-stats distributions over **all** males / all females respectively (zero-included for the unmated). The previous fields `mating_pairs.male_mate_count` / `female_mate_count` summed only over parents-with-children. Means, quantiles, and totals will shift. Use `n_mates == 0` count if you want the old denominator.

### Section renames

| 0.9 | 0.10 |
|---|---|
| `pedigree.demography.family_size` | `pedigree.demography.sibship_size` (per-Sibship stats only; per-individual contents moved out) |
| `pedigree.lineage.lineage` | split into `pedigree.individuals.reproduction` and `pedigree.individuals.genealogy` |
| `pedigree.lineage.founder_generation` | `pedigree.founders.founder_summary` |
| `pedigree.lineage.founder_contribution` | `pedigree.founders.founder_contribution` |
| `pedigree.relatedness.pairs` | `pedigree.relatedness.relationship_pairs` |
| `pedigree.strata.generation_summary` | `pedigree.strata.depth_summary` |

### Field renames (selected)

| 0.9 | 0.10 |
|---|---|
| `size_structure.gen_counts` | `size_structure.depth_counts` |
| `family_size.n_families` | `sibship_size.n_sibships` |
| `family_size.person_dist*` | `reproduction.offspring_count_hist*` (now over all individuals) |
| `family_size.frac_with_full_sib` | `reproduction.frac_with_full_sib` (same denominator) |
| `lineage.offspring` | `reproduction.offspring_count` |
| `lineage.mates` | `reproduction.mate_count` |
| `lineage.descendants` | `genealogy.descendant_paths` |
| `lineage.ancestors` | `genealogy.distinct_ancestors` |
| `founder_contribution.descendants_per_founder` | `founder_contribution.descendant_paths_per_founder` |
| `founder_generation.by_generation[*].gen` | `founder_summary.by_depth[*].depth` |
| `founder_generation.by_generation[*].founder_lines_per_individual` | `founder_summary.by_depth[*].founder_ancestors` |
| `founder_generation.bottleneck.min_active_generations` | `founder_summary.bottleneck.min_active_depths` |
| `founder_generation.bottleneck.min_effective_generations` | `founder_summary.bottleneck.min_effective_depths` |
| `relationship_summary.n_possible_pairs` | `relationship_summary.n_individual_pairs` |
| `relationship_summary.related_pair_density_by_generation[*].gen` | `relationship_summary.related_pair_density_by_depth[*].depth` |
| `pairs_engine` (sibling of `pairs`) | folded into `relationship_pairs.engine` |
| `generation_summary[*].gen` | `depth_summary[*].depth` |
| `sex_summary.<sex>.n_without_children` | `sex_summary.<sex>.n_terminal` |
| upstream `ne_inbreeding.n_generations_used` (and `ne_caballero_toro`) | normalised to `n_depths_used` inside pedsum |

### Fields dropped

- `mating_pairs.female_mate_count`, `male_mate_count`, `n_females_with_multiple_mates`, `n_males_with_multiple_mates` (per-individual mate-count stats moved to `reproduction:` as `mate_count` / `mate_count_male` / `mate_count_female`).
- `family_size.mates_female_mean`, `mates_male_mean` (derivable from `reproduction.mate_count`).
- `founder_contribution.descendant_count_semantics` (the new field names carry the semantics).

### Per-individual column renames in `annotated.tsv.gz`

| 0.9 | 0.10 |
|---|---|
| `n_descendants` | `n_descendant_paths` (path counts; same values) |
| `n_ancestors` | `n_distinct_ancestors` (distinct-individual counts; same values) |
| — | `n_founder_ancestors` (new — count of distinct **Founder Ancestors** per individual) |

### Migration

- Zero downstream consumers parse pedsum YAML/TSV today (verified against fitACE and simACE). No grace period is provided.
- If you have scripts that read these outputs, rewrite the paths per the tables above. A complete diff of expected keys lives in `tests/test_07_redesign.py` and `tests/test_summary_split.py`.

### Performance (internal; output unchanged)

`summarize` peak RSS lowered with no change to any YAML/TSV/annotated value
(verified by semantic before/after parity). Measured on a deterministic ~1M-row
pedigree via the new `benchmarks/` profiler:

- **Release streaming pair matrices.** After `count_pairs_streaming`, the
  cached `_A … _A5` adjacency powers are dropped (they were held through the
  inbreeding / Ne / write phases, where the streaming run actually peaks).
  Now delivered upstream: pedigree-graph **v0.5.2** releases them on exit
  (pedigree-graph#4), so the pin moves to `@v0.5.2` and pedsum's private
  `_release_pair_matrices()` workaround is removed. Overall peak −20%
  (narrow) / −24% (wide).
- **Drop `_pair_lists` after the burden summary** (`--per-individual-pairs`):
  removes the materialised pair lists (~2–2.8 GiB at 200K–400K rows) from the
  post-extraction resident floor.
- **Compute sex-role ambiguity rows on demand:** the two per-parent
  first-row lookup tables are gone; the rare ambiguous-row finding now resolves
  its row reference directly, trimming the read/validate phase (~190 MiB at
  1M rows).

## 0.9.0 — 2026-05-19 — sex-from-role override + `sex_source` per-row audit column

Breaking. Continues the [ADR-0001](docs/adr/0001-collaborator-cli-redesign.md)
collaborator-friendly principle — pedsum tries harder to produce a
usable pedigree, and the audit column lets you see exactly what it
decided.

### Imputation extended

`_impute_sex_from_roles` now does two passes by default:

1. (existing 0.8) Missing→role: unsexed used only as mother → F;
   unsexed used only as father → M; both → ambiguous (-1); neither →
   orphan (-1).
2. **NEW** Asserted→role: asserted M used only as mother → override to
   F; asserted F used only as father → override to M. Used as both
   (topology ambiguous) → assertion preserved AND
   `_check_sex_role_consistency` still hard-blocks (pedsum cannot
   choose M-or-F from contradictory topology). Used as neither →
   assertion preserved.

`_check_sex_role_consistency` is re-purposed under the new default:
the override has already cleared contradictions when topology was
unambiguous, so the check produces zero findings. The grouped stderr
summary now reports `sex consistent with parent role .... PASS (N
overridden from role)`.

### New column

`sex_source` is a per-row string column on **both**
`annotated.tsv.gz` (summarize) and `validate.tsv.gz` (validate),
with four categorical values:

- `input` — assertion preserved (no role conflict or no role).
- `imputed_from_missing` — was missing in input; role implied F or M.
- `imputed_from_role` — was asserted but topology disagreed; topology won.
- `unresolved` — still SEX_UNKNOWN after both passes (orphan or
  ambiguous, tolerated under `--allow-missing-sex`).

### New flag

- **`--no-override-asserted-sex`** (opt-out) disables the new Pass 2
  override behaviour. Restores 0.8's hard-block on sex/role
  contradictions via `sex_role_consistency`. The existing
  missing→F/M imputation is unaffected.

### Migrating from 0.8

| 0.8 | 0.9 |
|---|---|
| `sex_role_consistency` hard-blocked any asserted-M-used-as-mother row | Auto-fixed by default; pass `--no-override-asserted-sex` to restore the hard-block |
| Fixed `validate.tsv.gz` rewrote only missing-sex rows | Also rewrites overridden rows (asserted token replaced with role-implied F/M) |
| No per-row sex provenance in outputs | `sex_source` column added to annotated.tsv.gz + validate.tsv.gz |
| Grouped stderr summary: `sex consistent with parent role .... PASS` (always plain PASS) | `PASS (N overridden from role)` when override fired |


## 0.8.0 — 2026-05-19 — unified missing-sex tolerance

Breaking. Same collaborator-friendly principle as 0.7
([ADR-0001](docs/adr/0001-collaborator-cli-redesign.md)) — fewer flags
to learn, uniform behaviour across the two missing-sex populations.

### Renamed

- `--allow-unknown-sex` → `--allow-missing-sex`. Hard break, no alias
  (`allow_abbrev=False` on the argparse parser rejects the old name with
  rc=2).

### Broader scope

- `--allow-missing-sex` now tolerates BOTH orphan unsexed rows (sex
  missing, not used as a parent) AND role-ambiguous rows (sex missing,
  used as BOTH mother and father). The latter was an unconditional
  hard-block in 0.7. The flag mirrors the existing SKIP-with-count
  pattern: when set, `_run_validate` does not BLOCK on either
  `unknown_sex` or `sex_role_ambiguity`, and per-row findings are not
  written to `validate.log` for tolerated rows (matches the existing
  precedent — the check-level SKIP CheckResult is the audit channel).

### Auto-fix

- Fixed `validate.tsv.gz` output now writes `"-1"` (pedsum's canonical
  missing token) for every row whose final sex is `SEX_UNKNOWN`. Orphan
  rows that 0.7 left at their original missing token (often empty or
  `NA`) now normalise to `-1` so the fixed file is self-consistent. The
  rewrite no longer requires `n_imputed > 0` — it runs whenever
  imputation ran, even on orphan-only pedigrees.

### Sex-stratified refusal

- The error message at `_run_summarize` (`--inbreeding` / `--effective-size`
  with any `SEX_UNKNOWN` row) now names `--allow-missing-sex` instead of
  the deleted `--allow-unknown-sex`.

### Migrating from 0.7

| 0.7 | 0.8 |
|---|---|
| `--allow-unknown-sex` | `--allow-missing-sex` (now also tolerates `sex_role_ambiguity`) |
| Orphan-unsexed row in fixed TSV kept its input token | Orphan-unsexed row in fixed TSV is `"-1"` |
| `sex_role_ambiguity` was an unconditional hard-block | Hard-block by default; `--allow-missing-sex` opts in to SKIP |


## 0.7.0 — 2026-05-19 — collaborator-friendly CLI redesign

Breaking. Rationale and rejected alternatives are recorded in
[`docs/adr/0001-collaborator-cli-redesign.md`](docs/adr/0001-collaborator-cli-redesign.md).

### CLI changes

- **`--out` is now a directory**, not a basename. Files inside get fixed
  names: `summary.yaml`, `summary.extra.yaml`, `annotated.tsv.gz`, and
  (with `--tsv`) `summary.pedigree.tsv` + `summary.individual.tsv`.
  `validate` writes `validate.log` + `validate.tsv.gz` into the same
  directory.

- **`--inbreeding` and `--effective-size` are now on by default.** Pass
  `--no-inbreeding` or `--no-effective-size` to skip. `--ne-coancestry`
  remains opt-in (RAM-hungry on > ~500K rows).

- **`--burden` renamed to `--per-individual-pairs`.** The output schema's
  `relationship_burden` section name is unchanged.

- **`--tsv` (new, opt-in)** writes the long-form
  `summary.pedigree.tsv` + `summary.individual.tsv`. Off by default;
  collaborators typically need only the YAML.

### Deleted flags

| Flag | Why |
|---|---|
| `--engine` | Implementation detail. Pedsum always uses the matrix engine; the BFS engine remains available via `pedigree_graph.experimental.count_pairs_bfs`. |
| `--bfs-threshold` | Paired with the deleted `--engine`. |
| `--zero-as-missing` | Niche; preprocess your data instead. |
| `--single-file` | Two-YAML split is now unconditional. |

argparse runs with `allow_abbrev=False` so abbreviation of a deleted
long-option will not silently match a surviving one.

### Other behaviour

- Cost-estimate INFO line emitted before the F kernel when N > 1,000,000:
  `computing F on N=… rows; may take several minutes — pass --no-inbreeding to skip`.

- `--out PATH` exits rc=1 with a single-line error if `PATH` already
  exists as a regular file (not a directory).

### Migrating from 0.6

| 0.6 | 0.7 |
|---|---|
| `--out my_run` (basename → `my_run.summary.yaml` next to it) | `--out my_run/` (directory → `my_run/summary.yaml` inside) |
| omit `--inbreeding` to skip F | pass `--no-inbreeding` to skip F |
| omit `--effective-size` to skip Ne | pass `--no-effective-size` to skip Ne |
| `--burden` | `--per-individual-pairs` |
| `--single-file` | (removed; two-YAML split is always written) |
| `--engine matrix` / `--engine bfs` / `--bfs-threshold N` | (removed; matrix engine only) |
| `--zero-as-missing` | (removed; preprocess input) |
| `--out X` then read `X.summary.pedigree.tsv` | pass `--tsv` then read `X/summary.pedigree.tsv` |


## 0.6.x — 2026-05-18 — birth-year validation in `validate`

- `--birth-year-col NAME` is now accepted by the `validate` subcommand and runs three checks: `birth_year_dtype` (numeric parsing), `birth_year_range` (within `[--birth-year-min, --birth-year-max]`; defaults `1800` and the current calendar year + 1), and `birth_year_topology` (child's `birth_year >=` each known parent's `birth_year`). Findings appear in `.validate.log` like any other check.
- `summarize` now also surfaces range / topology violations as a clean `PedigreeError` (return code 1) instead of letting `PedigreeGraph` raise a `ValueError` mid-pipeline with a traceback.


## 0.6.0 — 2026-05-18 — collaborator-friendly polish (sex auto-detect + missing-sex imputation)

- **Sex-encoding auto-detection.** `--sex-encoding=auto` (the new default) picks `default` (0=F, 1=M) or `plink` (1=M, 2=F, 0=unknown) from the observed tokens. `--plink-sex` remains as a legacy alias. Word-only columns get no warning (encoding moot); only-`1` numeric columns get a WARNING since the encoding is genuinely ambiguous. Under `plink`, `sex=0` is unconditionally treated as missing (PLINK fam spec) without needing `--zero-as-missing`.
- **Missing-sex imputation.** Missing-sex tokens (the parent-missing set plus `-1`, `U`, `Unknown`) decode to a sentinel and are then imputed from parent role: F if the row is used as a mother, M if used as a father. Two new checks surface the residual cases: `sex_role_ambiguity` (present individual with unknown sex referenced as BOTH mother and father — always a hard block) and `unknown_sex` (orphan unsexed row — hard block unless `--allow-unknown-sex` is passed, in which case it shows as SKIP with a tolerated-count note, mirroring the existing `--no-sex-check` precedent). The imputed sex is folded into `validate.tsv.gz` so the fixed output reflects the auto-fix.
- **Row-order check dropped, reorder silent.** The `topological_row_order` check is gone. Both `summarize` and `validate` reorder rows into topological order silently (single INFO log line with the moved count). The `validate.tsv.gz` output is always parents-before-children.
- **Grouped validation summary.** The stderr summary now prints four named sections ("Columns & parsing", "IDs", "Parent references", "Graph structure") with human-friendly dot-padded labels. Internal check names are preserved in `.validate.log` so parsing contracts are unchanged.
- **`--effective-size` / `--inbreeding` refuse `sex=-1`.** When `--allow-unknown-sex` leaves rows with sentinel sex AND the user asks for sex-stratified Ne or the F kernel, `summarize` exits 1 with a clear message rather than silently miscounting.


## 0.5.0 — 2026-05-15 — streaming default for pair counts

Default pair-counting engine flipped from the matrix path to the streaming path. The matrix and BFS engines materialise pair arrays before counting (peak memory scales with `O(answer size)`, not `O(N)`); on pair-dense pedigrees this OOMs.

- `summarize` now uses `pedigree_graph.PedigreeGraph.count_pairs_streaming` (scalar, O(N) memory) by default to populate the 23 pair counts. Bit-identical to the matrix engine for the 10 simple codes; ~1% scalar approximation on the 13 cousin / collateral codes when the pedigree has inbreeding, twins, or shallow depth.
- Per-individual relationship-burden summary stays a stub under the streaming default.
- The opt-in `--burden` flag restores the matrix / BFS engine path with full pair-list output (and the per-individual burden summary). OOMs on pair-dense pedigrees; use only when burden is required.
- The previous `--no-pairs` flag is removed.

See [`pedigree-graph`'s `LIMITATIONS.md`](https://github.com/rwaples/pedigree-graph/blob/main/LIMITATIONS.md) for the precision contract and what would unlock exact matrix-engine semantics at scale.


## 0.4.0 — 2026-05-13 — pedigree-graph v0.5.0 consolidation + BFS engine port

Internals refactor. Pedsum becomes a thin wrapper over `pedigree-graph` for inbreeding, lineage, and pair-counting primitives; the BFS / boolean-matmul / numba pair-counting engine moves out of pedsum entirely.

### Primitives consolidated into pedigree-graph

- Pedsum's recursive memoized kinship `compute_inbreeding` (and helper `_merge_sorted_rows`) deleted; F is now sourced from `pedigree_graph.PedigreeGraph.compute_inbreeding()` (Meuwissen-Luo).
- Pedsum's `compute_descendants` (path-count BFS) and `compute_generations` (Kahn) deleted; both moved upstream to `PedigreeGraph.compute_n_descendants()` and `PedigreeGraph.generation`.
- The `n_ancestors` column is now sourced from `PedigreeGraph.compute_n_ancestors()` (distinct-count, sparse transitive closure).
- `PedigreeGraph` is built once in `_run_summarize` and threaded through every primitive that needs it (relationship pairs, F, lineage counts, Ne), rather than each engine constructing its own.

### Effective population size

- New flags `--effective-size`, `--ne-coancestry`, `--ne-threads N` for pedigree-based Ne estimation.

### BFS engine ported to pedigree-graph

The BFS / boolean-matmul / numba pair-counting engine that used to live inline in `pedigree_summary.py` (under the misleading `_count_pairs_graphtool` / `--engine=graph-tool` name) is now `pedigree_graph.experimental.count_pairs_bfs`.

- Engine renamed `graph-tool` → `bfs` everywhere (CLI flag, dispatcher, YAML `pairs_engine` field, README, environment.yml).
- Dead `import graph_tool.all` probe + `_GRAPH_TOOL_AVAILABLE` gate removed.
- Hostile module-load `numba.set_num_threads(8)` global call removed.
- BFS-only helpers (`_dedup_pairs`, `_pairs_from_groups`, `_po_pairs`, `_lineal_pairs_lo_hi`, `_subtract_pair_keys`, `_run_per_anchor`, `_enumerate_pairs_kernel`) deleted from pedsum.
- Latent bug in `_count_after_subtract` (a dead list-comprehension that silently dropped accumulated subtract keys on a base-id bump) fixed during the port; covered by a unit test in `pedigree-graph/tests/test_experimental.py::test_count_after_subtract_handles_subtract_with_larger_hi`.
- BFS engine now reports `MZ` correctly (delegating to `PedigreeGraph._mz_twin_pairs()`) for callers who supply twin data. Pedsum's input format has no twin column, so pedsum users still see `MZ=0` from both engines — by design, not a regression.
