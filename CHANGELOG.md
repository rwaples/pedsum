# Changelog

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
