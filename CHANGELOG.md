# Changelog

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
