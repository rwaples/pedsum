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
| `pedsum/pedigree_ops.py` | low-level array helpers (parent rows, sib groups, topological depth) |
| `pedsum/parse.py` | delimiter sniffing, column coercion, sex decoding |
| `pedsum/checks.py` | per-check finding producers + check metadata (`_CHECK_*`) |
| `pedsum/validate.py` | `load_and_validate` (fail-fast) / `validate_pedigree` (accumulating) + sex imputation |
| `pedsum/pairs.py` | relationship-pair enumeration + `PedigreeGraph` construction |
| `pedsum/sections.py` | per-section summary computations |
| `pedsum/schema.py` | categorised YAML schema + slim/extra split machinery |
| `pedsum/report.py` | report payload builders, safe-attempt redaction, output writers |
| `pedsum/cli.py` | argparse + `summarize` / `validate` runners + `main` |

## CLI design rationale

See [docs/adr/0001-collaborator-cli-redesign.md](docs/adr/0001-collaborator-cli-redesign.md).

## Output schema

Output schema follows [CONTEXT.md](CONTEXT.md) as of 0.10. The glossary fixes section names, key names, and the convention for distribution naming (`<noun>_count` / `<noun>_count_hist` / `<noun>_count_<sex>`). When extending the output, add the term to CONTEXT.md first, then emit keys that match it.

## Flag and behavior version history

See [CHANGELOG.md](CHANGELOG.md). Notable:
- `validate --drop-offending` emits a **Reduced Pedigree** (0.11; see
  [docs/adr/0003](docs/adr/0003-drop-offending-reduction.md)). `--no-sex-check`
  now lives in the registry (`ValidationContext.no_sex_check`) so the tolerance
  composes inside `_run_checks` rather than as a cli post-filter.
- `--allow-unknown-sex` → `--allow-missing-sex` (0.8)
- `ped_depth` sourced from `PedigreeGraph.generation` (0.4)

## Engine selection & semantics

Pedsum uses two pair-counting paths, picked by `--per-individual-pairs`
(no engine auto-tiering — per ADR 0001, the matrix/BFS dispatch was
removed). Both delegate to `pedigree-graph`:

- Default: `PedigreeGraph.count_pairs_streaming` (`_engine` reported as
  `streaming_scalar`). O(N) memory; aggregate counts only.
- `--per-individual-pairs`: `_count_pairs_matrix_with_lists` (`_engine`
  reported as `matrix`). Materialises full pair lists so the
  per-individual relationship-burden summary can be computed.
- Streaming counts are bit-identical to matrix on the 10 simple codes
  (`MO`, `FO`, `FS`, `MHS`, `PHS`, `MZ`, `GP`, `GGP`, `GGGP`, `G3GP`);
  ~1% scalar approximation on the 13 cousin/collateral codes when
  the pedigree has inbreeding, twins, or shallow depth. The YAML
  `pairs_engine` field records which path produced each summary.
- The experimental BFS enumerator (matrix counts *paths* / multiplicity;
  BFS counts *distinct shared ancestors*, disagreeing on `1C1R`, `H1C1R`,
  `1C2R`, `2C`) is no longer reachable from pedsum. It remains available
  to direct callers via `pedigree_graph.experimental.count_pairs_bfs`.
  Open upstream issues:
  [pedigree-graph#2](https://github.com/rwaples/pedigree-graph/issues/2),
  [pedigree-graph#3](https://github.com/rwaples/pedigree-graph/issues/3).

## Performance thresholds

- F kernel (Meuwissen-Luo): logs INFO above `N = 1,000,000` so naive
  runs don't silently hang. See `_F_KERNEL_WARN_THRESHOLD` (`pedsum/base.py`)
  and its use in `_run_summarize` (`pedsum/cli.py`).
- `--per-individual-pairs`: matrix engine OOMs on pair-dense
  pedigrees above ~500K rows.
- `--ne-coancestry`: kinship DP scales with cumulative ancestor set;
  blows up RAM above ~500K rows.

## Upstream integration

Both pair-counting engines delegate to `pedigree-graph`; bug fixes
propagate on `pip install -U`. Sparse/non-contiguous IDs are
compacted internally to dense `0..n-1` so the underlying machinery
never allocates `max(id)+1`-sized arrays.

## Stance: opt-outs vs auto-tiering

Size-tiered behaviors stay as opt-outs the user types
(`--no-inbreeding`, `--no-effective-size`, `--per-individual-pairs`,
`--ne-coancestry`), not as auto-tiered defaults. Reasoning is
informally documented here pending a future ADR.
