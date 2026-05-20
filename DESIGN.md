# DESIGN.md — pedsum internals (maintainer notes)

Pointers to where implementation details live. The README is the
collaborator-facing surface; this file is for maintainers.

## CLI design rationale

See [docs/adr/0001-collaborator-cli-redesign.md](docs/adr/0001-collaborator-cli-redesign.md).

## Flag and behavior version history

See [CHANGELOG.md](CHANGELOG.md). Notable:
- `--allow-unknown-sex` → `--allow-missing-sex` (0.8)
- `ped_depth` sourced from `PedigreeGraph.generation` (0.4)

## Engine selection & semantics

- Matrix vs streaming vs BFS engines: see `_select_engine` in
  `pedigree_summary.py` (line ~1824).
- Streaming counts are bit-identical to matrix on the 10 simple codes
  (`MO`, `FO`, `FS`, `MHS`, `PHS`, `MZ`, `GP`, `GGP`, `GGGP`, `G3GP`);
  ~1% scalar approximation on the 13 cousin/collateral codes when
  the pedigree has inbreeding, twins, or shallow depth.
- Matrix vs BFS on inbred pedigrees: matrix counts *paths*
  (multiplicity), BFS counts *distinct shared ancestors* — they
  disagree on cousin-style codes (`1C1R`, `H1C1R`, `1C2R`, `2C`).
  The YAML `pairs_engine` field records which engine produced each
  summary. See docstrings around lines 3477, 3647, 3649.
- BFS auto-selects at `N ≥ 5M` within the `--per-individual-pairs`
  path and emits a `FutureWarning`; never reached without that flag.
  Open upstream issues:
  [pedigree-graph#2](https://github.com/rwaples/pedigree-graph/issues/2),
  [pedigree-graph#3](https://github.com/rwaples/pedigree-graph/issues/3).

## Performance thresholds

- F kernel (Meuwissen-Luo): logs INFO above `N = 1,000,000` so naive
  runs don't silently hang. See `pedigree_summary.py:2176, 3443`.
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
