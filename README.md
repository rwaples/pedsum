# pedsum — standalone pedigree summary CLI

A single-file Python CLI for validating pedigrees and producing
machine-readable summaries: size, structure, family-size distribution,
relationship-pair counts (23 named codes through degree 5 plus
`by_degree` rollup), and per-individual statistics.

The relationship-pair enumerator is a vendored copy of
`simace.core.pedigree_graph` so the script ships as a single file with
no `simace` dependency at runtime.

## Install

Python ≥ 3.10 required (uses `functools.cached_property`, modern
type-hint syntax, `datetime.UTC`). Activate your env
(`conda activate myenv` or `source venv/bin/activate`) and install
dependencies:

**conda / mamba:**

```bash
conda install -c conda-forge numpy scipy pandas pyyaml
```

**pip:**

```bash
pip install numpy scipy pandas pyyaml
```

Optional: install `pigz` for ~3× faster `.tsv.gz` writes (the script
falls back to gzip-1 when absent):

```bash
conda install -c conda-forge pigz   # cross-platform, into the conda env
apt-get install pigz                # Debian/Ubuntu
brew install pigz                   # macOS
dnf install pigz                    # Fedora/RHEL
```

## Quick start

A 200-individual, 5-generation example pedigree
(`example_pedigree.tsv`) ships with the repo. Try it:

```bash
python pedigree_summary.py summarize --in example_pedigree.tsv --out /tmp/demo --inbreeding
python pedigree_summary.py validate  --in example_pedigree.tsv --out /tmp/demo
```

The example has columns `id, sex, mother, father, generation,
liability`. Only the first four are required; `generation` and
`liability` are preserved unchanged in `BASENAME.annotated.tsv.gz`.
The script also adds its own `ped_depth` column (topological depth
from founders) and the per-individual derived columns — see
"Topological depth" and "Column preservation" below.

## Usage

### Summarize a pedigree

```bash
python pedigree_summary.py summarize --in PED.tsv --out BASENAME [--inbreeding] [--safe-attempt]
```

Writes:

| File | Contents |
|---|---|
| `BASENAME.summary.yaml` | combined pedigree + individual summary |
| `BASENAME.summary.pedigree.tsv` | long-form pedigree-level summary |
| `BASENAME.summary.individual.tsv` | long-form per-individual distribution |
| `BASENAME.annotated.tsv.gz` | input pedigree + per-individual columns *(suppressed under `--safe-attempt`)* |

Flags:

- `--inbreeding` — compute `F` (inbreeding coefficient) per individual.
  Off by default; ~5 min on 10M rows.
- `--safe-attempt` — best-effort GDPR-style redaction: skip the
  per-individual `annotated.tsv.gz`, drop `min`/`max` from
  distributions, and null any count or stratum below cell-size 5.
  **Not a safe-harbor guarantee** — review before sharing.

### Validate a pedigree

```bash
python pedigree_summary.py validate --in PED.tsv --out BASENAME
```

Runs all integrity checks (duplicate IDs, missing parents, sex
conflicts, cycles, …) and writes:

| File | Contents |
|---|---|
| `BASENAME.validate.log` | per-finding TSV (one row per issue) |
| `BASENAME.validate.tsv.gz` | the pedigree with auto-fixes applied (omitted on hard-block findings) |

Auto-fix synthesizes founder rows for missing parent IDs.
Hard-blocks (cycles, duplicates, sex conflicts on missing parents,
etc.) cause the fixed TSV to be skipped — fix the source data first.

## Input format

TSV with at minimum the columns `id`, `sex`, `mother`, `father`
(column names overridable via `--id-col` / `--sex-col` /
`--mother-col` / `--father-col`).

- `id`: non-negative integer, unique per row.
- `sex`: `M`/`F` or `1`/`0` (1 = male, 0 = female).
- `mother`, `father`: parent IDs; `-1` for unknown/founder.

Half-founders (one parent `-1`, the other a valid ID) are accepted.

### Topological depth (`ped_depth`)

The script **always computes** a topological-depth column called
`ped_depth` (founders = 0, offspring = `max(parent_depth) + 1` via
Kahn's algorithm) and uses it as the grouping variable for
`f_by_generation`, `gen_counts`, and the per-individual ancestor /
descendant columns. The output column is named `ped_depth` and is
always a 32-bit integer.

You do not need to supply a depth column at all. If your input
already has a column named `ped_depth`, it is treated as a
*user-supplied extra* (see Column preservation, below) — it is **not**
trusted as ground truth and the script computes its own depth.

### Column preservation

`BASENAME.annotated.tsv.gz` keeps every column from your input.
The four canonical columns (`id`, `sex`, `mother`, `father`) are
deduplicated against the validated copies. Any other input column
that collides with a derived column (`ped_depth`, `is_founder`,
`F`, `n_*`, `component_id`) is preserved under the suffix `_input`,
and a `WARNING` is logged so the rename is never silent. Example:
an input column named `F` will appear in the output as both `F`
(the script's inbreeding coefficient) and `F_input` (your value).

Example:

```
id	sex	mother	father
1	F	-1	-1
2	M	-1	-1
3	F	1	2
4	M	1	2
5	F	3	2
```

## Output highlights

The `summary.yaml` contains:

- `size_structure` — counts, generation depth, connected components.
- `family_size` — sibship size distribution and offspring/mate counts.
- `pairs` — 23 named relationship codes (MZ, MO, FO, FS, MHS, PHS, GP,
  Av, GGP, HAv, GAv, 1C, GGGP, HGAv, GGAv, H1C, 1C1R, G3GP, HGGAv,
  G3Av, H1C1R, 1C2R, 2C) plus `PO = MO + FO` and `by_degree[0..5]`
  rollup.
- `inbreeding` — distribution of F (only when `--inbreeding`).
- `individual.distributions` — mean/std/quartiles/`nz` (non-zero
  count) for each per-individual numeric column.
- `individual.f_by_generation` — mean and max F per generation.

Floats are rounded to 4 decimal places.

## Logging

- Default: per-section progress at `INFO` level on stderr.
- `-v` / `--verbose`: also shows per-degree timings and matrix-product
  diagnostics from the relationship enumerator.
- `-q` / `--quiet`: warnings only.

## Notes

- This is a snapshot. The relationship enumerator was vendored from
  `simace.core.pedigree_graph`; if the upstream algorithm is updated
  for correctness, refresh Section 3a of `pedigree_summary.py`.
- For pedigrees with sparse/non-contiguous IDs the script
  internally compacts to dense `0..n-1` before enumeration, so the
  vendored matrix-power machinery never allocates `max(id)+1`-sized
  arrays.
