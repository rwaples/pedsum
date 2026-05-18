# pedsum — standalone pedigree summary CLI

A single-file Python CLI for validating pedigrees and producing
machine-readable summaries: size, structure, family-size distribution,
relationship-pair counts (23 named codes through degree 5 plus
`by_degree` rollup), mating-pair structure, founder contribution,
component/lineage/sex/generation aggregates, and per-individual
statistics.

Both relationship-pair engines (matrix and an experimental BFS path)
delegate to the [`pedigree-graph`](https://github.com/rwaples/pedigree-graph)
package — pedsum is a thin CLI wrapper plus pedigree validation, family-size
distribution, and per-individual outputs.

## Install

Python ≥ 3.13 required (matches `pedigree-graph`). Activate your env
(`conda activate myenv` or `source venv/bin/activate`) and install
dependencies:

**conda / mamba:**

```bash
conda install -c conda-forge numpy scipy pandas pyyaml
pip install "pedigree-graph @ git+https://github.com/rwaples/pedigree-graph.git@v0.5.0"
```

**pip:**

```bash
pip install numpy scipy pandas pyyaml
pip install "pedigree-graph @ git+https://github.com/rwaples/pedigree-graph.git@v0.5.0"
```

(`pedigree-graph` brings in `numba` transitively for the BFS engine's
parallel kernel.)

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
python pedigree_summary.py summarize --in example_pedigree.tsv --out /tmp/demo --effective-size
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
| `BASENAME.summary.pedigree.tsv` | long-form pedigree-level summary: size/link structure, family sizes, mating pairs, lineage, founder contribution, components, sex/generation aggregates, relationship-pair counts, and optional inbreeding |
| `BASENAME.summary.individual.tsv` | long-form per-individual distribution |
| `BASENAME.annotated.tsv.gz` | input pedigree + per-individual columns *(suppressed under `--safe-attempt`)* |

Flags:

- `--inbreeding` — emit per-individual `F` and the inbreeding summary
  section.  Off by default; ~minutes on 10M rows (F is the single most
  expensive computation in pedsum, even with the upstream
  Meuwissen-Luo kernel).  When `--effective-size` is also passed, F is
  shared with the Ne pipeline and the cost is paid once.
- `--effective-size` — compute the eight pedigree-based effective
  population size estimators
  (`Ne_I`, `Ne_C`, `Ne_V`, `Ne_sr`, `Ne_iΔF`, `Ne_LTC`, `Ne_H`,
  `Ne_CT`) via `pedigree-graph.compute_all_ne`.  Off by default.
  Works standalone — pair with `--inbreeding` to additionally emit
  the per-individual F section.
- `--ne-coancestry` — include the coancestry-rate `Ne_C` estimator.
  Off by default because the kinship DP can blow up RAM on very
  large pedigrees (>~500K rows).  No-op without `--effective-size`.
- `--ne-threads N` — number of threads for independent Ne estimator
  dispatch (default `1`, serial).  Validated `>= 1` by argparse.  No-op
  without `--effective-size`.
- `--burden` — opt into per-individual relationship-burden summary
  (`relationship_summary.relatives_total`, `.relatives_by_degree`,
  closest-degree distribution, etc.).  Requires the matrix or BFS
  engine to materialise full pair lists, which OOMs on pair-dense
  pedigrees (stallion-heavy livestock, large half-sib clusters).
  Without this flag pedsum uses
  `pedigree_graph.PedigreeGraph.count_pairs_streaming` (scalar,
  O(N) memory) to populate the 23 pair counts and leaves the burden
  summary as a stub.  The streaming counts are bit-identical to the
  matrix engine for the 10 simple codes
  (`MO`, `FO`, `FS`, `MHS`, `PHS`, `MZ`, `GP`, `GGP`, `GGGP`,
  `G3GP`); ~1% scalar approximation on the 13 cousin / collateral
  codes when the pedigree has inbreeding, twins, or shallow depth.
- `--safe-attempt` — best-effort GDPR-style redaction: skip the
  per-individual `annotated.tsv.gz`, drop `min`/`max` from
  distributions, and null any count or stratum below cell-size 5.
  **Not a safe-harbor guarantee** — review before sharing.
- `--sex-encoding {auto,default,plink}` — how to decode the sex
  column. `auto` (default) detects from the observed tokens; `default`
  forces `0 = female, 1 = male` (pedsum convention); `plink` forces
  `1 = male, 2 = female` with `0 = unknown` (PLINK fam spec). See
  "Sex auto-detection" below.
- `--plink-sex` — legacy alias for `--sex-encoding=plink`.
- `--allow-unknown-sex` — tolerate rows whose sex is missing AND
  cannot be imputed from parent role (kept as the sentinel `-1`).
  Without this flag, unresolved rows are an error. Incompatible with
  `--effective-size` / `--inbreeding` in `summarize` (sex-stratified
  estimators require resolved sex).
- `--zero-as-missing` — treat `0` in `mother`/`father` columns as
  missing (PLINK fam convention).
- `--engine {auto,matrix,bfs}` — relationship-pair enumeration engine.
  `auto` (default) picks `bfs` when `n` is at or above the threshold,
  otherwise `matrix`. The `bfs` engine is **experimental** — see
  "Choosing an engine" below.
- `--bfs-threshold N` — auto-select threshold for the bfs engine
  (default: 5,000,000).

### Validate a pedigree

```bash
python pedigree_summary.py validate --in PED.tsv --out BASENAME
```

Runs all integrity checks (duplicate IDs, missing parents, sex
conflicts, cycles, unsexed individuals, …) and writes:

| File | Contents |
|---|---|
| `BASENAME.validate.log` | per-finding TSV (one row per issue) |
| `BASENAME.validate.tsv.gz` | the pedigree with auto-fixes applied (omitted on hard-block findings) |

When `--birth-year-col NAME` is passed, validate also runs three
optional checks: `birth_year_dtype` (numeric parsing),
`birth_year_range` (each year within `[--birth-year-min,
--birth-year-max]`; defaults to `[1800, current_year + 1]`), and
`birth_year_topology` (`child.birth_year >= parent.birth_year` for
every edge where both endpoints are known). Findings are written to
`.validate.log` like any other check.

Auto-fixes folded into `BASENAME.validate.tsv.gz`:
- Synthesized founder rows for missing parent IDs.
- Sex imputed from parent role (F if used as a mother, M if used as
  a father) for any row whose original sex was missing.
- Rows reordered so parents always precede children (topological
  order), if the input was not already ordered.

Hard-blocks (cycles, duplicates, sex conflicts on missing parents,
unresolved sex without `--allow-unknown-sex`, sex-role ambiguity)
cause the fixed TSV to be skipped — fix the source data first.

## Input format

TSV (tab-separated; `.tsv` or `.tsv.gz`) with at minimum the columns
`id`, `sex`, `mother`, `father` — header row required. Column names
are overridable via `--id-col` / `--sex-col` / `--mother-col` /
`--father-col`.

- `id`: non-negative integer, unique per row. **Strings (e.g.
  `"P001"`) are not accepted** — see "Preparing your data" below.
- `sex`: `M`/`F` (any case), `Male`/`Female`, or numeric tokens whose
  meaning depends on the resolved encoding (see "Sex auto-detection"
  below). The default pedsum encoding is `0 = female, 1 = male`;
  PLINK's encoding is `1 = male, 2 = female` with `0 = unknown`.
  Missing tokens (`""`, `NA`, `NaN`, `N/A`, `.`, `?`, `None`, `null`,
  `-1`, `U`, `Unknown`, any case) are recognised and are either
  imputed from parent role (F if used as a mother, M if used as a
  father) or surfaced as findings.
- `mother`, `father`: parent IDs; `-1` for unknown/founder. The
  tokens `NA`, `NaN`, `N/A`, `.`, `?`, blank, `None`, `null` (any
  case) are also recognised as missing. To treat literal `0` as
  missing (PLINK fam convention), pass `--zero-as-missing`.

Half-founders (one parent missing, the other a valid ID) are
accepted.

### Sex auto-detection

`--sex-encoding=auto` (the default) picks an encoding from the tokens
in the sex column:

| Tokens seen | Resolved encoding |
|---|---|
| any `2` | `plink` (1=M, 2=F, 0=unknown) |
| any `0` (without `--zero-as-missing`) | `default` (0=F, 1=M) |
| any `0` plus `--zero-as-missing` | `plink` (0 = unknown) |
| only `M`/`F`/`Male`/`Female` | `default` (encoding choice is moot) |
| only `1` tokens | `default` + WARNING — pass `--sex-encoding=plink` if the file is actually PLINK-encoded |

Under the resolved encoding, missing-sex tokens decode to a sentinel
`-1`. pedsum then imputes those rows from parent role where possible:
an unsexed individual used as a mother is imputed F, and as a father
is imputed M. If a present individual is referenced as BOTH mother
and father with unknown sex, that's a data contradiction (`sex_role_ambiguity`)
and a hard block — pedsum cannot pick a side. Truly orphan unsexed
rows (not used as a parent) are a hard block unless `--allow-unknown-sex`
is set.

## Preparing your data

Most real-world pedigree files need a small preprocessing step to fit
the input format. Common cases:

### PLINK `.fam` files

A PLINK `.fam` file is whitespace-separated, has six columns
(`FID IID PAT MAT SEX PHENO`), no header, uses `0` for missing
parents, and encodes sex as `1 = male, 2 = female`. To run pedsum on
it directly:

```bash
# Add a header, convert spaces → tabs:
{ printf 'FID\tIID\tPAT\tMAT\tSEX\tPHENO\n'; tr -s ' ' '\t' < cohort.fam; } > cohort.tsv

python pedigree_summary.py summarize \
    --in cohort.tsv --out cohort \
    --id-col IID --mother-col MAT --father-col PAT --sex-col SEX \
    --plink-sex --zero-as-missing
```

### CSV (comma-separated)

```bash
tr ',' '\t' < input.csv > input.tsv
```

(or `python -c "import pandas as pd; pd.read_csv('input.csv').to_csv('input.tsv', sep='\t', index=False)"` if quoted commas matter.)

### String IDs (`"P001"`, `"FAM01-003"`, …)

The relationship enumerator requires integer IDs. Map them to ints
once and keep a lookup table:

```python
import pandas as pd
df = pd.read_csv("clinical.tsv", sep="\t", dtype=str)
all_ids = pd.unique(df[["id", "mother", "father"]].values.ravel())
all_ids = all_ids[all_ids != "-1"]
lut = {sid: i for i, sid in enumerate(sorted(all_ids))}
for col in ("id", "mother", "father"):
    df[col] = df[col].map(lambda x: lut.get(x, -1)).astype(int)
df.to_csv("clinical_int.tsv", sep="\t", index=False)
pd.Series(lut).to_csv("id_lookup.tsv", sep="\t")
```

### Excel (`.xlsx`)

Open in Excel/LibreOffice and "Save As" → tab-delimited, or:

```python
import pandas as pd
pd.read_excel("input.xlsx").to_csv("input.tsv", sep="\t", index=False)
```

## Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `input appears to be CSV` | comma-separated input | convert as above |
| `column 'id' must be integer-valued` | string/alphanumeric IDs | map to ints |
| `sex column has N invalid value(s)` showing `'2'` | PLINK 1/2 encoding | add `--plink-sex` |
| `id=0 referenced as mother/father` | file uses `0` for missing | add `--zero-as-missing` |
| `column 'mother' must be integer-valued` showing `NA` or non-finite | non-standard missing token | replace with `-1`, `NA`, blank, or `.` |
| `missing required columns` | wrong column names | use `--id-col` / `--sex-col` / `--mother-col` / `--father-col` |

### Topological depth (`ped_depth`)

The script **always computes** a topological-depth column called
`ped_depth` (founders = 0, offspring = `max(parent_depth) + 1`).
Since pedsum 0.4 the depth is sourced from `PedigreeGraph.generation`
in the upstream `pedigree-graph` library — semantics are unchanged,
output dtype is still `int32`, and it remains the grouping variable
for `f_by_generation`, `gen_counts`, and the per-individual ancestor
/ descendant columns.  Inputs must be in topological row order
(parents precede children); rows that violate this raise a clear
`PedigreeError`.

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
- `family_size` — sibship and per-person offspring distributions.
- `mating_pairs` — mating-pair counts, children-per-pair distribution,
  mate-count distributions, and effective pair count.
- `relationship_summary` — unique related/unrelated pair counts,
  related-pair density, closest-degree and relatives-by-degree
  distributions, and within-generation related-pair density. These
  pair-list-derived fields are exact for the matrix engine and marked
  unavailable for the experimental BFS engine.
- `lineage` — reproductive, terminal, and descendant distributions;
  ancestor distributions are populated only with `--inbreeding`.
- `founder_contribution` — founder descendant-path contribution and
  effective founder count.
- `founder_generation` — active founder lines and effective founder
  contribution by generation, plus simple bottleneck minima. This
  section is skipped on very large founder × individual products to
  avoid memory blow-ups.
- `components`, `sex_summary`, `generation_summary` — aggregate
  component, sex-stratified, and depth-stratified summaries, including
  generation-level offspring and mate-count distributions.
- `pairs` — 23 named relationship codes (MZ, MO, FO, FS, MHS, PHS, GP,
  Av, GGP, HAv, GAv, 1C, GGGP, HGAv, GGAv, H1C, 1C1R, G3GP, HGGAv,
  G3Av, H1C1R, 1C2R, 2C) plus `PO = MO + FO` and `by_degree[0..5]`
  rollup.
- `inbreeding` — distribution of F (only when `--inbreeding`).
- `effective_size` — scenario-level Ne from eight pedigree-based
  estimators (`Ne_I`, `Ne_C`, `Ne_V`, `Ne_sr`, `Ne_iΔF`, `Ne_LTC`,
  `Ne_H`, `Ne_CT`).  Each estimator emits its scalar `ne` plus a
  per-generation (or per-transition) breakdown vector.  Only present
  when `--effective-size` is passed; `Ne_C` carries `ne: null` unless
  `--ne-coancestry` is also passed.
- `individual.distributions` — mean/std/quartiles/`nz` (non-zero
  count) for each per-individual numeric column.

Floats are rounded to 4 decimal places.  Ne scalar values also appear
in `BASENAME.summary.pedigree.tsv` under the `effective_size_scalars`
section (the per-generation vectors live only in the YAML, to keep
the long-form TSV scannable).

## Logging

- Default: per-section progress at `INFO` level on stderr.
- `-v` / `--verbose`: also shows per-degree timings and matrix-product
  diagnostics from the relationship enumerator.
- `-q` / `--quiet`: warnings only.

## Notes

- Both engines delegate to the `pedigree-graph` package; bug fixes
  there propagate automatically on `pip install -U`.
- For pedigrees with sparse/non-contiguous IDs the script
  internally compacts to dense `0..n-1` before enumeration, so the
  underlying machinery never allocates `max(id)+1`-sized arrays.
- **Choosing an engine**:
    - The `matrix` engine (default for n < 5M) uses sparse matrix
      powers `A^k`. Fastest when its intermediate sparse products fit
      in RAM.
    - The `bfs` engine is **experimental** — auto-selected for
      n ≥ 5M. It uses boolean (set-union) sparse matmul plus a parallel
      numba kernel. Its scalability advantage at very large pedigrees
      (where the matrix engine OOMs) is unverified — at n=2M the
      matrix engine actually wins head-to-head (see
      `STATUS.md`). The engine may change or be removed in any
      `pedigree-graph` minor release; both pedsum's CLI warning and
      the package's `FutureWarning` fire on selection.
    - Both engines emit identical counts on non-inbred pedigrees. On
      inbred pedigrees they disagree on cousin-style codes (`1C1R`,
      `H1C1R`, `1C2R`, `2C`): matrix counts *paths* (multiplicity),
      bfs counts *distinct shared ancestors*. The YAML
      `pairs_engine` field records which engine produced each summary.
