# pedsum — pedigree summary CLI

Source: <https://github.com/rwaples/pedsum>

A Python command line tool for summarizing and validating pedigrees.
Reports the size, structure, sibship-size distribution, relationship-pair
counts (up to degree 5), mating-pair structure, founder contributions,
per-individual reproduction and genealogy aggregates, depth- and
sex-stratified summaries, and per-individual statistics. Output
terminology follows [CONTEXT.md](CONTEXT.md).

Uses the [`pedigree-graph`](https://github.com/rwaples/pedigree-graph) package for relationship-pair detection.

## Install

Python ≥ 3.13 required. 

### Get the code

```bash
# clone from git 
git clone https://github.com/rwaples/pedsum.git
cd pedsum
# Install dependencies in a conda environment
conda env install -f environment.yml
# activate conda environment
conda activate pedsum
```

## Quick start

```bash 
# run like this:
python pedigree_summary.py validate --in ...
python pedigree_summary.py summarize --in ...
```

```bash 
# help:
python pedigree_summary.py --help 
```

A 200-individual, 5-generation example pedigree
(`example_pedigree.tsv`) ships with the repo. Try it:

```bash
python pedigree_summary.py validate  --in example_pedigree.tsv --out /tmp/demo-validate
python pedigree_summary.py summarize --in example_pedigree.tsv --out /tmp/demo
```

`--out` is a directory (created if needed). The bare `summarize` writes
three files inside: `summary.yaml`, `summary.extra.yaml`, and
`annotated.tsv.gz`. Estimators of inbreeding (F) and Effective population size (Ne) are computed by default. 
Pass `--no-inbreeding` or `--no-effective-size` to skip them.

The example has columns `id, sex, mother, father, generation,
liability, birth_year`. Only the first four are required; the rest are
preserved unchanged in `annotated.tsv.gz`. Pass `--birth-year-col
birth_year` to include a birth year for each individual. The script
also adds its own `ped_depth` column (topological depth from founders)
and the per-individual derived columns.

## Usage

### Validate a pedigree

```bash
python pedigree_summary.py validate --in PED.tsv --out DIR
```

`--out DIR` is a directory (created if needed). Runs all integrity
checks (duplicate IDs, missing parents, sex conflicts, cycles, unsexed
individuals, …) and writes:

| File | Contents |
|---|---|
| `DIR/validate.log` | per-finding TSV (one row per issue) |
| `DIR/validate.tsv.gz` | the pedigree with auto-fixes applied (omitted on hard-block findings) |

When `--birth-year-col NAME` is passed, validate also runs three
optional checks: `birth_year_dtype` (numeric parsing),
`birth_year_range` (each year within `[--birth-year-min,
--birth-year-max]`; defaults to `[1800, current_year + 1]`), and
`birth_year_topology` (`child.birth_year >= parent.birth_year` for
every edge where both endpoints are known). Findings are written to
`.validate.log` like any other check.

Auto-fixes folded into `DIR/validate.tsv.gz`:
- Synthesized founder rows for missing parent IDs.
- Sex imputed from parent role (F if used as a mother, M if used as
  a father) for any row whose original sex was missing.
- Rows reordered so parents always precede children (topological
  order), if the input was not already ordered.

Hard-blocks (cycles, duplicates, sex conflicts on missing parents,
unresolved sex without `--allow-missing-sex`, sex-role ambiguity)
cause the fixed TSV to be skipped — fix the source data first.

### Summarize a pedigree

```bash
python pedigree_summary.py summarize --in PED.tsv --out DIR [options]
```

`--out DIR` is a directory (created if needed). Files written inside:

| File | When | Contents |
|---|---|---|
| `summary.yaml` | always | slim categorised summary (~500 lines) |
| `summary.extra.yaml` | always | per-depth / per-cohort arrays + full per-individual quantiles |
| `annotated.tsv.gz` | always (unless `--safe-attempt`) | input pedigree + per-individual columns |
| `summary.pedigree.tsv` | with `--tsv` | long-form pedigree-level summary |
| `summary.individual.tsv` | with `--tsv` | long-form per-individual distribution |

Flags:

- `--inbreeding` / `--no-inbreeding` — compute per-individual `F` and
  the inbreeding summary section. On by default. F is the single most
  expensive computation in pedsum (~minutes on 10M-row pedigrees);
  when `--effective-size` is also on, F is shared with the Ne pipeline
  and the cost is paid once.
- `--no-effective-size` — skip seven pedigree-based effective population size estimators (`Ne_I`, `Ne_V`,
  `Ne_sr`, `Ne_iΔF`, `Ne_LTC`, `Ne_H`, `Ne_CT`).  The
  eighth estimator (`Ne_C`, coancestry rate) is opt-in via
  `--ne-coancestry`..
- `--ne-coancestry` — additionally compute `Ne_C`. Off by default —
  its kinship DP can blow up RAM on pedigrees larger than ~500K rows.
- `--ne-threads N` — number of threads for independent Ne estimator
  dispatch (default `1`, serial). Validated `>= 1` by argparse.
- `--per-individual-pairs` — opt into the per-individual
  relationship-burden summary (`relationship_summary.relatives_total`,
  `.relatives_by_degree`, closest-degree distribution). Requires
  materialising full pair lists, which OOMs on pair-dense pedigrees
  above ~500K rows (stallion-heavy livestock, large half-sib clusters).
  Off by default; the 23 pair counts and the standard summary are
  produced without it.
- `--tsv` — additionally write the long-form `summary.pedigree.tsv`
  and `summary.individual.tsv`. Off by default; collaborators
  typically need only the YAML outputs.
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
- `--allow-missing-sex` — tolerate rows whose sex is missing after
  imputation, either because the row is unsexed and not used as a
  parent (orphan), or because it is used as BOTH mother and father
  with unknown sex (role-ambiguous). Such rows are auto-fixed to
  `sex=-1` in the validate-fixed output. Without this flag, either
  case hard-blocks. Incompatible with `--effective-size` /
  `--inbreeding` in `summarize` (sex-stratified estimators require
  resolved sex; pass `--no-effective-size` and/or `--no-inbreeding`
  if you want to keep `--allow-missing-sex`).
- `--no-override-asserted-sex` — disable the default behavior of
  overriding asserted sex when topology unambiguously implies the
  opposite (asserted M used only as mother → F; asserted F used only
  as father → M). The missing→F/M imputation is unaffected. Reverts
  to hard-blocking on sex/role contradictions.

## Input format

Header-row required, one individual per line. By default `--sep auto`
sniffs the first non-empty line for `\t`, `,`, `;`, or `|`; if none
are present and the line splits into multiple whitespace-separated
tokens, it routes through whitespace mode (PLINK fam-style). Override
with `--sep {tab,comma,semicolon,pipe,whitespace}` if you want to pin
it. Gzip is auto-detected from a `.gz` extension. A minimal pedigree:

```
id	sex	mother	father
1	F	-1	-1
2	M	-1	-1
3	F	1	2
4	M	1	2
5	F	3	2
```

### Required columns

| Column | Type | Notes |
|---|---|---|
| `id` | int ≥ 0, unique | **Strings (e.g. `"P001"`) are not accepted** — see "Preparing your data" below. |
| `sex` | `M`/`F`/`Male`/`Female` (any case) or numeric token | Numeric meaning depends on the resolved encoding (see "Sex auto-detection" below); default pedsum is `0 = female, 1 = male`, PLINK is `1 = male, 2 = female` with `0 = unknown`. Missing tokens (`""`, `NA`, `NaN`, `N/A`, `.`, `?`, `None`, `null`, `-1`, `U`, `Unknown`, any case) are recognised and are either imputed from parent role (F if used as a mother, M if used as a father) or surfaced as findings. |
| `mother` | int parent ID or missing | `-1` for unknown/founder. Tokens `NA`, `NaN`, `N/A`, `.`, `?`, blank, `None`, `null` (any case) are also recognised as missing. If your file uses literal `0` for "missing parent" (PLINK fam convention), replace it with `-1` first. |
| `father` | int parent ID or missing | Same conventions as `mother`. |

Required column names are overridable via `--id-col` / `--sex-col` /
`--mother-col` / `--father-col`. Half-founders (one parent missing,
the other a valid ID) are accepted.

### Optional columns

| Column | Recognised when | Effect |
|---|---|---|
| `birth_year` | `--birth-year-col NAME` is passed | Integer calendar year (sentinel `-1` for unknown). Threads into the Hill overlapping-generation Ne estimator — without it, Ne_H collapses to Ne_V. Range-checked against `[--birth-year-min, --birth-year-max]` (default `[1800, current_year + 1]`) and topologically checked (`child.birth_year >= parent.birth_year` on every known edge). |
| *(any other)* | always | Carried through verbatim into `annotated.tsv.gz` as a user-supplied extra. Collisions with derived columns (`ped_depth`, `F`, `n_*`, etc.) are preserved under a `_input` suffix with a `WARNING` — see "Column preservation" below. |

### Row order

Out-of-order rows are tolerated. Both `summarize` and `validate` run a
depth sweep up front; if any row is out of topological order pedsum
logs an INFO line and re-sorts parents-before-children for downstream
processing. `validate` writes the sorted pedigree to
`DIR/validate.tsv.gz`; `summarize` keeps the re-sorted frame
in-memory. Only cycles or rows that cannot be reached from any root
raise a `PedigreeError` (hard block) — fix the source data first.

### Sex auto-detection

`--sex-encoding=auto` (the default) picks an encoding from the tokens
in the sex column:

| Tokens seen | Resolved encoding |
|---|---|
| any `2` | `plink` (1=M, 2=F, 0=unknown) |
| any `0` | `default` (0=F, 1=M) |
| only `M`/`F`/`Male`/`Female` | `default` (encoding choice is moot) |
| only `1` tokens | `default` + WARNING — pass `--sex-encoding=plink` if the file is actually PLINK-encoded |

Missing-sex tokens decode to a sentinel `-1`. pedsum then imputes
from parent role: F if used as mother, M if used as father. Two
cases hard-block unless `--allow-missing-sex` is set — an individual
used as BOTH mother and father (pedsum cannot pick a side), and an
unsexed row not used as a parent at all.

## Preparing your data

Common preprocessing comes down to (a) adding a header row,
(b) remapping non-standard missing tokens, or (c) integerising
string IDs. `--sep auto` handles delimiter routing automatically.

### PLINK `.fam` files

No header, `0` for missing parents, sex `1=male, 2=female`. Add a
header and remap `0 → -1`:

```bash
{ printf 'FID IID PAT MAT SEX PHENO\n'; \
  awk 'BEGIN{OFS=" "} {if($3==0)$3=-1; if($4==0)$4=-1; print}' cohort.fam; } \
    > cohort.fam.headed

python pedigree_summary.py summarize \
    --in cohort.fam.headed --out cohort/ \
    --id-col IID --mother-col MAT --father-col PAT --sex-col SEX \
    --plink-sex
```

### Non-TSV inputs (CSV, Excel)

CSV: pass directly — `--sep auto` reads commas (or `--sep comma` to
be explicit). For Excel, or CSVs with quoted commas inside fields,
re-export through pandas:

```bash
python -c "import pandas as pd; pd.read_excel('input.xlsx').to_csv('input.tsv', sep='\t', index=False)"
```

### String IDs (`"P001"`, `"FAM01-003"`, …)

The relationship enumerator requires integer IDs. Map them once and
keep a lookup table:

```python
import pandas as pd
df = pd.read_csv("clinical.tsv", sep="\t", dtype=str)
all_ids = pd.unique(df[["id", "mother", "father"]].values.ravel())
lut = {sid: i for i, sid in enumerate(sorted(all_ids[all_ids != "-1"]))}
for col in ("id", "mother", "father"):
    df[col] = df[col].map(lambda x: lut.get(x, -1)).astype(int)
df.to_csv("clinical_int.tsv", sep="\t", index=False)
pd.Series(lut).to_csv("id_lookup.tsv", sep="\t")
```

## Large pedigrees

pedsum's defaults include the most expensive computations in the tool
— per-individual inbreeding (F) and the seven cheap effective-size
estimators. For very large pedigrees, the recommended first pass skips
both and produces size + structure only:

```bash
python pedigree_summary.py summarize \
    --in cohort.tsv --out cohort/ \
    --no-inbreeding --no-effective-size
```

Once that pass succeeds, re-run without `--no-inbreeding
--no-effective-size` to add F and the cheap Ne estimators.

## Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `column 'id' must be integer-valued` | string/alphanumeric IDs | map to ints |
| `sex column has N invalid value(s)` showing `'2'` | PLINK 1/2 encoding | add `--plink-sex` |
| `id=0 referenced as mother/father` | file uses `0` for missing | preprocess: replace `0` in mother/father columns with `-1` |
| `column 'mother' must be integer-valued` showing `NA` or non-finite | non-standard missing token | replace with `-1`, `NA`, blank, or `.` |
| `missing required columns` | wrong column names | use `--id-col` / `--sex-col` / `--mother-col` / `--father-col` |

### Topological depth (`ped_depth`)

The script always computes a topological-depth column called
`ped_depth` (founders = 0, offspring = `max(parent_depth) + 1`,
dtype `int32`). It is the grouping variable for `depth_summary`,
`depth_counts`, and the per-individual `n_distinct_ancestors` /
`n_descendant_paths` / `n_founder_ancestors` columns.

You do not need to supply a depth column. If your input already has
a column named `ped_depth`, it is treated as a *user-supplied extra*
(see Column preservation below) — it is **not** trusted as ground
truth and the script computes its own depth.

### Column preservation

`DIR/annotated.tsv.gz` keeps every column from your input.
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

`summary.yaml` is organised under the categories defined by `SUMMARY_SCHEMA`. Every key follows the glossary in [CONTEXT.md](CONTEXT.md) — see that document for term definitions and the naming convention (`n_<noun>` for per-individual columns, `<noun>_count` for summary-stats distributions, `<noun>_count_hist` for binned PMFs).

| Category → Section | Contents |
|---|---|
| `structure.size_structure` | counts, max/mean/median depth, `depth_counts`, connected-component aggregates |
| `structure.components` | component-size distribution and singleton stats |
| `demography.sibship_size` | per-Sibship size distribution (n_sibships, mean/median, size_dist) |
| `demography.mating_pairs` | Mating Pair count, children-per-pair, effective pair count |
| `reproduction.reproduction` | per-individual offspring/mate counts and Reproductive/Terminal classification (`offspring_count`, `offspring_count_hist`, `mate_count`, sex-stratified variants, `frac_with_full_sib`) |
| `genealogy.genealogy` | per-individual `descendant_paths` summary; `distinct_ancestors` summary when `--inbreeding` is set |
| `founders.founder_contribution` | Founders with descendants + `descendant_paths_per_founder` distribution + `effective_founders_by_descendant_paths` |
| `founders.founder_summary` | active Founders and effective-Founder contribution by depth, `founder_ancestors` per-depth distribution, bottleneck minima (may be skipped on very large pedigrees) |
| `relatedness.relationship_pairs` | 23 named Relationship codes through Degree 5 plus `PO = MO + FO`, with `by_degree[0..5]` rollup (TSV-only) |
| `relatedness.relationship_summary` | unique related/unrelated pair counts, related-pair density, closest-degree and relatives-by-degree distributions |
| `relatedness.inbreeding` | distribution of F (only with `--inbreeding`) |
| `popgen.effective_size` | per-estimator dict for the eight Ne estimators (`ne_inbreeding`, `ne_coancestry`, `ne_variance_family_size`, `ne_sex_ratio`, `ne_individual_delta_f`, `ne_long_term_contributions`, `ne_hill_overlapping`, `ne_caballero_toro`); `ne_coancestry` is null without `--ne-coancestry` |
| `strata.sex_summary` | per-sex sub-aggregates (n, n_founders, n_reproductive, n_terminal, …) |
| `strata.depth_summary` | per-depth sub-aggregates (one row per depth) |
| `individual.distributions` | mean/std/quartiles/`nz` for each per-individual numeric column |

Floats are rounded to 4 decimal places. With `--tsv`, Ne scalar
values also appear in `summary.pedigree.tsv` under
`effective_size_scalars`; per-depth vectors stay YAML-only.

## Logging

- Default: per-section progress at `INFO` level on stderr.
- `-v` / `--verbose`: also shows per-degree timings and matrix-product
  diagnostics from the relationship enumerator.
- `-q` / `--quiet`: warnings only.

---

Internals, engine semantics, and performance thresholds: see [DESIGN.md](DESIGN.md).
