# What `pedsum summarize --safe-attempt` does

This describes the safe mode of `pedsum summarize` in the unreleased work
after v0.17.0. It is taken from the release rules in
`pedsum/safe_summary.py` and the engine in `pedsum/disclosure.py`.

```bash
pedsum summarize --in PED.tsv --out SAFE_DIR --safe-attempt [--min-cell N] [--id-log]
```

## Purpose

Safe mode is for a pedigree held in a secure environment, such as Statistics
Denmark's research servers, where only aggregate results may leave. It writes
a report meant to be transferable under a minimum-cell rule.

It is an attempt, not a guarantee. Nothing checks your report when it is
written; the protection is only as good as the rules. The person transferring
the file keeps final responsibility, and the report must be read before it is
transferred. See [Known limits](#known-limits).

## How it works

Safe mode does not redact the ordinary report. It has a release list: every
value that may be written has a rule with a written reason. The report is
built from that list and nothing else. A value with no rule is absent,
including anything a later version of pedsum adds to the ordinary report.

Which rules apply depends on the options and the input's columns, not on the
data. Inside a table the labels do come from the data: the depths present, the
sexes present, the birth decades present. A blanked table has no cells.

### The minimum cell

`--min-cell N` is the fewest individuals, families or couples a released count
may rest on. The default is 5 and the smallest accepted value is 3. The report
records it as `safe_attempt_min_cell`.

A zero is always released. "At least N" below means every nonzero cell.

### Tables

Each frequency table is written with a `status` and its `total`.

| `status` | Meaning |
|---|---|
| `intact` | Every nonzero cell is at least N. The table is written as counted. |
| `merged` | An ordered table (depth, the F histogram) had a small bin at an end. End bins were merged inward until the merged bin reached N. A zero bin at the end is absorbed into the merged bin. |
| `blanked` | The table is withheld. Only its total is written, when the total itself may be. |

An ordered table is blanked, not merged, when a small bin sits between two
larger ones or when merging would leave fewer than two bins. Depth counts of
30, 3, 50, 40 blank the depth table.

Two-way tables are never merged: one small cell blanks the whole table.

### Linked tables

Some tables repeat the totals of a finer table. Where the rules declare that
link, the coarser table is written only while the finer one is intact, so a
released total cannot give back a cell of the withheld table. A table that
waits this way still shows its own total.

The declared links are:

- The by-status and by-sex offspring tables wait on the depth-by-offspring
  table.
- The inbred table, the F histogram and the by-sex inbred table wait on the
  depth-by-inbred table.

Not every such pair is linked. The twin pairs by sex do not wait on the twin
pairs by birth decade.

### Statistics

A statistic is written only when the tables and counts its rule names are
intact. A statistic of a merged or blanked table is `null`. The rule names the
main inputs, not every input a reader could combine with it.

Decimal values are rounded to four places.

### Pair counts

A count of pairs is not a count of people, so it is gated on the people
behind it. A relationship code needs N distinct families among the
individuals who have a relative of that code, and N distinct ancestors.

A family is the individuals who share a recorded mother and father. An
individual with no recorded parent is a family alone.

| Relationship | Ancestors counted |
|---|---|
| Maternal half sibs | distinct shared mothers |
| Paternal half sibs | distinct shared fathers |
| Full sibs, MZ twins | the smaller of distinct mothers and distinct fathers |
| Grandparent and deeper lineal pairs | distinct ancestors at that distance |
| Cousins, avuncular and similar pairs | a lower bound: the fewest ancestors that could account for them |

One woman's five children by five men count as one mother, so their ten
half-sib pairs are withheld. One small code blanks the whole table.

The ancestor counts are approximate. They are taken over all recorded
ancestors of the individuals involved, not over the pairs themselves, and
under inbreeding an ancestor who is also a closer relative is counted again.

## What the report contains

`N` below is the minimum cell.

### Run description

| Value | Rule |
|---|---|
| `input`, `command`, `version`, `generated_at` | Always written. See the warning below. |
| `safe_attempt_min_cell` | Always written. |
| `n_total` | The number of individuals. Withheld when 1 to N−1. |
| `max_degree_enumerated` | The `--max-degree` option. Always written. |

The recorded command is only partly protected. The `--in` and `--out` values
and every absolute path (a word or `--opt=value` value that starts with `/` or
`~`) are reduced to their base names, whatever option they follow. Everything
else is written as typed: the file name itself, relative paths given to other
options, column names, and tokens such as `N/A`. Do not put an identifying name
in any of them.

### Whole-pedigree tables

Each splits every individual into classes.

| Table | Classes | Rule |
|---|---|---|
| `size_structure.sex` | female, male, unknown | Every nonzero class at least N, or blanked. |
| `size_structure.parents_known` | both parents, mother only, father only, founder with descendants, founder without | Every nonzero class at least N, or blanked. |
| `size_structure.depth` | depth 0, 1, 2, … | Merged or blanked as under [Tables](#tables). The labels show the deepest depth. |
| `size_structure.depth_stats` | mean, sd and quartiles of depth | Only when the depth table is intact. |
| `size_structure.sex_source` | where each sex value came from | Only when the input has a `sex_source` column. Every nonzero class at least N, or blanked. |
| `reproduction.by_status` | has recorded offspring or not | Both classes at least N, and only with the depth-by-offspring table intact. |

### Counts

| Value | Rule |
|---|---|
| `components.n_components` | The number of connected components. Withheld when 1 to N−1. No component size is released. |
| `mating_pairs.n_pairs` | The number of distinct couples. Withheld when 1 to N−1. Zero is written. |
| `relationship_pairs.by_code` | Pairs by closest relationship, gated as in [Pair counts](#pair-counts). Parent-offspring codes are left out because they are sums of parents-known cells. |
| `relationship_pairs.engine` | The name of the counting engine. Always written. |

### Inbreeding

Written unless `--no-inbreeding` is given. Inbred means F above 1e-9.

| Value | Rule |
|---|---|
| `inbreeding.inbred` | Individuals by whether they are inbred. Both classes at least N, and only with the depth-by-inbred table intact. |
| `inbreeding.F_hist` | Individuals by F range. Merged or blanked as under [Tables](#tables). No F value is released. Same condition as the inbred table. |

No mean or standard deviation of F is written, pedigree-wide or per depth. F
takes few distinct values, so a mean beside a count can give back how many
individuals hold each value.

### Two-way tables

Each is written whole, and only when no cell is small.

| Table | Rows by columns | Extra condition |
|---|---|---|
| `depth_summary.sex` | depth by sex | |
| `depth_summary.reproductive` | depth by has offspring | |
| `depth_summary.inbred` | depth by inbred | Absent with `--no-inbreeding`. |
| `depth_summary.parents_recorded` | depth by which parents are recorded | |
| `sex_summary.reproductive` | sex by has offspring | Only with the depth-by-offspring table intact. |
| `sex_summary.inbred` | sex by inbred | Absent with `--no-inbreeding`. Only with the depth-by-inbred table intact. |

### Offspring sex concordance

Written with `--sex-concordance`, or with `--sex-concordance-permutations`
above zero, which turns it on. There are three groupings: full sibships,
maternal offspring groups and paternal offspring groups.

| Value | Rule |
|---|---|
| `null_model.*` | Six fixed fields describing the test. Always written. |
| `n_groups_eligible` | The offspring groups the test uses. Withheld when 1 to N−1. |
| `eligible_by_sex` | The offspring the test uses, by sex. Both classes at least N, or blanked with its total. |
| `test` | Only when the group count and the by-sex table are released. No per-group or per-size row is written. |
| `holm` | The Holm-adjusted p-values `p_analytical_holm`, `p_holm` and `permutations.p_holm`. Only when the group count and the by-sex table of every grouping are released, else `null`. |

The `test` block holds the observed and excess pair concordance, the
direction, z, the analytical p-value and `p_source`, the mean
and standard deviation of the groups' male proportions, and the permutation
settings and p-values. A test that was not computed reads `not_computed`.

A grouping's group count and by-sex table are also withheld when its
offspring differ by 1 to N−1 from a set they are nested in: the offspring of
the full sibships, the individuals with that parent recorded, or the
individuals of a sex. The full
sibships' group count and by-sex table are also withheld when the couples
(`mating_pairs.n_pairs`) outnumber the eligible full sibships by 1 to N−1,
and a maternal or paternal grouping's when the reproductive females or males
(`sex_summary.reproductive`) outnumber its eligible groups by 1 to N−1.

### Multiple births

Written with `--multiple-birth-col`.

| Value | Rule |
|---|---|
| `twin_pairs_by_sex` | Sets of two by the sexes of the pair, with a class for unknown sex. Every nonzero class at least N, or blanked. Larger sets are not released. |
| `weinberg.mz_fraction_estimate` | Weinberg's MZ share, from the known-sex classes. Only when the twin-pair table is intact. |
| `twin_pairs_by_birth_decade` | Sets of two by birth decade and sexes. Needs `--birth-year-col`. Written whole, only when no cell is small. The labels show the range of decades. |

The unit here is a delivery, not a family. Two twin deliveries to one couple
count twice.

The `FS` and `MZ` pair counts are released too, so a reader can subtract
the twin pairs from them. The full-sib pairs (`FS` plus `MZ`, since linked
MZ co-twins leave `FS`) are the twin pairs that are full sibs and the
full-sib pairs beyond them, and the twin pairs are those that are full sibs
and those that are not. With `--zygosity-col`, the same-sex twin pairs are
those linked as MZ and the rest, and `MZ` is those linked pairs and the MZ
pairs outside the table, such as in triplets. These parts are written
nowhere. When the full-sib pairs beyond the twin pairs come from 1 to N−1
families, or any other part is 1 to N−1, `twin_pairs_by_sex` and
`twin_pairs_by_birth_decade` are withheld with their totals, and Weinberg's
share with them. The pair table by code stays: it holds every relationship,
so the twin tables are the smaller loss.

## What is never written

- An explicit minimum or maximum, such as `max_F`. Table labels still show the
  deepest depth and the range of birth decades.
- A per-individual file. `annotated.tsv.gz` and `summary.individual.tsv` are
  not written.
- A histogram of a per-individual count of relatives (offspring, mates, sibs,
  ancestors) or any sibship-size table. Pair counts are sums over those
  tables, and the two together would give back a merged tail.
- The table of related pairs at the same depth.
- Any mean or standard deviation of the inbreeding coefficient.
- A per-depth mean of a per-individual count: `mean_distinct_ancestors` and
  `mean_descendant_paths`. Such a count takes few values, so its mean times
  the depth size can give back how many individuals hold each value.
- A statistic of how descendants are spread over founders:
  `effective_founders_by_descendant_paths` and the mean and standard deviation
  of equivalent complete generations.
- Founder, link and edge counts, the degree rollups of pair counts, and
  fractions beside their counts. Each is a sum or ratio of cells already
  written.

## Files

| Ordinary file | Safe file |
|---|---|
| `summary.yaml` | `summary.txt` |
| `summary.extra.yaml` | `summary.extra.txt` |
| `summary.pedigree.tsv` (with `--tsv`) | `summary.pedigree.tab` |
| `summary.individual.tsv` (with `--tsv`) | not written |
| `annotated.tsv.gz` | not written |

`summary.extra.txt` holds the sex-source table, the F histogram, the per-depth
tables, the twin pairs by decade and the concordance by-sex tables.
`summary.txt` holds the rest. Both repeat the run description.
`summary.pedigree.tab` holds the same released values as the two text files in
one flat table, so writing it adds no information.

The content is YAML text and tab-separated text. The extensions are ones on
Statistics Denmark's list of transferable file types. Whether YAML text under
a `.txt` name is acceptable has not been confirmed with Statistics Denmark.

A safe run refuses, with exit 1 and nothing written, an `--out` directory that
holds any file other than the ones this run writes. The check is on file names
only. A run that fails part-way can leave the reports of an earlier safe run
in place, so transfer only after a run that completed, and check that
`generated_at` matches in every file.

## The log

The log is never transferable.

Under `--safe-attempt` the log is scrubbed. Counts and other numbers drawn
from the input print as `[withheld]`, as does any text pedsum cannot show and
every line from another library. A traceback is replaced by the exception's
type. The numbers left are option values, the thread count, the minimum cell,
elapsed seconds and the memory limit.

The scrubbing has two known holes:

- A Rust panic message is written to stderr before pedsum sees it.
- Any direct write to stderr, such as a progress bar on a terminal, is not
  scrubbed.

`--id-log` writes the withheld lines in full to `summarize.ids.local.log` in
the output directory, for use inside the secure environment only.

## Known limits

- **The tests cover the rules, not your report.** Tests check, by exact
  integer programming over the declared relations between tables, that no
  small withheld cell has a single possible value. They run on test pedigrees,
  the generated ones of at most 70 rows. A relation nobody declared is not
  checked. Code reviews of this feature each found such relations, and more
  may remain.
- **Each run is judged alone.** Nothing compares a report with an earlier
  export of the same data, with a report run under other options, or with
  totals published elsewhere. Differences between two transferred reports can
  disclose what neither does alone.
- **The recorded command is only partly protected.** Relative paths and
  tokens are written as typed. See [Run description](#run-description).
- **No dominance rule.** A count that passes the minimum cell can still be
  shaped by one very large family. Sib pair counts are an example.
- **Statistics gate on named inputs only.** See [Statistics](#statistics).
- **Approximate pair support.** See [Pair counts](#pair-counts).
- **Rare classes blank tables at any size.** The gates act on cell sizes, not
  on the size of the pedigree. Almost every leak found so far needed a group
  of one to four people, which is likelier in a small pedigree, but a large
  one with a rare class is treated the same way.

## Before transferring a report

1. Confirm the run completed and that `generated_at` matches in every file.
2. Read every entry marked `blanked` or `merged`, and check that no released
   total beside it gives the hidden number back.
3. Read the `input` and `command` lines for anything identifying.
4. Compare with anything already transferred from the same data.
5. Transfer the report files only. Never the log or the `.ids.local.log` file.
6. To cut the exposure further, run with `--no-inbreeding` and without
   `--sex-concordance` and `--sex-concordance-permutations`.
7. Keep `--min-cell` at 5 or higher, or at the value the register requires.
