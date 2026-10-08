# pedsum — pedigree summary CLI

Source: <https://github.com/rwaples/pedsum>

A Python command line tool for summarizing and validating pedigrees.
Reports the size, structure, sibship-size distribution, relationship-pair
counts (up to degree 5), mating-pair structure, founder contributions,
per-individual reproduction and genealogy aggregates, depth- and
sex-stratified summaries, and per-individual statistics, and estimates
effective population size eight ways. Output terminology follows
[CONTEXT.md](CONTEXT.md).

Uses the [`pedigree-graph`](https://github.com/rwaples/pedigree-graph) package for relationship-pair detection.

## Install

Python ≥ 3.14 required. 

### Get the code

```bash
# clone from git 
git clone https://github.com/rwaples/pedsum.git
cd pedsum
# Install dependencies in a conda environment
conda env create -f environment.yml
# activate conda environment
conda activate pedsum
```

## Quick start

```bash 
# run like this:
python pedigree_summary.py validate --in ...
python pedigree_summary.py summarize --in ...
python pedigree_summary.py effective-size --in ...
python pedigree_summary.py epimight-input --in ...
```

```bash 
# help:
python pedigree_summary.py --help 
```

A 200-individual, 5-generation example pedigree
(`example_pedigree.tsv`) ships with the repo. Try it:

```bash
python pedigree_summary.py validate      --in example_pedigree.tsv --out /tmp/demo-validate
python pedigree_summary.py summarize     --in example_pedigree.tsv --out /tmp/demo
python pedigree_summary.py effective-size --in example_pedigree.tsv --out /tmp/demo-ne --birth-year-col birth_year
python pedigree_summary.py epimight-input --in example_pedigree.tsv --out /tmp/demo-epimight --pairs
```

`--out` is a directory (created if needed). The bare `summarize` writes
three files inside: `summary.yaml`, `summary.extra.yaml`, and
`annotated.tsv.gz`. It computes inbreeding (F) by default; pass
`--no-inbreeding` to skip it. Effective population size (Ne) has its own
command, `effective-size`, which writes `effective_size.yaml`.

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
| `DIR/validate.tsv.gz` | the pedigree with auto-fixes applied, plus a `sex_source` column (omitted on hard-block findings) |
| `DIR/validate.dropped.tsv` | with `--drop-offending`: the removal manifest (`id`, `check`, `round`) |

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
- The sex column written in one encoding, `0` = female, `1` = male,
  `-1` = unknown, whatever encoding the input used. `sex_source` says
  where each row's sex came from: `input`, `imputed_from_missing`,
  `imputed_from_role` or `unresolved`. To re-run pedsum on the file,
  leave `--sex-encoding` at `auto` or set it to `default`.
- Rows reordered so parents always precede children (topological
  order), if the input was not already ordered.

**`--fill-half-founders`** also gives every half-founder (an individual with
exactly one known parent) its own new founder in the missing slot: female (`0`) for
a missing mother, male (`1`) for a missing father. The new founders take IDs above every
ID in the input and are prepended with the other synthesized founders.
`ne_long_term_contributions` needs this, because it refuses any pedigree with
a half-founder. A new founder is unrelated to everyone, so kinship and F among
the input individuals do not change. Founder counts do: each half-founder
adds one founder, and founder-based statistics shift with them. Under
`--drop-offending` the fill also covers the half-founders that dropping
creates.

Hard-blocks (cycles, duplicates, sex conflicts on missing parents,
unresolved sex without `--allow-missing-sex`, sex-role ambiguity)
cause the fixed TSV to be skipped — fix the source data first.

**`--drop-offending`** turns those hard-blocks into a passing pedigree by
*removing* the offending individuals rather than refusing. It iteratively drops
every individual named in a droppable finding — clearing references to it, so
its children become half-founders (no cascade) — and re-runs until the pedigree
passes under the flags you gave, after the auto-fixes above. The result in
`DIR/validate.tsv.gz` is then a **Reduced Pedigree**: a *different*, smaller
pedigree, so relatedness, Ne, and founder counts computed on it differ from the
input. Every removal is recorded in `DIR/validate.dropped.tsv` (`id`, `check`,
the round it was dropped); pedsum warns when more than 10% of rows are removed
and exits non-zero whenever anything was dropped. Column- and parse-level
failures (a missing column, non-integer or negative IDs) still hard-block — no
removal can fix those.

The Reduced Pedigree is individual-level data (IDs, parents, and the dropped
IDs in the manifest), not aggregate statistics — `--safe-attempt` redaction
applies only to `summarize` output, never to `validate`. Treat
`validate.tsv.gz` / `validate.dropped.tsv` as the cleaned source pedigree, not
as a shareable artifact.

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
  expensive computation in pedsum, and its time grows with rows and
  steeply with **Depth** (see [Deep pedigrees](#deep-pedigrees)).
- `--max-degree N` — count relationship pairs up to degree `N` (1 to 5,
  default 5). Codes past `N`, and their `by_degree` entries, are null:
  not counted, which is different from zero. `structure.max_degree_enumerated`
  records `N`. Cost climbs steeply past degree 2. On the 783,029-row horse
  pedigree at `--threads 10`, `relationship_counts` took:

  | `--max-degree` | wall time |
  |---:|---:|
  | 1 | 0.61 s |
  | 2 | 1.36 s |
  | 3 | 221 s |
  | 4 | 632 s |
  | 5 | 1,833 s |

  Cannot be combined with `--per-individual-burden`, which always counts
  degrees 1 to 5.
- `--per-individual-burden` (`--per-individual-pairs` also accepted) — opt into the per-individual
  relationship-burden summary (`relationship_summary.relatives_total`,
  `.relatives_by_degree`, closest-degree distribution). Both paths run the
  same pedigree-graph row-streaming engine and report the same 23 exact
  counts. The burden sink accumulates per-person degree counts in O(N)
  memory without materialising pair lists.
  Off by default, and the standard summary is produced without it.
- `--sex-concordance` — opt into **Offspring Sex Concordance** (see
  below). Off by default. Adds
  `demography.offspring_sex_concordance` to both YAML files.
- `--sex-concordance-permutations N` — calibrate the sex-concordance
  p-values against `N` fixed-margin permutations (default `0`,
  analytical only). Implies `--sex-concordance`. **Any claim at
  p < 0.01 needs this** — see below.
- `--sex-concordance-seed INT` — seed for the permutation sampler
  (default `0`). No effect without
  `--sex-concordance-permutations`.
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
  case hard-blocks. `summarize` runs normally on such rows, F included.
  `effective-size` refuses them only when `ne_sex_ratio`,
  `ne_variance_family_size` or `ne_hill_overlapping` is selected, since
  those need every sex resolved; the other five estimators run.
- `--no-override-asserted-sex` — disable the default behavior of
  overriding asserted sex when topology unambiguously implies the
  opposite (asserted M used only as mother → F; asserted F used only
  as father → M). The missing→F/M imputation is unaffected. Reverts
  to hard-blocking on sex/role contradictions.

### Offspring Sex Concordance (`--sex-concordance`)

Asks whether resolved offspring sex is more or less concordant *within*
**Offspring Groups** than a pooled fixed-margin exchangeability null
predicts. Three groupings are analysed independently:

| Grouping | Group key | Requires |
|---|---|---|
| `sibship` | `(mother, father)` | both parents known |
| `maternal_offspring_group` | `mother` | mother known |
| `paternal_offspring_group` | `father` | father known |

```bash
# analytical screen
python pedigree_summary.py summarize --in PED.tsv --out DIR --sex-concordance

# with permutation calibration (implies --sex-concordance)
python pedigree_summary.py summarize --in PED.tsv --out DIR \
    --sex-concordance-permutations 1000 --sex-concordance-seed 7
```

**Statistic.** For each eligible group `g` with `M_g` males and `F_g`
females, the number of concordant within-group **Individual Pairs** is
summed:

```
C = Σ_g [ choose(M_g, 2) + choose(F_g, 2) ]
```

Conditioning holds the eligible group sizes and the global male/female
totals fixed and treats sex labels as exchangeable. The reported
`conditioning_male_fraction` is the *margin being conditioned on*, not
an estimated parameter. With `P = Σ_g choose(n_g, 2)`,
`S = Σ_g 3·choose(n_g, 3)` (indicator pairs sharing one offspring)
and `D = choose(P, 2) − S` (disjoint indicator pairs), the exact
conditional moments are

```
q2  = ([M]₂ + [F]₂) / [N]₂
q3  = ([M]₃ + [F]₃) / [N]₃
q22 = ([M]₄ + [F]₄ + 2[M]₂[F]₂) / [N]₄
E[C]   = P·q2
Var(C) = P·q2·(1−q2) + 2·S·(q3−q2²) + 2·D·(q22−q2²)
```

with `[x]_r` a falling factorial and `N = M + F`. These are computed as
exact rationals, so the zero-variance degeneracy test is a real
equality rather than a float64 near-miss.

**Eligibility, and why provenance is the headline axis.** Pedsum
resolves missing sex only for individuals used as a *parent*
(`validate.py`). Admitting imputed sex therefore makes eligibility
conditional on having reproduced. Fixed-margin conditioning absorbs a
*uniform* sex bias in reproduction, so that alone is harmless — but
**group-level** heterogeneity in selection (retaining a dam's daughters
and culling her sons, decided per group) is not absorbed, and it
inflates the test badly. Crucially, the harmless and harmful cases have
*identical* `sex_source` counts, so reporting provenance counts is not
enough to tell them apart. Consequently:

- the **headline** analysis admits `sex_source == "input"` only;
- an `all_resolved` block repeats the analysis admitting
  `imputed_from_missing` and `imputed_from_role` as a **sensitivity**;
- a pedigree with no input sex at all is refused
  (`skip_reason: no_input_sex`) rather than reported.

A group informs concordance only from two eligible offspring up;
eligible members are retained even when their siblings are not
(`n_groups_incomplete` records how often that happened).

**Inference.** `z = (C − E[C]) / √Var(C)` with a two-sided normal
p-value. The moments are exact and conditional; the **p-value is
asymptotic and screening-only**. Measured under the null it holds its
size at conventional levels for every level of group dominance, but in
the far tail it runs up to ~18× too liberal — and the excess sits
*entirely* in the positive (over-concordant) tail, i.e. exactly the
direction a user wants to find. `max_group_pair_share`
(`choose(n_max, 2) / P`) predicts the effect: ~0.003 is fine, 0.10 gives
~4.6×, 0.97+ gives 16–18×. **Any claim at p < 0.01 requires
permutations**; pedsum warns when you are about to make one without.

Permutations use the same fixed-margin null and the same two-sided
deviation from `E[C]`, report `(b+1)/(B+1)` (never zero), and run on
the headline analysis only. They are drawn with numba, a pedsum
dependency since 0.15, and the same seed gives the same p-values.
1,000 permutations is a reasonable starting point; cost scales
linearly in permutations × groups. Measured on a 10M-row pedigree with
2M groups: ~80 ms per draw, i.e. ~4 min for 1,000 permutations across
all three groupings. A guideline, not a runtime promise — scale it to your
own pedigree. Memory is `O(groups)`: the same run's peak RSS rose by
8 MB over the analysis-free baseline.

Everything except the sampler is effectively free: on that 10M-row
pedigree the grouping phase costs ~0.1 s and the analytical phase
(headline *and* sensitivity) ~0.3 s per grouping, against a 58 s
baseline run.

**Multiplicity.** Raw and Holm-adjusted p-values are reported across
whichever of the three groupings are computable. Holm is valid under
arbitrary dependence but conservative here: Sibship pairs are a subset
of both the maternal and paternal pair sets, so these are not three
independent hypotheses. Holm applies across groupings only — the
`by_group_size` rows are descriptive and are not tested.

**Interpretation limits.** Pedsum has no twin/multiple-birth
annotation and no birth order, pools offspring across the whole
multigenerational pedigree, generally cannot tell whether a
reproductive history is complete, and does not adjust for depth-,
cohort- or secular variation in sex probability. Multiple births,
sex-dependent stopping rules, missingness, ascertainment, and temporal
or depth structure can all produce or obscure concordance.

Between-family overdispersion in offspring sex is real and documented —
Wang et al. fit a beta-binomial rather than a binomial, associated with
older maternal age at first birth and the maternal variants *NSUN6* and
*TSHZ1*. It is **not** genetic: Zietsch et al. estimate the
heritability of offspring sex ratio at zero across 4.7 million births
(upper 95% CI 0.002). Sex-dependent stopping is a competing explanation
producing the same signature (Long & Zhang's coupon-collection
behaviour). The classical analysis framework is overdispersion
modelling (Lindsey & Altham; James). So a positive finding here is
plausible rather than automatically artefactual — but the *genetic*
reading is specifically ruled out, which raises rather than lowers the
stakes on the input-only headline. Excess pair concordance is a
moment-based test for exactly the beta-binomial overdispersion Wang et
al. fit; fitting that model is deferred, not merely omitted.

- Lindsey & Altham, "Analysis of the Human Sex Ratio by Using
  Overdispersion Models," *J. R. Stat. Soc. C* **47**(1), 149–157
  (1998). [doi:10.1111/1467-9876.00103](https://doi.org/10.1111/1467-9876.00103)
- James, "The variation of the probability of a son within and across
  couples," *Human Reproduction* **15**(5), 1184–1188 (2000).
  [doi:10.1093/humrep/15.5.1184](https://doi.org/10.1093/humrep/15.5.1184)
- Zietsch, Walum, Lichtenstein, Verweij & Kuja-Halkola, "No genetic
  contribution to variation in human offspring sex ratio: a total
  population study of 4.7 million births," *Proc. R. Soc. B*
  **287**(1921) (2020).
  [doi:10.1098/rspb.2019.2849](https://doi.org/10.1098/rspb.2019.2849)
- Long & Zhang, "The Coupon Collection Behavior in Human
  Reproduction," *Current Biology* **30**(19), 3856–3861.e1 (2020).
  [doi:10.1016/j.cub.2020.07.040](https://doi.org/10.1016/j.cub.2020.07.040)
- Wang, Rosner, Huang, Rich-Edwards, Laden, Hart, Penney & Chavarro,
  "Is sex at birth a biological coin toss? Insights from a
  longitudinal and GWAS analysis," *Science Advances* **11**(29),
  eadu7402 (2025).
  [doi:10.1126/sciadv.adu7402](https://doi.org/10.1126/sciadv.adu7402)

**Output fields.** `summary.yaml` carries the headline verdict per
grouping — `computed`, `skip_reason`, `n_groups_eligible`,
`n_offspring_eligible`, `excess_concordance`, `direction`, `p_holm`,
`p_source`, `max_group_pair_share`, and
`all_resolved_excess_concordance` — plus a shared `null_model` block.
`summary.extra.yaml` carries everything else: the full eligibility and
provenance counts, `conditioning_male_fraction`,
`n_within_group_pairs`, observed/expected concordance, `z`, raw and
Holm-adjusted analytical p-values, the permutation block
(`requested`, `completed`, `seed`, `p_raw`, `p_holm`), the
whole `all_resolved` sensitivity, the unweighted
`male_proportion_distribution`, and descriptive `by_group_size` rows.
No per-parent, per-Mating-Pair or per-Sibship record is ever emitted.

Under `--safe-attempt`, a grouping resting on fewer than five eligible
groups keeps its eligibility metadata but has concordance, direction,
inference and distributions nulled; counts of one through four are
nulled; `by_group_size` rows get small-cell redaction; permutation
count and seed may remain.

### Estimate effective population size

```bash
python pedigree_summary.py effective-size --in PED.tsv --out DIR [--estimators NAME[,NAME...]]
```

`--out DIR` is a directory (created if needed). The command writes one
file, `DIR/effective_size.yaml`. It holds the run metadata, a `status`
(`complete`, or `stopped_memory_limit`; see [Memory limit](#memory-limit)),
the `estimators_requested`, and one record per estimator under
`effective_size:` with its scalar `ne` and its per-depth arrays.

pedsum offers eight pedigree-based estimators. They measure different
things and are not interchangeable; [CONTEXT.md](CONTEXT.md) explains why
every name carries its estimator. All eight run by default.

| Estimator | Short name |
|---|---|
| `ne_inbreeding` | Ne_I |
| `ne_variance_family_size` | Ne_V |
| `ne_sex_ratio` | Ne_sr |
| `ne_individual_delta_f` | Ne_iΔF |
| `ne_long_term_contributions` | Ne_LTC |
| `ne_hill_overlapping` | Ne_H |
| `ne_coancestry` | Ne_C |
| `ne_group_coancestry` | Ne_GC |

All eight records are always written, and every `ne: null` carries a
`reason`. An estimator you did not select reports `{ne: null, reason:
not_requested}`; one that cannot run for want of metadata reports its own
`reason`, for example `missing_metadata`.
`ne_long_term_contributions` reports `missing_metadata` with `code:
incomplete_parentage` for any pedigree with a half-founder; run `validate
--fill-half-founders` first and pass it the written `validate.tsv.gz`.
An estimator that ran but found no estimate in these data reports
`reason: no_estimate` and a `code`. The rest of the record shows the
evidence:

| Estimator | `code` | Meaning |
|---|---|---|
| `ne_inbreeding`, `ne_coancestry`, `ne_group_coancestry` | `too_few_cohorts` | Fewer than 2 depths after the first have a usable mean (`n_depths_used`). |
| | `no_positive_rate` | The mean does not rise over depth, so the fitted `slope` gives no rate. |
| `ne_individual_delta_f` | `empty_reference` | No reference individual has a known parent (`n_reference: 0`). |
| | `reference_not_inbred` | The reference individuals have mean ΔF 0, so none is inbred. |
| `ne_variance_family_size` | `too_few_parents` | No depth has at least 2 males and 2 females, with offspring from each sex. |
| | `no_family_size_variance` | At each such depth, all males have the same number of offspring, and so do all females, so ΔF is 0. |
| `ne_sex_ratio` | `no_depth_with_both_sexes` | No depth holds both a male and a female. |
| `ne_long_term_contributions` | `no_founders` | The pedigree has no founder. |
| | `no_founder_contributions` | The last depth carries no founder contribution. |
| `ne_hill_overlapping` | `no_estimable_transition` | Collapsed to `ne_variance_family_size`, which has no estimate. |
| | `no_eligible_cohorts` | No birth-year cohort in the window has at least 2 of each sex. |
| | `no_estimable_cohort` | Eligible cohorts exist, but none gives a finite Ne. |

When `ne_individual_delta_f` has no estimate but its per-depth
`ne_per_gen` has one, the command logs a warning that names the reference
subpopulation and its `n_reference`. The default reference is the last
observed depth, which can hold few individuals; `--reference-col` picks
another.

Flags:

- `--estimators NAME[,NAME...]` — the estimators to run, as a comma list
  or repeated flags; `all` selects all eight, which is also the default.
  On the 783,029-row horse pedigree the eight take 1.11 s, and the whole
  command peaks at 394 MiB.
- `--birth-year-col NAME` — gives `ne_hill_overlapping` its cohort
  window and sex-decomposed `Ne_m` / `Ne_f`. Without it,
  `ne_hill_overlapping` collapses to `ne_variance_family_size`
  (`collapses_to_ne_v: true`).
- `--reference-col NAME` — a column flagging the **reference
  subpopulation** (`1`/`0` or `true`/`false`; missing counts as `0`) over
  which `ne_individual_delta_f` averages the individual increase in
  inbreeding, as Gutiérrez et al. (2008) prescribe. Without it the
  reference is the last observed depth. The record gains
  `reference_column`; the other seven estimators are unchanged. Its
  `ne_unrelated_founders` field divides by `t − 1` and is the value purgeR
  (`pop_Ne`) reports.
- The column, `--sep`, `--sex-encoding`, `--threads` and `--max-memory`
  flags work as in `summarize`.

Examples:

```bash
# all eight, with Hill's cohort window
python pedigree_summary.py effective-size --in PED.tsv --out DIR --birth-year-col birth_year

# only the two coancestry estimators, under an explicit 16 GiB limit
python pedigree_summary.py effective-size --in PED.tsv --out DIR \
    --estimators ne_coancestry,ne_group_coancestry --max-memory 16G
```

If the run crosses the memory limit, `effective_size.yaml` is still
written: the estimators that were running report
`{ne: null, reason: memory_limit, rss_gib, limit_gib}`, and the command
exits 3.

### Estimate the Mate Correlation

```bash
python pedigree_summary.py assortative-mating --in PED.tsv --out DIR --trait COL [COL2] [options]
```

`--out DIR` is a directory (created if needed). The command writes one
file, `DIR/assortative_mating.yaml`, with the run metadata and an
`assortative_mating:` block. It never runs as part of `summarize`.

pedsum reads and types the traits, assigns strata and writes the YAML.
The estimates and their inference come from
[pg-phenotype](https://github.com/rwaples/pg-phenotype)'s
`mate_correlation`, the method pedsum#13 designed, ported unchanged. Its
[reference](https://github.com/rwaples/pg-phenotype/blob/v0.1.0/docs/assortative-mating.md)
and [design notes](https://github.com/rwaples/pg-phenotype/blob/v0.1.0/docs/assortative-mating-design.md)
give the method in full; this section covers what the command reads and
writes.

The command reports the **Mate Correlation**, the correlation between the
mother's and the father's values of a trait, one observation per
**Mating Pair** ([CONTEXT.md](CONTEXT.md) defines both terms). One trait
gives one cell. Two traits give four cells, `R_mf[i][j] = corr(mother
trait i, father trait j)`, so rows are the mother's trait and columns the
father's. The two off-diagonal cells are estimated separately and need
not be equal. Two traits also add the **Within-Person Cross-Trait
Correlation** under `within_person`, once for mothers and once for
fathers. It counts each individual once who has both traits and at least
one analysed Mating Pair, uses the primary estimator of the cell with
the same trait types, and has no SE, CI or p-value.

**Trait types.** Each `--trait` column is typed from its non-missing
tokens:

| Non-missing tokens | Type |
|---|---|
| two distinct numbers, or `false`/`true`, or `no`/`yes` in any case | binary |
| more than 20 distinct numbers | continuous |
| 3 to 20 distinct numbers | error until `--trait-type` states the type |
| a token that is not a number | error |

Ordinal is never inferred; pass `--trait-type COL=ordinal`. An ordinal
trait must be numeric, and `--trait-type COL=binary` needs exactly two
levels. `traits[].levels` records the levels of a binary or ordinal
trait in order. Numbers sort numerically (`10` after `9`), and the first
level is the reference, coded 0. `type_source` records whether the type
was `inferred` or `stated`. A trait column that is absent, all missing or
constant, or that holds `inf` or a non-numeric token in a numeric trait,
makes the command log the error and exit 1 without writing the YAML.

**Missing values.** A token is missing when, after stripping whitespace,
it is blank, `.`, `?`, or `NA`, `NaN`, `N/A`, `None` or `NULL` in any
case. The reader's other null tokens, such as `#N/A` and `<NA>`, also
count. `--trait-missing TOK` adds a token, matched exactly after
stripping whitespace. There is no default numeric sentinel, so a code
such as `-9` must be passed. A founder that `validate` adds has no row in
the file, so its traits are missing. Each cell uses the pairs where both
of its values are present and counts the rest under
`n_dropped.mother_missing`, `father_missing` and `both_missing`.

**Estimators.** Each cell reports the estimators for its mother × father
trait types. The primary estimator, listed first, is the only one with a
permutation p-value and the only one with a stratified form.

| Mother × father | Primary | Also reported |
|---|---|---|
| continuous × continuous | `pearson` | `spearman` |
| binary × binary | `tetrachoric` | `table` (the 2×2 counts), `odds_ratio`, `phi` |
| binary × ordinal, ordinal × binary, ordinal × ordinal | `polychoric` | |
| continuous × binary, binary × continuous | `biserial` | `point_biserial` |
| continuous × ordinal, ordinal × continuous | `polyserial` | |

`phi` and `point_biserial` are Pearson's r with each binary side coded
0/1. `odds_ratio` is `ad / bc` on the 2×2 table, and `.inf` when
`bc = 0`. `spearman` uses average ranks for ties.

`tetrachoric` and `polychoric` are two-step maximum likelihood (Olsson
1979). Thresholds come from each sex's margins in closed form, then one
ρ maximises the likelihood of the pair table. `biserial` and
`polyserial` are the two-step estimator of Olsson, Drasgow & Dorans
(1982). The continuous side is standardised, the discrete side gets
thresholds from its margins, then ρ is fitted. Newton's method on the
analytic score finds ρ̂. It starts at 0 for `tetrachoric` and
`polychoric`, and at the ad hoc estimate of Olsson, Drasgow & Dorans
(1982, eq 38) for `biserial` and `polyserial`. If the curvature is not
positive, a step leaves (−0.9999, 0.9999), or 12 steps do not converge,
a bounded Brent search over that interval takes over, so every fit
reaches the optimum Brent would. The bivariate normal CDF comes from
Owen's T function (Owen 1956), and the thresholds from the AS 241 normal
quantile (Wichura 1988). Each latent record carries `boundary`, which is
`true` when the fit ends at a bound; the rule is in pg-phenotype's
[design notes](https://github.com/rwaples/pg-phenotype/blob/v0.1.0/docs/assortative-mating-design.md)
(threshold estimators). Means, SDs,
thresholds and the 2×2 table count each Mating Pair once, so a father
with three mates counts three times.

**Phi or tetrachoric.** For a binary trait, read `tetrachoric`, unless
you want a statement about the observed 0/1 values. Phi depends on
prevalence. Under a bivariate-normal liability with correlation 0.3 and
the same prevalence in both sexes, phi is 0.194 at 50% prevalence, 0.129
at 10% and 0.046 at 1%, so phi from rare diagnoses does not compare
across traits or cohorts. The same holds for `point_biserial` against
`biserial`. `tetrachoric`, `polychoric`, `biserial` and `polyserial`
estimate the correlation of a latent liability and assume that it is
bivariate normal, cut by thresholds into the observed levels. The other
estimators describe the observed values. Every estimate is phenotypic;
none is a genetic correlation.
On simACE pedigrees whose mate liabilities are bivariate normal by
construction, the tetrachoric recovers the liability correlation and phi
does not. With a liability Mate Correlation of 0.3506 over 3.5 million
Mating Pairs, cutting both liabilities at 10% prevalence gives a
tetrachoric of 0.3510 and a phi of 0.1567; at 30% the two are 0.3519 and
0.2154. When the liability is not bivariate normal, the tetrachoric
misses too. On a simACE scenario that pairs mates by matching
correlations rather than by a normal model, the 10% cut gives 0.4112
against a liability correlation of 0.3494.
[benchmarks/results/assortative_mating_simace.md](benchmarks/results/assortative_mating_simace.md)
lists every cell.

**Stratification.** `--stratify-by depth` puts each mate in the stratum
of their own **Depth**. `--stratify-by birth_year` uses their birth-year
bin, `--birth-year-bin` years wide (default 10). Bins start at multiples
of the width, so 1950 to 1959 is bin `1950`. A mother and her mate can
be in different strata. Continuous values are standardised to mean 0 and
SD 1 within each sex × stratum. Binary and ordinal sides get thresholds
per sex × stratum and share one ρ. Each cell then gains a `stratified`
block with the primary estimator, its SE, CI and p-value,
`n_strata_mothers` and `n_strata_fathers`.

With `--stratify-by`, pairs with a mate of unknown stratum (a birth year
of `-1`) are dropped before any cell and counted under
`mating_pairs.n_dropped.unknown_stratum`. Each cell then drops every
sex × stratum whose pairs come from fewer than `--min-stratum-networks`
**Mate Networks** (default 10, an arbitrary choice) or whose values are
constant. Dropping one stratum can thin another, so both rules repeat
until no pair is dropped. The cell counts the dropped pairs under
`n_dropped.small_stratum` and `n_dropped.degenerate_stratum`. `crude`
and `stratified` both use the pairs that are left, so their difference
shows only the adjustment.

**Inference.** Each estimate gets a standard error and a CI, and each
primary estimate also gets a permutation p-value. The YAML states the
assumptions under `inference:`.

- *Standard error.* `se` is a cluster-robust sandwich over the cell's
  Mate Networks of the stacked two-step estimating equations: the
  thresholds, stratum means and 1/N variances of the first step, then
  the ρ score. The uncertainty of the first step therefore reaches ρ̂.
  Influences are summed within each network, squared, summed over
  networks and scaled by `G/(G−1)` for `G` networks. Clustering by
  network keeps the dependence between pairs that share a remating
  parent. Networks are assumed independent; dependence between them
  through shared ancestry or siblings is not modelled. `se` is on the
  scale of the estimate (r or ρ), except for `odds_ratio`, where it is
  the SE of `log OR`. Spearman has no sandwich.
- *CI.* By default (`--bootstrap 0`) the CI is the Wald interval from
  `se`: `tanh(atanh(ρ̂) ± 1.96 · se / (1 − ρ̂²))` on the Fisher-z scale for
  correlations, and `exp(log OR ± 1.96 · se)` for the odds ratio, so the
  bounds stay in range (`ci_level: 0.95`, `ci_method: sandwich`).
- *Bootstrap.* With `--bootstrap N`, each of the N draws resamples the
  cell's Mate Networks with replacement. Pearson, Spearman, phi,
  point-biserial and the odds ratio are recomputed exactly on the draw.
  A latent correlation refits its thresholds and stratum moments on the
  draw, then takes one Newton step from ρ̂ with the draw's score and the
  full-sample Hessian. A draw whose Hessian at ρ̂ is not positive is
  refit in full. On 2,500 remated pairs the one-step CI bounds came
  within 0.0012 of a full refit on the same draws, and the gap shrinks
  about as 1/n. The CI runs from the 2.5th to the 97.5th percentile of
  the valid draws (`ci_method: bootstrap`); `se` is still the sandwich.
- *Permutation.* Each permutation shuffles the father trait vectors,
  both traits together, among distinct fathers in the same block, and
  leaves mothers in place. A block is father stratum × which traits the
  father has, so a father keeps one vector across all his mates, and
  each cell keeps its pairs. The null hypothesis is that, given the
  Mating Pair structure, father trait vectors are exchangeable within
  each block. That null is stronger than zero correlation. It also
  excludes non-linear association and any link between a father's values
  and his number of mates.
- *Permutation under stratification.* With `--stratify-by`, a father
  swaps only with fathers of his own stratum, for the crude estimate as
  well as the stratified one. Every `p_perm` then tests within-stratum
  exchangeability, while the crude estimate and its CI describe the
  pooled correlation. A cohort trend shared by both mates, with no
  assortment within a stratum, can give a crude CI that excludes 0 and a
  crude `p_perm` that does not reject. Without `--stratify-by` there is
  one stratum, and the crude `p_perm` tests the pooled correlation.
- *Permutation statistic.* For a latent primary
  (`permutation_statistic: score_at_zero`), each side's value is replaced
  by its latent score: the conditional mean of the liability given the
  level, under that side's thresholds, or the standardised value of a
  continuous side. `T` is the Pearson r of the mother and father scores
  over the cell's pairs, which is the score of the likelihood at ρ = 0
  scaled to a correlation. The father side's thresholds, means and SDs
  are refit on every permuted sample, because pair-weighted margins move
  when remating fathers swap. For two continuous traits `T` is the
  Pearson r itself (`permutation_statistic: pearson`), of the
  within-stratum standardised values for the stratified form. No
  optimiser runs per permutation.
- *Sequential p-value.* The p-value is two-sided by magnitude and
  sequential (Besag & Clifford 1991, closed scheme, `h = 20`). The draws
  are read in order. At the 20th valid draw with `|T*| ≥ |T_obs|`, ties
  included, the cell stops with `p_perm = 20 / l`, where `l` counts the
  valid draws so far. A cell that never reaches 20 runs all `N`
  permutations and reports `p_perm = (b + 1) / (B + 1)`, where `B`
  counts the valid permutations and `b` those at least as extreme. The
  sequential p is exact under the null at every sample size. A null cell
  stops after about 98 draws on average at `N = 999`; a cell with a
  strong signal never collects 20 exceedances and runs every
  permutation.

**Seeds, threads and the first run.** Each permutation and bootstrap
draw has its own random stream, SplitMix64 keyed by `--seed` and the
draw number, with separate streams for permutations and the bootstrap.
The stopping rule batches draws by draw count, never by thread count, and
every parallel sum adds fixed blocks in a fixed order. The same input and
`--seed` therefore give the same YAML, apart from `generated_at`, under
any `--threads`. pg-phenotype ships compiled, so the first run after an
install has no compile step.

**Output.** The `assortative_mating:` block holds:

| Field | Contents |
|---|---|
| `traits` | per trait: `name`, `type`, `type_source`, `levels` (binary and ordinal), `n_values`, `n_missing` |
| `settings` | `permutations`, `bootstrap`, `seed`, `threads` (the pg-phenotype thread count used), `ci_level`, `ci_method`, `stratify_by`, `birth_year_bin`, `min_stratum_networks` (the last two null when unused) |
| `inference` | `se_method`, `ci_scale`, `bootstrap_unit`, `bootstrap_assumption`, `bootstrap_method`, `permutation_null`, `permutation_blocks`, `permutation_statistic`, `permutation_stopping` (`besag_clifford_closed`) and `permutation_stop_h` (20), as fixed strings |
| `mating_pairs` | `n_total`, `n_dropped.unknown_stratum`, `n_mate_networks`, `largest_mate_network_share`, `n_mothers_multiple_mates`, `n_fathers_multiple_mates` |
| `mate_correlation` | one record per cell, named by its `mother` and `father` trait, in `R_mf` row order |
| `within_person` | two traits only: per sex, `estimator`, `n`, and the estimate |
| `notes` | the fixed caveats below |

Floats are rounded to 4 decimal places, or to 4 significant digits below
0.1, so a small p-value, SE or share is not written as 0.

A cell record holds `n`, `n_dropped` (`mother_missing`, `father_missing`,
`both_missing`, `small_stratum`, `degenerate_stratum`), the cell's own
`n_mate_networks` and `largest_mate_network_share`, and its `crude` and
`stratified` estimators. A sparse cell can rest on few networks even
when the whole pedigree does not. Each estimator record holds:

- the estimate, under `r`, `rho` (with `boundary`) or `value`
  (`odds_ratio`). An estimate the full cell cannot support is `null`
  with a `reason`, for example `constant_margin`.
- `se`, or `se: null` when the sandwich does not exist.
- `ci` with `ci_method` (`sandwich` or `bootstrap`), or `ci: null` with
  `ci_unavailable_reason`. Without `--bootstrap` the reasons are
  `single_mate_network` (the cell's pairs form one network, so no
  zero-width interval is published), `boundary` (a latent fit at a bound
  or a correlation of exactly ±1), `infinite_odds_ratio` (a zero cell),
  `sandwich_undefined` (the ρ equation is not concave at ρ̂), and
  `bootstrap_not_requested` (Spearman). With `--bootstrap`, they are
  `single_mate_network` and `too_many_failed_draws` (fewer than 95% of
  draws are valid).
- under `--bootstrap N` only, `bootstrap: {requested, valid, failed,
  failure_reasons}`. A draw fails when the estimate is undefined in it:
  `no_complete_pairs`, `constant_margin`, `degenerate_stratum`, or
  `empty_category` (a level that a stratum shows in the cell is absent
  from that stratum in the draw; levels are never merged). A ρ̂ at a
  bound and an infinite odds ratio are valid draws, so an odds-ratio CI
  can reach `.inf`.
- for the primary estimator only, `p_perm`, or `p_perm: null` with
  `p_perm_unavailable_reason` (`not_requested`,
  `no_informative_permutations`, or `no_valid_permutations`), its
  `permutation_statistic`, and `permutations: {requested, valid, failed,
  failure_reasons, seed, n_fixed_fathers, stopped_early, draws_used,
  sequential_h}`. `draws_used` is the number of draws read (`l` plus the
  failed draws among them; `valid`, `failed` and `failure_reasons` count
  those draws only), and `stopped_early` says whether it is below
  `requested`. `n_fixed_fathers` counts the cell's fathers who are alone
  in their block, so no permutation moves them. When all of them are,
  the reason is `no_informative_permutations`.

**Caveats.** `notes` repeats these in every file:

- Mating Pairs are seen only through offspring. A partnership with no
  recorded child is absent, so the Mate Correlation describes
  reproducing pairs.
- The estimates are phenotypic correlations. The latent estimators
  assume a bivariate-normal liability.
- Censored age-dependent diagnoses are not corrected. A mate who has not
  yet been diagnosed counts as unaffected.

Flags:

- `--trait COL [COL2]` names one or two trait columns, each once
  (required).
- `--trait-type COL=TYPE` states the type of a trait column, one of
  `continuous`, `binary` or `ordinal`. Repeat it for the second trait.
- `--trait-missing TOK` adds a token that means missing. Repeat it for
  more tokens.
- `--stratify-by {depth,birth_year}` turns on stratification.
  `birth_year` needs `--birth-year-col`.
- `--birth-year-bin YEARS` sets the bin width for `--stratify-by
  birth_year` (default 10).
- `--min-stratum-networks N` sets the Mate Network minimum per sex ×
  stratum (default 10). Without `--stratify-by` it has no effect, and
  `settings` records it as null.
- `--permutations N` (default 999) sets the most permutations a cell
  runs. A cell stops after about `20 / p_perm` of them; `0` turns the
  p-value off, and the records say `not_requested`.
- `--bootstrap N` (default 0) replaces the sandwich CI with a percentile
  CI from N one-step draws.
- `--seed INT` (default 0) keys every permutation and bootstrap draw.
  It takes any integer from -2^63 to 2^63-1.
- `--threads N` sets pg-phenotype's threads (the fits, the sandwich, the
  bootstrap and the permutations). The default is the physical cores the
  process may run on, with SMT siblings counted once: the permutation
  pass is memory-bound, and hyperthreads do not speed it up (see
  [Cost](#cost)). A `NUMBA_NUM_THREADS` below the request caps it.
  pg-phenotype takes one thread count per process, so a second run in
  the same process keeps the first one's. Results are identical under any
  value. The pedigree-graph engine keeps one thread unless `--threads` is
  given.
- The column, `--sep`, `--sex-encoding` and `--max-memory` flags work as
  in `summarize`.

Usage errors exit 2: more than two `--trait` columns, a column named
twice in `--trait` or in `--trait-type`, a `--trait-type` that is not
`COL=TYPE` or names a column that is not a trait, and `--stratify-by
birth_year` without `--birth-year-col`.

Examples:

```bash
# one binary diagnosis, with the default inference
python pedigree_summary.py assortative-mating --in PED.tsv --out DIR --trait dx

# two traits, -9 as missing, stratified by birth decade
python pedigree_summary.py assortative-mating --in PED.tsv --out DIR \
    --trait liability dx --trait-missing -9 \
    --stratify-by birth_year --birth-year-col birth_year
```

#### Cost

The figures below are from one workstation: an Intel i7-9750H with 6
cores, 12 threads and 31 GiB of RAM, shared with other jobs. The input came
from `benchmarks/generate_assortative_mating.py --pairs N --seed 0`: 2.76
rows per Mating Pair, 8 birth-decade strata, remating, and no missing
values. The command ran `--trait liab dx --stratify-by birth_year
--birth-year-col birth_year`, so all four cell types, with the default
inference (999 permutations, no bootstrap). Each figure is the median of
10 runs of pg-phenotype 0.1.1's code, each a fresh process, with the
1-minute load at most 12; peak memory is the cgroup `memory.peak` of the
run.

| Mating Pairs | Wall time, 6 threads | Wall time, 1 thread | Peak memory (6 threads) |
|---|---|---|---|
| 10^4 | 0.98 s | 1.08 s | 106 MiB |
| 10^5 | 2.36 s | 3.17 s | 233 MiB |
| 10^6 | 15.88 s | 28.07 s | 1,014 MiB |

- Against pedsum's own numba implementation, which pg-phenotype replaced,
  every configuration took 0.52 to 0.89 of the wall time and 0.74 to 1.00 of
  the peak memory (medians over 10 alternating pairs).
  [benchmarks/results/assortative_mating_cutover.md](benchmarks/results/assortative_mating_cutover.md)
  has every configuration, one and two traits at 1 and 6 threads.
- `--bootstrap 1000` at 10^5 took about 11 s with 6 threads.
- Load and validation take 2 to 2.5 s of the 10^6 runs.
- 3 × 10^6 and 10^7 pairs were measured only on the numba implementation
  (34.9 s and 3,273 MiB; 168.3 s and 8,315 MiB);
  [benchmarks/results/assortative_mating_cost.md](benchmarks/results/assortative_mating_cost.md)
  has those runs. pedsum's default memory watchdog sets its limit to 80% of
  the memory available at start; pass `--max-memory` on a machine with less
  free memory.

#### References

- Besag, J. & Clifford, P. (1991). Sequential Monte Carlo p-values.
  *Biometrika*, 78(2), 301–304. <https://doi.org/10.1093/biomet/78.2.301>
- Olsson, U. (1979). Maximum likelihood estimation of the polychoric
  correlation coefficient. *Psychometrika*, 44(4), 443–460.
  <https://doi.org/10.1007/BF02296207>
- Olsson, U., Drasgow, F. & Dorans, N. J. (1982). The polyserial
  correlation coefficient. *Psychometrika*, 47(3), 337–347.
  <https://doi.org/10.1007/BF02294164>
- Owen, D. B. (1956). Tables for computing bivariate normal
  probabilities. *Annals of Mathematical Statistics*, 27(4), 1075–1090.
  <https://doi.org/10.1214/aoms/1177728074>
- Wichura, M. J. (1988). Algorithm AS 241: The percentage points of the
  normal distribution. *Applied Statistics*, 37, 477.
  <https://doi.org/10.2307/2347330>

### Emit EPIMIGHT input

```bash
python pedigree_summary.py epimight-input --in PED.tsv --out DIR [options]
```

Builds the *structural skeleton* of an [EPIMIGHT](https://github.com/BioPsyk/epimight)
long-form input — one row per `person × disorder × relationship_kind` over the
eight relationship codes `PO, FS, HS, mHS, pHS, Av, 1G, 1C`. The columns a
pedigree determines are computed; the columns that need phenotype/affection/
demography are left as empty placeholders to fill downstream:

| Column | Source |
|---|---|
| `person_id`, `relationship_kind`, `relatives`, `born_at_year` | computed from the pedigree |
| `failure_status`, `failure_time`, `relatives_diagnosed`, `dead_at_year` | empty placeholders |

`--out DIR` is a directory (created if needed). Files written inside:

| File | When | Contents |
|---|---|---|
| `pipeline_input.tsv` | always | the long-form skeleton (one row per person × disorder × relationship_kind) |
| `relative_pairs.tsv` | with `--pairs` | the relative pairs backing the counts (`id1, id2, relationship_kind, kinship`) |
| `pipeline_input.parquet` / `relative_pairs.parquet` | with `--parquet` | parquet form of each emitted table (the format EPIMIGHT reads natively) |

In `relative_pairs.tsv`, the `kinship` column is the **nominal** coefficient
looked up by `relationship_kind` (the same value for every pair of a kind, e.g.
`FS → 0.25`, `1C → 0.0625`) — it is *not* computed from the pedigree, so it does
not reflect inbreeding or multiple relatedness paths. Pass `--exact-kinship` for
the pedigree-derived value.

Flags:

- `--pairs` — also write `relative_pairs.tsv`, the list of relative pairs that
  the skeleton's `relatives` counts aggregate. For directional kinds (`PO`,
  `Av`, `1G`) `id1` is the younger member; symmetric kinds are canonicalized
  `id1 < id2`. Materialises every pair, so it can be large on pair-dense
  pedigrees (cousins scale ~quadratically).
- `--exact-kinship` — add a `kinship_exact` column to `relative_pairs.tsv` with
  the **exact pedigree** kinship (inbreeding-, MZ-, and multi-path-aware), which
  can exceed the nominal value — e.g. inbred sibs (`0.375` not `0.25`) or double
  first cousins (`0.125` not `0.0625`). Runs the kinship recurrence over every
  pair, so cost scales with pair count × pedigree depth. No-op without `--pairs`.
- `--parquet` — additionally write the parquet form of each emitted table
  (written natively by polars).
- `--rels CODES` — comma-separated relationship codes to emit, in order
  (default: all eight).
- `--disorder NAME` — label for the single emitted `disorder` block (default
  `trait1`; a pedigree carries no trait).
- `--base-year YEAR` — calendar offset for the derived `born_at_year`
  (`base-year + generation`, default `1960`). No-op when `--birth-year-col` is
  set, in which case the real birth year is used.
- `--drop-founders` — drop founder-generation rows (off by default; useful when
  the output feeds an estimator, where a founder's degenerate full-sib stratum
  can break h² estimation).

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
| `birth_year` | `--birth-year-col NAME` is passed | Integer calendar year (sentinel `-1` for unknown). In `effective-size` it feeds the Hill overlapping-generation estimator; without it, Ne_H collapses to Ne_V. Range-checked against `[--birth-year-min, --birth-year-max]` (default `[1800, current_year + 1]`) and topologically checked (`child.birth_year >= parent.birth_year` on every known edge). |
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
re-export through polars:

```bash
python -c "import polars as pl; pl.read_excel('input.xlsx').write_csv('input.tsv', separator='\t')"
```

### String IDs (`"P001"`, `"FAM01-003"`, …)

The relationship enumerator requires integer IDs. Map them once and
keep a lookup table:

```python
import polars as pl

df = pl.read_csv("clinical.tsv", separator="\t", infer_schema=False)
all_ids = set()
for col in ("id", "mother", "father"):
    all_ids |= set(df[col].to_list())
lut = {sid: i for i, sid in enumerate(sorted(all_ids - {"-1"}))}
df = df.with_columns(
    pl.col(col).replace_strict(lut, default=-1, return_dtype=pl.Int64) for col in ("id", "mother", "father")
)
df.write_csv("clinical_int.tsv", separator="\t")
pl.DataFrame({"id": list(lut), "row": list(lut.values())}).write_csv("id_lookup.tsv", separator="\t")
```

## Large pedigrees

Two `summarize` defaults dominate the cost on large pedigrees: counting
relationship pairs through degree 5, and per-individual inbreeding (F).
For a first pass, stop pair counting at degree 2 and skip F:

```bash
python pedigree_summary.py summarize \
    --in cohort.tsv --out cohort/ --threads 10 \
    --max-degree 2 --no-inbreeding
```

On the 783,029-row horse pedigree, degree-2 counting takes 1.4 s where
degree 5 takes about 30 minutes (see `--max-degree`). Once the first
pass succeeds, raise `--max-degree` and drop `--no-inbreeding` as time
allows. `effective-size` is a separate run; its default eight estimators
take about a second on the same pedigree.

### Deep pedigrees

Row count is a poor guide to F's cost; **Depth** matters more. F walks
each individual's whole ancestor set, and that set can double with every
level of depth until it covers most of the earlier population. Measured
on generated 1M-row pedigrees (pedigree-graph 0.12.1, one thread):

| max depth | F | distinct-ancestor counts |
|---|---|---|
| 7 | 3.7 s | 0.3 s, 267 MB |
| 11 | 29 s | 2.3 s, 626 MB |
| 15 | 3.6 min | 18 s, 4.1 GB |
| 19 | 13.4 min | over 12 GiB, killed |

`summarize` computes the distinct-ancestor counts (`n_distinct_ancestors`)
together with F, so on pedigree-graph 0.12.1 a deep pedigree can run out
of memory as well as time. Before F starts, pedsum logs a WARNING when
rows times the largest possible ancestor set (`min(2^(depth+1) - 2,
rows)`) passes 4e9, about 1M rows at depth 11. `--no-inbreeding` skips
both.

`n_descendant_paths` counts paths, not individuals, so it also grows
with depth: on the same generated pedigrees it reaches 2e16 at depth 49
and overflows int64 by depth 59. `summarize` counts descendant paths
first and exits 1 with a one-line error when they overflow, before any
expensive phase.

## Memory limit

Every command runs under a best-effort memory limit. A background thread
reads pedsum's resident memory about once a second. When it passes the
limit, pedsum logs the step that was running and how much memory it
held, then exits with code 3.

- The default limit is 80% of the memory available at start: the
  smallest of the host's `MemAvailable` and the headroom (`memory.max`
  minus `memory.current`) of pedsum's cgroup and every enclosing cgroup.
  pedsum logs the limit it chose when it starts.
- `--max-memory SIZE` sets the limit, e.g. `--max-memory 12G` or
  `--max-memory 500M`. `--max-memory 0` turns it off.
- The limit narrows the window for the kernel's OOM killer; it does not
  close it. Memory is sampled once a second, so a fast enough allocation
  can still cross it before pedsum notices. To make a limit binding, run
  pedsum in a cgroup, e.g.
  `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 python pedigree_summary.py ...`.
- Every output file is written to a hidden `.<name>.partial-<pid>`
  beside it and renamed into place when complete, so a stop never leaves
  a truncated output. A stop mid-write can leave the `.partial-<pid>`
  file behind; delete it.
- `effective-size` keeps the estimators that finished before the stop
  (see above). A stop while the input is still being read writes the
  file with `n_total: null` and every requested estimator at
  `reason: memory_limit`, replacing any file an earlier run left. The
  other commands write nothing more after a stop.

Exit codes:

| Code | Meaning |
|---:|---|
| 0 | success |
| 1 | `validate` reported findings or `--drop-offending` removed rows; another command's input failed validation; `effective-size` refused unresolved sex; or `--out` names an existing file |
| 2 | bad arguments, an unreadable input file, or a `validate` hard-block |
| 3 | stopped at the memory limit |

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

### Equivalent complete generations (`ecg`)

`annotated.tsv.gz` carries `ecg`, each individual's **equivalent
complete generations** (Maignel, Boichard & Verrier 1996): the sum over
every known ancestor of `(1/2)^n`, `n` generations back, counting an
ancestor once per path. A founder has 0, a child of two founders 1, and
a child with one known founder parent 0.5. It measures pedigree depth
the way ENDOG, optiSel (`equiGen`), purgeR (`pop_t`) and visPedigree
(`ECG`) do, and its distribution is under
`individual.distributions.ecg` in `summary.extra.yaml`.

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
| `demography.offspring_sex_concordance` | within-group sex concordance against a fixed-margin null, per Sibship / Maternal / Paternal Offspring Group (only with `--sex-concordance`) |
| `individuals.reproduction` | per-individual offspring/mate counts and Reproductive/Terminal classification (`offspring_count`, `offspring_count_hist`, `mate_count`, sex-stratified variants, `frac_with_full_sib`) |
| `individuals.genealogy` | per-individual `descendant_paths` summary; `distinct_ancestors` summary when `--inbreeding` is set |
| `founders.founder_contribution` | Founders with descendants + `descendant_paths_per_founder` distribution + `effective_founders_by_descendant_paths` |
| `founders.founder_summary` | active Founders and effective-Founder contribution by depth, `founder_ancestors` per-depth distribution, bottleneck minima (may be skipped on very large pedigrees) |
| `relatedness.relationship_pairs` | 23 named Relationship codes through Degree 5 plus `PO = MO + FO`, with `by_degree[0..5]` rollup (TSV-only); codes past `--max-degree` are null |
| `relatedness.relationship_summary` | unique related/unrelated pair counts, related-pair density, closest-degree and relatives-by-degree distributions |
| `relatedness.inbreeding` | distribution of F (only with `--inbreeding`) |
| `strata.sex_summary` | per-sex sub-aggregates (n, n_founders, n_reproductive, n_terminal, …) |
| `strata.depth_summary` | per-depth sub-aggregates (one row per depth) |
| `individual.distributions` | mean/std/quartiles/`nz` for each per-individual numeric column |

Floats are rounded to 4 decimal places. Effective population size is not
in `summary.yaml`; `effective-size` writes it to `effective_size.yaml`.

## Logging

- Default: per-section progress at `INFO` level on stderr.
- `-v` / `--verbose`: also shows per-degree timings and matrix-product
  diagnostics from the relationship enumerator.
- `-q` / `--quiet`: warnings only.
- Relationship counting in `summarize` and `epimight-input` can run for
  many minutes on a large pedigree. In a terminal it shows a progress bar
  in rows walked; when stderr is redirected, it logs a progress line every
  30 s instead. `--quiet` turns off both.

---

Internals, engine semantics, and performance thresholds: see [DESIGN.md](DESIGN.md).
