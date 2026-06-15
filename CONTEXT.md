# pedsum

Pedsum is a CLI for summarizing and validating pedigrees. This document is the glossary of terms used in pedsum input, output, and discussion. Implementation details belong in `DESIGN.md` or `docs/adr/`.

## Language

**Depth**:
An individual's topological distance from the founder set: founders have depth 0, and every other individual has depth `max(parent_depth) + 1`. Always derivable from the pedigree alone.
_Avoid_: generation, ped_depth, gen

**Birth Year**:
The calendar year of birth for one individual, supplied as the optional input column `birth_year` (sentinel `-1` for unknown). One of two analytical stratification axes in pedsum — the other is **Depth**. Required by `ne_hill_overlapping`, which models populations whose parents and offspring may coexist in time. Range-checked against `[--birth-year-min, --birth-year-max]` and topologically validated (`child.birth_year >= parent.birth_year` on every known edge) when supplied.
_Avoid_: "year of birth", "DOB"

**Mating Pair**:
An ordered `(mother, father)` tuple that produced at least one offspring in the pedigree.
_Avoid_: bare "pair" in this sense; "couple", "parental dyad"

**Relationship Pair**:
An unordered pair of two individuals classified by their pedigree-topological relationship (e.g. full sibs `FS`, first cousins `1C`, grandparent `GP`).
_Avoid_: bare "pair" in this sense

**Individual Pair**:
An unordered pair of two distinct individuals, used as a combinatorial denominator (`n choose 2`). Carries no relationship semantics.
_Avoid_: "possible pair", "all-pairs"

**Sibship**:
The set of full siblings sharing the same mother and father — equivalently, the offspring of one **Mating Pair**. *Sibship size* is the cardinality of this set.
_Avoid_: family, nuclear family, brood

**Descendant Path**:
A directed parent-to-child path from a focal individual to one of its descendants. Counted by `n_descendant_paths`. An inbred descendant is reachable by multiple paths and contributes multiply.
_Avoid_: bare "descendant count" in this sense

**Distinct Ancestor**:
A unique individual that appears anywhere in the focal individual's ancestry. Counted by `n_distinct_ancestors` (computed by `pedigree-graph` via sparse transitive closure). An ancestor reached through multiple paths is counted once.
_Avoid_: bare "ancestor count"; "ancestor path" (pedsum does not compute path-counted ancestors today)

**Distinct Descendant** *(reserved)*:
A unique individual reachable as a descendant of the focal individual via any path. Pedsum does **not** compute this today — only `n_descendant_paths` is available. The name `n_distinct_descendants` is reserved for future use; until then, use `n_descendant_paths` and divide mentally by inbreeding multiplicity if needed.
_Avoid_: claiming `n_descendant_paths` equals distinct-descendant count on inbred pedigrees

**Reproduction**:
The output section holding per-individual *reproductive-output* statistics aggregated over all individuals: offspring counts, mate counts, and the classification of individuals into **Reproductive** vs **Terminal**.
_Avoid_: lineage (now reserved/dropped — see Flagged ambiguities)

**Reproductive** / **Terminal**:
An individual is **Reproductive** if it has at least one offspring in the pedigree, and **Terminal** otherwise. A property of the individual, not of any line of descent. Terminal does not mean "leaf of the pedigree as observed" — it means biologically childless within the pedigree.
_Avoid_: "leaf", "tip", "childless line"

**Genealogy**:
The output section holding per-individual *ancestry* statistics aggregated over all individuals: the summary-stats distributions of `n_descendant_paths` (as `descendant_paths`) and `n_distinct_ancestors` (as `distinct_ancestors`). The asymmetry between paths-for-descendants and distinct-for-ancestors reflects what `pedigree-graph` currently computes; see those term entries.
_Avoid_: lineage (now dropped — see Flagged ambiguities)

**Mate Count**:
For one individual, the number of distinct opposite-sex partners with whom that individual shares at least one offspring in the pedigree (`0` for unmated individuals). Per-individual column is `n_mates`; aggregated under **Reproduction** as `mate_count` (summary stats over **all** individuals) and the sex-stratified `mate_count_male` / `mate_count_female` (over all males / all females respectively, zero-included).
_Avoid_: "mates" as a noun for the count; "partners"

**Founder**:
An individual whose mother and father are both missing in the input pedigree (in-degree 0). Counted by `n_founders`. Founder is a structural property of the pedigree; it does not imply genetic, reproductive, or temporal status.
_Avoid_: "ancestor" (unqualified), "progenitor", "root"

**Half-Founder**:
An individual with exactly one parent missing in the input pedigree. Counted by `n_half_founders`. Not a **Founder**.
_Avoid_: "partial founder"

**Founder Ancestor**:
For one (non-founder) individual, a **Founder** that appears anywhere in that individual's ancestry. The per-individual count of *distinct* Founder Ancestors is the column `n_founder_ancestors`; its summary-stats distribution within each depth cohort is `founder_summary.by_depth[*].founder_ancestors`.
_Avoid_: "founder line per individual", "ancestral founder count" (use the canonical term)

**Effective Founders**:
A scalar variance-weighted equivalent count, defined as `(Σ wᵢ)² / Σ wᵢ²` for a vector of per-Founder contribution weights `w`. The weight choice **must** be named in the field: `effective_founders_by_descendant_paths` (weights = `n_descendant_paths` per Founder), `effective_founders_by_descendants_at_depth_d` (weights = descendants of each Founder at depth *d*), etc. Bare `effective_founders` is forbidden.
_Avoid_: "founder equivalent number" (unqualified)

**Effective Population Size (Ne)**:
The size of an idealised Wright-Fisher population that would exhibit the same value of a chosen genetic-drift parameter (rate of inbreeding accumulation, variance in allele frequency, coalescent rate, …) as the observed pedigree. Multiple operational definitions exist; **they are not interchangeable**. Pedsum reports eight named estimators under `popgen.effective_size:`, each carrying its model in its name (`ne_inbreeding`, `ne_variance_family_size`, `ne_sex_ratio`, `ne_individual_delta_f`, `ne_long_term_contributions`, `ne_hill_overlapping`, `ne_caballero_toro`, `ne_coancestry`). Bare `ne` is forbidden — every reference carries the estimator qualifier. Definitions, assumptions, and references for each estimator live in `DESIGN.md` (and README), not here.
_Avoid_: bare "Ne", "the effective size"

**Component**:
A maximal connected subgraph of the pedigree, treated as an undirected graph over parent-child edges and Mating-Pair partnerships. Counted by `n_components`. A *Singleton* is a Component of size 1.
_Avoid_: "cluster", "family" (lay use)

**F (Inbreeding Coefficient)**:
For one individual, the pedigree-based probability that the two alleles at a locus are identical-by-descent given the pedigree. Per-individual column is `F`; computed via Meuwissen-Luo recursion. Range [0, 1].
_Avoid_: "inbreeding" as a noun standing alone

**Inbred**:
An individual with `F > 0`. Counted by `n_inbred`. No tolerance threshold — pedigree F is exact under the infinite-loci model; users wanting a non-trivial threshold should read from the `inbreeding.hist` bins.
_Avoid_: using "inbred" to mean "above a threshold"; that's a user choice, not a pedsum definition

**Degree** (of a **Relationship Pair**):
The number of meioses separating the two individuals along their shortest pedigree path. Degree 0 = monozygotic twin (`MZ`); degree 1 = parent-offspring (`PO`) or full sib (`FS`); degree 2 = grandparent (`GP`), half-sib (`MHS`/`PHS`), avuncular (`Av`); … up to degree 5. Half relationships share the degree of their full counterpart but half the expected IBD share.
_Avoid_: "degree of relatedness" (kinship coefficient), "generations apart"

**Bottleneck**:
The minimum across depths of a per-depth founder-contribution statistic (e.g., minimum active-founder count, minimum effective founders by descendants). Reported under `founder_summary.bottleneck:` with the depth(s) at which the minimum occurs. Not a separate genetic event — purely a minimum over a per-depth series.
_Avoid_: implying a demographic crash event; the metric is descriptive, not causal

## Naming conventions

How terms compose into output keys:

| Pattern | Meaning | Example |
|---|---|---|
| `n_<noun>` | A per-individual integer column (appears in the wide `annotated.tsv.gz`). | `n_offspring`, `n_mates`, `n_descendant_paths`, `n_distinct_ancestors`, `n_founder_ancestors` |
| `<noun>_count` | Summary-statistics distribution `{mean, std, min, q1, median, q3, max, nz}` of the matching `n_<noun>` across some population. | `offspring_count`, `mate_count`, `descendant_paths`, `distinct_ancestors` |
| `<noun>_count_hist` | Binned PMF (e.g. `{'0', '1', '2', '3', '4+'}`) of the same per-individual integer. | `offspring_count_hist`, `inbreeding.hist` |
| `<noun>_count_<sex>` / `<noun>_count_hist_<sex>` | Sex-stratified variant of either; the `_male` / `_female` suffix always comes last. | `mate_count_male`, `offspring_count_hist_female` |

Bare `descendant_paths` and `distinct_ancestors` (and other plurals from the glossary) refer to *summary-stats distributions* of the matching `n_*` column when they appear as keys inside an output section.

## Validation

**Check**:
One named integrity test applied to an input pedigree (e.g. `duplicate_ids`, `acyclic`, `sex_role_consistency`). The `validate` command reports every Check; `summarize` runs the same Checks but stops at the first failure.
_Avoid_: test, rule, assertion

**Finding**:
One recorded violation of a **Check**, scoped to a specific `id` / row where applicable. The `validate.log` carries one line per Finding.
_Avoid_: error, issue, warning, problem

**Check Status**:
The outcome of a **Check**: `PASS` (ran, no **Findings**), `FAIL` (ran, ≥1 **Finding**), or `SKIP` (did not run — a prerequisite Check did not PASS, the Check is inapplicable to this input, or a flag tolerates it).
_Avoid_: result, state, outcome (as bare nouns in this sense)

**Blocking Check**:
A **Check** whose `FAIL` prevents `validate` from writing a fixed pedigree, surfaced to the user as `BLOCKED`. Non-blocking failures still write the fixed output and report **Findings**.
_Avoid_: hard error, fatal, fatal error

**Reduced Pedigree**:
The pedigree `validate --drop-offending` emits after removing every individual named in a droppable **Finding** — and clearing references to each removed individual — so that the result passes every applicable **Check** under the flags it was produced with. A Reduced Pedigree is a *different* pedigree from the input: it has fewer individuals, individuals that lost a parent become **Half-Founders**, and relationships routed through removed individuals are gone — so relatedness, **F (Inbreeding Coefficient)**, **Effective Population Size (Ne)**, and **Founder** counts computed on it differ from the input. Column/file-level Checks (a missing column, a non-integer id column) are never resolvable by removal and still `BLOCK`.
_Avoid_: "cleaned pedigree", "filtered pedigree", "fixed pedigree" (the latter is the un-reduced imputed + founder-added output)

## Relationships

- **Depth** is a property of an individual, derivable from the pedigree.
- **Birth Year** is a property of an individual, supplied as input — not derivable from the pedigree.
- A **Mating Pair** produces zero or more **Relationship Pairs** among its offspring (`FS`, and through descendants, `1C`, `Av`, etc.).
- Every **Mating Pair** and every **Relationship Pair** is also an **Individual Pair**, but not vice versa.
- A **Sibship** is the offspring set of exactly one **Mating Pair**; the mapping Mating Pair ↔ Sibship is one-to-one.
- A **Check** produces zero or more **Findings**; its **Check Status** is `FAIL` iff it produced at least one. A Check may depend on other Checks: if a prerequisite did not `PASS`, the dependent Check `SKIP`s.

## Example dialogue

> **User:** "I ran pedsum on a 200-individual pedigree and `n_descendants` for one founder is 74 — but the pedigree only has 170 non-founders. How can one person have 74 descendants out of 170?"
>
> **Maintainer:** "That field is a **Descendant Path** count, not a count of **Distinct Descendants**. Your founder has 74 distinct directed parent-to-child paths reaching descendants — when there's inbreeding, the same descendant is reached by multiple paths and contributes multiply. In the new schema this field is named `n_descendant_paths` to make that explicit; bare `n_descendants` is forbidden in output."

> **User:** "The `ne_variance_family_size` says 102 and `ne_inbreeding` says 29. Which one is the real Ne?"
>
> **Maintainer:** "Neither is *the* Ne — there is no single Ne. Each estimator measures a different aspect of genetic drift under different model assumptions. `ne_variance_family_size` uses the variance of offspring counts under random mating; `ne_inbreeding` uses the slope of mean F across depths. They disagree because your pedigree violates the assumptions of at least one — likely both. Pick the estimator that matches the assumption you care about; see `DESIGN.md`."

> **User:** "What's the difference between `n_founders_with_descendants` and `active_founders` at depth 4?"
>
> **Maintainer:** "Both count **Founders** by their descendants, but at different scopes. `n_founders_with_descendants` is Founders with ≥1 descendant *anywhere in the pedigree*. `active_founders` at depth 4 is Founders with ≥1 descendant *at depth 4*. A Founder whose only descendants are at depths 1 and 2 counts in the first but not the second."

> **User:** "Why does `sibship_size.size_dist` only sum 99 sibships but the pedigree has 170 non-founders?"
>
> **Maintainer:** "A **Sibship** is the offspring set of one **Mating Pair**. Multiple non-founders share a Mating Pair — your 170 non-founders are partitioned across 99 Mating Pairs, hence 99 Sibships. The per-individual offspring-count distribution lives separately under `reproduction.offspring_count:`."

## Flagged ambiguities

- "generation" was used in pedsum output to mean topological depth (`gen_counts`, `by_generation`, `generation_summary`, `n_generations_used`). Resolved: these are all renamed to use `depth`. The word "generation" is not a pedsum analytical axis — outputs are stratified by **Depth** (topological) or **Birth Year** (temporal). The term is intentionally omitted from the glossary so readers don't expect a "generation"-named output anywhere.
- "pair" historically meant three distinct things — parental dyads, topologically-classified relationship pairs, and combinatorial all-pairs. Resolved: bare "pair" is forbidden; every use is **Mating Pair**, **Relationship Pair**, or **Individual Pair**. The YAML section is `relationship_pairs:`.
- "family" historically meant a sibship (in `family_size:`), and the section mixed per-sibship and per-individual distributions. Resolved: "family" is not a pedsum term; the section is `sibship_size:` (per-sibship only); per-individual distributions (`offspring_count_hist`, `frac_with_full_sib`) live under `reproduction:`.
- "ancestors" / "descendants" silently meant different things — descendants were *paths*, ancestors were *distinct* — with a buried `descendant_count_semantics: path_count` field hinting at the asymmetry. Resolved: bare `n_ancestors` / `n_descendants` is forbidden; the columns are `n_distinct_ancestors` (today's computation) and `n_descendant_paths` (today's computation). `n_distinct_descendants` is reserved for future work but not yet computed.
- "lineage" historically named a section bag of per-individual properties, but in pop-gen "lineage" means a single ancestral line. Resolved: the section is split into **Reproduction** (offspring, mates, reproductive/terminal) and **Genealogy** (`descendant_paths` and `distinct_ancestors`). "Lineage" is not a pedsum term.
- "mates" appeared four times across three sections under three names — the same per-individual quantity sliced inconsistently. Resolved: canonical term is **Mate Count**; lives once under `reproduction:` (`mate_count`, `mate_count_male`, `mate_count_female`) as summary-stats distributions over **all** individuals (resp. all males, all females), zero-included. The per-individual mate-count fields are removed from `mating_pairs:` (per-Mating-Pair statistics only) and from `sibship_size:` (per-sibship only).
- "founder" was historically unqualified across five concepts (structural Founder; Founders with descendants; Founders surviving to depth *d*; variance-weighted equivalent count; founder lines per individual). Resolved: only the structural **Founder** uses the bare term; everything else is expressed via Founders + their descendants. `active_founders` is "Founders with ≥1 descendant at this depth"; per-individual count of distinct Founder Ancestors is the column `n_founder_ancestors`, aggregated as `founder_summary.by_depth[*].founder_ancestors`; `effective_founders` always carries a `_by_<weight>` qualifier. "Founder Line" is not a pedsum term.
- "inbred" sometimes informally means "F above some threshold." Resolved: in pedsum, **Inbred** means strictly `F > 0`; users who want a threshold use the `inbreeding.hist` bins.
- "degree" can mean meiotic-path degree (pedsum's usage) or "degree of relatedness" / "degree of kinship" (a kinship-coefficient interpretation). Resolved: pedsum **Degree** always means *meiotic distance*; the kinship coefficient is reported separately.
- validation vocabulary was informal — "error" / "issue" / "warning" for a violation, "test" / "rule" for an integrity test, "hard error" / "fatal" for a fail that stops output. Resolved: a single integrity test is a **Check**; one recorded violation is a **Finding**; a Check's outcome is its **Check Status** (`PASS` / `FAIL` / `SKIP`); a Check whose failure stops the fixed-pedigree write is a **Blocking Check** (`BLOCKED`).
