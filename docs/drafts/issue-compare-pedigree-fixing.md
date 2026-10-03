Title: Compare pedsum's `validate` repairs with those of other pedigree programs
Labels: question, documentation

## Question

`pedsum validate` repairs some pedigree problems and refuses others. How do those choices compare with the pedigree-cleaning functions in the R packages that the candidate benchmark pedigrees come from (`docs/benchmark-pedigrees.md`)? Where pedsum differs, is that deliberate, and is it documented?

This matters for the benchmark work. When pedsum's numbers disagree with a published value, we need to know whether the cause is the statistic itself or a different cleaned pedigree. Two examples:
- `optiSel::PedigWithErrors` contains deliberate errors that `prePed()` repairs.
- `purgeR::ped_rename` turns every unmatched parent into "unknown" instead of adding a founder row.

## How each program handles each problem

The other packages' behaviour was read from their source on the github.com/cran mirror and has **not** been checked by running them. The pedsum column is from `pedsum/validate.py` and `pedsum/checks.py` at `6f71852`. GeneticsPed could not be checked because its source was unreachable.

Legend:
- **E**: refuses (error / hard block)
- **W**: warns and continues
- **M**: prints a message
- **S**: changes the data silently
- **–**: not checked

| Problem | pedsum `validate` | optiSel `prePed` | visPedigree `tidyped` | nadiv `prepPed` | kinship2 `pedigree()` / `fixParents()` | purgeR `ped_rename` / `ped_sort` | pedtools `ped()` / `as.ped()` | pedigreemm `editPed` | nprcgenekeepr `qcStudbook` |
|---|---|---|---|---|---|---|---|---|---|
| Parent referenced but not a row | Adds a founder row; recorded in `validate.log` | S adds a row | S adds a row | S adds a row | E / S adds a row | **S sets the parent to unknown (no row added)** | E / adds a row + M | S adds a row | S adds a row (`recordStatus="added"`) |
| Duplicate ID | E; can be dropped with `--drop-offending`. Exact copies and conflicting rows are treated the same. | M; keeps the row with the most known parents | Exact copy: W + removed; conflicting: E | E | E | E | E | E (from base R) | Exact copy: S removed; conflicting: E |
| Cycle / own ancestor | E (`acyclic`); droppable | **M; breaks the loop by setting parents to NA** | E, listing the path | E only for dam-only or sire-only chains (mixed loops can get through) | E only later, in `kindepth` | E, but reported as "not sorted"; `ped_sort` has no check | E | No guard (unbounded recursion) | **W only** |
| Own parent | E (`self_loops`) | Via loop handling | E | E (numPed) | Via `kindepth` | E | E | E (validity) | W |
| Mother ID = father ID in the same row | E (`parents_distinct`) | Via the "both sire and dam" check | E, unless `selfing=TRUE` | **W, treated as selfing** | E | – (check commented out) | Allowed if sex is 0 | – | E |
| Individual used as both mother and father | E (`sex_role_ambiguity`), unless `--allow-missing-sex` and sex unknown | E | E, unless `selfing=TRUE` | W | E / S made female | – | Allowed only if sex is 0 | – | E |
| Recorded sex contradicts parent role | **Overrides by default** when the individual has a single role (counted as "overridden from role"); E with `--no-override-asserted-sex` | M + S overwrite | E | – | E / S overwrite | – | E | – | S overwrite (or listed with `reportErrors`) |
| Missing sex | Filled from role; an unsexed non-parent is E unless `--allow-missing-sex` | S filled from role | S filled from role | Only for added founders | Coded unknown (W if more than 25%) | – | Filled from role + M (no sex column) | – | **Not filled**; becomes "U" |
| Only one parent known | Kept as a half-founder | Kept; dummy parents only with `lastNative` | Kept | Kept | E / S adds a dummy parent | Kept | **E** | Kept | **S adds a dummy parent** |
| Parents listed after offspring | Topological reorder, recorded | S sort | S sort | S sort | Not needed | `ped_sort` sorts; `ped_rename` gives E | S sort (`reorder=TRUE`) | S sort | S sort |
| Missing-parent codes | `-1`, `NA`, blank, `.`. **`0` gives E** ("id=0 referenced as mother/father"). | `""`, `"0"`, `" "`, NA | `""`, `" "`, `"0"`, `"*"`, `"NA"`, NA | NA, 0 (W), `"*"`, -998 | NA, `missid` (0 or `""`) | Anything unmatched | `""`, `"0"`, NA | NA only (`"0"` becomes an ID) | NA, `"UNKNOWN"` |
| ID type / renumbering | Integers required; no renumbering | Optional numeric columns | Adds numeric columns | `numPed`, as a separate object | Invents `addin-N` IDs | **Always renumbers 1..N** | Character | Integer slots | Invents `U%04d` IDs |
| Parent born after offspring | E (`birth_year_topology`), only with `--birth-year-col`; droppable | – (a negative generation interval is set to NA) | – | – | – | – | – | – | **E** (also flags parents below a minimum age) |
| Rows removed | Only with `--drop-offending`, iterated to a pass; written to `validate.dropped.tsv`; warns above 10% | Duplicates; `keep` pruning | Rows with a missing ID (W) | Rows with a missing ID (W) | Separate `pedigree.trim` | `ped_clean` | – | – | Rows whose ID is `UNKNOWN` |
| Record of changes | `validate.log` (one row per finding) plus the drop manifest | Console messages | `ped_meta` | Warnings | – | – | Messages | – | `errorLst` / CSV when asked |

## Where pedsum differs, and what to decide

1. **Overriding recorded sex.** pedsum replaces a recorded sex that contradicts a single parent role by default. visPedigree and pedtools refuse; optiSel and nprcgenekeepr overwrite. Is overriding by default the right choice? Should the override count appear in `summary.yaml` as well as `validate.log`?
2. **Cycles.** pedsum refuses, or drops every member of the cycle under `--drop-offending`. optiSel keeps the individuals and cuts their parent links. Should pedsum offer the optiSel behaviour, for example `--break-cycles`, as a gentler reduction than dropping the individuals?
3. **Exact vs conflicting duplicates.** visPedigree and nprcgenekeepr remove exact copies quietly and refuse only conflicting ones. pedsum treats both as blocking. Should exact copies be an automatic fix?
4. **`0` as a missing parent.** Most of these packages and the PLINK `.fam` format use `0` for a missing parent, and pedsum refuses it. This overlaps with the `convert` issue. Should `validate` also accept `--missing-parent 0`?
5. **Unmatched parents.** purgeR turns them into "unknown"; pedsum adds founder rows. The two produce different founder counts, half-founder counts and Ne. This is expected, but the benchmark comparisons need to document it.
6. **Mother ID = father ID (selfing).** nadiv and visPedigree (`selfing=TRUE`) allow it for plant and monoecious pedigrees. pedsum always refuses. Is that in scope?
7. **Minimum parent age.** nprcgenekeepr flags parents younger than a species minimum. pedsum only checks `child.birth_year >= parent.birth_year`. Is an optional `--min-parent-age` worth adding?

## Proposed work

- [ ] Run each package's cleaner on the same hand-built fixtures (one fixture per row of the table) and record what it outputs. This confirms the table, which so far comes only from reading the code.
- [ ] Run `pedsum validate` on the same fixtures, and on `optiSel::PedigWithErrors`, and compare the cleaned pedigrees: rows, founders, parent links and sex.
- [ ] Decide items 1–7, and record the decisions in an ADR (`docs/adr/`).
- [ ] Add a "How pedsum's validation compares to other tools" section to the README, so users can tell which differences in published statistics come from cleaning.

## Bugs found in other packages while reading their code (unconfirmed, not run)

These came from reading the source, not from running it. Confirm them before filing upstream.
- **kinship2 `fixParents`:** the father-sex check tests the wrong index (`sex[mindex]` instead of `sex[findex]`).
- **pedigree `add.Inds`:** a missing ID used as both dam and sire is added twice.
- **nadiv `prepPed`:** a cycle that mixes dam and sire links can pass the loop check.

---
_Generated by [Claude Code](https://claude.ai/code)_
