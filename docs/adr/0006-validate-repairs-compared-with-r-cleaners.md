# ADR 0006: `validate`'s repairs, compared with the R pedigree cleaners

Status: accepted; implemented ([pedsum#25](https://github.com/rwaples/pedsum/issues/25)). Two points went
past the decisions below in implementation, both recorded under Consequences: each link `--break-cycles`
clears is a Finding of a Check, and loading keeps the `sex_source` column of its input.

## Context

[pedsum#7](https://github.com/rwaples/pedsum/issues/7) compares the repairs `validate` makes and refuses
with those of the pedigree-cleaning functions in eight R packages (optiSel `prePed`, visPedigree `tidyped`,
nadiv `prepPed`, kinship2 `pedigree`/`fixParents`, purgeR `ped_rename`/`ped_sort`, pedtools `ped`/`as.ped`,
pedigreemm `editPed`, nprcgenekeepr `qcStudbook`). The comparison was run, not read from source: the
fixtures, the R environment and the results are in
[pedsum-validate `cleaners/`](https://github.com/rwaples/pedsum-validate/tree/main/cleaners), against
pedsum 542ef12.

## Decisions

### A parent written `0`

Most of the R cleaners and PLINK `.fam` write a missing parent as `0`; optiSel, visPedigree, nadiv,
pedtools and purgeR read it so. pedsum reads `0` as an ID. An absent parent `0` used in both roles (the
usual `0 0` founder) blocked with two misleading lines ("sex conflict on missing parent(s)", "rows with
mother == father"), and `--drop-offending` then deleted every founder written that way. An absent `0` in
one role was synthesized as a founder: optiSel's `PedigWithErrors` has one father written `0`, which
`prePed` reads as missing and pedsum turned into an added founder named `0`.

- `0` stays an ordinary ID by default. In simACE's pedigrees ID 0 is a real individual, and in a subset
  that lost its row, reading `0` as missing would turn its offspring from half-sibs into unrelated
  **Half-Founders**.
- A new option, `--missing-parent-token TOKEN` (repeatable), adds tokens to the missing-parent set of
  every command that reads a pedigree.
- `validate` writes a parent read as missing through such a token as missing (blank) in
  `validate.tsv.gz`. The **Fixed Pedigree** then needs no option downstream: `summarize` and
  `effective-size` read the same pedigree `validate` wrote (ADR 0005) without repeating the flag.
- A new Check, `zero_parent_present`, Fail Policy `block`, FAILs when a parent ID `0` has no row of its own, in any role. Its
  BLOCKED line names the option. Being `block`, it also stops `--drop-offending`, which no longer removes
  founders over it.

### Exact duplicate rows

An ID on two identical rows blocked like an ID on two conflicting rows, and `--drop-offending` removed
both, so the individual vanished and its offspring lost a parent. visPedigree, optiSel and nprcgenekeepr
remove an exact copy; the other five refuse it. All but optiSel refuse a conflicting duplicate.

- A new Check, `duplicate_rows`, Fail Policy `autofix`: rows that say the same thing collapse to the
  first, with one Finding per removed copy, counted as a repair like a synthesized founder. Rows are
  compared as pedsum reads them: ID, parents, sex and birth year after parsing (so `NA` and `.` are the
  same missing parent, `M` and `1` the same sex), and every other column as text.
- `duplicate_ids` then covers only an ID whose rows differ by that comparison, a trait column included,
  and stays `block_or_drop`. pedsum never chooses between differing rows; under `--drop-offending` the
  individual is still removed.

### Cycles

A cycle BLOCKs plain `validate`, and `--drop-offending` removes its members (ADR 0003). optiSel's `prePed`
instead clears both parents of one member per loop and keeps everyone; visPedigree, pedtools, kinship2
(`kindepth`) and purgeR (`ped_rename`) refuse; nadiv (mixed dam/sire cycles) and nprcgenekeepr return the
cycle. On `PedigWithErrors`, `prePed` broke a four-member cycle that runs through mothers only by clearing
both parents of 276000802875148 (born 1976): the mother, born 2004, and a father born 1973 who is not in the
cycle.

- A new opt-in option, `--break-cycles`, clears only links the data show to be wrong: a self-link, and
  within a cycle, each link whose parent is born after the child (this needs `--birth-year-col`, or
  since [pedsum#28](https://github.com/rwaples/pedsum/issues/28) `--birth-date-col`: a link is
  compared on the two birth dates when both are known, else on the birth years). The
  individuals stay, and each cleared link is written to a manifest beside `validate.tsv.gz`.
- A cycle with no such link is left to the existing rule: BLOCK, or removal of its members under
  `--drop-offending`. pedsum does not pick a link to cut without evidence.

### Overriding recorded sex

pedsum overrides a recorded sex that contradicts the only parent role an individual has (ADR 0001, 0.9),
as optiSel, nprcgenekeepr and kinship2 `fixParents` do; visPedigree, pedtools and kinship2 `pedigree()`
refuse. The override was reported in the Check summary (`PASS (1 overridden from role)`), in one INFO
line on stderr that `-q` hides, and in the `sex_source` column; `validate.log` had no row for it and
`validate` exited 0. `summarize`, reading `validate.tsv.gz`, never saw it.

- The override stays the default, and `--no-override-asserted-sex` still turns it off.
- Each override is a Finding of a new Check, `sex_role_override`, with Fail Policy `autofix`, so it is in `validate.log` and
  counts as a repair like a synthesized founder: plain `validate` exits 1. It is not a `sex_role_consistency` Finding: that Check
  is `keep_or_drop`, and `--drop-offending` would remove every overridden individual.
- `summarize` counts `sex_source` values in `summary.yaml` when its input carries the column, so a summary
  of a Fixed Pedigree shows how many sexes came from parent roles.

### Unsexed individuals who are never parents

All eight R cleaners accept an individual with no recorded sex and no offspring. pedsum blocked it unless
`--allow-missing-sex` was given. Without the flag, 21 of the 25 published pedigrees in pedsum-validate
block, 20 of them only for such individuals: from 1 row in purgeR `dama` (of 1,316) to 4,691 in Soay sheep
Pedigree 2, mostly in files with no sex column. Such an individual cannot be sexed from roles, but it changes
no relationship and no F.

- The `unknown_sex` Check is retired. The only individuals it flagged were unsexed and never a parent:
  one used in one role gets its sex from that role, and one used in both is `sex_role_ambiguity`'s. Such an
  individual is kept with sex `0`, `validate` logs a WARNING with the count, and `summarize` accepts it.
  `--drop-offending` no longer removes it, so a parent that loses its only offspring in a reduction stays
  as well (ADR 0003 dropped it in the next round).
- `effective-size` still refuses its sex-dependent estimators (`ne_sex_ratio`, `ne_variance_family_size`,
  `ne_hill_overlapping`) while any row is unsexed.
- An unsexed individual used as both mother and father (`sex_role_ambiguity`) is unchanged: it blocks
  unless `--allow-missing-sex` is given, which now covers only that case.

### Selfing

A row whose mother and father are the same individual blocks (`parents_distinct`). nadiv allows it with a
warning, visPedigree with `selfing = TRUE` when that parent's sex is NA or `monoecious`, pedtools when the
sex is 0; the rest refuse or do not check. pedsum's **Mating Pair**, **Mate Count**, sex imputation and
three Ne estimators assume two sexes, so supporting selfing needs a third sex value and a definition of
each. Out of scope; tracked in [pedsum#24](https://github.com/rwaples/pedsum/issues/24). The README says
pedsum models two-sex pedigrees.

### Minimum parent age

`birth_year_topology` flagged only a child born before a parent. nprcgenekeepr, alone among the eight,
also refuses a parent younger than a species floor (2 years by default). A new option,
`--min-parent-age N` (default 0, today's rule), makes the Check `child.birth_year - parent.birth_year >= N`
on every known edge, with the same Fail Policy (`keep_or_drop`). Ages are whole years, since pedsum has birth
years, not dates; there is no per-species table and no separate floor per sex.

### Parents with no row

pedsum, like optiSel, visPedigree, nadiv, kinship2 `fixParents`, pedtools `as.ped(addMissingFounders =
TRUE)`, pedigreemm and nprcgenekeepr, adds a **Founder** row for a parent ID that has no row. purgeR instead
sets that parent to unknown, so the child becomes a **Half-Founder** (or a Founder) and the parent's offspring
are no longer half-sibs. Founder and Half-Founder counts and the Ne estimators differ between the two. This is
expected and stays as it is; the README and pedsum-validate's `differences.md` name it as a cleaning
difference.

### Smaller fixes

- `--missing-parent-token 0` with a row whose ID is 0 exits 2: a token cannot also be an individual.
- The `parent_refs_present_*` Finding text "(if file uses 0/NA/blank as missing token, convert to -1)" is
  wrong, since NA and blank are already missing; it names `--missing-parent-token` instead.
- The `--no-override-asserted-sex` help says it restores a hard-block. It does not: `sex_role_consistency`
  is `keep_or_drop` (ADR 0005), and `validate` writes the pedigree and exits 1. The help is corrected.
- Filling a missing sex from parent role stays a silent repair recorded in `sex_source`; only overriding a
  recorded sex is a Finding.
- `--allow-missing-sex` keeps its name, although with `unknown_sex` retired it covers only
  `sex_role_ambiguity`; renaming it would break invocations for little gain.

## Considered alternatives

- **Read `0` as missing whenever no row has ID 0**, with a WARNING. No flag needed for PLINK- and R-style
  files, but a pedigree subset that lost individual 0's row would silently lose that individual's
  sibships. Rejected for the explicit option.
- **Block an absent `0` only in both roles.** PedigWithErrors would keep running, still with a founder
  named `0`. Rejected: one rule for every absent `0`, and the user decides what it means.
- **Keep blocking exact duplicates, and only have `--drop-offending` keep one copy.** Leaves plain
  `validate` refusing a repair three of the R cleaners make and none can get wrong.
- **optiSel-style cycle breaking** (clear both parents of one member per cycle). Matches `prePed`'s counts,
  but discards links outside the cycle on a rule with no evidence behind it.
- **Refuse a sex/role contradiction by default** (visPedigree, pedtools). Reverses ADR 0001's 0.9 choice;
  a mistyped sex token is still more likely than a wrong parent link.
- **Warn on unsexed non-parents in `validate` but keep `summarize` requiring the flag.** The published
  pedigrees would still need the flag to summarize, for individuals that change no statistic `summarize`
  reports except the sex counts.
- **`--allow-selfing`**: deferred to pedsum#24.
- **Separate dam and sire age floors** (nprcgenekeepr's `minDamAge`/`minSireAge`). More options for a
  check few users set.

## Consequences

- Three new Checks with Fail Policy `autofix` (`duplicate_rows`, `sex_role_override` and `cycle_breaks`) and
  one with `block` (`zero_parent_present`); `unknown_sex` is removed. `tests/test_fail_policy.py` pins them.
- `cycle_breaks` was added in implementation: each link `--break-cycles` clears is one of its Findings, so a
  run that clears links exits 1 like any other repair and `validate.log` lists them, beside the manifest
  `validate.cleared_links.tsv`. It SKIPs without the flag.
- The Fixed Pedigree keeps an input `sex_source` column, when there is one, as the provenance of each sex
  this run did not change itself, so every command (`summarize`'s `annotated.tsv.gz`, `--sex-concordance`
  and counts among them) sees it. Loading `validate.tsv.gz` again finds every sex resolved, so without the
  column a sex `validate` filled or overrode passed for a recorded one. Founders `validate` adds are
  written with `sex_source = imputed_from_missing`.
- Findings after a collapsed duplicate row name the input's row numbers: the context keeps each remaining
  row's input position.
- Plain `validate` exits 1, not 0, when it overrides a recorded sex or collapses duplicate rows. Under
  `--drop-offending`, a run whose only FAILs are `autofix` Checks exits 0, as it already does for a
  synthesized founder.
- `summarize` stops on any FAIL, `autofix` ones included, as it does today for a parent with no row. Run
  on a raw input with a sex contradicted by role or a duplicate row, it now refuses where it used to
  succeed; run on `validate.tsv.gz`, which has neither, it is unchanged.
- Inputs that ran before now BLOCK when a parent written `0` has no row; `--missing-parent-token 0` is the
  fix. On PedigWithErrors it reads the father written `0` as missing, as `prePed` does, instead of adding
  a founder named `0`.
- Inputs that blocked before now run: exact duplicate rows, and unsexed individuals who are never parents.
  20 of the 21 published pedigrees that pass `--allow-missing-sex` in pedsum-validate no longer need it;
  the black robin pedigree still does, for two individuals used as both mother and father.
- `summary.yaml` gains the `sex_source` counts.
- ADR 0001's 0.9 sentence that a contradicted, used-as-both sex "stays a `sex_role_consistency`
  hard-block" was superseded by ADR 0005 and is marked so.
