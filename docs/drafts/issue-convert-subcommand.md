Title: Add a `convert` subcommand that turns other pedigree formats into pedsum input
Labels: enhancement

## Problem

We want to check pedsum against published pedigrees that come with published statistics. None of the candidates we found can be passed to `--in` as they are. Today pedsum reads a delimited table with **integer** IDs. Missing parents must be written as `-1`, `NA`, blank or `.`, and sex must use the default or PLINK encoding. Converting anything else is left to the README ("Preparing your data": the `awk` recipe for PLINK `.fam`, the polars snippet for string IDs, the `pl.read_excel` one-liner for Excel). Each recipe covers a single case, nothing tests them, and none of them keeps a record of what it changed.

Formats the candidate pedigrees actually come in:

| Format | Examples | What goes wrong today |
|---|---|---|
| GEDCOM 5.5 (`INDI`/`FAM` records) | royal92.ged; the Habsburg (Charles II) and Darwin–Wedgwood GEDCOMs; Ptolemaic dynasty | No reader. Parents come from `FAM` `HUSB`/`WIFE`/`CHIL` links, not from columns. |
| R `.rda` / `.RData` data frames | `GENLIB::genea140` (`ind, father, mother, sex`), `purgeR::atlas/dama/arrui/dorcas` (`id, dam, sire`), `kinship2::minnbreast`, `MCMCglmm::BTped`, `optiSel::PedigWithErrors`, `visPedigree::deep_ped`, `nadiv::Mrode2` | No reader. Missing parent is often `0`, which pedsum rejects with "id=0 referenced as mother/father". |
| `animal, dam, sire` tables (quantitative-genetics convention) | Soay sheep (`animal,dam,sire`, `NA` for missing, **no sex column**), Rum red deer, Florida scrub-jay (`ID, SIRE_ID, DAM_ID`) | Sex must be inferred from the parent's role. Rows that never appear as a parent have no sex at all. |
| String / alphanumeric IDs | optiSel (`276000812496744`-style numeric strings), visPedigree (`"0"` as missing), most GEDCOM `@I123@` | `column 'id' must be integer-valued` |
| PLINK `.fam` | — | Works only after the README's `awk` header and `0 → -1` recipe. |
| Excel sheets | Dryad deposits (e.g. scrub-jay `Core_Region_pedigree` sheet) | Only the README's polars one-liner. |
| Mother-only pedigrees | SRKW `orca.rda` (`mom`, no father column) | Needs an explicit empty father column. |

## Proposal

Add a subcommand, `pedsum convert --in FILE --from {auto,gedcom,rda,table,fam,excel} --out DIR`. It writes a canonical pedsum TSV plus an ID lookup table, so the output feeds straight into `validate` and `summarize`.

1. **Canonical output.** `DIR/pedigree.tsv` with `id, sex, mother, father`, integer IDs, `-1` for missing parents, and sex in the pedsum default encoding (`0` = F, `1` = M, missing left blank). Other columns are carried through.
2. **ID mapping.** `DIR/id_map.tsv` maps each `original_id` to its `id`. Assignment is deterministic, in first-appearance order, so the same input always gives the same IDs.
3. **Missing-parent tokens.** Default set `{0, -1, NA, "", ".", "0"}`, overridable with `--missing-parent TOKEN,...`.
4. **Column mapping.** Recognise common aliases: `animal/ind/id/IID`, `dam/mother/mom/MAT/motherid`, `sire/father/dad/PAT/fatherid`, `sex/gender/SEX`. Explicit `--id-col`, `--mother-col` etc. override them, consistent with the existing flags.
5. **Sex decoding.**
   - Reuse `parse._decode_sex` for the default and PLINK encodings.
   - Add string tokens: `M/F`, `male/female` in any case, and `1/2` with `--sex-encoding plink`.
   - When there is no sex column, leave sex missing and let `validate`'s existing role-based imputation fill it. Do not duplicate that logic.
6. **Readers.**
   - **GEDCOM:** a minimal 5.5 parser covering `INDI` with `SEX` and `FAMC`/`FAMS`, and `FAM` with `HUSB`/`WIFE`/`CHIL`.
     - A child listed in several families with conflicting parents is reported, not silently chosen.
     - Same-sex `HUSB`/`WIFE` and `SEX U` are passed through for `validate` to handle.
     - Names and dates are dropped by default, since they are personal data; `--keep-gedcom-fields` keeps them.
   - **R data:** read `.rda` through `pyreadr` as an optional dependency with a clear install hint, and pick the data frame with `--object NAME`.
   - **Excel:** `polars.read_excel`, with `--sheet NAME`.
7. **Conversion log.** `DIR/convert.log` (TSV) records each mapping decision: which column became which, the tokens treated as missing, how many IDs were renumbered, how many parents were referenced but absent as rows, and how many rows had no sex. This keeps the conversion auditable, in line with the `validate` / `validate.dropped.tsv` approach.
8. **No silent fixes.** `convert` only reshapes and recodes. Founder synthesis, sex imputation, topological reordering and drops stay in `validate`, so the integrity checks live in one place.

## Acceptance

- Round-trip tests on small fixtures for each reader: a hand-written GEDCOM with one family that has two marriages and one child with conflicting `FAMC`, an `.rda` with `0`-coded parents, an `animal,dam,sire` CSV with `NA` and no sex column, and a mother-only table.
- `convert`, then `validate`, then `summarize` runs end to end on `purgeR::atlas`, and reproduces N = 948 and 5 founders (both parents missing).
- The README's "Preparing your data" section points to `convert` and keeps the manual recipes as a fallback.

## Out of scope / follow-ups

- Downloading the benchmark pedigrees, and a harness that compares pedsum against published statistics (separate issue).
- Writing to other formats (PLINK `.fam`, GEDCOM).

---
_Generated by [Claude Code](https://claude.ai/code)_
