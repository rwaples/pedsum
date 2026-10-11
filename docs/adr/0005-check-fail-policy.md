# ADR 0005: Check Fail Policies, and one Fixed Pedigree for `validate` and `summarize`

Status: accepted

## Context

ADR 0002 made the **Checks** a declarative registry, and ADR 0003 added one
fact to each entry: `droppable`, whether `--drop-offending` can resolve a
`FAIL` by removing the flagged individual(s). What a `FAIL` does to plain
`validate` stayed outside the registry, as a hand-written list of six Checks
and their messages in `cli.py`. The two modes therefore answered "does this
block?" in two places:

- `--drop-offending` blocked on every non-droppable Check except
  `parent_refs_present_*`, derived from the registry;
- plain `validate` blocked on its six listed Checks only.

The column- and parse-level Checks were on neither list for plain
`validate`. A missing required column, a row without an id or an
unrecognized sex token wrote the input unchanged to `validate.tsv.gz` and
exited 1, although the `--out` help says the file is "not written if a block
is detected" ([pedsum#21](https://github.com/rwaples/pedsum/issues/21)).

A Check's `FAIL` actually has one of four fates, and `droppable` recorded
only half of the information needed to tell them apart:

| Fail Policy | plain `validate` | `--drop-offending` |
|---|---|---|
| `block` | BLOCK | BLOCK |
| `block_or_drop` | BLOCK | remove the flagged ids |
| `keep_or_drop` | write the pedigree with the contradiction kept | remove the flagged ids |
| `autofix` | synthesize founders | synthesize founders |

The pedigree `validate` writes was assembled in `cli.py` (founder rows for
missing parents, phantom parents, the PLINK sex column and `sex_source`, the
topological reorder) with helpers in `report.py`, while `load_and_validate`
repeated the sex-imputation logging and the reorder to build the frame
`summarize` reads. The `--drop-offending` orchestration and its self-verify
also lived in the CLI, so both `validate` paths could be tested only through
argv.

## Decision

- **Each `Check` declares a `fail_policy: FailPolicy`** with the four values
  above, replacing `droppable`. A `block_or_drop` Check may also carry a
  `block_message`: the line plain `validate` prints under `BLOCKED`, with any
  flag that tolerates it.
- **`blocking_failures(results, *, drop_offending)` is the one place that
  answers "is this run BLOCKED, and why"**, in registry order. Plain
  `validate` blocks on `block` and `block_or_drop`; `--drop-offending` blocks
  on `block` only. `DROPPABLE_CHECKS` (`block_or_drop` and `keep_or_drop`) is
  derived from the same field; `NON_REDUCIBLE_BLOCK_CHECKS` and the CLI list
  are deleted.
- **Every column- and parse-level Check is `block`**: `required_columns`,
  `empty_pedigree`, `id_dtype`, `sex_tokens`, `negative_ids`,
  `parent_token_range_mother`, `parent_token_range_father`,
  `birth_year_dtype`. This changes plain `validate`: these inputs now exit 2
  without `validate.tsv.gz`, the same as under `--drop-offending`.
- **`sex_role_consistency` stays `keep_or_drop`**, also under
  `--no-override-asserted-sex`. The README had said that flag "reverts to
  hard-blocking"; the code never did, and the README is corrected.

- **One Fixed Pedigree.** `pedsum/fixed_pedigree.py` builds it:
  `fix_pedigree(ctx, fill_half_founders=..., next_id=...)` returns a
  `FixedPedigree` (ID codes, resolved sex, parents, depth, the input row of
  each output row, the synthesized founders, labels), and
  `render_fixed_pedigree` turns it into the string frame `validate` writes.
  `load_and_validate` returns `fix_pedigree(ctx).frame()`, so `summarize`
  reads the same rows, sex and order `validate` writes for that input. Every
  Check has passed there, so no founder is synthesized; an assert holds that.
- **`fix_validated(ctx, results, ..., write=...)` decides what `validate`
  writes**: BLOCKED, the Fixed Pedigree, or the Fixed Pedigree of a Reduced
  Pedigree, which it re-validates from the written file. The CLI passes the
  writer, prints the outcome, and exits with its code.

## Considered alternatives

- **Add only a second boolean (`blocks_plain`) next to `droppable`.** Two
  booleans allow a fifth combination (blocks plain, never dropped, but not
  column-level) that no Check has, and still leave `autofix` as a name check
  on `parent_refs_present_*`. One four-valued field names every fate.
- **Block in plain `validate` only when parsing leaves no ids**
  (`required_columns`, `id_dtype`, `empty_pedigree`). This would keep writing
  a file for an unrecognized sex token or an unparsable birth year, whose
  output still carries the bad token. Rejected: one rule for both modes, and
  the file is only written when it can be read back.

- **Self-verify the Reduced Pedigree in memory**, by rebuilding a context
  from the rendered frame. It would drop the file round-trip, but then nothing
  checks that the written TSV reads back the same; the writer is passed in
  instead.
- **Let `summarize` accept `keep_or_drop` failures**, as `validate` writes
  them. Out of scope: `summarize` stays fail-fast, and only the source of its
  frame changed.

## Consequences

- A new Check must pick a Fail Policy; the dataclass has no default, and
  `tests/test_fail_policy.py` pins every Check's policy and the derived
  blocking for both modes.
- Plain `validate` exits 2 on inputs where it used to exit 1 and write a raw
  file. Pipelines that read `validate.tsv.gz` after an exit 1 on such input
  were reading an unvalidated pedigree.
- The BLOCKED lines of plain `validate` follow registry order rather than the
  old hand-written order. No test or documented interface depended on it.
- `validate.tsv.gz` and the `summarize` outputs were byte-identical (after
  decompression) before and after the move to one Fixed Pedigree, on a corpus
  of 30 `validate` and 6 `summarize` runs. `validate`'s INFO lines now use
  `summarize`'s wording for sex imputation and reordering.
- `acyclic` no longer requires `parent_refs_present_*`
  ([pedsum#22](https://github.com/rwaples/pedsum/issues/22)). It had been
  SKIPped whenever a parent ID was missing, an `autofix` failure that never
  blocks, so a cycle reached `validate.tsv.gz` with exit 1, or exit 0 under
  `--drop-offending`. The depth it reads already counts a parent with no row
  as a founder, so it now runs, and BLOCKs or drops the cycle, in every
  pedigree whose IDs parse. `fix_pedigree` asserts that every row has a depth.
- `self_loops` is `keep_or_drop`, but a self-loop is also a one-person
  cycle, so `acyclic` FAILs with it and plain `validate` BLOCKs. Under
  `--drop-offending` the `self_loops` policy is what removes it: `acyclic`
  drops only the members of cycles of two or more (ADR 0003).
