# 0003 — `validate --drop-offending` pedigree reduction

Status: accepted

## Context

`validate` already emits a lightly-fixed pedigree (sex imputation, **Founder**
synthesis, topological reorder). But some **Checks** report contradictions that
no imputation can resolve — most commonly `sex_role_consistency` (an individual
used as a **mother** in one row and a **father** in another) and the other
per-individual contradictions (`sex_role_ambiguity`, `unknown_sex`,
`self_loops`, `parents_distinct`, `acyclic`, `duplicate_ids`,
`birth_year_range`, `birth_year_topology`, `parent_refs_sex_conflict`). Today
these either `BLOCK` (no fixed output) or survive in the fixed output as
genuine contradictions. Users with a large real pedigree want a way to obtain a
pedigree that *passes*, accepting that the irreconcilable individuals are
removed.

The naive reading — "drop the offending individuals" — hides two traps:
dropping a parent leaves dangling child references (and the existing
add-founders step would *resurrect* the dropped parent as a synthesized
**Founder**), and a single drop can spawn new **Findings**.

## Decision

Add an opt-in `--drop-offending` flag to `validate` that produces a **Reduced
Pedigree** (see CONTEXT.md):

- **Drop the flagged id.** The unit removed is exactly the `id` named in each
  *droppable* **Finding** — no per-Check bespoke target, no guessing. Removing
  an id means deleting its row (if it has one) and setting any parent slot that
  references it to `-1` (**clear-refs**, not cascade — see alternatives).
  *One exception:* `_check_acyclic` over-flags — it reports every node it cannot
  topologically order, i.e. cycle members **and** their descendants. Dropping
  all of them would be a de-facto cascade. So for `acyclic` the reduction drops
  only the true cycle members (strongly-connected components of size ≥ 2, via
  `scipy.sparse.csgraph`); descendants survive (as **Half-Founders** once the
  cycle edge is cleared) and re-validate the next round. The `acyclic`
  **Findings** in `validate.log` are unchanged.
- **Droppable vs blocking.** Only the per-individual Checks above are
  droppable. Column/file-level Checks (`required_columns`, `*_dtype`,
  `negative_ids`, `parent_token_range`, `empty_pedigree`) still `BLOCK` — no
  row removal can fix a missing column or a non-integer id column.
- **Iterate to a fixpoint.** Each round re-imputes sex from current role usage,
  detects droppable Findings, and drops + clears refs; repeat until a round
  finds none (each round removes ≥1 id, so it terminates; bounded by N).
  Re-imputing every round keeps "after all possible imputations/corrections"
  true at the fixpoint. Then run the existing add-founders step and a final
  self-verify asserting the result passes (a bug guard, not an expected path).
- **Compose with tolerances.** A tolerated Check (`--allow-missing-sex`,
  `--no-sex-check`) `SKIP`s and emits no Finding, so it is simply never in the
  droppable set. "PASS" means "passes under the flags the run was invoked
  with."
- **Loud provenance.** Write a `validate.dropped.tsv` manifest (one row per
  distinct `id`, `check`, `round`); `validate.log` keeps the *initial-input*
  Findings while the manifest records every drop including later-round spawned
  offenders. Log `dropped N individual(s) / R row(s) of M (X%) over K rounds;
  cleared P reference(s)`; `WARNING` when the **rows-removed fraction** exceeds
  10% (rows, not distinct ids, so `duplicate_ids` removing many rows counts);
  exit `1` whenever anything was dropped (`0` only if `--drop-offending` was
  given but nothing needed dropping) so a reduced pedigree never masquerades as
  clean input.

## Considered alternatives

- **Cascade-drop descendants instead of clear-refs.** Removing an offender and
  *all* its descendants yields a true closed sub-pedigree (every retained
  individual keeps a real parent). Rejected: it deletes non-offenders (the
  descendants did nothing wrong, contradicting "drop offending individuals"),
  one bad ancestor can delete an entire lineage, and the only benefit over
  clear-refs — every retained individual keeps an input parent — is marginal
  because the relationships severed by clear-refs ran *through* an individual
  whose role was already contradictory. **Half-Founder** is an honest
  representation of "parent removed", so clear-refs is information loss, not
  falsification, and is made visible by the manifest.
- **Per-Check bespoke drop targets** (e.g. keep the first duplicate, repair the
  `self_loops` self-edge). Rejected in favour of the uniform "drop the flagged
  id" rule: bespoke targets reintroduce guessing (which duplicate row is real?),
  whereas dropping every flagged id never guesses. Consequence: `duplicate_ids`
  drops all copies — conservative toward correctness. The single principled
  exception is `acyclic` (cycle SCC members only), because its Findings
  over-report descendants; dropping those too would silently cascade, which the
  clear-refs decision explicitly rejects.
- **Single pass, no fixpoint.** Rejected: a drop can spawn new Findings (drop a
  child → its unsexed parent loses its only role → new `unknown_sex`), so a
  single pass can emit a file that does not actually pass.

## Consequences

- With `--drop-offending`, `validate.tsv.gz` is a **Reduced Pedigree** — a
  *different* pedigree from the input. Relatedness, `F`, every **Ne**
  estimator, and **Founder** counts computed on it differ from the input; the
  manifest, summary log, >10% warning, and non-zero exit make the reduction
  impossible to mistake for a clean input.
- Without the flag, `validate` is unchanged (still `BLOCK`s where it did).
- The droppable/blocking split and the clear-refs semantics are user-facing and
  costly to change once pipelines consume the Reduced Pedigree; that is why
  they are recorded here.
