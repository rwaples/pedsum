# 0002 — Validation as a declarative Check registry

Status: accepted

## Context

Validation lived in two functions in `pedsum/validate.py` that ran the same
**Checks** in the same order but diverged in surrounding work: `load_and_validate`
(used by `summarize`) raises on the first failure, then imputes sex, topo-reorders
rows, and returns a dataframe; `validate_pedigree` (used by `validate`) records
every Check's **Check Status** and accumulates **Findings**, hand-coding ~170 lines
of prerequisite gating (`if ids is not None` / `can_index` / `_skip_many(...)`) so
later Checks skip when an earlier parse failed. The two sequences had to be kept in
sync by hand, and the gating ladder was the messiest code in the package.

## Decision

Express the Checks as a single declarative registry and drive both modes from one
runner:

- **One `CHECKS` list** of `Check(name, requires, run, label, group)` entries, in
  canonical order. `_CHECK_ORDER`, `_CHECK_LABELS`, and `_CHECK_GROUPS` are
  *derived* from it (single source of truth; the three hand-synced tables are
  deleted).
- **One runner** parametrized by `on_fail`: `raise` (fail-fast, for `summarize`)
  or `accumulate` (for `validate`). A Check whose prerequisite did not `PASS` is
  auto-`SKIP`ped with `skip_reason = f"{first_failed_prereq} failed"`.
  `required_columns` is the universal root prerequisite.
- **A mutable `ValidationContext`** is threaded to every `run(ctx)`. The five
  parse Checks (`id_dtype`, `mother_dtype`, `father_dtype`, `sex_tokens`,
  `birth_year_dtype`) and only those populate `ctx` (parsing *is* the Check);
  every other Check reads from it. Sex imputation is a `@cached_property` on the
  context — shared by the three sex Checks, computed once, never a reported Check.
- **Each `run(ctx)` returns a rich `CheckOutcome`** (`status` + `findings` +
  `count` + optional `skip_reason`) so flag-dependent annotations
  (`PASS (N overridden from role)`, `SKIP (N tolerated)`) live inside the Check
  rather than as runner special-cases. The underlying `_check_*` helpers stay
  pure (`list[Finding]`).
- `load_and_validate` = `run(raise)` then its post-steps (impute / reorder /
  `children_csr` / dataframe). `validate_pedigree` = `run(accumulate)` then its
  ctx return. Both return shapes are otherwise preserved; the `validate`-mode
  `ctx` is now the `ValidationContext` (was an ad-hoc dict — only `cli.py` read
  it).

## Considered alternatives

- **Registry for the accumulate path only** — kills the gating ladder but leaves
  two Check sequences that can still drift. Rejected: the drift risk is half the
  motivation.
- **Bare `list[Finding]` from each Check, cosmetics as runner special-cases** —
  recreates name-keyed branches in the runner, partially rebuilding the spaghetti
  being removed. Rejected in favour of the rich `CheckOutcome`.
- **Don't merge; just extract helpers** — rejected: leaves the two-sequence
  duplication intact.

## Consequences

- Behavior-preserving by intent and verified against a captured golden baseline
  (`summarize`/`validate` outputs bit-identical modulo timestamp) plus the full
  test suite. The pinned strings `required_columns failed` and `sex_tokens failed`
  fall out of the uniform skip rule.
- **One intentional normalization:** fail-fast (`summarize`) error messages for
  parse failures now carry the `"<check>: N finding(s) — "` prefix from
  `_summarize_findings`, where they previously propagated the raw parser
  exception. All existing error-message assertions match on substrings (check
  names / `unknown sex encoding`) that survive, so no test breaks; messages for
  malformed inputs read slightly differently.
- Non-pinned `SKIP` reasons for malformed inputs change to the uniform
  `"<prereq> failed"` form (e.g. `acyclic`'s former friendly `"missing mother
  references"`). These appear only in `validate.log` / stderr for invalid
  pedigrees and are not asserted anywhere.
