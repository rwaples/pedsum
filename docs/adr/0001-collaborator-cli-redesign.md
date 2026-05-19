# 0001 — Collaborator-friendly CLI redesign for pedsum 0.7

Status: accepted (2026-05-19). Supersedes the internal-tidy plan that previously lived in `PLAN.md`.

## Context

Pedsum's intended audience is collaborators handed a human pedigree TSV — researchers, not pipeline authors. After the 0.4 / 0.5 / 0.6 rounds of internals work (see `STATUS.md`), the remaining sharp edge is the CLI surface: too many flags, defaults that hide useful output, an opaque output-file naming convention, and engine knobs that leak implementation details.

Pedsum currently has no programmatic consumers (neither simACE nor fitACE invoke it), so the cost of a hard break is low.

## Decision

Cut a breaking 0.7 release that reshapes the CLI around collaborator mental models:

| Area | Change |
|---|---|
| Defaults | `--inbreeding` and `--effective-size` flip to opt-out (`--no-inbreeding`, `--no-effective-size`). `--ne-coancestry` stays opt-in (RAM-hungry). |
| Cost transparency | Before the F kernel runs on large inputs (N > 1,000,000), log a single INFO line: "computing F on N=… rows; may take several minutes — pass --no-inbreeding to skip." No estimate formula; size threshold only. |
| Pair-counting | Collapse `--burden` / `--engine` / `--bfs-threshold` → single `--per-individual-pairs`. Matrix engine only inside pedsum. BFS stays available via `pedigree_graph.experimental.count_pairs_bfs` for callers reaching the package directly. |
| `--out` | Directory, not basename. Files inside get fixed names. |
| Output footprint | Always written: `summary.yaml`, `summary.extra.yaml`, `annotated.tsv.gz`. `--tsv` opt-in adds `summary.pedigree.tsv` + `summary.individual.tsv` (the `summary.` prefix avoids the ambiguity of bare `pedigree.tsv` next to an actual input pedigree). `--single-file` deleted (two-YAML split is unconditional). |
| Sex flags | Delete `--zero-as-missing` (niche; collaborators with 0/1-only columns where 0 means missing preprocess instead). Keep `--plink-sex` legacy alias, `--sex-encoding`, `--sex-col`, `--allow-unknown-sex`. |

Output schemas (YAML keys, TSV columns, validation log format) do **not** change. The internal `relationship_burden` section keeps that name even though the CLI flag is renamed.

## Rejected alternatives

- **Size-tiered defaults** (auto-compute F + Ne below a threshold). Hidden invariant; "the tool behaves differently at size X" is itself a usability bug.
- **Keep `--engine` speculatively for future BFS performance.** BFS has never been demonstrated to beat matrix at any pedigree size pedsum cares about; the numba kernel was killed at ~102% CPU on a 2M-row pedigree (STATUS.md open follow-up). Add the flag back if/when BFS wins on real data.
- **`--single-file` toggle.** Adds CLI surface area for a layout decision that should be unconditional.
- **Deprecation cycle before deletion.** Pedsum has no programmatic consumers and no public users; a deprecation release would cost a cycle without buying anything.
- **Delete `--plink-sex`.** Legacy alias, but PLINK is the dominant tool in the audience's ecosystem and `--plink-sex` reads naturally to that audience. Keep until a real motivation to drop it surfaces.

## Consequences

- **Breaking change.** Version bumps to 0.7. New `CHANGELOG.md` records every deleted flag, rename, and the `--out` semantics change.
- **README + STATUS rewrite.** Every example command updates; STATUS gets a "Resolved in pedsum 0.7" section.
- **Forward note on downstream consumers.** If simACE or fitACE later wire pedsum into snakemake rules, those rules use the new CLI. No migration work needed today because no programmatic consumer exists.
- The original `PLAN.md`'s structural-split work (split `_run_summarize`, `validate_pedigree`, `_parse_args` into helpers) is **re-evaluated after** the CLI redesign. Flag deletions and default flips will shrink `_run_summarize` substantially on their own; the original Phase 2 may be largely moot.
