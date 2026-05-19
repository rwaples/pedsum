# Plan: pedsum 0.7 collaborator-friendly CLI redesign

## Context

Pedsum's audience is collaborators handed a human pedigree TSV — researchers, not pipeline authors. After the 0.4 / 0.5 / 0.6 rounds of internals work captured in `STATUS.md`, the remaining sharp edges are CLI-facing: too many flags, defaults that hide the most useful output, opaque output filenames, and engine knobs that leak implementation details.

Rationale and rejected alternatives (size-tiered defaults, keeping `--engine` speculatively, the `--single-file` toggle, a deprecation cycle, deleting `--plink-sex`) live in [docs/adr/0001-collaborator-cli-redesign.md](docs/adr/0001-collaborator-cli-redesign.md). This document is the *how*: the concrete change list, the load-bearing call sites, and verification.

No programmatic consumers exist today — `rg pedigree_summary` returned empty across simACE and fitACE on 2026-05-19 — so this lands as a hard-break 0.7 release with no deprecation cycle. Output schemas (YAML keys, TSV columns, validate-log format) are explicitly preserved; only the CLI surface and on-disk file names change.

## The redesign

**Flags deleted.** Remove `--engine`, `--bfs-threshold`, `--zero-as-missing`, and `--single-file` from the argparse surface. Pass `allow_abbrev=False` on the parser so a deleted long-option cannot be silently abbreviated into something else.

**Flag renamed.** `--burden` → `--per-individual-pairs`. The internal `relationship_burden` output-schema section keeps that name; only the CLI flag changes.

**Defaults flipped to opt-out.** `--inbreeding` and `--effective-size` are on by default; new `--no-inbreeding` and `--no-effective-size` skip them. `--ne-coancestry` stays opt-in (it's the RAM-hungry estimator). Invert the flag-combination warning at `pedigree_summary.py:3379-3382` to match.

**`--out` becomes a directory.** `mkdir -p` the path; if it already exists and is not a directory, exit rc=1 with a single-line error. The output-write call sites at `pedigree_summary.py:3494-3525` rewrite from `_at(base, ".summary.yaml")` to `DIR / "summary.yaml"`, etc. The `--single-file` branch at `pedigree_summary.py:3502-3518` collapses into the unconditional two-YAML write.

**Default output footprint inside `DIR/`.** Three files always written:

```
DIR/
├── summary.yaml
├── summary.extra.yaml
└── annotated.tsv.gz
```

`--tsv` opt-in additionally writes `summary.pedigree.tsv` and `summary.individual.tsv`. The `summary.` prefix is deliberate: bare names like `pedigree.tsv` next to an input pedigree TSV were ambiguous.

**Cost-estimate INFO line.** Before the F kernel at `pedigree_summary.py:3439` runs on inputs with N > 1,000,000, log:

```
computing F on N={n:,} rows; may take several minutes — pass --no-inbreeding to skip
```

Size threshold only; no formula. This is the safety valve for the opt-out default on large pedigrees.

## Tests and docs

**Tests** in `external/pedsum/tests/`: migrate every existing `summarize` / `validate` invocation to directory `--out`. Add three opt-out-default cases (bare `summarize` populates the inbreeding section; bare `summarize` populates the effective-size section; `--no-inbreeding --no-effective-size` omits both). Add one test that `--per-individual-pairs` produces the `relationship_burden` YAML section. Add one test per deleted flag asserting argparse rc=2 — `allow_abbrev=False` ensures `--eng` cannot resurrect `--engine`.

**Docs**: new `external/pedsum/CHANGELOG.md` with a 0.7 entry listing every deleted flag, the rename, the two default flips, the `--out` change, the new `--tsv` opt-in, and the cost-estimate INFO line. Rewrite `README.md` examples to the new CLI and add a "Migrating from 0.6" subsection at the top mapping old → new. Add a "Resolved in pedsum 0.7" section at the top of `STATUS.md` linking the ADR. Refresh the module docstring at the top of `pedigree_summary.py`.

## Re-evaluate internal cleanup

The original `PLAN.md`'s Phase 2 (extract helpers from `_run_summarize`, `_parse_args`, `validate_pedigree`) is parked until after the CLI changes land. Flag deletions and default-flip simplification will shrink `_run_summarize` substantially; if both `_run_summarize` and `_parse_args` are under 120 lines post-Wave 1, the structural-split is moot and is skipped.

Verified dead-code notes carried over:

- `_id_list` at `pedigree_summary.py:184` — likely dead; confirm with `rg _id_list` before deleting.
- `_parent_parser` nested in `validate_pedigree` at `pedigree_summary.py:1080`, called at `:1084-1085` — **alive, do not delete** despite the prior plan's "dead-code candidate" label.
- `_positive_int` at `pedigree_summary.py:3143` — **alive**, used as argparse `type=`; do not delete.

## Out of scope for 0.7

Splitting `pedigree_summary.py` into multiple modules; reworking output schemas; performance work without profiling data; reworking relationship-pair semantics or pedigree-graph delegation; resurrecting the BFS engine inside pedsum (add it back when there is real data showing it beats matrix). If simACE or fitACE later wire pedsum into snakemake rules, those rules use the new CLI.

## Verification

The redesign is done when these all hold:

```bash
# Default footprint: 3 files in directory, F + cheap Ne computed.
python pedigree_summary.py summarize \
  --in example_pedigree.tsv --out /tmp/pedsum-07-smoke
ls /tmp/pedsum-07-smoke/
# → summary.yaml summary.extra.yaml annotated.tsv.gz

# Full opt-in: 5 files, both expensive computations off.
python pedigree_summary.py summarize \
  --in example_pedigree.tsv --out /tmp/pedsum-07-smoke-full \
  --tsv --no-inbreeding --no-effective-size
ls /tmp/pedsum-07-smoke-full/
# → summary.yaml summary.extra.yaml annotated.tsv.gz
#   summary.pedigree.tsv summary.individual.tsv

# Validate writes into the directory.
python pedigree_summary.py validate \
  --in example_pedigree.tsv --out /tmp/pedsum-07-smoke-validate
ls /tmp/pedsum-07-smoke-validate/
# → validate.tsv.gz validate.log

# Deleted flags absent from --help.
python pedigree_summary.py summarize --help \
  | grep -E -- '--engine|--bfs-threshold|--zero-as-missing|--single-file|--burden\b' \
  | wc -l
# → 0

pytest -q
ruff check pedigree_summary.py tests
```

Both `pytest -q` and `ruff check` must be green; the README, STATUS, CHANGELOG, and module docstring must reflect the new CLI; the repo is tagged `v0.7.0`.

If a regression surfaces after release: `git revert` the merge, tag `v0.7.1`. No data-migration concerns — output schemas are unchanged and no programmatic consumers exist today.
