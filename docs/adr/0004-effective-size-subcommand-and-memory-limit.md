# ADR 0004: Effective population size moves to its own subcommand, behind a memory limit

Status: accepted (2026-10-02). Ships in pedsum 0.15.0. Builds on [ADR 0001](0001-collaborator-cli-redesign.md)'s hard-break precedent.

Revised 2026-10-03, before 0.15.0 shipped: pedigree-graph 0.12.1 took the kinship DP out of Ne_C and Ne_GC (pedigree-graph#38), so §1's default became all eight and §4's two steps became one. The Context below is why this ADR exists and stays as written; earlier wording of §1 and §4 is in git history.

## Context

On the 783,029-row horse pedigree, `summarize` ran for 30 minutes and was then OOM-killed in the Ne step. It wrote nothing. Ne_GC grew past 22 GB, and the kernel's global OOM kill also took down the terminal that launched pedsum. Two estimators, Ne_C and Ne_GC, run pedigree-graph's kinship DP. Its peak depends on how long ancestors stay live, so it can't be predicted from N, and the native call can't be interrupted part-way through. The other six estimators take under a second on the same pedigree. Every other `summarize` phase ran within 1.1 GB.

## Decision

1. **`pedsum effective-size` is the only command that computes Ne.** `summarize` no longer computes any Ne, and `summary.yaml` loses its `popgen` category.
   - The new command writes one `effective_size.yaml` holding scalars and arrays together.
   - `--estimators NAME[,NAME...]` or `--estimators all` picks the estimators. The default is all eight: on pedigree-graph 0.12.1 they take 1.11 s on the horse pedigree, and the whole command peaks at 394 MiB.
   - All eight keys are always emitted. An unselected estimator reports `reason: not_requested`.
   - The command is named for the glossary term. A bare `ne` is forbidden by CONTEXT.md.
2. **The removed `summarize` flags fail loudly.** `--effective-size`, `--no-effective-size` and `--ne-coancestry` exit 2 with a message naming `pedsum effective-size`. There is no deprecation release, per ADR 0001.
3. **Every subcommand runs under a best-effort memory limit.**
   - A watchdog thread compares the process RSS with a limit about once a second.
   - The default limit is 80% of the smallest headroom at start: host `MemAvailable`, and `memory.max − memory.current` for the process's cgroup and each ancestor up to the root, since cgroup v2 limits are hierarchical. `--max-memory SIZE` overrides it, and `--max-memory 0` disables it.
   - When the limit is crossed, pedsum logs the running phase and its RSS, then exits 3.
   - This is protection, not a guarantee. Native allocation continues between samples and while the watchdog writes partial output, so a fast enough allocation can still reach the kernel OOM killer. The 20% margin of the default limit is what absorbs that overshoot.
   - The limit is a safety net. It never changes a result, so it is not the size-tiered behaviour that ADR 0001 rejected.
4. **A limit stop in `effective-size` still writes its output.**
   - The requested estimators run as one step, and their results replace the placeholders in one assignment, so a stop sees all of them or none.
   - Exactly one writer publishes `effective_size.yaml`, through a lock and an atomic replace. Whichever thread takes the lock first decides the outcome:
     - If the main thread wins, it publishes `status: complete` and disarms the watchdog.
     - If the watchdog wins, it publishes `status: stopped_memory_limit` and exits 3. The requested estimators report `reason: memory_limit`.
     - A reader never sees a truncated file.

## Considered options

- **Keep the six cheap estimators in `summarize` and move only Ne_C and Ne_GC.** Nothing in `summary.yaml` would break, but two commands would emit the same estimators. A Ne failure would also still cost the whole `summarize` output.
- **Leave `--effective-size` in `summarize` and also add the subcommand.** This is the smallest change, but `--ne-coancestry` would still reach the DP from `summarize`.
- **Run the DP estimators in a child process.** The parent would survive the child's death and record the failure with exit 0. This is the most robust option, but the child re-reads and rebuilds the graph (about 6 s on horse), and spawning a child adds complexity. The watchdog and partial YAML recover the same results with less machinery. Neither keeps the kernel OOM killer away for certain without a kernel-enforced limit.
- **Rely on the kernel: `RLIMIT_AS` or a `systemd-run` scope.** `RLIMIT_AS` caps virtual size, which overstates use: Ne_GC had 17.9 GB virtual at 12.5 GB RSS. A `systemd-run` scope is Linux- and systemd-only, so pedsum can only recommend it, not enforce it.
- **Leave the limit off by default.** The default run, which is what a collaborator types, would keep the terminal-killing failure mode.
- **Predict DP memory before running it.** pedigree-graph exposes no estimate, and the peak follows the shape of the live frontier, not N (pedigree-graph LIMITATIONS.md).

## Consequences

- **Breaking change.**
  - Version 0.15.0.
  - `summary.yaml`, `summary.extra.yaml` and `summary.pedigree.tsv` lose their effective-size content.
  - The unreleased 0.14.x change that gated Ne_GC behind `--ne-coancestry` is superseded before it ships.
- **F is computed twice** when a user runs both commands. That takes 0.74 s on horse, and minutes at 10M+ rows.
- **New exit code 3** means pedsum stopped itself at the memory limit. Codes 1 (validation) and 2 (usage or file errors) keep their meanings.
- **The watchdog reads `/proc`.** Where `/proc` is missing, pedsum logs that the limit is inactive and runs without it.
- **Progress reporting is out of scope.** It waits on pedigree-graph's `progress=` argument (pedigree-graph#37).
