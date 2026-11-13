# DESIGN.md — pedsum internals (maintainer notes)

Pointers to where implementation details live. The README is the
collaborator-facing surface; this file is for maintainers.

## Package layout

The implementation is the `pedsum/` package. The installed `pedsum` command
and `python -m pedsum` (`pedsum/__main__.py`) both call `pedsum.cli:main`.
`pedigree_summary.py` is a thin runnable shim (`python pedigree_summary.py …`)
for an uninstalled checkout; it re-exports nothing, and tests import from
`pedsum.*`. Modules, in dependency order (each imports only
from those above it):

| Module | Responsibility |
|---|---|
| `pedsum/base.py` | shared constants, the `pedigree_summary` logger, `PedigreeError` |
| `pedsum/progress.py` | `timed` phase logging (and the phase stack the memory watchdog and the benchmark profiler read); `relationship_progress`, the terminal progress bar for pedigree-graph's long relationship calls |
| `pedsum/pedigree_ops.py` | low-level array helpers (parent rows, parent references, sib groups, topological depth, sort-based distinct integers) |
| `pedsum/parse.py` | delimiter sniffing, column coercion, sex decoding, trait-column reading |
| `pedsum/checks.py` | the Check registry: each Check as one `(ctx) -> CheckResult` function, its Fail Policy, `blocking_failures`, and the constants derived from the registry |
| `pedsum/validate.py` | `ValidationConfig`, `ValidationContext`, sex imputation, the runner, `load_and_validate` (fail-fast) / `validate_pedigree` (accumulating) |
| `pedsum/fixed_pedigree.py` | the Fixed Pedigree: founder synthesis, phantom parents, resolved sex, topological order (`fix_pedigree`), and the string frame `validate` writes (`render_fixed_pedigree`) |
| `pedsum/reduce.py` | `fix_validated` (what `validate` writes: BLOCKED, or the Fixed Pedigree, after the `--drop-offending` reduction and its self-verify) and `reduce_pedigree` |
| `pedsum/disclosure.py` | the `--safe-attempt` engine: declared tables merged or blanked at the minimum cell, statistics withheld with their source table, and `build_release`, which copies only the leaves a `ReleaseSpec` lists |
| `pedsum/safe_summary.py` | `summarize --safe-attempt`: the integer source payload, its release list and links (`summarize_spec`), and the layout of the release into the output files |
| `pedsum/pairs.py` | relationship-pair enumeration + `PedigreeGraph` construction |
| `pedsum/sections.py` | per-section summary computations |
| `pedsum/sex_concordance.py` | Offspring Sex Concordance: group projection, exact conditional moments, Holm, permutation samplers |
| `pedsum/assortative_mating.py` | Mate Correlation: trait typing, row strata, the call to pg-phenotype's `mate_correlation` and the YAML payload built from its result |
| `pedsum/schema.py` | categorised YAML schema + slim/extra split machinery |
| `pedsum/memory.py` | the best-effort memory limit: `MemoryWatchdog`, default-limit discovery, `--max-memory` parsing |
| `pedsum/safe_effective_size.py`, `pedsum/safe_assortative_mating.py` | the same for `effective-size` and `assortative-mating`: source payload, release list, layout |
| `pedsum/safe_log.py` | the `--safe-attempt` log record factory: arguments printed by type, the rest withheld, and the `--id-log` file |
| `pedsum/report.py` | report payload builders, output writers, `atomic_output`, and the safe file names (`SAFE_SUFFIX`, `output_name`) with the `--out` rule (`prepare_out_dir(only=...)`) |
| `pedsum/cli/` | `parser.py` (argparse), `common.py` (logging, thread budget, `load_or_fail`), one runner module per subcommand, and `main` in `__init__.py` |

## CLI design rationale

See [docs/adr/0001-collaborator-cli-redesign.md](docs/adr/0001-collaborator-cli-redesign.md).

## Output schema

Output schema follows [GLOSSARY.md](GLOSSARY.md) as of 0.10. The glossary fixes section names, key names, and the convention for distribution naming (`<noun>_count` / `<noun>_count_hist` / `<noun>_count_<sex>`). When extending the output, add the term to GLOSSARY.md first, then emit keys that match it.

## Flag and behavior version history

See [CHANGELOG.md](CHANGELOG.md). Notable:
- `assortative-mating` reports the Mate Correlation (0.15; pedsum#13).
- Effective population size moves from `summarize` to `effective-size`,
  and every command runs under a memory limit (0.15; see
  [docs/adr/0004](docs/adr/0004-effective-size-subcommand-and-memory-limit.md)).
- `validate --drop-offending` emits a **Reduced Pedigree** (0.11; see
  [docs/adr/0003](docs/adr/0003-drop-offending-reduction.md)). `--no-sex-check`
  now lives in the registry (`ValidationConfig.no_sex_check`) so the tolerance
  composes inside `run_checks` rather than as a cli post-filter.
- `--allow-unknown-sex` → `--allow-missing-sex` (0.8)
- `ped_depth` sourced from `PedigreeGraph.depth` (0.4; the attribute was
  named `generation` before pedigree-graph 0.8)

## Engine selection & semantics

Pedsum uses two pair-counting paths, picked by `--per-individual-burden`
(alias `--per-individual-pairs`; no engine auto-tiering — per ADR 0001, the
matrix/BFS dispatch was removed). Both delegate to `pedigree-graph` and run
its Rust row-streaming engine, so neither builds a pair list:

- Default: `PedigreeGraph.relationship_counts(max_degree=N)`, with `N`
  from `--max-degree` (default 5; `_engine` reported as `rust_streaming`). The engine classifies every pair one row
  at a time, so the 23 counts are exact in O(N) memory; aggregate counts
  only.
- `--per-individual-burden`: `PedigreeGraph.relationship_burden()` (`_engine`
  reported as `rust_streaming_burden`). Besides the 23 category counts it
  returns each person's relative counts by degree (1-5) and the related-pair
  count within each depth, in O(N) output storage.
  `compute_relationship_summary_from_burden` (`pedsum/sections.py`) turns
  those arrays into the per-individual relationship-burden summary. It reports the same 23 counts as the default.
  Its parity oracle, a fold over explicit pair lists, lives in
  `tests/relationship_summary_oracle.py`.
- Both paths assign each pair its single closest relationship category, so
  the 23 counts partition the related pairs. A pair that is both
  parent-offspring and half sib (parent-offspring incest) counts only as
  parent-offspring. The YAML `pairs_engine` field records which path
  produced each summary.
- The experimental BFS enumerator (matrix counts *paths* / multiplicity;
  BFS counts *distinct shared ancestors*, disagreeing on `1C1R`, `H1C1R`,
  `1C2R`, `2C`) is no longer reachable from pedsum. It remains available
  to direct callers via `pedigree_graph.experimental.count_pairs_bfs`.
  Open upstream issues:
  [pedigree-graph#2](https://github.com/rwaples/pedigree-graph/issues/2),
  [pedigree-graph#3](https://github.com/rwaples/pedigree-graph/issues/3).

## Performance thresholds

- F kernel (Meuwissen-Luo): logs a WARNING when rows times the largest
  possible ancestor set, `min(2^(depth+1) - 2, rows)`, passes 4e9, so
  large or deep runs don't silently hang. See `F_WALK_WARN_VISITS`
  (`pedsum/base.py`) and its use in `run_summarize` (`pedsum/cli/summarize.py`).
- Descendant path counts run right after the graph build: they cost one
  pass, and an int64 overflow on a deep pedigree then exits 1 before any
  expensive phase.
- `--per-individual-burden`: O(N) output storage like the default; it
  builds no pair list.
- `epimight-input --pairs`: the one path that still materialises pairs,
  because `relative_pairs.tsv` lists them. It requests only the categories
  the `--rels` kinds read, with `execution="memory"`, and writes one kind
  at a time: beyond the engine's pair blocks it holds one kind's frame,
  and it never sorts the whole list.
- `effective-size`: all eight estimators run in memory linear in N since
  pedigree-graph 0.12.1, which took the kinship DP out of Ne_C and Ne_GC
  (its issue #38; Ne_GC had passed 12 GiB within 100 s on the 783K-row
  horse pedigree). The default selection runs all eight there in 1.11 s.
- `summarize --max-degree`: `relationship_counts` wall time on the horse
  pedigree at 10 threads was 0.61 s at degree 1, 1.36 s at 2, 221 s at 3,
  632 s at 4 and 1,833 s at 5. Codes past the cutoff are null.
- Memory limit (ADR 0004 §3): every command runs under `MemoryWatchdog`,
  which samples `/proc/self/statm` once a second and exits 3 past the
  limit (default 80% of the smallest of `MemAvailable` and each enclosing
  cgroup's headroom). It runs breach callbacks first; `effective-size`
  uses one to publish its finished estimators. Final publication and the
  breach share a lock through `MemoryWatchdog.disarm()`. Every writer
  publishes through `atomic_output`, so a stop leaves at most a
  `.<name>.partial-<pid>` file, never a truncated output.

## Assortative mating

`assortative-mating` reports the Mate Correlation over Mating Pairs; the
README covers its use and output. The computation is pg-phenotype's
`pg_phenotype.assortative.mate_correlation`. The method was designed here
in pedsum#13 and ported to pg-phenotype 0.1 unchanged. Its
[design notes](https://github.com/rwaples/pg-phenotype/blob/v0.3.0/docs/assortative-mating-design.md)
give the reasons for every choice (the Mate Network as cluster, the
two-step sandwich, the one-step bootstrap, the permutation null and its
sequential stopping, the threshold fits, thin strata, thread
independence, calibration), and pg-phenotype's
[ADR 0005](https://github.com/rwaples/pg-phenotype/blob/v0.3.0/docs/adr/0005-assortative-mating-reproduces-pedsums-numerics.md)
records how closely the port reproduces pedsum's numbers: counts, flags
and p-values exactly, estimates to the parity tests' 1e-12 (relative) for
closed-form estimators and 1e-7 for latent fits. Three kinds of fit
differ, each on rounding noise in a near-empty cell: pg-phenotype reports
an SE that pedsum withheld, and it fixes two pedsum errors, a
tetrachoric or polychoric Newton that stopped short of the maximum and a
`boundary` flag at an interior maximum. All three are in the CHANGELOG.

pedsum keeps the shell. `pedsum/assortative_mating.py`:

- types each trait column (`classify_trait`) and codes binary and
  ordinal levels `0..k-1`;
- labels each row's stratum (`strata`: its Depth, or the first year of
  its birth-year bin, `-1` unknown), passed to pg-phenotype as NaN;
- maps the result onto the YAML payload: the trait records, `settings`,
  the `inference` block (pg-phenotype's `MateCorrelation.method`, written
  unchanged so the description cannot drift from the code), the counts,
  the cells, the Within-Person Cross-Trait Correlation, then `NOTES`.

**Threads.** `pedsum.cli.assortative_mating._set_assortative_threads` hands `--threads` to numba,
which caps it at `NUMBA_NUM_THREADS`, then passes numba's count to
`pg_phenotype.configure_threads`. pg-phenotype commits one budget per
process, as pedigree-graph does, so a later run in the same process (the
test suite drives several) keeps the first budget, and `settings.threads`
reports the budget the run used.

**Refusals.** pg-phenotype's errors become `PedigreeError`. The CLI never
reaches them: `classify_trait` refuses an all-missing or constant column,
and the parser takes one or two traits. A direct caller of
`compute_assortative_mating` can, with a constant trait or three traits.

**Tests.** `tests/test_assortative_mating_payload.py` pins the YAML of six
small pedigrees byte for byte. `tests/test_assortative_calibration.py`
(marked `slow`) checks size and coverage through
`compute_assortative_mating`; a three-trait design runs as its three
trait pairs. pg-phenotype carries the estimator, sandwich, bootstrap and
permutation tests, and its golden gate against pedsum#13.

## Disclosure engine (`--safe-attempt`)

Plan and decisions: simACE `plans/pedsum-safe-attempt-dst.md` (D1-D13).
User-facing rules: README "Safe output".

**Release list, not redaction.** Each safe command builds a *source*
payload of integer counts at full resolution and declares a `ReleaseSpec`
(`summarize_spec`, `effective_size_spec`, `assortative_mating_spec`): one
rule per leaf, each with a reason string. `disclosure.build_release`
copies the listed leaves and nothing else, and the command lays its files
out from that release. The ordinary payload is never read, so a key that
pedsum or pedigree-graph adds later is absent until a rule names it. The
earlier design (`redaction.py`, a `redact` hook per section) nulled known
cells of the ordinary payload and missed shares, complements and scalar
extrema; it is deleted.

**Rules.** `Plain` (run metadata), `Count`, `Table` (1-D; an ordered one
merges small end bins before it is blanked), `Contingency` (blanked by any
small nonzero cell, which covers a zero in a small row or column, since
that row or column holds one), `Continuous`
(mean and sd at `n >= min_cell`, quartiles at `n >= 4 * min_cell`, never
min or max) and `Derived`. `Count`, `Table` and `Contingency` take a
`support` path: the number of distinct units behind each cell, which is
what the minimum cell is tested against. A rule whose support the payload
lacks is withheld, never gated on its raw count. `Table` and `Contingency`
take `requires`: finer tables whose totals they repeat, without which
they are withheld down to their total. A leaf the payload holds as `None`
is released as `null`, as a withheld one is, so the key set of a release
follows the options and never the data.

**Taint.** A `Derived` rule names its `sources`. It is released only when
every source is `INTACT`. A statistic of a merged table is withheld too:
`n`, the mean and the sd of an integer variable fix the sum and the sum of
squares, and those can leave one possible tail behind a merged bin. For
the same reason a per-depth or per-cohort Ne array is one leaf with its
scalar, released whole or not at all.

**Links and the oracle.** The engine protects one table at a time. It
cannot see that two tables split the same population. A spec therefore
lists each fact once (a count or its fraction, not both; no complement, no
rollup) and declares what is still shared as a `Link`: the `whole` cell is
the sum of the `parts` cells. The engine does not act on links.
`tests/disclosure_oracle.py` does: for each spec it builds the integer
system of the released cells, the links and the moment equations of each
released statistic at serialised precision, and checks with scipy `milp`
that every withheld cell has at least two integer solutions. An LP
relaxation is not enough, since it accepts systems that one integer
solution fits. The oracle covers the relations a spec declares; the
release list and the one-copy rule are what keep the undeclared ones few.

**No founder-contribution statistic (D17).** Three statistics are
concentration indices over per-founder contributions:
`effective_founders_by_descendant_paths`, the mean and sd of equivalent
complete generations, and `ne_long_term_contributions`. Each narrows how
the descendants are spread over the founders, and no gate short of a
dominance rule closes that, so none is on a release list and a safe run
computes none of them. The estimator's record is
`reason: not_released_in_safe_mode` whatever the run, so the key set of
`effective_size.txt` does not depend on the data.

**Pair counts, not histograms (D11).** A relationship pair count is a
weighted sum over a per-individual table: `FS` is the sum of `k(k-1)/2`
over sibships of size `k`, and `MO + FO` is the total offspring count. A
`Link` holds plain sums, not weighted ones, so the oracle could not prove
a merged tail of a sibship-size or relatives-per-person histogram safe
beside the pair counts. One of the two had to go. The pair counts stay;
every per-individual relative-count histogram and sibship-size table is
off the list. The same-depth related-pairs table is also off (D13): it
would need a pair pass per depth.

**Pair support (D9, D12).** DST's rules protect "persons, households,
families" and do not say whether a pair is an observation, so a pair count
is not tested against the minimum cell directly. `safe_summary.pair_counts`
reads each code's pairs and its support from one pass of
`PedigreeGraph.relatives_per_person`, whose memory is rows x codes with no
pair list. The support is the smaller of two counts, so pairs that all
hang on one person or one couple have a support of 1 from either side.

The first is the distinct families among the individuals credited with a
relative of that code, a family being the individuals sharing a recorded
(mother, father). A directional code credits the junior side only, which
errs toward withholding.

The second is the distinct ancestors the pairs go through, `up`
generations above the credited individual (`RELATIONSHIPS[code].up`):

| Codes | Unit |
|---|---|
| `MHS`; `PHS` | distinct shared mothers; distinct shared fathers |
| `FS`, `MZ` | the smaller of the distinct mothers and the distinct fathers |
| lineal (`GP`, `GGP`, `GGGP`, `G3GP`) | distinct ancestors at that distance, the seniors themselves; over by the few that inbreeding also places closer |
| through a couple (`Av`, `GAv`, `GGAv`, `G3Av`, `1C`, `1C1R`, `1C2R`, `2C`) | a lower bound: the fewest ancestors at that distance that could have every credited individual below them, the smaller of the bound over mothers and over fathers |
| through one ancestor (`HAv`, `HGAv`, `HGGAv`, `H1C`, `H1C1R`) | the same lower bound over ancestors of either sex |

The pass does not say which ancestor a cousin pair shares, so those rows
are bounds and can withhold a code that an exact count would release,
mostly in a small pedigree with few founders. In `assortative-mating` the
unit is the couple, backed by the smaller of its distinct mothers and
fathers. The individuals are also a gate table of three parts whose total
is the released `n_total`: the couples' mothers, their fathers and everyone
else. When the third part is below the minimum cell and not zero, the
counts that fix the first two are withheld: the parents with several
mates, the parents with both traits and each cell's Mate Networks. The
counts of couples stay, since without those three they only bound the
mothers and the fathers from above. Two more floors rest on the same
counts. The Mate Networks of a cell whose couples never remate are its
distinct mothers and its distinct fathers, so where someone remates
elsewhere they are withheld when the mothers, or the fathers, in no couple
of the cell are fewer than the minimum cell: in a stratified run those are
the parents of dropped thin-stratum couples alone. The Mate Networks of a
cell that holds every extra mate are the distinct parents of one sex, so
the parents of that sex with both traits take the cell's networks floor.

**Structural zeros and tables that wait (review S1, S2).** No individual
of the deepest depth has offspring, none of depth 0 or 1 is inbred, and
only depth 0 lacks both parents. A reader knows this, so each is declared
as a `Link` with no parts and the oracle fixes the cell at zero. With such
a row known, the offspring total less the founders with descendants is the
count of non-founders with offspring, a sum of depth-by-offspring cells.
The by-status and by-sex offspring tables therefore `require` the
depth-by-offspring table, and the inbred table, the F histogram and the
by-sex inbred table the depth-by-inbred table. The per-depth means of
ancestor and descendant-path counts are weighted sums of the
depth-by-recorded-parents row of depth 1 (in a two-depth pedigree, for the
founders' descendant paths), so they name that table as a source.

**Nested populations (review F1, F2; D18).** `summarize` releases no mean
or sd of F, per depth or over the inbred. F takes few distinct values, so
a mean beside a count gives how many individuals hold each value, and two
values can share a histogram bucket. The eligible offspring of the three
sex-concordance groupings are nested: a full sibship's are eligible in
both single-parent groupings, and each grouping's are among the
individuals with its parent recorded and among the individuals of their
sex in the `sex` table. The differences are gate tables
under `gates` in the source, which have rules and `Link`s so the oracle
sees them, and which `safe_outputs` writes to no file. The smallest
nonzero difference is the `support` of the grouping's group count and
by-sex table, so a difference of 1 to min_cell - 1 withholds both, the
by-sex total included. Group counts are not differenced: one mother's two
sibships are one maternal group, so the groups of one grouping are not a
subset of another's and their difference counts no set.

**F estimates of `effective-size` (review V1).** A depth's mean F times
its size is its F sum, and with two F values at the depth that sum gives
how many individuals hold each. The gate `inbred_f_by_depth` therefore
counts individuals by depth and by distinct F value (sorted values no
further apart than `INBRED_TOL` are one value), and mean F per depth,
`ne_group_coancestry` and the per-depth delta-F Ne are released only when
every value at every depth is held by none or at least min_cell. Bins of
the F histogram are not enough: two values share a bin. The scalar `ne`
of `ne_inbreeding` stays in the gated group on purpose: with one inbred
depth it is a function of that depth's mean F, so releasing it on counts
alone gives the mean back. Most real pedigrees fail this gate.

**Logs (D8).** `cli.main` runs a safe command inside
`safe_log.scrubbing()`, which installs a log record factory, so every
handler, including tqdm's redirect and pytest's capture, sees scrubbed
records only. A Rust panic is outside it: `cli.main` catches PyO3's
`PanicException`, but Rust's panic hook has by then written the message
and location to stderr unscrubbed, one more reason the log is never
transferred. On exit it puts back the factory and the
`warnings.showwarning` it found; an ordinary run changes neither. An
argument of a `pedigree_summary` record prints when its type says it
carries no ID and nothing drawn from the data rows: a `Public` string or
number, a path, an `OSError`, a `PedigreeError` with a `public` text.
Anything else prints as `[withheld]`, an unmarked number included, and
tracebacks are dropped. A log call that passes an ID or a count as an
argument is therefore withheld without being marked. `Public` marks a
number only when it is not computed from the rows: an option value, the
thread count, the minimum cell, elapsed seconds, a memory limit. No count
is marked, large or small: a printed total less its printed large parts is
the withheld small part (`21 - 10 - 10`). The row total is marked in one
line, the F-walk warning of `summarize`, whose `summary.txt` carries the
same `n_total`; the load lines withhold it, because a run can stop before
it writes a report.
A record of any other logger
(`pedigree_graph.*`, a library) prints as `[withheld]` whole, since nothing
checks that its message is a literal, and a Python warning prints its
category only. One thing can still get a value through, and a test covers
it: a message that is not a literal
(`test_every_log_message_in_the_package_is_a_literal`). The DEBUG-level
token tests in `tests/test_safe_log.py` run each safe command on IDs, birth
years and trait values that no other text can contain, and the trio test
there allows no digit in a safe log but the public ones.

**Files (D10).** `report.SAFE_SUFFIX` is the one map from an ordinary
extension to a safe one, and `cli.common.prepare_outputs` is the one place
a runner gets its output paths, so a safe run cannot name a `.yaml`. The
same call applies the `--out` rule and opens the `--id-log` file.

## Upstream integration

Both pair-counting engines delegate to `pedigree-graph`; bug fixes
propagate on `pip install -U`. Sparse/non-contiguous IDs are still
compacted internally to dense `0..n-1`, though since pedigree-graph 0.8
that is for pedsum's own convenience — a compacted id and the graph row
it names are the same number — rather than a memory necessity: the
library indexes its own ids densely and accepts rows in any order.

## Stance: opt-outs vs auto-tiering

Size-tiered behaviors stay as flags the user types (`--no-inbreeding`,
`--max-degree`, `--per-individual-pairs`, `effective-size --estimators`),
not as auto-tiered defaults. The memory limit is not one of these: it
never changes a result, only stops a run that would otherwise be
OOM-killed ([ADR 0004](docs/adr/0004-effective-size-subcommand-and-memory-limit.md)).
The rest of the reasoning is informally documented here pending a future
ADR.
