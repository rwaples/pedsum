# DESIGN.md — pedsum internals (maintainer notes)

Pointers to where implementation details live. The README is the
collaborator-facing surface; this file is for maintainers.

## Package layout

The implementation is the `pedsum/` package; `pedigree_summary.py` is a thin
runnable shim (`python pedigree_summary.py …`) that re-exports the symbols the
test suite imports directly. Modules, in dependency order (each imports only
from those above it):

| Module | Responsibility |
|---|---|
| `pedsum/base.py` | shared constants, the `pedigree_summary` logger, `PedigreeError` |
| `pedsum/progress.py` | `relationship_progress`: the terminal progress bar for pedigree-graph's long relationship calls |
| `pedsum/pedigree_ops.py` | low-level array helpers (parent rows, parent references, sib groups, topological depth, sort-based distinct integers) |
| `pedsum/parse.py` | delimiter sniffing, column coercion, sex decoding, trait-column reading |
| `pedsum/checks.py` | per-check finding producers + check metadata (`_CHECK_*`) |
| `pedsum/validate.py` | `load_and_validate` (fail-fast) / `validate_pedigree` (accumulating) + sex imputation |
| `pedsum/pairs.py` | relationship-pair enumeration + `PedigreeGraph` construction |
| `pedsum/sections.py` | per-section summary computations |
| `pedsum/sex_concordance.py` | Offspring Sex Concordance: group projection, exact conditional moments, Holm, permutation samplers |
| `pedsum/assortative_mating.py` | Mate Correlation: trait typing, Mate Networks, thin strata, the per-cell estimators and their sandwich SEs, the father permutation with sequential stopping, the opt-in Mate Network bootstrap, the YAML payload |
| `pedsum/assortative_kernels.py` | the numba kernels behind `assortative_mating.py`: blocked moment, count and score passes, influence functions, the counter-based RNG, the permutation and bootstrap draw loops |
| `pedsum/schema.py` | categorised YAML schema + slim/extra split machinery |
| `pedsum/memory.py` | the best-effort memory limit: `MemoryWatchdog`, default-limit discovery, `--max-memory` parsing |
| `pedsum/report.py` | report payload builders, safe-attempt redaction, output writers, `atomic_output` |
| `pedsum/cli.py` | argparse, the `summarize` / `validate` / `effective-size` / `epimight-input` / `assortative-mating` runners, `main` |

## CLI design rationale

See [docs/adr/0001-collaborator-cli-redesign.md](docs/adr/0001-collaborator-cli-redesign.md).

## Output schema

Output schema follows [CONTEXT.md](CONTEXT.md) as of 0.10. The glossary fixes section names, key names, and the convention for distribution naming (`<noun>_count` / `<noun>_count_hist` / `<noun>_count_<sex>`). When extending the output, add the term to CONTEXT.md first, then emit keys that match it.

## Flag and behavior version history

See [CHANGELOG.md](CHANGELOG.md). Notable:
- `assortative-mating` reports the Mate Correlation (0.15; pedsum#13).
- Effective population size moves from `summarize` to `effective-size`,
  and every command runs under a memory limit (0.15; see
  [docs/adr/0004](docs/adr/0004-effective-size-subcommand-and-memory-limit.md)).
- `validate --drop-offending` emits a **Reduced Pedigree** (0.11; see
  [docs/adr/0003](docs/adr/0003-drop-offending-reduction.md)). `--no-sex-check`
  now lives in the registry (`ValidationContext.no_sex_check`) so the tolerance
  composes inside `_run_checks` rather than as a cli post-filter.
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
  large or deep runs don't silently hang. See `_F_WALK_WARN_VISITS`
  (`pedsum/base.py`) and its use in `_run_summarize` (`pedsum/cli.py`).
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
README covers its use and output. `pedsum/assortative_mating.py` types
the traits, builds each cell's sample and writes the records.
`pedsum/assortative_kernels.py` holds the numba kernels that every pass
over the pairs runs in. The choices below can be reversed in code, so
they live here and not in an ADR.

**Cluster unit: the Mate Network.** Mating Pairs that share a parent
are not independent. A father with three mates puts his value into three
pairs. Treating pairs as independent would understate the variance. A
Mate Network is the smallest unit that holds every pair sharing a
parent, so the sandwich SE sums influences within networks and the
bootstrap resamples whole networks. Each cell finds its networks over
its own analysed pairs. Clustering by Component would also hold
relatives together, but a pedigree that is one Component would leave
one cluster. The cost is one assumption. Networks are taken as
independent, and dependence between them through ancestry or siblings
is not modelled. The YAML (`inference.bootstrap_assumption`) and the
README state it, and each cell reports `n_mate_networks` and
`largest_mate_network_share` so a reader can see when a few networks
dominate.

**Default CI: a cluster-robust two-step sandwich.** Refitting every
estimator 1,000 times per cell does not scale to 10^7 pairs, so the
bootstrap is opt-in. The default `se` stacks the first-step estimating
equations (thresholds per sex × stratum, stratum means and 1/N
variances) with the ρ score. Influences are summed within Mate Networks
and scaled by `G/(G−1)`, so first-step uncertainty reaches ρ̂ through
`A_ρθ` (Olsson 1979 eqs 21-28; Olsson, Drasgow & Dorans 1982 eq 37).
Each nuisance equation involves one parameter, so `A_θθ` is diagonal
and every estimator's influence is one pass over its pairs
(`Estimator.influence`). The CI is Wald on the Fisher-z scale (log for
the odds ratio) so bounds stay in range. At a boundary fit the sandwich
is not valid and the CI is withheld. Spearman has no influence function
and gets a CI only under `--bootstrap`. On 400 remated pairs in 200
networks the sandwich SE came within a ratio of 0.93 to 1.06 of the
refit bootstrap SD on two seeds; `tests/test_assortative_sandwich.py`
asserts 15%, and checks the SE against a finite-difference oracle to
1e-6.

**Bootstrap draws refit the first step and take one Newton step.**
Thresholds, stratum means and SDs are estimates from the same sample as
ρ. Holding them at their full-sample values in each draw would treat
them as known and narrow the CI, so every draw recomputes them on its
weighted sample. ρ then moves by one Newton step from ρ̂, with the
draw's score and the full-sample Hessian. The full-sample Hessian was
chosen over the draw's own by measurement. Its worst CI bound came
within 0.0012 of a full refit at 2,500 remated pairs and 0.0006 at
5,000, against 0.0048 and 0.0024 with the draw's own Hessian (the
development measurement quoted in `assortative_kernels.py`).
`tests/test_assortative_bootstrap.py` holds the gap under 0.002 at 2,500
pairs. A draw whose Hessian is not positive is refit in full. Each
`Estimator.fn` takes the whole cell, strata included, so the observed
fit, that fallback and the test oracles call one function.

**Permutation null: father vectors within blocks.** A permutation moves
trait vectors between distinct fathers, both traits together, and leaves
mothers in place. A father therefore carries one vector to all his
mates, and the Mating Pair structure, who mated with how many, stays
fixed. Fathers swap only within their block, father stratum × which
traits the father has. Under the stratum part, a father takes only the
vector of another father in his stratum, so the null never mixes
cohorts. The missingness part keeps each cell's eligible pairs, and so
its `n`, the same in every permutation. The null is that, given the
Mating Pair structure, father trait vectors are exchangeable within each
block. That is stronger than zero correlation, and the README says so.
All cells share one donor array per permutation. The blocks are the same
for the crude and the stratified form, so under `--stratify-by` the crude
`p_perm` also tests within-stratum exchangeability, while the crude
estimate and CI describe the pooled correlation. A cohort trend with no
within-stratum assortment therefore moves the crude CI away from 0 but
not the crude p-value. Pooling the crude donors across strata would make
the crude test reject on the trend itself, which is the confound the
stratification is there to remove.

**Calibration.** `tests/test_assortative_calibration.py` (marked `slow`)
simulates repeated datasets with a known latent mate correlation and
checks the inference against binomial Monte Carlo bounds of 3 SD. Under
`R_mf = 0`, with no remating, with heavy remating (5 mates per father)
and with a sparse binary trait (prevalence 0.05), the permutation test
rejects at most α + 3 SD at α = 0.05 over 500 datasets, in the
Pearson, tetrachoric and biserial cells, crude and stratified (every
dataset runs `--stratify-by birth_year`). Under a birth-decade trend
shared by both mates, the stratified test holds the same bound, and the
crude Pearson sandwich CI excludes 0 in at least half of 100 datasets,
so the trend is real. At `R_mf = 0.3`, with and without remating, every
sandwich CI (the five primaries, phi, point-biserial and the odds ratio)
covers the truth at 95% within 3 SD over 500 datasets, and coverage above
0.995 fails as too wide. The one-step bootstrap (399 draws) is checked
the same way for tetrachoric and Pearson over 200 remated datasets.

**Permutation statistic: the score at ρ = 0.** Refitting ρ̂ on every
permutation needs an optimiser per draw and per cell. The score of the
log-likelihood at ρ = 0 needs none. With `e` each side's conditional
latent mean given its value (`z` for a continuous side,
`(φ(τ_c) − φ(τ_{c+1})) / p_c` for level `c` of a discrete side), the
statistic is `Σ e_m e_f / √(Σ e_m² Σ e_f²)`, the ρ = 0 score scaled to a
correlation. For two continuous traits it is the Pearson r. Every father
margin is refit on the permuted pairs, because the pair-weighted margins
move when remating fathers swap. The mothers' scores do not move and
are summed per father once. The kernel then streams each cell's distinct
fathers once per draw.

**Sequential stopping: the Besag-Clifford closed scheme.** Most cells
of a run are null or weak, and 999 draws of each are wasted work once it
is clear there is no evidence against the null. The p-value is the
closed sequential scheme of Besag & Clifford (1991). Read the draws in
order and stop at the `h`-th valid one at least as extreme as the
observed statistic (`|T*| ≥ |T_obs|`, ties counted, which is the
conservative end `h / l⁻` of the range the paper quotes for a discrete
statistic). Then `p = h / l`, with `l` the valid draws so far. Without
`h` exceedances at the last of the `n − 1 = --permutations` draws,
`p = (g + 1) / n` as in the fixed-size test. `h = 20` (`SEQUENTIAL_H`;
the paper suggests 10 or 20). Under the null, `p` is a uniform rounded
up to the support `{1, h/(h+1), …, h/(n−1), h/n, …, 1/n}`, so the test
is exact at every size, and the expected draw count of a null cell is
about `h + h·log(n/h)` (97.7 for n = 1000, h = 20). Failed draws (an
undefined statistic in the permuted sample) count toward neither `l`
nor `n`; the record carries `draws_used`, which counts them, next to
`valid`.

The decision must be a deterministic function of the draw sequence, so
the kernel runs draws in batches whose sizes depend on the draws, never
on `--threads`. The first has `PERMUTATION_BATCH = 64` draws. Each later
one at least doubles the draws so far and runs on to the draw where the
nearest open form would reach `h` at its exceedance rate so far. A
strong-signal run therefore takes two launches instead of sixteen. The
Python side scans each batch in draw order per cell and form to find the
exact stopping draw, and a cell leaves the next batch once every tested
form has stopped. The record truncates the sequence it is given at the
`h`-th exceedance, so a kernel that computed more draws than needed
gives the same record. The scheme saves time only for null and weak
cells: a strongly significant cell never collects `h` exceedances and
runs every draw.

**Threshold estimators: two-step ML with a boundary flag from the fit.**
`polychoric` (tetrachoric at two levels) and `polyserial` (biserial at
two levels) take thresholds in closed form from the margins (Olsson 1979
eqs 15-18; Olsson, Drasgow & Dorans 1982 eq 36), which leaves a
one-dimensional search for ρ on (−0.9999, 0.9999), since the bivariate
normal is singular at ±1. Newton's method on the analytic score
(Olsson 1979 eq 9; Olsson, Drasgow & Dorans 1982 eq 26) finds ρ̂ from 0,
or from the eq 38 ad hoc estimate for `polyserial`. It hands over to
bounded Brent (`minimize_scalar`) when the curvature is not positive, a
step leaves the interval, or `NEWTON_MAX_ITER = 12` steps do not reach
`NEWTON_TOL = 1e-8`, so every fit reaches the optimum Brent would.
`FIT_OUTCOMES` counts each ending for the `-v` log. Olsson's appendix A2
prints `∂φ2/∂ρ` with two errors; the docstring of `bvn_pdf_and_drho`
gives the corrected form, which a finite-difference test checks.

Φ2 is Owen's (1956) closed form in Owen's T, evaluated by
`scipy.special.owens_t` on the corner grids of a table, never per pair.
The tests hold it to 1e-12 of `scipy.stats.multivariate_normal.cdf`.
numba kernels cannot call `scipy.special`, so the kernels carry their
own Φ (through `math.erfc`) and Φ⁻¹ (Wichura's (1988) AS 241, held to
1e-15 relative of `scipy.special.ndtri` on a test grid).

`boundary: true` comes from the fit, never from a zero count, because a
zero cell does not put ρ̂ at a bound. The 3×3 table with an empty centre
fits ρ̂ ≈ 0, with NLL 173.15 at 0 and 516.22 at ±0.99. The flag is set
when ρ̂ is within `BOUNDARY_MARGIN` (1e-3, an arbitrary choice) of a
bound, or when the NLL at the nearer bound is within `BOUNDARY_NLL_TOL`
(1e-6, relative) of the optimum. The second rule exists because a
likelihood can be flat up to the bound. On `[[30,10],[0,20]]` the NLL is
60.684256 from 0.998 to 0.9999, the optimiser stops at 0.9985, 1.4e-3
from the bound, and the distance rule alone misses it.

**Pair-weighted margins.** The Mate Correlation has one observation per
Mating Pair (CONTEXT.md), so every nuisance parameter comes from the
same observations. Means, SDs, thresholds and tables are computed over
the cell's pairwise-complete pairs, and a father with three mates counts
three times. Margins weighted per person would standardise against a
different sample from the one the correlation runs over. The
Within-Person Cross-Trait Correlation is a statistic about people, so it
counts each person once.

**Thin strata are dropped by Mate Network count.** Under
`--stratify-by`, a sex × stratum whose pairs come from a few networks
gives thresholds and SDs that swing between draws, and it is often
absent or constant in a draw, which fails the draw and can withhold the
CI. The alternatives were withholding the CI and skipping the stratum
inside a draw; the second changes the estimand from draw to draw. So
`drop_thin_strata` drops, before any fit, each sex × stratum whose pairs
span fewer than `--min-stratum-networks` networks (default 10, an
arbitrary choice), and each constant one. It counts networks, not
pairs, because the network is the cluster unit. A pair's mother and
father strata can differ, so dropping one stratum's pairs can remove
another stratum's networks or leave it constant. Both rules therefore
repeat, with networks recomputed on the kept pairs, until neither drops
a pair. Both rules depend on the cell's values, so the counts are per
cell (`n_dropped.small_stratum`, `n_dropped.degenerate_stratum`), and
the crude and stratified estimates share the kept sample.

**One numba kernel path, with results independent of `--threads`.**
numba is a hard dependency, so there is one production path and no
NumPy fallback to keep in step; the NumPy estimators it replaced live on
in `tests/assortative_oracles.py` as oracles. numba pins numpy (below
2.6 for numba 0.68), a tighter ceiling than pedsum's own. Three rules
keep the output a function of the input and `--seed` alone:

- Every draw takes its random numbers from SplitMix64 keyed by
  (seed, draw), with a separate stream domain for the bootstrap, so a
  draw is the same whichever worker runs it.
- Every O(n) sum of the point fits and the sandwich adds fixed blocks of
  16,384 pairs in block order, so the floating-point result does not
  depend on how many workers share the blocks.
- The sequential stopping batches by draw count (above).

The permutation pass streams the fathers' trait values and summed
mother scores as float32 and accumulates in float64. That halved the
kernel's time at 12 threads on 3 × 10^6 pairs and moves a statistic by
about 1e-7 relative; the tests hold observed statistics within 1e-6 and
null draws within 1e-5 of the float64 path. Once the traffic was
halved, 6 and 12 threads tied on that kernel, so the default is the
physical cores in the process's affinity set, with SMT siblings counted
once (`cli.physical_cores`).

References:

- Besag, J. & Clifford, P. (1991). Sequential Monte Carlo p-values.
  *Biometrika*, 78(2), 301–304. <https://doi.org/10.1093/biomet/78.2.301>
- Olsson, U. (1979). Maximum likelihood estimation of the polychoric
  correlation coefficient. *Psychometrika*, 44(4), 443–460.
  <https://doi.org/10.1007/BF02296207>
- Olsson, U., Drasgow, F. & Dorans, N. J. (1982). The polyserial
  correlation coefficient. *Psychometrika*, 47(3), 337–347.
  <https://doi.org/10.1007/BF02294164>
- Owen, D. B. (1956). Tables for computing bivariate normal
  probabilities. *Annals of Mathematical Statistics*, 27(4), 1075–1090.
  <https://doi.org/10.1214/aoms/1177728074>
- Wichura, M. J. (1988). Algorithm AS 241: The percentage points of the
  normal distribution. *Applied Statistics*, 37, 477.
  <https://doi.org/10.2307/2347330>

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
