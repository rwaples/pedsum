# assortative-mating cost (phase 5 draft; phase 6 finalises)

Config (b): `--trait liab dx --stratify-by birth_year --birth-year-col birth_year --threads 12`, default
inference (sandwich CI, 999 score-statistic permutations, no bootstrap). Inputs from
`benchmarks/generate_assortative_mating.py --pairs N --seed 0` (2.76 rows per pair; 8 birth-decade strata,
0.85 distinct fathers per pair, no missing trait values, so all four cells share one pair set and no
stratum is dropped).

Machine: i7-9750H (6 cores / 12 threads, 12 MiB L3, 31 GiB RAM), numpy 2.5.2, numba 0.68.0, polars
1.44.2, Python 3.14. Busy-loop clock 3.1-4.1 GHz, `platform_profile` performance, `no_turbo` 0,
`max_perf_pct` 100, `scaling_max_freq` 4.5 GHz. Load average 3-11 from other sessions throughout, so
single runs carry roughly 10-20% noise.

## Method

A stage driver runs the CLI path (`load_and_validate`, `read_trait_columns`, `classify_trait`,
`compute_assortative_mating`) with the module's functions wrapped by a timer. A top-level stage resets
the RSS high-water mark (`/proc/self/clear_refs` = 5) on entry and reads `VmHWM` on exit, so its peak is
exact. Rows starting with `>` are calls nested in the stage above them (wall only). Each run sits in its
own `systemd-run --user --scope` for cgroup `memory.peak`. `ru_maxrss` is invalid under the driver
because the resets clear it. Before = the worktree at the start of this phase (frozen copy); after = the
worktree now.

## Before / after at 1e6 pairs (medians of 3)

| stage | before 1e6 wall s | after 1e6 wall s | saving s | before peak MiB | after peak MiB |
|---|---:|---:|---:|---:|---:|
| load_and_validate | 2.431 | 2.397 | 0.034 | 1094 | 1104 |
| read_trait_columns | 0.439 | 0.432 | 0.007 | 1124 | 1137 |
| classify_trait | 0.567 | 0.311 | 0.255 | 1123 | 1159 |
| _parent_rows | 0.111 | 0.122 | -0.011 | 922 | 1080 |
| _group_mating_pairs | 0.078 | 0.079 | -0.001 | 869 | 1026 |
| strata | 0.013 | 0.014 | -0.001 | 834 | 986 |
| father_blocks | 0.715 | 0.024 | 0.691 | 851 | 907 |
| mate_networks | 0.901 | 0.064 | 0.837 | 1195 | 906 |
| > mate_networks | 0.715 | 0.283 | 0.431 |  |  |
| > stratum_networks | 2.618 | 0.191 | 2.427 |  |  |
| > degenerate_strata | 0.164 | 0.163 | 0.001 |  |  |
| drop_thin_strata | 3.658 | 0.726 | 2.932 | 1240 | 1131 |
| > sandwich_se | 0.944 | 0.993 | -0.049 |  |  |
| fit_cell | 1.724 | 1.833 | -0.110 | 1195 | 1131 |
| _mother_scores | 0.129 | 0.134 | -0.005 | 1195 | 1131 |
| > _PermutationInput.pack | 0.213 | 0.246 | -0.033 |  |  |
| > kernels.permutation_statistics | 35.072 | 6.389 | 28.683 |  |  |
| > kernels.arrangement_statistics | 0.063 | 0.026 | 0.037 |  |  |
| _permutation_statistics | 35.378 | 6.658 | 28.720 | 1388 | 1429 |
| permutation_record | 0.002 | 0.002 | -0.000 | 1267 | 1132 |
| within_person | 1.066 | 0.527 | 0.539 | 1267 | 1160 |
| compute: unwrapped remainder | 2.109 | 0.839 | 1.270 | 1388 | 1429 |
| **total (driver)** | 49.65 | 14.83 | 34.82 | 1388 | 1429 |

Before, top-level `mate_networks` counted 5 calls (all pairs plus one per cell); after, it is the
all-pairs call only, because cells reuse the labels from `drop_thin_strata`'s last round.

Whole runs (driver total, median of 3 unless noted; peak = max per-stage `VmHWM`; cgroup `memory.peak`
in brackets):

| pairs | before wall | after wall | before peak MiB | after peak MiB |
|---:|---:|---:|---:|---:|
| 1e5 | 2.56 s | 1.58 s | 384 (259) | 402 (277) |
| 1e6 | 49.65 s | 14.83 s | 1388 (1269) | 1429 (1308) |
| 3e6 (1 run) | 204.6 s EXTRAPOLATED (64.88 s at 99 permutations; kernel 15.38 s x 999/99) | 55.97 s | 3507 (3398) at P=99 | 3678 (3565) |
| 1e7 (1 run) | not run | **288.13 s** (`/usr/bin/time`; driver total 286.47 s) | | 11434 (cgroup 11,339 MiB) |

The 1e7 run predates the last kernel change (one contiguous column per trait in the arranged values,
measured -12% on the kernel at 3e6, see below). Free memory was about 13 GiB at the start and other
sessions pushed 4 GiB more into swap during the run (PSI memory "some" about 11 s), so the 1e7 stages
that run near the memory peak (pack, drop_thin_strata, the remainder) came out well above their
projections.

## Fixes (measured savings; outputs equal)

| fix | where | measured |
|---|---|---|
| Permutation kernel: each cell collapsed onto its distinct fathers (pair count, summed crude and stratified mother scores) and one pass per cell with sums about the stratum's observed mean, instead of two passes over every pair with a random gather per pair | `_cell_fathers`, `_continuous_cell`, `_discrete_cell` | 1e6, 199 draws, isolated kernel: 8.0-10.2 s -> 2.7 s |
| Fathers renumbered block by block; the shuffle permutes int32 row indices inside contiguous blocks and gathers the trait rows once per draw (same Fisher-Yates draws as `shuffle_blocks`) | `shuffle_index`, `permutation_statistics` | 3e6, 99 draws: 3.33 s -> 2.79 s (12 threads) |
| int32 father rows and pair counts in the streamed cell arrays | `_cell_fathers` | 3e6, 99 draws: 3.58 s -> 3.33 s |
| Arranged values stored one contiguous column per trait, so a cell streams only its father trait | `row_statistics` | 3e6, 99 draws, interleaved A/B x3: median 3.02 s -> 2.65 s (noisy) |
| Cell collapse by `bincount` instead of a stable argsort + `reduceat` (bit-identical sums) | `_cell_fathers` | 1e6: pack 0.59 s -> 0.25 s |
| `stratum_networks` with `unique_ints` (sort) instead of the hash `np.unique` | `stratum_networks` | 1e6: 2.62 s -> 0.19 s |
| `drop_thin_strata` returns its last round's Mate Network labels; cells stop recomputing them | `drop_thin_strata`, `compute_assortative_mating` | 4 `mate_networks` calls per run removed (1e6 ~0.3 s) |
| `mate_networks` uses pedigree rows as graph nodes and an O(n) first-pair renumbering | `mate_networks` | 1e6 nested calls: 0.72 s -> 0.28 s |
| `father_blocks` keyed on one integer instead of `np.unique(axis=0)` | `father_blocks` | 1e6: 0.72 s -> 0.02 s |
| `within_person`, `n_fixed_fathers` with `unique_ints`; mother remating count by `bincount`; stratum codes by `searchsorted` | `within_person`, `compute_assortative_mating` | 1e6: within_person 1.07 -> 0.53 s; remainder 2.11 -> 0.84 s |
| `classify_trait` builds one polars Series from the tokens (no Python loop for the missing mask) | `classify_trait` | 1e6: 0.57 s -> 0.31 s |

Equality. Twelve configurations through the full compute path give payloads identical to the
before tree, every float equal (maximum absolute difference 0.0): config (b) at 1e4 and 1e5; 1e5 with
10% / 5% missing values (stratified, stratified with `--bootstrap 50`, unstratified two traits,
unstratified `--trait dx`); thin strata (1e4, `--birth-year-bin 1 --min-stratum-networks 80`, two
drop rounds, 92 pairs dropped); two binary traits crude and `--stratify-by depth`; binary x continuous by
depth; null traits (shuffled across rows, p-values 0.03-0.83) stratified at 1e5 and crude at 1e4. The raw
permutation statistics differ by float summation order only: maximum |difference| 2.3e-14 over all
null draws, 2.1e-13 on an observed statistic of 0.30. NaN patterns (failed draws) are identical.

## Scaling model (after) and 10^7

| stage | 1e6 s | 3e6 s | exponent | 1e7 projected s (EXTRAPOLATED) | 1e7 measured s |
|---|---:|---:|---:|---:|---:|
| load_and_validate | 2.397 | 7.713 | 1.06 | 27.8 | 43.8 |
| read_trait_columns | 0.432 | 1.361 | 1.05 | 4.8 | 4.5 |
| classify_trait | 0.311 | 1.474 | 1.41 | 8.1 | 4.3 |
| _parent_rows | 0.122 | 0.362 | 0.99 | 1.2 | 1.7 |
| _group_mating_pairs | 0.079 | 0.297 | 1.21 | 1.3 | 1.5 |
| strata | 0.014 | 0.054 | 1.25 | 0.2 | 0.5 |
| father_blocks | 0.024 | 0.073 | 1.00 | 0.2 | 0.8 |
| mate_networks | 0.064 | 0.205 | 1.07 | 0.7 | 1.3 |
| > mate_networks | 0.283 | 0.861 | 1.01 | 2.9 | 7.3 |
| > stratum_networks | 0.191 | 0.671 | 1.14 | 2.7 | 3.7 |
| > degenerate_strata | 0.163 | 0.249 | 0.39 | 0.4 | 1.1 |
| drop_thin_strata | 0.726 | 2.011 | 0.93 | 6.1 | 16.1 |
| > sandwich_se | 0.993 | 2.724 | 0.92 | 8.2 | 10.0 |
| fit_cell | 1.833 | 5.307 | 0.97 | 17.0 | 21.8 |
| _mother_scores | 0.134 | 0.252 | 0.58 | 0.5 | 3.4 |
| > _PermutationInput.pack | 0.246 | 0.798 | 1.07 | 2.9 | 15.3 |
| > kernels.permutation_statistics | 6.389 | 32.475 | 1.48 | 192.9 | 152.1 |
| > kernels.arrangement_statistics | 0.026 | 0.056 | 0.69 | 0.1 | 0.1 |
| _permutation_statistics | 6.658 | 33.354 | 1.47 | 195.0 | 167.7 |
| permutation_record | 0.002 | 0.002 | 0.00 | 0.0 | 0.0 |
| within_person | 0.527 | 1.293 | 0.82 | 3.5 | 4.3 |
| compute: unwrapped remainder | 0.839 | 2.209 | 0.88 | 6.4 | 14.7 |
| **total** | 14.83 | 55.97 | 1.21 | 273 (sum of stages) | 286.5 |

Peak (max per-stage VmHWM): 1e6 1429 MiB, 3e6 3678 MiB; linear slope 1124 MiB per 1e6 pairs; 1e7 projection 11550 MiB (EXTRAPOLATED); 1e7 measured 11434 MiB.

Target: about 5 min and at most 20 GiB at 10^7 pairs. Measured 288 s and 11.1 GiB (cgroup). The
power-law projection from 1e6 and 3e6 gives 273 s (EXTRAPOLATED).

## Remaining top costs at 10^7 (measured run)

1. The permutation kernel, 152 s (53%). It is memory-bound. The cell passes stream about 40 bytes per
   father per cell per draw; at 3e6 that is about 35 GB/s (estimated from bytes over time), close to the
   DRAM peak, and the Fisher-Yates swaps plus
   the gather are random accesses into blocks of about 1M fathers. At 3e6 the same run took 22.4 s of
   kernel with `--threads 6` against 31.5 s with `--threads 12`: hyperthreads add contention and do not
   help here. Untried: processing 2-4 draws per pass over the cell arrays, which would halve their
   traffic (about -20% kernel) at 1.6-3.3 GiB more workspace at 1e7.
2. `load_and_validate`, 44 s (load path).
3. `fit_cell`, 22 s (point estimates about 12 s, sandwich 10 s), close to linear.
4. `drop_thin_strata` 16 s, pack 15 s and the remainder 15 s. All three are linear at 1e6 -> 3e6 and
   only grew at 1e7 under memory pressure (UNVERIFIED as the cause; one run).

Memory at 1e7: `load_and_validate` peaks at 8.4 GiB, and compute's stages start from 5-9 GiB of
retained state (the frame, the traits, and the raw token arrays that `_run_assortative_mating` keeps
alive through compute, projected at about 1.9 GiB at 1e7 in the phase-5a notes, EXTRAPOLATED). The
permutation stage peaks at 11.2 GiB: 12 worker workspaces (int32 order and the float64 arranged values,
about 2 GiB at 8.5M fathers) plus the collapsed cell arrays.

## numba cold compile

Fresh `NUMBA_CACHE_DIR`, the CLI at 1e4: 13.46 s / 12.62 s cold against 1.43 s / 1.80 s warm, so about
11-12 s of compile. The cache holds 26 kernels with one specialisation each, so no redundant
signatures were found and no cheap reduction is evident.

## Unverified

- The 1e7 figure is one run, taken under load from other sessions and with swap activity, on the code
  before the column-layout change.
- 1e7 inputs here have no missing values and no thin strata. Real data takes more `drop_thin_strata`
  rounds (one `mate_networks` and two `stratum_networks` calls each) and different cell father sets.
- The `--threads 6` advantage was measured at 3e6 only. At 1e7 every worker's random working set
  exceeds L3 either way.

## Phase 4: opt-in one-step bootstrap (`--bootstrap 1000`)

Config (b) with `--threads 6`, 999 permutations, `/usr/bin/time` wall and `ru_maxrss`
(`benchmarks/bench_assortative_mating.py`, inputs `am_{100000,1000000}.tsv` from the generator with seed 0).
Clock under a 2 s busy loop 4.27-4.29 GHz, `platform_profile` performance, `no_turbo` 0, `max_perf_pct` 100,
`scaling_max_freq` 4.5 GHz; load average about 1 before the runs.

| pairs | bootstrap | wall s (runs) | peak RSS MiB |
|---:|---:|---|---:|
| 1e5 | 0 | 2.4 (2.4, 2.4, 2.3) | 380 |
| 1e5 | 1000 | 7.5 (7.3, 7.5, 7.8) | 363 |
| 1e6 | 1000 | 101.9 (one run) | 1208 |

The 1000 draws cost about 5 s at 1e5 and about 87 s at 1e6 (against the 14.8 s phase-5 driver total
without a bootstrap at 12 threads); the growth from 1e5 to 1e6 is superlinear and not yet profiled.

## Phase 6: thread default, float32 streams, parallel fits (2026-10-06)

Same driver and config (b), default inference. Before = the worktree at the start of this phase (frozen
copy, numba cache warm after its first run); after = the worktree now. Load average 1-6 from other sessions;
free memory 4-9 GiB, so the 1e7 run again overlapped swap from other processes.

Changes measured here:

| change | where |
|---|---|
| `--threads` on `assortative-mating` defaults to the physical cores the process may run on (affinity set, SMT siblings counted once: 6 here); `settings.threads` records the numba count; pedigree-graph keeps 1 | `cli.physical_cores`, `_run_assortative_mating` |
| The streamed permutation arrays (arranged father values, per-father mother-score sums) are float32; the kernel accumulates in float64 | `STREAM_DTYPE`, `_cell_fathers`, `_PermutationInput.pack` |
| Every O(n) pass of the point fits and the sandwich is a blocked parallel reduction (fixed 16,384-pair blocks summed in block order, so results are bit-identical for any thread count; the draw kernels call serial twins over the same blocks) | `stratum_moments`, `pearson`, `count_table`, `margin`, `polyserial_nll/terms`, `polyserial_influence`, `pearson_influence`, `gather_influence` |
| The bootstrap Spearman pass walks pre-sorted copies of both sides (one random access per pair instead of six) | `continuous_draws`, `_spearman_sorted`, `sorted_ranks_into` |

### 1e6 pairs (medians of 3)

| stage | before, `--threads 12` | after, `--threads 12` | after, default (6) |
|---|---:|---:|---:|
| fit_cell | 1.684 | 1.559 | 0.950 |
| > sandwich_se | 0.914 | 1.078 | 0.558 |
| > kernels.permutation_statistics | 5.426 | 4.560 | 4.269 |
| within_person | 0.430 | 0.415 | 0.194 |
| **total (driver)** | 12.68 | 11.80 | 10.37 |
| peak MiB (VmHWM / cgroup) | 1487 / 1363 | 1478 / 1352 | 1275 / 1155 |

One of the three after `--threads 12` runs overlapped a test-suite run (42 s total; the median hides it,
but its fit_cell and sandwich medians are above the default-thread figures for that reason).

### 3e6 pairs (one run each)

| stage | before t12 | before t6 | after t12 | after default (6) |
|---|---:|---:|---:|---:|
| fit_cell | 5.312 | 5.029 | 2.718 | 2.401 |
| > sandwich_se | 2.786 | 2.550 | 1.481 | 1.215 |
| > _PermutationInput.pack | 0.836 | 0.804 | 1.471 | 1.389 |
| > kernels.permutation_statistics | 28.549 | 22.915 | 14.203 | 14.017 |
| within_person | 1.136 | 1.189 | 0.453 | 0.517 |
| **total (driver)** | 51.10 | 45.61 | 33.99 | 34.90 |
| peak MiB (VmHWM / cgroup) | 3678 / 3565 | 3386 / 3272 | 3447 / 3333 | 3273 / 3427 |

float32 storage halves the kernel at 12 threads (28.5 -> 14.2 s) and takes 39% off at 6 (22.9 -> 14.0 s).
With the traffic halved, 6 and 12 threads tie on the kernel (14.0 vs 14.2 s), so the physical-core default
costs nothing against 12 and saves the 12-worker workspace (VmHWM -174 MiB; the cgroup peak, which counts page cache, went the other way by 94 MiB). The pack stage grew by
0.6 s at 3e6 in both after runs (the float32 casts measure 0.18 s in isolation; cause UNVERIFIED).

### Opt-in bootstrap (`--bootstrap 200`, `--threads 6`, 999 permutations at 1e6, 0 at 1e5)

| pairs | before fit_cell | after fit_cell |
|---:|---:|---:|
| 1e5 (3 runs) | 1.468 | 1.581 |
| 1e6 (2 warm runs) | 19.232 | 13.927 |

Cause of the superlinear growth (profiled at 1e5 and 1e6 before): `_continuous_draws` went 0.37 -> 8.9 s
(24x for 10x pairs) while the polyserial draws went 0.93 -> 8.2 s (linear). The Spearman pass gathered
`v[order[i]]`, `w[order[i]]` and `rank_other[order[i]]` through the sort permutation and
`weighted_ranks_into` wrote `rank[order[t]]`, six random streams over 8 MB arrays per draw, which fall out
of L3 between 1e5 and 1e6. Walking pre-sorted copies leaves one random read per pair (the father rank).
The remaining random access is `mult[labels[i]]` in `network_weights` (one int32 per Mate Network per
worker, 3 MB at 1e6 and three gathers per continuous draw), not changed here. Results: the 1e5 payload
with `--bootstrap 200 --permutations 199` differs from the before tree only in float summation order
(36 fields, max |difference| 2.6e-13 on an `r`, 8e-15 on an `se`; p-values and Spearman draws identical).

### 1e7 pairs (one run, default threads = 6, cgroup scope, 5-minute cap)

| stage | phase 5 (t12) | now |
|---|---:|---:|
| load_and_validate | 43.8 | 31.0 |
| drop_thin_strata | 16.1 | 25.9 |
| fit_cell | 21.8 | 9.0 |
| > _PermutationInput.pack | 15.3 | 11.8 |
| > kernels.permutation_statistics | 152.1 | 88.9 |
| within_person | 4.3 | 4.9 |
| compute: unwrapped remainder | 14.7 | 11.9 |
| **total (driver)** | 286.5 | **196.2** |
| peak MiB (VmHWM / cgroup) | 11434 / 11339 | 9736 / 10759 |

Target 5 min and 20 GiB: 196 s and 10.5 GiB (cgroup). Free memory was 4.1 GiB at launch (27 GiB in use
by other sessions), so the stages near the peak (`drop_thin_strata`, pack, the remainder) again ran under
memory pressure; `drop_thin_strata` at 25.9 s is above its 3e6 scaling (UNVERIFIED as the cause; one run).

### Equality

Sandwich SEs, estimates and CIs move by float summation order only (blocked sums): 1e5 config (b) with
bootstrap and permutations, 36 fields, max |difference| 2.6e-13. The permutation statistics move by the
float32 rounding of the streamed arrays: tests hold the observed statistic within 1e-6 and the null
statistics within 1e-5 of the float64 path on every cell type, with identical failure patterns; a p-value
changes only where a null draw ties |T_obs| within 1e-5 (discrete cells repeat their table under many
permutations, so exact ties are common there and rounding decides the side).

## Sequential stopping (decision 21, Besag & Clifford 1991 closed scheme, h = 20)

Config (b) at `--threads 6`, 999 permutations, medians of 3 interleaved before/after runs per point
(`bench_assortative_mating.py`; before = the tree frozen before the change). The null inputs come from
`generate_assortative_mating.py --pairs N --seed 0 --r-mf 0 0 0 0` (same remating and strata, R_mf = 0).
Load average from other sessions rose from 5 to 10 during the sequence (each point's before ran before
its after), so the signal-input differences are inside the noise of this run and UNVERIFIED as real
costs; a quiet-machine re-measurement is needed before reading more than the null-input saving from them.

| input | pairs | before wall s (runs) | after wall s (runs) | before RSS MiB | after RSS MiB |
|---|---:|---|---|---:|---:|
| assortative (R_mf 0.30/0.15/0.05/0.25) | 1e5 | 3.20 (2.48, 3.20, 4.51) | 4.26 (4.30, 4.25, 4.26) | 349 | 343 |
| null (R_mf = 0) | 1e5 | 3.94 (3.94, 3.80, 3.97) | 3.56 (3.36, 3.56, 3.58) | 342 | 336 |
| assortative | 1e6 | 18.53 (18.93, 18.53, 18.35) | 19.69 (19.69, 19.98, 19.69) | 1134 | 1112 |
| null | 1e6 | 17.63 (17.63, 17.30, 17.95) | 11.07 (11.07, 11.93, 9.36) | 1113 | 1125 |

A standalone after run at 1e5 on the assortative input took 2.61 s (`/usr/bin/time`, idle-ish machine,
`NUMBA_DEBUG_CACHE=1` showed no cache saves), i.e. the same as the before tree's warm runs (2.51-2.60 s),
so the 1e5 "after" column above is load, not code. On the assortative input every cell has a strong
signal and never collects 20 exceedances, so all 999 draws run as before plus 16 kernel launches of 64
draws (one `prange` tail per launch instead of one per run); the possible few-percent kernel cost of that
is what the quiet-machine run has to settle. On the null input every cell stops after about
`20 + 20 log(1000/20)` draws (97.7 expected), so the kernel runs 2 batches (128 draws) instead of 999 and
the 1e6 wall falls 37%.

### Launch schedule: fixed 64-draw batches cost 3-5% on strong signals (2026-10-06)

The first stopping version ran the kernel in fixed batches of 64 draws, 16 launches for 999 permutations.
On the assortative input no cell stops, so all 999 draws ran as before, and an interleaved A/B (5 pairs,
load average 6-15) measured after/before median wall 1.032 at 1e5 and 1.054 at 1e6, with after slower in
all five 1e6 pairs.

Profile. cProfile and per-call timers over the permutation phase at 1e5 and 1e6 put the extra time in the
kernel launches. The Python side between launches (`_Stopping.scan`, the batch `_standardised`, the
`trait_needed` mask) took under 5 ms per run, and numba cache loads were about 210 ms in both trees.
Kernel timings of one packed input per size in one process, with the order rotated each repetition (medians of
8 at 1e6 and 24 at 1e5, load 4.7-6.5):

| schedule (999 draws) | launches | 1e5 kernel s | 1e6 kernel s | 1e6 vs single launch |
|---|---:|---:|---:|---:|
| one launch | 1 | 0.397 | 4.475 | 1.000 |
| fixed batches of 64 | 16 | 0.422 | 4.639 | 1.037 |
| doubling (64, 128, 256, 512, 999) | 5 | 0.403 | 4.565 | 1.020 |

Every launch ends at a barrier, and with 6 workers a 64-draw launch has 4 workers on 11 draws and 2 on 10,
so idle workers on the last draws add about 3% at 16 launches. The before tree's kernel and the after
kernel at one launch timed the same within run-to-run noise, both with the order rotated in one process
(1e6 medians 4.70 and 4.46 s), so the kernel changes that support stopping (a `cells` subset, the
`trait_needed` mask) cost nothing measurable.

Fix. The first launch runs 64 draws. Each later one at least doubles the draws so far and runs on to
`_Stopping.horizon()`, the draw where the nearest open form would reach `h` exceedances at its rate so far
(`h · l / max(g, 1)`). A form with no exceedances after 64 draws cannot stop before draw 1280, so a
strong-signal run takes 2 launches (0-64, 64-999). The null 1e5 input stops inside the first launch. The
payloads of the fixed-64 and horizon trees are identical apart from `generated_at`, at 1e5 on the
assortative and the null input with `--bootstrap 200`, and the stopping tests pass unchanged.

Interleaved CLI A/B after the fix: config (b), `--threads 6`, 999 permutations, no bootstrap. Before is the
tree frozen before stopping. The order alternates by pair (odd pairs run before first). The runs were gated
to start below load average 3, and load from other sessions then read 2.8-6.8 during the runs. The mean
clock across cores read 3.5-4.2 GHz (`platform_profile` performance, `no_turbo` 0, `scaling_max_freq`
4.5 GHz).

| input | pairs | n pairs | before wall s (runs) | after wall s (runs) | after/before median | after slower |
|---|---:|---:|---|---|---:|---:|
| assortative | 1e5 | 8 | 2.60 (2.46, 2.62, 2.59, 2.59, 2.60, 2.66, 2.60, 2.65) | 2.51 (2.61, 2.45, 2.44, 2.59, 2.45, 2.51, 2.51, 2.86) | 0.965 | 2/8 |
| assortative | 1e6 | 8 | 11.45 (11.31, 11.58, 11.94, 11.29, 11.52, 11.47, 11.43, 11.37) | 11.48 (11.31, 11.38, 11.51, 11.54, 11.43, 11.45, 11.80, 12.16) | 1.003 | 3/8 |
| null | 1e5 | 6 | 2.66 (2.67, 2.52, 2.47, 2.67, 2.77, 2.65) | 2.24 (2.24, 2.23, 2.30, 2.29, 2.20, 2.11) | 0.840 | 0/6 |
| null | 1e6 | 6 | 11.64 (11.57, 11.23, 11.71, 11.44, 11.73, 11.83) | 7.89 (7.76, 7.91, 7.79, 8.14, 8.19, 7.88) | 0.678 | 0/6 |

Paired differences (after minus before) on the assortative input were +0.15, -0.17, -0.15, +0.00, -0.15,
-0.15, -0.09, +0.21 s at 1e5 and +0.00, -0.20, -0.43, +0.25, -0.09, -0.02, +0.37, +0.79 s at 1e6. Their
order-corrected mean was -0.04 s at 1e5 and +0.08 s (0.7%) at 1e6, inside the run-to-run spread of about
±0.4 s at this load. The in-process table did not time the two-launch schedule. At 6 workers its busiest
worker runs 11 + 156 = 167 draws, the same as one launch, which leaves one extra barrier and launch
(UNVERIFIED as a kernel measurement).

## Final gate (2026-10-06)

One run of config (b) (`--trait liab dx --stratify-by birth_year --birth-year-col birth_year`), default inference (sandwich CI, 999 score-statistic permutations with Besag-Clifford stopping, no bootstrap), default threads (6 physical cores), on the 10^7-pair synthetic input (27,575,758 rows; 963 MB TSV), warm numba cache, in a `systemd-run --user --scope` unit. Load average 2.5 at start, 5.2 at end (other sessions); CPU 4.19 GHz at start.

| stage | wall |
|---|---:|
| load + validate | 22.6 s |
| assortative mating | 136.9 s |
| **total (`/usr/bin/time`)** | **168.3 s** |

Peak memory: 8,579,968 KB max RSS (8.18 GiB); cgroup `memory.peak` 8,719,052,800 B (8.12 GiB). Target was ~5 min and <= 20 GiB. One run, not a median. pedsum's default memory watchdog set its limit to 12.9 GiB (80% of the 16.2 GiB available at start); a machine with less free memory needs `--max-memory`.
