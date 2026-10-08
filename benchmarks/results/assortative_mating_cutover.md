# assortative-mating cutover: pedsum #13 vs pedsum on pg-phenotype 0.1.1

`assortative-mating` computes through pg-phenotype's `mate_correlation`
instead of pedsum #13's numba implementation. The gate for the cutover: over
repeated paired runs of the whole command, the median ratio (pg-phenotype
over numba) of wall time and of peak memory is at most 1.05 at every
configuration.

Result: **pass**. Worst median ratio: wall 0.885, peak 0.997.

## How it was run

`benchmarks/bench_assortative_mating.py run`, data in
[assortative_mating_cutover.jsonl](assortative_mating_cutover.jsonl):

- A: pedsum #13 at `142adf3` in a `git archive` snapshot with its own locked
  pixi env (numba 0.68.0, cache warm). B: this branch with pg-phenotype 0.1.1
  from PyPI (`933cbb5`).
- Inputs from `benchmarks/generate_assortative_mating.py --pairs N --seed 0`.
  Config `a` is `--trait dx`; config `b` is `--trait liab dx --stratify-by
  birth_year --birth-year-col birth_year`. 999 permutations, the default;
  `--bootstrap 1000` at 10^5 pairs only.
- 10 pairs per configuration, alternating which side runs first. Each run is
  a fresh process in its own systemd scope: wall from `/usr/bin/time`, peak
  from the scope's cgroup `memory.peak`. Each pair asserts both sides used the
  same permutation and bootstrap draws.
- Machine: i7-9750H (6 cores, 12 threads, 31 GiB), `platform_profile`
  performance, `scaling_max_freq` 4.5 GHz, 2026-10-08 16:17 to 17:07. Other
  sessions' jobs ran throughout, so the absolute times are high (the `max
  load` column is the 1-minute load at a pair's start). The pairing puts that
  load on both sides.

| config | Mating Pairs | P/B | threads | pairs | wall s A / B | wall B/A [min, max] | peak MiB A / B | peak B/A [min, max] | max load |
|---|---:|---|---:|---:|---|---|---|---|---:|
| a | 10,000 | 999/0 | 1 | 10 | 2.00 / 1.34 | 0.690 [0.621, 0.925] | 134 / 101 | 0.754 [0.749, 0.757] | 2.8 |
| a | 10,000 | 999/0 | 6 | 10 | 2.04 / 1.34 | 0.630 [0.493, 0.688] | 135 / 102 | 0.754 [0.747, 0.757] | 7.5 |
| b | 10,000 | 999/0 | 1 | 10 | 1.76 / 1.19 | 0.663 [0.642, 0.730] | 142 / 105 | 0.736 [0.731, 0.742] | 5.2 |
| b | 10,000 | 999/0 | 6 | 10 | 2.42 / 1.48 | 0.634 [0.504, 0.719] | 143 / 106 | 0.738 [0.733, 0.742] | 7.0 |
| a | 100,000 | 999/0 | 1 | 10 | 4.48 / 3.05 | 0.729 [0.642, 0.762] | 223 / 200 | 0.896 [0.850, 0.955] | 5.0 |
| a | 100,000 | 999/0 | 6 | 10 | 3.83 / 2.55 | 0.672 [0.583, 0.730] | 214 / 201 | 0.939 [0.873, 0.972] | 8.4 |
| a | 100,000 | 999/1000 | 6 | 10 | 5.79 / 4.12 | 0.720 [0.682, 0.754] | 208 / 197 | 0.944 [0.917, 0.976] | 22.0 |
| b | 100,000 | 999/0 | 1 | 10 | 5.82 / 3.86 | 0.689 [0.532, 0.864] | 239 / 224 | 0.940 [0.900, 1.014] | 7.5 |
| b | 100,000 | 999/0 | 6 | 10 | 5.71 / 2.67 | 0.524 [0.325, 0.629] | 222 / 231 | 0.997 [0.957, 1.109] | 10.9 |
| b | 100,000 | 999/1000 | 6 | 10 | 18.77 / 15.04 | 0.799 [0.710, 0.928] | 245 / 222 | 0.908 [0.867, 0.926] | 23.9 |
| a | 1,000,000 | 999/0 | 1 | 10 | 18.17 / 16.30 | 0.885 [0.735, 1.087] | 1011 / 984 | 0.962 [0.904, 1.025] | 6.4 |
| a | 1,000,000 | 999/0 | 6 | 10 | 8.66 / 8.62 | 0.884 [0.615, 1.127] | 1010 / 998 | 0.992 [0.945, 1.037] | 9.6 |
| b | 1,000,000 | 999/0 | 1 | 10 | 43.99 / 34.73 | 0.815 [0.540, 1.093] | 1020 / 1012 | 0.987 [0.929, 1.075] | 6.6 |
| b | 1,000,000 | 999/0 | 6 | 10 | 32.81 / 27.40 | 0.818 [0.610, 1.241] | 973 / 965 | 0.991 [0.972, 1.003] | 15.6 |

Worst median ratio: wall 0.885, peak 0.997. Gate 1.05: PASS.

The first run after an install, with an empty numba cache, is not gated.
A is pedsum #13, B is pedsum on pg-phenotype:

| config | Mating Pairs | threads | side | wall s | peak MiB |
|---|---:|---:|---|---:|---:|
| a | 10,000 | 1 | a | 24.73 | 498 |
| a | 10,000 | 1 | b | 1.94 | 206 |
| a | 10,000 | 6 | a | 35.39 | 311 |
| a | 10,000 | 6 | b | 1.21 | 102 |
| b | 10,000 | 1 | a | 39.01 | 398 |
| b | 10,000 | 1 | b | 1.07 | 105 |
| b | 10,000 | 6 | a | 67.53 | 397 |
| b | 10,000 | 6 | b | 1.87 | 106 |

pedsum #13 compiled its kernels on that run (25 to 68 s at 10^4 pairs);
pg-phenotype ships compiled.

## History

pg-phenotype 0.1.0 failed this gate: wall 1.090 at `b`, 10^6 pairs, 1 thread
(a slower single-threaded permutation pass) and peak 1.051 at `b`, 10^5
pairs, 6 threads (memory pedsum frees while parsing, which pg-phenotype's
pool threads could not reuse). 0.1.1 fixes both. pg-phenotype's
[benchmark report](https://github.com/rwaples/pg-phenotype/blob/v0.1.1/docs/gates/assortative-mating/benchmark.md#pg-phenotype-011-in-pedsums-cli)
has the 0.1.0 runs and a quieter run of the 0.1.1 code (worst median ratio:
wall 0.969, peak 0.996; load at most 12).
