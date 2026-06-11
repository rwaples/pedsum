# pedsum summarize — peak-RSS measurements

Rerunnable record for the measure-first memory work. Regenerate inputs and
re-run the profiler with the commands in `README.md`; raw per-run JSON lives in
`benchmarks/results/`.

- **Machine:** 12 cores, 30 GiB RAM, Linux. Sampler: `/proc/self/statm`, ~20 ms.
- **In-process profiler** (`profile_memory.py`): RSS is per-process, so it is
  immune to other processes; runtimes are secondary and may be noisy when cells
  run concurrently.
- **Inputs:** deterministic, `generate_pedigree.py --generations 8 --seed 0`
  (observed max depth 7, exercising `_A` … `_A5`). Streaming cells: 1,000,000
  rows. Matrix (`--per-individual-pairs`) cells: 200,000 rows — the largest size
  that completes comfortably; pair-list materialization is the memory-dangerous
  behavior being measured.
- Repeats: 1 warm-up + 2 measured (1M cells) / 3 measured (200K cells); peak RSS
  is the **median**, with observed range.

## Phase 0 baselines (no optimization applied)

Runtimes were measured with cells running concurrently, so treat them as
indicative; peak RSS (sampled per-process) is unaffected.

| Cell | Rows | Engine | F/Ne | Peak RSS (MiB, median [range]) | Peak phase | Runtime (s) |
|------|-----:|--------|------|-------------------------------:|------------|------------:|
| narrow/streaming | 1,000,000 | streaming | on | 2041.7 [2018.5–2064.9] | inbreeding (F + n_ancestors) | 45.5 |
| narrow/streaming (diag) | 1,000,000 | streaming | **off** | 1567.6 [1557.0–1578.3] | individual table built | 21.7 |
| wide/streaming | 1,000,000 | streaming | on | 2163.7 [2133.4–2193.9] | inbreeding (F + n_ancestors) | 62.1 |
| narrow/matrix | 200,000 | matrix | on | 6130.4 [5780.3–6346.7] | relationship pairs (extraction) | 41.8 |
| wide/matrix | 200,000 | matrix | on | 6136.5 [6020.6–6263.9] | relationship pairs (extraction) | 35.5 |

### matrix per-phase peak (narrow, baseline)

```
6130.4 MiB  relationship pairs            <- overall peak (extract_pairs; self-releases _A..A5 internally)
5387.5 MiB  relationship burden summary   <- holds _pair_lists + compute transients
3353.2 MiB  inbreeding / effective size / individual table / write  <- post-burden floor (still holds _pair_lists)
1120.8 MiB  build PedigreeGraph / size+structure / sibship / mating
```

**Idea 2 target:** `_pair_lists` persist in the ~3353 MiB post-burden floor
through inbreeding / Ne / write. Dropping them after the burden summary lowers
that floor. It does **not** lower the overall matrix peak (6130 MiB during
extraction, which Idea 1 leaves alone because the matrix engine already releases
`_A…_A5` internally).

### narrow/streaming per-phase peak (baseline)

```
2041.7 MiB  inbreeding (F + n_ancestors)   <- overall peak
1570.2 MiB  individual table built
1490.2 MiB  (unprofiled)
1490.2 MiB  aggregate pedigree sections
1466.3 MiB  wrote annotated.tsv.gz
1106.3 MiB  relationship pair counts (count_pairs_streaming)
1091.1 MiB  effective size (7 estimators)
 843.5 MiB  load+validate
 748.0 MiB  built PedigreeGraph / size+structure / sibship / mating-pair
```

**Key observation (Idea 1 target):** the overall peak is during the
**inbreeding** phase, *after* `count_pairs_streaming`. The `_A … _A5` adjacency
powers built by the streaming counter are never released, so they stay resident
through inbreeding / Ne / individual-table / write. Releasing them right after
the streaming block should lower the peaks of every later phase.

## Tier 1 before/after

Staged for attribution: each row is measured against the immediately preceding
state. Runtime medians here are from single (uncontended) profiler runs.

### Idea 1 — release `_A … _A5` after `count_pairs_streaming` (streaming path)

| Cell | Peak RSS before → after (MiB) | Δ | Runtime (s) |
|------|------------------------------:|----:|------------:|
| narrow/streaming | 2041.7 → 1625.9 | **−415.8 (−20.4%)** | 45.5 → 45.4 |
| wide/streaming | 2163.7 → 1644.7 | **−519.0 (−24.0%)** | 62.1 → 53.0 |

Both streaming cells confirm Idea 1: the overall peak (during the inbreeding
phase, after the matrices are released) drops 415–519 MiB. The overall peak is a
stable large allocation, so it is insensitive to the CPU-contention differences
between the (concurrent) baselines and the (single) after-runs; per-phase
transients sampled under different contention are noisier and not over-read.

narrow/streaming per-phase after Idea 1 (every post-streaming phase drops
~360–416 MiB; the streaming counter phase and total runtime are unchanged):

```
1625.9 MiB  inbreeding (F + n_ancestors)     (was 2041.7; −416)
1213.9 MiB  individual table built           (was 1570.2; −356)
1128.7 MiB  aggregate pedigree sections      (was 1490.2; −361)
1105.9 MiB  wrote annotated.tsv.gz           (was 1466.3; −360)
1108.0 MiB  relationship pair counts (streaming)  (was 1106.3; unchanged — released after)
```

The profiler flagged two <0.2 s phases (size+structure, sibship sizes) as >20%
slower; these are sampling jitter on trivial phases, not real regressions (total
runtime is flat). Matrix path: Idea 1 is a no-op (extract_pairs already releases
`_A…_A5` internally), confirmed by the unchanged matrix-cell behavior.

### Idea 2 — drop `pairs["_pair_lists"]` after the burden summary (matrix path)

Idea 2 frees the materialised pair lists from the **post-burden resident floor**
(inbreeding / Ne / individual-table / write). It does **not** lower the overall
matrix peak, which is during `extract_pairs` (before the pop). before = baseline
(no Idea 2); after = current tree.

| Cell | Post-burden floor before → after (MiB) | Δ floor | Overall peak before → after |
|------|---------------------------------------:|--------:|----------------------------:|
| narrow/matrix 200K | 3353.2 → 1384.2 | **−1969 (−59%)** | 6130.4 → 6083.9 (unchanged) |
| wide/matrix 200K | 3467.9 → 1208.3 | **−2259 (−65%)** | 6136.5 → 6213.5 (unchanged, noise) |
| narrow/matrix 400K | 4572.4 → 1774.7 | **−2797.7 (−61%)** | 11847.5 → 11726.8 (unchanged) |

Post-burden floor = peak RSS of the inbreeding phase (first phase after the
pop). The matrix peak (extraction) is unchanged because Idea 1 leaves the matrix
engine's internal release alone and Idea 2 acts only after extraction.

### Idea 7a — lazy first-row dicts (read/validate phase, both paths)

The `mother_first_row` / `father_first_row` dicts are now built only when an
ambiguous unsexed row exists. The synthetic inputs are fully sexed, so the dicts
become empty and the `load+validate` phase peak drops. Isolated on
narrow/streaming (7a is the only change vs the Idea-1-only run):

| Phase | Idea-1 only | + Idea 7a | Δ |
|-------|------------:|----------:|----:|
| load+validate peak (MiB) | 867.5 | 675.1 | **−192** |

Like Idea 2, this lowers a phase floor, not the overall peak (the dicts are
freed when `load_and_validate` returns, before the inbreeding-phase peak). It is
a safe, local saving that matters most when read/validate is the binding phase.

## Tier 1 summary

| Idea | Effect | Magnitude |
|------|--------|-----------|
| 1 — release `_A…_A5` after streaming | **overall peak** (streaming) | −416 MiB / −20% (narrow), −519 MiB / −24% (wide) |
| 2 — drop `_pair_lists` after burden | post-burden **floor** (matrix) | −1969 MiB (200K), −2798 MiB (400K), ~−61% |
| 7a — lazy first-row dicts | read/validate **floor** | −192 MiB (1M streaming) |

Correctness: all 127 pedsum tests pass; semantic output parity (HEAD vs tree)
holds for summarize on both engines + example_pedigree, and validate.log is
byte-identical on an ambiguous-sex input (7a preserves row-X/row-Y detail).

Where the overall peak now lives after Tier 1:

- **Streaming:** the `inbreeding (F + n_ancestors)` phase (~1626 MiB narrow) —
  inside pedigree-graph's Meuwissen–Luo / ancestor kernels.
- **Matrix:** the `extract_pairs` phase (6.1 GiB @200K, 11.7 GiB @400K) — also
  inside pedigree-graph.

Both are **out of pedsum-only scope** (no upstream `pedigree-graph` edits).

## Real-data check: the 783K stallion-heavy horse pedigree

The synthetic cells above deliberately avoid pathological pair density. The real
worst case is the horse-breed pedigree in `external/pedigrees/horse/horse.fixed.tsv`
(783,029 individuals; top sire 516736 has 2,593 offspring — the "all-time-great
stallion" of `pedigree-graph`'s `LIMITATIONS.md`). The matrix engine OOMs on it
(C(50K,2) candidate pairs through one stallion grandparent); the **streaming**
path is the only viable one, and it is exactly what Idea 1 optimises.

Command (not bundled in pedsum; columns map FoalID/Sex/Dam/Sire/BirthYear):

```bash
python benchmarks/profile_memory.py --label horse/streaming/full --repeats 2 --warmup 1 \
  -- summarize --in .../horse/horse.fixed.tsv --out /tmp/horse \
     --id-col FoalID --sex-col Sex --mother-col Dam --father-col Sire --birth-year-col BirthYear
```

Idea 1 before/after (matched 2-repeat runs; F is cheap here, mean F≈0.007, so the
peak is the individual-table phase, not inbreeding):

| Cell | Peak RSS before → after (MiB) | Δ | Runtime (s) |
|------|------------------------------:|----:|------------:|
| horse/streaming, full (F + 7 Ne) | 1103.2 → 963.0 | **−140.2 (−12.7%)** | 18.1 → 18.9 |
| horse/streaming, no F/Ne (diag) | 1096.4 → 947.7 | **−148.7 (−13.6%)** | 15.4 → 13.9 |

Full-run per-phase drop (before → after): inbreeding 918.0→756.6, individual
table 1103.2→963.0, write 1045.0→905.0, effective size 798.4→620.5 — every
post-streaming phase falls ~140–180 MiB as the cached `_A…_A5` are released. The
whole default summarize finishes in ~19 s at <1 GiB peak on the pedigree both
non-streaming engines OOM on. Confirms Idea 1 generalises to the real stallion
worst case (smaller absolute matrices than the synthetic 1M cell, ~150 MiB).

## Tier 2 / Tier 3 stop-go review

Gate (from the plan): promote only if the target accounts for ≥10% of peak RSS
**or** has a credible expected saving of ≥50 MiB on the representative input.

| Item | Targeted allocation (measured / estimated) | Reduces overall peak? | Verdict |
|------|---------------------------------------------|-----------------------|---------|
| 3 — `usecols` on summarize read | `df_raw` extra cols ≈ **316 MiB** in the read phase on wide (8 cols) | No — read phase (~990 MiB) sits below the inbreeding peak (~1645) | **Judgment call** (see below) |
| 7b — `sex_source` int8 enum | object array = N pointers to ~4 interned strings ≈ **8 MiB** at 1M rows | No | **Drop** — far below the 50 MiB gate |
| 7c — adaptive int32 IDs/parents | id+mother+father int64 ≈ 24 MiB; saving ≈ **12 MiB**, and originals are transient (remapped to compact int64 in PedigreeGraph) | No | **Drop** — below gate |
| 6 — avoid `idf` materialization (Tier 3) | `individual table built` phase = 1214 MiB, **below** the inbreeding peak | No — `idf` does not dominate; F does | **Defer** — gate not met |

**Recommendation: stop after Tier 1.** The Tier-1 changes captured the
meaningful pedsum-only wins. The remaining overall peak lives in pedigree-graph
kernels (F for streaming, `extract_pairs` for matrix), which are out of the
pedsum-only scope. Of the Tier-2 items only **Idea 3** touches a >50 MiB
allocation, and it lowers a transient read-phase floor rather than the headline
peak — worth doing only if lower read-phase steady-state RAM on wide inputs is
itself a goal. 7b and 7c are below the gate; Idea 6 does not meet it.
