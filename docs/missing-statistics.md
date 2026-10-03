# Pedigree statistics pedsum does not report

This is a gap analysis against the statistics that published pedigree software
reports: ENDOG, CFC, optiSel, purgeR, visPedigree, GENLIB, PMx/nprcgenekeepr
and PyPedal. It is also against the statistics that the candidate benchmark
pedigrees (`docs/benchmark-pedigrees.md`) publish. It was compiled
2026-10-03 against pedsum `6f71852` and the pedigree-graph checkout next to
it.

**Already covered, and so not listed below:**
- N, sexes, founders and half-founders, and depth (`ped_depth`).
- Per-individual F (Meuwissen–Luo), with mean F by depth and an F histogram.
- Sibship and offspring-count distributions, mate counts, and components.
- Relationship-pair counts through degree 5, with per-pair exact kinship via
  `epimight-input --pairs --exact-kinship`.
- Distinct ancestors, descendant paths, and founder ancestors.
- `effective_founders_by_descendant_paths` / `_by_descendants`.
- The eight Ne estimators in `effective_size.yaml`.

Citations were checked as described in `docs/benchmark-pedigrees.md`: web
search restricted to publisher and PubMed domains, with doi.org and Crossref
blocked. Fields that could not be seen are marked `TODO: verify`.

## A. Pedigree depth and completeness

| # | Statistic | Who reports it | pedsum today | Notes |
|---|---|---|---|---|
| A1 | **Equivalent complete generations** (ECG, t), per individual and summarised | ENDOG, CFC, optiSel (`equiGen`), purgeR (`pop_t`), visPedigree | Computed inside pedigree-graph (`_ne_rates._equivalent_generations`) for `ne_individual_delta_f`, but never written out | Cheapest gap to close: add an `ecg` column to `annotated.tsv.gz` and its distribution by depth. The benchmarks need it: the purgeR atlas sum is 5500.627, and optiSel gives it for 6 named IDs. |
| A2 | **Complete generations** (fully traced generations) per individual | ENDOG, optiSel (`fullGen`), the bear paper (mean 1.30) | Missing | `ped_depth` is the *maximum* generations traced (`maxGen`), so only the fully-traced count is missing. |
| A3 | **Pedigree completeness by ancestral generation** (share of the 2^g ancestors known) | ENDOG, GENLIB (`gc`), optiSel | Missing | Reported per reference cohort. |
| A4 | **Pedigree completeness index** (MacCluer PCI) | optiSel (`PCI`), ENDOG | Missing | Per individual; needs a generation cutoff `d`. |
| A5 | **Expected genealogical depth** (Cazes) | GENLIB (`depth`); genea140 mean 9.4 | Missing | Computed over probands. |

## B. Founder and ancestor contributions

| # | Statistic | Who reports it | pedsum today | Notes |
|---|---|---|---|---|
| B1 | **Expected genetic contribution of each founder to a reference population** (gene-flow p_i), and **founder equivalents** fe = 1/Σp_i² (Lacy) | ENDOG, CFC, PMx, purgeR (`Nfe`), visPedigree (`fe`), optiSel | pedsum's Effective Founders are weighted by descendant paths or descendant counts, which is a **different** statistic, not Lacy's fe | pedigree-graph already propagates per-cohort mean founder-genome contributions (`_ne_founders._per_gen_founder_means`) for `ne_long_term_contributions`. A reference-population variant would give p_i directly. A convention is needed for half-founders, whose contributions do not sum to 1: in `purgeR::dorcas`, raw p gives 15.20 and renormalised p gives 13.39. |
| B2 | **Effective number of ancestors** fa, from marginal contributions (Boichard 1997) | ENDOG, CFC, purgeR (`Nae`), visPedigree (`fa`), PyPedal | Missing | Iterative; the number of ancestors explaining 50% of the genes is usually reported with it. |
| B3 | **Founder genome equivalents** fg (Lacy 1989), by gene dropping or as 1/(2·mean kinship) | PMx, purgeR (`Ng`), visPedigree (`fg`), nprcgenekeepr, ENDOG | Missing | The kinship form is deterministic; the gene-drop form needs a seed. |
| B4 | **Genetic conservation index** (Alderson) | ENDOG | Missing | Low priority. No citable defining source has been confirmed yet. |

## C. Kinship, inbreeding decomposition and purging

| # | Statistic | Who reports it | pedsum today | Notes |
|---|---|---|---|---|
| C1 | **Mean kinship (coancestry)** in a reference population, and **per-individual mean kinship** (MK) | optiSel, PMx, ENDOG, CFC, visPedigree (`MeanCoan`), GENLIB (`phi.mean`) | Per-pair only (`--pairs --exact-kinship`); `ne_coancestry` uses it internally | MK is the core statistic for conservation management. GENLIB publishes mean kinship among genea140 probands per subpopulation. |
| C2 | **Partial inbreeding** (the part of F due to each founder or ancestor) | purgeR, visPedigree, GRAIN | Missing | Exact recursion (Lacy, Alaks & Walsh 1996) or gene dropping. |
| C3 | **Ancestral inbreeding**: Ballou's F_ANC, Kalinowski's F_ANC/F_NEW split, and Baumung's ancestral history coefficient | purgeR, GRAIN | Missing | Kalinowski needs gene dropping. |
| C4 | **Purging statistics**: opportunity of purging (Gulisija & Crow), purged inbreeding g (García-Dorado) | purgeR | Missing | Niche; only if the benchmarks call for it. |
| C5 | **Non-random mating / Hardy–Weinberg deviation** α, and **subpopulation F-statistics** from coancestry (Caballero & Toro) | ENDOG, purgeR (`pop_hwd`) | Missing | Needs a subpopulation column for the F_IS/F_ST partition. |

## D. Effective size and demography

| # | Statistic | Who reports it | pedsum today | Notes |
|---|---|---|---|---|
| D1 | **Reference-population selection for Ne** | ENDOG, purgeR (`target`), optiSel (`Phen`), visPedigree | pedigree-graph `ne_individual_delta_f(reference=...)` supports it; the pedsum CLI does not expose it | Needed for exact comparison with purgeR (8.18 for all rows vs 14.01 for target) and with optiSel (Ne_C 97.96 over `Phen` with equiGen ≥ 4). Proposed: `--reference-col` or `--reference-ids FILE`. |
| D2 | **Ne_iΔF eligibility convention** | purgeR counts rows with t ≤ 1 as ΔF = 0 (atlas: 8.18); pedigree-graph requires t > 0 (atlas: 8.03) | Different convention, documented in pedigree-graph | Either add an option, or document the expected difference in the benchmark harness. |
| D3 | **Generation interval** by the four pathways (sire→son, sire→daughter, dam→son, dam→daughter) | ENDOG, optiSel, CFC | pedigree-graph has `PedigreeGraph.generation_interval` (Hill 1979, sex split); pedsum does not report it as its own statistic | Needs birth years; a natural addition to `effective-size` output when `--birth-year-col` is given. |
| D4 | **Ne from regressing F or ECG on birth year** (realised ΔF per year scaled by L) | ENDOG, Leroy et al. 2013 | Missing | Needs D3. |
| D5 | **Fraction of genes from a founder group** (migrant vs native contributions, native Ne) | optiSel | Missing | Needs a founder-group label column. |

## E. Input and tooling gaps found while surveying

- **Format conversion:** GEDCOM, `.rda`, `animal/dam/sire` tables, string IDs, `0` as missing, and Excel. Drafted as a separate issue.
- **Benchmark harness:** a script under `benchmarks/` that fetches the Tier-1 pedigrees, runs `summarize` and `effective-size`, and checks pedsum's values against the published ones within stated tolerances.

## Suggested order

1. **A1** ECG output and **D1** reference population. pedigree-graph already does both; this is just wiring.
2. **B1** Lacy fe with an explicit half-founder convention, **C1** mean kinship / MK, and **B3** fg in its kinship form.
3. **A2–A4** completeness, **B2** fa, and **D3** generation interval.
4. **C2–C5** and **D4–D5** on demand.

## References

- Ballou JD (1997). Ancestral inbreeding only minimally affects inbreeding depression in mammalian populations. *J Hered* 88(3):169–178. https://doi.org/10.1093/oxfordjournals.jhered.a023085
- Baumung R, Farkas J, Boichard D, Mészáros G, Sölkner J, Curik I (2015). GRAIN: a computer program to calculate ancestral and partial inbreeding coefficients using a gene dropping approach. *J Anim Breed Genet* 132 (TODO: verify issue):100–108 (TODO: verify author list and pages). https://doi.org/10.1111/jbg.12145
- Boichard D, Maignel L, Verrier E (1997). The value of using probabilities of gene origin to measure genetic variability in a population. *Genet Sel Evol* 29(1):5–23. https://doi.org/10.1186/1297-9686-29-1-5
- Caballero A, Toro MA (2000). Interrelations between effective population size and other pedigree tools for the management of conserved populations. *Genet Res* 75(3):331–343. https://doi.org/10.1017/S0016672399004449
- Caballero A, Toro MA (2002). Analysis of genetic diversity for the management of conserved subdivided populations. *Conserv Genet* 3 (TODO: verify issue):289–299. https://doi.org/10.1023/A:1019956205473
- Cazes P, Cazes MH (1996). Comment mesurer la profondeur généalogique d'une ascendance ? *Population* 51(1):117–140. TODO: verify DOI (Persée pop_0032-4663_1996_num_51_1_6119).
- Cervantes I, Goyache F, Molina A, Valera M, Gutiérrez JP (2011). Estimation of effective population size from the rate of coancestry in pedigreed populations. *J Anim Breed Genet* 128(1):56–63. https://doi.org/10.1111/j.1439-0388.2010.00881.x
- García-Dorado A (2012). Understanding and predicting the fitness decline of shrunk populations: inbreeding, purging, mutation, and standard selection. *Genetics* 190(4):1461–1476. TODO: verify DOI.
- Gulisija D, Crow JF (2007). Inferring purging from pedigree data. *Evolution* 61(5):1043–TODO: verify end page. https://doi.org/10.1111/j.1558-5646.2007.00088.x
- Gutiérrez JP, Cervantes I, Molina A, Valera M, Goyache F (2008). Individual increase in inbreeding allows estimating effective sizes from pedigrees. *Genet Sel Evol* 40(4):359–378. https://doi.org/10.1051/gse:2008008 (TODO: verify the Springer-form DOI 10.1186/1297-9686-40-4-359; the two checks disagreed)
- Gutiérrez JP, Cervantes I, Goyache F (2009). Improving the estimation of realized effective population sizes in farm animals. *J Anim Breed Genet* 126(4):327–332. https://doi.org/10.1111/j.1439-0388.2009.00810.x
- Hill WG (1979). A note on effective population size with overlapping generations. *Genetics* 92(1):317–322. https://doi.org/10.1093/genetics/92.1.317 (TODO: verify DOI on publisher page)
- James JW (1977). A note on selection differential and generation length when generations overlap. *Anim Prod* 24(1):109–112. TODO: verify DOI.
- Kalinowski ST, Hedrick PW, Miller PS (2000). Inbreeding depression in the Speke's gazelle captive breeding program. *Conserv Biol* 14 (TODO: verify issue):1375–1384. https://doi.org/10.1046/j.1523-1739.2000.98209.x
- Lacy RC (1989). Analysis of founder representation in pedigrees: founder equivalents and founder genome equivalents. *Zoo Biol* 8(2):111–123. https://doi.org/10.1002/zoo.1430080203
- Lacy RC, Alaks G, Walsh A (1996). Hierarchical analysis of inbreeding depression in *Peromyscus polionotus*. *Evolution* 50(6):2187–2200. TODO: verify DOI.
- Leroy G, Mary-Huard T, Verrier E, Danvy S, Charvolin E, Danchin-Burge C (2013). Methods to estimate effective population size using pedigree data: Examples in dog, sheep, cattle and horse. *Genet Sel Evol* 45:1. https://doi.org/10.1186/1297-9686-45-1
- MacCluer JW, TODO: verify remaining authors (1983). Inbreeding and pedigree structure in Standardbred horses. *J Hered* 74(6):394–399. https://doi.org/10.1093/oxfordjournals.jhered.a109824
- Maignel L, Boichard D, Verrier E (1996). Genetic variability of French dairy breeds estimated from pedigree information. *Interbull Bulletin* TODO: verify volume, pages and author order. https://journal.interbull.org/index.php/ib/article/view/523/523
- Meuwissen THE, Luo Z (1992). Computing inbreeding coefficients in large populations. *Genet Sel Evol* 24(4):305–313. https://doi.org/10.1186/1297-9686-24-4-305
- Woolliams JA, Bijma P, Villanueva B (1999). Expected genetic contributions and their impact on gene flow and genetic gain. *Genetics* 153(2):1009–1020. https://doi.org/10.1093/genetics/153.2.1009
- Software: ENDOG (Gutiérrez & Goyache 2005, https://doi.org/10.1111/j.1439-0388.2005.00512.x); optiSel (Wellmann 2019, https://doi.org/10.1186/s12859-018-2450-5); purgeR (López-Cortegano 2022, https://doi.org/10.1093/bioinformatics/btab599); GENLIB (Gauvin et al. 2015, https://doi.org/10.1186/s12859-015-0581-5); visPedigree (Luan et al. 2026, *Bioinformatics Advances* 6(1):vbag210, TODO: verify DOI); CFC (Sargolzaei, Iwaisaki & Colleau 2006, TODO: verify whole entry).
