# Candidate benchmark pedigrees

A survey of pedigrees that pedsum could be run on and compared against
published statistics. Compiled 2026-10-03.

## How the citations were checked

The cloud sandbox used for this survey blocks doi.org, Crossref, PubMed,
Europe PMC, OpenAlex, Dryad and publisher pages. Citations were checked
through web-search results restricted to publisher, PubMed/PMC or Dryad
domains, and through the R packages' own `DESCRIPTION`/`CITATION` files on
the github.com/cran mirror. A field is given only where it was seen in such a
source. Everything else is written `TODO: verify <field>`. Before quoting any
of these, resolve the DOI on a machine with normal network access.

"Status" columns:

- **fetched**: the file was downloaded and parsed during the survey.
- **listed**: the deposit was found in search results but not downloaded.
- **blocked**: the host was unreachable from the sandbox; not checked.
- **restricted**: the data are not publicly downloadable.

Numbers marked *recomputed* were reproduced from the downloaded file with a
throwaway script (not pedsum). Others come from the source as quoted.

---

## 1. Exact numeric targets (downloadable; published values recomputed)

### 1.1 purgeR zoo studbooks: `atlas`, `dama`, `arrui`, `dorcas`, `darwin`

- **Data:** R package purgeR 1.8.3, `data(atlas)` etc.
  https://github.com/cran/purgeR (`data/*.rda`). GPL-2. Status: fetched.
- **Size:** atlas 948 (*Gazella cuvieri*), dama 1,316 (*Nanger dama*),
  arrui 380 (*Ammotragus lervia*), dorcas 1,279 (*Gazella dorcas*),
  darwin 63 (Darwin/Wedgwood family).
- **Published values:**
  - Heredity abstract: Ne = 14, 11, 4 and 39 from individual ΔF over
    `target == 1`. Recomputed as 14.01041, 11.09927, 3.83865 and 39.32125.
  - atlas package tests: total F 197.2809, max F 0.4277344, equivalent
    complete generations summing to 5500.627, individual-ΔF Ne 8.184803
    (all rows) and 14.01041 (target).
  - Vignette `pop_Nancestors`: founder equivalents Nfe
    1.769424 / 3.583086 / 13.386468 / 2.614750 (arrui / atlas / dorcas /
    dama). The first, second and fourth were recomputed.
  - darwin: F = 0.06298828 for William Erasmus Darwin.
- **Citations:**
  - López-Cortegano E, Moreno E, García-Dorado A (2021). Genetic purging in
    captive endangered ungulates with extremely low effective population
    sizes. *Heredity* 127(5):433–442. https://doi.org/10.1038/s41437-021-00473-2
  - López-Cortegano E (2022). purgeR: inbreeding and purging in pedigreed
    populations. *Bioinformatics* 38(2):564–565.
    https://doi.org/10.1093/bioinformatics/btab599

### 1.2 GENLIB `genea140` (Quebec, extract of the BALSAC register)

- **Data:** R package GENLIB, `data(genea140)`.
  https://raw.githubusercontent.com/cran/GENLIB/master/data/genea140.rda
  (`ind, father, mother, sex`; 0 = unknown parent). GPL. Status: fetched.
- **Size:** 41,523 individuals, 140 probands.
- **Published values, all recomputed:**
  - 20,773 males and 20,750 females.
  - 7,399 founders.
  - 21,230 nuclear families.
  - 5,994 full sibships, the largest of size 14.
  - 18 generations.
  - Mean genealogical depth 9.4.
- **Citations:**
  - Gauvin H, Lefebvre J-F, Moreau C, Lavoie E-M, Labuda D, Vézina H,
    Roy-Gagnon M-H (2015). GENLIB: an R package for the analysis of
    genealogical data. *BMC Bioinformatics* 16:160 (TODO: verify volume and
    article number). https://doi.org/10.1186/s12859-015-0581-5
  - Roy-Gagnon M-H, Moreau C, Bherer C, St-Onge P, Sinnett D, Laprise C,
    Vézina H, Labuda D (2011). Genomic and genealogical investigation of the
    French Canadian founder population structure. *Hum Genet*
    129(5):521–531. https://doi.org/10.1007/s00439-010-0945-x

### 1.3 visPedigree `deep_ped`

- **Data:** R package visPedigree 1.10.1, `data(deep_ped)`.
  https://github.com/cran/visPedigree. GPL-3. Status: fetched.
- **Size:** 4,396 rows; 4,399 individuals once missing parents are added.
- **Published values** (package vignette; the package's own output, not peer
  reviewed):
  - N 4,399, 483 sires, 554 dams, 138 founders, 13 generations. N and the
    sire and dam counts were recomputed.
  - fe 64.73344 and fa 44.12033.
  - Mean coancestry 0.0260693.
  - NeInbreeding 98.00425.
- **Citation:** Luan S, Kong J, Xia Z, Kang Z, Qiang G, Luo K, Sui J (2026).
  visPedigree: a comprehensive R package for tidying, analyzing, and
  visualizing breeding pedigrees. *Bioinformatics Advances* 6(1):vbag210.
  TODO: verify DOI.

### 1.4 optiSel `PedigWithErrors` / `Phen` (Hinterwald cattle)

- **Data:** R package optiSel 2.1.0. https://github.com/cran/optiSel. GPL-2.
  Status: fetched.
- **Size:** 10,863 rows; 178 candidates in `Phen`. The file contains
  deliberate errors that optiSel's `prePed()` repairs.
- **Published values** (vignette `ped-vignette.html`):
  - Mean F of the `Phen` animals 0.01943394.
  - Coancestry Ne 97.95922.
  - Per-animal equiGen, fullGen, maxGen, PCI and F for 6 named IDs.
- **Citation:** Wellmann R (2019). Optimum contribution selection for animal
  breeding and conservation: the R package optiSel. *BMC Bioinformatics*
  20:25. https://doi.org/10.1186/s12859-018-2450-5

### 1.5 Small fixtures

| Pedigree | Data | Published value | Citation |
|---|---|---|---|
| Mrode example | `nadiv::Mrode2`; `AGHmatrix::ped.mrode`; the pedigreemm `?getA` example. Status: fetched. | F = 0.125 for animals 5 and 6; textbook A matrix | Mrode RA (2005). *Linear Models for the Prediction of Animal Breeding Values*, 2nd ed. CABI Publishing (TODO: verify place). ISBN 0-85199-000-2. https://doi.org/10.1079/9780851990002.0000 · Wolak ME (2012). nadiv: an R package to create relatedness matrices for estimating non-additive genetic variances in animal models. *Methods Ecol Evol* 3(5):792–796. https://doi.org/10.1111/j.2041-210X.2012.00213.x |
| Lacy (1989) example | `nprcgenekeepr::lacy1989Ped`; 7 rows in `tests/testthat/test_calcFE.R`. Status: fetched. | fe = 2.91 (package test 2.9090909) | Lacy RC (1989). Analysis of founder representation in pedigrees: founder equivalents and founder genome equivalents. *Zoo Biol* 8(2):111–123. https://doi.org/10.1002/zoo.1430080203 |
| Boichard toy pedigrees | Must be typed in from the paper's figures. PyPedal's test suite asserts fe = 4.0 / 2.0 / 5.6 and fa = 2.0 / 2.0 / 2.94. Status: the `.ped` files are on SourceForge (blocked). | fe, fa | Boichard D, Maignel L, Verrier E (1997). The value of using probabilities of gene origin to measure genetic variability in a population. *Genet Sel Evol* 29(1):5–23. https://doi.org/10.1186/1297-9686-29-1-5 |
| MCMCglmm `BTped` (blue tit) | `data(BTped)`, 1,040 rows. Status: fetched. | 212 founders, 106 full-sib families, all F = 0 (by construction) | Hadfield JD, Nutall A, Osorio D, Owens IPF (2007). Testing the phenotypic gambit: phenotypic, genetic and environmental correlations of colour. *J Evol Biol* 20(2):549–557. https://doi.org/10.1111/j.1420-9101.2006.01262.x · Hadfield JD (2010). MCMC methods for multi-response generalized linear mixed models: the MCMCglmm R package. *J Stat Softw* 33(2):1–22. TODO: verify DOI. |
| nprcgenekeepr unit tests | `tests/testthat/`. Status: fetched. | Ne by sex ratio 6 / 3.6 / 2; Ne by variance 1.875 / 2.5; fg = 32/21 | nprcgenekeepr (CRAN package, v2.0.0), Raboin M, Therneau T, Vinson A, Sharp RM, Schultz M. *Genetic Tools for Colony Management*. TODO: verify year. No paper. |
| visPedigree `small_ped`, `inbred_ped` | In the package. Status: fetched. | fe 6.585209 and fa 2.666667; mean F by generation 0, 0, 0.25, 0.25, 0.4375 | As 1.3 |

---

## 2. Approximate or partial targets (downloadable)

### 2.1 Human and historical

| Pedigree | Data, status | N | Comparable values | Citation |
|---|---|---|---|---|
| Spanish Habsburgs | No published data file. Stand-in GEDCOM: https://raw.githubusercontent.com/jlmborges/VisAC/main/R_package/Charles_II_2022_04.ged (MIT). Status: fetched. | 198 (stand-in) | Published F: Philip I 0.025, Charles I 0.037, Philip II 0.123, Philip III 0.218, Philip IV 0.115, Charles II 0.254. Don Carlos 0.211 is TODO: verify. The stand-in gives lower values (Charles II 0.238) because it is shallower. | Alvarez G, Ceballos FC, Quinteiro C (2009). The role of inbreeding in the extinction of a European royal dynasty. *PLoS ONE* 4(4):e5174 (TODO: verify vol/issue). https://doi.org/10.1371/journal.pone.0005174 · Ceballos FC, Álvarez G (2013). Royal dynasties as human inbreeding laboratories: the Habsburgs. *Heredity* 111(2):114–121. https://doi.org/10.1038/hdy.2013.25 |
| Darwin–Wedgwood | `purgeR::darwin` (1.1). GEDCOM stand-in: https://raw.githubusercontent.com/samoilev/swarm/main/Examples/darwin-wedgwood/darwin-wedgwood.ged (40 individuals; gives 0.0625 because the deeper link is missing). Status: fetched. | 63 | Darwin's children F = 0.0630 | Berra TM, Alvarez G, Ceballos FC (2010). Was the Darwin/Wedgwood dynasty adversely affected by consanguinity? *BioScience* 60(5):376–383. https://doi.org/10.1525/bio.2010.60.5.7 |
| royal92.ged (European royalty, Denis R. Reid 1992) | https://raw.githubusercontent.com/R-Computing-Lab/BGmisc/main/data-raw/royal92.ged, and as CSV: `.../royal92.csv`. Status: fetched. | 3,010 | No published pedigree statistics; use as a regression baseline. Recomputed: 992 founders, mean F 0.0022, max F 0.1875. | No pedigree-statistics paper. Used as one network in: Boyd ZM, Callor N, Gledhill T, Jenkins A, Snellman R, Webb B, Wonnacott R (2023). The persistent homology of genealogical networks. *Applied Network Science* (TODO: verify volume and article number). https://doi.org/10.1007/s41109-023-00538-7 |
| kinship2 `minnbreast` (Minnesota breast cancer families) | https://raw.githubusercontent.com/cran/kinship2/master/data/minnbreast.rda, plus `sample.ped.tab.gz` (55 rows). Status: fetched. | 28,081 | Package docs: 426 families. The paper describes 544 families, which does not match the packaged subset, so there is no reproduction target. | Sinnwell JP, Therneau TM, Schaid DJ (2014). The kinship2 R package for pedigree data. *Hum Hered* 78(2):91–93. https://doi.org/10.1159/000363105 · Sellers TA, Anderson VE, Potter JD, Bartow SA, Chen PL, Everson L, King RA, Kuni CC, Kushi LH, McGovern PG, et al. (TODO: verify remaining authors) (1995). Epidemiologic and genetic follow-up study of 544 Minnesota breast cancer families: design and methods. *Genet Epidemiol* 12(4):417–429. https://doi.org/10.1002/gepi.1370120409 |
| Ptolemaic dynasty | https://raw.githubusercontent.com/D-Jeffrey/gedcom-samples/main/famous%20family%20trees/royalty/Ptolemaic+Dynasty.ged (MIT). Status: fetched. | 38 | No published F. Extreme-inbreeding stress test only (recomputed: Ptolemy XII F = 0.426). | None. |
| GENLIB `geneaJi` / ribd `jicaque` (Jicaque, Honduras) | https://raw.githubusercontent.com/cran/GENLIB/master/data/geneaJi.rda. Status: fetched. | 29 (GENLIB) / 22 (ribd) | Described as a "modified version" of the original, so published values are not expected to match. | Chapman AM, Jacquard A (1971). Un isolat d'Amérique Centrale: les Indiens Jicaques de Honduras. In: *Génétique et Population*. Paris: Presses Universitaires de France (TODO: verify editors and pages). |

### 2.2 Wild populations

| Population | Data, status | N | Comparable values | Citation |
|---|---|---|---|---|
| Soay sheep, St Kilda (current) | https://raw.githubusercontent.com/eamittell/PriorViabilitySelectionSoaySheep/main/SheepPedigreeRandomIds_May2025.csv (`animal,dam,sire`, no sex). GPL-3. Status: fetched. | 14,334 | No statistics published for this version. Recomputed: 3,715 founders, 3,349 half-founders, maximum depth 17. | Mittell EA, Pemberton JM, Kruuk LEB, Morrissey MB (2025). Unmeasured prior viability selection resolves the paradox of stasis for body size in wild Soay sheep. *PNAS* 122(48) (TODO: verify article number). https://doi.org/10.1073/pnas.2513969122 |
| Soay sheep (Bérénos pedigrees 1 and 2) | Dryad https://doi.org/10.5061/dryad.367s2 (`pedigree1dryad.txt`, `pedigree2dryad.txt`). Status: listed. | about 5,500 | Pedigree 2: 5,516 individuals, 4,531 maternal and 4,158 paternal links, as cited by Johnston 2016 (TODO: verify). A later version: 6,740 individuals and a depth of 10 generations (TODO: verify). | Bérénos C, Ellis PA, Pilkington JG, Pemberton JM (2014). Estimating quantitative genetic parameters in wild populations: a comparison of pedigree and genomic approaches. *Mol Ecol* 23 (TODO: verify issue):3434–3451. https://doi.org/10.1111/mec.12827 · Johnston SE, Bérénos C, Slate J, Pemberton JM (2016). Conserved genetic architecture underlying individual recombination rate variation in a wild population of Soay sheep (*Ovis aries*). *Genetics* 203(1):583–598. TODO: verify DOI. · James C, Pemberton JM, Navarro P, Knott S (2024). Investigating pedigree- and SNP-associated components of heritability in a wild population of Soay sheep. *Heredity* 132(4):202–210. https://doi.org/10.1038/s41437-024-00673-6 |
| Rum red deer | Figshare, `ped_anon.txt`. TODO: verify DOI 10.6084/m9.figshare.25053941. Status: listed. | about 3,900 | Older pedigree: 22% have F > 0 when both parents and at least one grandparent are known, and 42% when all four grandparents are known. The more precise figures 21.9% and mean F 0.00724 are TODO: verify. | Hewett AM, Johnston SE, Morris A, Morris S, Pemberton JM (2024). Genetic architecture of inbreeding depression may explain its persistence in a population of wild red deer. *Mol Ecol* 33(9):e17335. https://doi.org/10.1111/mec.17335 · Walling CA, Nussey DH, Morris A, Clutton-Brock TH, Kruuk LEB, Pemberton JM (2011). Inbreeding depression in red deer calves. *BMC Evol Biol* 11:318. https://doi.org/10.1186/1471-2148-11-318 |
| Mandarte song sparrow | Dryad https://doi.org/10.5061/dryad.p7p1jb3 and https://doi.org/10.5061/dryad.p9s04 (per-individual F). Also the GitHub repo `matthewwolak/Wolak_etal_SongSparrowFitnessQG`, described as "a restricted subset". Status: listed. | about 2,800+ | 26 immigrants; per-individual F; "mean f ≈ 0.06 (range 0–0.31)" (TODO: verify source) | Wolak ME, Arcese P, Keller LF, Nietlisbach P, Reid JM (2018). Sex‐specific additive genetic variances and correlations for fitness in a song sparrow (*Melospiza melodia*) population subject to natural immigration and inbreeding. *Evolution* 72(10):2057–2075. https://doi.org/10.1111/evo.13575 · Nietlisbach P, Keller LF, Camenisch G, Guillaume F, Arcese P, Reid JM, Postma E (2017). Pedigree-based inbreeding coefficient explains more variation in fitness than heterozygosity at 160 microsatellites in a wild bird population. *Proc R Soc B* 284(1850):20162763. https://doi.org/10.1098/rspb.2016.2763 |
| Helgeland house sparrow | Dryad https://doi.org/10.5061/dryad.m0cfxpp10 (`pedigree.txt`). Status: listed. | about 3,100 | F computed with the R package `pedigree`; n = 1,241 with at least 2 complete generations | Niskanen AK, Billing AM, Holand H, Hagen IJ, Araya-Ajoy YG, Husby A, Rønning B, Myhre AM, Ranke PS, Kvalnes T, Pärn H, Ringsby TH, Lien S, Sæther B-E, Muff S, Jensen H (2020). Consistent scaling of inbreeding depression in space and time in a house sparrow metapopulation. *PNAS* 117(25):14584–14592. https://doi.org/10.1073/pnas.1909599117 |
| Chatham Island black robin | Dryad https://doi.org/10.5061/dryad.81tg6 (`DatasetS1.csv`: parentage, sex, F). Status: listed. | hundreds | Per-individual F, 0.25–0.65 | Weiser EL, Grueber CE, Kennedy ES, Jamieson IG (2016). Unexpected positive and negative effects of continuing inbreeding in one of the world's most inbred wild animals. *Evolution* 70(1):154–166. https://doi.org/10.1111/evo.12840 · Kennedy ES, Grueber CE, Duncan RP, Jamieson IG (2014). Severe inbreeding depression and no evidence of purging in an extremely inbred wild species—the Chatham Island black robin. *Evolution* 68(4):987–995. TODO: verify DOI (10.1111/evo.12315). |
| Pyrenean brown bear | Dryad https://doi.org/10.5061/dryad.w9ghx3g1s (litter-level table with `Inbreeding_individual`). Status: listed. | about 100–150 | Depth 0–6 generations; mean 1.30 complete generations; mean 3.11 maximum generations; F 0–0.375 | Auclair L, Vanpé C, Chapron G, Quenette P-Y, Robert A (2025). Inbreeding depression across multiple life-history traits in a long-lived mammal. *Mol Ecol* 34 (TODO: verify issue):e70123. https://doi.org/10.1111/mec.70123 |
| Florida scrub-jay | Dryad https://doi.org/10.5061/dryad.z612jm6j0 (sheet `Core_Region_pedigree`: `ID, SIRE_ID, DAM_ID`). Status: listed. | hundreds | One breeding pair accounts for about 24% of expected genetic contributions since 2008 | Linderoth T, Deaner L, Chen N, Bowman R, Boughton RK, Fitzpatrick SW (2025). Translocations spur population growth but fail to prevent genetic erosion in imperiled Florida Scrub-Jays. *Curr Biol* 35(6):1391–1399.e6. TODO: verify DOI (10.1016/j.cub.2025.01.058). |
| Southern Resident killer whales | https://raw.githubusercontent.com/noaa-nwfsc/srkw-status/main/data/orca.rda (mothers only, no father column). GPL-3. Status: fetched. | 227 | Published: 4 inbred offspring; two males sired 52% of sampled offspring. The sires are not in the file. | Ford MJ, Parsons KM, Ward EJ, Hempelmann JA, Emmons CK, Hanson MB, Balcomb KC, Park LK (2018). Inbreeding in an endangered killer whale population. *Anim Conserv* 21(5):423–432. https://doi.org/10.1111/acv.12413 |
| Seychelles warbler | Dryad https://doi.org/10.5061/dryad.vt4b8gtr1 (pedigree with README). Status: listed. | — | 10-generation pedigree | Sparks AM, Spurgin LG, van der Velde M, Fairfield EA, Komdeur J, Burke T, Richardson DS, Dugdale HL (2022). Telomere heritability and parental age at conception effects in a wild avian population. *Mol Ecol* 31(23):6324–6338. https://doi.org/10.1111/mec.15804 |
| Scandinavian arctic fox | Dryad 10.5061/dryad.6g8t8 (TODO: verify; may not contain a pedigree). Status: listed. | 205 | 5 founders; mean F about 0.125 | Norén K, Godoy E, Dalén L, Meijer T, Angerbjörn A (2016). Inbreeding depression in a critically endangered carnivore. *Mol Ecol* 25(14):3309–3318. https://doi.org/10.1111/mec.13674 |
| Kalahari meerkats | Dryad 10.5061/dryad.h90j32v2 (TODO: verify; identifier format looks wrong for 2012). The project restricts its data. | — | 44% have F > 0 | Nielsen JF, English S, Goodall-Copestake WP, Wang J, Walling CA, Bateman AW, Flower TP, Sutcliffe RL, Samson J, Thavarajah NK, Kruuk LEB, Clutton-Brock TH, Pemberton JM (2012). Inbreeding and inbreeding depression of early life traits in a cooperative mammal. *Mol Ecol* 21(11):2788–2804. https://doi.org/10.1111/j.1365-294X.2012.05565.x |
| Kluane red squirrel | Dryad 10.5061/dryad.b064t9c5 (TODO: verify), `pedigree.csv`. Status: listed. | — | No pedigree statistics found | Taylor RW, Boon AK, Dantzer B, Réale D, Humphries MM, Boutin S, Gorrell JC, Coltman DW, McAdam AG (2012). Low heritabilities, but genetic and maternal correlations between red squirrel behaviours. *J Evol Biol* 25(4):614–624. https://doi.org/10.1111/j.1420-9101.2012.02456.x |

### 2.3 Captive, aquaculture and livestock

| Population | Data, status | Comparable values | Citation |
|---|---|---|---|
| Steelhead hatchery | Dryad 10.5061/dryad.bn249 (TODO: verify). Status: listed. | 6,602 fish over 4 generations; Ne 107.9; 15.7% of F4 have F > 0 | Naish KA, Seamons TR, Dauer MB, Hauser L, Quinn TP (2013). Relationship between effective population size, inbreeding and adult fitness‐related traits in a steelhead (*Oncorhynchus mykiss*) population released in the wild. *Mol Ecol* 22(5):1295–1309. https://doi.org/10.1111/mec.12185 |
| Brookfield Zoo *Peromyscus* | Dryad https://doi.org/10.5061/dryad.8q153 (reportedly includes an F column). Status: listed. | 6 populations, about 20 generations; per-individual F | Willoughby JR, Fernandez NB, Lamb MC, Ivy JA, Lacy RC, DeWoody JA (2015). The impacts of inbreeding, drift and selection on genetic diversity in captive breeding populations. *Mol Ecol* 24(1):98–110. https://doi.org/10.1111/mec.13020 |
| Holstein (whole-genome-sequenced pedigree) | Dryad https://doi.org/10.5061/dryad.vx0k6djq8. Status: listed. | FPED for 245 sequenced animals | Alemu SW, Kadri NK, Harland C, Faux P, Charlier C, Caballero A, Druet T (2021). An evaluation of inbreeding measures using a whole-genome sequenced cattle pedigree. *Heredity* 126(3):410–423. https://doi.org/10.1038/s41437-020-00383-9 |
| PIC pig common dataset | Supplement to the G3 paper. Status: not fetched. | None published (scale and runtime only) | Cleveland MA, Hickey JM, Forni S (2012). A common dataset for genomic analysis of livestock populations. *G3* 2(4):429–435. https://doi.org/10.1534/g3.111.001453 |

---

## 3. Considered but blocked, restricted, or without a usable file

| Resource | Why not usable | Citation |
|---|---|---|
| FamiLinx (Geni.com, about 86M profiles) | Download hosts (familinx.org, OSF) were blocked; current availability unknown. | Kaplanis J, Gordon A, Shor T, Weissbrod O, Geiger D, Wahl M, Gershovits M, Markus B, Sheikh M, Gymrek M, Bhatia G, MacArthur DG, Price AL, Erlich Y (2018). Quantitative analysis of population-scale family trees with millions of relatives. *Science* 360(6385):171–175 (TODO: verify author order). https://doi.org/10.1126/science.aam9309 · Colasurdo A, Omenti R (2024). Using online genealogical data for demographic research: An empirical examination of the FamiLinx database. *Demographic Research* 51(41):1299–1350. https://doi.org/10.4054/DemRes.2024.51.41 |
| Kinsources.net (more than 100 anthropological kinship networks) | Host blocked; licence listed as "not specified". | No citation collected. |
| Kinship-algorithm benchmark sets | Datasets not identified. | Kirkpatrick B, Ge S, Wang L (2019). Efficient computation of the kinship coefficients. *Bioinformatics* 35(6):1002–TODO: verify end page. TODO: verify DOI. Preprint: arXiv:1602.04368. |
| Speke's gazelle studbook | No public file. | Templeton AR, Read B (1984). Factors eliminating inbreeding depression in a captive herd of Speke's gazelle (*Gazella spekei*). *Zoo Biol* 3:177–199 (TODO: verify issue, pages, DOI). |
| Scandinavian wolves | No public file. Published mean F among breeding pairs 0.25–0.30 (TODO: verify). | Åkesson M, Liberg O, Sand H, Wabakken P, Bensch S, Flagstad Ø (2016). Genetic rescue in a severely inbred wolf population. *Mol Ecol* 25(19):4745–4756. https://doi.org/10.1111/mec.13797 |
| ENDOG example data | Host (ucm.es) unreachable; no published outputs for the example. | Gutiérrez JP, Goyache F (2005). A note on ENDOG: a computer program for analysing pedigree information. *J Anim Breed Genet* 122(3):172–176. https://doi.org/10.1111/j.1439-0388.2005.00512.x |
| CFC example data | Software site offline. | Sargolzaei M, Iwaisaki H, Colleau JJ (2006). CFC: a tool for monitoring genetic diversity. *Proc. 8th World Congress on Genetics Applied to Livestock Production*, Belo Horizonte (TODO: verify whole entry). |
| PyPedal example pedigrees | Only on SourceForge (blocked). | No citation collected. |
| AGHmatrix `ped.mrode` | Duplicate of Mrode (1.5). | Amadeu RR, Garcia AAF, Munoz PR, Ferrão LFV (2023). AGHmatrix: genetic relationship matrices in R. *Bioinformatics* 39(7):btad445. https://doi.org/10.1093/bioinformatics/btad445 |
| Restricted human cohorts: Hutterites, Old Order Amish, Utah Population Database, full BALSAC, deCODE (Iceland), Framingham | Access by application only. | No citation collected. |
| Pedigrees with no public file found: Tristan da Cunha, Pitcairn, Wright's shorthorn bull Comet, Rothschild, Mennonite, Dutch genealogies | No public file. | No citation collected. |
| Long-term wild studies with data on request only: Wytham great tits (SPI-Birds), collared flycatcher, Cayo Santiago macaques, Amboseli baboons, Gombe chimpanzees, takahē, kākāpō (Māori data governance), Mauritius kestrel, Alpine ibex, superb fairy-wren, Isle Royale wolves, Ram Mountain bighorn sheep | Not public, or no pedigree file found. | No citation collected. |
| Zoo and conservation studbooks: Mexican wolf, California condor, Przewalski's horse, Florida panther, dog breed studbooks; PMx examples; mouse wheel-running lines | No public file found. | No citation collected. |
