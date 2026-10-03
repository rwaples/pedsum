# pedsum against published pedigree statistics

Runs pedsum on public pedigrees whose statistics have been published, and
compares pedsum's output with those values. The latest results are in
[`results/results.md`](results/results.md). Background on each source, and the
citations, are in [`docs/benchmark-pedigrees.md`](../../docs/benchmark-pedigrees.md).

## Run it

From the repo root, in an environment with pedsum's dependencies plus
[`rdata`](https://pypi.org/project/rdata/), which reads R `.rda` files:

```bash
pip install rdata
python benchmarks/published/run_published.py                    # all datasets
python benchmarks/published/run_published.py --only atlas,btped  # a subset
```

The script needs network access to `raw.githubusercontent.com`. It exits 1 if
any check is FAIL or ERROR. A full run takes about 15 seconds once the files are cached.

## What it does

For each dataset in [`cases.yaml`](cases.yaml), the script:

1. **Downloads** the file from a pinned commit of the read-only CRAN mirror on
   GitHub into `.cache/`, and refuses it if its SHA-256 differs from the pinned
   value.
2. **Converts** it to pedsum input (`.work/<name>/input.tsv`):
   - Original IDs are mapped to integers in first-appearance order.
   - The dataset's missing-parent tokens (and R `NA`) become `-1`.
   - Sex is recoded.
   - The original ID and the reference-population flag are kept as extra
     columns.

   This step stands in for the `convert` subcommand proposed in #6.
3. **Runs pedsum**: `pedigree_summary.py validate`, then `summarize` on the repaired
   output (`validate.tsv.gz`). Its output goes to `.work/<name>/`, with the
   commands' output logged in `pedsum.log`.
4. **Compares** each check in `cases.yaml` with the published value, within
   that check's tolerance.

## Reading the results

| Status | Meaning |
|---|---|
| PASS | pedsum's own output matches the published value. |
| PASS (engine) | Matches, but computed by calling the `pedigree_graph` API directly, because the pedsum CLI does not report this statistic yet. The `Gap` column names the item in `docs/missing-statistics.md` that would bring it into pedsum. |
| DIFF | A known convention difference; the `note` column explains it. |
| GAP | Neither pedsum nor `pedigree_graph` computes it. The published value is recorded for when it does. |
| FAIL / ERROR | A mismatch, or a run failure. Investigate. |

## Findings so far

- **Exact matches.**
  - purgeR atlas: total F (197.2809), max F (0.4277344), F of the last row, and
    the founder count.
  - GENLIB `genea140`: every published count (N, sexes, founders, nuclear
    families, full sibships, largest sibship, generation depth).
  - visPedigree `deep_ped`: N, founders and maximum generation, once the 3
    absent parents are added.
  - The F of William Erasmus Darwin (0.06298828).
  - The F values of the Mrode example.
  - `BTped`'s structure.
- **Ne from individual ΔF (Gutiérrez et al. 2008).** For all four purgeR
  studbooks, `pedigree_graph` reproduces the published reference-population Ne
  (14.01, 11.10, 3.84, 39.32) only through its `ne_unrelated_founders`
  diagnostic. That diagnostic uses `t − 1` in the exponent, as purgeR does
  (`purgeR/R/Ne.R:44`). The headline `ne` uses `t` and gives 15.79 for atlas.
  pedsum's CLI cannot select a reference population yet (gap D1).
- **Ne over all rows.** purgeR keeps rows with `t ≤ 1` and counts them as ΔF = 0,
  giving 8.18 for atlas. `pedigree_graph` drops those rows, giving 8.03 (gap D2).
- **Missing sex blocks real pedigrees.** `validate` refuses any individual with
  unknown sex that is not a parent. That rejects dama (1 animal), arrui (1),
  dorcas (7) and deep_ped (521 rows) unless `--allow-missing-sex` is passed,
  which this harness does for those datasets. See issue #7.
- **Rounding in `summary.yaml`.** It rounds `max_F` and `mean_F` to 4 decimals.
  Exact comparisons therefore read `annotated.tsv.gz`.

## Adding a dataset

1. Add an entry under `datasets:` in `cases.yaml`: the URL pinned to a commit,
   its `sha256`, the column mapping, missing-parent tokens, and `sex_map`.
   - R files also need the `object` name.
   - Use `format: whitespace_table` for gzipped whitespace-delimited text.
2. Add `checks:` entries. Each needs the published value, its `source`, a `via`
   (see the header of `cases.yaml`), and a `tol` for non-integer values.
3. Add the source and its citation to `docs/benchmark-pedigrees.md`.

Only files that can be fetched without authentication belong here. Dryad,
Zenodo and Figshare deposits are blocked in the cloud sandbox this was built
in. They can be added the same way where the network allows.
