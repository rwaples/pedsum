# pedsum validation checks

This lists every integrity check that `pedsum validate` runs on a pedigree
file, taken from the check registry (`CHECKS` in `pedsum/checks.py`) in the
unreleased work after v0.17.0. There are 33 checks.

The tables below group the checks by topic. The order they run in is given
under [Run order](#run-order). A check is skipped when one it depends on did
not pass.

## What a failed check does

Each check has one of four fail policies. The policy decides what plain
`pedsum validate` does and what `pedsum validate --drop-offending` does.

| Policy | Plain `validate` | With `--drop-offending` |
|---|---|---|
| block | Stops. No `validate.tsv.gz` is written. | Stops. |
| block or drop | Stops. | Removes the offending individuals and continues. |
| keep or drop | Writes the pedigree with the problem left in, and reports it. | Removes the offending individuals. |
| autofix | Repairs the pedigree and reports what it changed. | Same. |

All checks run before the run decides whether it is blocked, so one run
reports every failure.

A problem can fail more than one check, and the strictest policy wins. A
self-loop fails `self_loops` (keep or drop) and also `acyclic` (block or
drop), so plain `validate` stops on it.

### Exit codes

| Exit code | Plain `validate` | With `--drop-offending` |
|---|---|---|
| 0 | Every check passed. | Nothing was dropped. Autofixes alone still exit 0. |
| 1 | A check failed, including one that was autofixed, and the pedigree was written. | At least one individual was dropped. |
| 2 | Blocked. | Blocked, or every individual was dropped. |

### `validate.log`

Findings are written to `validate.log` as tab-separated rows with the columns
`check`, `id`, `row`, `detail` and `n_refs`.

- `row` is the zero-based position among the data rows, not the line number in
  the file.
- A parse failure is one finding for the whole column, with `id` and `row`
  blank.
- `n_refs` is filled only for `parent_refs_present_*`: the number of rows
  naming the missing parent.
- Under `--drop-offending` the log holds the findings of the first pass.
  Problems that appear only after a removal round are in
  `validate.dropped.tsv`, not in the log.

## File and column checks

All of these block.

| Check | What it tests |
|---|---|
| `required_columns` | The ID, sex, mother and father columns are present. A multiple-birth or zygosity column must be present when its option names it. A birth-year column is required only with `--break-cycles`, and a birth-date column only with `--break-cycles` or `--twin-max-days`; otherwise a named birth column that is absent makes its checks skip. |
| `empty_pedigree` | The file has at least one row. |
| `id_dtype` | Every row has an ID. Any non-missing text is a valid ID, and `001` and `1` are different IDs. |
| `sex_tokens` | Every sex value is recognised: M/F in any case, Male/Female, or PLINK 1, 2, 0. Blank, `NA`, `NaN`, `N/A`, `.`, `?`, `None`, `null`, `-1`, `U` and `Unknown` are accepted as unknown sex. |
| `birth_year_dtype` | The birth-year column parses as numbers. A fractional year is truncated to a whole year. Missing tokens and `-1` mean unknown, and are the only ways to write it: any other negative value (such as `-1.5`), `inf`, and a value above 2147483647 (such as 2147483647.5) fail, and the detail names the value. Runs only with `--birth-year-col`. |
| `birth_date_dtype` | Every known birth date is ISO `YYYY-MM-DD`. Runs only with `--birth-date-col`. |
| `zygosity_tokens` | The zygosity column holds MZ or DZ in any case, or a missing value. Runs only with `--zygosity-col`. |

## ID checks

| Check | Policy | What it tests |
|---|---|---|
| `duplicate_rows` | autofix | No row repeats an earlier row exactly. Copies are removed. Rows are compared after parsing, so `NA` and `.` are the same missing parent and `M` and `1` the same sex. |
| `negative_ids` | block | No ID is written as a negative integer. |
| `duplicate_ids` | block or drop | No ID appears on two rows that differ. |
| `parent_token_range_mother` | block | No mother value is below -1. A missing parent is `-1`, blank, `NA`, `NaN`, `N/A`, `.`, `?`, `None` or `null` in any case, or a token named with `--missing-parent-token`. |
| `parent_token_range_father` | block | The same for the father column. |
| `zero_parent_present` | block | If a parent is written as `0`, an individual with ID 0 exists. PLINK and many R tools write a missing parent as 0; pass `--missing-parent-token 0` if the file does. |

## Parent link checks

| Check | Policy | What it tests |
|---|---|---|
| `cycle_breaks` | autofix | With `--break-cycles`: clears every self-link and, inside each descent cycle, every link whose parent is born after the child. Reports each link cleared. Clears nothing unless every named birth column parsed. A cycle with no such link is left for `acyclic`. |
| `role_conflict_breaks` | autofix | With `--break-role-conflicts`: for an individual with a recorded sex who is a mother on one row and a father on another, clears the links that contradict the recorded sex. A row whose mother and father are the same ID is left to `parents_distinct`. |
| `parent_refs_present_mother` | autofix | Every mother ID has a row of its own. A missing mother is added as a founder row. |
| `parent_refs_present_father` | autofix | The same for fathers. |
| `parent_child_count` | keep or drop | No mother has more than `--max-children-mother` children and no father more than `--max-children-father`. Off by default. It catches a placeholder code such as 9999 used for an unknown parent. 20 and 50 suit a human register. `--drop-offending` drops the parent and keeps the children. |
| `self_loops` | keep or drop | No individual is listed as its own mother or father. A self-loop also fails `acyclic`, so plain `validate` stops on it. |
| `parents_distinct` | block or drop | No row has the same ID as mother and father. |
| `acyclic` | block or drop | No individual is its own ancestor. Reports every row in or below a cycle. `--drop-offending` removes the members of the cycle and keeps their descendants. |

## Sex checks

| Check | Policy | What it tests |
|---|---|---|
| `parent_refs_sex_conflict` | block or drop | No parent that lacks a row is named as both a mother and a father. |
| `sex_role_ambiguity` | block or drop | No individual with unknown sex is used as both a mother and a father. `--allow-missing-sex` tolerates it and writes the sex as unknown, but `summarize`, `effective-size` and `assortative-mating` still refuse an individual in both parent roles. |
| `parent_role_conflict` | block or drop | No individual with a recorded sex is used as both a mother and a father. Rows whose mother and father are the same ID are not counted here. `--break-role-conflicts` repairs it. |
| `sex_role_override` | autofix | Reports each recorded sex that was overridden because the individual is used in only one parent role: a recorded male used only as a mother becomes female, and the reverse. `--no-override-asserted-sex` turns the repair off. |
| `sex_role_consistency` | keep or drop | No mother has a resolved sex other than female and no father other than male. It does not cover individuals whose sex started unknown (`sex_role_ambiguity`) or those used in both roles (`parent_role_conflict`). |

## Birth year and birth date checks

These run only when the matching column is named. All are keep or drop.

| Check | What it tests |
|---|---|
| `birth_year_range` | Every known birth year is within `--birth-year-min` and `--birth-year-max` (default 1800 to next year). |
| `birth_year_topology` | Each parent is at least `--min-parent-age` years older than the child (default 0) and, with `--max-parent-age`, at most that many years older, judged on birth years. A year difference overstates the age by up to 1, so a difference is flagged only above `--max-parent-age` + 1. If `birth_date_dtype` fails, every link is judged on birth years, and the note says so. A row with no year in the year column takes the year of its birth date. Used for a parent-child link where the two birth dates are not both known. A link is skipped unless both birth years are known. |
| `birth_date_range` | Every known birth date has a year within the same range. |
| `birth_date_year_consistency` | A row's birth date falls inside its birth year, when both are given. |
| `birth_date_topology` | The same parent-age limits judged on birth dates to the day, in completed years, for links where both dates are known. |

## Multiple birth checks

Both are autofix.

| Check | What it tests |
|---|---|
| `multiple_birth_sets` | With `--multiple-birth-col`: every member of a multiple-birth set shares both known parents with another member. A member that does not has its set cleared. A set that holds two or more groups of full sibs is cleared for every member. A set whose full sibs were born apart is cleared for every member too: their known birth dates, sorted, have a gap of more than 7 days, or, when some date is unknown, their known birth years (a missing one taken from the date) span more than 1. Members cleared for another reason do not count. Unknown birth data clears nothing. |
| `mz_links` | With `--zygosity-col`, which needs `--multiple-birth-col`: MZ marks form pairs of co-twins. A mark, MZ or DZ, is cleared when its row is in no set. An MZ mark is also cleared when it is the only MZ in its set, when the set has three or more MZ, or when the two co-twins have known and different sex. |

## Options that change what the checks do

| Option | Effect |
|---|---|
| `--drop-offending` | Removes the offending individuals for block-or-drop and keep-or-drop checks, and lists them in `validate.dropped.tsv`. For `acyclic` only the cycle members are removed. After each removal the checks run again, and further rounds follow until nothing fails. |
| `--break-cycles` | Turns on `cycle_breaks`. Needs `--birth-year-col` or `--birth-date-col`. Cleared links go to `validate.cleared_links.tsv`. |
| `--break-role-conflicts` | Turns on `role_conflict_breaks`. Cleared links go to `validate.cleared_links.tsv`. |
| `--allow-missing-sex` | Tolerates `sex_role_ambiguity`. |
| `--no-override-asserted-sex` | Turns off the `sex_role_override` repair. |
| `--no-sex-check` | Bypasses `parent_refs_sex_conflict`. If such a parent is kept, the other pedsum commands refuse the pedigree. |
| `--missing-parent-token TOKEN` | Names an extra token that means a missing parent, such as 0 or 9999. Repeatable. It is an error if the token is also the ID of a row. |
| `--min-parent-age N`, `--max-parent-age N` | Set the parent-age limits for `birth_year_topology` and `birth_date_topology`. The minimum defaults to 0, which flags only a child born before a parent. The maximum defaults to 0, which is off; 70 suits a human register. |
| `--max-children-mother N`, `--max-children-father N` | Set the limits for `parent_child_count`. |
| `--twin-max-days N` | Groups full sibs into multiple-birth sets, where each is born at most N days after the one before. N is the gap between adjacent births, not the span of the set. N runs from 0 to 7: a larger N is a usage error (exit 2), since a later validate with `--multiple-birth-col` clears a set whose adjacent births are more than 7 days apart. Needs `--birth-date-col`, and cannot be combined with `--multiple-birth-col`. |

## Run order

`required_columns`, `empty_pedigree`, `id_dtype`, `sex_tokens`,
`birth_year_dtype`, `birth_date_dtype`, `zygosity_tokens`, `duplicate_rows`,
`negative_ids`, `duplicate_ids`, `parent_token_range_mother`,
`parent_token_range_father`, `zero_parent_present`, `cycle_breaks`,
`role_conflict_breaks`, `parent_refs_present_mother`,
`parent_refs_present_father`, `parent_child_count`,
`parent_refs_sex_conflict`, `sex_role_ambiguity`, `parent_role_conflict`,
`self_loops`, `parents_distinct`, `sex_role_override`, `sex_role_consistency`,
`acyclic`, `birth_year_range`, `birth_year_topology`, `birth_date_range`,
`birth_date_year_consistency`, `birth_date_topology`, `multiple_birth_sets`,
`mz_links`.
