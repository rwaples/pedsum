"""Characterization goldens: every artifact each subcommand writes, pinned as normalized text.

Each case runs the CLI on an input under ``tests/golden/inputs`` and compares
``tests/golden/expected/<case>.txt``: the exit code, the sorted artifact names
in ``--out``, and every artifact's text. ``.gz`` artifacts are compared
decompressed and parquet as CSV, because both carry run-specific bytes. The
YAML meta values ``input``, ``command`` and ``version`` (and the ``--tsv``
``meta`` rows) become placeholders, ``generated_at`` is dropped, and thread
counts are blanked. After a deliberate output change, regenerate with
``PEDSUM_REGENERATE=1 pixi run pytest tests/test_golden.py`` and review the diff.
"""

from __future__ import annotations

import gzip
import io
import os
import re
from pathlib import Path

import polars as pl
import pytest
from conftest import run_pedsum

GOLDEN = Path(__file__).parent / "golden"
INPUTS = GOLDEN / "inputs"
EXPECTED = GOLDEN / "expected"

BY = ["--birth-year-col", "birth_year", "--birth-year-max", "2030"]
MALFORMED = {
    "required_columns": ("missing_sex_col.tsv", []),
    "empty_pedigree": ("empty.tsv", []),
    "id_dtype": ("id_blank.tsv", []),
    "sex_tokens": ("sex_bad.tsv", []),
    "negative_ids": ("negative_id.tsv", []),
    "parent_token_range_mother": ("mother_range.tsv", []),
    "parent_token_range_father": ("father_range.tsv", []),
    "birth_year_dtype": ("birth_year_bad.tsv", BY),
}
SUMMARIZE_FULL = ["--sex-concordance", "--sex-concordance-permutations", "50", "--sex-concordance-seed", "3", "--tsv"]

CASES: dict[str, list[str]] = {
    "summarize_clean": ["summarize", "traits.tsv", *BY, "--per-individual-burden", *SUMMARIZE_FULL],
    "summarize_max_degree_2": ["summarize", "traits.tsv", "--max-degree", "2"],
    "summarize_safe_clean": ["summarize", "traits.tsv", *BY, "--per-individual-burden", *SUMMARIZE_FULL, "--safe-attempt"],
    "summarize_safe_small": [
        "summarize", "small.tsv", *BY, "--allow-missing-sex", "--per-individual-burden", *SUMMARIZE_FULL, "--safe-attempt",
    ],
    "summarize_small": ["summarize", "small.tsv", *BY, "--allow-missing-sex", "--per-individual-burden", *SUMMARIZE_FULL],
    "summarize_no_inbreeding": ["summarize", "small.tsv", "--allow-missing-sex", "--no-inbreeding", "--tsv"],
    "summarize_reingest_no_inbreeding": ["summarize", "reingest.tsv", "--allow-missing-sex", "--no-inbreeding"],
    "summarize_strings": ["summarize", "strings_clean.tsv", *BY, "--tsv"],
    "summarize_blocked": ["summarize", "sex_bad.tsv"],
    "validate_clean": ["validate", "traits.tsv", *BY],
    "validate_strings_fill": ["validate", "strings.tsv", *BY, "--fill-half-founders"],
    "validate_flagged": ["validate", "flagged.tsv", *BY, "--fill-half-founders"],
    "validate_droppable": ["validate", "droppable.tsv", *BY],
    "validate_droppable_drop": ["validate", "droppable.tsv", *BY, "--drop-offending", "--fill-half-founders"],
    "validate_cycle_absent_parent": ["validate", "cycle_absent_parent.tsv"],
    "validate_cycle_absent_parent_drop": ["validate", "cycle_absent_parent.tsv", "--drop-offending"],
    "validate_droppable_drop_tolerant": [
        "validate", "droppable.tsv", *BY, "--drop-offending", "--allow-missing-sex", "--no-sex-check",
    ],
    "effective_size_clean": ["effective-size", "traits.tsv", *BY],
    "effective_size_reference": ["effective-size", "traits.tsv", "--reference-col", "ref", "--estimators", "ne_individual_delta_f"],
    "effective_size_strings": ["effective-size", "strings_clean.tsv", "--reference-col", "ref", "--estimators", "ne_individual_delta_f,ne_coancestry"],
    "assortative_clean": [
        "assortative-mating", "traits.tsv", "--trait", "liability", "dx", "--trait-type", "dx=binary",
        "--permutations", "49", "--seed", "5", "--threads", "1",
    ],
    "assortative_strings": [
        "assortative-mating", "strings_clean.tsv", *BY, "--trait", "birth_year", "--trait-type", "birth_year=continuous",
        "--permutations", "0", "--threads", "1",
    ],
    "epimight_clean": ["epimight-input", "traits.tsv", *BY, "--pairs", "--exact-kinship", "--parquet"],
    "epimight_strings": ["epimight-input", "strings_clean.tsv", "--pairs", "--drop-founders"],
    **{f"validate_block_{check}": ["validate", path, *extra] for check, (path, extra) in MALFORMED.items()},
    **{
        f"validate_block_{check}_drop": ["validate", path, *extra, "--drop-offending"]
        for check, (path, extra) in MALFORMED.items()
    },
}  # fmt: skip

# A long command wraps onto indented continuation lines.
META = re.compile(r"^(input|command|version): .*(?:\n .*)*$", re.MULTILINE)
GENERATED_AT = re.compile(r"^generated_at: .*\n", re.MULTILINE)
THREADS = re.compile(r"^(\s+threads: )\d+$", re.MULTILINE)
TSV_META = re.compile(r"^meta\t(input|command|version)\t\t.*$", re.MULTILINE)
TSV_GENERATED_AT = re.compile(r"^meta\tgenerated_at\t\t.*\n", re.MULTILINE)


def _normalize(text: str) -> str:
    text = META.sub(r"\1: <\1>", text)
    text = GENERATED_AT.sub("", text)
    text = THREADS.sub(r"\1N", text)
    text = TSV_META.sub(r"meta\t\1\t\t<\1>", text)
    return TSV_GENERATED_AT.sub("", text)


def _artifact_text(path: Path) -> str:
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode()
    if path.suffix == ".parquet":
        buf = io.StringIO()
        pl.read_parquet(path).write_csv(buf, separator="\t")
        return buf.getvalue()
    return path.read_text()


def render(out_dir: Path, returncode: int) -> str:
    """The case's exit code, artifact names and normalized artifact texts as one document."""
    names = sorted(p.name for p in out_dir.iterdir())
    parts = [f"exit: {returncode}", f"artifacts: {names}"]
    parts += [f"=== {name} ===\n{_normalize(_artifact_text(out_dir / name))}" for name in names]
    return "\n".join(parts) + "\n"


@pytest.mark.parametrize("case", sorted(CASES))
def test_artifacts_match_the_golden(case, tmp_path):
    """The CLI writes the pinned artifacts, and exits with the pinned code."""
    command, input_name, *flags = CASES[case]
    out_dir = tmp_path / "out"
    res = run_pedsum([command, "--in", str(INPUTS / input_name), "--out", str(out_dir), *flags])
    got = render(out_dir, res.returncode)
    expected = EXPECTED / f"{case}.txt"
    if os.environ.get("PEDSUM_REGENERATE"):
        expected.parent.mkdir(exist_ok=True)
        expected.write_text(got)
    assert got == expected.read_text()
