"""Pinned ``assortative_mating.yaml`` for small pedigrees: how pedsum writes pg-phenotype's result.

Each case runs the CLI on ``tests/data/assortative_mating/<case>.tsv`` and
compares the YAML from ``n_total`` on with ``<case>.yaml``; the input path,
command, version and timestamp above it, and the thread count, vary by run.
The expected files are pedsum #13's output from before the computation moved
to pg-phenotype, except ``binary_ordinal_birth_year``'s grade x grade stratified
polychoric, whose SE and sandwich CI pedsum withheld (empty-cell rounding
noise; see the CHANGELOG).  After a deliberate output change, regenerate with
``PEDSUM_REGENERATE=1 pixi run pytest tests/test_assortative_mating_payload.py``
and review the diff.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from conftest import run_pedsum

DATA = Path(__file__).parent / "data" / "assortative_mating"
BIRTH_YEAR = ["--birth-year-col", "birth_year", "--birth-year-max", "2100"]
CASES = {
    "binary_2x2_boot": ["--trait", "dx", "--trait-type", "dx=binary", "--permutations", "199", "--bootstrap", "300", "--seed", "4"],
    "continuous_one_network": ["--trait", "bmi", "--trait-type", "bmi=continuous", "--permutations", "199", "--seed", "0"],
    "continuous_binary_boot": [
        "--trait", "liability", "smoker", "--trait-type", "liability=continuous", "--trait-type", "smoker=binary",
        "--permutations", "199", "--bootstrap", "200", "--seed", "8",
    ],
    "continuous_ordinal_depth": [
        "--trait", "height", "grade", "--trait-type", "height=continuous", "--trait-type", "grade=ordinal",
        "--stratify-by", "depth", "--min-stratum-networks", "2", "--permutations", "299", "--bootstrap", "200", "--seed", "2",
    ],
    "binary_ordinal_birth_year": [
        "--trait", "dx", "grade", "--trait-type", "dx=binary", "--trait-type", "grade=ordinal", *BIRTH_YEAR,
        "--stratify-by", "birth_year", "--min-stratum-networks", "2", "--permutations", "299", "--seed", "9",
    ],
    "thin_birth_year_strata": [
        "--trait", "liability", "dx", "--trait-type", "liability=continuous", "--trait-type", "dx=binary", *BIRTH_YEAR,
        "--stratify-by", "birth_year", "--birth-year-bin", "5", "--permutations", "199", "--seed", "0",
    ],
}  # fmt: skip
THREADS = re.compile(r"^(    threads: )\d+$", re.MULTILINE)


def _stable(text: str) -> str:
    """The YAML from ``n_total`` on, with the thread count blanked."""
    _, sep, tail = text.partition("\nn_total:")
    assert sep
    return THREADS.sub(r"\1N", sep.lstrip("\n") + tail)


@pytest.mark.parametrize("case", sorted(CASES))
def test_payload_matches_the_pinned_yaml(case, tmp_path):
    """The CLI writes the pinned YAML byte for byte."""
    out_dir = tmp_path / "out"
    res = run_pedsum(["assortative-mating", "--in", str(DATA / f"{case}.tsv"), "--out", str(out_dir), *CASES[case]])
    assert res.returncode == 0, res.stderr
    got = _stable((out_dir / "assortative_mating.yaml").read_text())
    expected = DATA / f"{case}.yaml"
    if os.environ.get("PEDSUM_REGENERATE"):
        expected.write_text(got)
    assert got == expected.read_text()
