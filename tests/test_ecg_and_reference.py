"""Equivalent complete generations (``ecg``) and ``effective-size --reference-col``."""

from __future__ import annotations

import gzip

import numpy as np
import polars as pl
import pytest
import yaml
from conftest import EXAMPLE, run_pedsum, write_ped
from pedigree_graph import PedigreeGraph
from pedigree_graph._ne_rates import _equivalent_generations
from pedigree_graph.effective_size import ne_individual_delta_f

from pedsum.sections import equivalent_complete_generations

# 1 x 2 -> 3; 3 x 4 -> 5; 6 has mother 3 only; 7 = 5 x 3 (inbred, two paths to 1 and 2).
_ROWS = [
    {"id": 1, "sex": "M", "mother": -1, "father": -1},
    {"id": 2, "sex": "F", "mother": -1, "father": -1},
    {"id": 3, "sex": "F", "mother": 2, "father": 1},
    {"id": 4, "sex": "M", "mother": -1, "father": -1},
    {"id": 5, "sex": "M", "mother": 3, "father": 4},
    {"id": 6, "sex": "F", "mother": 3, "father": -1},
    {"id": 7, "sex": "F", "mother": 3, "father": 5},
]


def _frame(rows) -> tuple[pl.DataFrame, tuple[np.ndarray, np.ndarray]]:
    """The frame with ``ped_depth``, and each row's mother and father row (``-1`` when unknown)."""
    df = pl.DataFrame(rows).select("id", "mother", "father")
    pg = PedigreeGraph.from_arrays(
        ids=df["id"].to_numpy(), mother_ids=df["mother"].to_numpy(), father_ids=df["father"].to_numpy()
    )
    df = df.with_columns(pl.Series("ped_depth", np.asarray(pg.depth, dtype=np.int32)))
    return df, (np.asarray(pg.mother_rows), np.asarray(pg.father_rows))


def test_ecg_hand_values():
    """Founders 0; child of founders 1; one known founder parent 0.5; paths counted once each."""
    df, idx = _frame(_ROWS)
    ecg = equivalent_complete_generations(df, *idx)
    # 5: parents 3 (ECG 1) and 4 (ECG 0) -> (1+1)/2 + (1+0)/2 = 1.5
    # 6: mother 3 only -> (1+1)/2 = 1
    # 7: parents 3 (1) and 5 (1.5) -> (2 + 2.5)/2 = 2.25
    np.testing.assert_allclose(ecg, [0, 0, 1, 0, 1.5, 1, 2.25])


def test_ecg_matches_pedigree_graph_on_example():
    """The recurrence equals pedigree_graph's internal ECG on the bundled example."""
    raw = pl.read_csv(EXAMPLE, separator="\t")
    df, idx = _frame(raw.select("id", "mother", "father").to_dicts())
    pg = PedigreeGraph.from_arrays(
        ids=df["id"].to_numpy(), mother_ids=df["mother"].to_numpy(), father_ids=df["father"].to_numpy()
    )
    np.testing.assert_allclose(equivalent_complete_generations(df, *idx), _equivalent_generations(pg), atol=1e-12)


def test_ecg_does_not_depend_on_row_order():
    """Shuffled rows give each individual the same ECG."""
    df, idx = _frame(_ROWS)
    expected = dict(zip(df["id"].to_list(), equivalent_complete_generations(df, *idx), strict=True))
    shuffled, sidx = _frame([_ROWS[i] for i in (6, 4, 2, 0, 5, 3, 1)])
    got = dict(zip(shuffled["id"].to_list(), equivalent_complete_generations(shuffled, *sidx), strict=True))
    assert got == pytest.approx(expected)


def test_summarize_writes_ecg_column_and_distribution(tmp_path):
    """``annotated.tsv.gz`` gains an ``ecg`` column and summary.extra.yaml its distribution."""
    ped = write_ped(tmp_path / "ped.tsv", _ROWS)
    out = tmp_path / "out"
    res = run_pedsum(["summarize", "--in", str(ped), "--out", str(out)])
    assert res.returncode == 0, res.stderr
    with gzip.open(out / "annotated.tsv.gz", "rt") as fh:
        ann = pl.read_csv(fh, separator="\t")
    assert dict(zip(ann["id"].to_list(), ann["ecg"].to_list(), strict=True)) == pytest.approx(
        {1: 0, 2: 0, 3: 1, 4: 0, 5: 1.5, 6: 1, 7: 2.25}
    )
    extra = yaml.safe_load((out / "summary.extra.yaml").read_text())
    assert extra["individual"]["distributions"]["ecg"]["max"] == pytest.approx(2.25)


# --- effective-size --reference-col ----------------------------------------


def _with_reference(tmp_path, flags):
    raw = pl.read_csv(EXAMPLE, separator="\t")
    path = tmp_path / "ref.tsv"
    raw.with_columns(pl.Series("ref", flags)).write_csv(path, separator="\t")
    return raw, path


def _run_es(tmp_path, ped, *extra):
    out = tmp_path / "out"
    res = run_pedsum(["effective-size", "--in", str(ped), "--out", str(out), *extra])
    return res, out / "effective_size.yaml"


def test_reference_col_matches_pedigree_graph(tmp_path):
    """The record equals pedigree_graph's ne_individual_delta_f over the flagged rows."""
    raw = pl.read_csv(EXAMPLE, separator="\t")
    flags = ["1" if g >= 3 else "0" for g in raw["generation"].to_list()]
    raw, ped = _with_reference(tmp_path, flags)
    res, path = _run_es(tmp_path, ped, "--estimators", "ne_individual_delta_f", "--reference-col", "ref")
    assert res.returncode == 0, res.stderr
    record = yaml.safe_load(path.read_text())["effective_size"]["ne_individual_delta_f"]

    pg = PedigreeGraph.from_arrays(
        ids=raw["id"].to_numpy(), mother_ids=raw["mother"].to_numpy(), father_ids=raw["father"].to_numpy()
    )
    expected = ne_individual_delta_f(pg, reference=np.flatnonzero(np.array(flags) == "1"))
    assert record["reference_column"] == "ref"
    assert record["n_reference"] == expected.n_reference
    assert record["ne"] == pytest.approx(expected.ne, abs=1e-4)
    assert record["ne_unrelated_founders"] == pytest.approx(expected.ne_unrelated_founders, abs=1e-4)


def test_reference_col_changes_only_individual_delta_f(tmp_path):
    """Other estimators are identical with and without --reference-col."""
    raw = pl.read_csv(EXAMPLE, separator="\t")
    _, ped = _with_reference(tmp_path, ["true" if g == 2 else "" for g in raw["generation"].to_list()])
    res, path = _run_es(tmp_path / "a", ped)
    assert res.returncode == 0, res.stderr
    base = yaml.safe_load(path.read_text())["effective_size"]
    res, path = _run_es(tmp_path / "b", ped, "--reference-col", "ref")
    assert res.returncode == 0, res.stderr
    with_ref = yaml.safe_load(path.read_text())["effective_size"]
    assert {k: v for k, v in with_ref.items() if k != "ne_individual_delta_f"} == {
        k: v for k, v in base.items() if k != "ne_individual_delta_f"
    }
    assert with_ref["ne_individual_delta_f"]["n_reference"] != base["ne_individual_delta_f"]["n_reference"]


@pytest.mark.parametrize(
    ("flags", "column", "message"),
    [
        (None, "nope", "reference column 'nope' not in input"),
        ("0", "ref", "marks no row"),
        ("maybe", "ref", "must hold 1/0 or true/false"),
    ],
)
def test_reference_col_errors(tmp_path, flags, column, message):
    """A missing column, an empty reference or an unreadable token exits 1 without output."""
    raw = pl.read_csv(EXAMPLE, separator="\t")
    _, ped = _with_reference(tmp_path, [flags or "1"] * len(raw))
    res, path = _run_es(tmp_path, ped, "--reference-col", column)
    assert res.returncode == 1
    assert message in res.stderr
    assert not path.exists()


def test_unknown_sex_allowed_for_sex_free_estimators(tmp_path):
    """Unsexed rows only block the estimators that use sex."""
    ped = write_ped(
        tmp_path / "ped.tsv",
        [
            {"id": 1, "sex": "M", "mother": -1, "father": -1},
            {"id": 2, "sex": "F", "mother": -1, "father": -1},
            {"id": 8, "sex": "", "mother": -1, "father": -1},
            {"id": 3, "sex": "M", "mother": 2, "father": 1},
            {"id": 4, "sex": "", "mother": 2, "father": 1},
        ],
    )
    res, path = _run_es(tmp_path, ped, "--allow-missing-sex", "--estimators", "ne_inbreeding,ne_individual_delta_f")
    assert res.returncode == 0, res.stderr
    assert yaml.safe_load(path.read_text())["status"] == "complete"
