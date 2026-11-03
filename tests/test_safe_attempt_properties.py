"""``apply_safe_attempt`` is idempotent: redacting a redacted payload changes nothing.

The payload template is captured from one real ``summarize --safe-attempt`` run
with every optional section on (multiple births, sex concordance, relationship
burden). Hypothesis then redraws every integer leaf around the minimum cell, so
each rule's small and large branches both fire, and asserts that a second pass
neither raises nor changes the first pass's output.
"""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace

import polars as pl
import pytest
from conftest import EXAMPLE, run_pedsum
from hypothesis import given, settings
from hypothesis import strategies as st

import pedsum.cli.summarize as summarize_cli
from pedsum.redaction import SAFE_MIN_CELL
from pedsum.report import apply_safe_attempt


def _with_twin_sets(path):
    """``example_pedigree.tsv`` plus a ``set`` column pairing the first two children of each sibship."""
    df = pl.read_csv(EXAMPLE, separator="\t")
    kids = df.filter(pl.col("mother") != -1).with_columns(
        pl.int_range(pl.len()).over("mother", "father").alias("birth_order"),
        pl.len().over("mother", "father").alias("sibship_size"),
        pl.format("{}_{}", "mother", "father").alias("sib"),
    )
    sets = kids.filter((pl.col("birth_order") < 2) & (pl.col("sibship_size") >= 2)).select(
        "id", pl.col("sib").alias("set")
    )
    df.join(sets, on="id", how="left").write_csv(path, separator="\t", null_value="")
    return path


@pytest.fixture(scope="module")
def payload_template(tmp_path_factory):
    """(ped_data, ind_data) as summarize hands them to ``apply_safe_attempt``."""
    tmp = tmp_path_factory.mktemp("safe_attempt")
    ped = _with_twin_sets(tmp / "ped.tsv")
    captured = []

    def capture(ped_data, ind_data, *args, **kwargs):
        captured.append(copy.deepcopy((ped_data, ind_data)))
        apply_safe_attempt(ped_data, ind_data, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(summarize_cli, "apply_safe_attempt", capture)
        r = run_pedsum(
            [
                "summarize",
                "--in",
                str(ped),
                "--out",
                str(tmp / "out"),
                "--safe-attempt",
                "--multiple-birth-col",
                "set",
                "--birth-year-col",
                "birth_year",
                "--sex-concordance",
                "--per-individual-burden",
            ]
        )
    assert r.returncode == 0, r.stderr
    (ped_data, ind_data) = captured[0]
    for section in ("multiple_births", "offspring_sex_concordance", "inbreeding", "relationship_summary"):
        assert ped_data.get(section) is not None, section
    assert ped_data["multiple_births"]["by_birth_decade"], "no decade rows: the group-n rule would go unexercised"
    return ped_data, ind_data


def _map_ints(obj, fill):
    """``obj`` with every int leaf (also inside dataclasses such as Distribution) replaced by ``fill(leaf)``."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return fill(obj)
    if isinstance(obj, dict):
        return {k: _map_ints(v, fill) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_ints(v, fill) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_map_ints(v, fill) for v in obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return replace(obj, **{f.name: _map_ints(getattr(obj, f.name), fill) for f in fields(obj) if f.init})
    return obj


def _n_ints(obj) -> int:
    seen = []
    _map_ints(obj, seen.append)
    return len(seen)


def _redact(ped_data, ind_data):
    ped_data, ind_data = copy.deepcopy((ped_data, ind_data))
    apply_safe_attempt(ped_data, ind_data)
    return ped_data, ind_data


def test_real_payload_redaction_is_idempotent(payload_template):
    """The captured payload itself survives a second pass unchanged."""
    once = _redact(*payload_template)
    assert _redact(*once) == once


@settings(deadline=None)
@given(data=st.data())
def test_redaction_is_idempotent(payload_template, data):
    """With every count redrawn near the minimum cell, a second pass neither raises nor changes anything."""
    n = _n_ints(payload_template)
    values = iter(data.draw(st.lists(st.integers(0, 3 * SAFE_MIN_CELL), min_size=n, max_size=n)))
    ped_data, ind_data = _map_ints(payload_template, lambda _: next(values))
    ped_data["n_total"] = payload_template[0]["n_total"]
    once = _redact(ped_data, ind_data)
    assert _redact(*once) == once
