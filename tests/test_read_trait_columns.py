"""read_trait_columns: projected read, id alignment, and missing-token handling."""

from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pedsum.base import PedigreeError
from pedsum.parse import read_trait_columns
from pedsum.pedigree_ops import unique_ints

if TYPE_CHECKING:
    from pathlib import Path

_HEADER = "id\tsex\tmother\tfather\tx\ty\n"
_ROWS = ["1\tM\t-1\t-1\t 0.5 \tNA\n", "2\tF\t-1\t-1\tnull\t-9\n", "3\tF\t2\t1\t?\t B \n"]


def _write(tmp_path, text, name="ped.tsv") -> Path:
    path = tmp_path / name
    if name.endswith(".gz"):
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)
    return path


def test_rows_aligned_to_ids_in_any_order(tmp_path):
    """Ids in another order than the file, and ids absent from it, align by id; absent ids are None."""
    path = _write(tmp_path, _HEADER + "".join(_ROWS))
    out = read_trait_columns(path, "auto", "id", ["x", "y"], np.array([3, 99, 1, 2]), ["-9"])
    assert out["x"].dtype == object
    assert out["x"].tolist() == [None, None, "0.5", None]
    assert out["y"].tolist() == ["B", None, None, None]


def test_same_order_matches_reordered_path(tmp_path):
    """The file-order fast path and the gather path give the same tokens."""
    path = _write(tmp_path, _HEADER + "".join(_ROWS))
    in_order = read_trait_columns(path, "auto", "id", ["x", "y"], np.array([1, 2, 3]))
    reordered = read_trait_columns(path, "auto", "id", ["x", "y"], np.array([2, 3, 1]))
    for column in ("x", "y"):
        assert in_order[column][[1, 2, 0]].tolist() == reordered[column].tolist()
    assert in_order["y"].tolist() == [None, "-9", "B"]


@pytest.mark.parametrize("name", ["ped.tsv", "ped.tsv.gz"])
def test_duplicate_and_id_trait_names(tmp_path, name):
    """A trait named twice, or named like the id column, reads like any other column."""
    path = _write(tmp_path, _HEADER + "".join(_ROWS), name)
    out = read_trait_columns(path, "auto", "id", ["x", "x", "id"], np.array([1, 2, 3]))
    assert list(out) == ["x", "id"]
    assert out["id"].tolist() == ["1", "2", "3"]


def test_whitespace_separated_input(tmp_path):
    """PLINK-style whitespace input projects the requested columns too."""
    text = "id sex mother father x\n1 M -1 -1 1.5\n2 F -1 -1 NA\n"
    path = _write(tmp_path, text, "ped.txt")
    out = read_trait_columns(path, "auto", "id", ["x"], np.array([2, 1]))
    assert out["x"].tolist() == [None, "1.5"]


def test_absent_trait_column_names_file_header(tmp_path):
    """An absent trait column is a PedigreeError naming it and the file's columns."""
    path = _write(tmp_path, _HEADER + "".join(_ROWS))
    with pytest.raises(PedigreeError) as err:
        read_trait_columns(path, "auto", "id", ["x", "z"], np.array([1, 2, 3]))
    assert str(err.value) == "trait column(s) ['z'] not in input; file has ['id', 'sex', 'mother', 'father', 'x', 'y']"


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int8])
def test_unique_ints_matches_np_unique(dtype):
    """unique_ints returns np.unique's sorted distinct values with the same dtype, empty input included."""
    rng = np.random.default_rng(0)
    for values in (rng.integers(-50, 50, 1000).astype(dtype), np.array([], dtype=dtype), np.array([7], dtype=dtype)):
        got = unique_ints(values)
        expected = np.unique(values)
        assert got.dtype == expected.dtype
        assert np.array_equal(got, expected)
