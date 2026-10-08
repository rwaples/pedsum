"""Mate Correlation (pedsum#13): trait reading and typing, and the payload pedsum writes from pg-phenotype's result."""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from conftest import run_pedsum, write_ped
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pedsum.assortative_mating import (
    MIN_STRATUM_NETWORKS,
    Trait,
    classify_trait,
    compute_assortative_mating,
)
from pedsum.base import PedigreeError
from pedsum.parse import read_trait_columns
from pedsum.report import ASSORTATIVE_MATING_FIGURES, _write_yaml
from pedsum.validate import load_and_validate


def _tokens(*values) -> np.ndarray:
    return np.array(list(values), dtype=object)


# ---------------------------------------------------------------------------
# Step 1: reading and typing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tokens", "levels", "codes"),
    [
        (("9", "-4", None, "9"), ("-4", "9"), [1, 0, np.nan, 1]),
        (("-2.5", "1e3", "-2.5"), ("-2.5", "1000"), [0, 1, 0]),
        (("0", "1", "1"), ("0", "1"), [0, 1, 1]),
        (("TRUE", "false", "True"), ("false", "true"), [1, 0, 1]),
        (("Yes", "no", None), ("no", "yes"), [1, 0, np.nan]),
    ],
)
def test_two_levels_infer_binary(tokens, levels, codes):
    """Any two numeric levels (the lower is the reference), true/false or yes/no infer binary."""
    trait = classify_trait("t", _tokens(*tokens))
    assert (trait.kind, trait.type_source, trait.levels) == ("binary", "inferred", levels)
    np.testing.assert_array_equal(trait.values, codes)


def test_many_numeric_values_infer_continuous():
    """More than 20 distinct numeric values infer continuous, values kept as parsed."""
    trait = classify_trait("t", _tokens(*(str(i / 2) for i in range(21)), None))
    assert (trait.kind, trait.type_source, trait.levels) == ("continuous", "inferred", None)
    np.testing.assert_array_equal(trait.values, [i / 2 for i in range(21)] + [np.nan])


@pytest.mark.parametrize("n_levels", [3, 20])
def test_ambiguous_numeric_needs_a_stated_type(n_levels):
    """3-20 distinct numeric values without --trait-type is an error naming the flag."""
    with pytest.raises(PedigreeError, match="--trait-type t=ordinal"):
        classify_trait("t", _tokens(*(str(i) for i in range(n_levels))))


def test_ordinal_levels_order_numerically():
    """Ordinal levels sort as numbers (10 after 9), are coded in that order, and are recorded in levels."""
    trait = classify_trait("t", _tokens("10", "9", "2", None, "1", "10"), "ordinal")
    assert (trait.kind, trait.type_source, trait.levels) == ("ordinal", "stated", ("1", "2", "9", "10"))
    np.testing.assert_array_equal(trait.values, [3.0, 2.0, 1.0, np.nan, 0.0, 3.0])


def test_stated_continuous_keeps_values():
    """A stated continuous trait keeps its numeric values whatever its level count."""
    trait = classify_trait("t", _tokens("1", "2", "4", None), "continuous")
    assert trait.levels is None
    np.testing.assert_array_equal(trait.values, [1.0, 2.0, 4.0, np.nan])


@pytest.mark.parametrize(
    ("tokens", "stated", "match"),
    [
        ((None, None), None, "missing in every row"),
        (("5", "5.0", None), None, "constant"),
        (("yes", "YES"), None, "constant"),
        (("1", "inf"), None, "non-finite"),
        (("1", "-Infinity", "2"), "continuous", "non-finite"),
        (("1", "abc", "2"), None, "must be numeric"),
        (("true", "false"), "ordinal", "must be numeric"),
        (("true", "yes"), None, "must be numeric"),
        (("1", "2", "3"), "binary", "stated binary"),
    ],
)
def test_classify_errors(tokens, stated, match):
    """All-missing, constant, non-finite, non-numeric and mis-stated columns are data errors."""
    with pytest.raises(PedigreeError, match=match):
        classify_trait("t", _tokens(*tokens), stated)


def _read(tmp_path, rows, columns, missing=()) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    path = write_ped(tmp_path / "ped.tsv", rows)
    df, labels = load_and_validate(path)
    return df, read_trait_columns(path, "auto", "id", columns, df["id"].to_numpy(), labels, missing)


def test_missing_tokens_trimming_and_case(tmp_path):
    """Reader null tokens, '.', '?' and their padded or recased forms are missing; other tokens are stripped."""
    tokens = [" 1.5 ", ".", "?", " na ", "Null", "NAN", "", "nan", " 2 "]
    rows = [{"id": i + 1, "sex": "M", "mother": -1, "father": -1, "x": t} for i, t in enumerate(tokens)]
    df, out = _read(tmp_path, rows, ["x"])
    by_id = dict(zip(df["id"].to_list(), out["x"].tolist(), strict=True))
    assert [by_id[i + 1] for i in range(len(tokens))] == ["1.5", None, None, None, None, None, None, None, "2"]


def test_trait_missing_tokens(tmp_path):
    """--trait-missing tokens are missing after stripping, and only exact matches count."""
    rows = [
        {"id": 1, "sex": "M", "mother": -1, "father": -1, "x": "-9"},
        {"id": 2, "sex": "F", "mother": -1, "father": -1, "x": " -9 "},
        {"id": 3, "sex": "M", "mother": -1, "father": -1, "x": "-9.0"},
        {"id": 4, "sex": "F", "mother": -1, "father": -1, "x": "3"},
    ]
    df, out = _read(tmp_path, rows, ["x"], missing=[" -9"])
    by_id = dict(zip(df["id"].to_list(), out["x"].tolist(), strict=True))
    assert by_id == {1: None, 2: None, 3: "-9.0", 4: "3"}


def test_ids_added_by_validate_are_missing(tmp_path):
    """Founders that validate adds for absent parent ids carry no trait value in its output."""
    rows = [
        {"id": 1, "sex": "M", "mother": -1, "father": -1, "x": "0.5"},
        {"id": 3, "sex": "F", "mother": 2, "father": 1, "x": "1.5"},
    ]
    write_ped(tmp_path / "ped.tsv", rows)
    run_pedsum(["validate", "--in", str(tmp_path / "ped.tsv"), "--out", str(tmp_path / "v")])
    fixed = tmp_path / "v" / "validate.tsv.gz"
    df, labels = load_and_validate(fixed)
    out = read_trait_columns(fixed, "auto", "id", ["x"], df["id"].to_numpy(), labels)
    by_id = dict(zip(df["id"].to_list(), out["x"].tolist(), strict=True))
    assert by_id == {1: "0.5", 2: None, 3: "1.5"}


def test_missing_trait_column(tmp_path):
    """A trait column absent from the file is a data error naming it."""
    rows = [{"id": 1, "sex": "M", "mother": -1, "father": -1}]
    with pytest.raises(PedigreeError, match="'y'"):
        _read(tmp_path, rows, ["y"])


# ---------------------------------------------------------------------------
# Step 2: the payload
# ---------------------------------------------------------------------------


def _pedigree(pairs: list[tuple[int, int]], children_per_pair: int = 1) -> pl.DataFrame:
    """A validated-shape frame: founder parents, then children of each (mother, father) pair."""
    parents = sorted({m for m, _ in pairs} | {f for _, f in pairs})
    sex = {m: 0 for m, _ in pairs} | {f: 1 for _, f in pairs}
    rows = [(p, sex[p], -1, -1, 0) for p in parents]
    next_id = max(parents) + 1
    for m, f in pairs:
        for _ in range(children_per_pair):
            rows.append((next_id, 0, m, f, 1))
            next_id += 1
    ids, sexes, mothers, fathers, depths = (list(col) for col in zip(*rows, strict=True))
    return pl.DataFrame(
        {"id": ids, "sex": sexes, "mother": mothers, "father": fathers, "ped_depth": depths},
        schema={"id": pl.Int64, "sex": pl.Int8, "mother": pl.Int64, "father": pl.Int64, "ped_depth": pl.Int32},
    )


def _trait(df: pl.DataFrame, by_id: dict[int, float], name: str = "x") -> Trait:
    values = np.array([by_id.get(i, np.nan) for i in df["id"].to_list()], dtype=np.float64)
    return Trait(name, "continuous", "stated", None, values)


def _pair_values(rng, n_pairs, remate_every=3) -> tuple[list[tuple[int, int]], dict[int, float]]:
    """Mating Pairs with one mother each; every ``remate_every``-th pair reuses the previous pair's father."""
    pairs, father = [], 1000
    for k in range(n_pairs):
        if k % remate_every:
            father += 1
        pairs.append((k + 1, father))
    ids = sorted({p for pair in pairs for p in pair})
    return pairs, dict(zip(ids, rng.normal(size=len(ids)).tolist(), strict=True))


def test_crude_pearson_is_corrcoef_over_mating_pairs():
    """The cell's r is np.corrcoef over one observation per Mating Pair, whatever the sibship sizes."""
    rng = np.random.default_rng(0)
    pairs, values = _pair_values(rng, 30)
    df = _pedigree(pairs, children_per_pair=3)
    out = compute_assortative_mating(df, [_trait(df, values)], permutations=19, bootstrap=50, seed=0)
    m = [values[mo] for mo, _ in pairs]
    f = [values[fa] for _, fa in pairs]
    cell = out["mate_correlation"][0]
    assert cell["n"] == 30
    assert cell["crude"]["pearson"]["r"] == pytest.approx(np.corrcoef(m, f)[0, 1], abs=1e-12)
    assert out["mating_pairs"]["n_total"] == 30
    assert out["mating_pairs"]["n_fathers_multiple_mates"] == sum(n > 1 for n in Counter(f for _, f in pairs).values())


def test_one_network_withholds_the_ci():
    """A cell whose pairs form one Mate Network (a remating chain) publishes no CI."""
    pairs = [(1, 100), (1, 101), (2, 101), (2, 102), (3, 102)]
    df = _pedigree(pairs)
    trait = _trait(df, {1: 0.0, 2: 1.0, 3: 3.0, 100: 0.5, 101: 2.0, 102: 1.0})
    for bootstrap in (100, 0):
        cell = compute_assortative_mating(df, [trait], permutations=0, bootstrap=bootstrap, seed=0)["mate_correlation"][
            0
        ]
        assert (cell["n_mate_networks"], cell["largest_mate_network_share"]) == (1, 1.0)
        pearson_record = cell["crude"]["pearson"]
        assert isinstance(pearson_record["r"], float)
        assert (pearson_record["se"], pearson_record["ci"]) == (None, None)
        assert pearson_record["ci_unavailable_reason"] == "single_mate_network"
        assert pearson_record["p_perm_unavailable_reason"] == "not_requested"


def test_constant_draws_are_counted_as_failures():
    """Draws that resample one network only are constant; they fail, and too many withhold the CI."""
    pairs = [(1, 11), (2, 12), (3, 13)]
    df = _pedigree(pairs)
    trait = _trait(df, {1: 0.0, 2: 1.0, 3: 2.0, 11: 0.5, 12: 0.0, 13: 2.0})
    record = compute_assortative_mating(df, [trait], permutations=0, bootstrap=400, seed=0)["mate_correlation"][0][
        "crude"
    ]["pearson"]
    boot = record["bootstrap"]
    assert set(boot["failure_reasons"]) == {"constant_margin"}
    assert boot["failed"] == boot["failure_reasons"]["constant_margin"] > 0
    assert boot["valid"] + boot["failed"] == 400
    assert record["ci"] is None
    assert record["ci_unavailable_reason"] == "too_many_failed_draws"


def test_all_fixed_fathers_have_no_informative_permutations():
    """With one father per missingness block, no permutation can move a value."""
    pairs = [(1, 11), (2, 12), (3, 12)]
    df = _pedigree(pairs)
    x = _trait(df, {1: 0.0, 2: 1.0, 3: 2.0, 11: 0.5, 12: 1.5})
    y = _trait(df, {1: 1.0, 2: 0.0, 3: 2.0, 11: 1.0}, "y")
    cell = compute_assortative_mating(df, [x, y], permutations=50, bootstrap=0, seed=0)["mate_correlation"][0]
    pearson_record = cell["crude"]["pearson"]
    assert pearson_record["p_perm"] is None
    assert pearson_record["p_perm_unavailable_reason"] == "no_informative_permutations"
    assert pearson_record["permutations"]["n_fixed_fathers"] == 2


def test_unknown_stratum_pairs_are_dropped():
    """With --stratify-by birth_year, a pair with a mate of unknown birth year leaves every cell."""
    rng = np.random.default_rng(2)
    pairs, values = _pair_values(rng, 12)
    df = _pedigree(pairs)
    years = [-1 if i == 1 else 1990 + i % 20 for i in df["id"].to_list()]
    df = df.with_columns(pl.Series("birth_year", years, dtype=pl.Int32))
    out = compute_assortative_mating(
        df, [_trait(df, values)], permutations=9, bootstrap=9, seed=0, stratify_by="birth_year", birth_year_bin=10
    )
    assert out["mating_pairs"]["n_total"] == 12
    assert out["mating_pairs"]["n_dropped"] == {"unknown_stratum": 1}
    cell = out["mate_correlation"][0]
    assert cell["n"] + cell["n_dropped"]["small_stratum"] + cell["n_dropped"]["degenerate_stratum"] == 11
    assert set(cell["stratified"]) == {"pearson"}


def _binary(df: pl.DataFrame, by_id: dict[int, float], name: str = "b", cut: float = 0.0) -> Trait:
    """``by_id`` thresholded at ``cut`` as a binary trait coded 0/1."""
    values = _trait(df, by_id).values
    return Trait(name, "binary", "inferred", ("0", "1"), np.where(np.isnan(values), np.nan, values > cut))


def _ordinal(df: pl.DataFrame, by_id: dict[int, float], cuts: list[float], name: str = "o") -> Trait:
    """``by_id`` binned at ``cuts`` as an ordinal trait coded ``0..len(cuts)``."""
    values = _trait(df, by_id).values
    codes = np.where(np.isnan(values), np.nan, np.digitize(values, cuts))
    return Trait(name, "ordinal", "stated", tuple(str(k) for k in range(len(cuts) + 1)), codes)


@pytest.mark.parametrize("binary", [False, True])
def test_determinism_per_seed(binary):
    """The same seed reproduces every resample; another seed changes them."""
    rng = np.random.default_rng(5)
    pairs, values = _pair_values(rng, 40)
    df = _pedigree(pairs)
    trait = _binary(df, values) if binary else _trait(df, values)
    run = [compute_assortative_mating(df, [trait], permutations=99, bootstrap=99, seed=s) for s in (3, 3, 4)]
    assert run[0] == run[1]
    name, key = ("tetrachoric", "rho") if binary else ("pearson", "r")
    first, other = run[0]["mate_correlation"][0]["crude"][name], run[2]["mate_correlation"][0]["crude"][name]
    assert first[key] == other[key]
    assert first["ci"] != other["ci"]


@settings(deadline=None, max_examples=25)
@given(
    n_pairs=st.integers(2, 25),
    remate_every=st.integers(1, 4),
    seed=st.integers(0, 10_000),
    stratify=st.booleans(),
    binary=st.booleans(),
)
def test_row_order_invariance(n_pairs, remate_every, seed, stratify, binary):
    """Shuffling the pedigree rows changes nothing in the payload, resamples and strata included."""
    rng = np.random.default_rng(seed)
    pairs, values = _pair_values(rng, n_pairs, remate_every)
    df = _pedigree(pairs, children_per_pair=2)
    missing = {i for i in values if rng.random() < 0.15}
    values = {i: v for i, v in values.items() if i not in missing}
    df = _with_birth_years(df, {i: int(rng.integers(1950, 1980)) for i in df["id"].to_list()})
    options = {"stratify_by": "birth_year", "birth_year_bin": 10} if stratify else {}
    draws = 9 if binary else 29

    def run(frame):
        trait = _binary(frame, values) if binary else _trait(frame, values)
        return compute_assortative_mating(frame, [trait], permutations=draws, bootstrap=draws, seed=seed, **options)

    present = _binary(df, values).values if binary else _trait(df, values).values
    assume(len(np.unique(present[~np.isnan(present)])) > 1)  # pg-phenotype refuses a constant trait
    assert run(df[rng.permutation(len(df)).tolist()]) == run(df)


# ---------------------------------------------------------------------------
# Step 4: two traits
# ---------------------------------------------------------------------------


def _one_to_one(n_pairs: int) -> list[tuple[int, int]]:
    return [(k + 1, 100_000 + k) for k in range(n_pairs)]


def _cells(out: dict) -> dict[tuple[str, str], dict]:
    return {(c["mother"], c["father"]): c for c in out["mate_correlation"]}


def test_full_r_mf_recovers_an_asymmetric_matrix():
    """From (mother t1, mother t2, father t1, father t2) normal, every R_mf cell and both within-person r recover."""
    within_m, within_f = 0.4, 0.3
    r_mf = np.array([[0.30, 0.15], [0.05, 0.25]])
    cov = np.block(
        [[np.array([[1, within_m], [within_m, 1]]), r_mf], [r_mf.T, np.array([[1, within_f], [within_f, 1]])]]
    )
    n = 20_000
    rng = np.random.default_rng(11)
    draws = rng.multivariate_normal(np.zeros(4), cov, size=n)
    pairs = _one_to_one(n)
    df = _pedigree(pairs)
    t1 = {m: draws[k, 0] for k, (m, _) in enumerate(pairs)} | {f: draws[k, 2] for k, (_, f) in enumerate(pairs)}
    t2 = {m: draws[k, 1] for k, (m, _) in enumerate(pairs)} | {f: draws[k, 3] for k, (_, f) in enumerate(pairs)}
    out = compute_assortative_mating(
        df, [_trait(df, t1, "t1"), _trait(df, t2, "t2")], permutations=0, bootstrap=0, seed=0
    )

    # SE of r is about (1 - r^2) / sqrt(n) < 0.0075; 0.03 is four of them.
    cells = _cells(out)
    for i, mother in enumerate(("t1", "t2")):
        for j, father in enumerate(("t1", "t2")):
            assert cells[mother, father]["n"] == n
            assert cells[mother, father]["crude"]["pearson"]["r"] == pytest.approx(r_mf[i, j], abs=0.03)
    assert out["within_person"]["mothers"]["r"] == pytest.approx(within_m, abs=0.03)
    assert out["within_person"]["fathers"]["r"] == pytest.approx(within_f, abs=0.03)
    assert out["within_person"]["mothers"]["estimator"] == "pearson"


def test_each_cell_uses_its_pairwise_complete_pairs():
    """Each cell keeps the pairs with its own two values, reports that n, and counts the rest by side."""
    rng = np.random.default_rng(12)
    pairs = _one_to_one(60)
    df = _pedigree(pairs)
    ids = [p for pair in pairs for p in pair]
    x = {i: v for i, v in zip(ids, rng.normal(size=len(ids)), strict=True) if rng.random() > 0.2}
    y = {i: v for i, v in zip(ids, rng.normal(size=len(ids)), strict=True) if rng.random() > 0.3}
    out = compute_assortative_mating(df, [_trait(df, x, "x"), _trait(df, y, "y")], permutations=0, bootstrap=0, seed=0)
    for (mother, father), cell in _cells(out).items():
        mv, fv = {"x": x, "y": y}[mother], {"x": x, "y": y}[father]
        complete = [(mv[m], fv[f]) for m, f in pairs if m in mv and f in fv]
        assert cell["n"] == len(complete)
        assert cell["n_dropped"] == {
            "mother_missing": sum(m not in mv and f in fv for m, f in pairs),
            "father_missing": sum(m in mv and f not in fv for m, f in pairs),
            "both_missing": sum(m not in mv and f not in fv for m, f in pairs),
            "small_stratum": 0,
            "degenerate_stratum": 0,
        }
        assert cell["crude"]["pearson"]["r"] == pytest.approx(np.corrcoef(np.array(complete).T)[0, 1], abs=1e-12)


def test_within_person_counts_each_eligible_individual_once():
    """A remating father counts once; children and parents missing a trait do not count."""
    pairs = [(1, 10), (2, 10), (3, 10), (4, 11), (5, 12), (6, 13)]
    df = _pedigree(pairs)
    x = {1: 0.1, 2: 0.9, 3: 0.4, 4: 1.2, 5: -0.3, 6: 0.7, 10: 1.0, 11: 0.2, 12: -1.0, 13: 0.5, 14: 3.0}
    y = {1: 0.3, 2: 0.5, 3: -0.2, 4: 1.0, 5: 0.0, 10: 2.0, 11: 0.1, 12: -0.5, 13: 1.5, 14: -3.0}
    out = compute_assortative_mating(df, [_trait(df, x, "x"), _trait(df, y, "y")], permutations=0, bootstrap=0, seed=0)
    mothers, fathers = [1, 2, 3, 4, 5], [10, 11, 12, 13]
    assert out["within_person"] == {
        "mothers": {
            "estimator": "pearson",
            "r": pytest.approx(np.corrcoef([x[i] for i in mothers], [y[i] for i in mothers])[0, 1], abs=1e-12),
            "n": 5,
        },
        "fathers": {
            "estimator": "pearson",
            "r": pytest.approx(np.corrcoef([x[i] for i in fathers], [y[i] for i in fathers])[0, 1], abs=1e-12),
            "n": 4,
        },
    }


# ---------------------------------------------------------------------------
# Step 5: stratification
# ---------------------------------------------------------------------------


def _with_birth_years(df: pl.DataFrame, years: dict[int, int]) -> pl.DataFrame:
    return df.with_columns(pl.Series("birth_year", [years.get(i, 2000) for i in df["id"].to_list()], dtype=pl.Int32))


def _manual_standardise(x: np.ndarray, code: np.ndarray) -> np.ndarray:
    z = np.empty_like(x)
    for c in np.unique(code):
        sel = code == c
        z[sel] = (x[sel] - x[sel].mean()) / x[sel].std()
    return z


def _stratified_fixture(
    seed: int, n_pairs: int = 80
) -> tuple[pl.DataFrame, list[tuple[int, int]], dict[int, float], dict[int, int]]:
    """Remating pairs whose mates' birth years fall in a few decades, one value per individual."""
    rng = np.random.default_rng(seed)
    pairs, values = _pair_values(rng, n_pairs)
    df = _pedigree(pairs)
    years = {i: int(rng.integers(1950, 1980)) for i in values}
    return _with_birth_years(df, years), pairs, values, years


def test_stratified_pearson_uses_pair_weighted_margins_and_the_crude_sample():
    """The stratified r standardises over one entry per pair (a remating father counts per mate), on the crude sample."""
    df, pairs, values, years = _stratified_fixture(14)
    out = compute_assortative_mating(
        df, [_trait(df, values)], permutations=19, bootstrap=19, seed=0, stratify_by="birth_year", birth_year_bin=10
    )
    cell = out["mate_correlation"][0]
    assert cell["n_dropped"]["degenerate_stratum"] == 0
    m = np.array([values[mo] for mo, _ in pairs])
    f = np.array([values[fa] for _, fa in pairs])
    ms = np.array([years[mo] // 10 for mo, _ in pairs])
    fs = np.array([years[fa] // 10 for _, fa in pairs])
    stratified = cell["stratified"]["pearson"]
    assert cell["n"] == len(pairs)
    assert cell["crude"]["pearson"]["r"] == pytest.approx(np.corrcoef(m, f)[0, 1], abs=1e-12)
    assert stratified["r"] == pytest.approx(
        np.corrcoef(_manual_standardise(m, ms), _manual_standardise(f, fs))[0, 1], abs=1e-12
    )
    assert (stratified["n_strata_mothers"], stratified["n_strata_fathers"]) == (3, 3)
    assert stratified["bootstrap"]["requested"] == stratified["permutations"]["requested"] == 19
    assert stratified["p_perm"] is not None


@pytest.mark.parametrize(("min_networks", "n_small", "n_degenerate"), [(1, 0, 3), (MIN_STRATUM_NETWORKS, 3, 0)])
def test_degenerate_strata_leave_both_crude_and_stratified(min_networks, n_small, n_degenerate):
    """Pairs in a sex x stratum with one pair or one value leave the cell's crude and stratified sample alike.

    With a one-network minimum only the degenerate rule drops them; at the
    default both strata are also small, and small takes the count.
    """
    rng = np.random.default_rng(15)
    pairs = _one_to_one(30)
    df = _pedigree(pairs)
    values = {p: float(v) for p, v in zip([p for pair in pairs for p in pair], rng.normal(size=60), strict=True)}
    years = {m: 1960 + 10 * (k % 2) for k, (m, _) in enumerate(pairs)} | {
        f: 1960 + 10 * (k % 2) for k, (_, f) in enumerate(pairs)
    }
    years[pairs[0][0]] = 1990  # a mother stratum of one pair
    for _, f in pairs[1:3]:  # a father stratum of two pairs with one value
        years[f] = 2000
        values[f] = 0.25
    df = _with_birth_years(df, years)
    out = compute_assortative_mating(
        df,
        [_trait(df, values)],
        permutations=0,
        bootstrap=0,
        seed=0,
        stratify_by="birth_year",
        birth_year_bin=10,
        min_stratum_networks=min_networks,
    )
    cell = out["mate_correlation"][0]
    kept = pairs[3:]
    assert (cell["n_dropped"]["small_stratum"], cell["n_dropped"]["degenerate_stratum"]) == (n_small, n_degenerate)
    assert cell["n"] == len(kept)
    assert "degenerate_stratum" not in out["mating_pairs"]["n_dropped"]
    m = [values[mo] for mo, _ in kept]
    f = [values[fa] for _, fa in kept]
    assert cell["crude"]["pearson"]["r"] == pytest.approx(np.corrcoef(m, f)[0, 1], abs=1e-12)
    assert cell["stratified"]["pearson"]["n_strata_mothers"] == 2


def test_small_strata_leave_both_crude_and_stratified():
    """A sex x stratum spanning fewer Mate Networks than the minimum leaves both samples and counts as small."""
    rng = np.random.default_rng(20)
    pairs = _one_to_one(30)
    df = _pedigree(pairs)
    values = {p: float(v) for p, v in zip([p for pair in pairs for p in pair], rng.normal(size=60), strict=True)}
    years = {p: 1960 + 10 * (k % 2) for k, pair in enumerate(pairs) for p in pair}
    for m, _ in pairs[:2]:  # a mother stratum of two networks, values distinct
        years[m] = 1990
    for _, f in pairs[2:5]:  # a father stratum of three networks with one value
        years[f] = 2000
        values[f] = 0.25
    df = _with_birth_years(df, years)
    cell = compute_assortative_mating(
        df,
        [_trait(df, values)],
        permutations=0,
        bootstrap=0,
        seed=0,
        stratify_by="birth_year",
        birth_year_bin=10,
        min_stratum_networks=3,
    )["mate_correlation"][0]
    kept = pairs[5:]
    assert (cell["n_dropped"]["small_stratum"], cell["n_dropped"]["degenerate_stratum"]) == (2, 3)
    assert cell["n"] == len(kept)
    m = np.array([values[mo] for mo, _ in kept])
    f = np.array([values[fa] for _, fa in kept])
    code = np.array([k % 2 for k in range(5, 30)])
    assert cell["crude"]["pearson"]["r"] == pytest.approx(np.corrcoef(m, f)[0, 1], abs=1e-12)
    assert cell["stratified"]["pearson"]["r"] == pytest.approx(
        np.corrcoef(_manual_standardise(m, code), _manual_standardise(f, code))[0, 1], abs=1e-12
    )
    assert cell["stratified"]["pearson"]["n_strata_mothers"] == 2


def test_min_stratum_networks_is_recorded_only_when_stratified():
    """Settings carry the minimum (default 10) under stratification and null without it."""
    df, _, values, _ = _stratified_fixture(21, n_pairs=20)
    trait = [_trait(df, values)]
    stratified = compute_assortative_mating(
        df, trait, permutations=0, bootstrap=0, seed=0, stratify_by="birth_year", birth_year_bin=10
    )
    crude = compute_assortative_mating(df, trait, permutations=0, bootstrap=0, seed=0)
    assert stratified["settings"]["min_stratum_networks"] == MIN_STRATUM_NETWORKS == 10
    assert crude["settings"]["min_stratum_networks"] is None
    assert crude["mate_correlation"][0]["n_dropped"]["small_stratum"] == 0


def test_degenerate_stratum_in_a_draw_fails_that_draw():
    """A draw that leaves a stratum constant fails the stratified statistic only, counted by reason."""
    rng = np.random.default_rng(16)
    pairs = _one_to_one(24)
    df = _pedigree(pairs)
    values = {p: float(v) for p, v in zip([p for pair in pairs for p in pair], rng.normal(size=48), strict=True)}
    years = {p: 1900 + 10 * (k // 2) for k, pair in enumerate(pairs) for p in pair}
    df = _with_birth_years(df, years)
    cell = compute_assortative_mating(
        df,
        [_trait(df, values)],
        permutations=0,
        bootstrap=200,
        seed=0,
        stratify_by="birth_year",
        birth_year_bin=10,
        min_stratum_networks=1,
    )["mate_correlation"][0]
    stratified = cell["stratified"]["pearson"]["bootstrap"]
    assert stratified["failure_reasons"]["degenerate_stratum"] == stratified["failed"] > 0
    assert "degenerate_stratum" not in cell["crude"]["pearson"]["bootstrap"]["failure_reasons"]


def test_permutations_stay_inside_father_strata():
    """When stratum x missingness leaves each father alone in his block, stratification turns permutations off."""
    rng = np.random.default_rng(17)
    pairs = _one_to_one(40)
    df = _pedigree(pairs)
    x = {p: float(v) for p, v in zip([p for pair in pairs for p in pair], rng.normal(size=80), strict=True)}
    # Two fathers per decade, one with y and one without: one father per stratum x missingness block.
    years = {f: 1800 + 10 * (k // 2) for k, (_, f) in enumerate(pairs)} | {m: 2000 for m, _ in pairs}
    y = {m: 1.0 + k for k, (m, _) in enumerate(pairs)} | {f: 1.0 for k, (_, f) in enumerate(pairs) if k % 2}
    df = _with_birth_years(df, years)
    traits = [_trait(df, x, "x"), _trait(df, y, "y")]

    def p_record(**options):
        out = compute_assortative_mating(df, traits, permutations=49, bootstrap=0, seed=0, **options)
        return _cells(out)["x", "x"]

    crude_only = p_record()["crude"]["pearson"]
    assert crude_only["p_perm"] is not None
    stratified_cell = p_record(stratify_by="birth_year", birth_year_bin=10, min_stratum_networks=1)
    for record in (stratified_cell["crude"]["pearson"], stratified_cell["stratified"]["pearson"]):
        assert record["p_perm"] is None
        assert record["p_perm_unavailable_reason"] == "no_informative_permutations"
        assert record["permutations"]["n_fixed_fathers"] == 40


def test_cohort_trend_without_assortment_vanishes_when_stratified():
    """Mates share a cohort whose mean moves for both sexes; the crude r picks it up, the stratified r does not."""
    rng = np.random.default_rng(18)
    n = 12_000
    pairs = _one_to_one(n)
    df = _pedigree(pairs)
    cohort = rng.integers(0, 6, n)
    m = cohort + rng.normal(size=n)
    f = cohort + rng.normal(size=n)
    values = {mo: m[k] for k, (mo, _) in enumerate(pairs)} | {fa: f[k] for k, (_, fa) in enumerate(pairs)}
    years = {p: 1950 + 10 * int(cohort[k]) for k, pair in enumerate(pairs) for p in pair}
    df = _with_birth_years(df, years)
    cell = compute_assortative_mating(
        df, [_trait(df, values)], permutations=0, bootstrap=0, seed=0, stratify_by="birth_year", birth_year_bin=10
    )["mate_correlation"][0]
    # Crude r is var(cohort) / (var(cohort) + 1) = 0.74 in expectation; the stratified SE is about 0.009.
    assert cell["crude"]["pearson"]["r"] > 0.6
    assert abs(cell["stratified"]["pearson"]["r"]) < 0.04


# ---------------------------------------------------------------------------
# Step 3: threshold estimators in the payload
# ---------------------------------------------------------------------------


def test_constant_margin_cell_in_the_payload():
    """A cell whose analysed mothers all share one level reports the undefined estimators by reason."""
    pairs = _one_to_one(20)
    df = _pedigree(pairs)
    values = {m: 0.0 for m, _ in pairs} | {f: float(k % 2) for k, (_, f) in enumerate(pairs)}
    trait = Trait("b", "binary", "inferred", ("0", "1"), _trait(df, values).values)
    cell = compute_assortative_mating(df, [trait], permutations=9, bootstrap=9, seed=0)["mate_correlation"][0]
    assert cell["crude"]["table"] == [[10, 10], [0, 0]]
    assert cell["crude"]["tetrachoric"] == {"rho": None, "reason": "constant_margin"}
    assert cell["crude"]["odds_ratio"] == {"value": None, "reason": "constant_margin"}
    assert cell["crude"]["phi"] == {"r": None, "reason": "constant_margin"}


def _mates(rng, r_mf, n, within_m=0.4, within_f=0.3) -> tuple[pl.DataFrame, dict[int, float], dict[int, float]]:
    """(mother t1, mother t2, father t1, father t2) draws from a 4-variate normal with mate matrix ``r_mf``."""
    cov = np.block(
        [[np.array([[1, within_m], [within_m, 1]]), r_mf], [r_mf.T, np.array([[1, within_f], [within_f, 1]])]]
    )
    draws = rng.multivariate_normal(np.zeros(4), cov, size=n)
    pairs = _one_to_one(n)
    t1 = {m: draws[k, 0] for k, (m, _) in enumerate(pairs)} | {f: draws[k, 2] for k, (_, f) in enumerate(pairs)}
    t2 = {m: draws[k, 1] for k, (m, _) in enumerate(pairs)} | {f: draws[k, 3] for k, (_, f) in enumerate(pairs)}
    return _pedigree(pairs), t1, t2


def test_large_n_recovers_rho_from_binary_and_ordinal_mates():
    """Tetrachoric and polychoric cells and the within-person polychoric recover the latent R_mf within 0.04 at n = 20000."""
    r_mf = np.array([[0.30, 0.15], [0.05, 0.25]])
    df, t1, t2 = _mates(np.random.default_rng(21), r_mf, 20_000)
    traits = [_binary(df, t1, "b", cut=0.5), _ordinal(df, t2, [-1.0, 0.0, 1.0], "o")]
    out = compute_assortative_mating(df, traits, permutations=0, bootstrap=0, seed=0)
    cells = _cells(out)
    # Latent-correlation SEs at n = 20000 are below 0.02; 0.04 is at least two of them.
    assert cells["b", "b"]["crude"]["tetrachoric"]["rho"] == pytest.approx(0.30, abs=0.04)
    assert cells["b", "o"]["crude"]["polychoric"]["rho"] == pytest.approx(0.15, abs=0.04)
    assert cells["o", "b"]["crude"]["polychoric"]["rho"] == pytest.approx(0.05, abs=0.04)
    assert cells["o", "o"]["crude"]["polychoric"]["rho"] == pytest.approx(0.25, abs=0.04)
    assert all(
        not c["crude"][name]["boundary"]
        for c in cells.values()
        for name in ("tetrachoric", "polychoric")
        if name in c["crude"]
    )
    assert cells["b", "b"]["crude"]["phi"]["r"] < cells["b", "b"]["crude"]["tetrachoric"]["rho"]
    assert cells["b", "b"]["crude"]["odds_ratio"]["value"] > 1
    assert cells["b", "b"]["crude"]["table"] == [[int(v) for v in row] for row in cells["b", "b"]["crude"]["table"]]
    assert sum(map(sum, cells["b", "b"]["crude"]["table"])) == 20_000
    assert out["within_person"]["mothers"]["estimator"] == "polychoric"
    assert out["within_person"]["mothers"]["rho"] == pytest.approx(0.4, abs=0.04)
    assert out["within_person"]["fathers"]["rho"] == pytest.approx(0.3, abs=0.04)


def test_large_n_recovers_rho_from_mixed_mates():
    """Biserial and polyserial cells in both orientations recover the latent R_mf within 0.04 at n = 20000."""
    r_mf = np.array([[0.30, 0.15], [0.05, 0.25]])
    df, t1, t2 = _mates(np.random.default_rng(22), r_mf, 20_000)
    binary = compute_assortative_mating(
        df, [_trait(df, t1, "x"), _binary(df, t2, "b", cut=0.8)], permutations=0, bootstrap=0, seed=0
    )
    cells = _cells(binary)
    assert cells["x", "b"]["crude"]["biserial"]["rho"] == pytest.approx(0.15, abs=0.04)
    assert cells["b", "x"]["crude"]["biserial"]["rho"] == pytest.approx(0.05, abs=0.04)
    assert cells["b", "b"]["crude"]["tetrachoric"]["rho"] == pytest.approx(0.25, abs=0.04)
    assert abs(cells["x", "b"]["crude"]["point_biserial"]["r"]) < abs(cells["x", "b"]["crude"]["biserial"]["rho"])
    assert binary["within_person"]["mothers"]["estimator"] == "biserial"
    assert binary["within_person"]["mothers"]["rho"] == pytest.approx(0.4, abs=0.04)

    ordinal = compute_assortative_mating(
        df, [_ordinal(df, t1, [-0.5, 0.5], "o"), _trait(df, t2, "x")], permutations=0, bootstrap=0, seed=0
    )
    cells = _cells(ordinal)
    assert cells["o", "x"]["crude"]["polyserial"]["rho"] == pytest.approx(0.15, abs=0.04)
    assert cells["x", "o"]["crude"]["polyserial"]["rho"] == pytest.approx(0.05, abs=0.04)
    assert set(cells["o", "x"]["crude"]) == {"polyserial"}
    assert ordinal["within_person"]["fathers"]["estimator"] == "polyserial"
    assert ordinal["within_person"]["fathers"]["rho"] == pytest.approx(0.3, abs=0.04)


def test_within_person_mixed_kinds_count_a_remating_individual_once():
    """The within-person biserial is individual-level: father 10 with three mates counts once."""
    pairs = [(1, 10), (2, 10), (3, 10), (4, 11), (5, 12), (6, 13)]
    df = _pedigree(pairs)
    x = {1: 0.1, 2: 0.9, 3: 0.4, 4: 1.2, 5: -0.3, 6: 0.7, 10: 1.0, 11: 0.2, 12: -1.0, 13: 0.5}
    b = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 10: 1, 11: 0, 12: 0, 13: 1}
    out = compute_assortative_mating(
        df, [_trait(df, x, "x"), _binary(df, b, "b", cut=0.5)], permutations=0, bootstrap=0, seed=0
    )
    fathers = out["within_person"]["fathers"]
    assert list(fathers) == ["estimator", "n", "rho", "boundary"]
    assert (fathers["estimator"], fathers["n"]) == ("biserial", 4)
    assert -1 < fathers["rho"] < 1
    assert out["within_person"]["mothers"]["n"] == 6


def test_empty_category_in_a_draw_is_counted():
    """A level present in the cell but absent from a draw's margin fails that draw with reason empty_category."""
    pairs = _one_to_one(30)
    df = _pedigree(pairs)
    rng = np.random.default_rng(23)
    values = {p: float(rng.integers(0, 2)) for pair in pairs for p in pair}
    values[pairs[0][1]] = 2.0  # the only father at level 2
    trait = Trait("o", "ordinal", "stated", ("0", "1", "2"), _trait(df, values).values)
    record = compute_assortative_mating(df, [trait], permutations=0, bootstrap=200, seed=0)["mate_correlation"][0][
        "crude"
    ]["polychoric"]
    boot = record["bootstrap"]
    assert boot["failure_reasons"]["empty_category"] == boot["failed"] > 0
    assert boot["valid"] + boot["failed"] == 200


def test_odds_ratio_inf_is_a_value_and_the_ci_is_never_nan():
    """A zero off-diagonal gives .inf; bootstrap bounds are order statistics, so a draw at inf gives an inf bound, not NaN."""
    ped = _pedigree(_one_to_one(60))
    rows = [(0, 0)] * 30 + [(0, 1)] * 10 + [(1, 1)] * 20
    values = {m: float(mv) for (m, _), (mv, _) in zip(_one_to_one(60), rows, strict=True)} | {
        f: float(fv) for (_, f), (_, fv) in zip(_one_to_one(60), rows, strict=True)
    }
    trait = Trait("b", "binary", "inferred", ("0", "1"), _trait(ped, values).values)
    out = compute_assortative_mating(ped, [trait], permutations=0, bootstrap=200, seed=0)
    cell = out["mate_correlation"][0]
    odds = cell["crude"]["odds_ratio"]
    assert odds["value"] == np.inf
    assert cell["crude"]["table"] == [[30, 10], [0, 20]]
    # No pair has (mother 1, father 0), so no draw can either: every draw is inf and so is the whole CI.
    assert odds["ci"] == [np.inf, np.inf]
    assert odds["bootstrap"]["failed"] == 0
    assert cell["crude"]["tetrachoric"]["boundary"] is True
    assert odds["se"] is None
    path = _write_yaml_tmp(out)
    text = path.read_text()
    assert ".inf" in text
    assert "nan" not in text.lower()


def _write_yaml_tmp(payload: dict) -> Path:
    path = Path(tempfile.mkdtemp()) / "am.yaml"
    _write_yaml({"assortative_mating": payload}, path, ASSORTATIVE_MATING_FIGURES)
    return path


def test_stratified_tetrachoric_in_the_payload():
    """With --stratify-by, the binary cell reports a stratified tetrachoric with per-stratum thresholds and one ρ."""
    rng = np.random.default_rng(24)
    n = 400
    pairs = _one_to_one(n)
    df = _pedigree(pairs)
    cohort = rng.integers(0, 2, n)
    draws = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], size=n) + cohort[:, None]
    values = {m: draws[k, 0] for k, (m, _) in enumerate(pairs)} | {f: draws[k, 1] for k, (_, f) in enumerate(pairs)}
    years = {p: 1950 + 10 * int(cohort[k]) for k, pair in enumerate(pairs) for p in pair}
    df = _with_birth_years(df, years)
    trait = _binary(df, values, cut=0.5)
    cell = compute_assortative_mating(
        df, [trait], permutations=19, bootstrap=19, seed=0, stratify_by="birth_year", birth_year_bin=10
    )["mate_correlation"][0]
    stratified = cell["stratified"]["tetrachoric"]
    assert set(cell["stratified"]) == {"tetrachoric"}
    assert -1 < stratified["rho"] < 1
    assert stratified["boundary"] is False
    assert (stratified["n_strata_mothers"], stratified["n_strata_fathers"]) == (2, 2)
    assert stratified["p_perm"] is not None
    assert stratified["rho"] < cell["crude"]["tetrachoric"]["rho"]


def test_pg_phenotype_refusals_raise_pedigree_error():
    """A constant trait or three traits, which the CLI never passes, come back as ``PedigreeError``."""
    df = _pedigree(_one_to_one(10))
    constant = Trait("c", "binary", "stated", ("0", "1"), np.where(df["mother"].to_numpy() == -1, 1.0, np.nan))
    with pytest.raises(PedigreeError, match="constant"):
        compute_assortative_mating(df, [constant], permutations=0, bootstrap=0, seed=0)
    varied = _trait(df, {i: float(i % 7) for i in df["id"].to_list()})
    with pytest.raises(PedigreeError, match="traits"):
        compute_assortative_mating(df, [varied] * 3, permutations=0, bootstrap=0, seed=0)
