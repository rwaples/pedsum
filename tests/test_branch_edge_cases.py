"""Focused edge-case coverage for important pedsum branch behavior."""

from __future__ import annotations

import gzip
from argparse import ArgumentTypeError
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pedsum.base import PedigreeError
from pedsum.checks import (
    Finding,
    _check_birth_year_topology,
    _check_duplicate_ids,
    _check_empty_pedigree,
    _check_parent_refs_present,
    _check_sex_role_consistency,
    _summarize_findings,
)
from pedsum.cli import _positive_int
from pedsum.pairs import _build_pedigree_graph, _count_pairs_matrix_with_lists
from pedsum.parse import (
    _as_parent_int_col,
    _decode_sex,
    _format_id_sample,
    _maybe_warn_csv,
    _read_pedigree_table,
    _sniff_delimiter,
)
from pedsum.pedigree_ops import _compute_depth_unordered, _full_sib_groups, _id_list
from pedsum.report import (
    _apply_safe_attempt,
    _build_individual_data,
    _build_pedigree_data,
    _build_summary_data,
    _drop_distribution_extrema,
    _flatten_long,
    _prepare_out_dir,
    _round_floats,
    _write_annotated_tsv,
    _write_dropped_manifest,
    _write_validate_tsv_gz,
)
from pedsum.schema import (
    SectionSpec,
    _categorise_pedigree,
    _drop_dotted_path,
    _split_effective_size,
    _split_individual_distributions,
    _split_section,
    _split_sex_summary,
)
from pedsum.sections import (
    _build_inbreeding_summary,
    _effective_count_from_weights,
    build_individual_df,
    compute_aggregate_sections,
    compute_founder_summary,
    compute_mating_pair_summary,
    compute_relationship_summary,
    compute_sibship_sizes,
    compute_size_structure,
)


def test_positive_int_rejects_zero_and_accepts_positive() -> None:
    """The CLI positive-int type accepts positives and rejects non-positive ints."""
    assert _positive_int("3") == 3
    with pytest.raises(ArgumentTypeError, match=">= 1"):
        _positive_int("0")


class TestParseEdgeCases:
    """Direct parser edge cases that are awkward through the subprocess CLI."""

    def test_id_sample_empty_and_deterministic_subset(self) -> None:
        """ID samples handle empty inputs and return deterministic bounded samples."""
        assert _format_id_sample(np.array([], dtype=np.int64)) == ""
        assert _format_id_sample(np.arange(10), k=3) == "5, 6, 9"

    def test_decode_sex_explicit_default_plink_and_bad_encoding(self) -> None:
        """Explicit sex encodings and invalid encoding names are handled distinctly."""
        default = _decode_sex(pd.Series(["0", "1", ""]), encoding="default")
        plink = _decode_sex(pd.Series(["2", "1", "0"]), encoding="plink")
        np.testing.assert_array_equal(default, np.array([0, 1, -1], dtype=np.int8))
        np.testing.assert_array_equal(plink, np.array([0, 1, -1], dtype=np.int8))
        with pytest.raises(PedigreeError, match="unknown sex encoding"):
            _decode_sex(pd.Series(["F"]), encoding="mystery")

    def test_maybe_warn_csv_allows_normal_input_and_rejects_csv_shape(self) -> None:
        """The friendly CSV warning distinguishes split tables from one-column CSV-looking input."""
        _maybe_warn_csv(pd.DataFrame({"id": [1], "sex": ["F"]}))
        with pytest.raises(PedigreeError, match="appears to be CSV"):
            _maybe_warn_csv(pd.DataFrame({"id,sex,mother,father": ["1,F,-1,-1"]}))

    def test_sniff_delimiter_handles_gzip_empty_whitespace_and_fallback(self, tmp_path) -> None:
        """Delimiter sniffing covers compressed, empty, whitespace, and tab-fallback inputs."""
        gz_path = tmp_path / "ped.tsv.gz"
        with gzip.open(gz_path, "wt") as fh:
            fh.write("id,sex,mother,father\n1,F,-1,-1\n")
        assert _sniff_delimiter(gz_path) == ","

        empty = tmp_path / "empty.tsv"
        empty.write_text("\n\n")
        with pytest.raises(PedigreeError, match="empty"):
            _sniff_delimiter(empty)

        whitespace = tmp_path / "ped.fam"
        whitespace.write_text("id sex mother father\n1 F -1 -1\n")
        assert _sniff_delimiter(whitespace) == r"\s+"

        single_token = tmp_path / "single.txt"
        single_token.write_text("id\n1\n")
        assert _sniff_delimiter(single_token) == "\t"

    def test_read_pedigree_table_missing_and_explicit_sep(self, tmp_path) -> None:
        """Table reading reports missing files and honors explicit separators."""
        with pytest.raises(PedigreeError, match="input file not found"):
            _read_pedigree_table(tmp_path / "missing.tsv")
        table = tmp_path / "ped.psv"
        table.write_text("id|sex|mother|father\n1|F|-1|-1\n")
        df = _read_pedigree_table(table, sep="pipe", dtype=str)
        assert df.to_dict("records") == [{"id": "1", "sex": "F", "mother": "-1", "father": "-1"}]

        comma = tmp_path / "ped.csv"
        comma.write_text("id,sex,mother,father\n1,F,-1,-1\n")
        auto = _read_pedigree_table(comma, dtype=str)
        assert auto.to_dict("records") == [{"id": "1", "sex": "F", "mother": "-1", "father": "-1"}]

    def test_parent_int_col_zero_as_missing(self) -> None:
        """PLINK-style zero parent IDs are normalised to -1 when requested."""
        arr = _as_parent_int_col(pd.Series(["0", "2", "NA"]), "mother", zero_as_missing=True)
        np.testing.assert_array_equal(arr, np.array([-1, 2, -1], dtype=np.int64))


class TestCheckEdgeCases:
    """Validation check producers cover clean, offending, and formatting cases."""

    def test_duplicate_ids_clean_and_offending(self) -> None:
        """Duplicate-ID checking returns no findings for unique IDs and one per duplicate value."""
        assert _check_duplicate_ids(np.array([1, 2, 3], dtype=np.int64)) == []
        findings = _check_duplicate_ids(np.array([2, 1, 2, 3, 1], dtype=np.int64))
        assert [f.id for f in findings] == [1, 2]

    def test_parent_refs_present_absent_clean_missing_and_zero_hint(self) -> None:
        """Missing-parent detection distinguishes no refs, all-present refs, and the zero-token hint."""
        id_index = pd.Index([1, 2, 3])
        assert _check_parent_refs_present(np.array([-1, -1]), "mother", id_index) == []
        assert _check_parent_refs_present(np.array([1, -1, 2]), "mother", id_index) == []
        findings = _check_parent_refs_present(np.array([0, 99, 99]), "father", id_index)
        assert [f.id for f in findings] == [0, 99]
        assert "convert to -1" in findings[0].detail

    def test_empty_pedigree_check(self) -> None:
        """Empty-pedigree checking fires only for n == 0."""
        assert _check_empty_pedigree(3) == []
        assert _check_empty_pedigree(0)[0].check == "empty_pedigree"

    def test_sex_role_consistency_without_skip_mask(self) -> None:
        """Sex-role consistency works when no imputation skip mask is supplied."""
        ids = np.array([1, 2, 3], dtype=np.int64)
        mothers = np.array([-1, -1, 2], dtype=np.int64)
        fathers = np.array([-1, -1, 1], dtype=np.int64)
        sex = np.array([1, 1, 0], dtype=np.int8)  # id=2 is male but used as mother.
        findings = _check_sex_role_consistency(mothers, fathers, sex, pd.Index(ids))
        assert len(findings) == 1
        assert findings[0].id == 2
        assert "used as mother" in findings[0].detail

    def test_summarize_findings_variants(self) -> None:
        """Finding summaries render empty, id-only, row-only, and overflow samples."""
        assert _summarize_findings([]) == ""
        assert "id=7" in _summarize_findings([Finding(check="c", id=7)])
        assert "row 4" in _summarize_findings([Finding(check="c", row=4)])
        many = [Finding(check="c", id=i, row=i) for i in range(7)]
        assert "and 2 more" in _summarize_findings(many)

    def test_birth_year_topology_no_edges_and_bad_edge(self) -> None:
        """Birth-year topology skips no-edge pedigrees and flags child-before-parent years."""
        ids = np.array([1, 2], dtype=np.int64)
        no_edges = _check_birth_year_topology(
            ids, np.array([-1, -1]), np.array([-1, -1]), np.array([1980, 2000]), pd.Index(ids)
        )
        assert no_edges == []
        bad = _check_birth_year_topology(
            ids, np.array([-1, 1]), np.array([-1, -1]), np.array([2000, 1990]), pd.Index(ids)
        )
        assert len(bad) == 1
        assert bad[0].check == "birth_year_topology"


class TestPedigreeOpsAndPairsEdgeCases:
    """Low-level pedigree helper edge cases."""

    def test_id_list_short_and_long(self) -> None:
        """ID list rendering shows all short inputs and truncates long inputs."""
        assert _id_list([3, 4]) == "3, 4"
        assert _id_list(range(7), max_show=3) == "0, 1, 2, ... (7 total)"

    def test_full_sib_groups_empty_and_present_parent_pairs(self) -> None:
        """Full-sib grouping handles no-full-parent and full-parent sibling groups."""
        df = pd.DataFrame({"mother": [-1, 1], "father": [-1, -1]})
        counts, groups, both_present = _full_sib_groups(df)
        np.testing.assert_array_equal(counts, np.array([0, 0], dtype=np.int64))
        assert groups.empty
        assert not both_present.any()

        sibs = pd.DataFrame({"mother": [-1, -1, 1, 1], "father": [-1, -1, 2, 2]})
        counts, groups, both_present = _full_sib_groups(sibs)
        np.testing.assert_array_equal(counts, np.array([0, 0, 1, 1], dtype=np.int64))
        assert int(groups.loc[(1, 2)]) == 2
        assert both_present.tolist() == [False, False, True, True]

    def test_compute_depth_unordered_detects_cycles(self) -> None:
        """Depth computation raises when parent rows form a true cycle."""
        depth = _compute_depth_unordered(np.array([-1, 0]), np.array([-1, -1]), 2)
        np.testing.assert_array_equal(depth, np.array([0, 1], dtype=np.int32))
        with pytest.raises(PedigreeError, match="cycle"):
            _compute_depth_unordered(np.array([1, 0]), np.array([-1, -1]), 2)

    def test_count_pairs_matrix_builds_graph_when_not_supplied(self) -> None:
        """The matrix pair counter builds a PedigreeGraph when the caller does not pass one."""
        df = pd.DataFrame(
            {
                "id": [10, 20, 30],
                "sex": [1, 0, 1],
                "mother": [-1, -1, 20],
                "father": [-1, -1, 10],
            }
        )
        out = _count_pairs_matrix_with_lists(df)
        assert out["PO"] == out.get("MO", 0) + out.get("FO", 0)
        assert "_pair_lists" in out

    def test_count_pairs_matrix_reuses_supplied_graph(self) -> None:
        """The matrix pair counter accepts a prebuilt PedigreeGraph."""
        df = pd.DataFrame(
            {
                "id": [10, 20, 30],
                "sex": [1, 0, 1],
                "mother": [-1, -1, 20],
                "father": [-1, -1, 10],
            }
        )
        pg = _build_pedigree_graph(df)
        out = _count_pairs_matrix_with_lists(df, pg=pg)
        assert out["PO"] == out.get("MO", 0) + out.get("FO", 0)


class TestSchemaEdgeCases:
    """Summary schema split edge cases."""

    def test_categorise_pedigree_merges_pair_engine_and_drops_empty_sections(self) -> None:
        """Categorisation folds pair engine into relationship pairs and skips empty sections."""
        nested = _categorise_pedigree(
            {"relationship_pairs": {"FS": 2}, "pairs_engine": "matrix", "components": {}, "sex_summary": []}
        )
        assert nested["relatedness"]["relationship_pairs"] == {"FS": 2, "engine": "matrix"}
        assert "structure" not in nested
        assert "strata" not in nested

    def test_split_effective_size_routes_scalars_arrays_and_ne_none_stub(self) -> None:
        """Effective-size splitting keeps scalars slim and named arrays extra."""
        slim, extra = _split_effective_size(
            {
                "scalar_estimator": 12.0,
                "ne_coancestry": {"ne": None, "v_mf": [1, 2]},
                "ne_demo": {"ne": 50.0, "cohort_years": [2000, 2001], "cohort_window": {"lo": 1}},
            }
        )
        assert slim["scalar_estimator"] == 12.0
        assert slim["ne_coancestry"] == {"ne": None}
        assert slim["ne_demo"] == {"ne": 50.0, "cohort_window": {"lo": 1}}
        assert extra["ne_demo"] == {"cohort_years": [2000, 2001]}

    def test_split_sex_summary_routes_non_dict_and_extra_stats(self) -> None:
        """Sex-summary splitting preserves scalar strata and routes distributions to extra."""
        slim, extra = _split_sex_summary({"unknown": 2, "female": {"n": 3, "depth": {"mean": 1.0}}})
        assert slim == {"unknown": 2, "female": {"n": 3}}
        assert extra == {"female": {"depth": {"mean": 1.0}}}

    def test_split_section_scalar_list_dict_and_special_variants(self) -> None:
        """Generic section splitting handles scalars, lists, dict specs, and special sections."""
        assert _split_section(5, SectionSpec("scalar")) == (5, None)
        rows = [1, {"depth": 0, "n": 2}]
        assert _split_section(rows, SectionSpec("plain_list")) == (rows, None)
        slim, extra = _split_section(
            [1, {"depth": 0, "n": 2, "mean_F": 0.1}],
            SectionSpec("depth_summary", list_of_dict_slim_keys=("depth", "n")),
        )
        assert slim == [1, {"depth": 0, "n": 2}]
        assert extra == [None, {"mean_F": 0.1}]

        assert _split_section({"a": 1}, SectionSpec("plain_dict")) == ({"a": 1}, None)
        eff_slim, eff_extra = _split_section(
            {"est": {"ne": 1, "cohort_years": [2000]}},
            SectionSpec("effective_size"),
        )
        assert eff_slim == {"est": {"ne": 1}}
        assert eff_extra == {"est": {"cohort_years": [2000]}}
        sex_slim, sex_extra = _split_section(
            {"female": {"n": 2, "depth": {"mean": 0.0}}},
            SectionSpec("sex_summary"),
        )
        assert sex_slim == {"female": {"n": 2}}
        assert sex_extra == {"female": {"depth": {"mean": 0.0}}}
        pairs_slim, pairs_extra = _split_section(
            {"FS": 2, "unknown_extra": 9},
            SectionSpec("relationship_pairs", slim_keys=("FS",)),
        )
        assert pairs_slim == {"FS": 2}
        assert pairs_extra == {"unknown_extra": 9}

    def test_drop_dotted_path_handles_empty_missing_and_present_paths(self) -> None:
        """Dotted-path dropping is a no-op for invalid paths and deletes present leaves."""
        data = {"a": {"b": 1}, "x": 2}
        _drop_dotted_path(data, ())
        _drop_dotted_path(data, ("missing", "leaf"))
        _drop_dotted_path(data, ("a", "b"))
        assert data == {"a": {}, "x": 2}

    def test_split_individual_distributions_routes_non_dict_full_extra_and_drops_f_max(self) -> None:
        """Individual distribution splitting covers scalar, slim, extra, and F.max-drop paths."""
        slim, extra = _split_individual_distributions(
            {
                "flag": 1,
                "F": {"mean": 0.1, "median": 0.0, "max": 0.5, "q1": 0.0},
                "n_offspring": {"mean": 1.0, "median": 1.0},
                "custom": {"mean": 2.0},
            }
        )
        assert slim == {
            "flag": 1,
            "F": {"mean": 0.1, "median": 0.0},
            "n_offspring": {"mean": 1.0, "median": 1.0},
        }
        assert extra == {"F": {"q1": 0.0}, "custom": {"mean": 2.0}}


class TestReportEdgeCases:
    """Report redaction and writer edge cases."""

    def test_build_report_payloads_handle_empty_optional_sections(self, tmp_path) -> None:
        """Pedigree and individual payload builders omit empty optional sections cleanly."""
        size = {
            "n_total": 2,
            "n_founders": 2,
            "founder_frac": 1.0,
            "n_nonfounders": 0,
            "nonfounder_frac": 0.0,
            "n_male": 1,
            "n_female": 1,
            "n_unknown_sex": 0,
            "n_mother_links": 0,
            "n_father_links": 0,
            "n_parent_child_edges": 0,
            "n_with_both_parents": 0,
            "n_with_mother_only": 0,
            "n_with_father_only": 0,
            "n_half_founders": 0,
            "max_depth": 0,
            "mean_depth": 0.0,
            "median_depth": 0.0,
            "depth_counts": [2],
            "n_components": 2,
            "largest_component": 1,
            "largest_component_frac": 0.5,
            "next_components": [1],
        }
        ped, extras = _build_pedigree_data(
            tmp_path / "p.tsv",
            "cmd",
            size,
            {"empty": True},
            {"FS": 0, "by_degree": {1: 0}, "_engine": "matrix"},
            None,
            None,
            None,
        )
        assert ped["sibship_size"] is None
        assert ped["inbreeding"] is None
        assert extras == {}

        idf = pd.DataFrame({"id": [1], "n_offspring": [0]})
        ind = _build_individual_data(idf, tmp_path / "p.tsv", "cmd", include_inbreeding=False)
        assert "F" not in ind["distributions"]
        assert "n_offspring" in ind["distributions"]

    def test_build_summary_data_yaml_extras_and_distribution_split(self, tmp_path) -> None:
        """Summary data construction splices YAML extras and splits individual distributions."""
        ped_data = {
            "input": str(tmp_path / "p.tsv"),
            "command": "cmd",
            "version": "test",
            "generated_at": "now",
            "n_total": 1,
            "effective_size_scalars": {"legacy": 1},
            "relationship_pairs": {"FS": 1},
        }
        ind_data = {
            "input": str(tmp_path / "p.tsv"),
            "command": "cmd",
            "version": "test",
            "generated_at": "now",
            "n_total": 1,
            "distributions": {"n_offspring": {"mean": 0.0, "median": 0.0, "max": 1}, "custom": {"mean": 2.0}},
        }
        slim, extra = _build_summary_data(ped_data, ind_data, yaml_extras={"effective_size": {"est": 1}})
        assert "effective_size_scalars" not in str(slim)
        assert slim["pedigree"]["popgen"]["effective_size"] == {"est": 1}
        assert "n_offspring" in slim["individual"]["distributions"]
        assert "custom" in extra["individual"]["distributions"]

    def test_flatten_long_round_floats_and_drop_extrema(self) -> None:
        """Long flattening, recursive rounding, and extrema dropping handle all container types."""
        assert list(_flatten_long(5)) == []
        rows = list(_flatten_long({"a": {"b": [1.23456]}}))
        assert rows == [("a", "b", "0", 1.23456)]
        assert _round_floats({"x": [1.23456, {"y": 2.34567}]}, ndigits=2) == {"x": [1.23, {"y": 2.35}]}
        obj = [{"mean": 1, "q1": 1, "median": 1, "q3": 1, "min": 0, "max": 2}, {"nested": {"min": 0}}]
        _drop_distribution_extrema(obj)
        assert "min" not in obj[0]
        assert obj[1] == {"nested": {"min": 0}}

    def test_redact_for_privacy_suppresses_small_groups(self) -> None:
        """Privacy redaction handles singleton sibships, mating-pair details, and inbreeding bins."""
        ped_data = {
            "n_total": 10,
            "size_structure": {"next_components": [1, 5], "depth_counts": [1, 6], "largest_component": 1},
            "sibship_size": {"size_dist": {"1": 1, "2": 0}},
            "mating_pairs": {"n_pairs": 1, "children_per_pair": {"mean": 1.0}, "n_pairs_with_multiple_children": 1},
            "relationship_summary": {
                "n_related_pairs": 1,
                "related_pairs_by_closest_degree": {"1": 1},
                "related_pair_density_by_depth": [{"depth": 0, "n": 1, "n_related_pairs": 1}],
            },
            "reproduction": {"n_reproductive": 1},
            "founder_contribution": {"n_founders_with_descendants": 1},
            "founder_summary": {
                "by_depth": [{"depth": 0, "n": 1, "active_founders": 1}],
                "bottleneck": {"min_active_founders": 1},
            },
            "components": {"singletons": 1},
            "sex_summary": {"female": {"n": 1, "n_founders": 1}},
            "depth_summary": [{"depth": 0, "n": 1, "n_male": 1}],
            "relationship_pairs": {"FS": 1, "by_degree": {1: 1}},
            "inbreeding": {
                "n_inbred": 1,
                "frac_inbred": 0.1,
                "mean_F": 0.1,
                "max_F": 0.2,
                "hist": {"<0.25": 0.1, "0": None},
            },
        }
        ind_data = {"distributions": {"F": {"min": 0.0, "max": 0.2, "nz": 1}}}
        _apply_safe_attempt(ped_data, ind_data, min_cell=3)
        assert ped_data["size_structure"]["next_components"] == [5]
        assert ped_data["sibship_size"]["size_dist"] == {"1": None, "2": None}
        assert ped_data["mating_pairs"]["children_per_pair"] is None
        assert ped_data["relationship_pairs"]["FS"] is None
        assert ped_data["relationship_pairs"]["by_degree"] == {1: None}
        assert ped_data["inbreeding"]["n_inbred"] is None
        assert ped_data["inbreeding"]["hist"]["<0.25"] is None
        assert "min" not in ind_data["distributions"]["F"]
        assert ind_data["distributions"]["F"]["nz"] is None

    def test_prepare_out_dir_rejects_existing_file(self, tmp_path) -> None:
        """Output-directory preparation refuses an existing non-directory path."""
        out = tmp_path / "already_file"
        out.write_text("x")
        assert _prepare_out_dir(out) == 1

    def test_write_dropped_manifest_deduplicates_rows(self, tmp_path) -> None:
        """Dropped manifests emit distinct id/check/round rows only once."""
        path = tmp_path / "dropped.tsv"
        _write_dropped_manifest([(1, "self_loops", 1), (1, "self_loops", 1), (2, "negative_ids", 2)], path)
        df = pd.read_csv(path, sep="\t")
        assert df.to_dict("records") == [
            {"id": 1, "check": "self_loops", "round": 1},
            {"id": 2, "check": "negative_ids", "round": 2},
        ]

    def test_write_validate_tsv_prepends_added_founders(self, tmp_path) -> None:
        """Validate TSV writing prepends synthesized founder rows when present."""
        df_raw = pd.DataFrame({"id": [3], "sex": ["M"], "mother": [1], "father": [2]})
        out = tmp_path / "validate.tsv.gz"
        _write_validate_tsv_gz(
            df_raw,
            [{"id": 1, "sex": "F"}, {"id": 2, "sex": "M"}],
            "id",
            "sex",
            "mother",
            "father",
            out,
        )
        with gzip.open(out, "rt") as fh:
            fixed = pd.read_csv(fh, sep="\t", dtype=str)
        assert fixed["id"].tolist() == ["1", "2", "3"]
        assert fixed["mother"].tolist() == ["-1", "-1", "1"]

    def test_write_annotated_tsv_detects_true_row_mismatch(self, tmp_path) -> None:
        """Annotated TSV writing raises when the input rows cannot realign to the individual table."""
        in_path = tmp_path / "p.tsv"
        pd.DataFrame({"id": [1], "sex": ["F"], "mother": [-1], "father": [-1]}).to_csv(in_path, sep="\t", index=False)
        idf = pd.DataFrame({"id": [2], "sex": [0], "mother": [-1], "father": [-1]})
        args = SimpleNamespace(id_col="id", sex_col="sex", mother_col="mother", father_col="father", sep="tab")
        with pytest.raises(PedigreeError, match="row order mismatch"):
            _write_annotated_tsv(in_path, args, idf, tmp_path / "annotated.tsv.gz")


class TestSectionEdgeCases:
    """Section-level summary edge cases."""

    def test_effective_count_no_positive_weights(self) -> None:
        """Effective-count helper returns zero when all weights are non-positive."""
        assert _effective_count_from_weights(np.array([0, -1])) == 0.0

    def test_size_structure_without_children_csr(self) -> None:
        """Size-structure computation has a no-graph path for component labels."""
        df = pd.DataFrame({"mother": [-1, -1], "father": [-1, -1], "sex": [1, 0], "ped_depth": [0, 0]})
        summary, labels = compute_size_structure(df, None)
        assert summary["n_components"] == 2
        np.testing.assert_array_equal(labels, np.array([0, 1], dtype=np.int32))

    def test_mating_and_sibship_empty_and_nonempty_outputs(self) -> None:
        """Mating-pair and sibship sections cover no-child and child-present outputs."""
        df = pd.DataFrame({"mother": [-1, -1], "father": [-1, -1]})
        assert compute_mating_pair_summary(df) is None
        assert compute_sibship_sizes(df) == {"empty": True}

        with_children = pd.DataFrame({"mother": [-1, -1, 1, 1], "father": [-1, -1, 2, 2]})
        mating = compute_mating_pair_summary(with_children)
        sibs = compute_sibship_sizes(with_children)
        assert mating is not None
        assert mating["n_pairs"] == 1
        assert sibs["n_sibships"] == 1
        assert sibs["size_dist"]["2"] == 1.0

    def test_founder_summary_empty_and_bounded_skip(self) -> None:
        """Founder summary handles empty inputs and max-cell skips."""
        empty = pd.DataFrame(columns=["id", "mother", "father", "ped_depth", "is_founder"])
        summary, counts = compute_founder_summary(empty)
        assert summary == {"computed": True, "by_depth": [], "bottleneck": None}
        assert counts.size == 0

        idf = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "mother": [-1, -1, 1],
                "father": [-1, -1, 2],
                "ped_depth": [0, 0, 1],
                "is_founder": [True, True, False],
            }
        )
        skipped, skipped_counts = compute_founder_summary(idf, max_lineage_cells=1)
        assert skipped["computed"] is False
        assert "exceeds" in skipped["skip_reason"]
        np.testing.assert_array_equal(skipped_counts, np.zeros(3, dtype=np.int32))

        summary, n_anc = compute_founder_summary(idf)
        assert summary["computed"] is True
        assert summary["bottleneck"] is not None
        np.testing.assert_array_equal(n_anc, np.array([1, 1, 2], dtype=np.int32))

    def test_aggregate_sections_empty_and_no_inbreeding(self) -> None:
        """Aggregate sections cover empty inputs and the no-inbreeding branch."""
        empty = pd.DataFrame()
        assert compute_aggregate_sections(empty, {"computed": True}, include_inbreeding=False)["depth_summary"] == []

        idf = pd.DataFrame(
            {
                "id": [1, 2],
                "sex": [1, 1],
                "mother": [-1, -1],
                "father": [-1, -1],
                "ped_depth": [0, 0],
                "is_founder": [True, True],
                "F": [0.0, 0.0],
                "n_offspring": [0, 0],
                "n_mates": [0, 0],
                "n_descendant_paths": [0, 0],
                "n_distinct_ancestors": [0, 0],
                "component_id": [0, 0],
                "n_full_sibs": [0, 0],
            }
        )
        sections = compute_aggregate_sections(idf, {"computed": True}, include_inbreeding=False)
        assert sections["reproduction"]["mate_count_female"] is None
        assert sections["genealogy"]["distinct_ancestors"] is None
        assert "F" not in sections["sex_summary"]["male"]

    def test_build_individual_df_parent_derived_counts(self) -> None:
        """Individual table construction covers parent-present, grandparent, and mate-count branches."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7],
                "sex": [1, 0, 1, 0, 1, 0, 1],
                "mother": [-1, -1, -1, 2, 2, 4, 4],
                "father": [-1, -1, -1, 1, 3, 5, 5],
                "ped_depth": [0, 0, 0, 1, 1, 2, 2],
            }
        )
        idf = build_individual_df(
            df,
            pd.Index(df["id"].to_numpy()),
            F=np.zeros(len(df)),
            n_distinct_ancestors=np.arange(len(df)),
            n_descendant_paths=np.arange(len(df)),
            component_labels=np.zeros(len(df), dtype=np.int32),
            sex_source=np.array(["input"] * len(df), dtype=object),
        )
        by_id = idf.set_index("id")
        assert by_id.loc[2, "n_mates"] == 2
        assert by_id.loc[6, "n_full_sibs"] == 1
        assert by_id.loc[6, "n_grandparents"] == 4
        assert by_id.loc[4, "n_mat_half_sibs"] == 1

    def test_relationship_summary_skip_empty_unknown_and_self_pairs(self) -> None:
        """Relationship summaries distinguish skipped, empty, no-valid-pair, and related-pair inputs."""
        df = pd.DataFrame({"ped_depth": [0, 0, 1]})
        skipped = compute_relationship_summary(df, None)
        assert skipped["computed"] is False

        empty = compute_relationship_summary(pd.DataFrame({"ped_depth": []}), {})
        assert empty["n_possible_pairs"] == 0

        no_valid = compute_relationship_summary(
            df,
            {
                "BOGUS": (np.array([0]), np.array([1])),
                "FS": (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
                "MZ": (np.array([1]), np.array([1])),
            },
        )
        assert no_valid["n_related_pairs"] == 0

        related = compute_relationship_summary(df, {"FS": (np.array([0, 1]), np.array([1, 2]))})
        assert related["n_related_pairs"] == 2
        assert related["related_pairs_by_closest_degree"]["1"] == 2

    def test_normalise_effective_size_and_inbreeding_empty(self) -> None:
        """Effective-size key normalisation and empty inbreeding summaries cover degenerate paths."""
        from pedsum.sections import _normalise_effective_size_keys

        assert _normalise_effective_size_keys({"n_generations_used": 3, "ne": 12}) == {
            "n_depths_used": 3,
            "ne": 12,
        }
        assert _normalise_effective_size_keys({"ne": 12}) == {"ne": 12}
        empty_inb = _build_inbreeding_summary(np.array([], dtype=float))
        assert empty_inb["frac_inbred"] == 0.0
        assert empty_inb["mean_F"] == 0.0
