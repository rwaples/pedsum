"""Focused edge-case coverage for important pedsum branch behavior."""

from __future__ import annotations

import gzip

import numpy as np
import polars as pl
import pytest
from conftest import EXAMPLE, parent_rows, run_check
from relationship_summary_oracle import compute_relationship_summary

from pedsum.base import PedigreeError
from pedsum.checks import _CHECK_ORDER, CheckStatus, Finding, summarize_findings
from pedsum.ids import IdLabels
from pedsum.parse import (
    _sniff_delimiter,
    decode_sex,
    format_id_sample,
    maybe_warn_csv,
    read_pedigree_table,
)
from pedsum.pedigree_ops import group_mating_pairs, structural_depth
from pedsum.report import (
    _drop_distribution_extrema,
    _flatten_long,
    _round_floats,
    apply_safe_attempt,
    build_individual_data,
    build_pedigree_data,
    build_summary_data,
    prepare_out_dir,
    write_annotated_tsv,
)
from pedsum.schema import (
    SectionSpec,
    _drop_dotted_path,
    _split_section,
    _split_sex_summary,
    categorise_pedigree,
    split_individual_distributions,
)
from pedsum.sections import (
    _effective_count_from_weights,
    build_inbreeding_summary,
    build_individual_df,
    compute_aggregate_sections,
    compute_founder_summary,
    compute_mating_pair_summary,
    compute_sibship_sizes,
    compute_size_structure,
)
from pedsum.validate import ValidationConfig, ValidationContext, run_checks, validate_pedigree


class TestParseEdgeCases:
    """Direct parser edge cases that are awkward through the subprocess CLI."""

    def test_id_sample_empty_and_deterministic_subset(self) -> None:
        """ID samples handle empty inputs and return deterministic bounded samples."""
        assert format_id_sample(np.array([], dtype=np.int64), str) == ""
        assert format_id_sample(np.arange(10), str, k=3) == "5, 6, 9"

    def test_decode_sex_reads_plink_and_rejects_other_numbers(self) -> None:
        """1/2/0 decode as PLINK and -1 as unknown; any other number is an error."""
        plink = decode_sex(pl.Series(["2", "1", "0", "-1"], dtype=pl.String))
        np.testing.assert_array_equal(plink, np.array([0, 1, -1, -1], dtype=np.int8))
        with pytest.raises(PedigreeError, match="PLINK 1=male, 2=female, 0=unknown"):
            decode_sex(pl.Series(["F", "3"], dtype=pl.String))

    def test_maybe_warn_csv_allows_normal_input_and_rejects_csv_shape(self) -> None:
        """The friendly CSV warning distinguishes split tables from one-column CSV-looking input."""
        maybe_warn_csv(pl.DataFrame({"id": [1], "sex": ["F"]}))
        with pytest.raises(PedigreeError, match="appears to be CSV"):
            maybe_warn_csv(pl.DataFrame({"id,sex,mother,father": ["1,F,-1,-1"]}))

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
            read_pedigree_table(tmp_path / "missing.tsv")
        table = tmp_path / "ped.psv"
        table.write_text("id|sex|mother|father\n1|F|-1|-1\n")
        df = read_pedigree_table(table, sep="pipe", dtype=str)
        assert df.to_dicts() == [{"id": "1", "sex": "F", "mother": "-1", "father": "-1"}]

        comma = tmp_path / "ped.csv"
        comma.write_text("id,sex,mother,father\n1,F,-1,-1\n")
        auto = read_pedigree_table(comma, dtype=str)
        assert auto.to_dicts() == [{"id": "1", "sex": "F", "mother": "-1", "father": "-1"}]


def _founders(ids: list[int]) -> list[dict]:
    """Founder rows with the given IDs, all female."""
    return [{"id": str(i), "sex": "F", "mother": "-1", "father": "-1"} for i in ids]


class TestCheckEdgeCases:
    """Validation check producers cover clean, offending, and formatting cases."""

    def test_duplicate_ids_clean_and_offending(self) -> None:
        """Duplicate-ID checking returns no findings for unique IDs and one per duplicate value."""
        assert run_check("duplicate_ids", _founders([1, 2, 3])).findings == []
        findings = run_check("duplicate_ids", _founders([2, 1, 2, 3, 1])).findings
        assert [f.id for f in findings] == [1, 2]

    def test_parent_refs_present_absent_clean_missing_and_zero_hint(self) -> None:
        """Missing-parent detection distinguishes no refs, all-present refs, and the zero-token hint."""

        def check(mothers: list[int], fathers: list[int], role: str) -> list[Finding]:
            rows = [
                {**row, "mother": str(m), "father": str(f)}
                for row, m, f in zip(_founders([1, 2, 3]), mothers, fathers, strict=True)
            ]
            return run_check(f"parent_refs_present_{role}", rows).findings

        assert check([-1, -1, -1], [-1, -1, -1], "mother") == []
        assert check([1, -1, 2], [-1, -1, -1], "mother") == []
        findings = check([-1, -1, -1], [0, 99, 99], "father")
        assert [f.id for f in findings] == [0, 99]
        assert "convert to -1" in findings[0].detail

    def test_empty_pedigree_check(self) -> None:
        """Empty-pedigree checking fires only for n == 0."""
        assert run_check("empty_pedigree", _founders([1, 2, 3])).status == CheckStatus.PASS
        empty = pl.DataFrame(schema=dict.fromkeys(("id", "sex", "mother", "father"), pl.String))
        result = run_checks(ValidationContext(empty, ValidationConfig()))[0]["empty_pedigree"]
        assert result.findings[0].check == "empty_pedigree"

    def test_sex_role_consistency_without_override(self) -> None:
        """Without the role override, an asserted male used as a mother fails sex-role consistency."""
        rows = [
            {"id": "1", "sex": "M", "mother": "-1", "father": "-1"},
            {"id": "2", "sex": "M", "mother": "-1", "father": "-1"},  # male, but used as mother
            {"id": "3", "sex": "F", "mother": "2", "father": "1"},
        ]
        findings = run_check("sex_role_consistency", rows, ValidationConfig(override_asserted_sex=False)).findings
        assert len(findings) == 1
        assert findings[0].id == 2
        assert "used as mother" in findings[0].detail

    def test_every_check_returns_a_fresh_result(self) -> None:
        """No CheckResult is shared between Checks or between runs, since the runner stamps each one."""
        runs = [validate_pedigree(EXAMPLE, ValidationConfig())[1] for _ in range(2)]
        ids = [id(r) for results in runs for r in results]
        assert len(set(ids)) == len(ids)
        assert all(r.name == name for r, name in zip(runs[0], _CHECK_ORDER, strict=True))

    def test_summarize_findings_variants(self) -> None:
        """Finding summaries render empty, id-only, row-only, and overflow samples."""
        assert summarize_findings([], str) == ""
        assert "id=7" in summarize_findings([Finding(check="c", id=7)], str)
        assert "row 4" in summarize_findings([Finding(check="c", row=4)], str)
        many = [Finding(check="c", id=i, row=i) for i in range(7)]
        assert "and 2 more" in summarize_findings(many, str)

    def test_birth_year_topology_no_edges_and_bad_edge(self) -> None:
        """Birth-year topology skips no-edge pedigrees and flags child-before-parent years."""
        config = ValidationConfig(birth_year_col="birth_year")
        rows = [
            {"id": "1", "sex": "F", "mother": "-1", "father": "-1", "birth_year": "2000"},
            {"id": "2", "sex": "M", "mother": "-1", "father": "-1", "birth_year": "1990"},
        ]
        assert run_check("birth_year_topology", rows, config).findings == []
        rows[1]["mother"] = "1"
        bad = run_check("birth_year_topology", rows, config).findings
        assert len(bad) == 1
        assert bad[0].check == "birth_year_topology"


class TestPedigreeOpsAndPairsEdgeCases:
    """Low-level pedigree helper edge cases."""

    def test_mating_pairs_empty_and_present_parent_pairs(self) -> None:
        """Mating-pair grouping handles no-full-parent and full-parent sibling groups."""
        df = pl.DataFrame({"id": [1, 2], "mother": [-1, 1], "father": [-1, -1]})
        mating = group_mating_pairs(df, *parent_rows(df))
        assert mating.children.size == 0
        assert mating.sizes.size == 0

        sibs = pl.DataFrame({"id": [1, 2, 3, 4], "mother": [-1, -1, 1, 1], "father": [-1, -1, 2, 2]})
        mating = group_mating_pairs(sibs, *parent_rows(sibs))
        # One (mother=1, father=2) mating pair with two children.
        assert mating.children.tolist() == [2, 3]
        assert mating.sizes.tolist() == [2]
        assert mating.mother_rows.tolist() == [0]
        assert mating.father_rows.tolist() == [1]

    def test_structural_depth_marks_cycles(self) -> None:
        """Depth places acyclic rows and leaves rows in a true cycle at -1."""
        depth = structural_depth(np.array([-1, 0]), np.array([-1, -1]))
        np.testing.assert_array_equal(depth, np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(structural_depth(np.array([1, 0]), np.array([-1, -1])), [-1, -1])


class TestSchemaEdgeCases:
    """Summary schema split edge cases."""

    def test_categorise_pedigree_merges_pair_engine_and_drops_empty_sections(self) -> None:
        """Categorisation folds pair engine into relationship pairs and skips empty sections."""
        nested = categorise_pedigree(
            {
                "relationship_pairs": {"FS": 2},
                "pairs_engine": "rust_streaming",
                "components": {},
                "sex_summary": [],
            }
        )
        assert nested["relatedness"]["relationship_pairs"] == {"FS": 2, "engine": "rust_streaming"}
        assert "structure" not in nested
        assert "strata" not in nested

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
        slim, extra = split_individual_distributions(
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
        ped = build_pedigree_data(
            tmp_path / "p.tsv",
            "cmd",
            size,
            None,
            {"FS": 0, "by_degree": {1: 0}, "_engine": "matrix"},
            None,
            None,
            None,
            compute_aggregate_sections(pl.DataFrame(), {"computed": True}),
            None,
        )
        assert ped["sibship_size"] is None
        assert ped["inbreeding"] is None

        idf = pl.DataFrame({"id": [1], "n_offspring": [0]})
        ind = build_individual_data(idf, tmp_path / "p.tsv", "cmd")
        assert "F" not in ind["distributions"]
        assert "n_offspring" in ind["distributions"]

    def test_build_summary_data_categorises_and_splits_distributions(self, tmp_path) -> None:
        """Summary data construction categorises pedigree sections and splits individual distributions."""
        ped_data = {
            "input": str(tmp_path / "p.tsv"),
            "command": "cmd",
            "version": "test",
            "generated_at": "now",
            "n_total": 1,
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
        slim, extra = build_summary_data(ped_data, ind_data)
        assert slim["pedigree"]["relatedness"]["relationship_pairs"] == {"FS": 1}
        assert "n_offspring" in slim["individual"]["distributions"]
        assert "custom" in extra["individual"]["distributions"]

    def test_flatten_long_round_floats_and_drop_extrema(self) -> None:
        """Long flattening, recursive rounding, and extrema dropping handle all container types."""
        assert list(_flatten_long(5)) == []
        rows = list(_flatten_long({"a": {"b": [1.23456]}}))
        assert rows == [("a", "b", "0", 1.23456)]
        assert _round_floats({"x": [1.23456, {"y": 2.34567}]}, ndigits=2) == {"x": [1.23, {"y": 2.35}]}
        values = [0.123456, 0.0123456, 4.00001e-5, -3.14159e-9, 123.456789, 0.0, float("inf")]
        assert _round_floats(values, figures=4) == [0.1235, 0.01235, 4e-05, -3.142e-09, 123.4568, 0.0, float("inf")]
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
        apply_safe_attempt(ped_data, ind_data, min_cell=3)
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
        assert prepare_out_dir(out) == 1

    def test_write_annotated_tsv_detects_true_row_mismatch(self, tmp_path) -> None:
        """Annotated TSV writing raises when the input rows cannot realign to the individual table."""
        in_path = tmp_path / "p.tsv"
        pl.DataFrame({"id": [1], "sex": ["F"], "mother": [-1], "father": [-1]}).write_csv(in_path, separator="\t")
        idf = pl.DataFrame({"id": [2], "sex": [0], "mother": [-1], "father": [-1]})
        with pytest.raises(PedigreeError, match="row order mismatch"):
            write_annotated_tsv(in_path, ValidationConfig(), "tab", idf, tmp_path / "annotated.tsv.gz", IdLabels())


class TestSectionEdgeCases:
    """Section-level summary edge cases."""

    def test_effective_count_no_positive_weights(self) -> None:
        """Effective-count helper returns zero when all weights are non-positive."""
        assert _effective_count_from_weights(np.array([0, -1])) == 0.0

    def test_size_structure_without_parent_edges(self) -> None:
        """Size-structure computation has a no-edge path for component labels."""
        df = pl.DataFrame({"id": [1, 2], "mother": [-1, -1], "father": [-1, -1], "sex": [1, 0], "ped_depth": [0, 0]})
        summary, labels = compute_size_structure(df, *parent_rows(df))
        assert summary["n_components"] == 2
        np.testing.assert_array_equal(labels, np.array([0, 1], dtype=np.int32))

    def test_mating_and_sibship_empty_and_nonempty_outputs(self) -> None:
        """Mating-pair and sibship sections cover no-child and child-present outputs."""
        df = pl.DataFrame({"id": [1, 2], "mother": [-1, -1], "father": [-1, -1]})
        sizes = group_mating_pairs(df, *parent_rows(df)).sizes
        assert compute_mating_pair_summary(sizes) is None
        assert compute_sibship_sizes(sizes) is None

        with_children = pl.DataFrame({"id": [1, 2, 3, 4], "mother": [-1, -1, 1, 1], "father": [-1, -1, 2, 2]})
        sizes = group_mating_pairs(with_children, *parent_rows(with_children)).sizes
        mating = compute_mating_pair_summary(sizes)
        sibs = compute_sibship_sizes(sizes)
        assert mating is not None
        assert mating["n_pairs"] == 1
        assert sibs["n_sibships"] == 1
        assert sibs["size_dist"]["2"] == 1.0

    def test_founder_summary_empty_and_bounded_skip(self) -> None:
        """Founder summary handles empty inputs and max-cell skips."""
        empty = pl.DataFrame(
            schema={
                "id": pl.Int64,
                "mother": pl.Int64,
                "father": pl.Int64,
                "ped_depth": pl.Int32,
                "is_founder": pl.Boolean,
            }
        )
        summary, counts = compute_founder_summary(empty, *parent_rows(empty))
        assert summary == {"computed": True, "by_depth": [], "bottleneck": None}
        assert counts.size == 0

        idf = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "mother": [-1, -1, 1],
                "father": [-1, -1, 2],
                "ped_depth": [0, 0, 1],
                "is_founder": [True, True, False],
            }
        )
        skipped, skipped_counts = compute_founder_summary(idf, *parent_rows(idf), max_lineage_cells=1)
        assert skipped["computed"] is False
        assert "exceeds" in skipped["skip_reason"]
        np.testing.assert_array_equal(skipped_counts, np.zeros(3, dtype=np.int32))

        summary, n_anc = compute_founder_summary(idf, *parent_rows(idf))
        assert summary["computed"] is True
        assert summary["bottleneck"] is not None
        np.testing.assert_array_equal(n_anc, np.array([1, 1, 2], dtype=np.int32))

    def test_aggregate_sections_empty_and_no_inbreeding(self) -> None:
        """Aggregate sections cover empty inputs and the no-inbreeding branch."""
        empty = pl.DataFrame()
        assert compute_aggregate_sections(empty, {"computed": True})["depth_summary"] == []

        idf = pl.DataFrame(
            {
                "id": [1, 2],
                "sex": [1, 1],
                "mother": [-1, -1],
                "father": [-1, -1],
                "ped_depth": [0, 0],
                "is_founder": [True, True],
                "n_offspring": [0, 0],
                "n_mates": [0, 0],
                "n_descendant_paths": [0, 0],
                "component_id": [0, 0],
                "n_full_sibs": [0, 0],
            }
        )
        sections = compute_aggregate_sections(idf, {"computed": True})
        assert sections["reproduction"]["mate_count_female"] is None
        assert sections["genealogy"]["distinct_ancestors"] is None
        assert "F" not in sections["sex_summary"]["male"]

    def test_build_individual_df_parent_derived_counts(self) -> None:
        """Individual table construction covers parent-present, grandparent, and mate-count branches."""
        df = pl.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7],
                "sex": [1, 0, 1, 0, 1, 0, 1],
                "mother": [-1, -1, -1, 2, 2, 4, 4],
                "father": [-1, -1, -1, 1, 3, 5, 5],
                "ped_depth": [0, 0, 0, 1, 1, 2, 2],
            }
        )
        mother_rows, father_rows = parent_rows(df)
        idf = build_individual_df(
            df,
            mother_rows,
            father_rows,
            group_mating_pairs(df, mother_rows, father_rows),
            F=np.zeros(len(df)),
            n_distinct_ancestors=np.arange(len(df)),
            n_descendant_paths=np.arange(len(df)),
            component_labels=np.zeros(len(df), dtype=np.int32),
            sex_source=np.array(["input"] * len(df), dtype=object),
        )
        by_id = {row["id"]: row for row in idf.iter_rows(named=True)}
        assert by_id[2]["n_mates"] == 2
        assert by_id[6]["n_full_sibs"] == 1
        assert by_id[6]["n_grandparents"] == 4
        assert by_id[4]["n_mat_half_sibs"] == 1

    def test_relationship_summary_skip_empty_unknown_and_self_pairs(self) -> None:
        """Relationship summaries distinguish skipped, empty, no-valid-pair, and related-pair inputs."""
        df = pl.DataFrame({"ped_depth": [0, 0, 1]})
        skipped = compute_relationship_summary(df, None)
        assert skipped["computed"] is False

        empty = compute_relationship_summary(pl.DataFrame(schema={"ped_depth": pl.Int32}), {})
        assert empty["n_individual_pairs"] == 0

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
        empty_inb = build_inbreeding_summary(np.array([], dtype=float))
        assert empty_inb["frac_inbred"] == 0.0
        assert empty_inb["mean_F"] == 0.0
