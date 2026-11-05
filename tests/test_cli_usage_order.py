"""With two usage errors on one command line, the subcommand's checks report the one 0.16.1 reported."""

from __future__ import annotations

import pytest

from pedsum.cli.parser import parse_args


def _usage_error(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    """The error line of a usage error (exit 2), after argparse's usage text."""
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err.splitlines()[-1]


def test_validate_reports_break_cycles_before_zygosity(capsys):
    """Validate checks its own flags before --zygosity-col."""
    err = _usage_error(["validate", "--in", "p.tsv", "--out", "o", "--break-cycles", "--zygosity-col", "z"], capsys)
    assert "--break-cycles needs --birth-year-col" in err
    assert "--zygosity-col" not in err


def test_summarize_reports_zygosity_before_burden(capsys):
    """Summarize checks --zygosity-col before its own flags."""
    argv = ["summarize", "--in", "p.tsv", "--out", "o", "--per-individual-burden", "--max-degree", "3"]
    err = _usage_error([*argv, "--zygosity-col", "z"], capsys)
    assert "--zygosity-col needs --multiple-birth-col" in err
    assert "--per-individual-burden" not in err
