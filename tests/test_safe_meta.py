"""``safe_meta_block`` cuts the value of a path option to its base name and records every other word as given."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pedsum.cli.parser import parse_args
from pedsum.report import PATH_OPTIONS, safe_meta_block


def recorded(cmd: str) -> str:
    """The command a safe meta block records for ``cmd``."""
    return safe_meta_block(Path("/home/alice/proj707/ped.tsv"), cmd, 3)["command"]


def test_only_path_option_values_are_cut_to_a_base_name():
    """``--opt=value`` keeps its option name, and a slash in a token or a trait name is not a path."""
    cmd = (
        "pedsum assortative-mating --in=/home/alice/proj707/ped.tsv --out /home/alice/out/ "
        "--missing-parent-token N/A --trait 'bmi/kg' --trait-missing=n/a"
    )
    assert recorded(cmd) == (
        "pedsum assortative-mating --in=ped.tsv --out out --missing-parent-token N/A --trait bmi/kg --trait-missing=n/a"
    )


def test_the_input_is_named_by_base_name():
    """The input and its ``--in`` word both lose their directory."""
    meta = safe_meta_block(Path("/home/alice/proj707/ped.tsv"), "pedsum summarize --in /home/alice/proj707/ped.tsv", 3)
    assert meta["input"] == "ped.tsv"
    assert meta["command"] == "pedsum summarize --in ped.tsv"


@pytest.mark.parametrize(
    ("given", "kept"),
    [("rel/dir/", "dir"), ("--out=rel/dir//", "--out=dir"), ("/", "/"), ("--out=", "--out="), ("ped.tsv", "ped.tsv")],
)
def test_a_trailing_slash_leaves_a_name(given, kept):
    """A directory given with a trailing slash keeps its name; a value with no name is recorded as given."""
    head = "pedsum summarize " if given.startswith("--") else "pedsum summarize --out "
    assert recorded(head + given) == head + kept


def test_a_word_after_an_equals_form_is_not_taken_for_its_value():
    """``--out=a/b`` carries its own value, so the next word is not a path."""
    assert recorded("pedsum summarize --out=a/b c/d") == "pedsum summarize --out=b c/d"


def test_path_options_are_the_parsers_path_typed_options(monkeypatch):
    """``PATH_OPTIONS`` names exactly the ``type=Path`` options of every subcommand."""
    parsers: list[argparse.ArgumentParser] = []
    real = argparse.ArgumentParser.parse_args

    def recording(self, *args, **kwargs):
        parsers.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", recording)
    parse_args(["summarize", "--in", "a", "--out", "b"])
    subparsers = [a for a in parsers[0]._actions if isinstance(a, argparse._SubParsersAction)]
    commands = subparsers[0].choices
    assert len(commands) >= 5
    for name, parser in commands.items():
        path_typed = {flag for action in parser._actions if action.type is Path for flag in action.option_strings}
        assert path_typed == PATH_OPTIONS, name
