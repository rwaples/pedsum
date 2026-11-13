"""``--safe-attempt`` logs name no ID and no data value, and its files have the safe names in a clean ``--out``.

The pedigrees here use IDs, birth years and trait values that no other text
of a run can contain, so finding one in stderr, stdout or a file is a leak.
"""

from __future__ import annotations

import ast
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import EXAMPLE, run_pedsum, write_ped

from pedsum import safe_log
from pedsum.base import PedigreeError
from pedsum.report import SAFE_SUFFIX, id_log_name, output_name
from pedsum.safe_log import WITHHELD, Public, detail

PACKAGE = Path(__file__).resolve().parents[1] / "pedsum"
YEAR0 = 4_100_000
BY = ["--birth-year-col", "birth_year", "--birth-year-max", "9999999"]

COMMANDS: dict[str, tuple[list[str], set[str]]] = {
    "summarize": (
        [*BY, "--sex-concordance", "--per-individual-burden", "--tsv"],
        {"summary.txt", "summary.extra.txt", "summary.pedigree.tab"},
    ),
    "effective-size": ([*BY], {"effective_size.txt"}),
    "assortative-mating": (
        [*BY, "--trait", "score", "dx", "--trait-type", "dx=binary", "--permutations", "9", "--threads", "1"],
        {"assortative_mating.txt"},
    ),
}


def pid(i: int) -> str:
    """The ``i``-th distinctive ID."""
    return f"ID9{i:04d}7"


def rows(n_couples: int = 12) -> list[dict[str, str]]:
    """Founder couples with three children each; the children pair up and have two children each."""
    out: list[dict[str, str]] = []

    def add(sex: str, mother: str, father: str, generation: int) -> str:
        i = len(out)
        out.append(
            {
                "id": pid(i),
                "sex": sex,
                "mother": mother,
                "father": father,
                "birth_year": str(YEAR0 + 1000 * generation + i),
                "score": f"{700.31 + 1.13 * i:.2f}",
                "dx": str(i % 2),
            }
        )
        return pid(i)

    children: list[tuple[str, str]] = []
    for _ in range(n_couples):
        father, mother = add("M", "NA", "NA", 0), add("F", "NA", "NA", 0)
        children += [(add("MF"[k % 2], mother, father, 1), "MF"[k % 2]) for k in range(3)]
    sons = [c for c, sex in children if sex == "M"]
    daughters = [c for c, sex in children if sex == "F"]
    # Shifted by one family, so no couple is a pair of sibs.
    for son, daughter in zip(sons, daughters[2:] + daughters[:2], strict=False):
        for k in range(2):
            add("MF"[k], daughter, son, 2)
    return out


def tokens(pedigree: list[dict[str, str]], extra: tuple[str, ...] = ()) -> list[str]:
    """Every ID, birth year and continuous trait value of a pedigree, plus ``extra``."""
    found = {row[key] for row in pedigree for key in ("id", "birth_year", "score") if row.get(key)}
    return sorted(found | set(extra))


def leaked(text: str, pedigree_tokens: list[str]) -> list[str]:
    """The tokens ``text`` holds as whole numbers or words, in any case."""
    return [t for t in pedigree_tokens if re.search(rf"(?<![\w.]){re.escape(t)}(?![\w])", text, re.IGNORECASE)]


def emitted(res, out_dir: Path, *, skip: tuple[str, ...] = ()) -> str:
    """stdout, stderr, and every file name and file text of ``out_dir`` but ``skip``."""
    parts = [res.stdout, res.stderr]
    if out_dir.exists():
        for path in sorted(out_dir.rglob("*")):
            parts.append(path.name)
            if path.is_file() and path.name not in skip:
                parts.append(path.read_text())
    return "\n".join(parts)


def safe_run(command: str, ped: Path, out_dir: Path, *more: str):
    """Run ``command`` with its full flags under ``--safe-attempt`` at DEBUG."""
    flags, _ = COMMANDS[command]
    return run_pedsum([command, "--in", str(ped), "--out", str(out_dir), *flags, "--safe-attempt", "--verbose", *more])


def broken(pedigree: list[dict[str, str]]) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    """``pedigree`` with a bad sex token, an absent mother, a repeated id and a child born before its parents."""
    bad = [dict(row) for row in pedigree]
    bad[5]["sex"] = "SEXTOKEN77"
    bad[-1]["mother"] = "ID9MISSING7"
    bad.append(dict(bad[3]))
    bad[-3]["birth_year"] = str(YEAR0 - 50)
    return bad, ("SEXTOKEN77", "ID9MISSING7", str(YEAR0 - 50))


# ---------------------------------------------------------------------------
# The scrubber
# ---------------------------------------------------------------------------


@pytest.fixture
def scrubbed(caplog):
    """A logger whose records are scrubbed, with the messages it produced."""
    log = logging.getLogger("pedigree_summary.test")
    caplog.set_level(logging.DEBUG)
    with safe_log.scrubbing():
        yield log, lambda: [record.getMessage() for record in caplog.records]


def test_an_unmarked_string_is_withheld_and_a_public_one_is_printed(scrubbed):
    """A plain string, a list and a set are withheld; a ``Public`` string and a path print."""
    log, messages = scrubbed
    log.info("id %s fails %s in %s", "ID90417", Public("sex_tokens"), Path("ped.tsv"))
    log.info("ids %s and %r", ["ID90417"], {"ID90418"})
    assert messages() == [f"id {WITHHELD} fails sex_tokens in ped.tsv", f"ids {WITHHELD} and {WITHHELD}"]


def test_an_unmarked_number_is_withheld_and_a_public_one_keeps_its_format(scrubbed):
    """An int, a float, a bool and a numpy number are withheld unless marked ``Public``."""
    log, messages = scrubbed
    log.info("%d missing of %d, %d unknown, %5.1f%% in %.2fs, %s", 4, 21, 0, 12.5, 0.25, True)
    log.info("%d rows, mean %.3f", np.int64(21), np.float64(0.5))
    log.info("%d threads, min cell %s, %5.1f%% in %.2fs", Public(4), Public(5), Public(12.5), Public(0.25))
    assert messages() == [
        f"{WITHHELD} missing of {WITHHELD}, {WITHHELD} unknown, {WITHHELD}% in {WITHHELD}s, {WITHHELD}",
        f"{WITHHELD} rows, mean {WITHHELD}",
        "4 threads, min cell 5,  12.5% in 0.25s",
    ]


def test_a_public_value_formats_as_its_own_type_in_an_ordinary_run(caplog):
    """``Public`` keeps the type of its value, and takes nothing but a string or a number."""
    caplog.set_level(logging.DEBUG)
    logging.getLogger("pedigree_summary.test").info("%d of %.2f, %s", Public(np.int64(7)), Public(0.5), Public("name"))
    assert caplog.records[0].getMessage() == "7 of 0.50, name"
    assert isinstance(Public("name"), str)
    assert isinstance(Public(7), int)
    assert isinstance(Public(0.5), float)
    with pytest.raises(TypeError):
        Public(["ID90417"])


def test_a_withheld_argument_does_not_break_a_numeric_format(scrubbed):
    """A ``%d`` or ``%.3g`` whose argument is withheld prints the placeholder."""
    log, messages = scrubbed
    log.warning("widest %d days, p = %.3g, 100%% sure", "ID90417", detail(0.0123))
    assert messages() == [f"widest {WITHHELD} days, p = {WITHHELD}, 100% sure"]


def test_an_error_prints_its_public_text_only(scrubbed):
    """A ``PedigreeError`` prints its public text or nothing; an ``OSError`` prints."""
    log, messages = scrubbed
    log.error(
        "failed: %s", PedigreeError("trait 'x' is constant ('TRAITVAL9' everywhere)", public="trait 'x' is constant")
    )
    log.error("failed: %s", PedigreeError("row 3 (id=ID90417)"))
    log.error("failed: %s", PedigreeError("trait 'x' is missing in every row", public=True))
    log.error("file error: %s", FileNotFoundError(2, "No such file", "ped.tsv"))
    assert messages() == [
        "failed: trait 'x' is constant",
        f"failed: {WITHHELD}",
        "failed: trait 'x' is missing in every row",
        "file error: [Errno 2] No such file: 'ped.tsv'",
    ]


def test_a_traceback_a_mapping_and_a_non_string_message_are_scrubbed(scrubbed, caplog):
    """A traceback is dropped, and mapping arguments and exception messages are scrubbed."""
    log, messages = scrubbed
    try:
        raise ValueError("ID90417")
    except ValueError as e:
        log.exception("callback failed")
        log.error(e)
    log.info("%(who)s has %(n)d of %(cap)d", {"who": "ID90417", "n": 7, "cap": Public(9)})
    assert messages() == ["callback failed", WITHHELD, f"{WITHHELD} has {WITHHELD} of 9"]
    assert all(record.exc_info is None for record in caplog.records)
    assert "ID90417" not in caplog.text


def test_a_record_of_any_other_logger_is_withheld_whole(scrubbed, tmp_path):
    """pedigree-graph, a library that formats its own message, and a Python warning print nothing of theirs."""
    _, messages = scrubbed
    safe_log.open_id_log(tmp_path / "ids.local.log")
    logging.getLogger("pedigree_graph._progress").info("%s", "relatives_per_person total: 2 pairs, pairs per lane [2]")
    logging.getLogger("somelib").warning("row 3 holds ID90417")
    logging.getLogger("pedigree_summary_other").info("%d rows", Public(3))
    warnings.warn("ID90417 is odd", UserWarning, stacklevel=1)
    assert messages() == [WITHHELD, WITHHELD, WITHHELD, f"a library warned (UserWarning): {WITHHELD}"]
    id_log = (tmp_path / "ids.local.log").read_text()
    assert "pairs per lane [2]" in id_log
    assert "row 3 holds ID90417" in id_log
    assert "UserWarning: ID90417 is odd" in id_log


def test_scrubbing_ends_with_the_block(caplog):
    """After the block a record is built as before, and the ID log is closed."""
    caplog.set_level(logging.DEBUG)
    with safe_log.scrubbing():
        pass
    logging.getLogger("pedigree_summary.test").info("id %s, %d", "ID90417", 2)
    assert [record.getMessage() for record in caplog.records] == ["id ID90417, 2"]
    assert not safe_log.id_log_is_open()
    assert not safe_log.anything_withheld()


class _CountingRecords:
    """A log record factory that keeps every record it builds."""

    def __init__(self) -> None:
        self.records: list[logging.LogRecord] = []

    def __call__(self, *args, **kwargs) -> logging.LogRecord:
        record = logging.LogRecord(*args, **kwargs)
        self.records.append(record)
        return record


@pytest.fixture
def logging_state():
    """A custom record factory and ``showwarning`` installed around a test, and what was there put back."""
    factory, shown = _CountingRecords(), []

    def showwarning(message, category, filename, lineno, file=None, line=None):
        shown.append(str(message))

    before = logging.getLogRecordFactory(), warnings.showwarning
    logging.setLogRecordFactory(factory)
    warnings.showwarning = showwarning
    try:
        yield factory, showwarning, shown
    finally:
        logging.setLogRecordFactory(before[0])
        warnings.showwarning = before[1]


def _warn_then(real):
    def run(*args, **kwargs):
        warnings.warn("ID90417 is odd", UserWarning, stacklevel=1)
        return real(*args, **kwargs)

    return run


def test_an_ordinary_run_leaves_the_record_factory_and_the_warnings_hook_alone(logging_state, tmp_path, monkeypatch):
    """Without ``--safe-attempt`` ``main`` builds its records with the factory it found and captures no warning."""
    from pedsum.cli import summarize

    factory, showwarning, shown = logging_state
    monkeypatch.setattr(summarize, "group_mating_pairs", _warn_then(summarize.group_mating_pairs))
    res = run_pedsum(["summarize", "--in", str(EXAMPLE), "--out", str(tmp_path / "out")])
    assert res.returncode == 0, res.stderr
    assert logging.getLogRecordFactory() is factory
    assert warnings.showwarning is showwarning
    assert shown == ["ID90417 is odd"]
    assert any("five random ids: " in record.getMessage() for record in factory.records)
    assert "ID90417" not in res.stderr


def test_a_safe_run_puts_back_the_record_factory_and_the_warnings_hook(logging_state, tmp_path, monkeypatch):
    """A safe run scrubs through the factory it found, then leaves that factory and ``showwarning`` in place."""
    from pedsum.cli import summarize

    factory, showwarning, shown = logging_state
    monkeypatch.setattr(summarize, "group_mating_pairs", _warn_then(summarize.group_mating_pairs))
    res = run_pedsum(["summarize", "--in", str(EXAMPLE), "--out", str(tmp_path / "out"), "--safe-attempt"])
    assert res.returncode == 0, res.stderr
    assert logging.getLogRecordFactory() is factory
    assert warnings.showwarning is showwarning
    assert shown == []
    messages = [record.getMessage() for record in factory.records]
    assert f"a library warned (UserWarning): {WITHHELD}" in messages
    assert not any("ID90417" in message for message in messages)
    assert not safe_log.id_log_is_open()

    boom = RuntimeError("row 3 holds ID90417")
    monkeypatch.setattr(summarize, "group_mating_pairs", lambda *_a, **_k: (_ for _ in ()).throw(boom))
    res = run_pedsum(["summarize", "--in", str(EXAMPLE), "--out", str(tmp_path / "out2"), "--safe-attempt"])
    assert res.returncode == 1
    assert logging.getLogRecordFactory() is factory
    assert warnings.showwarning is showwarning


def _log_calls(tree: ast.AST):
    """Every ``logger.<level>(...)`` call of a module."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
        ):
            yield node


def test_every_log_message_in_the_package_is_a_literal():
    """An f-string or a concatenated message would carry a value past the scrubber, which sees arguments only."""
    built = []
    n_calls = 0
    for path in sorted(PACKAGE.rglob("*.py")):
        for call in _log_calls(ast.parse(path.read_text())):
            n_calls += 1
            message = call.args[0]
            if not (isinstance(message, ast.Constant) and isinstance(message.value, str)):
                built.append(f"{path.relative_to(PACKAGE.parent)}:{call.lineno}")
    assert n_calls > 50
    assert built == []


def test_only_validate_and_the_parser_write_to_stderr_directly():
    """``print`` and ``sys.stderr.write`` bypass logging; the parser prints usage before any input is read."""
    direct = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name == "print" or name.endswith(("stderr.write", "stdout.write")):
                    direct.add(str(path.relative_to(PACKAGE)))
    assert direct == {"cli/validate.py", "cli/parser.py"}


# ---------------------------------------------------------------------------
# The three commands at DEBUG
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pedigree() -> list[dict[str, str]]:
    """The clean pedigree."""
    return rows()


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_clean_run_emits_no_id_and_no_value(command, pedigree, tmp_path):
    """IDs read as strings log a sample at WARNING and five ids at INFO in an ordinary run; neither survives."""
    ped = write_ped(tmp_path / "ped.tsv", pedigree)
    plain = run_pedsum([command, "--in", str(ped), "--out", str(tmp_path / "plain"), *COMMANDS[command][0], "-v"])
    assert plain.returncode == 0, plain.stderr
    assert leaked(plain.stderr, tokens(pedigree))

    res = safe_run(command, ped, tmp_path / "out")
    assert res.returncode == 0, res.stderr
    assert WITHHELD in res.stderr
    assert "--id-log writes those lines in full to" in res.stderr
    assert leaked(emitted(res, tmp_path / "out"), tokens(pedigree)) == []
    assert {path.name for path in (tmp_path / "out").iterdir()} == COMMANDS[command][1]


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_clean_run_with_integer_ids_emits_none(command, pedigree, tmp_path):
    """Integer IDs, which a ``%d`` would print, reach no output either."""
    numbered = [
        {**row, **{key: (row[key][3:-1] + "90417" if row[key] != "NA" else "NA") for key in ("id", "mother", "father")}}
        for row in pedigree
    ]
    ped = write_ped(tmp_path / "ped.tsv", numbered)
    res = safe_run(command, ped, tmp_path / "out")
    assert res.returncode == 0, res.stderr
    assert leaked(emitted(res, tmp_path / "out"), tokens(numbered)) == []


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_failed_validation_names_checks_and_counts_only(command, pedigree, tmp_path):
    """A failed validation names each failing Check and writes no ``validate.log``."""
    bad, extra = broken(pedigree)
    ped = write_ped(tmp_path / "ped.tsv", bad)
    plain = run_pedsum([command, "--in", str(ped), "--out", str(tmp_path / "plain"), *COMMANDS[command][0]])
    assert plain.returncode == 1
    assert (tmp_path / "plain" / "validate.log").exists()

    res = safe_run(command, ped, tmp_path / "out")
    assert res.returncode == 1
    assert leaked(emitted(res, tmp_path / "out"), tokens(bad, extra)) == []
    assert list((tmp_path / "out").iterdir()) == []
    failing = set(re.findall(r"check (\w+): ", res.stderr))
    assert {"sex_tokens", "duplicate_ids"} <= failing
    assert "add --id-log to write them to" in res.stderr
    assert id_log_name(command) in res.stderr


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_id_log_is_the_only_place_an_id_appears(command, pedigree, tmp_path):
    """``--id-log`` puts the findings, with their IDs, in ``<command>.ids.local.log`` and nowhere else."""
    bad, extra = broken(pedigree)
    ped = write_ped(tmp_path / "ped.tsv", bad)
    out = tmp_path / "out"
    res = safe_run(command, ped, out, "--id-log")
    assert res.returncode == 1
    name = f"{command}.ids.local.log"
    assert [path.name for path in out.iterdir()] == [name]
    assert leaked(emitted(res, out, skip=(name,)), tokens(bad, extra)) == []
    id_log = (out / name).read_text()
    assert {"SEXTOKEN77", bad[3]["id"]} <= set(leaked(id_log, tokens(bad, extra)))
    assert "check\tid\trow\tdetail\tn_refs" in id_log
    assert "never transfer that file" in res.stderr


def test_id_log_of_a_clean_run_holds_the_withheld_lines(pedigree, tmp_path):
    """The ID log of a clean run holds exactly the lines that lost something, and a rerun replaces it."""
    ped = write_ped(tmp_path / "ped.tsv", pedigree)
    out = tmp_path / "out"
    res = safe_run("summarize", ped, out, "--id-log")
    assert res.returncode == 0, res.stderr
    assert {path.name for path in out.iterdir()} == COMMANDS["summarize"][1] | {"summarize.ids.local.log"}
    assert leaked(emitted(res, out, skip=("summarize.ids.local.log",)), tokens(pedigree)) == []
    id_log = (out / "summarize.ids.local.log").read_text()
    assert "five random ids: ID9" in id_log
    # Lines that lost nothing are not copied.
    assert "wrote " not in id_log
    assert "--id-log writes those lines" not in res.stderr
    # Running again replaces the file.
    assert safe_run("summarize", ped, out, "--id-log").returncode == 0
    assert (out / "summarize.ids.local.log").read_text().count("five random ids") == 1


def test_id_log_needs_safe_attempt(pedigree, tmp_path):
    """``--id-log`` alone is a usage error."""
    ped = write_ped(tmp_path / "ped.tsv", pedigree)
    res = run_pedsum(["summarize", "--in", str(ped), "--out", str(tmp_path / "out"), "--id-log"])
    assert res.returncode == 2
    assert "--id-log needs --safe-attempt" in res.stderr


def test_an_id_in_both_parent_roles_is_not_named(pedigree, tmp_path):
    """The pedigree-graph refusal of an id in both parent roles is reported without the ids."""
    both = [dict(row) for row in pedigree]
    parent = both[0]["id"]
    both[0]["sex"] = ""
    both[2]["mother"], both[2]["father"] = parent, both[5]["id"]
    ped = write_ped(tmp_path / "ped.tsv", both)
    plain = run_pedsum(["summarize", "--in", str(ped), "--out", str(tmp_path / "plain"), "--allow-missing-sex"])
    assert plain.returncode == 1
    assert f"id {parent} is the mother of" in plain.stderr

    res = safe_run("summarize", ped, tmp_path / "out", "--allow-missing-sex")
    assert res.returncode == 1
    assert "an id is both a mother and a father" in res.stderr
    assert leaked(emitted(res, tmp_path / "out"), tokens(both)) == []


@pytest.mark.parametrize(
    ("column", "value", "public"),
    [
        ("score", "TRAITVAL9", "trait 'score' is constant"),
        ("dx", "TRAITVAL9", "trait 'dx' must be numeric"),
    ],
)
def test_a_trait_error_names_the_trait_and_not_its_values(pedigree, tmp_path, column, value, public):
    """A trait typing error names the column; its values go to the ID log only."""
    odd = [{**row, column: value} for row in pedigree]
    odd[1][column] = "1" if column == "dx" else value
    ped = write_ped(tmp_path / "ped.tsv", odd)
    res = safe_run("assortative-mating", ped, tmp_path / "out")
    assert res.returncode == 1
    assert public in res.stderr
    assert leaked(emitted(res, tmp_path / "out"), tokens(odd, ("TRAITVAL9",))) == []
    assert list((tmp_path / "out").iterdir()) == []

    res = safe_run("assortative-mating", ped, tmp_path / "out", "--id-log")
    assert "traitval9" in (tmp_path / "out" / "assortative-mating.ids.local.log").read_text().lower()
    assert "traitval9" not in res.stderr.lower()


def test_an_unexpected_exception_prints_its_type_only(pedigree, tmp_path, monkeypatch):
    """An exception no runner catches prints its type; its traceback goes to the ID log only."""
    from pedsum.cli import summarize

    def boom(*_args, **_kwargs):
        raise ValueError(f"row 3 holds {pedigree[3]['id']}")

    monkeypatch.setattr(summarize, "group_mating_pairs", boom)
    ped = write_ped(tmp_path / "ped.tsv", pedigree)
    res = safe_run("summarize", ped, tmp_path / "out")
    assert res.returncode == 1
    assert f"stopped on an unexpected ValueError: {WITHHELD}" in res.stderr
    assert "Traceback" not in res.stderr
    assert leaked(emitted(res, tmp_path / "out"), tokens(pedigree)) == []

    res = safe_run("summarize", ped, tmp_path / "out", "--id-log")
    assert res.returncode == 1
    assert f"ValueError: row 3 holds {pedigree[3]['id']}" in (tmp_path / "out" / "summarize.ids.local.log").read_text()
    assert leaked(res.stderr, tokens(pedigree)) == []


def test_a_missing_count_and_the_row_total_are_not_logged(pedigree, tmp_path):
    """Neither the count of rows without a trait value nor the total it would be subtracted from is printed."""
    sparse = [dict(row) for row in pedigree]
    for row in sparse[-3:]:
        row["score"] = "NA"
    ped = write_ped(tmp_path / "ped.tsv", sparse)
    plain = run_pedsum(
        ["assortative-mating", "--in", str(ped), "--out", str(tmp_path / "plain"), *COMMANDS["assortative-mating"][0]]
    )
    assert f"trait score: continuous (inferred), 3 missing of {len(sparse)}" in plain.stderr
    for min_cell in ("5", "3"):
        res = safe_run("assortative-mating", ped, tmp_path / f"out{min_cell}", "--min-cell", min_cell, "--id-log")
        assert res.returncode == 0, res.stderr
        assert f"trait score: continuous (inferred), {WITHHELD} missing of {WITHHELD}" in res.stderr
        id_log = (tmp_path / f"out{min_cell}" / "assortative-mating.ids.local.log").read_text()
        assert f"trait score: continuous (inferred), 3 missing of {len(sparse)}" in id_log


TRIO = [
    {"id": "11", "sex": "M", "mother": "NA", "father": "NA", "birth_year": "1950", "score": "1.5", "dx": "0"},
    {"id": "12", "sex": "F", "mother": "NA", "father": "NA", "birth_year": "1952", "score": "2.5", "dx": "1"},
    {"id": "13", "sex": "F", "mother": "12", "father": "11", "birth_year": "1980", "score": "3.5", "dx": "1"},
]

#: What a safe log prints of a ``Public`` number, and the digits its message literals hold.
PUBLIC_NUMBERS = re.compile(
    r"^\d\d:\d\d:\d\d "
    r"|\bin \d+\.\d\ds$"
    r"|memory limit \d+\.\d GiB \(\d+% of \d+\.\d GiB available"
    r"|\(min cell = \d+\)"
    r"|--threads=\d+",
    re.MULTILINE,
)


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_trio_logs_no_number_that_is_not_public(command, tmp_path):
    """Every count of a three-row pedigree is a small cell, so no digit of the log may come from the data.

    pedigree-graph logs ``relatives_per_person total: 2 pairs`` for it.
    """
    write_ped(tmp_path / "ped.tsv", TRIO)
    flags = [*COMMANDS[command][0], "--safe-attempt", "--verbose"]
    flags[flags.index("9999999")] = "2100"
    for out, more in (("out", []), ("logged", ["--id-log"])):
        res = run_pedsum([command, "--in", "ped.tsv", "--out", out, *flags, *more], cwd=tmp_path)
        assert WITHHELD in res.stderr
        assert re.findall(r"^.*\d.*$", PUBLIC_NUMBERS.sub("", res.stderr), re.MULTILINE) == []
    if command == "summarize":
        assert f"pedigree_graph._relatives_per_person: {WITHHELD}" in res.stderr
        assert "relatives_per_person total: 2 pairs" in (tmp_path / "logged" / "summarize.ids.local.log").read_text()


# ---------------------------------------------------------------------------
# File names and the --out directory
# ---------------------------------------------------------------------------


def test_safe_names_come_from_one_map():
    """``output_name`` maps each ordinary name through ``SAFE_SUFFIX`` and has no name for a ``.gz``."""
    assert SAFE_SUFFIX == {".yaml": ".txt", ".tsv": ".tab"}
    assert output_name("summary.extra.yaml", safe=True) == "summary.extra.txt"
    assert output_name("summary.pedigree.tsv", safe=True) == "summary.pedigree.tab"
    assert output_name("summary.yaml", safe=False) == "summary.yaml"
    with pytest.raises(KeyError):
        output_name("annotated.tsv.gz", safe=True)


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_safe_run_writes_only_txt_and_tab(command, pedigree, tmp_path):
    """Each command writes exactly its safe names: YAML text as ``.txt``, long tables as ``.tab``."""
    ped = write_ped(tmp_path / "ped.tsv", pedigree)
    out = tmp_path / "out"
    assert safe_run(command, ped, out).returncode == 0
    written = sorted(out.iterdir())
    assert {path.name for path in written} == COMMANDS[command][1]
    assert {path.suffix for path in written} <= {".txt", ".tab"}
    for path in written:
        if path.suffix == ".txt":
            assert yaml.safe_load(path.read_text())["safe_attempt_min_cell"] == 5
        else:
            assert path.read_text().splitlines()[0] == "section\tkey\tsubkey\tvalue"


def test_summarize_without_tsv_writes_the_two_reports(pedigree, tmp_path):
    """Without ``--tsv`` summarize writes the two reports only."""
    ped = write_ped(tmp_path / "ped.tsv", pedigree)
    res = run_pedsum(["summarize", "--in", str(ped), "--out", str(tmp_path / "out"), "--safe-attempt"])
    assert res.returncode == 0, res.stderr
    assert {path.name for path in (tmp_path / "out").iterdir()} == {"summary.txt", "summary.extra.txt"}


def _snapshot(out: Path) -> dict[str, tuple[bytes, int]]:
    """Every file of ``out`` with its bytes and modification time."""
    return {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in out.iterdir()}


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_safe_run_refuses_a_directory_an_ordinary_run_wrote(command, pedigree, tmp_path):
    """A safe run into an ordinary run's directory exits 1 and changes nothing there."""
    ped = write_ped(tmp_path / "ped.tsv", pedigree)
    out = tmp_path / "out"
    flags = COMMANDS[command][0]
    assert run_pedsum([command, "--in", str(ped), "--out", str(out), *flags]).returncode == 0
    before = _snapshot(out)
    res = safe_run(command, ped, out, "--id-log")
    assert res.returncode == 1
    assert "Pass a new or empty directory" in res.stderr
    assert sorted(before)[0] in res.stderr
    assert _snapshot(out) == before


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_a_safe_run_into_its_own_output_runs_again(command, pedigree, tmp_path):
    """A safe run into its own earlier output succeeds and leaves the same file set."""
    ped = write_ped(tmp_path / "ped.tsv", pedigree)
    out = tmp_path / "out"
    assert safe_run(command, ped, out).returncode == 0
    first = _snapshot(out)
    res = safe_run(command, ped, out)
    assert res.returncode == 0, res.stderr
    assert sorted(_snapshot(out)) == sorted(first)


def test_a_safe_run_refuses_the_files_of_another_safe_run(pedigree, tmp_path):
    """Another command's files, files of other flags and an ID log not asked for are all stale here."""
    ped = write_ped(tmp_path / "ped.tsv", pedigree)
    out = tmp_path / "out"
    assert safe_run("summarize", ped, out).returncode == 0
    assert safe_run("effective-size", ped, out).returncode == 1
    res = run_pedsum(["summarize", "--in", str(ped), "--out", str(out), "--safe-attempt"])
    assert res.returncode == 1
    assert "summary.pedigree.tab" in res.stderr

    logged = tmp_path / "logged"
    assert safe_run("effective-size", ped, logged, "--id-log").returncode == 0
    assert safe_run("effective-size", ped, logged).returncode == 1
    assert safe_run("effective-size", ped, logged, "--id-log").returncode == 0


def test_a_safe_run_creates_an_absent_directory_and_refuses_a_file(pedigree, tmp_path):
    """An absent ``--out`` is created; a file is refused."""
    ped = write_ped(tmp_path / "ped.tsv", pedigree)
    assert safe_run("effective-size", ped, tmp_path / "a" / "b").returncode == 0
    assert safe_run("effective-size", ped, ped).returncode == 1
