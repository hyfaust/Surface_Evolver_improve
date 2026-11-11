"""Integration tests: the CLI dispatch table, the runner's case schema and the report."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from semodern import cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "reports" / "results.json"
REPORT = PROJECT_ROOT / "reports" / "REPORT.md"


def test_cli_case_table_covers_every_benchmark() -> None:
    expected = {"cube", "catenoid", "cube_sphere", "sessile", "periodic", "aniso", "octa", "phasefield", "meshfree"}
    assert set(cli.CASES) == expected
    for module in cli.CASES.values():
        assert (PROJECT_ROOT / "benchmarks" / f"{module}.py").is_file()


def test_cli_list_prints_every_case(capsys) -> None:
    assert cli.main(["list"]) == 0
    printed = capsys.readouterr().out
    for name in cli.CASES:
        assert name in printed


def test_cli_rejects_unknown_case() -> None:
    with pytest.raises(SystemExit):
        cli.main(["run", "--case", "nonsense"])


def test_cli_run_cube_passes() -> None:
    # `cube` is the merge gate, so exercising it through the CLI also proves dispatch.
    assert cli.main(["run", "--case", "cube"]) == 0


@pytest.mark.skipif(not RESULTS.exists(), reason="run benchmarks/run_all.py first")
def test_results_schema_is_complete() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    for field in ("branch", "case", "optimizer", "passed", "notes"):
        assert field in payload["schema"]
    assert payload["case_count"] == len(payload["cases"])
    for case in payload["cases"]:
        assert case["passed"] is not None, f"unpopulated passed for {case}"
        assert case["branch"] in {"b1", "b2", "b3", "b4", "b5"}


@pytest.mark.skipif(not RESULTS.exists(), reason="run benchmarks/run_all.py first")
def test_every_branch_contributes_cases() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    branches = {case["branch"] for case in payload["cases"]}
    assert branches == {"b1", "b2", "b3", "b4", "b5"}


@pytest.mark.skipif(not REPORT.exists(), reason="run benchmarks/make_report.py first")
def test_report_mentions_every_branch_and_limit() -> None:
    text = REPORT.read_text(encoding="utf-8")
    for marker in ("B1", "B2", "B3", "B4", "B5", "推荐技术栈", "失败与限制汇总"):
        assert marker in text
    # The documented limitations must survive regeneration.
    assert "phelanc" in text
    assert "pop_vertex" in text


def test_run_all_fails_closed_when_nothing_is_selected(tmp_path) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))
    import run_all

    # `--out` keeps the real report untouched; an empty selection must not report success.
    output = tmp_path / "results.json"
    code = run_all.main(["--only", "no-such-label", "--out", str(output)])
    assert code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    # The merged file still aggregates whatever branch reports exist on disk; what must
    # not happen is a success verdict for a selection that verified nothing.
    assert payload["passed"] is False
    assert payload["notes"] == "no benchmark selected; nothing was verified"
    assert payload["runs"] == []
