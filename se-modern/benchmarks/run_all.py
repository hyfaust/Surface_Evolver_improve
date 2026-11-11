"""Integrated benchmark runner: executes every branch's cases and merges the reports.

Each benchmark script writes its own `reports/results.<branch>.json` on its branch, so
the branches never conflict.  This runner

1. calls each benchmark's `main()` in sequence, capturing pass/fail and timing;
2. merges every `results.*.json` into a single `reports/results.json` with a
   uniform case schema;
3. fails (non-zero exit) if any gate failed, while still writing the merged file so
   the report can show partial results.

Case schema (identical for every branch):

    branch, case, optimizer, iterations, wall_time, energy, grad_norm,
    area, volume, se_energy, se_area, passed, notes

Usage: `python benchmarks/run_all.py [--skip phasefield,meshfree]`
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from semodern import metrics  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
MERGED = REPORTS / "results.json"
CASE_FIELDS = (
    "branch",
    "case",
    "optimizer",
    "iterations",
    "wall_time",
    "energy",
    "grad_norm",
    "area",
    "volume",
    "se_energy",
    "se_area",
    "passed",
    "notes",
)

# Each entry: (label, module name, branch key used in the per-branch report).
BENCHMARKS = (
    ("smoke", "smoke_cube", "smoke"),
    ("b1-catenoid", "catenoid", "b1"),
    ("b1-cube-sphere", "cube_sphere", "b1"),
    ("b1-convergence", "convergence_cube", "b1"),
    ("b2-sessile", "sessile_cap", "b2"),
    ("b2-periodic", "periodic_graph", "b2"),
    ("b2-anisotropic", "anisotropic", "b2"),
    ("b3-topology", "octa_topology", "b3"),
    ("b4-phasefield", "phasefield_sphere", "b4"),
    ("b5-meshfree", "meshfree_catenoid", "b5"),
)


def run_benchmark(module_name: str) -> dict:
    """Import and run one benchmark module's `main()`, capturing the outcome."""
    started = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
    except Exception as error:  # noqa: BLE001 - the runner must survive any branch
        return {
            "module": module_name,
            "returncode": 1,
            "wall_time": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=4),
        }
    try:
        code = module.main()
    except Exception as error:  # noqa: BLE001 - report, do not abort the sweep
        return {
            "module": module_name,
            "returncode": 1,
            "wall_time": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=4),
        }
    return {"module": module_name, "returncode": int(code or 0), "wall_time": time.perf_counter() - started}


def load_branch_cases(reports: Path = REPORTS) -> list[dict]:
    """Merge every `results.*.json` into a uniform case list, skipping the merged file."""
    cases: list[dict] = []
    for path in sorted(reports.glob("results.*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for case in payload.get("cases", []):
            normalized = {field: case.get(field) for field in CASE_FIELDS}
            normalized.setdefault("source", path.name)
            normalized["source"] = path.name
            cases.append(normalized)
    return cases


def merge_reports(runs: list[dict], reports: Path = REPORTS) -> dict:
    """Build `reports/results.json` from the per-branch reports plus the run log."""
    cases = load_branch_cases(reports)
    branch_gates: dict[str, bool] = {}
    for path in sorted(reports.glob("results.*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        branch = payload.get("branch", path.stem.removeprefix("results."))
        branch_gates[branch] = bool(payload.get("passed", False))
    passed = bool(cases) and all(case["passed"] for case in cases) and all(
        run["returncode"] == 0 for run in runs
    )
    return {
        "schema": list(CASE_FIELDS),
        "generated_by": "benchmarks/run_all.py",
        "runs": runs,
        "branch_gates": branch_gates,
        "case_count": len(cases),
        "cases": cases,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    """Run every benchmark, merge the reports and report the overall gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip",
        default="",
        help="comma-separated label prefixes to skip, e.g. 'b4,b5'",
    )
    parser.add_argument(
        "--out",
        default=str(MERGED),
        help="path for the merged report (default: reports/results.json)",
    )
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated label prefixes to run exclusively",
    )
    args = parser.parse_args(argv)
    skip = tuple(item.strip() for item in args.skip.split(",") if item.strip())
    only = tuple(item.strip() for item in args.only.split(",") if item.strip())

    runs: list[dict] = []
    for label, module_name, _branch in BENCHMARKS:
        if only and not any(label.startswith(prefix) for prefix in only):
            continue
        if skip and any(label.startswith(prefix) for prefix in skip):
            print(f"--- skip {label}")
            continue
        print(f"--- {label}")
        outcome = run_benchmark(module_name)
        outcome["label"] = label
        runs.append(outcome)
        status = "ok" if outcome["returncode"] == 0 else "FAILED"
        print(f"    {status} in {outcome['wall_time']:.2f}s")
        if outcome["returncode"] != 0 and "error" in outcome:
            print(f"    {outcome['error']}")

    payload = merge_reports(runs)
    # A selection that ran nothing has no evidence, so it must not report success.
    if not runs:
        payload["passed"] = False
        payload["notes"] = "no benchmark selected; nothing was verified"
    metrics.write_json(args.out, payload)

    print()
    print(f"cases: {payload['case_count']}")
    for branch, gate in sorted(payload["branch_gates"].items()):
        print(f"  branch {branch}: {'PASS' if gate else 'FAIL'}")
    print(f"  overall: {'PASS' if payload['passed'] else 'FAIL'}")
    print(f"wrote {args.out}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
