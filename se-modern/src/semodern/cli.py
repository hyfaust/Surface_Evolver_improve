"""Command-line entry point for the prototype's benchmarks.

    python -m semodern.cli run --case cube
    python -m semodern.cli run --case catenoid
    python -m semodern.cli list

Every case maps to the corresponding benchmark script's `main()`, so the CLI adds no
algorithm of its own; it exists so a single case can be reproduced without running
the whole sweep.  Exit status is the case's own pass/fail.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = PROJECT_ROOT / "benchmarks"

# Public case name -> benchmark module implementing it.
CASES: dict[str, str] = {
    "cube": "smoke_cube",
    "catenoid": "catenoid",
    "cube_sphere": "cube_sphere",
    "sessile": "sessile_cap",
    "periodic": "periodic_graph",
    "aniso": "anisotropic",
    "octa": "octa_topology",
    "phasefield": "phasefield_sphere",
    "meshfree": "meshfree_catenoid",
}


def _load(module_name: str):
    """Import a benchmark module from the `benchmarks/` directory."""
    if str(BENCHMARKS) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS))
    return importlib.import_module(module_name)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested case."""
    parser = argparse.ArgumentParser(prog="semodern", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one benchmark case")
    run.add_argument("--case", required=True, choices=sorted(CASES), help="case to run")

    sub.add_parser("list", help="list available cases")

    args = parser.parse_args(argv)
    if args.command == "list":
        for name, module in sorted(CASES.items()):
            print(f"{name:12s} -> benchmarks/{module}.py")
        return 0

    module = _load(CASES[args.case])
    return int(module.main() or 0)


if __name__ == "__main__":
    sys.exit(main())
