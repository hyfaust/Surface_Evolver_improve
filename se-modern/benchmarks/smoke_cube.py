"""`cube.fe` smoke test: the per-branch merge gate.

Runs the phases fixed by the plan and writes `reports/smoke_cube.json`:

1. dump `cube.fe`'s initial mesh with SE and parse it;
2. assert 14 vertices and 24 faces in both the dump and the datafile;
3. compute Python area/volume and compare with SE's reported values;
4. evolve two steps in SE and re-parse;
5. assert SE exited 0, the dump exists, and the area did not increase;
6. write the JSON report; exit 1 with `SMOKE FAIL: ...` on any violation.

Usage: `python benchmarks/smoke_cube.py`
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from semodern import io_fe, mesh as mesh_mod, se_bridge  # noqa: E402

DATA = PROJECT_ROOT / "data" / "se" / "cube.fe"
WORKDIR_BASE = PROJECT_ROOT / "tmp" / "smoke_cube"
# Process-isolated scratch: prevents concurrent runs sharing run.cmd
WORKDIR = se_bridge.isolated_workdir(WORKDIR_BASE)
REPORT = PROJECT_ROOT / "reports" / "smoke_cube.json"
EXPECTED_VERTS = 14
EXPECTED_FACES = 24


class SmokeFailure(AssertionError):
    """Raised when one of the smoke assertions fails."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def run() -> dict:
    """Execute the seven smoke steps and return the report payload."""
    initial = se_bridge.dump_initial(DATA, WORKDIR)
    parsed = io_fe.parse_fe(DATA)
    check(initial.n_verts == EXPECTED_VERTS, f"dump has {initial.n_verts} vertices, expected {EXPECTED_VERTS}")
    check(initial.n_faces == EXPECTED_FACES, f"dump has {initial.n_faces} faces, expected {EXPECTED_FACES}")
    check(parsed.verts.shape[0] == EXPECTED_VERTS, f"datafile has {parsed.verts.shape[0]} vertices")
    check(parsed.faces.shape[0] == EXPECTED_FACES, f"datafile has {parsed.faces.shape[0]} faces")

    initial_area = float(mesh_mod.area(initial))
    initial_volume = float(mesh_mod.volume(initial))
    check(initial_area > 0.0, f"initial area is not positive: {initial_area}")
    check(abs(abs(initial_volume) - 1.0) < 1e-10, f"initial |volume| = {abs(initial_volume)!r}, expected 1")

    scalars = se_bridge.run_se_timed(DATA, "", WORKDIR, name="scalars.dmp")
    check(scalars.returncode == 0, f"SE scalar probe exited {scalars.returncode}")
    se_volume = scalars.dump_volume if scalars.dump_volume is not None else scalars.volume
    check(se_volume is not None, "SE did not report a body volume")
    check(abs(abs(se_volume) - 1.0) < 1e-10, f"SE |volume| = {abs(se_volume)!r}, expected 1")
    check(
        (initial_volume > 0) == (se_volume > 0),
        f"volume signs disagree: python {initial_volume!r} vs SE {se_volume!r}",
    )
    se_area = scalars.area
    check(se_area is not None, "SE did not report total_area")
    check(abs(se_area - initial_area) < 1e-10, f"SE area {se_area!r} vs python {initial_area!r}")

    evolved = se_bridge.run_se_timed(DATA, "g 2", WORKDIR, name="smoke.dmp")
    check(evolved.returncode == 0, f"SE evolution exited {evolved.returncode}: {evolved.stdout[-400:]}")
    check(evolved.dump_path.is_file(), f"SE produced no dump at {evolved.dump_path}")
    final = io_fe.load_dmp(evolved.dump_path)
    final_area = float(mesh_mod.area(final))
    final_volume = float(mesh_mod.volume(final))
    check(torch.isfinite(torch.tensor(final_area)).item(), "final area is not finite")
    check(final_area <= initial_area + 1e-10, f"area rose from {initial_area!r} to {final_area!r}")

    return {
        "case": "cube",
        "initial_area": initial_area,
        "final_area": final_area,
        "initial_volume": initial_volume,
        "final_volume": final_volume,
        "se_area": se_area,
        "se_volume": se_volume,
        "vertices": initial.n_verts,
        "faces": initial.n_faces,
        "passed": True,
        "notes": "SE g 2 lowers area; python area/volume match SE to 1e-10",
    }


def main() -> int:
    """Run the smoke test, print the outcome and return a process exit code."""
    try:
        payload = run()
    except Exception as error:  # noqa: BLE001 - smoke must always report a failure
        message = str(error) if isinstance(error, SmokeFailure) else f"{type(error).__name__}: {error}"
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps({"case": "cube", "passed": False, "notes": message}, indent=2) + "\n")
        print(f"SMOKE FAIL: {message}")
        if not isinstance(error, SmokeFailure):
            traceback.print_exc()
        return 1
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SMOKE PASS")
    print(
        f"  vertices={payload['vertices']} faces={payload['faces']} "
        f"area {payload['initial_area']:.12f} -> {payload['final_area']:.12f} "
        f"volume {payload['initial_volume']:.12f} -> {payload['final_volume']:.12f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
