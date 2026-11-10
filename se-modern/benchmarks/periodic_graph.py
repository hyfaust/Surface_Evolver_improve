"""B2 benchmark: periodic graph relaxation toward a plane.

A periodic height field `z(x, y)` on the torus `[0, L)^2` starts as a sinusoidal
bump.  Area minimization with no volume constraint must drive it to the flat
plane, whose area is exactly `L^2`.  Wrapped triangles are measured with the
minimum-image convention (`constraints.PeriodicGraph`), so the initial bump's
area is a genuine surface area rather than an artefact of the coordinate wrap.

Gate G2 requires the final area to reach the plane to within `1e-3` while
decreasing monotonically.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from semodern import constraints, metrics, optimize  # noqa: E402

REPORT = PROJECT_ROOT / "reports" / "results.b2.json"
GRID = 16
BOX = (1.0, 1.0)
AMPLITUDE = 0.05
PLANE_TOLERANCE = 1e-3


class GraphEnergy:
    """Adapter exposing `PeriodicGraph` through the `Energy` protocol."""

    def __init__(self, graph: constraints.PeriodicGraph) -> None:
        self.graph = graph

    def value(self, z: torch.Tensor) -> torch.Tensor:
        return self.graph.area(z)

    def grad(self, z: torch.Tensor) -> torch.Tensor:
        return self.graph.grad(z)

    def hvp(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        leaf = z.clone().detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(self.value(leaf), leaf, create_graph=True)
        (product,) = torch.autograd.grad(grad, leaf, grad_outputs=v)
        return product.detach()


def run() -> dict:
    """Relax the bump and report the area trajectory against the flat plane."""
    graph = constraints.PeriodicGraph.from_grid(GRID, BOX)
    heights = AMPLITUDE * torch.sin(2 * math.pi * graph.x) * torch.sin(2 * math.pi * graph.y)
    plane = BOX[0] * BOX[1]
    initial = float(graph.area(heights))
    result = optimize.lbfgs(GraphEnergy(graph), heights, max_iter=400, tol=1e-10)
    final = float(graph.area(result.x))
    monotonically_decreasing = all(
        after <= before + 1e-12 for before, after in zip(result.history, result.history[1:])
    )
    reached_plane = abs(final - plane) <= PLANE_TOLERANCE
    height_max = float(result.x.abs().max())
    return {
        "grid": GRID,
        "amplitude": AMPLITUDE,
        "initial_area": initial,
        "final_area": final,
        "plane_area": plane,
        "area_gap": abs(final - plane),
        "max_height": height_max,
        "iterations": result.iterations,
        "grad_norm": result.grad_norm,
        "monotone": monotonically_decreasing,
        "passed": bool(reached_plane and monotonically_decreasing),
    }


def append_to_report(outcome: dict) -> Path:
    """Merge this case into `results.b2.json`, keeping other B2 cases intact."""
    import json

    payload = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {"cases": []}
    payload["cases"] = [case for case in payload.get("cases", []) if case.get("case") != "periodic_graph"]
    payload["cases"].append(
        {
            "branch": "b2",
            "case": "periodic_graph",
            "optimizer": "lbfgs",
            "iterations": outcome["iterations"],
            "wall_time": None,
            "energy": outcome["final_area"],
            "grad_norm": outcome["grad_norm"],
            "area": outcome["final_area"],
            "volume": None,
            "se_energy": None,
            "se_area": None,
            "passed": outcome["passed"],
            "notes": (
                f"area {outcome['initial_area']:.9f} -> {outcome['final_area']:.9f} "
                f"(plane {outcome['plane_area']:.9f}, gap {outcome['area_gap']:.2e}); "
                f"monotone={outcome['monotone']}"
            ),
        }
    )
    payload["periodic_passed"] = outcome["passed"]
    return metrics.write_json(REPORT, payload)


def main() -> int:
    """Run the relaxation, print the trajectory summary and record the case."""
    outcome = run()
    print(
        f"periodic graph {outcome['grid']}x{outcome['grid']}: area "
        f"{outcome['initial_area']:.9f} -> {outcome['final_area']:.9f} "
        f"(plane {outcome['plane_area']:.9f})"
    )
    print(
        f"  iterations={outcome['iterations']} max_height={outcome['max_height']:.3e} "
        f"monotone={outcome['monotone']} gap={outcome['area_gap']:.2e}"
    )
    append_to_report(outcome)
    print(f"  [{'PASS' if outcome['passed'] else 'FAIL'}] plane reached within {PLANE_TOLERANCE:g}")
    print(f"wrote {REPORT}")
    return 0 if outcome["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
