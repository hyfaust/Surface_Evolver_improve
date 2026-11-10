"""B2 benchmark: cubic anisotropic surface energy relaxation.

Cubic anisotropy `gamma(n) = 1 + eps*(n_x^4 + n_y^4 + n_z^4)` makes axis-aligned
facets cheapest, so a sphere relaxes toward a rounded cube.  The benchmark starts
from an icosphere, records an energy history, and checks the plan's criteria: the
energy decreases on every accepted step, no NaN appears, and the relaxed shape
keeps the anisotropy's axis symmetry.

Gate G2 asks for 20 iterations of monotone decrease with no NaN; the run uses
`tr_newton_cg`, whose Hessian-vector products expose indefinite curvature, with
L-BFGS as a cross-check.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from semodern import constraints, energy, mesh as mesh_mod, metrics, optimize  # noqa: E402

REPORT = PROJECT_ROOT / "reports" / "results.b2.json"
EPS = 0.1
SUBDIVISIONS = 2
MIN_MONOTONE_STEPS = 20


def icosphere(subdivisions: int = SUBDIVISIONS, radius: float = 1.0) -> mesh_mod.Mesh:
    """Icosphere from an icosahedron with `subdivisions` 1-to-4 refinements."""
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    base_vertices = [
        [-1.0, golden, 0.0], [1.0, golden, 0.0], [-1.0, -golden, 0.0], [1.0, -golden, 0.0],
        [0.0, -1.0, golden], [0.0, 1.0, golden], [0.0, -1.0, -golden], [0.0, 1.0, -golden],
        [golden, 0.0, -1.0], [golden, 0.0, 1.0], [-golden, 0.0, -1.0], [-golden, 0.0, 1.0],
    ]
    base_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    verts = [torch.tensor(v, dtype=torch.float64) for v in base_vertices]
    faces = [list(face) for face in base_faces]
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                midpoint_vector = verts[a] + verts[b]
                verts.append(midpoint_vector / midpoint_vector.norm())
                cache[key] = len(verts) - 1
            return cache[key]

        refined: list[list[int]] = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        faces = refined
    table = torch.stack(verts) * radius
    return mesh_mod.Mesh(verts=table, faces=torch.tensor(faces, dtype=torch.int64))


def axis_symmetry_error(verts: torch.Tensor) -> float:
    """Max deviation of the point cloud from axis-permutation symmetry, normalized.

    For a cubic-anisotropic relaxed shape the sorted per-axis extents must agree; the
    reported number is their spread divided by the mean extent.
    """
    centered = verts - verts.mean(dim=0, keepdim=True)
    extents = centered.abs().max(dim=0).values
    return float((extents.max() - extents.min()) / extents.mean())


def run_solver(name: str, problem: energy.Energy, start_flat: torch.Tensor) -> dict:
    """Run one optimizer and summarize its monotone decrease and symmetry."""
    started_history: list[float] = []

    def callback(iteration: int, value: float, grad_norm: float) -> None:
        started_history.append(value)

    if name == "tr_newton_cg":
        result = optimize.tr_newton_cg(problem, start_flat, max_iter=40, tol=1e-8, callback=callback)
    else:
        result = optimize.lbfgs(problem, start_flat, max_iter=200, tol=1e-8, callback=callback)

    steps = [float(problem.value(start_flat))] + result.history
    monotone = all(after <= before + 1e-10 for before, after in zip(steps, steps[1:]))
    decreasing_steps = sum(1 for before, after in zip(steps, steps[1:]) if after < before)
    final = result.x.reshape(-1, 3)
    return {
        "optimizer": name,
        "iterations": result.iterations,
        "decreasing_steps": decreasing_steps,
        "initial_energy": float(problem.value(start_flat)),
        "final_energy": result.energy,
        "monotone": monotone,
        "finite": bool(torch.isfinite(result.x).all()),
        "grad_norm": result.grad_norm,
        "symmetry_error": axis_symmetry_error(final),
        "radius_spread": float(final.norm(dim=1).max() - final.norm(dim=1).min()),
    }


def main() -> int:
    """Relax the icosphere under cubic anisotropy and record both solvers."""
    sphere = icosphere()
    mesh_mod.validate(sphere)
    anisotropy = constraints.AnisotropicTension(sphere.faces, eps=EPS)
    problem = energy.Energy(sphere.faces, terms=[anisotropy])
    start_flat = sphere.verts.reshape(-1).clone()
    initial = float(problem.value(start_flat))
    print(
        f"anisotropic: V={sphere.n_verts} F={sphere.n_faces} eps={EPS} "
        f"initial energy={initial:.9f} (isotropic area {float(mesh_mod.area(sphere)):.9f})"
    )

    outcomes = [run_solver(name, problem, start_flat) for name in ("tr_newton_cg", "lbfgs")]
    records = []
    for outcome in outcomes:
        passed = bool(
            outcome["monotone"]
            and outcome["finite"]
            and outcome["decreasing_steps"] >= MIN_MONOTONE_STEPS
            and outcome["symmetry_error"] < 0.05
        )
        print(
            f"  {outcome['optimizer']:12s} iters={outcome['iterations']:3d} "
            f"E {outcome['initial_energy']:.6f} -> {outcome['final_energy']:.6f} "
            f"steps={outcome['decreasing_steps']:3d} monotone={outcome['monotone']} "
            f"sym_err={outcome['symmetry_error']:.3e}"
        )
        records.append(
            {
                "branch": "b2",
                "case": "anisotropic",
                "optimizer": outcome["optimizer"],
                "iterations": outcome["iterations"],
                "wall_time": None,
                "energy": outcome["final_energy"],
                "grad_norm": outcome["grad_norm"],
                "area": None,
                "volume": None,
                "se_energy": None,
                "se_area": None,
                "passed": passed,
                "notes": (
                    f"eps={EPS}; energy {outcome['initial_energy']:.6f} -> {outcome['final_energy']:.6f}; "
                    f"{outcome['decreasing_steps']} decreasing steps; monotone={outcome['monotone']}; "
                    f"axis symmetry error {outcome['symmetry_error']:.2e}"
                ),
            }
        )

    payload = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {"cases": []}
    payload["cases"] = [case for case in payload.get("cases", []) if case.get("case") != "anisotropic"]
    payload["cases"].extend(records)
    payload["anisotropic_passed"] = any(record["passed"] for record in records)
    metrics.write_json(REPORT, payload)

    for record in records:
        print(f"  [{'PASS' if record['passed'] else 'FAIL'}] {record['optimizer']}")
    gate = payload["anisotropic_passed"]
    print(f"  [{'PASS' if gate else 'FAIL'}] cubic anisotropy gate (>= {MIN_MONOTONE_STEPS} monotone steps, finite, symmetric)")
    print(f"wrote {REPORT}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
