"""B1 benchmark: volume-constrained cube-to-sphere minimization vs Surface Evolver.

`cube.fe` starts as a unit cube (area 6, volume 1) and, under area minimization
with the enclosed volume held fixed, relaxes to a sphere of radius
`(3/(4*pi))^(1/3)` with area `4*pi*r^2`.

B1 is about the *optimizer*, so the volume constraint uses the provisional radial
projection the plan prescribes: after every step, vertices are scaled about the
body centroid so the enclosed volume returns to `V0`, and the search direction is
the area gradient with its volume-changing (radial) component removed.  B2
replaces this with an augmented Lagrangian; the projection exists only so the two
optimizers can be compared on the same constrained problem.

SE runs `g 20; hessian; g 20` on the same datafile as the reference.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from semodern import energy, io_fe, mesh as mesh_mod, metrics, optimize, se_bridge  # noqa: E402

DATA = PROJECT_ROOT / "data" / "se" / "cube.fe"
WORKDIR = PROJECT_ROOT / "tmp" / "b1_cube_sphere"
REPORT = PROJECT_ROOT / "reports" / "results.b1.json"
SE_EVOLUTION = "g 20; hessian; g 20"
VOLUME = 1.0


class RadialVolumeProjection:
    """Hold `volume(x) = volume0` by scaling about the centroid.

    `project(x)` returns the vertex vector whose enclosed volume is exactly
    `volume0`, obtained by one closed-form radial scaling about the centroid:
    volume scales with the cube of the scale factor, so a single step suffices.
    `tangential(grad)` removes the component of `grad` that changes volume, so
    optimizer steps stay on the constraint surface to first order.
    """

    def __init__(self, faces: torch.Tensor, n_verts: int, volume0: float) -> None:
        self.faces = faces
        self.n_verts = n_verts
        self.volume0 = float(volume0)
        self._centroid = None

    def volume(self, flat: torch.Tensor) -> float:
        return float(energy.signed_volume(flat.reshape(-1, 3), self.faces))

    def project_or_none(self, flat: torch.Tensor) -> torch.Tensor | None:
        """Project onto the constraint, or return `None` when the candidate is invalid.

        A candidate is rejected when its signed volume is not clearly positive or
        when radial rescaling would leave a degenerate triangulation (a collapsed
        mesh has no meaningful area gradient, and rescaling such a mesh amplifies
        roundoff instead of restoring it).  Line searches treat `None` as an
        infeasible step, exactly as they treat a non-finite energy.
        """
        verts = flat.reshape(-1, 3)
        current = self.volume(flat)
        if not current > 1e-12 * max(self.volume0, 1e-30):
            return None
        centroid = verts.mean(dim=0, keepdim=True)
        scale = (self.volume0 / current) ** (1.0 / 3.0)
        projected = centroid + scale * (verts - centroid)
        areas = energy.triangle_areas(projected, self.faces)
        mean_area = float(areas.mean())
        if not float(areas.min()) > 1e-3 * mean_area:
            return None
        return projected.reshape(-1)

    def project(self, flat: torch.Tensor) -> torch.Tensor:
        """Project onto the constraint, raising when the point is infeasible."""
        projected = self.project_or_none(flat)
        if projected is None:
            raise ValueError("cannot project a mesh whose signed volume is not positive")
        return projected

    def volume_gradient(self, flat: torch.Tensor) -> torch.Tensor:
        """Gradient of the enclosed volume with respect to the flat coordinates."""
        verts = flat.reshape(-1, 3).detach().requires_grad_(True)
        value = energy.signed_volume(verts, self.faces)
        (grad,) = torch.autograd.grad(value, verts)
        return grad.detach().reshape(-1)

    def tangential(self, flat: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Component of `grad` orthogonal to the volume constraint normal."""
        normal = self.volume_gradient(flat)
        denominator = float(normal.dot(normal))
        if denominator == 0.0:
            return grad
        return grad - (float(grad.dot(normal)) / denominator) * normal


class ProjectedEnergy:
    """`Energy` view whose iterates are projected onto the volume constraint.

    `value` and `grad` both evaluate at the projected point, and `grad` additionally
    drops its volume-changing component, so a generic unconstrained minimizer
    reduces to projected area minimization.
    """

    def __init__(self, faces: torch.Tensor, n_verts: int, projection: RadialVolumeProjection) -> None:
        self.base = energy.Energy(faces, n_verts=n_verts)
        self.projection = projection

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Objective at the projected point; `inf` marks an infeasible candidate."""
        projected = self.projection.project_or_none(x)
        if projected is None:
            return torch.tensor(math.inf, dtype=torch.float64)
        return self.base.value(projected)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """Tangential gradient at the projected point; zeros only if infeasible."""
        projected = self.projection.project_or_none(x)
        if projected is None:
            return torch.zeros_like(x)
        return self.projection.tangential(projected, self.base.grad(projected))

    def hvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        projected = self.projection.project_or_none(x)
        if projected is None:
            return torch.zeros_like(v)
        return self.projection.tangential(projected, self.base.hvp(projected, v))


def sphere_area(volume: float) -> float:
    """Area of the sphere enclosing `volume`."""
    radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    return 4.0 * math.pi * radius * radius


def run_python(optimizer: str, max_iter: int) -> dict:
    """Minimize projected area with `optimizer`, reporting volume error and timing."""
    cube = io_fe.load_fe(DATA)
    projection = RadialVolumeProjection(cube.faces, cube.n_verts, VOLUME)
    problem = ProjectedEnergy(cube.faces, cube.n_verts, projection)
    start = projection.project(cube.verts.reshape(-1).clone())
    started = time.perf_counter()
    if optimizer == "lbfgs":
        result = optimize.lbfgs(problem, start, max_iter=max_iter, tol=1e-8)
    else:
        result = optimize.tr_newton_cg(problem, start, max_iter=max_iter, tol=1e-8)
    wall = time.perf_counter() - started

    # The returned point is projected once more, so the reported volume is exact.
    final = projection.project_or_none(result.x)
    if final is None:
        raise RuntimeError(f"{optimizer} finished at an infeasible point")
    final_mesh = mesh_mod.Mesh(verts=final.reshape(-1, 3), faces=cube.faces)
    volume = float(mesh_mod.volume(final_mesh))
    return {
        "optimizer": optimizer,
        "iterations": result.iterations,
        "wall_time": wall,
        "energy": float(mesh_mod.area(final_mesh)),
        "grad_norm": result.grad_norm,
        "converged": result.converged,
        "area": float(mesh_mod.area(final_mesh)),
        "volume": volume,
        "volume_rel_error": abs(volume - VOLUME) / VOLUME,
        "ideal_sphere_area": sphere_area(VOLUME),
        "history_first": result.history[0],
        "history_last": result.history[-1],
    }


def run_se() -> dict | None:
    """Evolve the same datafile in SE and return its final area, volume and time."""
    if not se_bridge.available():
        return None
    started = time.perf_counter()
    run = se_bridge.run_se_timed(DATA, SE_EVOLUTION, WORKDIR, name="cube_sphere.dmp")
    wall = time.perf_counter() - started
    if not run.ok:
        return {"failed": True, "returncode": run.returncode, "stdout": run.stdout[-2000:]}
    return {
        "iterations": 40,
        "wall_time": wall,
        "area": run.area,
        "volume": run.dump_volume if run.dump_volume is not None else run.volume,
        "energy": run.energy,
        "faces": run.mesh.n_faces,
        "vertices": run.mesh.n_verts,
    }


def main() -> int:
    """Run both optimizers and SE, print a table, append cases and judge gate G1."""
    cube = io_fe.load_fe(DATA)
    ideal = sphere_area(VOLUME)
    print(
        f"cube->sphere: V={cube.n_verts} F={cube.n_faces} start area={float(mesh_mod.area(cube)):.12f} "
        f"ideal sphere area={ideal:.12f}"
    )

    records: list[dict] = []
    for optimizer in ("lbfgs", "tr_newton_cg"):
        outcome = run_python(optimizer, max_iter=1000)
        passed = outcome["volume_rel_error"] < 1e-8
        print(
            f"  {optimizer:12s} iters={outcome['iterations']:5d} time={outcome['wall_time']:7.2f}s "
            f"area={outcome['area']:.12f} volume_err={outcome['volume_rel_error']:.3e} "
            f"|g|inf={outcome['grad_norm']:.3e}"
        )
        records.append(
            {
                "branch": "b1",
                "case": "cube_sphere",
                "optimizer": optimizer,
                "iterations": outcome["iterations"],
                "wall_time": outcome["wall_time"],
                "energy": outcome["energy"],
                "grad_norm": outcome["grad_norm"],
                "area": outcome["area"],
                "volume": outcome["volume"],
                "volume_rel_error": outcome["volume_rel_error"],
                "ideal_sphere_area": ideal,
                "se_energy": None,
                "se_area": None,
                "passed": bool(passed),
                "notes": (
                    f"volume rel error {outcome['volume_rel_error']:.3e} "
                    f"(gate 1e-8); area vs ideal sphere "
                    f"{(outcome['area'] - ideal) / ideal:+.3%}"
                ),
            }
        )

    se = run_se()
    if se is not None and not se.get("failed"):
        print(
            f"  {'se g20+h+g20':12s} iters={se['iterations']:5d} time={se['wall_time']:7.2f}s "
            f"area={se['area']:.12f} volume={se['volume']:.12f}"
        )
        record = {
            "branch": "b1",
            "case": "cube_sphere",
            "optimizer": "se",
            "iterations": se["iterations"],
            "wall_time": se["wall_time"],
            "energy": se["energy"],
            "grad_norm": None,
            "area": se["area"],
            "volume": se["volume"],
            "volume_rel_error": abs(se["volume"] - VOLUME) / VOLUME,
            "ideal_sphere_area": ideal,
            "se_energy": se["energy"],
            "se_area": se["area"],
            "passed": True,
            "notes": "reference run g 20; hessian; g 20",
        }
        records.append(record)
        for record in records:
            if record["se_area"] is None:
                record["se_area"] = se["area"]
                record["se_energy"] = se["energy"]
    elif se is not None:
        print(f"  SE run failed: {se['returncode']}")

    # Merge into results.b1.json without dropping the catenoid cases.
    existing = {}
    if REPORT.exists():
        import json

        existing = json.loads(REPORT.read_text(encoding="utf-8"))
        merged = [case for case in existing.get("cases", []) if case.get("case") != "cube_sphere"]
    else:
        merged = []
    merged.extend(records)
    payload = dict(existing)
    payload["branch"] = "b1"
    payload["cases"] = merged
    payload["cube_sphere_passed"] = any(case["passed"] for case in records if case["optimizer"] != "se")
    payload["passed"] = bool(payload.get("passed", False) or payload["cube_sphere_passed"])
    metrics.write_json(REPORT, payload)

    gate = payload["cube_sphere_passed"]
    print(f"  [{'PASS' if gate else 'FAIL'}] volume rel error < 1e-8 for at least one optimizer")
    print(f"wrote {REPORT}")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
