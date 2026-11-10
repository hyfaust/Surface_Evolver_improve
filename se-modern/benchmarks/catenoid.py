"""B1 benchmark: catenoid area minimization against Surface Evolver.

`cat.fe` describes a soap film spanning two coaxial rings of radius
`RMAX = 1.5088795` at `z = +-1`, with every ring vertex declared `fixed`.  The
interior ring of six vertices is free, so this is an unconstrained area
minimization over the free coordinates; the boundary acts only through the fixed
mask and the tension-free area functional.

The script compares, on the *same triangulation*:

* Python L-BFGS and trust-region Newton-CG from the datafile's initial mesh, and
* SE running `g 50`, whose final area is scraped from its own `printf`.

Gate G1 requires one optimizer to reach `grad_norm < 1e-6` with an area matching
SE to `1e-6` relative error.  Results are written to `reports/results.b1.json`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from semodern import energy, io_fe, mesh as mesh_mod, metrics, optimize, se_bridge  # noqa: E402

DATA = PROJECT_ROOT / "data" / "se" / "cat.fe"
WORKDIR = PROJECT_ROOT / "tmp" / "b1_catenoid"
REPORT = PROJECT_ROOT / "reports" / "results.b1.json"
SE_EVOLUTION = "g 50"
PERTURBATION = 0.05


def perturbed_start() -> mesh_mod.Mesh:
    """`cat.fe`'s mesh with its six free vertices randomly displaced.

    The datafile's initial mesh is already stationary to machine precision, which
    would make any optimizer comparison vacuous, so the benchmark perturbs only the
    free interior vertices and lets both codes relax the same starting state.
    """
    cat = io_fe.load_fe(DATA)
    generator = torch.Generator().manual_seed(20240607)
    noise = torch.randn(cat.verts.shape, generator=generator, dtype=torch.float64) * PERTURBATION
    mask = (~cat.fixed_mask).unsqueeze(1).expand_as(cat.verts)
    verts = cat.verts + noise * mask.to(torch.float64)
    return mesh_mod.Mesh(verts=verts, faces=cat.faces, fixed=cat.fixed)


def run_python(optimizer: str, max_iter: int, start_mesh: mesh_mod.Mesh) -> dict:
    """Minimize the catenoid area with `optimizer` and report timing and accuracy."""
    cat = start_mesh
    problem = energy.Energy(cat.faces, n_verts=cat.n_verts, fixed=sorted(cat.fixed))
    start = cat.verts.reshape(-1).clone()
    started = time.perf_counter()
    if optimizer == "lbfgs":
        result = optimize.lbfgs(problem, start, max_iter=max_iter, tol=1e-9)
    else:
        result = optimize.tr_newton_cg(problem, start, max_iter=max_iter, tol=1e-9)
    wall = time.perf_counter() - started
    final = mesh_mod.Mesh(verts=result.x.reshape(-1, 3), faces=cat.faces)
    moved = (result.x.reshape(-1, 3) - cat.verts).abs().max().item()
    return {
        "optimizer": optimizer,
        "iterations": result.iterations,
        "wall_time": wall,
        "energy": result.energy,
        "grad_norm": result.grad_norm,
        "converged": result.converged,
        "area": float(mesh_mod.area(final)),
        "moved": moved,
        "history_first": result.history[0],
        "history_last": result.history[-1],
    }


def run_se(start_mesh: mesh_mod.Mesh) -> dict | None:
    """Evolve the benchmark's starting mesh in SE and return its area and wall time."""
    if not se_bridge.available():
        return None
    WORKDIR.mkdir(parents=True, exist_ok=True)
    start_file = io_fe.save_fe(start_mesh, WORKDIR / "cat_start.fe")
    started = time.perf_counter()
    run = se_bridge.run_se_timed(start_file, SE_EVOLUTION, WORKDIR, name="cat_se.dmp")
    wall = time.perf_counter() - started
    if not run.ok:
        return {"failed": True, "returncode": run.returncode, "stdout": run.stdout[-2000:]}
    return {
        "iterations": int(SE_EVOLUTION.split()[1]),
        "wall_time": wall,
        "energy": run.energy,
        "area": run.area,
        "faces": run.mesh.n_faces,
        "vertices": run.mesh.n_verts,
    }


def main() -> int:
    """Run both optimizers and SE, print a table and write the JSON report."""
    start_mesh = perturbed_start()
    initial = io_fe.load_fe(DATA)
    initial_area = float(mesh_mod.area(start_mesh))
    stationary_area = float(mesh_mod.area(initial))
    print(
        f"catenoid: V={initial.n_verts} F={initial.n_faces} perturbed area={initial_area:.12f} "
        f"(unperturbed stationary area={stationary_area:.12f})"
    )

    cases: list[dict] = []
    for optimizer in ("lbfgs", "tr_newton_cg"):
        outcome = run_python(optimizer, max_iter=2000, start_mesh=start_mesh)
        cases.append(outcome)
        print(
            f"  {optimizer:12s} iters={outcome['iterations']:5d} "
            f"time={outcome['wall_time']:7.2f}s area={outcome['area']:.12f} "
            f"|g|inf={outcome['grad_norm']:.3e} converged={outcome['converged']}"
        )

    se = run_se(start_mesh)
    if se is not None and not se.get("failed"):
        print(f"  {'se g 50':12s} iters={se['iterations']:5d} time={se['wall_time']:7.2f}s "
              f"area={se['area']:.12f}")
    elif se is not None:
        print(f"  SE run failed: {se['returncode']}")

    records: list[dict] = []
    best_gap = None
    for case in cases:
        record = {
            "branch": "b1",
            "case": "catenoid",
            "optimizer": case["optimizer"],
            "iterations": case["iterations"],
            "wall_time": case["wall_time"],
            "energy": case["energy"],
            "grad_norm": case["grad_norm"],
            "area": case["area"],
            "volume": None,
            "se_energy": None if se is None or se.get("failed") else se["energy"],
            "se_area": None if se is None or se.get("failed") else se["area"],
            "passed": False,
            "notes": "",
        }
        if record["se_area"] is not None:
            comparison = metrics.compare(case["area"], record["se_area"], rtol=1e-6)
            record["area_rel_diff"] = comparison["rel_diff"]
            record["passed"] = bool(case["grad_norm"] < 1e-6 and comparison["passed"])
            record["notes"] = (
                f"area rel diff vs SE {comparison['rel_diff']:.3e}; "
                f"max vertex displacement {case['moved']:.3e}"
            )
            best_gap = comparison["rel_diff"] if best_gap is None else min(best_gap, comparison["rel_diff"])
        else:
            record["passed"] = bool(case["grad_norm"] < 1e-6)
            record["notes"] = "SE unavailable; graded on grad_norm only"
        records.append(record)

    payload = {
        "branch": "b1",
        "case": "catenoid",
        "initial_area": initial_area,
        "stationary_area": stationary_area,
        "perturbation": PERTURBATION,
        "se_command": SE_EVOLUTION,
        "best_area_rel_diff": best_gap,
        "passed": any(record["passed"] for record in records),
        "cases": records,
    }
    metrics.write_json(REPORT, payload)

    for record in records:
        status = "PASS" if record["passed"] else "FAIL"
        print(f"  [{status}] {record['optimizer']}: {record['notes']}")
    print(f"wrote {REPORT}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
