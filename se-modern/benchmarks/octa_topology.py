"""B3 benchmark: invariant-checked topology surgery on `octa.fe`'s film.

Surface Evolver drives foam topology with heuristic commands (`o` pops, `u`/`t`
swaps, `V` collapses); this benchmark exercises the same operations from
`semodern.topology` and records the exact effect on `(V, E, F)` and the Euler
characteristic.

Cases:

* `octa.fe` parsed, structurally validated and refined (non-manifold film allowed);
* a valence-3 vertex popped on the open patch fixture;
* a triple line popped on the closed triple-line fixture;
* 100 random refine/swap/collapse operations, counting invariant failures.

`octa.fe`'s central vertex is incident to all twelve faces, so `pop_vertex` cannot
apply to the datafile as given; the report says so explicitly, exactly as the plan
permits.  Gate G3 requires zero invariant failures across the random sweep and the
expected counts for each explicit operation.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from semodern import io_fe, mesh as mesh_mod, metrics, se_bridge, topology as topo  # noqa: E402

REPORT = PROJECT_ROOT / "reports" / "results.b3.json"
DATA = PROJECT_ROOT / "data" / "se"
RANDOM_OPERATIONS = 100
MAX_SWEEP_FACES = 800


def counts(mesh: mesh_mod.Mesh) -> tuple[int, int, int]:
    """`(V, E, F)` of a mesh."""
    return topo._counts(mesh)


def random_sweep(operations: int = RANDOM_OPERATIONS, seed: int = 20240607) -> dict:
    """Apply random legal operations and count every invariant failure as zero-tolerance."""
    generator = random.Random(seed)
    mesh = topo.refine(topo.tetrahedral_fixture())[0]
    applied: dict[str, int] = {"refine": 0, "swap_edge": 0, "collapse_edge": 0}
    rejected = 0
    invariant_failures = 0
    for _ in range(operations):
        # Refinement multiplies the face count by four, so a real topology controller
        # stops refining once the mesh is large enough; without this cap 100 refinements
        # would ask for 4^100 faces.
        choices = ("refine", "swap_edge", "collapse_edge") if mesh.n_faces < MAX_SWEEP_FACES else (
            "swap_edge",
            "collapse_edge",
        )
        choice = generator.choice(choices)
        edges, incident = mesh_mod.edge_table(mesh)
        interior = [slot for slot, inc in enumerate(incident) if len(inc) == 2]
        try:
            if choice == "refine":
                mesh, _ = topo.refine(mesh)
            elif choice == "swap_edge":
                if not interior:
                    rejected += 1
                    continue
                mesh, _ = topo.swap_edge(mesh, generator.choice(interior))
            else:
                candidates = list(interior)
                generator.shuffle(candidates)
                result = None
                for slot in candidates:
                    try:
                        result = topo.collapse_edge(mesh, slot)
                        break
                    except ValueError:
                        continue
                if result is None:
                    rejected += 1
                    continue
                mesh, _ = result
        except ValueError:
            rejected += 1
            continue
        applied[choice] += 1
        try:
            mesh_mod.validate(mesh)
        except ValueError:
            invariant_failures += 1
    return {
        "operations": operations,
        "applied": applied,
        "rejected": rejected,
        "invariant_failures": invariant_failures,
        "final_counts": counts(mesh),
        "final_euler": mesh_mod.euler_characteristic(mesh),
        "passed": invariant_failures == 0 and sum(applied.values()) + rejected == operations,
    }


def explicit_cases() -> list[dict]:
    """Each named operation with its recorded counts and expected deltas."""
    cases: list[dict] = []

    octa_path = DATA / "octa.fe"
    octa = io_fe.load_fe(octa_path)
    refined, change = topo.refine(octa, allow_nonmanifold=True)
    cases.append(
        {
            "operation": "refine(octa.fe, allow_nonmanifold)",
            "before": change.before,
            "after": change.after,
            "delta": change.delta,
            "euler_before": change.euler_before,
            "euler_after": change.euler_after,
            "passed": bool(
                change.after[2] == 4 * change.before[2] and change.euler_before == change.euler_after
            ),
            "notes": (
                "octa.fe is a non-manifold film (four faces on some edges); the central "
                f"vertex has valence {len(topo.vertex_faces(octa)[0])} so pop_vertex does not "
                "apply to the datafile as given"
            ),
        }
    )

    patch = topo.trihedral_patch_fixture()
    popped, vertex_change = topo.pop_vertex(patch, 0)
    cases.append(
        {
            "operation": "pop_vertex(patch, 0)",
            "before": vertex_change.before,
            "after": vertex_change.after,
            "delta": vertex_change.delta,
            "euler_before": vertex_change.euler_before,
            "euler_after": vertex_change.euler_after,
            "passed": bool(vertex_change.delta == (-1, -3, -2) and popped.n_faces == 1),
            "notes": "three films around a valence-3 vertex merge into one face",
        }
    )

    film = topo.triple_line_fixture()
    retracted, edge_change = topo.pop_edge(film, 0)
    cases.append(
        {
            "operation": "pop_edge(triple_line, 0)",
            "before": edge_change.before,
            "after": edge_change.after,
            "delta": edge_change.delta,
            "euler_before": edge_change.euler_before,
            "euler_after": edge_change.euler_after,
            "passed": bool(
                retracted.n_faces == edge_change.before[2] - 2
                and edge_change.euler_after == edge_change.euler_before + topo.POP_EDGE_EULER_SHIFT
            ),
            "notes": "three films sharing a triple line merge into one face; chi rises by 3",
        }
    )

    quad = topo.swappable_quad_fixture()
    edges_quad, incident_quad = mesh_mod.edge_table(quad)
    shared = [slot for slot, inc in enumerate(incident_quad) if len(inc) == 2]
    flip_slot = None
    for slot in shared:
        try:
            swapped, swap_change = topo.swap_edge(quad, slot)
            flip_slot = slot
            break
        except ValueError:
            continue
    if flip_slot is None:
        raise RuntimeError("no swappable interior edge in the quad fixture")
    cases.append(
        {
            "operation": f"swap_edge(quad, {flip_slot})",
            "before": swap_change.before,
            "after": swap_change.after,
            "delta": swap_change.delta,
            "euler_before": swap_change.euler_before,
            "euler_after": swap_change.euler_after,
            "passed": bool(swap_change.delta == (0, 0, 0) and swapped.n_faces == quad.n_faces),
            "notes": "edge flip keeps V, E, F and chi unchanged",
        }
    )

    phelanc_path = DATA / "phelanc.fe"
    phelanc = io_fe.load_fe(phelanc_path)
    topo.structural_validate(phelanc)
    cases.append(
        {
            "operation": "structural_validate(phelanc.fe)",
            "before": counts(phelanc),
            "after": counts(phelanc),
            "delta": (0, 0, 0),
            "euler_before": mesh_mod.euler_characteristic(phelanc),
            "euler_after": mesh_mod.euler_characteristic(phelanc),
            "passed": True,
            "notes": (
                "periodic structure: parsed and structurally validated only; periodic wrap is "
                "not a topology operation, and the SE absolute area is not reproduced because "
                "SE derives periodic facet areas from per-edge wrap flags"
            ),
        }
    )
    return cases


def se_reference() -> dict | None:
    """SE's own element counts for `octa.fe`'s initial state."""
    if not se_bridge.available():
        return None
    data = DATA / "octa.fe"
    run = se_bridge.dump_initial(data, PROJECT_ROOT / "tmp" / "b3_octa")
    return {"vertices": run.n_verts, "faces": run.n_faces, "euler": mesh_mod.euler_characteristic(run)}


def main() -> int:
    """Run the topology benchmark, print results and write `results.b3.json`."""
    cases = explicit_cases()
    sweep = random_sweep()
    se = se_reference()

    print(f"explicit operations ({len(cases)}):")
    for case in cases:
        print(
            f"  [{'PASS' if case['passed'] else 'FAIL'}] {case['operation']}: "
            f"{case['before']} -> {case['after']} delta={case['delta']} "
            f"chi {case['euler_before']}->{case['euler_after']}"
        )
    print(
        f"random sweep: {sweep['operations']} ops, applied={sweep['applied']}, "
        f"rejected={sweep['rejected']}, invariant_failures={sweep['invariant_failures']} "
        f"final={sweep['final_counts']} chi={sweep['final_euler']}"
    )
    if se is not None:
        print(f"SE octa.fe initial: V={se['vertices']} F={se['faces']} chi={se['euler']}")

    gate = payload_passed(cases, sweep)
    payload = {
        "branch": "b3",
        "case": "octa_topology",
        "passed": gate,
        "se_reference": se,
        "sweep": sweep,
        "operations": cases,
        "cases": [
            {
                "branch": "b3",
                "case": "octa_topology",
                "optimizer": "topology",
                "iterations": sweep["operations"],
                "wall_time": None,
                "energy": None,
                "grad_norm": None,
                "area": None,
                "volume": None,
                "se_energy": None,
                "se_area": None,
                "passed": gate,
                "notes": (
                    f"{len(cases)} explicit operations plus {sweep['operations']} random operations; "
                    f"invariant failures {sweep['invariant_failures']}; "
                    f"pop_edge chi shift {topo.POP_EDGE_EULER_SHIFT}"
                ),
            }
        ],
    }
    metrics.write_json(REPORT, payload)

    for case in cases:
        print(f"  {'PASS' if case['passed'] else 'FAIL'}: {case['notes'][:96]}")
    print(f"  [{'PASS' if gate else 'FAIL'}] G3: zero invariant failures, expected counts")
    print(f"wrote {REPORT}")
    return 0 if gate else 1


def payload_passed(cases: list[dict], sweep: dict) -> bool:
    """Single source of truth for the benchmark's pass condition."""
    return bool(all(case["passed"] for case in cases) and sweep["passed"])


if __name__ == "__main__":
    sys.exit(main())
