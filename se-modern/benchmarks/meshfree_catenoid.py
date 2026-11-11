"""B5 benchmark: mesh-free RBF-FD catenoid versus the exact profile.

The axisymmetric catenoid BVP is solved on a 1-D node set with RBF-FD derivative
stencils and a Newton-type root finder, then compared with the exact profile
`r(z) = a cosh(z/a)` and with the closed-form area.  This isolates what a mesh-free
method buys on a smooth variational problem, and where it stops: nothing here
touches arbitrary topology, volume constraints or contact angles.

Gate G5: at 40 nodes the profile must match to better than 1% with a residual below
`1e-6`.  A refinement study at 20/30/40/60/120 nodes and a stencil-width scan are
reported alongside, together with the physical existence limit on the ring radius
(`R > R_max ~ 1.5089 H`), which is the same number `cat.fe` encodes as `RMAX`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semodern import meshfree as mf, metrics  # noqa: E402

REPORT = PROJECT_ROOT / "reports" / "results.b5.json"
HALF_HEIGHT = 1.0
RADIUS = 1.6
NODE_COUNT = 40
STENCIL = 7
ERROR_TOLERANCE = 1e-2
RESIDUAL_TOLERANCE = 1e-6


def summarize(result: dict) -> dict:
    """Compact, JSON-safe summary of one solved case."""
    area = mf.area_of_revolution(result["profile"], result["nodes"])
    exact_area = mf.exact_area(result["a"], result["half_height"])
    return {
        "node_count": result["node_count"],
        "stencil": result["stencil"],
        "initial": result["initial"],
        "half_height": result["half_height"],
        "radius": result["radius"],
        "a": result["a"],
        "neck_numerical": result["neck_numerical"],
        "neck_exact": result["neck_exact"],
        "relative_error_max": result["relative_error_max"],
        "relative_error_mean": result["relative_error_mean"],
        "residual_inf": result["residual_inf"],
        "area": area,
        "exact_area": exact_area,
        "area_relative_error": abs(area - exact_area) / exact_area,
        "success": result["success"],
    }


def main() -> int:
    """Run the gate case, the refinement study and the stencil scan; write results."""
    print(f"mesh-free catenoid: H={HALF_HEIGHT} R={RADIUS} (R_max={mf.maximum_radius(HALF_HEIGHT):.7f})")

    gate_case = mf.solve_catenoid(
        HALF_HEIGHT, RADIUS, node_count=NODE_COUNT, stencil=STENCIL, initial="parabolic"
    )
    gate = summarize(gate_case)
    gate_passed = bool(
        gate["success"]
        and gate["relative_error_max"] < ERROR_TOLERANCE
        and gate["residual_inf"] < RESIDUAL_TOLERANCE
        and math.isfinite(gate["area"])
    )
    print(
        f"  gate n={gate['node_count']} k={gate['stencil']}: "
        f"profile error {gate['relative_error_max']:.3e}, residual {gate['residual_inf']:.2e}, "
        f"area error {gate['area_relative_error']:.3e}"
    )

    refinement = [
        summarize(mf.solve_catenoid(HALF_HEIGHT, RADIUS, node_count=n, stencil=9))
        for n in (20, 30, 40, 60, 120)
    ]
    for case in refinement:
        print(
            f"  n={case['node_count']:3d} k={case['stencil']} "
            f"error={case['relative_error_max']:.3e} residual={case['residual_inf']:.2e}"
        )

    stencil_scan = [
        summarize(mf.solve_catenoid(HALF_HEIGHT, RADIUS, node_count=60, stencil=k))
        for k in (5, 6, 7, 9)
    ]
    for case in stencil_scan:
        print(f"  stencil k={case['stencil']}: error={case['relative_error_max']:.3e}")

    payload = {
        "branch": "b5",
        "case": "meshfree_catenoid",
        "passed": gate_passed,
        "gate": gate,
        "refinement": refinement,
        "stencil_scan": stencil_scan,
        "existence_limit": {
            "maximum_radius_unit_height": mf.maximum_radius(1.0),
            "cat_fe_rmax": 1.5088795,
            "note": "cat.fe's RMAX is exactly the largest span a catenoid can bridge",
        },
        "limits": [
            "Only the smooth axisymmetric BVP is solved; no 3-D topology",
            "No volume constraint, contact angle or anisotropy support",
            "Percent-level accuracy needs consistent quadratic RBF augmentation; "
            "with linear augmentation the stencils are inconsistent",
        ],
        "cases": [
            {
                "branch": "b5",
                "case": "meshfree_catenoid",
                "optimizer": "rbf-fd+hybr",
                "iterations": None,
                "wall_time": None,
                "energy": None,
                "grad_norm": gate["residual_inf"],
                "area": gate["area"],
                "volume": None,
                "se_energy": None,
                "se_area": None,
                "passed": gate_passed,
                "notes": (
                    f"n={gate['node_count']} k={gate['stencil']}; profile error "
                    f"{gate['relative_error_max']:.3e}; residual {gate['residual_inf']:.2e}; "
                    f"area error {gate['area_relative_error']:.3e}"
                ),
            }
        ],
    }
    metrics.write_json(REPORT, payload)

    print(f"  [{'PASS' if gate_passed else 'FAIL'}] G5: profile error < 1% and residual < 1e-6 at 40 nodes")
    print(f"wrote {REPORT}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
