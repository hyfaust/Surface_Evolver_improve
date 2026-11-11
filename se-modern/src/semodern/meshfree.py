"""Mesh-free RBF-FD solver for the axisymmetric catenoid problem.

A catenoid is the surface of revolution of `r(z)` and, as a minimal surface, obeys

.. math::

    \\frac{r''}{(1 + r'^2)^{3/2}} - \\frac{1}{r\\sqrt{1 + r'^2}} = 0,
    \\qquad r(\\pm H) = R.

This module solves that boundary-value problem **without a mesh**: the interval
`[-H, H]` carries `n` nodes, and at every node the first and second derivatives are
approximated by a radial-basis-function finite-difference (RBF-FD) stencil built
from `k` nearest neighbours with the polyharmonic spline `phi(r) = r^3`.  The
nonlinear algebraic system is solved with `scipy.optimize.root`.

Why polyharmonic splines: `r^3` is conditionally positive definite of order 2, so
augmenting the stencil with a linear polynomial (`1, z`) makes the local system
solvable for any distinct nodes while keeping the weights independent of any shape
parameter.  Gaussian RBFs would need a tuned `eps` and are ill-conditioned at these
node counts.

Scope: this branch only solves the smooth axisymmetric BVP.  It makes no attempt at
arbitrary 3-D topology, volume constraints or contact angles; those are the sharp
interface branch's job, and `REPORT.md` states the boundary explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import root

Tensor = torch.Tensor


def stencil_weights(
    nodes: np.ndarray, index: int, neighbours: np.ndarray, *, quadratic: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """RBF-FD weights for the first and second derivatives at `nodes[index]`.

    The interpolant of the stencil values is

        s(x) = sum_a lambda_a * |x - x_a|^3 + c0 + c1 (x - x_i) [+ c2 (x - x_i)^2],

    and its coefficients solve the augmented system

        [ Phi  P ] [ lambda ]   [ f ]     Phi_ab = |x_a - x_b|^3
        [ P^T  0 ] [ c      ] = [ 0 ]     P columns  = 1, (x_a - x_i) [, (x_a - x_i)^2]

    The *derivative* at the centre is then a linear functional of `f`, so its weight
    row is the corresponding row of `A^{-1}` pushed through the derivative of `s`:

        L = [ dPhi/dx |_{x_i} , dP/dx |_{x_i} ] @ A^{-1},   w = L[:n].

    Computing `L` this way (rather than solving a second system whose right-hand side
    holds the RBF derivatives) is what makes the stencil reproduce polynomials; solving
    directly against the derivative of the basis gives a matrix that ignores the
    augmentation block and fails even `f = x` exactly.
    """
    points = nodes[neighbours]
    centre = nodes[index]
    offsets = points - centre
    size = points.shape[0]
    basis = np.abs(offsets[:, None] - offsets[None, :]) ** 3
    columns = [np.ones(size), offsets]
    if quadratic:
        columns.append(offsets**2)
    poly = np.column_stack(columns)
    order = poly.shape[1]
    system = np.zeros((size + order, size + order))
    system[:size, :size] = basis
    system[:size, size:] = poly
    system[size:, :size] = poly.T
    inverse = np.linalg.inv(system)

    # d/dx |x - x_a|^3 at x = x_i, where x_i - x_a = -offset_a; d2 is the second derivative.
    signed = -offsets
    first_row = np.concatenate([3.0 * signed * np.abs(signed), np.array([0.0, 1.0] + [0.0] * (order - 2))])
    second_row = np.concatenate([6.0 * np.abs(offsets), np.array([0.0, 0.0] + [2.0] * (order - 2))])
    first = (first_row @ inverse)[:size]
    second = (second_row @ inverse)[:size]
    return first, second


@dataclass
class RbfStencilSystem:
    """Precomputed RBF-FD derivative weights on a 1-D node set."""

    nodes: np.ndarray
    neighbours: list[np.ndarray]
    first: np.ndarray
    second: np.ndarray

    @classmethod
    def build(cls, nodes: np.ndarray, stencil: int = 7) -> "RbfStencilSystem":
        """Build first/second derivative weights for interior nodes.

        Boundary nodes are excluded from the unknowns, so only interior indices get a
        stencil; `stencil` nearest neighbours are used (clamped to the node count).
        """
        count = nodes.shape[0]
        width = min(stencil, count)
        neighbours: list[np.ndarray] = []
        first_rows: list[np.ndarray] = []
        second_rows: list[np.ndarray] = []
        for index in range(1, count - 1):
            order = np.argsort(np.abs(nodes - nodes[index]))[:width]
            neighbours.append(order)
            d1, d2 = stencil_weights(nodes, index, order)
            first_rows.append(d1)
            second_rows.append(d2)
        return cls(
            nodes=nodes,
            neighbours=neighbours,
            first=np.array(first_rows),
            second=np.array(second_rows),
        )

    def derivative(self, weights: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Apply the stencil rows to `values`, returning interior derivatives."""
        gathered = np.array([values[row] for row in self.neighbours])
        return weights @ np.zeros(0) if weights.size == 0 else (weights * gathered).sum(axis=1)

    def residuals(self, values: np.ndarray) -> np.ndarray:
        """Catenoid minimal-surface residual at every interior node.

        The axisymmetric minimal-surface equation is

            r'' / (1 + r'^2)^{3/2} - 1 / (r sqrt(1 + r'^2)) = 0.

        Both terms are `O(1)` and cancel at the solution, which makes the *discrete*
        residual a demanding test for the derivative stencils but keeps the root
        unique and physical.  (The first-order integral `r / sqrt(1 + r'^2) = a` is
        algebraically equivalent but degenerate: a straight line through the endpoints
        satisfies it with a different `a`, so the solver can converge to the wrong
        branch.)
        """
        radii = values[1:-1]
        first = self.derivative(self.first, values)
        second = self.derivative(self.second, values)
        slope_squared = first * first
        curvature = second / (1.0 + slope_squared) ** 1.5
        neck = 1.0 / (radii * np.sqrt(1.0 + slope_squared))
        return curvature - neck


def catenoid_constant(half_height: float, radius: float) -> float:
    """Solve `R = a cosh(H/a)` for the catenoid parameter `a`.

    `a cosh(H/a)` has a single minimum at `a = H/ sinh(u) ~ 0.858 H` (value
    `R_max ~ 1.5089 H` for the unit case), so a catenoid spanning `(-H, H)` at radius
    `R` exists only for `R > R_max`, and then on two branches.  The *stable* neck
    corresponds to the branch with the larger `a`, which is the root returned here.
    Requesting `R <= R_max` raises `ValueError`: that is the classic result that the
    soap film between rings further apart than `~1.325 R` collapses to two discs.
    """
    minimum = np.linspace(0.05 * half_height, 10.0 * half_height, 200_001)
    values = minimum * np.cosh(half_height / minimum)
    peak = float(values.min())
    neck = float(minimum[int(values.argmin())])
    if radius <= peak:
        raise ValueError(
            f"no catenoid spans H={half_height} at R={radius}: "
            f"the maximum attainable radius is {peak:.6f} at a={neck:.6f}"
        )
    # Bracket the large-a root and bisect; f is monotone increasing beyond the neck.
    low, high = neck, neck
    while low * np.cosh(half_height / low) > radius:
        low *= 0.999
    while high * np.cosh(half_height / high) < radius:
        high *= 1.001
        if high > 1e6 * half_height:
            raise ValueError("catenoid root search diverged")
    for _ in range(300):
        middle = 0.5 * (low + high)
        if middle * np.cosh(half_height / middle) < radius:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def maximum_radius(half_height: float) -> float:
    """Largest ring radius a catenoid can span over `(-half_height, half_height)`."""
    minimum = np.linspace(0.05 * half_height, 10.0 * half_height, 200_001)
    return float((minimum * np.cosh(half_height / minimum)).min())


def analytic_profile(z: np.ndarray, a: float) -> np.ndarray:
    """Exact catenoid profile `r(z) = a cosh(z/a)` about the neck at `z = 0`."""
    return a * np.cosh(z / a)


def solve_catenoid(
    half_height: float = 1.0,
    radius: float = 1.6,
    *,
    node_count: int = 40,
    stencil: int = 7,
    initial: str = "parabolic",
) -> dict:
    """Solve the axisymmetric catenoid BVP with RBF-FD and compare with the exact profile.

    `initial` selects the starting guess: `"parabolic"` uses `R + c(1 - z^2/H^2)` (the
    plan's default), `"analytic"` starts from the exact solution and therefore tests the
    discretization alone.  Returns the node coordinates, the numerical and exact
    profiles, the residual and the relative error.
    """
    nodes = np.linspace(-half_height, half_height, node_count)
    system = RbfStencilSystem.build(nodes, stencil=stencil)
    a = catenoid_constant(half_height, radius)
    exact = analytic_profile(nodes, a)

    if initial == "analytic":
        guess = exact.copy()
    elif initial == "parabolic":
        guess = radius - 0.1 * (1.0 - (nodes / half_height) ** 2)
        guess[0] = radius
        guess[-1] = radius
    else:
        raise ValueError(f"unknown initial guess {initial!r}")

    def residual(interior_values: np.ndarray) -> np.ndarray:
        """Catenoid residual for the interior unknowns, with the Dirichlet ends held."""
        full = np.concatenate(([radius], interior_values, [radius]))
        return system.residuals(full)

    solution = root(residual, guess[1:-1], method="hybr", options={"xtol": 1e-12})
    profile = np.concatenate(([radius], solution.x, [radius]))
    interior = slice(1, -1)
    relative = np.abs(profile[interior] - exact[interior]) / exact[interior]
    return {
        "nodes": nodes,
        "profile": profile,
        "exact": exact,
        "a": a,
        "half_height": half_height,
        "radius": radius,
        "node_count": node_count,
        "stencil": stencil,
        "initial": initial,
        "success": bool(solution.success),
        "message": str(solution.message),
        "residual_inf": float(np.abs(residual(profile[1:-1])).max()),
        "relative_error_max": float(relative.max()),
        "relative_error_mean": float(relative.mean()),
        "neck_numerical": float(profile[node_count // 2]),
        "neck_exact": float(a),
    }


def area_of_revolution(profile: np.ndarray, nodes: np.ndarray) -> float:
    """Surface area `2 pi integral r sqrt(1 + r'^2) dz` by the trapezoid rule."""
    slope = np.gradient(profile, nodes)
    integrand = profile * np.sqrt(1.0 + slope * slope)
    return float(2.0 * np.pi * np.trapezoid(integrand, nodes))


def exact_area(a: float, half_height: float) -> float:
    """Area of the catenoid `r = a cosh(z/a)` between `-H` and `H`.

    `A = 2 pi int a cosh^2(z/a) dz = 2 pi a (H + (a/2) sinh(2H/a))`, since
    `cosh^2 u = (1 + cosh 2u)/2`.  Omitting the leading factor `a` is a common slip and
    makes the reference disagree with the numerical integral by 10% for `a ~ 1.1`.
    """
    return float(2.0 * np.pi * a * (half_height + 0.5 * a * np.sinh(2.0 * half_height / a)))


def to_torch(profile: np.ndarray, nodes: np.ndarray) -> tuple[Tensor, Tensor]:
    """Convert a numeric profile to tensors for downstream tooling."""
    return torch.as_tensor(nodes, dtype=torch.float64), torch.as_tensor(profile, dtype=torch.float64)
