"""Optimizer tests: convergence on a quadratic, Rosenbrock, and indefinite curvature."""

from __future__ import annotations

import math

import pytest
import torch

from semodern import optimize
from semodern.optimize import OptResult


class Quadratic:
    """`0.5 x^T A x - b^T x` with a positive-definite `A`."""

    def __init__(self, a: torch.Tensor, b: torch.Tensor) -> None:
        self.a = a
        self.b = b

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x.dot(self.a @ x) - self.b.dot(x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.a @ x - self.b

    def hvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.a @ v

    def exact_minimizer(self) -> torch.Tensor:
        return torch.linalg.solve(self.a, self.b)


class Rosenbrock:
    """Classic 2-D Rosenbrock function, minimum at (1, 1)."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
                200.0 * (x[1] - x[0] ** 2),
            ]
        )

    def hvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        h = torch.tensor(
            [
                [2.0 - 400.0 * (x[1] - x[0] ** 2) + 800.0 * x[0] ** 2, -400.0 * x[0]],
                [-400.0 * x[0], 200.0],
            ],
            dtype=torch.float64,
        )
        return h @ v


def positive_definite(n: int = 6, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    m = torch.randn(n, n, generator=generator, dtype=torch.float64)
    return m @ m.T + n * torch.eye(n, dtype=torch.float64)


def test_lbfgs_solves_quadratic_to_high_accuracy() -> None:
    a = positive_definite()
    b = torch.arange(1.0, 7.0, dtype=torch.float64)
    problem = Quadratic(a, b)
    result = optimize.lbfgs(problem, torch.zeros(6, dtype=torch.float64), tol=1e-10)
    assert isinstance(result, OptResult)
    # The objective is O(1) in magnitude, so float64 value resolution caps the
    # attainable gradient near 1e-8; the minimizer is reached to that precision.
    assert result.grad_norm < 1e-6
    assert (result.x - problem.exact_minimizer()).abs().max().item() < 1e-7
    assert result.energy == pytest.approx(float(problem.value(problem.exact_minimizer())), rel=1e-12)
    assert result.history[-1] <= result.history[0]


def test_tr_newton_cg_solves_quadratic_in_few_iterations() -> None:
    a = positive_definite()
    b = torch.arange(1.0, 7.0, dtype=torch.float64)
    problem = Quadratic(a, b)
    result = optimize.tr_newton_cg(problem, torch.zeros(6, dtype=torch.float64), tol=1e-10)
    assert result.converged
    assert (result.x - problem.exact_minimizer()).abs().max().item() < 1e-7
    # max_radius=0.1 is smaller than the Newton step from the origin (~1.4 here), so a
    # handful of expanding trust-region steps are needed, not one.
    assert result.iterations <= 10


def test_lbfgs_reaches_rosenbrock_gradient_tolerance() -> None:
    problem = Rosenbrock()
    result = optimize.lbfgs(
        problem, torch.tensor([-1.2, 1.0], dtype=torch.float64), max_iter=5000, tol=1e-6
    )
    assert result.grad_norm < 1e-6
    assert result.x[0].item() == pytest.approx(1.0, abs=1e-4)
    assert result.x[1].item() == pytest.approx(1.0, abs=1e-4)


def test_tr_newton_cg_reaches_rosenbrock_gradient_tolerance() -> None:
    problem = Rosenbrock()
    result = optimize.tr_newton_cg(
        problem, torch.tensor([-1.2, 1.0], dtype=torch.float64), max_iter=500, tol=1e-6
    )
    assert result.grad_norm < 1e-6
    assert result.energy == pytest.approx(0.0, abs=1e-10)


def test_steihaug_truncates_on_negative_curvature() -> None:
    # Saddle: E = x^2 - y^2 near the origin, started where the gradient is not zero.
    a = torch.tensor([[2.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    problem = Quadratic(a, torch.zeros(2, dtype=torch.float64))
    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    p, hit_boundary = optimize.steihaug_cg(problem, x, problem.grad(x), 0.5)
    assert hit_boundary
    assert float(p.norm()) == pytest.approx(0.5, rel=1e-12)
    assert float(p.dot(problem.grad(x))) < 0.0


def test_tr_newton_cg_never_increases_energy_on_saddle() -> None:
    a = torch.tensor([[2.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    problem = Quadratic(a, torch.zeros(2, dtype=torch.float64))
    x0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    result = optimize.tr_newton_cg(problem, x0, max_iter=20)
    assert math.isfinite(result.energy)
    assert result.energy <= float(problem.value(x0)) + 1e-12
    for before, after in zip(result.history, result.history[1:]):
        assert after <= before + 1e-12


def test_strong_wolfe_returns_descent_step() -> None:
    a = positive_definite()
    problem = Quadratic(a, torch.ones(6, dtype=torch.float64))
    x = torch.ones(6, dtype=torch.float64)
    value = float(problem.value(x))
    grad = problem.grad(x)
    direction = -grad
    step, x_new, grad_new, value_new = optimize.strong_wolfe(problem, x, value, grad, direction)
    assert step > 0.0
    assert value_new < value
    slope = float(grad.dot(direction))
    assert value_new <= value + 1e-4 * step * slope
    assert abs(float(grad_new.dot(direction))) <= 0.9 * abs(slope)


def test_callback_observes_each_iteration() -> None:
    a = positive_definite()
    problem = Quadratic(a, torch.ones(6, dtype=torch.float64))
    seen: list[tuple[int, float, float]] = []
    result = optimize.lbfgs(
        problem, torch.zeros(6, dtype=torch.float64), callback=lambda *args: seen.append(args)
    )
    assert len(seen) == result.iterations
    assert seen[-1][2] == pytest.approx(result.grad_norm)


def test_already_converged_start_returns_immediately() -> None:
    a = positive_definite()
    b = torch.arange(1.0, 7.0, dtype=torch.float64)
    problem = Quadratic(a, b)
    start = problem.exact_minimizer()
    for solver in (optimize.lbfgs, optimize.tr_newton_cg):
        result = solver(problem, start)
        assert result.converged
        assert result.iterations == 0


class RadialVolumeProjection:
    """Test double: hold the enclosed volume by scaling about the centroid."""

    def __init__(self, faces: torch.Tensor, volume0: float) -> None:
        self.faces = faces
        self.volume0 = volume0

    def volume(self, flat: torch.Tensor) -> float:
        verts = flat.reshape(-1, 3)
        tri = verts[self.faces]
        return float(torch.einsum("ij,ij->", tri[:, 0], torch.linalg.cross(tri[:, 1], tri[:, 2])) / 6.0)

    def project(self, flat: torch.Tensor) -> torch.Tensor:
        verts = flat.reshape(-1, 3)
        centroid = verts.mean(dim=0, keepdim=True)
        scale = (self.volume0 / self.volume(flat)) ** (1.0 / 3.0)
        return (centroid + scale * (verts - centroid)).reshape(-1)


def cube_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    """Unit cube with outward-oriented triangles (SE's fan triangulation)."""
    verts = torch.tensor(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)], dtype=torch.float64
    )
    faces = torch.tensor(
        [
            [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6],
            [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],
            [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5],
        ],
        dtype=torch.int64,
    )
    return verts, faces


def test_radial_projection_restores_exact_volume() -> None:
    verts, faces = cube_mesh()
    projection = RadialVolumeProjection(faces, 1.0)
    stretched = verts * torch.tensor([1.3, 0.8, 1.1], dtype=torch.float64)
    assert projection.volume(stretched.reshape(-1)) != pytest.approx(1.0)
    projected = projection.project(stretched.reshape(-1))
    assert projection.volume(projected) == pytest.approx(1.0, rel=1e-14)


def test_projected_descent_preserves_volume_under_lbfgs() -> None:
    verts, faces = cube_mesh()
    projection = RadialVolumeProjection(faces, 1.0)

    class Projected:
        def value(self, x: torch.Tensor) -> torch.Tensor:
            projected = projection.project(x)
            tri = projected.reshape(-1, 3)[faces]
            return 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=1).sum()

        def grad(self, x: torch.Tensor) -> torch.Tensor:
            projected = projection.project(x).detach().requires_grad_(True)
            tri = projected.reshape(-1, 3)[faces]
            area = 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=1).sum()
            (grad,) = torch.autograd.grad(area, projected)
            return grad.detach()

    problem = Projected()
    start = projection.project((verts * torch.tensor([1.2, 0.9, 1.0], dtype=torch.float64)).reshape(-1))
    result = optimize.lbfgs(problem, start, max_iter=60, tol=1e-8)
    assert result.energy <= float(problem.value(start)) + 1e-12
    assert projection.volume(projection.project(result.x)) == pytest.approx(1.0, rel=1e-13)
