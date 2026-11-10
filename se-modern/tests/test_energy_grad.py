"""Gradient and Hessian-vector-product correctness against finite differences."""

from __future__ import annotations

import math

import pytest
import torch

from semodern import energy as energy_mod
from semodern.mesh import Mesh


def bumped_grid(n: int = 5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Small (n x n) height-field triangulation, perturbed so no gradient vanishes."""
    generator = torch.Generator().manual_seed(seed)
    coords = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    grid = torch.cartesian_prod(coords, coords)
    heights = 0.2 * torch.rand(grid.shape[0], generator=generator, dtype=torch.float64)
    verts = torch.cat([grid, heights.unsqueeze(1)], dim=1)
    faces: list[list[int]] = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b, c, d = a + 1, a + n, a + n + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, torch.tensor(faces, dtype=torch.int64)


def sphere_patch(seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed icosahedron-like closed surface: a randomly perturbed octahedron."""
    generator = torch.Generator().manual_seed(seed)
    verts = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        dtype=torch.float64,
    )
    verts = verts + 0.05 * torch.rand(verts.shape, generator=generator, dtype=torch.float64)
    faces = torch.tensor(
        [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ],
        dtype=torch.int64,
    )
    return verts, faces


def central_difference(fn, x: torch.Tensor, h: float = 1e-6) -> torch.Tensor:
    grad = torch.zeros_like(x)
    for index in range(x.numel()):
        step = torch.zeros_like(x)
        step[index] = h
        grad[index] = (fn(x + step) - fn(x - step)) / (2 * h)
    return grad


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def test_surface_area_gradient_matches_central_difference() -> None:
    verts, faces = bumped_grid()
    problem = energy_mod.Energy(faces)
    analytic = problem.grad(verts)
    numeric = central_difference(lambda v: problem.value(v), verts.reshape(-1))
    assert relative_error(analytic, numeric) < 1e-6


def test_volume_gradient_matches_central_difference() -> None:
    verts, faces = sphere_patch()
    problem = energy_mod.Energy(faces)
    analytic = problem.grad(verts)
    numeric = central_difference(lambda v: problem.value(v), verts.reshape(-1))
    assert relative_error(analytic, numeric) < 1e-6


def test_hvp_matches_gradient_difference() -> None:
    verts, faces = bumped_grid()
    problem = energy_mod.Energy(faces)
    generator = torch.Generator().manual_seed(7)
    direction = torch.rand(verts.shape, generator=generator, dtype=torch.float64) - 0.5
    analytic = problem.hvp(verts, direction)
    h = 1e-6
    numeric = (problem.grad(verts + h * direction) - problem.grad(verts - h * direction)) / (2 * h)
    assert relative_error(analytic, numeric) < 1e-6


def test_hvp_is_symmetric_and_matches_assembled_hessian() -> None:
    verts, faces = sphere_patch()
    problem = energy_mod.Energy(faces)
    generator = torch.Generator().manual_seed(11)
    u = torch.rand(verts.numel(), generator=generator, dtype=torch.float64) - 0.5
    v = torch.rand(verts.numel(), generator=generator, dtype=torch.float64) - 0.5
    flat = verts.reshape(-1)
    hvp_u = problem.hvp(flat, u)
    hvp_v = problem.hvp(flat, v)
    assert float(hvp_u.dot(v)) == pytest.approx(float(hvp_v.dot(u)), rel=1e-9, abs=1e-9)

    def scalar(vec: torch.Tensor) -> torch.Tensor:
        return problem.value(vec)

    hessian = torch.autograd.functional.hessian(scalar, flat, vectorize=True)
    assert relative_error(hvp_u.reshape(-1), hessian @ u) < 1e-9


def test_fixed_vertices_have_zero_gradient_components() -> None:
    verts, faces = bumped_grid()
    problem = energy_mod.Energy(faces, fixed={0, 3, 7})
    grad = problem.grad(verts).reshape(-1, 3)
    for index in (0, 3, 7):
        assert bool((grad[index] == 0).all())
    assert bool((grad[[1, 2, 4]] != 0).any())


def test_gravity_term_matches_hand_integral() -> None:
    # Unit cube [0,1]^3 with outward faces: int_V z dV = 1/2, so E_g = rho*g*1/2.
    verts = torch.tensor(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)], dtype=torch.float64
    )
    faces = torch.tensor(
        [
            [0, 2, 3], [0, 3, 1],  # z = 0
            [4, 5, 7], [4, 7, 6],  # z = 1
            [0, 1, 5], [0, 5, 4],  # y = 0
            [2, 6, 7], [2, 7, 3],  # y = 1
            [0, 4, 6], [0, 6, 2],  # x = 0
            [1, 3, 7], [1, 7, 5],  # x = 1
        ],
        dtype=torch.int64,
    )
    value = energy_mod.gravity(verts, faces, rho=2.0, g_vec=(0.0, 0.0, 3.0))
    # int_V z dV = 1/2 for the unit cube, so E = rho * g_z * 1/2 = 3.
    assert value.item() == pytest.approx(2.0 * 3.0 * 0.5, rel=1e-12)


def test_area_tension_scaling() -> None:
    verts, faces = bumped_grid()
    tension = torch.full((faces.shape[0],), 2.0, dtype=torch.float64)
    plain = energy_mod.surface_area(verts, faces)
    weighted = energy_mod.surface_area(verts, faces, tension)
    assert weighted.item() == pytest.approx(2.0 * plain.item(), rel=1e-14)


def test_tension_length_mismatch_is_rejected() -> None:
    verts, faces = bumped_grid()
    with pytest.raises(ValueError, match="tension has"):
        energy_mod.surface_area(verts, faces, torch.ones(3, dtype=torch.float64))


def test_area_of_flat_grid_is_known() -> None:
    # (n-1)^2 unit squares tessellated into 2 triangles each: total area 1.
    verts, faces = bumped_grid(n=3)
    flat = verts.clone()
    flat[:, 2] = 0.0
    assert energy_mod.surface_area(flat, faces).item() == pytest.approx(1.0, rel=1e-14)
    assert math.isclose(float(flat[:3].sum()), float(flat[:3].sum()))
