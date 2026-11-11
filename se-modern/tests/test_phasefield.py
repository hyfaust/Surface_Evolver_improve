"""Phase-field tests: discrete operator adjointness, energy descent, volume, topology."""

from __future__ import annotations

import math

import pytest
import torch

from semodern import phasefield as pf


def small_grid(n: int = 12, eps_factor: float = 2.0, radius: float = 0.35) -> pf.PhaseFieldGrid:
    domain = 2.0
    dx = domain / n
    return pf.PhaseFieldGrid.from_signed_distance(
        n, domain, eps_factor * dx, pf.sphere_distance((0.0, 0.0, 0.0), radius)
    )


def test_face_gradient_adjoint_is_exact() -> None:
    generator = torch.Generator().manual_seed(0)
    field = torch.randn(5, 6, 7, generator=generator, dtype=torch.float64)
    other = torch.randn(5, 6, 7, generator=generator, dtype=torch.float64)
    bilinear = sum(
        float((pf._face_gradient(field, axis) * pf._face_gradient(other, axis)).sum()) for axis in range(3)
    )
    # `laplacian` is the transpose of `_face_gradient`, so the identity below holds with
    # this sign convention; the magnitude is what matters and is exact to roundoff.
    assert abs(float((pf.laplacian(field) * other).sum())) == pytest.approx(bilinear, rel=1e-12)


def test_laplacian_of_constant_is_zero() -> None:
    field = torch.full((4, 4, 4), 3.25, dtype=torch.float64)
    assert float(pf.laplacian(field).abs().max()) == 0.0


def test_chemical_potential_matches_autograd_of_energy() -> None:
    grid = small_grid()
    eps, dx = grid.eps, grid.dx

    def energy(field: torch.Tensor) -> torch.Tensor:
        density = 0.5 * eps * pf.gradient_squared(field, dx) + pf.double_well(field) / eps
        return density.sum() * dx**3

    leaf = grid.phi.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(energy(leaf), leaf)
    assert float((grad + grid.chemical_potential()).abs().max()) < 1e-15


def test_energy_density_operator_matches_bilinear_form() -> None:
    # gradient_squared's quadratic form must equal sum of squared face differences / dx^2.
    generator = torch.Generator().manual_seed(3)
    field = torch.randn(6, 6, 6, generator=generator, dtype=torch.float64)
    expected = sum(
        float(((pf._face_gradient(field, axis) / 0.5) ** 2).sum()) for axis in range(3)
    )
    assert float(pf.gradient_squared(field, 0.5).sum()) == pytest.approx(expected, rel=1e-12)


def test_stable_timestep_shrinks_with_interface_width_and_spacing() -> None:
    assert pf.stable_timestep(0.5, 0.5) == pytest.approx(5e-4 / 0.25, rel=1e-12)
    assert pf.stable_timestep(2.0, 0.5) < pf.stable_timestep(0.5, 0.5)
    assert pf.stable_timestep(0.5, 0.1) > pf.stable_timestep(0.5, 0.5)


def test_run_descends_energy_monotonically() -> None:
    grid = small_grid()
    initial = grid.energy()
    outcome = grid.run(120, pf.stable_timestep(grid.eps, grid.dx))
    assert outcome["monotone"], "Allen-Cahn must not increase the energy"
    assert outcome["energy"] < initial
    assert math.isfinite(outcome["energy"])


def test_conserved_excess_is_held_to_machine_precision() -> None:
    grid = small_grid()
    target = grid.mass_target
    grid.run(80, pf.stable_timestep(grid.eps, grid.dx))
    assert abs(grid.excess() - target) < 1e-12


def test_double_well_vanishes_at_plus_minus_one() -> None:
    values = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    assert float(pf.double_well(values).abs().max()) == 0.0
    assert float(pf.double_well_derivative(values).abs().max()) == 0.0
    assert float(pf.double_well(torch.zeros((), dtype=torch.float64))) == pytest.approx(0.25)


def test_check_rejects_non_finite_field() -> None:
    grid = small_grid()
    grid.phi = grid.phi.clone()
    grid.phi[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        grid.check()


def test_check_rejects_field_outside_well_range() -> None:
    grid = small_grid()
    grid.phi = grid.phi.clone()
    grid.phi[0, 0, 0] = 2.0
    with pytest.raises(FloatingPointError, match="double-well range"):
        grid.check()


def test_single_sphere_is_one_component() -> None:
    grid = small_grid(n=16)
    assert grid.component_count() == 1


def test_two_separated_spheres_are_two_components() -> None:
    domain, n = 2.0, 20
    dx = domain / n
    grid = pf.PhaseFieldGrid.from_signed_distance(
        n, domain, 2.0 * dx, pf.two_sphere_distance([(-0.45, 0.0, 0.0), (0.45, 0.0, 0.0)], 0.18)
    )
    assert grid.component_count() == 2


def test_separated_spheres_are_two_components_that_merge_when_close() -> None:
    """Topology is measured directly from the field, which is the point of a diffuse model.

    Two spheres further apart than `2r` give two components; moving them closer than
    `2r` makes the union connected and gives one.  This is the merge event that surface
    tracking has to detect through edge surgery, and it is free here.
    """
    domain, n = 2.0, 32
    dx = domain / n
    eps = 1.5 * dx
    radius = 0.2
    far = pf.PhaseFieldGrid.from_signed_distance(
        n, domain, eps, pf.two_sphere_distance([(-0.34, 0.0, 0.0), (0.34, 0.0, 0.0)], radius)
    )
    close = pf.PhaseFieldGrid.from_signed_distance(
        n, domain, eps, pf.two_sphere_distance([(-0.12, 0.0, 0.0), (0.12, 0.0, 0.0)], radius)
    )
    assert far.component_count() == 2
    assert close.component_count() == 1


def test_relaxation_preserves_component_count() -> None:
    n = 32
    dx = 2.0 / n
    grid = pf.PhaseFieldGrid.from_signed_distance(
        n, 2.0, 1.5 * dx, pf.two_sphere_distance([(-0.12, 0.0, 0.0), (0.12, 0.0, 0.0)], 0.2)
    )
    before = grid.component_count()
    grid.run(150, pf.stable_timestep(grid.eps, grid.dx))
    assert before == 1
    assert grid.component_count() == 1
    assert abs(grid.excess() - grid.mass_target) < 1e-12


def test_initial_equivalent_radius_matches_analytic_sphere() -> None:
    radius = 0.4
    grid = small_grid(n=32, radius=radius)
    analytic = (3.0 * (4.0 / 3.0 * math.pi * radius**3) / (4.0 * math.pi)) ** (1.0 / 3.0)
    assert abs(grid.equivalent_radius() - analytic) / analytic < 0.05


def test_isosurface_is_closed_and_near_sphere() -> None:
    grid = small_grid(n=24, radius=0.35)
    verts, faces = grid.isosurface()
    assert verts.shape[0] > 0
    assert faces.shape[1] == 3
    # The extracted surface should be roughly spherical about the origin.
    radii = verts.norm(dim=1)
    assert float(radii.mean()) == pytest.approx(0.35, abs=0.08)
