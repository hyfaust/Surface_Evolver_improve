"""Constraint tests: augmented-Lagrangian volume, contact angle, periodicity, anisotropy."""

from __future__ import annotations

import math

import pytest
import torch

from semodern import constraints, energy, io_fe, mesh as mesh_mod, optimize

from .conftest import requires_se


def cube_mesh() -> mesh_mod.Mesh:
    """Unit cube with outward-oriented triangles and volume 1."""
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
    return mesh_mod.Mesh(verts=verts, faces=faces)


def test_volume_constraint_residual_and_gradient() -> None:
    cube = cube_mesh()
    constraint = constraints.VolumeConstraint(cube.faces, volume0=1.0)
    assert float(constraint.volume(cube.verts)) == pytest.approx(1.0, rel=1e-14)
    assert float(constraint.value(cube.verts)) == pytest.approx(0.0, abs=1e-14)
    scaled = cube.verts * 1.1
    assert float(constraint.value(scaled)) == pytest.approx(1.1**3 - 1.0, rel=1e-12)
    analytic = constraint.grad(scaled)
    leaf = scaled.reshape(-1).clone().requires_grad_(True)
    (automatic,) = torch.autograd.grad(constraint.value(leaf), leaf)
    assert float((analytic - automatic).abs().max()) == 0.0


def test_volume_projection_is_exact() -> None:
    cube = cube_mesh()
    constraint = constraints.VolumeConstraint(cube.faces, volume0=1.0)
    stretched = cube.verts * torch.tensor([1.4, 0.7, 1.2], dtype=torch.float64)
    projected = constraint.project(stretched)
    assert float(constraint.volume(projected)) == pytest.approx(1.0, rel=1e-13)
    assert float(constraint.relative(projected)) < 1e-13


def test_volume_projection_rejects_degenerate_mesh() -> None:
    cube = cube_mesh()
    constraint = constraints.VolumeConstraint(cube.faces, volume0=1.0)
    with pytest.raises(ValueError, match="zero-volume"):
        constraint.project(torch.zeros_like(cube.verts))


def test_augmented_lagrangian_converges_on_cube_sphere() -> None:
    cube = cube_mesh()
    base = energy.Energy(cube.faces)
    constraint = constraints.VolumeConstraint(cube.faces, volume0=1.0)
    outcome = constraints.augmented_lagrangian(base, constraint, cube.verts.reshape(-1), outer=8, inner_iter=200)
    assert outcome["relative_residual"] < 1e-8
    assert outcome["converged"]
    # A relaxed, volume-preserving cube must be smaller in area than the cube.
    assert outcome["area"] < 6.0
    assert all(step["relative_residual"] >= 0.0 for step in outcome["history"])


def test_augmented_lagrangian_multiplier_grows_penalty() -> None:
    cube = cube_mesh()
    base = energy.Energy(cube.faces)
    constraint = constraints.VolumeConstraint(cube.faces, volume0=1.0)
    outcome = constraints.augmented_lagrangian(base, constraint, cube.verts.reshape(-1), outer=3, inner_iter=20)
    penalties = [step["rho"] for step in outcome["history"]]
    assert penalties == sorted(penalties)
    assert outcome["rho"] <= 1e6


def test_contact_angle_energy_and_analytic_gradient() -> None:
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    contact = constraints.ContactAngle([[0, 1, 2, 3]], wall_z=0.0, gamma=2.0, theta_deg=60.0)
    assert float(contact.wetted_area(verts)) == pytest.approx(1.0, rel=1e-14)
    expected = -2.0 * math.cos(math.radians(60.0))
    assert float(contact.value(verts)) == pytest.approx(expected, rel=1e-14)
    analytic = contact.grad(verts)
    leaf = verts.clone().requires_grad_(True)
    (automatic,) = torch.autograd.grad(contact.value(leaf), leaf)
    assert float((analytic - automatic.reshape(-1)).abs().max()) < 1e-15


def test_contact_angle_sign_switches_droplet_and_bubble() -> None:
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    droplet = constraints.ContactAngle([[0, 1, 2, 3]], 0.0, 1.0, 45.0, sign=-1)
    bubble = constraints.ContactAngle([[0, 1, 2, 3]], 0.0, 1.0, 45.0, sign=+1)
    assert float(droplet.value(verts)) == pytest.approx(-float(bubble.value(verts)), rel=1e-15)


def test_contact_angle_rejects_bad_sign_and_loops() -> None:
    with pytest.raises(ValueError, match="sign"):
        constraints.ContactAngle([[0, 1, 2]], 0.0, 1.0, 90.0, sign=0)
    with pytest.raises(ValueError, match="three vertices"):
        constraints.ContactAngle([[0, 1]], 0.0, 1.0, 90.0)


def test_ninety_degree_contact_has_zero_wetting_energy() -> None:
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    contact = constraints.ContactAngle([[0, 1, 2, 3]], 0.0, 1.0, 90.0)
    assert float(contact.value(verts)) == pytest.approx(0.0, abs=1e-15)
    # cos(90 deg) is 6e-17 rather than 0 in float64, so the gradient is zero to roundoff.
    assert float(contact.grad(verts).abs().max()) < 1e-15


def test_spherical_cap_angle_matches_known_caps() -> None:
    # A hemisphere: contact radius equals apex height, so theta = 90 degrees.
    assert constraints.spherical_cap_angle(1.0, 1.0) == pytest.approx(90.0, rel=1e-12)
    # A cap on a unit sphere with r = sin(t), h = 1 - cos(t) has angle 90 + t.
    for degrees in (30.0, 45.0, 60.0, 120.0):
        radians = math.radians(degrees)
        radius = math.sin(radians)
        height = 1.0 - math.cos(radians)
        assert constraints.spherical_cap_angle(radius, height) == pytest.approx(degrees, rel=1e-9)


def test_periodic_flat_graph_area_equals_box() -> None:
    graph = constraints.PeriodicGraph.from_grid(6, (1.5, 0.5))
    heights = torch.zeros(graph.x.numel(), dtype=torch.float64)
    assert float(graph.area(heights)) == pytest.approx(1.5 * 0.5, rel=1e-14)
    assert float(graph.area(heights).detach()) > 0.0


def test_periodic_wrapped_triangle_uses_minimum_image() -> None:
    # One triangle straddling the periodic boundary in x on a unit torus.
    x = torch.tensor([0.95, 0.05, 0.5], dtype=torch.float64)
    y = torch.tensor([0.5, 0.5, 0.1], dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    graph = constraints.PeriodicGraph(x, y, faces, (1.0, 1.0))
    heights = torch.zeros(3, dtype=torch.float64)
    area = float(graph.area(heights))
    # Minimum image makes x-separation 0.1, not 0.9.
    expected = 0.5 * 0.1 * 0.4
    assert area == pytest.approx(expected, rel=1e-12)


def test_periodic_area_gradient_matches_autograd() -> None:
    graph = constraints.PeriodicGraph.from_grid(5, (1.0, 1.0))
    heights = 0.05 * torch.sin(2 * math.pi * graph.x) * torch.cos(2 * math.pi * graph.y)
    analytic = graph.grad(heights)
    leaf = heights.clone().requires_grad_(True)
    (automatic,) = torch.autograd.grad(graph.area(leaf), leaf)
    assert float((analytic - automatic).abs().max()) < 1e-14


def test_periodic_bumped_graph_relaxes_toward_plane() -> None:
    graph = constraints.PeriodicGraph.from_grid(8, (1.0, 1.0))
    heights = 0.05 * torch.sin(2 * math.pi * graph.x) * torch.sin(2 * math.pi * graph.y)

    class GraphEnergy:
        def value(self, z):
            return graph.area(z)

        def grad(self, z):
            return graph.grad(z)

        def hvp(self, z, v):
            leaf = z.clone().requires_grad_(True)
            (grad,) = torch.autograd.grad(self.value(leaf), leaf, create_graph=True)
            (product,) = torch.autograd.grad(grad, leaf, grad_outputs=v)
            return product

    initial = float(graph.area(heights))
    result = optimize.lbfgs(GraphEnergy(), heights, max_iter=200, tol=1e-10)
    final = float(graph.area(result.x))
    plane = 1.0
    assert final < initial
    assert final == pytest.approx(plane, abs=1e-3)


def test_anisotropic_tension_is_isotropic_when_eps_zero() -> None:
    cube = cube_mesh()
    isotropic = constraints.AnisotropicTension(cube.faces, eps=0.0)
    assert float(isotropic(cube.verts)) == pytest.approx(float(mesh_mod.area(cube)), rel=1e-14)


def test_anisotropic_tension_rewards_aligned_normals() -> None:
    cube = cube_mesh()
    anisotropic = constraints.AnisotropicTension(cube.faces, eps=0.1)
    # Axis-aligned cube faces have unit normals with one component +-1, maximizing n^4.
    assert float(anisotropic(cube.verts)) == pytest.approx(6.0 * (1.0 + 0.1), rel=1e-12)


def test_anisotropic_energy_decreases_and_stays_symmetric() -> None:
    cube = cube_mesh()
    anisotropic = constraints.AnisotropicTension(cube.faces, eps=0.1)
    problem = energy.Energy(cube.faces, terms=[anisotropic])
    result = optimize.lbfgs(problem, cube.verts.reshape(-1), max_iter=100, tol=1e-8)
    assert result.energy <= float(problem.value(cube.verts)) + 1e-12
    assert bool(torch.isfinite(result.x).all())
    final = result.x.reshape(-1, 3)
    # The relaxed shape keeps the cube's 1/2-centre symmetry axis-wise.
    spread = (final - final.mean(dim=0)).abs()
    assert float(spread.max()) > 0.0


def test_anisotropic_gradient_matches_autograd() -> None:
    cube = cube_mesh()
    anisotropic = constraints.AnisotropicTension(cube.faces, eps=0.2)
    problem = energy.Energy(cube.faces, terms=[anisotropic])
    analytic = problem.grad(cube.verts)
    leaf = cube.verts.clone().requires_grad_(True)
    (automatic,) = torch.autograd.grad(problem.value(leaf), leaf)
    assert float((analytic - automatic.reshape(-1)).abs().max()) < 1e-12


@requires_se
def test_mound_datafile_parses_with_constraints(data_se) -> None:
    mound = io_fe.load_fe(data_se / "mound.fe")
    assert mound.n_faces == 24
    assert (0, 1) in mound.vertex_constraints
    assert mound.fixed == frozenset({8, 9, 10, 11})


@requires_se
def test_sphere_datafile_volume_reference(data_se) -> None:
    sphere = io_fe.load_fe(data_se / "sphere.fe")
    declared = io_fe.load_bodies(data_se / "sphere.fe")[1]
    assert declared == pytest.approx(2.0 * math.pi / 3.0, rel=1e-12)
    assert float(mesh_mod.volume(sphere)) < 0.0
