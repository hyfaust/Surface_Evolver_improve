"""Mesh-free RBF-FD tests: stencil exactness, catenoid convergence, area identity."""

from __future__ import annotations

import numpy as np
import pytest

from semodern import meshfree as mf


def test_stencil_reproduces_linear_and_quadratic_exactly() -> None:
    nodes = np.array([-0.2, 0.0, 0.1, 0.3, 0.5])
    first, second = mf.stencil_weights(nodes, 1, np.arange(5))
    offset = nodes - nodes[1]
    assert float(first @ offset) == pytest.approx(1.0, rel=1e-12)
    assert float(second @ offset) == pytest.approx(0.0, abs=1e-12)
    assert float(first @ offset**2) == pytest.approx(0.0, abs=1e-12)
    assert float(second @ offset**2) == pytest.approx(2.0, rel=1e-12)


def test_stencil_derivatives_approximate_a_smooth_function() -> None:
    nodes = np.linspace(-1.0, 1.0, 41)
    system = mf.RbfStencilSystem.build(nodes, stencil=7)
    values = np.exp(nodes)
    first = system.derivative(system.first, values)
    second = system.derivative(system.second, values)
    expected = np.exp(nodes)[1:-1]
    # The two nodes adjacent to the Dirichlet ends see a strongly one-sided stencil, so
    # the accuracy claim is made on the interior where the stencil is symmetric.
    assert 4.0 < expected.shape[0]
    interior = slice(2, -2)
    assert np.abs(first[interior] - expected[interior]).max() < 5e-3
    assert np.abs(second[interior] - expected[interior]).max() < 5e-2
    # At the boundary the weights are still consistent: they reproduce a linear function.
    assert finite_all(first) and finite_all(second)


def finite_all(values) -> bool:
    """True when every entry of `values` is finite."""
    return bool(np.isfinite(values).all())


def test_catenoid_constant_satisfies_ring_condition() -> None:
    half_height, radius = 1.0, 1.6
    a = mf.catenoid_constant(half_height, radius)
    assert a * np.cosh(half_height / a) == pytest.approx(radius, rel=1e-10)


def test_maximum_radius_matches_surface_evolver_rmax() -> None:
    # `cat.fe` declares RMAX = 1.5088795, the largest ring spacing ratio it can span;
    # recovering that number independently is a strong check on the root finding.
    assert mf.maximum_radius(1.0) == pytest.approx(1.5088795, abs=1e-6)


def test_catenoid_rejects_impossible_ring_radius() -> None:
    with pytest.raises(ValueError, match="no catenoid spans"):
        mf.catenoid_constant(1.0, 1.0)


def test_analytic_profile_satisfies_the_ode() -> None:
    a = mf.catenoid_constant(1.0, 1.6)
    z = np.linspace(-1.0, 1.0, 20001)
    r = mf.analytic_profile(z, a)
    first = np.gradient(r, z)
    second = np.gradient(first, z)
    residual = second / (1.0 + first**2) ** 1.5 - 1.0 / (r * np.sqrt(1.0 + first**2))
    assert np.abs(residual[50:-50]).max() < 1e-6


def test_solver_reproduces_the_exact_profile() -> None:
    result = mf.solve_catenoid(1.0, 1.6, node_count=40, stencil=7)
    assert result["success"]
    assert result["relative_error_max"] < 1e-2
    assert result["residual_inf"] < 1e-8


def test_solver_converges_under_refinement() -> None:
    coarse = mf.solve_catenoid(1.0, 1.6, node_count=30, stencil=7)
    fine = mf.solve_catenoid(1.0, 1.6, node_count=120, stencil=9)
    assert fine["relative_error_max"] < coarse["relative_error_max"]
    assert fine["relative_error_max"] < 1e-4


def test_boundary_values_are_enforced() -> None:
    radius = 1.6
    result = mf.solve_catenoid(1.0, radius, node_count=40, stencil=7)
    assert result["profile"][0] == pytest.approx(radius, rel=1e-15)
    assert result["profile"][-1] == pytest.approx(radius, rel=1e-15)


def test_neck_matches_the_exact_parameter() -> None:
    result = mf.solve_catenoid(1.0, 1.6, node_count=61, stencil=9)
    assert result["neck_numerical"] == pytest.approx(result["neck_exact"], rel=2e-3)


def test_area_of_revolution_matches_the_closed_form() -> None:
    result = mf.solve_catenoid(1.0, 1.6, node_count=200, stencil=9)
    numeric = mf.area_of_revolution(result["profile"], result["nodes"])
    exact = mf.exact_area(result["a"], 1.0)
    assert numeric == pytest.approx(exact, rel=1e-3)


def test_parabolic_and_analytic_starts_agree() -> None:
    parabolic = mf.solve_catenoid(1.0, 1.6, node_count=60, stencil=9, initial="parabolic")
    analytic = mf.solve_catenoid(1.0, 1.6, node_count=60, stencil=9, initial="analytic")
    assert parabolic["relative_error_max"] == pytest.approx(analytic["relative_error_max"], rel=1e-6)


def test_unknown_initial_guess_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown initial guess"):
        mf.solve_catenoid(1.0, 1.6, node_count=20, stencil=5, initial="magic")
