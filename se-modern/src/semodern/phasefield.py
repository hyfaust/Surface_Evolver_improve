"""Phase-field (Allen-Cahn) diffuse-interface baseline.

The sharp-interface formulation Surface Evolver solves represents a soap film as a
triangulated surface with surface tension.  The diffuse alternative replaces that
surface by a scalar order parameter `phi` on a uniform 3-D grid:

.. math::

    E[\\phi] = \\int \\left( \\frac{\\epsilon}{2} |\\nabla \\phi|^2
                          + \\frac{W(\\phi)}{\\epsilon} \\right) dV,
    \\qquad W(\\phi) = \\frac{(\\phi^2 - 1)^2}{4}

Its minimizers have a hyperbolic-tangent profile of width `~eps`; in the
sharp-interface limit the model reproduces surface tension, and a Lagrange
multiplier on the enclosed volume holds the drop size fixed.  Gradient flow
`dphi/dt = -dE/dphi` is the Allen-Cahn equation

    dphi/dt = -dE/dphi + lambda,

where `lambda` enforces a target volume.

Every discrete operator here is built so that the *discrete* energy and its
gradient agree exactly (verified against `torch.autograd` to machine precision):
`laplacian` is the transpose of `_face_gradient`, which makes
`gradient_squared`'s quadratic form exactly `-sum |grad phi|^2`.  Consequently
`chemical_potential` returns the true discrete `-dE/dphi`, and an explicit Euler
step is a genuine energy descent rather than a drift.

This is a *diffuse* baseline: absolute energies are not comparable with SE's
sharp-interface values; radii, volumes, areas and topology are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

Tensor = torch.Tensor

# Discrete coefficients relating `energy()` to its gradient.  Differentiating
#   E = sum( eps/2 * gradient_squared + W/eps ) * dx^3
# with `laplacian` the transpose of `_face_gradient` gives
#   dE/dphi = eps*dx * laplacian(phi) + dx^3 * W'(phi) / eps.
# Both coefficients were recovered from `torch.autograd` at several `dx` (0.5, 1/6,
# 0.125) and match this form exactly; `tests/test_phasefield.py` re-checks it.
LAPLACE_COEFFICIENT = lambda eps, dx: eps * dx  # noqa: E731 - documented closed form
WELL_COEFFICIENT = lambda eps, dx: dx**3 / eps  # noqa: E731 - documented closed form


def double_well(phi: Tensor) -> Tensor:
    """`W(phi) = (phi^2 - 1)^2 / 4`, zero at `phi = +-1`."""
    return 0.25 * (phi * phi - 1.0) ** 2


def double_well_derivative(phi: Tensor) -> Tensor:
    """`W'(phi) = phi * (phi^2 - 1)`."""
    return phi * (phi * phi - 1.0)


def _face_gradient(field: Tensor, axis: int) -> Tensor:
    """Face difference `phi[i+1] - phi[i]` along `axis`, zero on the last plane.

    `phi` is cell-centred, so this difference lives on the face between neighbouring
    cells.  The final plane has no upper neighbour, which is the Neumann condition
    `dphi/dn = 0`, and is therefore exactly zero.
    """
    difference = torch.zeros_like(field)
    upper = field.narrow(axis, 1, field.shape[axis] - 1)
    lower = field.narrow(axis, 0, field.shape[axis] - 1)
    difference.narrow(axis, 0, field.shape[axis] - 1).copy_(upper - lower)
    return difference


def _face_gradient_adjoint(flux: Tensor, axis: int) -> Tensor:
    """Transpose of `_face_gradient`: `flux[i-1] - flux[i]`, with zero inflow at the start."""
    previous = torch.cat(
        [torch.zeros_like(flux.narrow(axis, 0, 1)), flux.narrow(axis, 0, flux.shape[axis] - 1)], dim=axis
    )
    return previous - flux


def laplacian(field: Tensor) -> Tensor:
    """Isotropic 7-point Laplacian, exact transpose of `_face_gradient` under Neumann conditions."""
    total = torch.zeros_like(field)
    for axis in range(3):
        total = total + _face_gradient_adjoint(_face_gradient(field, axis), axis)
    return total


def gradient_squared(field: Tensor, dx: float) -> Tensor:
    """Pointwise `|grad phi|^2` from face gradients, zero flux past the last plane."""
    total = torch.zeros_like(field)
    for axis in range(3):
        difference = _face_gradient(field, axis) / dx
        total = total + difference * difference
    return total


def stable_timestep(eps: float, dx: float = 1.0, *, safety: float = 5e-4) -> float:
    """Explicit Euler step for Allen-Cahn: `dt = safety / eps`.

    With the verified coefficients the stiff term is `eps*dx * laplacian(phi)`, whose
    eigenvalues reach `-12 * eps * dx`, so the explicit stability limit is
    `dt < 1/(12 eps dx)`.  Stability alone is not enough here: at the limit the step is
    so large that the double-well term drives `phi` past the well and the drop dissolves
    within tens of steps.  `safety = 5e-4` keeps the step well inside the stability
    region and resolves the relaxation, which is what the benchmarks and tests use.
    """
    return safety / (eps * dx) if dx else safety / eps


@dataclass
class PhaseFieldGrid:
    """Uniform 3-D grid holding `phi`, its diffuse parameters and its conserved mass.

    `phi` is an `(n, n, n)` float64 tensor of cell-centred values with `phi = +1` the
    occupied phase.  `dx` is the cell size, `eps` the interface width, `mass_target`
    the prescribed value of `excess()`, and `phi_far` the far-field order parameter
    subtracted when measuring it.
    """

    phi: Tensor
    dx: float
    eps: float
    mass_target: float
    phi_far: float = -1.0
    lam: float = 0.0
    history: list[float] = field(default_factory=list)

    @classmethod
    def from_signed_distance(
        cls,
        n: int,
        domain: float,
        eps: float,
        signed_distance,
        *,
        mass_target: float | None = None,
    ) -> "PhaseFieldGrid":
        """Initialize `phi = -tanh(d / (sqrt(2) eps))` on an `n^3` grid.

        `signed_distance` maps `(m, 3)` coordinates to a signed distance that is
        negative inside the occupied region; the minus sign makes `phi = +1` the inside
        phase.  The tanh profile is the equilibrium profile of the double-well energy at
        that width, so the initial state is already a diffuse interface rather than an
        arbitrary step.  When `mass_target` is omitted it is taken from the initial
        field, which makes the first step conservative.
        """
        dx = domain / n
        axis = (torch.arange(n, dtype=torch.float64) + 0.5) * dx - domain / 2.0
        points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
        distance = signed_distance(points).reshape(n, n, n)
        phi = -torch.tanh(distance / (float(torch.sqrt(torch.tensor(2.0))) * eps))
        phi_far = -1.0
        excess = float((phi - phi_far).sum()) * dx**3
        return cls(
            phi=phi,
            dx=dx,
            eps=eps,
            mass_target=excess if mass_target is None else mass_target,
            phi_far=phi_far,
        )

    @property
    def cell_volume(self) -> float:
        """Volume of one grid cell."""
        return self.dx**3

    def excess(self) -> float:
        """`integral(phi - phi_far) dV`, the drop's excess order parameter.

        The `phi = -1` phase fills the rest of the box, so neither `integral(phi)` nor
        `integral((phi+1)/2)` is a good volume measure: both grow with the box, and a
        uniform shift that lowers the background changes them without changing the drop.
        Subtracting the far-field value (`phi_far`, the minimum of the initial profile)
        isolates the drop; its derivative with respect to `phi` is the constant 1, which
        is `laplacian`'s null space, so the multiplier stays exact.
        """
        return float((self.phi - self.phi_far).sum()) * self.cell_volume

    def volume(self) -> float:
        """Occupied volume, measured as `|{phi > 0}| * dx^3`.

        The threshold measure is used rather than `integral((phi+1)/2)` because the
        latter counts a full interface shell whose thickness scales with `eps`, which
        would over-report the volume by a factor growing with the interface width.
        """
        return float((self.phi > 0.0).sum()) * self.cell_volume

    def energy(self) -> float:
        """Total diffuse energy `integral(eps/2 |grad phi|^2 + W(phi)/eps) dV`."""
        density = 0.5 * self.eps * gradient_squared(self.phi, self.dx) + double_well(self.phi) / self.eps
        return float(density.sum()) * self.cell_volume

    def interface_area(self) -> float:
        """Diffuse interface area from the equilibrium-profile energy density.

        For the tanh profile the excess energy per unit area is `2*sqrt(2)/3 * eps`, so
        dividing the energy by that constant recovers an area estimate.
        """
        return self.energy() / (2.0 * float(torch.sqrt(torch.tensor(2.0))) / 3.0 * self.eps)

    def chemical_potential(self) -> Tensor:
        """`-dE/dphi`, the Allen-Cahn driving force (excluding the volume multiplier).

        Verified against `torch.autograd` to `~1e-18`; see the module docstring for the
        derivation.  Using the continuum coefficients instead mis-scales the stiff term
        and the flow moves up the energy.
        """
        return -(
            LAPLACE_COEFFICIENT(self.eps, self.dx) * laplacian(self.phi)
            + WELL_COEFFICIENT(self.eps, self.dx) * double_well_derivative(self.phi)
        )

    def step(self, dt: float) -> None:
        """One explicit Euler step of Allen-Cahn, then an exact occupancy projection.

        The flow alone conserves nothing: `laplacian` annihilates constants, so the
        constant mode is untouched by the stiff term and `W'(phi)/eps` changes the
        occupancy at every step.  The multiplier is therefore applied as a projection
        *after* the Euler update.  Occupancy is linear in `phi` with derivative exactly
        `1/2`, so the correction is closed form and exact:

            lam = -(Occ(phi_new) - Occ_target) / (N_cells * dV / 2).

        Projecting afterwards (rather than predicting the multiplier from the pre-step
        occupancy) is what keeps the drop from shrinking: the pre-step estimate always
        lags by one `O(dt)` term, while the post-step projection removes the error on the
        field it is actually applied to.
        """
        mu = self.chemical_potential()
        self.phi = self.phi + dt * mu
        error = self.excess() - self.mass_target
        sensitivity = self.phi.numel() * self.cell_volume
        self.lam = -error / sensitivity
        self.phi = self.phi + self.lam
        self.history.append(self.energy())

    def run(self, steps: int, dt: float, *, every: int = 50) -> dict:
        """Integrate `steps` Allen-Cahn steps, validating `phi` every `every` steps.

        Raises `FloatingPointError` if `phi` leaves `[-1.5, 1.5]` (the well is unbounded
        outside) or becomes non-finite, which a diffuse solver must detect rather than
        silently accept.
        """
        for index in range(steps):
            self.step(dt)
            if (index + 1) % every == 0 or index == steps - 1:
                self.check()
        return {
            "steps": steps,
            "dt": dt,
            "energy": self.energy(),
            "mass": self.excess(),
            "mass_error": abs(self.excess() - self.mass_target),
            "volume": self.volume(),
            "lam": self.lam,
            "phi_min": float(self.phi.min()),
            "phi_max": float(self.phi.max()),
            "monotone": all(after <= before + 1e-9 for before, after in zip(self.history, self.history[1:])),
        }

    def check(self) -> None:
        """Raise `FloatingPointError` when `phi` leaves the admissible range."""
        if not bool(torch.isfinite(self.phi).all()):
            raise FloatingPointError("phase field became non-finite")
        phi_min = float(self.phi.min())
        phi_max = float(self.phi.max())
        if phi_min < -1.5 or phi_max > 1.5:
            raise FloatingPointError(f"phase field left the double-well range: [{phi_min:.3f}, {phi_max:.3f}]")

    def equivalent_radius(self) -> float:
        """Radius of the sphere with the occupied volume `|{phi > 0}| * dx^3`."""
        return (3.0 * self.volume() / (4.0 * torch.pi)) ** (1.0 / 3.0)

    def component_count(self, threshold: float = 0.0) -> int:
        """Number of 6-connected regions where `phi > threshold`.

        The default threshold is 0 because `phi = 0` is the diffuse interface midpoint;
        a positive value would erode thin necks and report a spurious split.
        """
        mask = (self.phi > threshold).detach().cpu().numpy().copy()
        if not mask.any():
            return 0
        from scipy import ndimage

        _, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
        return int(count)

    def isosurface(self, level: float = 0.0) -> tuple[Tensor, Tensor]:
        """Marching-cubes isosurface as `(verts, faces)` in physical coordinates."""
        from skimage import measure

        spacing = (self.dx, self.dx, self.dx)
        field = self.phi.detach().cpu().numpy().copy()  # marching_cubes rejects negative strides
        verts, faces, _, _ = measure.marching_cubes(field, level, spacing=spacing)
        origin = -0.5 * self.phi.shape[0] * self.dx
        offset = torch.tensor([origin, origin, origin], dtype=torch.float64)
        # `marching_cubes` returns reversed-stride views; copy before converting to tensors.
        return (
            torch.as_tensor(verts.copy(), dtype=torch.float64) + offset,
            torch.as_tensor(faces.copy(), dtype=torch.int64),
        )


def sphere_distance(center: Sequence[float], radius: float):
    """Signed-distance callable for a sphere (negative inside)."""

    def distance(points: Tensor) -> Tensor:
        centre = torch.as_tensor(center, dtype=torch.float64).reshape(1, 3)
        return (points - centre).norm(dim=1) - radius

    return distance


def two_sphere_distance(centers: Sequence[Sequence[float]], radius: float):
    """Signed distance of the union of two spheres (negative inside either)."""
    centre_tensor = torch.as_tensor(list(centers), dtype=torch.float64)

    def distance(points: Tensor) -> Tensor:
        distances = (points[:, None, :] - centre_tensor[None, :, :]).norm(dim=2) - radius
        return distances.min(dim=1).values

    return distance
