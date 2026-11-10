"""Modern constraint handling: volume, contact angle, periodicity, anisotropy.

The four pieces the plan calls for:

* `VolumeConstraint` — augmented Lagrangian (`E + lam*c + rho/2*c^2`, `c = V - V0`)
  with SE-style multiplier and penalty updates, plus a `project` fallback for
  cases where the outer loop stalls.
* `ContactAngle` — line-integral wetting energy `-gamma*cos(theta)*A_wet` on a
  planar wall (`z = wall_z`), with the analytic in-plane gradient of the wetted
  area; `sign` switches between a droplet and a bubble.
* `PeriodicGraph` — periodic height field on a `[0, L)^2` torus, using the
  minimum-image triangle area so a graph that wraps an edge is measured correctly.
* `AnisotropicTension` — cubic anisotropy `1 + eps*(nx^4 + ny^4 + nz^4)` on the
  unit face normal, giving an energy term suitable for `Energy(terms=...)`.

Nothing here mutates its inputs; every value/grad pair is autograd-differentiable
except where an analytic gradient is documented as exact.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch

from . import energy as energy_mod

Tensor = torch.Tensor


def _flat_to_verts(x: Tensor | Sequence[float]) -> Tensor:
    verts = x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=torch.float64)
    if verts.dtype != torch.float64:
        verts = verts.to(torch.float64)
    return verts.reshape(-1, 3)


class VolumeConstraint:
    """Enclosed-volume constraint `c(x) = V(x) - V0` for a closed triangle mesh.

    `value`/`grad` expose the constraint itself; `penalty(x, lam, rho)` is the
    augmented-Lagrangian contribution `lam*c + rho/2*c^2`.  `augmented(energy, ...)`
    wraps an `Energy` into an objective whose `value`/`grad`/`hvp` include that
    contribution, which is what the outer loop minimizes.
    """

    def __init__(self, faces: Tensor, volume0: float) -> None:
        self.faces = faces if isinstance(faces, Tensor) else torch.as_tensor(faces, dtype=torch.int64)
        self.volume0 = float(volume0)

    def volume(self, x: Tensor | Sequence[float]) -> Tensor:
        """Signed enclosed volume at `x`."""
        return energy_mod.signed_volume(_flat_to_verts(x), self.faces)

    def value(self, x: Tensor | Sequence[float]) -> Tensor:
        """Constraint residual `V(x) - V0`."""
        return self.volume(x) - self.volume0

    def relative(self, x: Tensor | Sequence[float]) -> float:
        """`|c| / V0`, the convergence measure used by the outer loop."""
        return abs(float(self.value(x))) / max(abs(self.volume0), 1e-300)

    def grad(self, x: Tensor | Sequence[float]) -> Tensor:
        """Gradient of the residual, shaped like a flat `(3n,)` vector."""
        verts = _flat_to_verts(x).detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(energy_mod.signed_volume(verts, self.faces), verts)
        return grad.detach().reshape(-1)

    def penalty(self, x: Tensor | Sequence[float], lam: float, rho: float) -> Tensor:
        """Augmented-Lagrangian term `lam*c + rho/2*c^2`."""
        residual = self.value(x)
        return lam * residual + 0.5 * rho * residual * residual

    def project(self, x: Tensor | Sequence[float]) -> Tensor:
        """Radial projection onto `V = V0` about the centroid (baseline only).

        Volume scales with the cube of a uniform scale about the centroid, so a
        single closed-form scaling satisfies the constraint exactly.
        """
        verts = _flat_to_verts(x)
        centroid = verts.mean(dim=0, keepdim=True)
        current = float(energy_mod.signed_volume(verts, self.faces))
        if current == 0.0:
            raise ValueError("cannot project a zero-volume mesh")
        scale = (self.volume0 / current) ** (1.0 / 3.0)
        return (centroid + scale * (verts - centroid)).reshape(-1)

    def project_preserving(
        self,
        x: Tensor | Sequence[float],
        clip: Callable[[Tensor], Tensor],
        *,
        iterations: int = 60,
        tol: float = 1e-14,
    ) -> Tensor:
        """Satisfy `V = V0` by scaling about the centroid while honouring `clip`.

        With pinned degrees of freedom (a wall, a fixed ring) a single radial scaling
        no longer sets the volume exactly, because `clip` restores the pinned
        components afterwards.  The scale factor is therefore found by a bracketed
        bisection on `f(s) = V(clip(centroid + s*(x - centroid))) - V0`, which is
        monotone in `s` for any mesh whose orientation does not change, and is
        evaluated exactly rather than by finite differences.
        """
        verts = _flat_to_verts(x)
        centroid = verts.mean(dim=0, keepdim=True)
        offset = verts - centroid

        def volume_at(scale: float) -> float:
            candidate = clip((centroid + scale * offset).reshape(-1))
            return float(energy_mod.signed_volume(_flat_to_verts(candidate), self.faces))

        low, high = 0.0, 2.0
        while volume_at(high) < self.volume0:
            low, high = high, high * 2.0
            if high > 1e12:
                raise ValueError("volume is unreachable by radial scaling")
        low = max(low, 1e-12)
        while volume_at(low) > self.volume0:
            high, low = low, low * 0.5
            if low < 1e-12:
                raise ValueError("volume is unreachable by radial scaling")

        for _ in range(iterations):
            middle = 0.5 * (low + high)
            difference = volume_at(middle) - self.volume0
            if abs(difference) <= tol * max(abs(self.volume0), 1.0):
                break
            if difference < 0.0:
                low = middle
            else:
                high = middle
        return clip((centroid + 0.5 * (low + high) * offset).reshape(-1))

    def augmented(
        self,
        base: energy_mod.Energy,
        *,
        lam: float = 0.0,
        rho: float = 10.0,
    ) -> "AugmentedEnergy":
        """Bundle `base` with this constraint at the given multiplier and penalty."""
        return AugmentedEnergy(base, self, lam=lam, rho=rho)


class AugmentedEnergy:
    """`base` energy plus a `VolumeConstraint` term at fixed `(lam, rho)`.

    Provides the `Energy` protocol (`value`/`grad`/`hvp`) so `optimize.lbfgs` and
    `optimize.tr_newton_cg` accept it directly.
    """

    def __init__(
        self,
        base: energy_mod.Energy,
        constraint: VolumeConstraint,
        *,
        lam: float,
        rho: float,
        reject_degenerate: bool = True,
        free_dofs: Tensor | None = None,
        clip: Callable[[Tensor], Tensor] | None = None,
        reference: Tensor | None = None,
    ) -> None:
        self.base = base
        self.constraint = constraint
        self.lam = float(lam)
        self.rho = float(rho)
        self.reject_degenerate = bool(reject_degenerate)
        self._free_dofs = None if free_dofs is None else free_dofs.to(torch.float64)
        self._clip = clip
        self._reference_normals: Tensor | None = None
        if reference is not None:
            reference_verts = _flat_to_verts(reference)
            self._reference_normals = energy_mod.face_normals(
                reference_verts, self.base.faces, normalize=True
            ).detach()

    def clip(self, x: Tensor | Sequence[float]) -> Tensor:
        """Apply the wall/dirichlet clip, if any, to a candidate coordinate vector."""
        flat = _flat_to_verts(x).reshape(-1)
        return flat if self._clip is None else self._clip(flat)

    @property
    def free_dofs(self) -> Tensor:
        """Flat multiplier that zeroes the pinned coordinate components."""
        if self._free_dofs is None:
            return torch.ones(self.base.n_verts * 3, dtype=torch.float64)
        return self._free_dofs

    def is_degenerate(self, verts: Tensor) -> bool:
        """True when a triangle has collapsed, where the area gradient is undefined.

        Area minimization has no barrier against collapsing a triangulation: a
        fully collapsed mesh has zero area and a well-defined penalty value, so an
        unguarded line search can step into it and then be unable to leave.  Such
        iterates are treated as infeasible, exactly as a line search treats a
        non-finite objective.
        """
        areas = energy_mod.triangle_areas(verts, self.base.faces)
        mean = float(areas.mean())
        if not float(areas.min()) > 1e-6 * mean:
            return True
        if self._reference_normals is not None:
            # A valid triangulation of a cap keeps every face pointing the same way it
            # started: a face whose normal reverses has passed through the surface, which
            # means the mesh self-intersects and its area is no longer meaningful.
            normals = energy_mod.face_normals(verts, self.base.faces, normalize=True)
            if bool((torch.sum(normals * self._reference_normals, dim=1) <= 0.0).any()):
                return True
        return False

    def energy(self, verts: Tensor) -> Tensor:
        """Full augmented objective at `(n, 3)` coordinates."""
        if self.reject_degenerate and self.is_degenerate(verts):
            return torch.tensor(float("inf"), dtype=torch.float64)
        return self.base.value(verts) + self.constraint.penalty(verts.reshape(-1), self.lam, self.rho)

    def value(self, x: Tensor | Sequence[float]) -> Tensor:
        """Augmented objective; `inf` marks a degenerate (collapsed) candidate."""
        clipped = self.clip(x)
        if self.reject_degenerate and self.is_degenerate(clipped.reshape(-1, 3)):
            return torch.tensor(float("inf"), dtype=torch.float64)
        return self.base.value(clipped) + self.constraint.penalty(clipped, self.lam, self.rho)

    def grad(self, x: Tensor | Sequence[float]) -> Tensor:
        """Gradient of the augmented objective, shaped like a flat vector."""
        flat = self.clip(x).detach()
        if self.reject_degenerate and self.is_degenerate(flat.reshape(-1, 3)):
            return torch.zeros_like(flat)
        leaf = flat.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(self.value(leaf), leaf)
        return grad.detach() * self.free_dofs

    def hvp(self, x: Tensor | Sequence[float], v: Tensor | Sequence[float]) -> Tensor:
        flat = self.clip(x).detach()
        direction = (v if isinstance(v, Tensor) else torch.as_tensor(v, dtype=torch.float64)).reshape(-1).detach()
        direction = direction * self.free_dofs
        leaf = flat.clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(self.value(leaf), leaf, create_graph=True)
        (product,) = torch.autograd.grad(grad, leaf, grad_outputs=direction)
        return product.detach() * self.free_dofs


def augmented_lagrangian(
    base: energy_mod.Energy,
    constraint: VolumeConstraint,
    x0,
    *,
    outer: int = 8,
    rho0: float = 10.0,
    rho_max: float = 1e6,
    rho_growth: float = 2.0,
    inner_iter: int = 100,
    volume_tol: float = 1e-8,
    grad_tol: float = 1e-6,
    solve: Callable | None = None,
    free_dofs: Tensor | None = None,
    clip: Callable[[Tensor], Tensor] | None = None,
    reference: Tensor | None = None,
) -> dict:
    """Minimize `base` subject to `V = V0` by the augmented Lagrangian method.

    Each outer iteration minimizes the augmented objective with `solve` (an
    optimizer with the `optimize.lbfgs` signature), then updates
    `lam += rho*c` and `rho = min(rho_growth*rho, rho_max)`.  Stops when
    `|c|/V0 < volume_tol` and the final gradient norm is below `grad_tol`.
    """
    from . import optimize

    solver = solve or optimize.lbfgs
    lam = 0.0
    rho = float(rho0)
    x = _flat_to_verts(x0).reshape(-1).clone()
    if clip is not None:
        x = clip(x)
    history: list[dict] = []

    for step in range(outer):
        problem = AugmentedEnergy(
            base, constraint, lam=lam, rho=rho, free_dofs=free_dofs, clip=clip, reference=reference
        )
        result = solver(problem, x, max_iter=inner_iter, tol=min(grad_tol, 1e-7))
        x = result.x
        residual = float(constraint.value(x))
        relative = constraint.relative(x)
        history.append(
            {
                "outer": step + 1,
                "energy": float(base.value(x)),
                "volume": float(constraint.volume(x)),
                "relative_residual": relative,
                "lam": lam,
                "rho": rho,
                "inner_iterations": result.iterations,
                "inner_grad_norm": result.grad_norm,
            }
        )
        if relative < volume_tol:
            break
        lam += rho * residual
        rho = min(rho * rho_growth, rho_max)

    final_problem = AugmentedEnergy(
        base, constraint, lam=lam, rho=rho, free_dofs=free_dofs, clip=clip, reference=reference
    )
    return {
        "x": x,
        "lam": lam,
        "rho": rho,
        "volume": float(constraint.volume(x)),
        "volume0": constraint.volume0,
        "relative_residual": constraint.relative(x),
        "energy": float(base.value(x)),
        "area": float(energy_mod.surface_area(_flat_to_verts(x), base.faces, base.tension)),
        "grad_norm": float(final_problem.grad(x).abs().max()),
        "converged": bool(constraint.relative(x) < volume_tol),
        "history": history,
    }


class ContactAngle:
    """Line-integral wetting energy for a contact line on the plane `z = wall_z`.

    The wetted area is the polygon enclosed by the contact line's projection onto
    the wall, `A_wet = |sum_i (x_i*y_{i+1} - x_{i+1}*y_i)| / 2`, and the energy is
    `-gamma*cos(theta)*A_wet` for a droplet (`sign = -1`) or the negative of that
    for a bubble.  The in-plane gradient is analytic and exact:

        dA/dx_i = (y_{i+1} - y_{i-1}) / 2,   dA/dy_i = (x_{i-1} - x_{i+1}) / 2

    with the sign of the enclosed area carried through so a reversed loop behaves
    consistently.  Contact vertices are expected to sit on the wall; their normal
    component of the gradient is zero, and callers keep them at `z = wall_z`.
    """

    def __init__(
        self,
        boundary_edges: Sequence[Sequence[int]],
        wall_z: float,
        gamma: float,
        theta_deg: float,
        *,
        n_verts: int | None = None,
        sign: int = -1,
    ) -> None:
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 (droplet) or +1 (bubble)")
        self.loops = [list(loop) for loop in boundary_edges]
        if not self.loops or any(len(loop) < 3 for loop in self.loops):
            raise ValueError("each contact loop needs at least three vertices")
        self.wall_z = float(wall_z)
        self.gamma = float(gamma)
        self.theta_deg = float(theta_deg)
        self.sign = int(sign)
        self.n_verts = int(n_verts) if n_verts is not None else max(max(loop) for loop in self.loops) + 1

    @property
    def coefficient(self) -> float:
        """Prefactor `sign * gamma * cos(theta)` multiplying the wetted area."""
        return self.sign * self.gamma * math.cos(math.radians(self.theta_deg))

    def wetted_area(self, x: Tensor | Sequence[float]) -> Tensor:
        """Total wetted area enclosed by the contact loops."""
        verts = _flat_to_verts(x)
        total = torch.zeros((), dtype=torch.float64)
        for loop in self.loops:
            points = verts[torch.tensor(loop, dtype=torch.int64)]
            xs, ys = points[:, 0], points[:, 1]
            total = total + 0.5 * torch.abs(torch.sum(xs * torch.roll(ys, -1) - torch.roll(xs, -1) * ys))
        return total

    def value(self, x: Tensor | Sequence[float]) -> Tensor:
        """Wetting energy `sign * gamma * cos(theta) * A_wet`."""
        return self.coefficient * self.wetted_area(x)

    def grad(self, x: Tensor | Sequence[float]) -> Tensor:
        """Analytic wetting-energy gradient, shaped like a flat `(3n,)` vector."""
        verts = _flat_to_verts(x)
        grad = torch.zeros((self.n_verts, 3), dtype=torch.float64)
        for loop in self.loops:
            index = torch.tensor(loop, dtype=torch.int64)
            points = verts[index]
            xs, ys = points[:, 0], points[:, 1]
            signed_area = 0.5 * torch.sum(xs * torch.roll(ys, -1) - torch.roll(xs, -1) * ys)
            if float(signed_area) == 0.0:
                continue
            orientation = 1.0 if float(signed_area) > 0.0 else -1.0
            d_area_dx = 0.5 * (torch.roll(ys, -1) - torch.roll(ys, 1))
            d_area_dy = 0.5 * (torch.roll(xs, 1) - torch.roll(xs, -1))
            contribution = self.coefficient * orientation
            grad[index, 0] += contribution * d_area_dx
            grad[index, 1] += contribution * d_area_dy
        return grad.reshape(-1)

    def contact_radius(self, x: Tensor | Sequence[float], *, axis: tuple[int, int] = (0, 1)) -> float:
        """Mean radius of the contact loop projected onto the wall plane.

        Each loop is centred on its own projected centroid, so an off-centre drop
        measures its own radius rather than the distance from the origin.
        """
        verts = _flat_to_verts(x)
        radii: list[float] = []
        for loop in self.loops:
            points = verts[torch.tensor(loop, dtype=torch.int64)][:, list(axis)]
            centroid = points.mean(dim=0, keepdim=True)
            radii.append(float((points - centroid).norm(dim=1).mean()))
        return sum(radii) / len(radii) if radii else float("nan")


def spherical_cap_angle(contact_radius: float, apex_height: float) -> float:
    """Contact angle, in degrees, of a spherical cap resting on a flat wall.

    A cap with contact radius `r` and apex height `h` lies on a sphere of radius
    `R = (r^2 + h^2) / (2h)`, and its contact angle satisfies
    `cos(theta) = (r^2 - h^2) / (r^2 + h^2)`.  Rearranged,

        theta / 2 = atan2(h, r),

    which is evaluated here because it stays well conditioned for very flat caps as
    well as for caps past the hemisphere (`theta > 90`).
    """
    if contact_radius <= 0.0 or apex_height <= 0.0:
        raise ValueError("contact_radius and apex_height must both be positive")
    return math.degrees(2.0 * math.atan2(apex_height, contact_radius))


def apex_height(x: Tensor | Sequence[float], loops: Sequence[Sequence[int]], wall_z: float) -> float:
    """Maximum height above the wall of the surface vertices, measured from `wall_z`."""
    verts = _flat_to_verts(x)
    return float(verts[:, 2].max()) - float(wall_z)


class PeriodicGraph:
    """Periodic height field `z(x, y)` on the torus `[0, Lx) x [0, Ly)`.

    Triangles whose corner displacement exceeds half a period are measured with the
    minimum-image convention, so a graph that wraps an edge keeps its true area.
    `area` and `grad` accept the flat height vector `(n,)`.
    """

    def __init__(self, x: Tensor, y: Tensor, faces: Tensor, box: Sequence[float]) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
        self.y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same number of nodes")
        self.faces = torch.as_tensor(faces, dtype=torch.int64).reshape(-1, 3)
        self.box = tuple(float(value) for value in box)
        if len(self.box) != 2:
            raise ValueError("box must be (Lx, Ly)")

    @classmethod
    def from_grid(cls, n: int, box: Sequence[float], heights: Tensor | None = None) -> "PeriodicGraph":
        """Uniform `n x n` triangulated grid on `[0, Lx) x [0, Ly)`."""
        coords = torch.arange(n, dtype=torch.float64) / n
        gx, gy = torch.meshgrid(coords, coords, indexing="ij")
        x = (gx.reshape(-1) * box[0]).clone()
        y = (gy.reshape(-1) * box[1]).clone()
        faces: list[list[int]] = []
        for i in range(n):
            for j in range(n):
                a = i * n + j
                b = ((i + 1) % n) * n + j
                c = i * n + (j + 1) % n
                d = ((i + 1) % n) * n + (j + 1) % n
                faces.append([a, b, d])
                faces.append([a, d, c])
        graph = cls(x, y, torch.tensor(faces, dtype=torch.int64), box)
        if heights is not None:
            graph.heights = torch.as_tensor(heights, dtype=torch.float64).reshape(-1)
        return graph

    def nodes(self, z: Tensor | Sequence[float]) -> Tensor:
        """Vertex coordinates `(n, 3)` assembled from the flat height vector."""
        heights = z if isinstance(z, Tensor) else torch.as_tensor(z, dtype=torch.float64)
        heights = heights.reshape(-1)
        if heights.shape[0] != self.x.shape[0]:
            raise ValueError(f"expected {self.x.shape[0]} heights, got {heights.shape[0]}")
        return torch.stack([self.x, self.y, heights], dim=1)

    def periods(self) -> Tensor:
        """Lattice matrix for `Energy(periods=...)`, with `z` unperiodized."""
        return torch.tensor(
            [[self.box[0], 0.0, 0.0], [0.0, self.box[1], 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
        )

    def triangle_areas(self, z: Tensor | Sequence[float]) -> Tensor:
        """Per-triangle area with minimum-image corner placements."""
        verts = self.nodes(z)
        corners = verts[self.faces]
        spans = torch.tensor([self.box[0], self.box[1], math.inf], dtype=torch.float64)
        anchor = corners[:, 0:1, :]
        offsets = corners[:, 1:, :] - anchor
        # Wrap in-plane displacements into (-L/2, L/2]; z is never wrapped.
        wrapped = offsets - spans * torch.round(offsets / spans)
        wrapped[:, :, 2] = offsets[:, :, 2]
        placed = torch.cat([anchor, anchor + wrapped], dim=1)
        cross = torch.linalg.cross(placed[:, 1] - placed[:, 0], placed[:, 2] - placed[:, 0])
        return 0.5 * torch.linalg.norm(cross, dim=1)

    def area(self, z: Tensor | Sequence[float]) -> Tensor:
        """Total surface area of the height field."""
        return self.triangle_areas(z).sum()

    def grad(self, z: Tensor | Sequence[float]) -> Tensor:
        """Gradient of `area` with respect to the flat heights."""
        heights = (z if isinstance(z, Tensor) else torch.as_tensor(z, dtype=torch.float64)).reshape(-1)
        leaf = heights.clone().detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(self.area(leaf), leaf)
        return grad.detach()


class AnisotropicTension:
    """Cubic anisotropic surface energy `sum_f gamma(n_f) * area_f`.

    `gamma(n) = 1 + eps * (n_x^4 + n_y^4 + n_z^4)` with `n` the unit face normal, so
    the energy is smooth wherever the mesh is non-degenerate and autograd supplies
    its gradient.  Instances are callables over `(n, 3)` coordinates, suitable for
    `Energy(terms=[...])`.
    """

    def __init__(self, faces: Tensor, eps: float = 0.1) -> None:
        self.faces = torch.as_tensor(faces, dtype=torch.int64).reshape(-1, 3)
        self.eps = float(eps)

    def gamma(self, x: Tensor) -> Tensor:
        """Per-face tension from the unit normal."""
        normals = energy_mod.face_normals(x, self.faces, normalize=True)
        return 1.0 + self.eps * (normals.pow(4).sum(dim=1))

    def __call__(self, verts: Tensor) -> Tensor:
        """Anisotropic surface energy for `(n, 3)` coordinates."""
        areas = energy_mod.triangle_areas(verts, self.faces)
        return (self.gamma(verts) * areas).sum()

    def energy(self, faces: Tensor | None = None) -> energy_mod.Energy:
        """An `Energy` whose terms include this anisotropy."""
        return energy_mod.Energy(self.faces, terms=[self])
