"""Differentiable surface energy terms and the `Energy` interface.

Every function accepts vertices either as `(n, 3)` or flattened `(3n,)`, is
differentiable with respect to them, and never modifies its input in place.
`Energy` is the single entry point used by the optimizers: `value`, `grad` and
`hvp` (Hessian-vector product) all run through `torch.autograd`, so a Hessian is
never assembled explicitly.

Periodic meshes (`periods` given, i.e. SE's `PERIODS`/`TORUS*`) use an automatic
minimum-image convention: each triangle keeps its first vertex as anchor and each
other corner is translated by the lattice vector bringing it closest to that
anchor (see `minimum_image`).  The choice is piecewise constant in the vertex
coordinates, so gradients stay locally exact.  SE instead stores an explicit wrap
flag per edge, so on very coarse periodic start files the two conventions can
differ; `benchmarks` therefore compares absolute areas only for non-periodic
cases and treats `phelanc` as a parse/topology case.
"""

from __future__ import annotations

import itertools
from typing import Callable, Sequence

import torch

Tensor = torch.Tensor

_SHIFT_GRID = torch.tensor(list(itertools.product((-1, 0, 1), repeat=3)), dtype=torch.float64)


def _vertices(x: Tensor | Sequence[float], faces: Tensor) -> Tensor:
    """Return `x` as an `(n, 3)` float64 tensor, without copying when possible."""
    verts = x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=torch.float64)
    if verts.dtype != torch.float64:
        verts = verts.to(torch.float64)
    if verts.dim() == 1:
        verts = verts.reshape(-1, 3)
    return verts


def _faces(faces: Tensor | Sequence) -> Tensor:
    table = faces if isinstance(faces, Tensor) else torch.as_tensor(faces, dtype=torch.int64)
    return table.reshape(-1, 3)


def _period_matrix(periods: Tensor | Sequence | None) -> Tensor | None:
    if periods is None:
        return None
    matrix = periods if isinstance(periods, Tensor) else torch.as_tensor(periods, dtype=torch.float64)
    return matrix.reshape(3, 3).to(torch.float64)


def _shifts(periods: Tensor) -> Tensor:
    return torch.einsum("kc,cd->kd", _SHIFT_GRID.to(periods.dtype), periods)


def minimum_image(verts: Tensor | Sequence[float], faces: Tensor, periods: Tensor | Sequence) -> Tensor:
    """Place every triangle's corners in one image cell, nearest to its anchor.

    Triangles are taken in index order and each corner is assigned the image
    nearest the triangle's first vertex, so a vertex shared by several triangles
    ends up with the placement of the lowest-indexed triangle that uses it.
    """
    vertex_table = _vertices(verts, faces)
    triangles = _faces(faces)
    matrix = _period_matrix(periods)
    if matrix is None or float(matrix.abs().sum()) == 0.0:
        return vertex_table

    shifts = _shifts(matrix)
    anchor = vertex_table[triangles[:, 0]]
    shifted = vertex_table.clone()
    rows = torch.arange(triangles.shape[0])
    for corner in (1, 2):
        candidates = vertex_table[triangles[:, corner]][:, None, :] - shifts[None, :, :]
        best = (candidates - anchor[:, None, :]).pow(2).sum(dim=-1).argmin(dim=1)
        shifted[triangles[:, corner]] = candidates[rows, best]
    return shifted


def _triangle_corners(x: Tensor, faces: Tensor, periods: Tensor | Sequence | None) -> Tensor:
    """`(m, 3, 3)` corner coordinates, in minimum image when `periods` is given."""
    vertex_table = _vertices(x, faces)
    table = _faces(faces)
    if periods is None or float(_period_matrix(periods).abs().sum()) == 0.0:
        return vertex_table[table]
    return minimum_image(vertex_table, table, periods)[table]


def triangle_areas(x: Tensor, faces: Tensor, *, periods: Tensor | Sequence | None = None) -> Tensor:
    """Area of every triangle, shape `(m,)`."""
    tri = _triangle_corners(x, faces, periods)
    cross = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * torch.linalg.norm(cross, dim=1)


def surface_area(
    x: Tensor,
    faces: Tensor,
    tension: Tensor | None = None,
    *,
    periods: Tensor | Sequence | None = None,
) -> Tensor:
    """Sum of triangle areas, optionally weighted by a per-face tension."""
    areas = triangle_areas(x, faces, periods=periods)
    if tension is None:
        return areas.sum()
    weights = tension if isinstance(tension, Tensor) else torch.as_tensor(tension, dtype=areas.dtype)
    weights = weights.reshape(-1)
    if weights.shape[0] != areas.shape[0]:
        raise ValueError(f"tension has {weights.shape[0]} entries for {areas.shape[0]} faces")
    return (weights * areas).sum()


def signed_volume(x: Tensor, faces: Tensor) -> Tensor:
    """Signed volume via the divergence theorem, `sum(dot(v0, cross(v1, v2))) / 6`."""
    tri = _triangle_corners(x, faces, None)
    return torch.einsum("ij,ij->", tri[:, 0], torch.linalg.cross(tri[:, 1], tri[:, 2])) / 6.0


def face_normals(x: Tensor, faces: Tensor, *, normalize: bool = True) -> Tensor:
    """Face normals by the right-hand rule, shape `(m, 3)`; unit length if `normalize`."""
    tri = _triangle_corners(x, faces, None)
    normals = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if normalize:
        normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-300)
    return normals


def gravity(x: Tensor, faces: Tensor, rho: float, g_vec: Sequence[float]) -> Tensor:
    """Gravitational energy in SE's divergence form: `rho * g_z * int_V z dV`.

    The volume integral is `(1/2) * sum_f n_z * int_f z^2 dA`.  With
    `cross_z = 2 * area * n_z` and `z_squared_mean` the exact mean of `z^2` over
    the triangle, that reduces to `0.25 * sum_f cross_z * z_squared_mean`, which
    is the form evaluated below.
    """
    tri = _triangle_corners(x, faces, None)
    cross = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    z0, z1, z2 = tri[:, 0, 2], tri[:, 1, 2], tri[:, 2, 2]
    z_squared_mean = (z0 * z0 + z1 * z1 + z2 * z2 + z0 * z1 + z1 * z2 + z2 * z0) / 6.0
    g = torch.as_tensor(g_vec, dtype=tri.dtype).reshape(3)
    return 0.25 * float(rho) * g[2] * torch.sum(z_squared_mean * cross[:, 2])


class Energy:
    """Scalar energy over vertex positions with autograd `value`/`grad`/`hvp`.

    Parameters
    ----------
    faces : (m, 3) int64 triangle table.
    n_verts : number of vertices; defaults to `faces.max() + 1`.
    tension : optional per-face surface tension (default 1 for every face).
    terms : extra energy contributions, each a callable `verts (n, 3) -> scalar`.
    rho, g_vec : optional gravity term (see `gravity`).
    periods : optional `(3, 3)` lattice enabling minimum-image triangle areas.
    fixed : vertex indices held fixed; their gradient and HVP rows are zero.

    `x` may be flat `(3n,)` or `(n, 3)`.  `grad(x)` mirrors the shape of `x`, and
    `hvp(x, v)` mirrors the shape of `v`, which must match `x`.
    """

    def __init__(
        self,
        faces: Tensor | Sequence,
        *,
        n_verts: int | None = None,
        tension: Tensor | Sequence[float] | None = None,
        terms: Sequence[Callable[[Tensor], Tensor]] | None = None,
        rho: float = 0.0,
        g_vec: Sequence[float] = (0.0, 0.0, 0.0),
        periods: Tensor | Sequence | None = None,
        fixed: Sequence[int] | Tensor | None = None,
    ) -> None:
        self.faces = _faces(faces)
        if self.faces.numel() == 0:
            raise ValueError("Energy requires at least one face")
        self.n_verts = int(n_verts) if n_verts is not None else int(self.faces.max()) + 1
        self.tension = None if tension is None else torch.as_tensor(tension, dtype=torch.float64)
        self.terms = tuple(terms or ())
        self.rho = float(rho)
        self.g_vec = tuple(float(value) for value in g_vec)
        self.periods = _period_matrix(periods)

        mask = torch.zeros(self.n_verts, dtype=torch.bool)
        if fixed is not None:
            raw = sorted(int(index) for index in fixed) if isinstance(fixed, (set, frozenset)) else list(fixed)
            indices = torch.as_tensor(raw, dtype=torch.int64).reshape(-1)
            if indices.numel():
                if int(indices.min()) < 0 or int(indices.max()) >= self.n_verts:
                    raise ValueError(f"fixed vertex index outside 0..{self.n_verts - 1}")
                mask[indices] = True
        self.fixed_mask = mask

    def free_dofs(self) -> Tensor:
        """Flat (3n,) mask of the free coordinate components."""
        return (~self.fixed_mask).unsqueeze(1).expand(self.n_verts, 3).reshape(-1).to(torch.float64)

    def energy(self, x: Tensor) -> Tensor:
        """Energy for `(n, 3)` coordinates, connected to the autograd graph."""
        value = surface_area(x, self.faces, self.tension, periods=self.periods)
        if self.rho:
            value = value + gravity(x, self.faces, self.rho, self.g_vec)
        for term in self.terms:
            value = value + term(x)
        return value

    def value(self, x: Tensor | Sequence[float]) -> Tensor:
        """Scalar energy; differentiable, so callers may use it in autograd."""
        return self.energy(_vertices(x, self.faces))

    def grad(self, x: Tensor | Sequence[float]) -> Tensor:
        """Gradient shaped like `x`, with fixed components set to zero."""
        flat = _vertices(x, self.faces).reshape(-1)
        leaf = flat.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(self.energy(leaf.reshape(-1, 3)), leaf)
        return grad.detach() * self.free_dofs()

    def hvp(self, x: Tensor | Sequence[float], v: Tensor | Sequence[float]) -> Tensor:
        """Hessian-vector product `H(x) v` shaped like `x`, fixed rows zeroed."""
        flat = _vertices(x, self.faces).reshape(-1)
        direction = v if isinstance(v, Tensor) else torch.as_tensor(v, dtype=torch.float64)
        direction = direction.reshape(-1).detach() * self.free_dofs()
        if direction.shape[0] != flat.shape[0]:
            raise ValueError(f"direction has {direction.shape[0]} entries, expected {flat.shape[0]}")
        leaf = flat.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(self.energy(leaf.reshape(-1, 3)), leaf, create_graph=True)
        (product,) = torch.autograd.grad(grad, leaf, grad_outputs=direction)
        return product.detach() * self.free_dofs()
