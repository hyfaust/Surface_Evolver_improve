"""Triangular surface mesh, edge topology and geometric invariants.

`Mesh` stores only vertices, triangles and an optional period lattice; every
derived quantity (edge table, area, volume, Euler characteristic) is recomputed
on demand, so no caller can observe a stale cache.  Vertices are float64
`(n, 3)`, faces int64 `(m, 3)`, and `periods` is an optional `(3, 3)` matrix
whose rows are the lattice translation vectors (SE's `PERIODS`).

Invariants enforced by `validate`:
  * face indices within range, no repeated vertex inside a face,
  * no duplicate face (compared as an unordered vertex set),
  * every triangle has strictly positive area,
  * every edge is incident to at most two faces,
  * an edge with two faces is traversed in opposite directions by them
    (consistent orientation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import energy as _energy


@dataclass
class Mesh:
    """Triangle surface mesh with an optional period lattice."""

    verts: torch.Tensor
    faces: torch.Tensor
    periods: torch.Tensor | None = None
    fixed: frozenset[int] = frozenset()
    vertex_constraints: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        verts = torch.as_tensor(self.verts, dtype=torch.float64)
        faces = torch.as_tensor(self.faces, dtype=torch.int64)
        self.verts = verts.reshape(-1, 3)
        self.faces = faces.reshape(-1, 3)
        if self.periods is not None:
            periods = torch.as_tensor(self.periods, dtype=torch.float64)
            self.periods = periods.reshape(3, 3)
        self.fixed = frozenset(int(index) for index in self.fixed)
        self.vertex_constraints = tuple((int(v), int(c)) for v, c in self.vertex_constraints)

    @property
    def n_verts(self) -> int:
        return int(self.verts.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def fixed_mask(self) -> torch.Tensor:
        """Boolean mask over vertices, true where the datafile declares `fixed`."""
        mask = torch.zeros(self.n_verts, dtype=torch.bool)
        if self.fixed:
            mask[torch.tensor(sorted(self.fixed), dtype=torch.int64)] = True
        return mask

    def with_periods(self, periods: torch.Tensor) -> "Mesh":
        """Copy of this mesh with the given period lattice, coordinates wrapped once."""
        wrapped = _energy.minimum_image(self.verts, self.faces, periods)
        return Mesh(
            verts=wrapped,
            faces=self.faces,
            periods=periods,
            fixed=self.fixed,
            vertex_constraints=self.vertex_constraints,
        )

    def detached(self) -> "Mesh":
        """Copy that shares no autograd history (needed before in-place updates)."""
        return Mesh(
            verts=self.verts.detach(),
            faces=self.faces,
            periods=None if self.periods is None else self.periods.detach(),
            fixed=self.fixed,
            vertex_constraints=self.vertex_constraints,
        )


def edge_table(mesh: Mesh) -> tuple[torch.Tensor, list[list[tuple[int, int]]]]:
    """Canonical undirected edges with their incident `(face, sign)` pairs.

    `edges` is `(e, 2)` int64 with `edges[:, 0] < edges[:, 1]`, and the incidence
    list is aligned with it.  `sign` is `+1` when a face traverses the edge from
    the lower to the higher index, `-1` otherwise, so a consistently oriented
    interior edge has one `+1` and one `-1`.
    """
    index: dict[tuple[int, int], int] = {}
    keys: list[tuple[int, int]] = []
    incident: list[list[tuple[int, int]]] = []
    for face, loop in enumerate(mesh.faces.tolist()):
        for u, v in zip(loop, loop[1:] + loop[:1]):
            key = (u, v) if u < v else (v, u)
            slot = index.get(key)
            if slot is None:
                slot = len(keys)
                index[key] = slot
                keys.append(key)
                incident.append([])
            incident[slot].append((face, 1 if u < v else -1))
    edges = torch.tensor(keys, dtype=torch.int64).reshape(-1, 2)
    return edges, incident


def edge_face_array(mesh: Mesh) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edge table as flat arrays: `(edges, face_a, sign_a, face_b)`.

    `face_b` is `-1` for boundary edges, which is convenient for vectorized
    topology checks (each edge has at most two incident faces, guaranteed by
    `validate`).
    """
    edges, incident = edge_table(mesh)
    count = edges.shape[0]
    face_a = torch.full((count,), -1, dtype=torch.int64)
    sign_a = torch.zeros((count,), dtype=torch.int64)
    face_b = torch.full((count,), -1, dtype=torch.int64)
    for slot, inc in enumerate(incident):
        face_a[slot], sign_a[slot] = inc[0]
        if len(inc) == 2:
            face_b[slot] = inc[1][0]
    return edges, face_a, sign_a, face_b


def validate(mesh: Mesh) -> None:
    """Raise `ValueError` if any mesh invariant above is violated."""
    n_verts = mesh.n_verts
    if mesh.n_faces == 0:
        raise ValueError("mesh has no faces")
    if mesh.faces.numel() and (int(mesh.faces.min()) < 0 or int(mesh.faces.max()) >= n_verts):
        raise ValueError(
            f"face index out of range 0..{n_verts - 1}: "
            f"min={int(mesh.faces.min())} max={int(mesh.faces.max())}"
        )

    seen: set[tuple[int, int, int]] = set()
    for face, loop in enumerate(mesh.faces.tolist()):
        if len(set(loop)) != 3:
            raise ValueError(f"face {face} repeats a vertex: {tuple(loop)}")
        key = tuple(sorted(loop))
        if key in seen:
            raise ValueError(f"face {face} duplicates another face (same vertex set {key})")
        seen.add(key)

    areas = _energy.triangle_areas(mesh.verts, mesh.faces, periods=mesh.periods)
    if bool((areas <= 0.0).any()):
        bad = int(torch.nonzero(areas <= 0.0, as_tuple=False)[0, 0])
        raise ValueError(
            f"face {bad} has non-positive area {float(areas[bad]):.6g}; degenerate triangles are not allowed"
        )

    edges, incident = edge_table(mesh)
    for edge, inc in enumerate(incident):
        if len(inc) > 2:
            faces = [face for face, _ in inc]
            raise ValueError(
                f"non-manifold edge {edge} {tuple(edges[edge].tolist())} is incident to "
                f"{len(inc)} faces {faces}"
            )
        if len(inc) == 2 and inc[0][1] == inc[1][1]:
            raise ValueError(
                f"inconsistently oriented edge {edge} {tuple(edges[edge].tolist())}: "
                f"faces {inc[0][0]} and {inc[1][0]} traverse it in the same direction"
            )


def area(mesh: Mesh) -> torch.Tensor:
    """Total surface area (differentiable in the vertex coordinates)."""
    return _energy.surface_area(mesh.verts, mesh.faces, periods=mesh.periods)


def volume(mesh: Mesh) -> torch.Tensor:
    """Signed enclosed volume; positive for outward-oriented closed surfaces."""
    return _energy.signed_volume(mesh.verts, mesh.faces)


def euler_characteristic(mesh: Mesh) -> int:
    """V - E + F using the canonical edge count."""
    edges, _ = edge_table(mesh)
    return mesh.n_verts - int(edges.shape[0]) + mesh.n_faces
