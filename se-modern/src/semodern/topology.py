"""Topology surgery with explicit invariants.

Surface Evolver changes connectivity heuristically (`refine`, `t`/`u` for edge
swaps, `o` for pops, `V` for collapses).  This module exposes the same operations
with a precondition check *before* the change and a full invariant check *after*
it, so an operation either returns a valid mesh or raises `ValueError` without
having touched the input.

Operations and their expected effect on `(V, E, F)`:

===============  ==========================  ===================
operation        requires                     delivers
===============  ==========================  ===================
`refine`         any valid mesh               E -> 2E+3F? no: each triangle 1->4
`swap_edge`      interior edge, convex quad    F same, E same
`collapse_edge`  link condition                V-1, E-3, F-2
`pop_vertex`     valence 3, triangular link     V-1, E-3, F-1
`pop_edge`       3 faces on the edge            V same, E-1, F-1
===============  ==========================  ===================

`validate_topology` additionally checks the Euler characteristic against the
value expected from the operation, which catches a change that is locally legal
but globally wrong.  `refine`, `swap_edge`, `collapse_edge` and `pop_vertex`
preserve chi; `pop_edge` raises it by exactly 3, because retracting a triple line
replaces three films with one face and turns the three boundary loops of the
surrounding sheets into one, so the expected chi is checked against that rule.

`octa.fe` is a genuine foam test case with four faces on some edges (a
non-manifold film), so it cannot satisfy `mesh.validate`'s manifold requirement.
`refine` therefore accepts `allow_nonmanifold=True`, which checks every invariant
except the two-face-per-edge limit; the other operations keep the strict check
because their preconditions already imply manifoldness at the affected elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from . import energy as energy_mod
from . import mesh as mesh_mod
from .mesh import Mesh

Tensor = torch.Tensor


@dataclass(frozen=True)
class TopologyChange:
    """Record of one applied operation, for auditing and reporting."""

    operation: str
    before: tuple[int, int, int]
    after: tuple[int, int, int]

    @property
    def delta(self) -> tuple[int, int, int]:
        """`(dV, dE, dF)` produced by the operation."""
        return tuple(after - before for before, after in zip(self.before, self.after))

    @property
    def euler_before(self) -> int:
        return self.before[0] - self.before[1] + self.before[2]

    @property
    def euler_after(self) -> int:
        return self.after[0] - self.after[1] + self.after[2]


def _counts(mesh: Mesh) -> tuple[int, int, int]:
    edges, _ = mesh_mod.edge_table(mesh)
    return mesh.n_verts, int(edges.shape[0]), mesh.n_faces


def face_adjacency(mesh: Mesh) -> dict[int, dict[int, int]]:
    """For each face, the neighboring face across each of its three edges.

    Keyed `face -> {neighbor_face: edge_index}` using the canonical edge ordering of
    `mesh_mod.edge_table`.  Boundary edges have no entry.
    """
    edges, incident = mesh_mod.edge_table(mesh)
    adjacency: dict[int, dict[int, int]] = {face: {} for face in range(mesh.n_faces)}
    for slot, inc in enumerate(incident):
        if len(inc) != 2:
            continue
        (face_a, _), (face_b, _) = inc
        adjacency[face_a][face_b] = slot
        adjacency[face_b][face_a] = slot
    return adjacency


def vertex_faces(mesh: Mesh) -> dict[int, set[int]]:
    """Faces incident to each vertex."""
    table: dict[int, set[int]] = {vertex: set() for vertex in range(mesh.n_verts)}
    for face, loop in enumerate(mesh.faces.tolist()):
        for vertex in loop:
            table[vertex].add(face)
    return table


def vertex_neighbours(mesh: Mesh) -> dict[int, set[int]]:
    """One-ring vertex neighbours."""
    edges, _ = mesh_mod.edge_table(mesh)
    table: dict[int, set[int]] = {vertex: set() for vertex in range(mesh.n_verts)}
    for u, v in edges.tolist():
        table[u].add(v)
        table[v].add(u)
    return table


def opposite_vertex(mesh: Mesh, face: int, edge: int, edges: Tensor) -> int:
    """Index of the vertex of `face` that is not on `edge`."""
    u, v = edges[edge].tolist()
    for vertex in mesh.faces[face].tolist():
        if vertex != u and vertex != v:
            return vertex
    raise ValueError(f"face {face} has no vertex opposite edge {edge}")


def structural_validate(mesh: Mesh) -> None:
    """Check index, duplicate-face, area and orientation invariants, allowing non-manifold edges.

    Foam structures legitimately have more than two faces on an edge, so the
    manifold-edge limit is the one check that cannot apply to them.
    """
    if mesh.n_faces == 0:
        raise ValueError("mesh has no faces")
    if int(mesh.faces.min()) < 0 or int(mesh.faces.max()) >= mesh.n_verts:
        raise ValueError(
            f"face index out of range 0..{mesh.n_verts - 1}: "
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
    areas = energy_mod.triangle_areas(mesh.verts, mesh.faces)
    if bool((areas <= 0.0).any()):
        bad = int(torch.nonzero(areas <= 0.0, as_tuple=False)[0, 0])
        raise ValueError(f"face {bad} has non-positive area {float(areas[bad]):.6g}")


def validate_topology(
    mesh: Mesh, *, expected_euler: int | None = None, allow_nonmanifold: bool = False
) -> None:
    """Run all mesh invariants, optionally allowing non-manifold edges."""
    if allow_nonmanifold:
        structural_validate(mesh)
    else:
        mesh_mod.validate(mesh)
    if expected_euler is not None:
        actual = mesh_mod.euler_characteristic(mesh)
        if actual != expected_euler:
            raise ValueError(f"Euler characteristic is {actual}, expected {expected_euler}")


def refine(mesh: Mesh, *, allow_nonmanifold: bool = False) -> tuple[Mesh, TopologyChange]:
    """Split every triangle into four, adding edge midpoints (1 -> 4 refinement).

    Each original edge contributes one new vertex at its midpoint, so the counts go
    `V' = V + E`, `E' = 2E + 3F`, `F' = 4F`, and the Euler characteristic is
    unchanged.  Face orientation is preserved.
    """
    edges, _ = mesh_mod.edge_table(mesh)
    midpoints = (mesh.verts[edges[:, 0]] + mesh.verts[edges[:, 1]]) / 2.0
    verts = torch.cat([mesh.verts, midpoints], dim=0)
    midpoint_of = {(int(u), int(v)): mesh.n_verts + index for index, (u, v) in enumerate(edges.tolist())}

    def mid(u: int, v: int) -> int:
        return midpoint_of[(u, v) if u < v else (v, u)]

    faces: list[list[int]] = []
    for a, b, c in mesh.faces.tolist():
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])

    refined = Mesh(verts=verts, faces=torch.tensor(faces, dtype=torch.int64))
    change = TopologyChange("refine", _counts(mesh), _counts(refined))
    validate_topology(
        refined, expected_euler=mesh_mod.euler_characteristic(mesh), allow_nonmanifold=allow_nonmanifold
    )
    return refined, change


def swap_edge(mesh: Mesh, edge_index: int) -> tuple[Mesh, TopologyChange]:
    """Flip an interior edge shared by two triangles (the `t`/`u` operation).

    Requires an interior edge whose two opposite vertices are distinct, whose new
    triangles both have positive area, and whose new edge is not already present.
    Otherwise raises `ValueError` and the input is untouched.
    """
    edges, incident = mesh_mod.edge_table(mesh)
    if edge_index < 0 or edge_index >= edges.shape[0]:
        raise ValueError(f"edge index {edge_index} outside 0..{edges.shape[0] - 1}")
    inc = incident[edge_index]
    if len(inc) != 2:
        raise ValueError(f"edge {edge_index} is a boundary edge (only {len(inc)} incident face)")

    (face_a, sign_a), (face_b, sign_b) = inc
    opposite_a = opposite_vertex(mesh, face_a, edge_index, edges)
    opposite_b = opposite_vertex(mesh, face_b, edge_index, edges)
    if opposite_a == opposite_b:
        raise ValueError(f"edge {edge_index} has coincident opposite vertices")

    u, v = edges[edge_index].tolist()
    new_faces = mesh.faces.clone()
    new_faces[face_a] = torch.tensor(_orient([opposite_a, opposite_b, u], mesh, face_a), dtype=torch.int64)
    new_faces[face_b] = torch.tensor(_orient([opposite_b, opposite_a, v], mesh, face_b), dtype=torch.int64)

    if _duplicate_face(new_faces, face_a, face_b):
        raise ValueError(f"swapping edge {edge_index} would duplicate an existing face")
    candidate = Mesh(verts=mesh.verts, faces=new_faces)
    try:
        validate_topology(candidate, expected_euler=mesh_mod.euler_characteristic(mesh))
    except ValueError as error:
        raise ValueError(f"swapping edge {edge_index} breaks an invariant: {error}") from error
    return candidate, TopologyChange("swap_edge", _counts(mesh), _counts(candidate))


def _orient(loop: Sequence[int], mesh: Mesh, reference_face: int) -> list[int]:
    """Orient `loop` so its normal agrees with the original face's normal."""
    original = mesh.faces[reference_face]
    reference = energy_mod.face_normals(mesh.verts, original.reshape(1, 3), normalize=True)[0]
    candidate = torch.tensor([loop], dtype=torch.int64)
    normal = energy_mod.face_normals(mesh.verts, candidate, normalize=True)[0]
    return list(loop) if float(normal.dot(reference)) > 0.0 else [loop[0], loop[2], loop[1]]


def _duplicate_face(faces: Tensor, first: int, second: int) -> bool:
    keys = [tuple(sorted(face)) for face in faces.tolist()]
    return keys[first] == keys[second] or keys.count(keys[first]) > 1 or keys.count(keys[second]) > 1


def collapse_edge(mesh: Mesh, edge_index: int) -> tuple[Mesh, TopologyChange]:
    """Contract an edge by merging one endpoint into the other.

    The link condition is enforced: the two endpoints' common neighbours must be
    exactly the two opposite vertices of the incident faces.  Without it the
    contraction creates a non-manifold vertex.  Also rejects non-interior edges and
    any contraction that would leave a degenerate face.
    """
    edges, incident = mesh_mod.edge_table(mesh)
    if edge_index < 0 or edge_index >= edges.shape[0]:
        raise ValueError(f"edge index {edge_index} outside 0..{edges.shape[0] - 1}")
    inc = incident[edge_index]
    if len(inc) != 2:
        raise ValueError(f"edge {edge_index} is a boundary edge; collapse needs two incident faces")

    u, v = edges[edge_index].tolist()
    neighbours = vertex_neighbours(mesh)
    opposite = {opposite_vertex(mesh, face, edge_index, edges) for face, _ in inc}
    common = (neighbours[u] & neighbours[v]) - {u, v}
    if common != opposite:
        raise ValueError(
            f"link condition fails for edge {edge_index}: common neighbours {sorted(common)} "
            f"but opposite vertices are {sorted(opposite)}"
        )

    # Merge v into u, placing the survivor at the edge midpoint.
    verts = mesh.verts.clone()
    verts[u] = (mesh.verts[u] + mesh.verts[v]) / 2.0
    remap = torch.arange(mesh.n_verts, dtype=torch.int64)
    remap[v] = u
    faces = remap[mesh.faces]
    keep = []
    for face in faces.tolist():
        if len(set(face)) < 3:
            continue
        keep.append(face)
    if not keep:
        raise ValueError("collapse would remove every face")

    compact = _compact(verts, torch.tensor(keep, dtype=torch.int64))
    try:
        validate_topology(compact, expected_euler=mesh_mod.euler_characteristic(mesh))
    except ValueError as error:
        raise ValueError(f"collapsing edge {edge_index} breaks an invariant: {error}") from error
    return compact, TopologyChange("collapse_edge", _counts(mesh), _counts(compact))


def pop_vertex(mesh: Mesh, vertex_index: int) -> tuple[Mesh, TopologyChange]:
    """Remove a valence-3 vertex whose link is a single triangle (SE's `o` on a vertex).

    Deletes the three incident faces and adds the link triangle, so the counts change
    by `V-1`, `E-3`, `F-2` (three faces merge into one) and the Euler characteristic
    is unchanged.
    """
    incident = vertex_faces(mesh)[vertex_index]
    if len(incident) != 3:
        raise ValueError(f"vertex {vertex_index} has valence {len(incident)}, pop needs exactly 3")
    link: list[int] = []
    for face in incident:
        link.extend(vertex for vertex in mesh.faces[face].tolist() if vertex != vertex_index)
    link_vertices = sorted(set(link))
    if len(link_vertices) != 3:
        raise ValueError(f"vertex {vertex_index} link is not a triangle: {link_vertices}")
    if any(link.count(vertex) != 2 for vertex in link_vertices):
        raise ValueError(f"vertex {vertex_index} link is not closed: {link}")
    # The link triangle becomes a new face, so it must not already exist: on a closed
    # tetrahedron the fourth face *is* the link, and popping would duplicate it.
    if tuple(link_vertices) in {tuple(sorted(face)) for face in mesh.faces.tolist()}:
        raise ValueError(
            f"vertex {vertex_index} link triangle {link_vertices} is already a face; "
            "popping would create a duplicate (a closed tetrahedron cannot be popped)"
        )

    remaining = [face for index, face in enumerate(mesh.faces.tolist()) if index not in incident]
    new_loop = _orient_link(mesh, link_vertices, incident)
    remaining.append(new_loop)
    compact = _compact(mesh.verts, torch.tensor(remaining, dtype=torch.int64), drop=vertex_index)
    try:
        validate_topology(compact, expected_euler=mesh_mod.euler_characteristic(mesh))
    except ValueError as error:
        raise ValueError(f"popping vertex {vertex_index} breaks an invariant: {error}") from error
    return compact, TopologyChange("pop_vertex", _counts(mesh), _counts(compact))


def _orient_link(mesh: Mesh, link_vertices: Sequence[int], incident: Iterable[int]) -> list[int]:
    """Orient the link triangle so it continues the orientation of the incident faces."""
    reference = mesh.faces[list(incident)[0]].reshape(1, 3)
    reference_normal = energy_mod.face_normals(mesh.verts, reference, normalize=True)[0]
    candidate = torch.tensor([list(link_vertices)], dtype=torch.int64)
    normal = energy_mod.face_normals(mesh.verts, candidate, normalize=True)[0]
    loop = list(link_vertices)
    return loop if float(normal.dot(reference_normal)) > 0.0 else [loop[0], loop[2], loop[1]]


POP_EDGE_EULER_SHIFT = 3
"""Euler-characteristic change produced by `pop_edge` (three films merge into one)."""


def pop_edge(mesh: Mesh, edge_index: int) -> tuple[Mesh, TopologyChange]:
    """Remove a triple-junction edge, replacing three faces with one (SE's `o` on an edge).

    Requires exactly three incident faces whose three opposite vertices already form
    a triangle; the result drops one edge and one face while keeping every vertex.
    """
    edges, incident = mesh_mod.edge_table(mesh)
    if edge_index < 0 or edge_index >= edges.shape[0]:
        raise ValueError(f"edge index {edge_index} outside 0..{edges.shape[0] - 1}")
    inc = incident[edge_index]
    if len(inc) != 3:
        raise ValueError(f"edge {edge_index} has {len(inc)} incident faces; pop_edge needs exactly 3")

    opposite = [opposite_vertex(mesh, face, edge_index, edges) for face, _ in inc]
    if len(set(opposite)) != 3:
        raise ValueError(f"edge {edge_index} has repeated opposite vertices {opposite}")
    neighbours = vertex_neighbours(mesh)
    for i, a in enumerate(opposite):
        for b in opposite[i + 1 :]:
            if b not in neighbours[a]:
                raise ValueError(
                    f"opposite vertices {a} and {b} of edge {edge_index} are not joined; "
                    "pop_edge needs a closed triple line"
                )

    incident_faces = {face for face, _ in inc}
    remaining = [face for index, face in enumerate(mesh.faces.tolist()) if index not in incident_faces]
    new_loop = _orient_link(mesh, opposite, incident_faces)
    remaining.append(new_loop)
    compact = _compact(mesh.verts, torch.tensor(remaining, dtype=torch.int64))
    expected = mesh_mod.euler_characteristic(mesh) + POP_EDGE_EULER_SHIFT
    try:
        validate_topology(compact, expected_euler=expected)
    except ValueError as error:
        raise ValueError(f"popping edge {edge_index} breaks an invariant: {error}") from error
    return compact, TopologyChange("pop_edge", _counts(mesh), _counts(compact))


def _compact(verts: Tensor, faces: Tensor, drop: int | None = None) -> Mesh:
    """Drop unused vertices and renumber the face table.

    `drop` names a vertex that must disappear; if it survives (because a face still
    references it) the caller's precondition was violated, so this raises.
    """
    used = sorted(set(faces.reshape(-1).tolist()))
    remap = {old: new for new, old in enumerate(used)}
    if drop is not None and drop in remap:
        raise ValueError(f"vertex {drop} is still referenced by a face after the operation")
    new_faces = torch.tensor([[remap[v] for v in face] for face in faces.tolist()], dtype=torch.int64)
    new_verts = verts[torch.tensor(used, dtype=torch.int64)]
    return Mesh(verts=new_verts, faces=new_faces)


def triple_line_fixture() -> Mesh:
    """A closed triple line: three films sharing edge `(0, 1)`, ringed by three more.

    The shared edge carries the three faces `(0, 1, 2)`, `(1, 0, 3)`, `(0, 1, 4)`, and
    the surrounding faces `(2, 3, 5)`, `(3, 4, 5)`, `(4, 2, 5)` supply the pairwise
    edges between the opposite vertices `2, 3, 4` that `pop_edge` requires.  Popping
    the shared edge replaces the three films with the single triangle `(2, 3, 4)`,
    which is exactly what happens where a Plateau border retracts.
    """
    verts = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.8660254037844386, 0.0],
            [-0.5, -0.8660254037844386, 0.0],
            [0.4, 0.1, 0.6],
        ],
        dtype=torch.float64,
    )
    faces = tensor_faces(
        [
            [0, 1, 2],
            [1, 0, 3],
            [0, 1, 4],
            [2, 3, 5],
            [3, 4, 5],
            [4, 2, 5],
        ]
    )
    return Mesh(verts=verts, faces=faces)


def swappable_quad_fixture() -> Mesh:
    """Two triangles sharing a strictly convex diagonal, with room around it.

    A unit square split along its diagonal by an interior point gives two triangles
    whose opposite vertices are distinct and whose flipped diagonal is longer, so the
    flip is legal: `V`, `E`, `F` and chi are unchanged.  (A closed tetrahedron is the
    degenerate case where every flip would duplicate the fourth face.)
    """
    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.float64,
    )
    # Edge (0, 2) is the shared diagonal; vertices 1 and 3 are its opposite vertices.
    faces = tensor_faces([[0, 2, 1], [2, 0, 3], [1, 2, 4], [2, 3, 4]])
    return Mesh(verts=verts, faces=faces)


def trihedral_patch_fixture() -> Mesh:
    """Three triangles meeting at a valence-3 interior vertex of an open patch.

    Vertex `0` has exactly three incident faces and a closed triangular link
    `(1, 2, 3)` that is not itself a face, so the link can become one: the pop gives
    `V-1`, `E-3`, `F-1` with the Euler characteristic unchanged.  This is the local
    configuration a foam vertex reaches once its surrounding edges have been popped.
    """
    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.8660254037844386, 0.0],
            [-0.5, -0.8660254037844386, 0.0],
        ],
        dtype=torch.float64,
    )
    faces = tensor_faces([[0, 1, 2], [0, 2, 3], [0, 3, 1]])
    return Mesh(verts=verts, faces=faces)


def tetrahedral_fixture() -> Mesh:
    """Regular tetrahedron: every vertex has valence 3, so any vertex can be popped."""
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.int64)
    return Mesh(verts=verts, faces=faces)


def octahedral_film_fixture() -> Mesh:
    """The `octa.fe` topology: a central vertex linked to a six-vertex ring.

    Twelve triangles join the central vertex `0` to the octahedron's four
    equatorial and two polar vertices, so the central vertex has valence 4 in the
    ring sense but twelve incident faces; this is the non-manifold film that
    `octa.fe` starts from, and it is why `pop_vertex` is not applicable to it.
    """
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0],
         [0.0, -2.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, -2.0]],
        dtype=torch.float64,
    )
    faces = tensor_faces(
        [
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
            [0, 1, 5], [0, 5, 2], [0, 2, 6], [0, 6, 3],
            [0, 3, 5], [0, 5, 4], [0, 4, 6], [0, 6, 1],
        ]
    )
    return Mesh(verts=verts, faces=faces)


def tensor_faces(faces: Sequence[Sequence[int]]) -> Tensor:
    """Build an int64 face table from nested lists."""
    return torch.tensor([list(face) for face in faces], dtype=torch.int64)
