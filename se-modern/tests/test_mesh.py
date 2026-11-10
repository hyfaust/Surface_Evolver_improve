"""Mesh geometry, edge table and invariant tests."""

from __future__ import annotations

import math

import pytest
import torch

from semodern import mesh as mesh_mod
from semodern.mesh import Mesh


def tetrahedron() -> Mesh:
    """Unit tetrahedron with outward-oriented faces and volume 1/6."""
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.int64)
    return Mesh(verts=verts, faces=faces)


def test_tetrahedron_area_volume_euler() -> None:
    tet = tetrahedron()
    edge_length_areas = (1.0 + 1.0 + 1.0 + math.sqrt(3.0)) / 2.0
    assert mesh_mod.area(tet).item() == pytest.approx(edge_length_areas, rel=1e-15)
    assert mesh_mod.volume(tet).item() == pytest.approx(1.0 / 6.0, rel=1e-15)
    assert mesh_mod.euler_characteristic(tet) == 2
    mesh_mod.validate(tet)


def test_edge_table_signs_and_count() -> None:
    edges, incident = mesh_mod.edge_table(tetrahedron())
    assert edges.shape == (6, 2)
    assert [len(inc) for inc in incident] == [2] * 6
    for inc in incident:
        assert sorted(sign for _, sign in inc) == [-1, 1]


def test_volume_sign_flips_with_orientation() -> None:
    tet = tetrahedron()
    flipped = Mesh(verts=tet.verts, faces=tet.faces.flip(1))
    assert mesh_mod.volume(flipped).item() == pytest.approx(-1.0 / 6.0, rel=1e-15)
    assert mesh_mod.area(flipped).item() == pytest.approx(mesh_mod.area(tet).item(), rel=1e-15)


def test_duplicate_face_rejected() -> None:
    tet = tetrahedron()
    faces = torch.cat([tet.faces, tet.faces[0:1]], dim=0)
    with pytest.raises(ValueError, match="duplicates"):
        mesh_mod.validate(Mesh(verts=tet.verts, faces=faces))


def test_non_manifold_edge_rejected() -> None:
    # Three triangles sharing the edge (0, 1).
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, -1.0]],
        dtype=torch.float64,
    )
    faces = torch.tensor([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=torch.int64)
    with pytest.raises(ValueError, match="non-manifold edge"):
        mesh_mod.validate(Mesh(verts=verts, faces=faces))


def test_inconsistent_orientation_rejected() -> None:
    tet = tetrahedron()
    faces = tet.faces.clone()
    faces[1] = faces[1].flip(0)
    with pytest.raises(ValueError, match="same direction"):
        mesh_mod.validate(Mesh(verts=tet.verts, faces=faces))


def test_degenerate_face_rejected() -> None:
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)
    with pytest.raises(ValueError, match="non-positive area"):
        mesh_mod.validate(Mesh(verts=verts, faces=faces))


def test_out_of_range_index_rejected() -> None:
    tet = tetrahedron()
    faces = tet.faces.clone()
    faces[0, 0] = 9
    with pytest.raises(ValueError, match="out of range"):
        mesh_mod.validate(Mesh(verts=tet.verts, faces=faces))


def test_area_and_volume_are_differentiable() -> None:
    tet = tetrahedron()
    verts = tet.verts.clone().requires_grad_(True)
    total = mesh_mod.area(Mesh(verts=verts, faces=tet.faces)) + mesh_mod.volume(Mesh(verts=verts, faces=tet.faces))
    (grad,) = torch.autograd.grad(total, verts)
    assert grad.shape == verts.shape
    assert bool(torch.isfinite(grad).all())


def test_fixed_mask_marks_declared_vertices() -> None:
    tet = tetrahedron()
    pinned = Mesh(verts=tet.verts, faces=tet.faces, fixed=frozenset({0, 2}))
    assert pinned.fixed_mask.tolist() == [True, False, True, False]


def test_minimum_image_shortens_split_triangle() -> None:
    # Vertex 1 sits one periodic image too far along x; wrapping must bring it back.
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.9, 0.0, 0.0], [0.1, 1.0, 0.0]], dtype=torch.float64)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    periods = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64)
    plain = Mesh(verts=verts, faces=faces)
    wrapped = Mesh(verts=verts, faces=faces, periods=periods)
    assert mesh_mod.area(plain).item() == pytest.approx(0.95, rel=1e-12)
    assert mesh_mod.area(wrapped).item() == pytest.approx(0.05, rel=1e-12)
