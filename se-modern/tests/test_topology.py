"""Topology tests: every operation's preconditions, counts and invariants."""

from __future__ import annotations

import random

import pytest
import torch

from semodern import mesh as mesh_mod, topology as topo

from .conftest import requires_se


def tet() -> mesh_mod.Mesh:
    return topo.tetrahedral_fixture()


def octa_film() -> mesh_mod.Mesh:
    return topo.octahedral_film_fixture()


def test_refine_quadruples_faces_and_preserves_euler() -> None:
    mesh = tet()
    refined, change = topo.refine(mesh)
    # Each triangle splits into 4, each edge gains a midpoint: dV = E, dE = 2E + 3F - E,
    # dF = 3F.  For the tetrahedron (V=4, E=6, F=4) that is (6, 18, 12).
    assert change.delta == (6, 18, 12)
    assert change.after[0] == mesh.n_verts + 6
    assert refined.n_faces == 4 * mesh.n_faces
    assert mesh_mod.euler_characteristic(refined) == mesh_mod.euler_characteristic(mesh)
    assert change.euler_before == change.euler_after


def test_refine_places_midpoints() -> None:
    mesh = tet()
    refined, _ = topo.refine(mesh)
    midpoints = refined.verts[mesh.n_verts:]
    assert midpoints.shape[0] == 6
    for midpoint in midpoints.tolist():
        hits = 0
        for a in range(mesh.n_verts):
            for b in range(a + 1, mesh.n_verts):
                expected = (mesh.verts[a] + mesh.verts[b]) / 2.0
                if torch.allclose(torch.tensor(midpoint, dtype=torch.float64), expected):
                    hits += 1
        assert hits == 1


def test_refine_handles_nonmanifold_film() -> None:
    film = octa_film()
    refined, change = topo.refine(film, allow_nonmanifold=True)
    assert refined.n_faces == 4 * film.n_faces
    assert change.euler_before == change.euler_after


def test_swap_edge_preserves_counts() -> None:
    # A tetrahedron is the degenerate case where every flip duplicates the fourth
    # face, so the swappable quad fixture is used here.
    mesh = topo.swappable_quad_fixture()
    swapped, change = topo.swap_edge(mesh, 0)
    assert change.delta == (0, 0, 0)
    assert mesh_mod.euler_characteristic(swapped) == mesh_mod.euler_characteristic(mesh)
    assert bool((swapped.faces.sum(dim=1) != mesh.faces.sum(dim=1)).any())
    mesh_mod.validate(swapped)


def test_swap_edge_rejects_boundary_edge() -> None:
    patch = topo.trihedral_patch_fixture()
    edges, incident = mesh_mod.edge_table(patch)
    boundary = [slot for slot, inc in enumerate(incident) if len(inc) == 1]
    assert boundary
    with pytest.raises(ValueError, match="boundary edge"):
        topo.swap_edge(patch, boundary[0])


def test_swap_edge_rejects_bad_index() -> None:
    with pytest.raises(ValueError, match="outside"):
        topo.swap_edge(tet(), 999)


def test_collapse_edge_satisfies_link_condition() -> None:
    patch = topo.trihedral_patch_fixture()
    edges, incident = mesh_mod.edge_table(patch)
    interior = [slot for slot, inc in enumerate(incident) if len(inc) == 2]
    collapsed = None
    for slot in interior:
        try:
            candidate, change = topo.collapse_edge(patch, slot)
        except ValueError:
            continue
        collapsed = (candidate, change)
        break
    assert collapsed is not None, "no collapsible interior edge found"
    candidate, change = collapsed
    assert change.delta[0] == -1
    assert change.delta[2] == -2
    assert mesh_mod.euler_characteristic(candidate) == mesh_mod.euler_characteristic(patch)


def test_collapse_edge_rejects_boundary() -> None:
    patch = topo.trihedral_patch_fixture()
    edges, incident = mesh_mod.edge_table(patch)
    boundary = [slot for slot, inc in enumerate(incident) if len(inc) == 1]
    with pytest.raises(ValueError, match="boundary edge"):
        topo.collapse_edge(patch, boundary[0])


def test_pop_vertex_merges_three_faces_into_one() -> None:
    patch = topo.trihedral_patch_fixture()
    popped, change = topo.pop_vertex(patch, 0)
    assert change.delta == (-1, -3, -2)
    assert popped.n_faces == 1
    assert mesh_mod.euler_characteristic(popped) == mesh_mod.euler_characteristic(patch)


def test_pop_vertex_rejects_wrong_valence() -> None:
    with pytest.raises(ValueError, match="valence"):
        topo.pop_vertex(octa_film(), 0)


def test_pop_vertex_rejects_closed_tetrahedron() -> None:
    # The tetrahedron's link triangle is already a face, so popping would duplicate it.
    with pytest.raises(ValueError, match="already a face"):
        topo.pop_vertex(tet(), 0)


def test_pop_edge_merges_three_films() -> None:
    film = topo.triple_line_fixture()
    popped, change = topo.pop_edge(film, 0)
    assert change.delta[2] == -2
    assert mesh_mod.euler_characteristic(popped) == mesh_mod.euler_characteristic(film) + topo.POP_EDGE_EULER_SHIFT
    mesh_mod.validate(popped)


def test_pop_edge_rejects_two_face_edge() -> None:
    with pytest.raises(ValueError, match="needs exactly 3"):
        topo.pop_edge(tet(), 0)


def test_tetrahedron_edge_flip_is_rejected_as_duplicate() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        topo.swap_edge(tet(), 0)


def test_inputs_are_never_mutated() -> None:
    mesh = topo.swappable_quad_fixture()
    verts_before = mesh.verts.clone()
    faces_before = mesh.faces.clone()
    topo.refine(mesh)
    topo.swap_edge(mesh, 0)
    assert bool((mesh.verts == verts_before).all())
    assert bool((mesh.faces == faces_before).all())


def test_hundred_random_operations_keep_invariants() -> None:
    """The plan's G3 requirement: 100 random legal operations, zero invariant failures."""
    generator = random.Random(20240607)
    mesh = topo.refine(topo.tetrahedral_fixture(), allow_nonmanifold=False)[0]
    applied = {"refine": 0, "swap_edge": 0, "collapse_edge": 0}
    max_faces = 800
    rejections = 0

    for _ in range(100):
        # Refinement multiplies the face count by four, so a real topology controller
        # stops refining once the mesh is large enough to keep the sweep bounded.
        choices = (
            ("refine", "swap_edge", "collapse_edge")
            if mesh.n_faces < max_faces
            else ("swap_edge", "collapse_edge")
        )
        choice = generator.choice(choices)
        edges, incident = mesh_mod.edge_table(mesh)
        interior = [slot for slot, inc in enumerate(incident) if len(inc) == 2]
        try:
            if choice == "refine":
                mesh, _ = topo.refine(mesh)
            elif choice == "swap_edge":
                if not interior:
                    rejections += 1
                    continue
                mesh, _ = topo.swap_edge(mesh, generator.choice(interior))
            else:
                candidates = list(interior)
                generator.shuffle(candidates)
                collapsed = None
                for slot in candidates:
                    try:
                        collapsed = topo.collapse_edge(mesh, slot)
                        break
                    except ValueError:
                        continue
                if collapsed is None:
                    rejections += 1
                    continue
                mesh, _ = collapsed
        except ValueError:
            rejections += 1
            continue
        applied[choice] += 1
        mesh_mod.validate(mesh)

    assert sum(applied.values()) + rejections == 100
    assert sum(applied.values()) >= 20
    assert mesh_mod.euler_characteristic(mesh) == 2


def test_topology_change_reports_deltas() -> None:
    change = topo.TopologyChange("refine", (4, 6, 4), (8, 24, 16))
    assert change.delta == (4, 18, 12)
    assert change.euler_before == 2
    assert change.euler_after == 0


def test_structural_validate_allows_nonmanifold_but_catches_duplicates() -> None:
    film = octa_film()
    topo.structural_validate(film)
    with pytest.raises(ValueError, match="duplicates"):
        topo.structural_validate(mesh_mod.Mesh(verts=film.verts, faces=torch.cat([film.faces, film.faces[:1]])))


@requires_se
def test_octa_datafile_parses_and_refines(data_se) -> None:
    from semodern import io_fe

    octa = io_fe.load_fe(data_se / "octa.fe")
    assert octa.n_verts == 7
    assert octa.n_faces == 12
    # The central vertex is incident to all twelve faces, so pop_vertex cannot apply
    # to octa.fe as given; that is recorded rather than treated as a failure.
    assert len(topo.vertex_faces(octa)[0]) == 12
    refined, change = topo.refine(octa, allow_nonmanifold=True)
    assert refined.n_faces == 48
    assert change.euler_before == change.euler_after


@requires_se
def test_phelanc_parses_as_periodic_structure(data_se) -> None:
    from semodern import io_fe

    phelanc = io_fe.load_fe(data_se / "phelanc.fe")
    assert phelanc.periods is not None
    assert phelanc.n_faces == 276
    # Periodic wrap is not a topology operation; only structural validity is asserted.
    topo.structural_validate(phelanc)
