"""Datafile/dump parsing and round-trip tests against the copied SE benchmarks."""

from __future__ import annotations

import pytest

from semodern import io_fe, mesh as mesh_mod, se_bridge
from semodern.mesh import Mesh

from .conftest import requires_se


def test_cube_counts_and_measures(data_se) -> None:
    cube = io_fe.load_fe(data_se / "cube.fe")
    assert cube.n_verts == 14
    assert cube.n_faces == 24
    assert mesh_mod.area(cube).item() == pytest.approx(6.0, rel=1e-14)
    assert abs(mesh_mod.volume(cube).item()) == pytest.approx(1.0, rel=1e-14)
    assert mesh_mod.euler_characteristic(cube) == 2


def test_cube_polygons_are_fan_triangulated_about_centroids(data_se) -> None:
    parsed = io_fe.parse_fe(data_se / "cube.fe")
    assert sized(parsed.faces_per_element) == [4] * 6
    centroids = parsed.verts[8:]
    assert centroids.shape == (6, 3)
    # Each cube face centre has exactly one coordinate at 0 or 1 and two at 0.5.
    for centre in centroids.tolist():
        assert sorted(round(value, 10) for value in centre) in ([0.0, 0.5, 0.5], [0.5, 0.5, 1.0])


def sized(elements: list[list[int]]) -> list[int]:
    return [len(element) for element in elements]


def test_cat_uses_boundary_parameterization(data_se) -> None:
    cat = io_fe.load_fe(data_se / "cat.fe")
    assert cat.n_verts == 18
    assert cat.n_faces == 24
    # Twelve ring vertices at radius RMAX = 1.5088795 on z = +-1, six interior points at z = 0.
    ring = cat.verts[cat.verts[:, 2].abs() == 1.0]
    radii = (ring[:, :2] ** 2).sum(dim=1).sqrt()
    assert radii.shape == (12,)
    assert float(radii.min()) == pytest.approx(1.5088795, rel=1e-12)
    assert float(radii.max()) == pytest.approx(1.5088795, rel=1e-12)
    assert int((cat.verts[:, 2] == 0.0).sum()) == 6
    assert int((cat.verts[:, 2] == 1.0).sum()) == 6
    assert int((cat.verts[:, 2] == -1.0).sum()) == 6


def test_fixed_vertices_are_read(data_se) -> None:
    cat = io_fe.load_fe(data_se / "cat.fe")
    assert cat.fixed == frozenset(range(12))
    assert cat.vertex_constraints == ()


def test_periods_section_is_read(data_se) -> None:
    phelanc = io_fe.load_fe(data_se / "phelanc.fe")
    assert phelanc.periods is not None
    assert phelanc.periods.tolist() == [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    assert phelanc.n_verts == 100
    assert phelanc.n_faces == 276


def test_sphere_vertex_constraints_are_read(data_se) -> None:
    sphere = io_fe.load_fe(data_se / "sphere.fe")
    # Vertices 1-4 sit on the liquid constraint, vertices 5-9 on the display constraint.
    assert [constraint for _, constraint in sphere.vertex_constraints] == [1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert sphere.fixed == frozenset({4, 5, 6, 7, 8})


def test_declared_body_volume_is_read(data_se) -> None:
    assert io_fe.load_bodies(data_se / "cube.fe")[1] == pytest.approx(1.0)
    assert io_fe.load_bodies(data_se / "sphere.fe")[1] == pytest.approx(2.0943951023931953, rel=1e-12)


def test_body_face_map_covers_all_triangles(data_se) -> None:
    parsed = io_fe.parse_fe(data_se / "cube.fe")
    assert sorted(parsed.bodies[1]) == list(range(24))


def test_save_load_round_trip(tmp_path, data_se) -> None:
    cube = io_fe.load_fe(data_se / "cube.fe")
    # save_fe writes triangle element ids, so the body must list all 24 triangles.
    target = io_fe.save_fe(cube, tmp_path / "cube_out.fe", bodies={1: range(1, 25)}, volume=1.0)
    reloaded = io_fe.load_fe(target)
    assert reloaded.n_verts == cube.n_verts
    assert reloaded.n_faces == cube.n_faces
    assert bool((reloaded.faces == cube.faces).all())
    assert (reloaded.verts - cube.verts).abs().max().item() == 0.0
    assert io_fe.load_bodies(target)[1] == pytest.approx(1.0)


def test_unclosed_edge_loop_is_rejected(tmp_path) -> None:
    text = """vertices
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
edges
1 1 2
2 2 3
3 3 1
4 1 4
faces
1 1 2 4
"""
    path = tmp_path / "broken.fe"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="not closed"):
        io_fe.load_fe(path)


def test_undefined_edge_is_rejected(tmp_path) -> None:
    text = """vertices
1 0 0 0
2 1 0 0
3 0 1 0
edges
1 1 2
2 2 3
faces
1 1 2 9
"""
    path = tmp_path / "broken.fe"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="undefined edge"):
        io_fe.load_fe(path)


def test_missing_section_is_rejected(tmp_path) -> None:
    path = tmp_path / "broken.fe"
    path.write_text("vertices\n1 0 0 0\n2 1 0 0\n3 0 1 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required section 'edges'"):
        io_fe.load_fe(path)


def test_unknown_boundary_is_rejected(tmp_path) -> None:
    text = """vertices
1 0.5 boundary 3
edges
1 1 1
faces
"""
    path = tmp_path / "broken.fe"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="undefined boundary 3"):
        io_fe.load_fe(path)


@requires_se
def test_dump_and_datafile_agree(data_se, tmp_path) -> None:
    dumped = se_bridge.dump_initial(data_se / "cube.fe", tmp_path)
    parsed = io_fe.load_fe(data_se / "cube.fe")
    assert dumped.n_verts == parsed.n_verts == 14
    assert dumped.n_faces == parsed.n_faces == 24
    assert (dumped.verts - parsed.verts).abs().max().item() == 0.0
    assert mesh_mod.area(dumped).item() == pytest.approx(mesh_mod.area(parsed).item(), rel=1e-15)
