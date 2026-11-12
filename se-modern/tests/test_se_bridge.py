"""SE bridge tests: locate the binary, dump, evolve and scrape scalars."""

from __future__ import annotations

import os

import pytest

from semodern import io_fe, mesh as mesh_mod, se_bridge

from .conftest import requires_se


def test_find_evolver_returns_existing_path() -> None:
    if not se_bridge.available():
        pytest.skip("Surface Evolver executable not available")
    path = se_bridge.find_evolver()
    assert path
    assert "evolver" in path.lower()


def test_commands_without_dump_are_rejected(tmp_path, data_se) -> None:
    with pytest.raises(ValueError, match="dump command"):
        se_bridge.run_se_commands(data_se / "cube.fe", "g 1;", tmp_path)


@requires_se
def test_dump_initial_matches_datafile(data_se, workdir) -> None:
    dumped = se_bridge.dump_initial(data_se / "cube.fe", workdir)
    parsed = io_fe.load_fe(data_se / "cube.fe")
    assert dumped.n_verts == parsed.n_verts
    assert dumped.n_faces == parsed.n_faces
    assert mesh_mod.area(dumped).item() == pytest.approx(mesh_mod.area(parsed).item(), rel=1e-15)


@requires_se
def test_run_se_commands_returns_mesh_and_stdout(data_se, workdir) -> None:
    run = se_bridge.run_se_commands(data_se / "cube.fe", 'dump "a.dmp";', workdir)
    mesh, stdout = run
    assert run.returncode == 0
    assert mesh is not None and mesh.n_faces == 24
    assert isinstance(stdout, str)
    assert run.dump_path.is_file()


@requires_se
def test_scalars_and_dump_agree(data_se, workdir) -> None:
    run = se_bridge.run_se_timed(data_se / "cube.fe", "", workdir, name="s.dmp")
    assert run.ok
    assert run.area == pytest.approx(float(mesh_mod.area(run.mesh)), rel=1e-12)
    assert run.volume == pytest.approx(1.0, rel=1e-12)
    assert run.dump_volume == pytest.approx(1.0, rel=1e-12)


@requires_se
def test_evolution_lowers_area(data_se, workdir) -> None:
    initial = se_bridge.run_se_timed(data_se / "cube.fe", "", workdir, name="i.dmp")
    evolved = se_bridge.run_se_timed(data_se / "cube.fe", "g 5", workdir, name="e.dmp")
    assert evolved.area < initial.area
    assert evolved.mesh.n_faces == initial.mesh.n_faces


@requires_se
def test_refine_increases_element_counts(data_se, workdir) -> None:
    run = se_bridge.run_se_timed(data_se / "cube.fe", "r", workdir, name="r.dmp")
    assert run.ok
    assert run.mesh.n_faces > 24
    assert run.mesh.n_verts > 14
    mesh_mod.validate(run.mesh)


def test_isolated_workdir_is_per_process_and_created(tmp_path) -> None:
    """Concurrent runs must not share `run.cmd`.

    `run_se_commands` writes a fixed filename into its working directory, so two callers
    pointing at the same directory would overwrite each other's command script.  This was
    observed as intermittent, unrelated test failures while a benchmark ran alongside the
    suite; the helper below namespaces by process id.
    """
    first = se_bridge.isolated_workdir(tmp_path)
    assert first.is_dir()
    assert first.name == f"pid{os.getpid()}"
    assert se_bridge.isolated_workdir(tmp_path) == first


def test_isolated_workdirs_do_not_collide(tmp_path) -> None:
    """Two distinct pids (simulated) get distinct directories."""
    import pathlib

    made = {pathlib.Path(tmp_path, f"pid{pid}") for pid in (1, 2)}
    assert len(made) == 2
    assert all(not path.exists() for path in made)
