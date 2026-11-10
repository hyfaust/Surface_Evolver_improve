"""Bridge to the reference Surface Evolver binary.

SE is used read-only as a baseline: it runs in a scratch directory with a
generated `run.cmd`, and the resulting dump is parsed back into a `Mesh`.
Nothing outside that scratch directory is written and no SE source is touched.

`run_se_commands` returns an `SeRun`, which unpacks as `(mesh, stdout)` and also
carries the process return code, the dump path and any `SE_SCALARS` line the
command script printed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .io_fe import load_dmp
from .mesh import Mesh

FALLBACK_BINARIES = (
    Path(r"E:\Mybat\evolver64.exe"),
    Path(r"E:\Mybat\evolver.exe"),
)
SE_TIMEOUT_SECONDS = 300
_DUMP_RE = re.compile(r'dump\s+"([^"]+)"')
_SCALARS_RE = re.compile(r"SE_SCALARS\s+(\S+)\s+(\S+)\s+(\S+)")
_ENERGY_RE = re.compile(r"//\s*Total energy:\s*([-\d.eEdD+]+)")
_ACTUAL_VOLUME_RE = re.compile(r"/\*actual:\s*([-\d.eEdD+]+)\s*\*/")
_SCALAR_PRINT = 'printf "SE_SCALARS %.17g %.17g %.17g\\n", total_area, body[1].volume, total_energy;'


@dataclass
class SeRun:
    """Outcome of one Surface Evolver invocation."""

    returncode: int
    stdout: str
    dump_path: Path
    mesh: Mesh | None
    area: float | None = None
    volume: float | None = None
    energy: float | None = None
    dump_volume: float | None = None

    def __iter__(self) -> Iterator[object]:
        yield self.mesh
        yield self.stdout

    @property
    def ok(self) -> bool:
        """True when SE exited cleanly and its dump parsed."""
        return self.returncode == 0 and self.mesh is not None


def find_evolver() -> str:
    """Locate `evolver64` (preferred), then the known install, then `evolver`.

    Raises `FileNotFoundError` when nothing is available, so callers can convert
    the situation into a pytest skip rather than a hard failure.
    """
    for name in ("evolver64", "evolver"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in FALLBACK_BINARIES:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("Surface Evolver executable not found (evolver64/evolver or E:\\Mybat)")


def available() -> bool:
    """True when a Surface Evolver binary can be located."""
    try:
        find_evolver()
    except FileNotFoundError:
        return False
    return True


def run_se_commands(fe_path: str | Path, commands: str, workdir: str | Path) -> SeRun:
    """Run SE on `fe_path` with `commands`; a final `quit 0;` is appended.

    `commands` must contain at least one `dump "<file>"`; the last one decides the
    file that is parsed.  The process runs with `cwd=workdir` and a 300 s cap, and
    its combined stdout/stderr is returned for scraping or diagnosis.
    """
    dump_names = _DUMP_RE.findall(commands)
    if not dump_names:
        raise ValueError("commands must contain a dump command")
    scratch = Path(workdir)
    scratch.mkdir(parents=True, exist_ok=True)
    command_file = scratch / "run.cmd"
    command_file.write_text(f"{commands.strip()}\nquit 0;\n", encoding="utf-8")

    completed = subprocess.run(
        [find_evolver(), "-Q", "-x", "-f", command_file.name, str(Path(fe_path).resolve())],
        cwd=str(scratch),
        capture_output=True,
        text=True,
        timeout=SE_TIMEOUT_SECONDS,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    dump_path = scratch / dump_names[-1]
    text = dump_path.read_text(encoding="utf-8", errors="replace") if dump_path.is_file() else ""
    scalars = _SCALARS_RE.search(output)
    energy = _ENERGY_RE.search(text)
    actual_volume = _ACTUAL_VOLUME_RE.search(text)
    return SeRun(
        returncode=completed.returncode,
        stdout=output,
        dump_path=dump_path,
        mesh=load_dmp(dump_path) if completed.returncode == 0 and dump_path.is_file() else None,
        area=_number(scalars.group(1)) if scalars else None,
        volume=_number(scalars.group(2)) if scalars else None,
        energy=_number(scalars.group(3)) if scalars else (_number(energy.group(1)) if energy else None),
        dump_volume=_number(actual_volume.group(1)) if actual_volume else None,
    )


def run_se(fe_path: str | Path, commands: str, workdir: str | Path) -> Path:
    """Run SE and return the dump path, raising when the run or the dump failed."""
    result = run_se_commands(fe_path, commands, workdir)
    if result.returncode != 0:
        raise RuntimeError(f"Surface Evolver failed ({result.returncode}) on {fe_path}:\n{result.stdout}")
    if not result.dump_path.is_file():
        raise RuntimeError(f"Surface Evolver produced no dump at {result.dump_path}:\n{result.stdout}")
    return result.dump_path


def run_se_timed(
    fe_path: str | Path,
    commands: str = "",
    workdir: str | Path = ".",
    *,
    name: str = "out.dmp",
    with_scalars: bool = True,
) -> SeRun:
    """Run SE with `commands`, optionally printing `total_area`/volume/energy first.

    `commands` must not contain its own dump: the dump for `name` is appended.
    """
    script = commands.strip().rstrip(";")
    prefix = f"{script};\n" if script else ""
    probe = f"{_SCALAR_PRINT}\n" if with_scalars else ""
    return run_se_commands(fe_path, f'{prefix}{probe}dump "{name}";', workdir)


def dump_initial(fe_path: str | Path, workdir: str | Path, *, name: str = "out.dmp") -> Mesh:
    """Dump the datafile's initial mesh without evolving and parse it."""
    result = run_se_timed(fe_path, "", workdir, name=name, with_scalars=False)
    if result.mesh is None:
        raise RuntimeError(f"Surface Evolver did not produce a usable dump: {result.stdout}")
    return result.mesh


def _number(token: str) -> float:
    return float(token.replace("d", "e").replace("D", "e"))
