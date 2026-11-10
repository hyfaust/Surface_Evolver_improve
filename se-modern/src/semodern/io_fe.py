"""Readers/writers for Surface Evolver `.fe` datafiles and `.dmp` dumps.

Parsed sections: `vertices`, `edges`, `faces`, `facets`, `bodies`, `periods`.
Vertex coordinates may be literals, expressions over `PARAMETER` values, or
boundary parameterizations (`<param> ... boundary <n>`) evaluated from the file's
`BOUNDARY` declarations.  Polygonal faces are fan-triangulated about their
centroid, exactly as SE's own `dump` does: a cube's six quadrilateral faces
become twenty-four triangles around six new vertices.

Recognized vertex attributes are `fixed` and `constraint <n>`; every other
attribute keyword (`color`, `density`, `no_refine`, `original`, ...) is ignored,
as is everything from `read` on (SE command procedures).  `\\` line
continuations are joined.  All parse failures raise `ValueError` carrying the
offending line number.

Returned `Mesh` objects use 0-based vertex indices; file ids are 1-based.

Note on periodic files (`TORUS`, `TORUS_FILLED`, `PERIODS`): SE stores an explicit
wrap flag on every edge and derives facet areas from those flags, which cannot be
recovered from bare vertex coordinates once polygons have been triangulated.  The
`periods` lattice is still recorded and `energy` applies a minimum-image
convention (see `energy.minimum_image`), but absolute areas of coarse periodic
files such as `phelanc.fe` are not comparable with SE; the benchmarks treat that
case as a parse/topology case only.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .mesh import Mesh

_SECTION_RE = re.compile(r"^\s*(vertices|edges|faces|facets|bodies|periods)\s*:?\s*(?:/\*.*)?$", re.IGNORECASE)
_KEYWORD_RE = re.compile(r"^[A-Za-z_]")
_PARAMETER_RE = re.compile(r"^\s*PARAMETER\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.IGNORECASE)
_BOUNDARY_RE = re.compile(r"^\s*BOUNDARY\s+(\d+)\s+PARAMETERS\s+(\d+)\s*$", re.IGNORECASE)
_COORD_RE = re.compile(r"^\s*X([1-9]\d*)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_INT_RE = re.compile(r"^[+-]?\d+$")
_SE_FLOAT_RE = re.compile(r"(?<=\d)[dD](?=[+-]?\d)")

_FUNCTIONS = {
    name: getattr(math, name)
    for name in (
        "cos", "sin", "tan", "acos", "asin", "atan", "atan2", "cosh", "sinh", "tanh",
        "exp", "log", "log10", "sqrt", "fabs", "floor", "ceil", "pow", "hypot",
        "degrees", "radians",
    )
}
_FUNCTIONS.update({"abs": abs, "min": min, "max": max, "pi": math.pi, "e": math.e})


@dataclass
class FeStructure:
    """Everything `parse_fe` extracts from a Surface Evolver file."""

    verts: torch.Tensor
    faces: torch.Tensor
    faces_per_element: list[list[int]] = field(default_factory=list)
    bodies: dict[int, list[int]] = field(default_factory=dict)
    volumes: dict[int, float] = field(default_factory=dict)
    periods: torch.Tensor | None = None
    fixed: frozenset[int] = frozenset()
    vertex_constraints: tuple[tuple[int, int], ...] = ()


def _strip_comments(text: str) -> str:
    """Remove `/* ... */` (possibly multi-line) and `//` comments, keeping the line count."""
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _eval_expr(expr: str, env: dict[str, float], line_no: int) -> float:
    """Evaluate an SE scalar expression (`^` is exponentiation, `1.5d0` a literal).

    SE identifiers are case-insensitive: dumps print `P1` where datafiles write
    `p1`, so every name is offered in original, lower- and upper-case spelling.
    """
    scope: dict[str, float] = {}
    for key, value in {**_FUNCTIONS, **env}.items():
        scope[key] = value
        scope[key.lower()] = value
        scope[key.upper()] = value
    python_expr = _SE_FLOAT_RE.sub("e", expr).replace("^", "**")
    try:
        value = eval(python_expr, {"__builtins__": {}}, scope)  # noqa: S307
    except NameError as exc:
        raise ValueError(f"line {line_no}: unknown name in expression {expr!r}: {exc}") from exc
    except (SyntaxError, TypeError, ZeroDivisionError, ValueError) as exc:
        raise ValueError(f"line {line_no}: cannot evaluate expression {expr!r}: {exc}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"line {line_no}: expression {expr!r} is not a scalar")
    return float(value)


def _join_continuations(lines: Sequence[str]) -> list[tuple[int, str]]:
    """Join SE backslash line continuations, keeping the first physical line number."""
    joined: list[tuple[int, str]] = []
    pending: list[str] = []
    start = 0
    for line_no, raw in enumerate(lines, start=1):
        text = raw.strip()
        if pending and not text:
            continue
        if text.endswith("\\"):
            if not pending:
                start = line_no
            pending.append(text[:-1].strip())
            continue
        if pending:
            pending.append(text)
            joined.append((start, " ".join(pending)))
            pending = []
            continue
        joined.append((line_no, text))
    if pending:
        raise ValueError(f"line {start}: dangling line continuation")
    return joined


def _scan(
    lines: Sequence[str],
) -> tuple[dict[str, list[tuple[int, str]]], dict[str, float], dict[int, dict[int, str]]]:
    """Split joined lines into geometry sections, `PARAMETER` values and boundary formulas.

    A section ends at the next keyword line that is not one of its own data lines, so
    the command procedures after `read` are never captured as geometry.  A boundary
    maps axis number -> formula; `PARAMETERS m` bounds the boundary parameter count
    (`p1..pm`), not the number of coordinate axes, and absent axes default to `0`.
    """
    sections: dict[str, list[tuple[int, str]]] = {}
    params: dict[str, float] = {}
    boundaries: dict[int, dict[int, str]] = {}
    current: str | None = None
    open_boundary: int | None = None

    for line_no, line in _join_continuations(lines):
        if not line:
            continue
        parameter = _PARAMETER_RE.match(line)
        if parameter:
            params[parameter.group(1)] = _eval_expr(parameter.group(2), params, line_no)
            current, open_boundary = None, None
            continue
        boundary = _BOUNDARY_RE.match(line)
        if boundary:
            boundaries[int(boundary.group(1))] = {}
            current, open_boundary = None, int(boundary.group(1))
            continue
        coordinate = _COORD_RE.match(line)
        if coordinate:
            if open_boundary is None:
                continue  # an `x1:` label outside a boundary block
            boundaries[open_boundary][int(coordinate.group(1))] = coordinate.group(2)
            continue
        section = _SECTION_RE.match(line)
        if section:
            current = section.group(1).lower()
            sections.setdefault(current, [])
            open_boundary = None
            continue
        if _KEYWORD_RE.match(line):
            current, open_boundary = None, None
            continue
        if current is not None:
            sections[current].append((line_no, line))
    return sections, params, boundaries


def _attribute(token_list: Sequence[str]) -> tuple[bool, int | None]:
    """Extract the `fixed` flag and `constraint <n>` id from an attribute token list."""
    lowered = [token.lower() for token in token_list]
    is_fixed = "fixed" in lowered
    constraint: int | None = None
    if "constraint" in lowered:
        slot = lowered.index("constraint") + 1
        if slot < len(token_list) and _INT_RE.match(token_list[slot]):
            constraint = int(token_list[slot])
    return is_fixed, constraint


def _parse_vertices(
    entries: Sequence[tuple[int, str]],
    params: dict[str, float],
    boundaries: dict[int, dict[int, str]],
) -> tuple[torch.Tensor, frozenset[int], tuple[tuple[int, int], ...]]:
    coords: dict[int, list[float]] = {}
    fixed: set[int] = set()
    constraints: list[tuple[int, int]] = []
    for line_no, line in entries:
        tokens = line.split()
        if not _INT_RE.match(tokens[0]):
            raise ValueError(f"line {line_no}: vertex id expected, got {tokens[0]!r}")
        vertex_id = int(tokens[0])
        lowered = [token.lower() for token in tokens]
        if "boundary" in lowered:
            slot = lowered.index("boundary")
            if slot + 1 >= len(tokens) or not _INT_RE.match(tokens[slot + 1]):
                raise ValueError(f"line {line_no}: vertex {vertex_id} has a malformed boundary reference")
            formulas = boundaries.get(int(tokens[slot + 1]))
            if formulas is None:
                raise ValueError(f"line {line_no}: vertex {vertex_id} uses undefined boundary {tokens[slot + 1]}")
            env = dict(params)
            for axis, raw in enumerate(tokens[1:slot], start=1):
                env[f"p{axis}"] = _eval_expr(raw, params, line_no)
            coords[vertex_id] = [
                _eval_expr(formulas[axis], env, line_no) if axis in formulas else 0.0 for axis in (1, 2, 3)
            ]
        elif len(tokens) < 4:
            raise ValueError(f"line {line_no}: vertex {vertex_id} needs three coordinates, got {tokens[1:]!r}")
        else:
            coords[vertex_id] = [_eval_expr(raw, params, line_no) for raw in tokens[1:4]]
        is_fixed, constraint = _attribute(tokens[1:])
        if is_fixed:
            fixed.add(vertex_id - 1)
        if constraint is not None:
            constraints.append((vertex_id - 1, constraint))
    if not coords:
        raise ValueError("no vertices found")
    expected = list(range(1, len(coords) + 1))
    if sorted(coords) != expected:
        raise ValueError(f"vertex ids must be 1..{len(coords)} in order, got {sorted(coords)}")
    table = torch.tensor([coords[i] for i in expected], dtype=torch.float64)
    return table, frozenset(fixed), tuple(constraints)


def _parse_edges(entries: Sequence[tuple[int, str]]) -> dict[int, tuple[int, int]]:
    """Edge id -> (v1, v2) as 0-based indices, preserving the file's orientation."""
    edges: dict[int, tuple[int, int]] = {}
    for line_no, line in entries:
        tokens = line.split()
        if len(tokens) < 3 or not all(_INT_RE.match(token) for token in tokens[:3]):
            raise ValueError(f"line {line_no}: edge needs '<id> <vertex> <vertex>', got {tokens!r}")
        edges[int(tokens[0])] = (int(tokens[1]) - 1, int(tokens[2]) - 1)
    if not edges:
        raise ValueError("no edges found")
    return edges


def _parse_faces(entries: Sequence[tuple[int, str]], edges: dict[int, tuple[int, int]]) -> list[list[int]]:
    """One ordered 0-based vertex ring per face, from its signed edge loop."""
    loops: list[list[int]] = []
    for line_no, line in entries:
        tokens = line.split()
        if not _INT_RE.match(tokens[0]):
            raise ValueError(f"line {line_no}: face id expected, got {tokens[0]!r}")
        directed: list[int] = []
        for token in tokens[1:]:
            if not _INT_RE.match(token):
                break
            directed.append(int(token))
        if len(directed) < 3:
            raise ValueError(f"line {line_no}: face {tokens[0]} needs an edge loop of >= 3 edges, got {directed}")
        loop: list[int] = []
        for position, signed in enumerate(directed):
            edge = edges.get(abs(signed))
            if edge is None:
                raise ValueError(f"line {line_no}: face {tokens[0]} references undefined edge {abs(signed)}")
            u, v = edge if signed > 0 else (edge[1], edge[0])
            if position == 0:
                loop.append(u)
            elif loop[-1] != u:
                raise ValueError(
                    f"line {line_no}: face {tokens[0]} edge loop is not closed: edge {signed} starts at "
                    f"vertex {u} but the previous edge ended at vertex {loop[-1]}"
                )
            loop.append(v)
        if loop[-1] == loop[0]:
            loop.pop()
        if len(loop) != len(directed):
            raise ValueError(f"line {line_no}: face {tokens[0]} edge loop never returns to its start vertex {loop[0]}")
        loops.append(loop)
    if not loops:
        raise ValueError("no faces found")
    return loops


def _fan_triangulate(
    loops: Sequence[Sequence[int]], verts: torch.Tensor
) -> tuple[torch.Tensor, list[list[int]], torch.Tensor]:
    """Triangulate polygons about their centroid; returns triangles, per-face ids, new verts."""
    triangles: list[list[int]] = []
    per_element: list[list[int]] = []
    extra: list[torch.Tensor] = []
    for loop in loops:
        ids: list[int] = []
        if len(loop) == 3:
            ids.append(len(triangles))
            triangles.append(list(loop))
        else:
            centre_index = verts.shape[0] + len(extra)
            extra.append(verts[torch.tensor(list(loop), dtype=torch.int64)].mean(dim=0))
            for i, u in enumerate(loop):
                ids.append(len(triangles))
                triangles.append([u, loop[(i + 1) % len(loop)], centre_index])
        per_element.append(ids)
    new_verts = torch.stack(extra) if extra else torch.zeros(0, 3, dtype=verts.dtype)
    return torch.tensor(triangles, dtype=torch.int64), per_element, new_verts


def _parse_bodies(
    entries: Sequence[tuple[int, str]], params: dict[str, float], face_count: int
) -> tuple[dict[int, list[int]], dict[int, float]]:
    """Body -> element ids (dumps stamp orientation with a sign) and body -> declared volume."""
    bodies: dict[int, list[int]] = {}
    volumes: dict[int, float] = {}
    for line_no, line in entries:
        tokens = line.split()
        if not _INT_RE.match(tokens[0]):
            raise ValueError(f"line {line_no}: body id expected, got {tokens[0]!r}")
        body_id = int(tokens[0])
        ids: list[int] = []
        for token in tokens[1:]:
            if not _INT_RE.match(token):
                break
            ids.append(abs(int(token)))
        if not ids:
            raise ValueError(f"line {line_no}: body {body_id} lists no faces")
        if min(ids) < 1 or max(ids) > face_count:
            raise ValueError(f"line {line_no}: body {body_id} references a face outside 1..{face_count}")
        bodies[body_id] = ids
        lowered = [token.lower() for token in tokens]
        for keyword in ("volume", "volconst"):
            if keyword in lowered:
                slot = lowered.index(keyword) + 1
                if slot >= len(tokens):
                    raise ValueError(f"line {line_no}: body {body_id} has {keyword} with no value")
                volumes[body_id] = _eval_expr(tokens[slot], params, line_no)
                break
    return bodies, volumes


def _parse_periods(entries: Sequence[tuple[int, str]], params: dict[str, float]) -> torch.Tensor | None:
    """Read the `PERIODS` matrix rows (up to three vectors)."""
    rows: list[list[float]] = []
    for line_no, line in entries:
        tokens = line.split()
        if len(tokens) < 3:
            raise ValueError(f"line {line_no}: period row needs three numbers, got {tokens!r}")
        rows.append([_eval_expr(raw, params, line_no) for raw in tokens[:3]])
    if not rows:
        return None
    if len(rows) != 3:
        raise ValueError(f"expected three period vectors, got {len(rows)}")
    return torch.tensor(rows, dtype=torch.float64)


def parse_fe(path: str | Path) -> FeStructure:
    """Parse a datafile or dump into vertices, triangles and body/period bookkeeping."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sections, params, boundaries = _scan(_strip_comments(text).splitlines())
    for required in ("vertices", "edges"):
        if required not in sections:
            raise ValueError(f"{path}: required section '{required}' not found")
    face_key = "faces" if "faces" in sections else "facets" if "facets" in sections else None
    if face_key is None:
        raise ValueError(f"{path}: required section 'faces' not found")

    verts, fixed, constraints = _parse_vertices(sections["vertices"], params, boundaries)
    edges = _parse_edges(sections["edges"])
    for edge_id, (u, v) in edges.items():
        if max(u, v) >= verts.shape[0] or min(u, v) < 0:
            raise ValueError(f"edge {edge_id} references undefined vertex ({u + 1}, {v + 1})")
    loops = _parse_faces(sections[face_key], edges)
    triangles, per_element, extra = _fan_triangulate(loops, verts)
    if extra.numel():
        verts = torch.cat([verts, extra], dim=0)

    bodies: dict[int, list[int]] = {}
    volumes: dict[int, float] = {}
    if "bodies" in sections:
        raw_bodies, volumes = _parse_bodies(sections["bodies"], params, len(loops))
        bodies = {
            body: [triangle for face in members for triangle in per_element[face - 1]]
            for body, members in raw_bodies.items()
        }
    periods = _parse_periods(sections["periods"], params) if "periods" in sections else None
    return FeStructure(
        verts=verts,
        faces=triangles,
        faces_per_element=per_element,
        bodies=bodies,
        volumes=volumes,
        periods=periods,
        fixed=fixed,
        vertex_constraints=constraints,
    )


def load_fe(path: str | Path) -> Mesh:
    """Load a datafile or dump as a triangle `Mesh` (polygons are fan-triangulated)."""
    parsed = parse_fe(path)
    return Mesh(
        verts=parsed.verts,
        faces=parsed.faces,
        periods=parsed.periods,
        fixed=parsed.fixed,
        vertex_constraints=parsed.vertex_constraints,
    )


def load_dmp(path: str | Path) -> Mesh:
    """Load an SE `dump` file; dump syntax is a superset of the datafile syntax."""
    return load_fe(path)


def load_bodies(path: str | Path) -> dict[int, float]:
    """Declared volume/volconst per body id (empty when no body declares one)."""
    return parse_fe(path).volumes


def load_body_faces(path: str | Path) -> dict[int, list[int]]:
    """Triangle indices per body id, ordered as in `load_fe`'s faces."""
    return parse_fe(path).bodies


def save_fe(
    mesh: Mesh,
    path: str | Path,
    *,
    bodies: dict[int, Iterable[int]] | None = None,
    volume: float | None = None,
    read_commands: str = "",
) -> Path:
    """Write `mesh` as a triangle datafile, optionally with a `read` command section.

    Coordinates use `repr`, which round-trips float64 exactly.
    """
    edges = _edge_ids(mesh)
    out: list[str] = ["// generated by semodern.io_fe", "", "vertices"]
    for index, (x, y, z) in enumerate(mesh.verts.tolist(), start=1):
        suffix = " fixed" if index - 1 in mesh.fixed else ""
        out.append(f"{index:6d} {x!r} {y!r} {z!r}{suffix}")
    out += ["", "edges"]
    for index, (u, v) in enumerate(edges, start=1):
        out.append(f"{index:6d} {u + 1:6d} {v + 1:6d}")
    out += ["", "faces"]
    for index, triangle in enumerate(mesh.faces.tolist(), start=1):
        out.append(f"{index:6d} " + " ".join(f"{signed:6d}" for signed in _edge_loop(triangle, edges)))
    if bodies is not None:
        out += ["", "bodies"]
        for body_id, members in bodies.items():
            line = f"{body_id:6d} " + " ".join(f"{face:6d}" for face in members)
            if volume is not None:
                line += f"  volume {volume!r}"
            out.append(line)
    if read_commands:
        out += ["", "read", "", read_commands]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    return target


def _edge_ids(mesh: Mesh) -> list[tuple[int, int]]:
    seen: dict[tuple[int, int], None] = {}
    for loop in mesh.faces.tolist():
        for u, v in zip(loop, loop[1:] + loop[:1]):
            seen.setdefault((u, v) if u < v else (v, u), None)
    return list(seen)


def _edge_loop(triangle: Sequence[int], edges: Sequence[tuple[int, int]]) -> list[int]:
    lookup = {edge: index + 1 for index, edge in enumerate(edges)}
    loop: list[int] = []
    for u, v in zip(triangle, list(triangle[1:]) + [triangle[0]]):
        key = (u, v) if u < v else (v, u)
        loop.append(lookup[key] if u < v else -lookup[key])
    return loop
