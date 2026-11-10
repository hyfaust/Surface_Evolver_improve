"""Metrics, comparisons and JSON persistence helpers for the benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from . import mesh as mesh_mod
from .mesh import Mesh

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT_ROOT / "reports"
DATA_SE = PROJECT_ROOT / "data" / "se"
TMP = PROJECT_ROOT / "tmp"


def metrics(mesh: Mesh) -> dict:
    """Area, signed volume, element counts and Euler characteristic."""
    edges, _ = mesh_mod.edge_table(mesh)
    return {
        "area": float(mesh_mod.area(mesh)),
        "volume": float(mesh_mod.volume(mesh)),
        "vertices": mesh.n_verts,
        "edges": int(edges.shape[0]),
        "faces": mesh.n_faces,
        "euler": mesh_mod.euler_characteristic(mesh),
        "body_count": 1,
        "periodic": mesh.periods is not None,
    }


def compare(a: float, b: float, rtol: float = 1e-9, atol: float = 1e-12) -> dict:
    """Relative/absolute agreement of two scalars."""
    difference = abs(a - b)
    scale = max(abs(a), abs(b), 1.0)
    return {
        "a": float(a),
        "b": float(b),
        "abs_diff": float(difference),
        "rel_diff": float(difference / scale),
        "rtol": rtol,
        "atol": atol,
        "passed": bool(difference <= atol + rtol * scale),
    }


def energy_delta(before: float, after: float, *, tol: float = 0.0) -> dict:
    """Energy change report; `passed` means the energy did not rise above `tol`."""
    delta = after - before
    return {
        "before": float(before),
        "after": float(after),
        "delta": float(delta),
        "passed": bool(delta <= tol),
    }


def grad_norm(grad: torch.Tensor) -> float:
    """Infinity norm of a gradient tensor."""
    return float(grad.abs().max()) if grad.numel() else 0.0


def write_json(path: str | Path, payload: dict) -> Path:
    """Write `payload` as pretty JSON, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def append_case(path: str | Path, case: dict) -> Path:
    """Append `case` to the `cases` list of a JSON report file."""
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {"cases": []}
    payload.setdefault("cases", []).append(case)
    return write_json(target, payload)
