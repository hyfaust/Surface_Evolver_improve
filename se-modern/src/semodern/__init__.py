"""A modern Python prototype for explicit-surface foam/capillary problems.

The package treats Surface Evolver as a reference implementation: `se_bridge`
runs it as a baseline, `io_fe` reads its datafiles and dumps, and the numerical
modules (`mesh`, `energy`, `optimize`, `constraints`, `topology`, `phasefield`,
`meshfree`) provide the replacements under test.

Submodules are resolved lazily so that importing the package never requires a
module a branch has not landed yet; `import semodern.optimize` still works.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .mesh import Mesh

__version__ = "0.1.0"

_SUBMODULES = (
    "constraints",
    "energy",
    "io_fe",
    "mesh",
    "meshfree",
    "metrics",
    "optimize",
    "phasefield",
    "se_bridge",
    "topology",
)

__all__ = ["Mesh", *_SUBMODULES]

if TYPE_CHECKING:  # pragma: no cover - import-cycle-free type checking only
    from . import (  # noqa: F401
        constraints,
        energy,
        io_fe,
        mesh,
        meshfree,
        metrics,
        optimize,
        phasefield,
        se_bridge,
        topology,
    )


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
