from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.fem2d.mesh import TriangleMesh


@dataclass(frozen=True, slots=True)
class LagrangeP1Space2D:
    """Узловое пространство P1 на треугольной сетке (dofs = вершины). 2D-аналог `LagrangeP1Space`."""

    mesh: TriangleMesh

    @property
    def ndofs(self) -> int:
        return self.mesh.n_vertices

    @property
    def cell_to_global_vertices(self) -> np.ndarray:
        return np.asarray(self.mesh.cells, dtype=int)

    def cell_dof_indices(self, cell_idx: int) -> np.ndarray:
        return np.asarray(self.mesh.cells[int(cell_idx)], dtype=int)

    def boundary_dofs(self) -> tuple[int, ...]:
        return self.mesh.boundary_vertices()
