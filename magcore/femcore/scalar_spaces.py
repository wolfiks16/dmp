from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.femcore.mesh import TetraMesh


@dataclass(frozen=True, slots=True)
class LagrangeP1Space:
    mesh: TetraMesh

    @property
    def ndofs(self) -> int:
        return self.mesh.n_vertices

    @property
    def cell_to_global_vertices(self) -> np.ndarray:
        return np.asarray(self.mesh.cells, dtype=int)

    def cell_dof_indices(self, cell_idx: int) -> np.ndarray:
        return np.asarray(self.mesh.cells[int(cell_idx)], dtype=int)

    def boundary_vertex_indices(self) -> tuple[int, ...]:
        verts: set[int] = set()
        for face in self.mesh.boundary_faces():
            verts.update(int(v) for v in face)
        return tuple(sorted(verts))

    def boundary_dofs(self) -> tuple[int, ...]:
        return self.boundary_vertex_indices()
