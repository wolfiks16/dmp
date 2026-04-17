from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.femcore.edge_topology import EdgeTopology, boundary_edges, build_edge_topology
from magcore.mesh.mesh import TetraMesh


@dataclass(frozen=True, slots=True)
class NedelecP1Space:
    mesh: TetraMesh
    edge_topology: EdgeTopology

    @classmethod
    def from_mesh(cls, mesh: TetraMesh) -> "NedelecP1Space":
        return cls(mesh=mesh, edge_topology=build_edge_topology(mesh))

    @property
    def global_edges(self) -> tuple[tuple[int, int], ...]:
        return self.edge_topology.global_edges

    @property
    def cell_to_global_edges(self) -> np.ndarray:
        return self.edge_topology.cell_to_global_edges

    @property
    def cell_edge_signs(self) -> np.ndarray:
        return self.edge_topology.cell_edge_signs

    @property
    def ndofs(self) -> int:
        return self.edge_topology.n_edges

    def edge_to_dof_map(self) -> dict[tuple[int, int], int]:
        return {edge: i for i, edge in enumerate(self.global_edges)}

    def cell_dof_indices(self, cell_idx: int) -> np.ndarray:
        return np.asarray(self.cell_to_global_edges[int(cell_idx)], dtype=int)

    def cell_dof_signs(self, cell_idx: int) -> np.ndarray:
        return np.asarray(self.cell_edge_signs[int(cell_idx)], dtype=int)

    def boundary_dofs(self) -> tuple[int, ...]:
        b_edges = boundary_edges(self.mesh)
        edge_to_dof = self.edge_to_dof_map()
        dofs = [edge_to_dof[e] for e in b_edges]
        return tuple(sorted(dofs))