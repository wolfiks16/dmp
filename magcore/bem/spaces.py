from __future__ import annotations

from dataclasses import dataclass
from abc import ABC
import numpy as np

from magcore.enums import BasisKind
from magcore.mesh.surface_mesh import SurfaceMesh


@dataclass(frozen=True, slots=True)
class DiscreteSpace(ABC):
    name: str
    basis_kind: BasisKind
    ndofs: int


@dataclass(frozen=True, slots=True)
class FaceP0Space(DiscreteSpace):
    mesh: SurfaceMesh
    face_indices: tuple[int, ...]
    dof_to_face: tuple[int, ...]
    face_to_dof: dict[int, int]

    @classmethod
    def from_faces(
        cls,
        mesh: SurfaceMesh,
        face_indices: tuple[int, ...],
        name: str = "flux_trace_space",
    ) -> "FaceP0Space":
        face_indices = tuple(sorted(face_indices))
        face_to_dof = {f: i for i, f in enumerate(face_indices)}
        return cls(
            name=name,
            basis_kind=BasisKind.P0,
            ndofs=len(face_indices),
            mesh=mesh,
            face_indices=face_indices,
            dof_to_face=face_indices,
            face_to_dof=face_to_dof,
        )


@dataclass(frozen=True, slots=True)
class VertexP1Space(DiscreteSpace):
    mesh: SurfaceMesh
    face_indices: tuple[int, ...]
    active_vertices: tuple[int, ...]
    dof_to_vertex: tuple[int, ...]
    vertex_to_dof: dict[int, int]

    @classmethod
    def from_faces(
        cls,
        mesh: SurfaceMesh,
        face_indices: tuple[int, ...],
        name: str = "phi_trace_space",
    ) -> "VertexP1Space":
        face_indices = tuple(sorted(face_indices))
        vertices: set[int] = set()
        for f in face_indices:
            face = mesh.faces[f]
            vertices.update(int(v) for v in face)

        active_vertices = tuple(sorted(vertices))
        vertex_to_dof = {v: i for i, v in enumerate(active_vertices)}

        return cls(
            name=name,
            basis_kind=BasisKind.P1,
            ndofs=len(active_vertices),
            mesh=mesh,
            face_indices=face_indices,
            active_vertices=active_vertices,
            dof_to_vertex=active_vertices,
            vertex_to_dof=vertex_to_dof,
        )