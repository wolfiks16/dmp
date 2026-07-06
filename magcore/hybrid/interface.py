from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.bem.spaces import FaceP0Space, VertexP1Space
from magcore.mesh.mesh import TetraMesh
from magcore.mesh.surface_mesh import SurfaceMesh


@dataclass(frozen=True, slots=True)
class CouplingInterface:
    """
    Геометрический мост FEM(объём)↔BEM(поверхность) для фазы B (см. docs/math/coupling.md).

    Извлекает интерфейс Γ = ∂Ω из тетраэдральной сетки Ω и строит:
      - surface_mesh: SurfaceMesh граней Γ, ориентированных так, что
        SurfaceMesh.face_normal(f) совпадает с ВНЕШНЕЙ нормалью из Ω (решающе для знаков связи);
      - face_to_cell: для каждой грани Γ — индекс смежной (владеющей) ячейки FEM;
      - outward_normals, face_areas: внешние нормали и площади граней Γ;
      - phi_space (P1 по вершинам Γ) — след ψ; flux_space (P0 по граням Γ) — след λ=∂ψ/∂n;
      - surface_to_global_vertex: вершина Γ → глобальная вершина тетра-сетки.
    """

    tetra_mesh: TetraMesh
    surface_mesh: SurfaceMesh
    face_to_cell: np.ndarray              # (n_faces,) int
    surface_to_global_vertex: np.ndarray  # (n_surf_verts,) int
    outward_normals: np.ndarray           # (n_faces, 3) float
    face_areas: np.ndarray                # (n_faces,) float
    phi_space: VertexP1Space
    flux_space: FaceP0Space

    @property
    def n_faces(self) -> int:
        return int(self.surface_mesh.n_faces)

    @property
    def n_phi_dofs(self) -> int:
        return int(self.phi_space.ndofs)

    @property
    def n_flux_dofs(self) -> int:
        return int(self.flux_space.ndofs)

    def global_to_surface_vertex(self) -> dict[int, int]:
        return {int(g): i for i, g in enumerate(self.surface_to_global_vertex)}

    @classmethod
    def from_tetra_mesh(cls, tetra_mesh: TetraMesh) -> "CouplingInterface":
        # Владелец и противоположная вершина для каждой грани (по канонической грани).
        face_owner: dict[tuple[int, int, int], int] = {}
        face_opp: dict[tuple[int, int, int], int] = {}
        for c in range(tetra_mesh.n_cells):
            cv = tetra_mesh.cell_vertex_indices(c)
            for face in tetra_mesh.cell_faces(c):
                opp = next(v for v in cv if v not in face)
                face_owner[face] = c
                face_opp[face] = opp

        bfaces = tetra_mesh.boundary_faces()
        verts = tetra_mesh.vertices

        b_verts = sorted({int(v) for f in bfaces for v in f})
        g2s = {g: i for i, g in enumerate(b_verts)}
        surf_vertices = verts[b_verts]

        surf_faces: list[list[int]] = []
        face_to_cell: list[int] = []
        normals: list[np.ndarray] = []
        areas: list[float] = []

        for f in bfaces:
            i, j, k = f
            c = face_owner[f]
            opp = face_opp[f]
            pi, pj, pk, po = verts[i], verts[j], verts[k], verts[opp]
            n = np.cross(pj - pi, pk - pi)
            centroid = (pi + pj + pk) / 3.0
            if float(np.dot(n, centroid - po)) < 0.0:
                tri = (i, k, j)  # развернуть обход → внешняя нормаль
                n = -n
            else:
                tri = (i, j, k)
            nrm = float(np.linalg.norm(n))
            normals.append(n / nrm)
            areas.append(0.5 * nrm)
            surf_faces.append([g2s[tri[0]], g2s[tri[1]], g2s[tri[2]]])
            face_to_cell.append(c)

        surface_mesh = SurfaceMesh(
            vertices=np.asarray(surf_vertices, dtype=float),
            faces=np.asarray(surf_faces, dtype=int),
        )
        issues = surface_mesh.validate_basic()
        if issues:
            raise ValueError(f"Coupling surface mesh invalid: {issues}")

        all_faces = tuple(range(surface_mesh.n_faces))
        phi_space = VertexP1Space.from_faces(surface_mesh, all_faces, name="coupling_phi")
        flux_space = FaceP0Space.from_faces(surface_mesh, all_faces, name="coupling_flux")

        return cls(
            tetra_mesh=tetra_mesh,
            surface_mesh=surface_mesh,
            face_to_cell=np.asarray(face_to_cell, dtype=int),
            surface_to_global_vertex=np.asarray(b_verts, dtype=int),
            outward_normals=np.asarray(normals, dtype=float),
            face_areas=np.asarray(areas, dtype=float),
            phi_space=phi_space,
            flux_space=flux_space,
        )
