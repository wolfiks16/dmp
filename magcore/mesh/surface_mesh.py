from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.constants import EPS_AREA
from magcore.domain.validation import ValidationIssue, error


@dataclass(frozen=True, slots=True)
class SurfaceMesh:
    vertices: np.ndarray  # shape (N, 3)
    faces: np.ndarray     # shape (M, 3)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def face_vertices(self, face_idx: int) -> np.ndarray:
        return self.vertices[self.faces[face_idx]]

    def face_area(self, face_idx: int) -> float:
        tri = self.face_vertices(face_idx)
        e1 = tri[1] - tri[0]
        e2 = tri[2] - tri[0]
        return 0.5 * float(np.linalg.norm(np.cross(e1, e2)))

    def face_centroid(self, face_idx: int) -> np.ndarray:
        tri = self.face_vertices(face_idx)
        return tri.mean(axis=0)

    def face_normal(self, face_idx: int) -> np.ndarray:
        tri = self.face_vertices(face_idx)
        e1 = tri[1] - tri[0]
        e2 = tri[2] - tri[0]
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n)
        if norm <= 0.0:
            raise ValueError(f"Degenerate face {face_idx} has zero normal.")
        return n / norm

    def validate_basic(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []

        if not isinstance(self.vertices, np.ndarray):
            issues.append(error("mesh.vertices.type", "vertices must be a numpy.ndarray."))
            return tuple(issues)

        if not isinstance(self.faces, np.ndarray):
            issues.append(error("mesh.faces.type", "faces must be a numpy.ndarray."))
            return tuple(issues)

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            issues.append(error("mesh.vertices.shape", "vertices must have shape (N, 3).", shape=getattr(self.vertices, "shape", None)))

        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            issues.append(error("mesh.faces.shape", "faces must have shape (M, 3).", shape=getattr(self.faces, "shape", None)))

        if issues:
            return tuple(issues)

        if self.n_vertices == 0:
            issues.append(error("mesh.vertices.empty", "Mesh must contain at least one vertex."))
        if self.n_faces == 0:
            issues.append(error("mesh.faces.empty", "Mesh must contain at least one face."))

        if not np.isfinite(self.vertices).all():
            issues.append(error("mesh.vertices.nonfinite", "Mesh vertices contain NaN or Inf."))

        if not np.issubdtype(self.faces.dtype, np.integer):
            issues.append(error("mesh.faces.dtype", "Mesh faces must have integer dtype.", dtype=str(self.faces.dtype)))

        if np.min(self.faces) < 0 or np.max(self.faces) >= self.n_vertices:
            issues.append(error("mesh.faces.out_of_range", "Mesh faces contain invalid vertex indices."))

        if issues:
            return tuple(issues)

        for face_idx in range(self.n_faces):
            area = self.face_area(face_idx)
            if area <= EPS_AREA:
                issues.append(error("mesh.face.degenerate", "Mesh contains a degenerate triangle face.", face_idx=face_idx, area=area))

        return tuple(issues)