from __future__ import annotations

from dataclasses import dataclass
import numpy as np

REFERENCE_TETRA_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
)


def reference_barycentric(point: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=float)
    if p.shape != (3,):
        raise ValueError("point must have shape (3,).")
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    return np.array([1.0 - x - y - z, x, y, z], dtype=float)


def reference_barycentric_gradients() -> np.ndarray:
    return np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)


@dataclass(frozen=True, slots=True)
class AffineTetraMap:
    physical_vertices: np.ndarray

    def __post_init__(self) -> None:
        verts = np.asarray(self.physical_vertices, dtype=float)
        object.__setattr__(self, "physical_vertices", verts)
        if verts.shape != (4, 3):
            raise ValueError("physical_vertices must have shape (4, 3).")
        if not np.isfinite(verts).all():
            raise ValueError("physical_vertices must be finite.")
        if self.jacobian_determinant() <= 0.0:
            raise ValueError("Affine tetra map requires positively oriented physical tetrahedron.")

    @property
    def v0(self) -> np.ndarray:
        return self.physical_vertices[0]

    def jacobian_matrix(self) -> np.ndarray:
        v0, v1, v2, v3 = self.physical_vertices
        return np.column_stack((v1 - v0, v2 - v0, v3 - v0))

    def jacobian_determinant(self) -> float:
        return float(np.linalg.det(self.jacobian_matrix()))

    def inverse_jacobian(self) -> np.ndarray:
        return np.linalg.inv(self.jacobian_matrix())

    def inverse_transpose_jacobian(self) -> np.ndarray:
        return self.inverse_jacobian().T

    def volume(self) -> float:
        return self.jacobian_determinant() / 6.0

    def map_to_physical(self, ref_point: np.ndarray) -> np.ndarray:
        xi = np.asarray(ref_point, dtype=float)
        if xi.shape != (3,):
            raise ValueError("ref_point must have shape (3,).")
        return self.v0 + self.jacobian_matrix() @ xi

    def map_to_reference(self, physical_point: np.ndarray) -> np.ndarray:
        x = np.asarray(physical_point, dtype=float)
        if x.shape != (3,):
            raise ValueError("physical_point must have shape (3,).")
        return self.inverse_jacobian() @ (x - self.v0)

    def barycentric_at_physical_point(self, physical_point: np.ndarray) -> np.ndarray:
        xi = self.map_to_reference(physical_point)
        return reference_barycentric(xi)

    def physical_barycentric_gradients(self) -> np.ndarray:
        grads_ref = reference_barycentric_gradients()
        JTinv = self.inverse_transpose_jacobian()
        return grads_ref @ JTinv.T
