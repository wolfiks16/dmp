from __future__ import annotations

import numpy as np

from magcore.bem.element_integrals import (
    single_layer_point_potential_regular,
    triangle_vertices,
)
from magcore.bem.quadrature import get_triangle_quadrature
from magcore.mesh.surface_mesh import SurfaceMesh


def evaluate_single_layer_potential_p0(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    density: np.ndarray,
    target_points: np.ndarray,
    quadrature_order: int = 2,
) -> np.ndarray:
    """
    Evaluate off-surface single-layer potential at target points for a P0 surface density.

    Potential:
        phi(x) = sum_j sigma_j * ∫_{T_j} G(x, y) dS_y

    Parameters
    ----------
    mesh:
        Surface mesh containing source faces.
    face_indices:
        Tuple of source face indices. The density vector is assumed to be aligned
        with this ordering after sorting.
    density:
        Array of shape (n_faces_selected,).
    target_points:
        Array of shape (N, 3).
    quadrature_order:
        Triangle quadrature order for regular integration.

    Notes
    -----
    At R1.4 this routine assumes target points are sufficiently far from source
    triangles so that regular quadrature is acceptable.
    """
    face_indices = tuple(sorted(face_indices))
    density = np.asarray(density, dtype=float)
    target_points = np.asarray(target_points, dtype=float)

    if density.shape != (len(face_indices),):
        raise ValueError("density shape must match number of source faces.")
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError("target_points must have shape (N, 3).")

    q = get_triangle_quadrature(quadrature_order)
    out = np.zeros(target_points.shape[0], dtype=float)

    for p_idx, x in enumerate(target_points):
        val = 0.0
        for local_f, face_idx in enumerate(face_indices):
            tri = triangle_vertices(mesh, face_idx)
            val += density[local_f] * single_layer_point_potential_regular(
                target_point=x,
                source_tri=tri,
                quadrature=q,
            )
        out[p_idx] = val

    return out