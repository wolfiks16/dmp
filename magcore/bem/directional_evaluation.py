from __future__ import annotations

import numpy as np

from magcore.bem.element_integrals import (
    integrate_over_triangle_regular,
    triangle_vertices,
)
from magcore.bem.normal_derivative_kernels import laplace_dgreen_dn_x
from magcore.bem.quadrature import get_triangle_quadrature
from magcore.mesh.surface_mesh import SurfaceMesh


def normalize_directions(directions: np.ndarray) -> np.ndarray:
    """
    Normalize an array of direction vectors of shape (N, 3).
    """
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (N, 3).")

    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("direction vectors must be nonzero.")

    return directions / norms[:, None]


def single_layer_directional_derivative_regular(
    target_point: np.ndarray,
    target_direction: np.ndarray,
    source_tri: np.ndarray,
    quadrature_order: int = 2,
) -> float:
    """
    Evaluate the directional derivative with respect to the target point x:

        ∂_{d_x} S[chi_j](x)
        = ∫_{T_j} grad_x G(x, y) · d_x dS_y

    Notes
    -----
    We reuse laplace_dgreen_dn_x(...) because mathematically it computes
    grad_x G · v for an arbitrary direction vector v, not only for a true normal.
    """
    q = get_triangle_quadrature(quadrature_order)
    d = np.asarray(target_direction, dtype=float)
    if d.shape != (3,):
        raise ValueError("target_direction must have shape (3,).")

    nrm = float(np.linalg.norm(d))
    if nrm == 0.0:
        raise ValueError("target_direction must be nonzero.")
    d = d / nrm

    return integrate_over_triangle_regular(
        source_tri,
        lambda y: laplace_dgreen_dn_x(target_point, y, d),
        q,
    )


def evaluate_single_layer_directional_derivative_p0(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    density: np.ndarray,
    target_points: np.ndarray,
    target_directions: np.ndarray,
    quadrature_order: int = 2,
) -> np.ndarray:
    """
    Evaluate the directional derivative of a P0 single-layer potential:

        ∂_{d_x} S[sigma](x)
        = sum_j sigma_j ∫_{T_j} grad_x G(x, y) · d_x dS_y

    Parameters
    ----------
    mesh:
        Source mesh
    face_indices:
        Source face indices
    density:
        Shape (N_faces,)
    target_points:
        Shape (N_targets, 3)
    target_directions:
        Shape (N_targets, 3)

    Returns
    -------
    values:
        Shape (N_targets,)
    """
    face_indices = tuple(sorted(face_indices))
    density = np.asarray(density, dtype=float)
    target_points = np.asarray(target_points, dtype=float)
    target_directions = np.asarray(target_directions, dtype=float)

    if density.shape != (len(face_indices),):
        raise ValueError("density shape must match number of source faces.")
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError("target_points must have shape (N, 3).")
    if target_directions.ndim != 2 or target_directions.shape != target_points.shape:
        raise ValueError("target_directions must have the same shape as target_points.")

    dirs = normalize_directions(target_directions)
    out = np.zeros(target_points.shape[0], dtype=float)

    for p_idx, (x, d_x) in enumerate(zip(target_points, dirs, strict=False)):
        val = 0.0
        for local_f, face_idx in enumerate(face_indices):
            tri = triangle_vertices(mesh, face_idx)
            val += density[local_f] * single_layer_directional_derivative_regular(
                target_point=x,
                target_direction=d_x,
                source_tri=tri,
                quadrature_order=quadrature_order,
            )
        out[p_idx] = val

    return out


def estimate_zone_H_parallel_from_faces(
    directional_derivative_values: np.ndarray,
) -> float:
    """
    Convert sampled directional derivatives of the scalar potential into
    a zonewise operating field estimate along the easy axis.

    Since:
        H = -grad(phi)
    we use:
        H_parallel = - mean( d(phi)/d(a) )
    """
    vals = np.asarray(directional_derivative_values, dtype=float)
    if vals.ndim != 1:
        raise ValueError("directional_derivative_values must be a 1D array.")
    if len(vals) == 0:
        raise ValueError("directional_derivative_values must be nonempty.")
    if not np.isfinite(vals).all():
        raise ValueError("directional_derivative_values must be finite.")

    return float(-np.mean(vals))