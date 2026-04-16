from __future__ import annotations

import numpy as np

from magcore.bem.element_integrals import (
    integrate_over_triangle_regular,
    triangle_area,
    triangle_vertices,
)
from magcore.bem.normal_derivative_kernels import laplace_dgreen_dn_x
from magcore.bem.quadrature import get_triangle_quadrature
from magcore.mesh.surface_mesh import SurfaceMesh


def face_unit_normals(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> np.ndarray:
    """
    Return unit normals for the selected faces in the given ordering.
    """
    face_indices = tuple(sorted(face_indices))
    return np.array([mesh.face_normal(f) for f in face_indices], dtype=float)


def face_characteristic_lengths(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> np.ndarray:
    """
    Characteristic face size based on area:
        h_i = sqrt(area_i)
    """
    face_indices = tuple(sorted(face_indices))
    areas = np.array([mesh.face_area(f) for f in face_indices], dtype=float)
    return np.sqrt(areas)


def face_centroids(mesh: SurfaceMesh, face_indices: tuple[int, ...]) -> np.ndarray:
    """
    Return face centroids in the given ordering.
    """
    face_indices = tuple(sorted(face_indices))
    return np.array([mesh.face_centroid(f) for f in face_indices], dtype=float)


def offset_face_centroids(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    offset_factor: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build collocation points slightly inside and outside the surface:

        x_i^- = c_i - delta_i n_i
        x_i^+ = c_i + delta_i n_i

    where:
        delta_i = offset_factor * sqrt(area_i)

    Returns
    -------
    centroids:
        shape (N, 3)
    inner_points:
        shape (N, 3)
    outer_points:
        shape (N, 3)
    """
    if offset_factor <= 0.0:
        raise ValueError("offset_factor must be positive.")

    face_indices = tuple(sorted(face_indices))
    ctrs = face_centroids(mesh, face_indices)
    normals = face_unit_normals(mesh, face_indices)
    h = face_characteristic_lengths(mesh, face_indices)

    delta = offset_factor * h
    inner_points = ctrs - delta[:, None] * normals
    outer_points = ctrs + delta[:, None] * normals

    return ctrs, inner_points, outer_points


def single_layer_normal_derivative_regular(
    target_point: np.ndarray,
    target_normal: np.ndarray,
    source_tri: np.ndarray,
    quadrature_order: int = 2,
) -> float:
    """
    Evaluate:
        ∫_{T_j} dG/dn_x (x, y) dS_y

    for a target point x and target normal n_x, assuming a regular configuration.
    """
    q = get_triangle_quadrature(quadrature_order)

    return integrate_over_triangle_regular(
        source_tri,
        lambda y: laplace_dgreen_dn_x(target_point, y, target_normal),
        q,
    )


def evaluate_single_layer_normal_derivative_p0(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    density: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    quadrature_order: int = 2,
) -> np.ndarray:
    """
    Evaluate the normal derivative of a P0 single-layer potential at target points:

        d/dn_x S[sigma](x)
        = sum_j sigma_j ∫_{T_j} dG/dn_x(x, y) dS_y

    Parameters
    ----------
    mesh:
        Source mesh
    face_indices:
        Source face indices
    density:
        Array of shape (N_faces,)
    target_points:
        Array of shape (N_targets, 3)
    target_normals:
        Array of shape (N_targets, 3)
    quadrature_order:
        Regular quadrature order

    Notes
    -----
    This routine assumes target points are offset from the surface enough
    that regular quadrature is acceptable.
    """
    face_indices = tuple(sorted(face_indices))
    density = np.asarray(density, dtype=float)
    target_points = np.asarray(target_points, dtype=float)
    target_normals = np.asarray(target_normals, dtype=float)

    if density.shape != (len(face_indices),):
        raise ValueError("density shape must match number of source faces.")
    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError("target_points must have shape (N, 3).")
    if target_normals.ndim != 2 or target_normals.shape != target_points.shape:
        raise ValueError("target_normals must have the same shape as target_points.")

    out = np.zeros(target_points.shape[0], dtype=float)

    for p_idx, (x, n_x) in enumerate(zip(target_points, target_normals, strict=False)):
        val = 0.0
        for local_f, face_idx in enumerate(face_indices):
            tri = triangle_vertices(mesh, face_idx)
            val += density[local_f] * single_layer_normal_derivative_regular(
                target_point=x,
                target_normal=n_x,
                source_tri=tri,
                quadrature_order=quadrature_order,
            )
        out[p_idx] = val

    return out


def assemble_single_layer_normal_trace_matrix_offset(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    target_points: np.ndarray,
    target_normals: np.ndarray,
    quadrature_order: int = 2,
) -> np.ndarray:
    """
    Assemble collocation matrix A for the normal derivative trace of single-layer potential:

        A[i, j] ≈ ∫_{T_j} dG/dn_x (x_i, y) dS_y

    with x_i taken as offset target points.
    """
    face_indices = tuple(sorted(face_indices))
    target_points = np.asarray(target_points, dtype=float)
    target_normals = np.asarray(target_normals, dtype=float)

    if target_points.ndim != 2 or target_points.shape[1] != 3:
        raise ValueError("target_points must have shape (N, 3).")
    if target_normals.ndim != 2 or target_normals.shape != target_points.shape:
        raise ValueError("target_normals must have the same shape as target_points.")

    n_targets = target_points.shape[0]
    n_faces = len(face_indices)

    A = np.zeros((n_targets, n_faces), dtype=float)

    for i, (x, n_x) in enumerate(zip(target_points, target_normals, strict=False)):
        for j, face_idx in enumerate(face_indices):
            tri = triangle_vertices(mesh, face_idx)
            A[i, j] = single_layer_normal_derivative_regular(
                target_point=x,
                target_normal=n_x,
                source_tri=tri,
                quadrature_order=quadrature_order,
            )

    return A