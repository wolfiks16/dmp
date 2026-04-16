from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from magcore.bem.normal_evaluation import (
    assemble_single_layer_normal_trace_matrix_offset,
    face_unit_normals,
    offset_face_centroids,
)
from magcore.mesh.surface_mesh import SurfaceMesh


@dataclass(frozen=True, slots=True)
class OffsetTraceConfig:
    quadrature_order: int = 2
    offset_factor: float = 0.05


@dataclass(slots=True)
class OffsetTraceMatrices:
    face_indices: tuple[int, ...]
    centroids: np.ndarray
    inner_points: np.ndarray
    outer_points: np.ndarray
    normals: np.ndarray
    A_minus: np.ndarray
    A_plus: np.ndarray
    Kt_avg: np.ndarray
    jump: np.ndarray
    metadata: dict


def assemble_single_layer_normal_trace_matrices(
    mesh: SurfaceMesh,
    face_indices: tuple[int, ...],
    offset_config: OffsetTraceConfig | None = None,
) -> OffsetTraceMatrices:
    """
    Assemble offset-collocation approximations for the normal traces of a
    single-layer potential on selected faces.

    Definitions:
        A_minus[i, j] ≈ ∫_{T_j} dG/dn_x (x_i^-, y) dS_y
        A_plus [i, j] ≈ ∫_{T_j} dG/dn_x (x_i^+, y) dS_y

    Then define:
        Kt_avg ≈ 0.5 * (A_minus + A_plus)
        jump   ≈ A_minus - A_plus

    Notes
    -----
    This is an offset-trace collocation surrogate, not a final singular on-surface
    trace discretization.
    """
    if offset_config is None:
        offset_config = OffsetTraceConfig()

    face_indices = tuple(sorted(face_indices))
    normals = face_unit_normals(mesh, face_indices)
    centroids, inner_points, outer_points = offset_face_centroids(
        mesh=mesh,
        face_indices=face_indices,
        offset_factor=offset_config.offset_factor,
    )

    A_minus = assemble_single_layer_normal_trace_matrix_offset(
        mesh=mesh,
        face_indices=face_indices,
        target_points=inner_points,
        target_normals=normals,
        quadrature_order=offset_config.quadrature_order,
    )
    A_plus = assemble_single_layer_normal_trace_matrix_offset(
        mesh=mesh,
        face_indices=face_indices,
        target_points=outer_points,
        target_normals=normals,
        quadrature_order=offset_config.quadrature_order,
    )

    Kt_avg = 0.5 * (A_minus + A_plus)
    jump = A_minus - A_plus

    return OffsetTraceMatrices(
        face_indices=face_indices,
        centroids=centroids,
        inner_points=inner_points,
        outer_points=outer_points,
        normals=normals,
        A_minus=A_minus,
        A_plus=A_plus,
        Kt_avg=Kt_avg,
        jump=jump,
        metadata={
            "quadrature_order": offset_config.quadrature_order,
            "offset_factor": offset_config.offset_factor,
            "n_faces": len(face_indices),
            "backend": "offset_trace_collocation",
        },
    )


def apply_offset_trace_jump(
    trace_mats: OffsetTraceMatrices,
    density: np.ndarray,
) -> np.ndarray:
    """
    Apply numerical jump surrogate to a P0 density:
        jump @ density
    """
    density = np.asarray(density, dtype=float)
    n = len(trace_mats.face_indices)
    if density.shape != (n,):
        raise ValueError("density shape must match number of selected faces.")

    return trace_mats.jump @ density


def apply_offset_trace_avg(
    trace_mats: OffsetTraceMatrices,
    density: np.ndarray,
) -> np.ndarray:
    """
    Apply numerical K'-average surrogate:
        Kt_avg @ density
    """
    density = np.asarray(density, dtype=float)
    n = len(trace_mats.face_indices)
    if density.shape != (n,):
        raise ValueError("density shape must match number of selected faces.")

    return trace_mats.Kt_avg @ density