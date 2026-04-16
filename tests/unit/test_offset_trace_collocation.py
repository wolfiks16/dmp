from __future__ import annotations

import numpy as np

from magcore.bem.adjoint_trace_collocation import (
    OffsetTraceConfig,
    apply_offset_trace_avg,
    apply_offset_trace_jump,
    assemble_single_layer_normal_trace_matrices,
)
from magcore.mesh.surface_mesh import SurfaceMesh


def make_tetra_surface_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_assemble_single_layer_normal_trace_matrices_shapes():
    mesh = make_tetra_surface_mesh()
    cfg = OffsetTraceConfig(quadrature_order=2, offset_factor=0.05)

    trace = assemble_single_layer_normal_trace_matrices(
        mesh=mesh,
        face_indices=(0, 1, 2, 3),
        offset_config=cfg,
    )

    assert trace.centroids.shape == (4, 3)
    assert trace.inner_points.shape == (4, 3)
    assert trace.outer_points.shape == (4, 3)
    assert trace.normals.shape == (4, 3)
    assert trace.A_minus.shape == (4, 4)
    assert trace.A_plus.shape == (4, 4)
    assert trace.Kt_avg.shape == (4, 4)
    assert trace.jump.shape == (4, 4)


def test_offset_trace_matrices_are_finite():
    mesh = make_tetra_surface_mesh()
    cfg = OffsetTraceConfig(quadrature_order=2, offset_factor=0.05)

    trace = assemble_single_layer_normal_trace_matrices(
        mesh=mesh,
        face_indices=(0, 1, 2, 3),
        offset_config=cfg,
    )

    assert np.isfinite(trace.A_minus).all()
    assert np.isfinite(trace.A_plus).all()
    assert np.isfinite(trace.Kt_avg).all()
    assert np.isfinite(trace.jump).all()


def test_offset_trace_jump_is_nontrivial():
    mesh = make_tetra_surface_mesh()
    cfg = OffsetTraceConfig(quadrature_order=2, offset_factor=0.05)

    trace = assemble_single_layer_normal_trace_matrices(
        mesh=mesh,
        face_indices=(0, 1, 2, 3),
        offset_config=cfg,
    )

    density = np.ones(4, dtype=float)
    jump_vals = apply_offset_trace_jump(trace, density)

    assert jump_vals.shape == (4,)
    assert np.isfinite(jump_vals).all()
    assert np.linalg.norm(jump_vals) > 0.0


def test_offset_trace_avg_is_finite():
    mesh = make_tetra_surface_mesh()
    cfg = OffsetTraceConfig(quadrature_order=2, offset_factor=0.05)

    trace = assemble_single_layer_normal_trace_matrices(
        mesh=mesh,
        face_indices=(0, 1, 2, 3),
        offset_config=cfg,
    )

    density = np.array([1.0, -0.5, 0.25, 0.75], dtype=float)
    vals = apply_offset_trace_avg(trace, density)

    assert vals.shape == (4,)
    assert np.isfinite(vals).all()


def test_offset_trace_jump_and_avg_check_density_shape():
    mesh = make_tetra_surface_mesh()
    cfg = OffsetTraceConfig(quadrature_order=2, offset_factor=0.05)

    trace = assemble_single_layer_normal_trace_matrices(
        mesh=mesh,
        face_indices=(0, 1, 2, 3),
        offset_config=cfg,
    )

    try:
        apply_offset_trace_jump(trace, np.ones(3, dtype=float))
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for wrong jump density shape."

    try:
        apply_offset_trace_avg(trace, np.ones(3, dtype=float))
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for wrong avg density shape."


def test_offset_trace_inner_outer_matrices_are_not_identical():
    mesh = make_tetra_surface_mesh()
    cfg = OffsetTraceConfig(quadrature_order=2, offset_factor=0.05)

    trace = assemble_single_layer_normal_trace_matrices(
        mesh=mesh,
        face_indices=(0, 1, 2, 3),
        offset_config=cfg,
    )

    diff = np.linalg.norm(trace.A_minus - trace.A_plus, ord="fro")
    assert diff > 0.0