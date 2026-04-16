from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import (
    AdaptiveIntegrationConfig,
    assemble_single_layer_p0p0_full,
    single_layer_face_face_full,
)
from magcore.mesh.surface_mesh import SurfaceMesh


def make_convergence_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],   # 0
            [1.0, 0.0, 0.0],   # 1
            [0.0, 1.0, 0.0],   # 2
            [1.0, 1.0, 0.0],   # 3
            [2.0, 0.0, 0.0],   # 4
            [0.0, 0.0, 5.0],   # 5
            [1.0, 0.0, 5.0],   # 6
            [0.0, 1.0, 5.0],   # 7
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],  # 0
            [1, 3, 2],  # 1 shared-edge with 0
            [1, 4, 3],  # 2 shared-vertex with 0
            [5, 6, 7],  # 3 regular-separated
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_self_entry_stabilizes_with_depth():
    mesh = make_convergence_mesh()

    vals = []
    for depth in (3, 4, 5, 6):
        config = AdaptiveIntegrationConfig(
            quadrature_order=2,
            near_factor=2.0,
            max_depth=depth,
            self_max_depth=depth + 1,
        )
        vals.append(single_layer_face_face_full(mesh, 0, 0, config=config))

    vals = np.array(vals, dtype=float)
    diffs = np.abs(np.diff(vals))

    assert np.all(np.isfinite(vals))
    assert diffs[-1] <= diffs[0]


def test_shared_edge_entry_stabilizes_with_depth():
    mesh = make_convergence_mesh()

    vals = []
    for depth in (3, 4, 5, 6):
        config = AdaptiveIntegrationConfig(
            quadrature_order=2,
            near_factor=2.0,
            max_depth=depth,
            self_max_depth=depth + 1,
        )
        vals.append(single_layer_face_face_full(mesh, 0, 1, config=config))

    vals = np.array(vals, dtype=float)
    diffs = np.abs(np.diff(vals))

    assert np.all(np.isfinite(vals))
    assert diffs[-1] <= diffs[0]


def test_shared_vertex_entry_stabilizes_with_depth():
    mesh = make_convergence_mesh()

    vals = []
    for depth in (3, 4, 5, 6):
        config = AdaptiveIntegrationConfig(
            quadrature_order=2,
            near_factor=2.0,
            max_depth=depth,
            self_max_depth=depth + 1,
        )
        vals.append(single_layer_face_face_full(mesh, 0, 2, config=config))

    vals = np.array(vals, dtype=float)
    diffs = np.abs(np.diff(vals))

    assert np.all(np.isfinite(vals))
    assert diffs[-1] <= diffs[0]


def test_matrix_is_cauchy_stable_with_depth_in_frobenius_norm():
    mesh = make_convergence_mesh()
    mats = []

    for depth in (4, 5, 6):
        config = AdaptiveIntegrationConfig(
            quadrature_order=2,
            near_factor=2.0,
            max_depth=depth,
            self_max_depth=depth + 1,
        )
        V, metadata = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config)
        mats.append(V)

    d45 = np.linalg.norm(mats[1] - mats[0], ord="fro")
    d56 = np.linalg.norm(mats[2] - mats[1], ord="fro")
    scale = max(np.linalg.norm(mats[2], ord="fro"), 1.0e-30)

    assert d45 / scale < 0.1
    assert d56 / scale < 0.1


def test_matrix_changes_mildly_between_quadrature_orders():
    mesh = make_convergence_mesh()

    config_1 = AdaptiveIntegrationConfig(
        quadrature_order=1,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )
    config_2 = AdaptiveIntegrationConfig(
        quadrature_order=2,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )

    V1, _ = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config_1)
    V2, _ = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config_2)

    rel = np.linalg.norm(V2 - V1, ord="fro") / max(np.linalg.norm(V2, ord="fro"), 1.0e-30)

    assert rel < 0.5