from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig, assemble_single_layer_p0p0_full
from magcore.mesh.surface_mesh import SurfaceMesh


def make_full_assembly_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],   # face 0
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],   # face 1 shared-edge with 0
            [2.0, 0.0, 0.0],   # face 2 shared-vertex with 0
            [0.0, 0.0, 5.0],   # face 3 regular-separated
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],  # 0
            [1, 3, 2],  # 1
            [1, 4, 3],  # 2
            [5, 6, 7],  # 3
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_assemble_single_layer_p0p0_full_shape():
    mesh = make_full_assembly_mesh()
    config = AdaptiveIntegrationConfig(quadrature_order=2, max_depth=4, self_max_depth=5)

    mat, metadata = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config)

    assert mat.shape == (4, 4)
    assert metadata["backend"] == "adaptive_complete_p0p0_single_layer"


def test_assemble_single_layer_p0p0_full_is_finite():
    mesh = make_full_assembly_mesh()
    config = AdaptiveIntegrationConfig(quadrature_order=2, max_depth=4, self_max_depth=5)

    mat, metadata = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config)

    assert np.isfinite(mat).all()


def test_assemble_single_layer_p0p0_full_is_symmetric():
    mesh = make_full_assembly_mesh()
    config = AdaptiveIntegrationConfig(quadrature_order=2, max_depth=4, self_max_depth=5)

    mat, metadata = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config)

    assert np.allclose(mat, mat.T, rtol=1e-12, atol=1e-12)


def test_assemble_single_layer_p0p0_full_diagonal_positive():
    mesh = make_full_assembly_mesh()
    config = AdaptiveIntegrationConfig(quadrature_order=2, max_depth=4, self_max_depth=5)

    mat, metadata = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config)

    assert np.all(np.diag(mat) > 0.0)


def test_assemble_single_layer_p0p0_full_offdiagonal_finite():
    mesh = make_full_assembly_mesh()
    config = AdaptiveIntegrationConfig(quadrature_order=2, max_depth=4, self_max_depth=5)

    mat, metadata = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config)

    offdiag = mat.copy()
    np.fill_diagonal(offdiag, 0.0)
    assert np.isfinite(offdiag).all()