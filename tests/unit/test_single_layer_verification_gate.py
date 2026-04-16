from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import (
    AdaptiveIntegrationConfig,
    assemble_single_layer_p0p0_full,
)
from magcore.bem.evaluation import evaluate_single_layer_potential_p0
from magcore.mesh.surface_mesh import SurfaceMesh


def make_verification_mesh() -> SurfaceMesh:
    """
    Small mixed-topology mesh:
    - face 0 and 1 share an edge
    - face 0 and 2 share a vertex
    - face 3 is far separated
    """
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
            [0, 1, 2],  # face 0
            [1, 3, 2],  # face 1 shared-edge with 0
            [1, 4, 3],  # face 2 shared-vertex with 0
            [5, 6, 7],  # face 3 regular-separated
        ],
        dtype=int,
    )
    return SurfaceMesh(vertices=vertices, faces=faces)


def make_single_triangle_mesh() -> SurfaceMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    return SurfaceMesh(vertices=vertices, faces=faces)


def test_single_layer_matrix_is_symmetric_to_tight_tolerance():
    mesh = make_verification_mesh()
    config = AdaptiveIntegrationConfig(
        quadrature_order=2,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )

    V, metadata = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config)

    skew = V - V.T
    skew_norm = np.linalg.norm(skew, ord="fro")
    V_norm = np.linalg.norm(V, ord="fro")

    rel = skew_norm / max(V_norm, 1.0e-30)
    assert rel < 1.0e-10


def test_single_layer_matrix_has_positive_diagonal():
    mesh = make_verification_mesh()
    config = AdaptiveIntegrationConfig(
        quadrature_order=2,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )

    V, metadata = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config)

    assert np.all(np.isfinite(np.diag(V)))
    assert np.all(np.diag(V) > 0.0)


def test_single_layer_quadratic_form_positive_for_nonzero_density():
    mesh = make_verification_mesh()
    config = AdaptiveIntegrationConfig(
        quadrature_order=2,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )

    V, metadata = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config)

    sigma = np.array([1.0, -0.7, 0.4, 0.9], dtype=float)
    q = float(sigma @ V @ sigma)

    assert np.isfinite(q)
    assert q > 0.0


def test_single_layer_matrix_entries_are_all_finite():
    mesh = make_verification_mesh()
    config = AdaptiveIntegrationConfig(
        quadrature_order=2,
        near_factor=2.0,
        max_depth=5,
        self_max_depth=6,
    )

    V, metadata = assemble_single_layer_p0p0_full(mesh, (0, 1, 2, 3), config=config)

    assert np.isfinite(V).all()


def test_far_field_potential_is_positive_and_decays():
    mesh = make_single_triangle_mesh()
    density = np.array([1.0], dtype=float)

    pts = np.array(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 20.0],
            [0.0, 0.0, 40.0],
        ],
        dtype=float,
    )

    phi = evaluate_single_layer_potential_p0(
        mesh=mesh,
        face_indices=(0,),
        density=density,
        target_points=pts,
        quadrature_order=2,
    )

    assert np.all(phi > 0.0)
    assert phi[0] > phi[1] > phi[2]


def test_far_field_r_times_phi_is_approximately_constant():
    mesh = make_single_triangle_mesh()
    density = np.array([1.0], dtype=float)

    pts = np.array(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 20.0],
            [0.0, 0.0, 40.0],
        ],
        dtype=float,
    )

    phi = evaluate_single_layer_potential_p0(
        mesh=mesh,
        face_indices=(0,),
        density=density,
        target_points=pts,
        quadrature_order=2,
    )

    r = np.linalg.norm(pts, axis=1)
    scaled = r * phi

    # not exact, but should be fairly stable in the far field
    rel_spread = (np.max(scaled) - np.min(scaled)) / max(np.mean(scaled), 1.0e-30)
    assert rel_spread < 0.15