from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import (
    assemble_coulomb_coupling_matrix,
    assemble_discrete_gradient_matrix,
    assemble_mixed_coulomb_system,
)
from magcore.femcore.boundary_conditions import apply_zero_mixed_dirichlet_bc
from magcore.femcore.gauge_diagnostics import project_to_gradient_subspace
from magcore.femcore.mesh import TetraMesh, oriented_tetra_volume6
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space


def _single_tetra_mesh() -> TetraMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=int)
    return TetraMesh(vertices=vertices, cells=cells)


def _cube_with_center_mesh() -> TetraMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
            [0.5, 0.5, 0.5],  # 8 center
        ],
        dtype=float,
    )

    # 12 треугольников на границе куба
    boundary_tris = [
        (0, 1, 2), (0, 2, 3),  # z = 0
        (4, 6, 5), (4, 7, 6),  # z = 1
        (0, 5, 1), (0, 4, 5),  # y = 0
        (3, 2, 6), (3, 6, 7),  # y = 1
        (0, 3, 7), (0, 7, 4),  # x = 0
        (1, 5, 6), (1, 6, 2),  # x = 1
    ]

    cells: list[list[int]] = []
    for a, b, c in boundary_tris:
        tet = [a, b, c, 8]
        vol6 = oriented_tetra_volume6(vertices[np.array(tet, dtype=int)])
        if vol6 <= 0.0:
            tet = [a, c, b, 8]
        cells.append(tet)

    return TetraMesh(vertices=vertices, cells=np.asarray(cells, dtype=int))


def test_assemble_mixed_coulomb_system_shapes_and_symmetry() -> None:
    mesh = _single_tetra_mesh()
    vector_space = NedelecP1Space.from_mesh(mesh)
    scalar_space = LagrangeP1Space(mesh)

    def J_fn(x: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=float)

    A, b = assemble_mixed_coulomb_system(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        nu=1.0,
        J_fn=J_fn,
    )

    nA = vector_space.ndofs
    nP = scalar_space.ndofs

    assert A.shape == (nA + nP, nA + nP)
    assert b.shape == (nA + nP,)
    assert np.allclose(A, A.T, atol=1e-12)
    assert np.allclose(b[nA:], 0.0, atol=1e-12)


def test_apply_zero_mixed_dirichlet_bc_single_tetra_all_boundary() -> None:
    mesh = _single_tetra_mesh()
    vector_space = NedelecP1Space.from_mesh(mesh)
    scalar_space = LagrangeP1Space(mesh)

    def J_fn(x: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=float)

    A, b = assemble_mixed_coulomb_system(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        nu=2.0,
        J_fn=J_fn,
    )

    vector_bnd = vector_space.boundary_dofs()
    scalar_bnd = scalar_space.boundary_dofs()

    A_bc, b_bc = apply_zero_mixed_dirichlet_bc(
        A=A,
        b=b,
        vector_dofs=vector_bnd,
        scalar_dofs=scalar_bnd,
        n_vector_dofs=vector_space.ndofs,
    )

    assert np.allclose(A_bc, np.eye(A_bc.shape[0]), atol=1e-12)
    assert np.allclose(b_bc, 0.0, atol=1e-12)


def test_gauge_projection_recovers_pure_gradient_on_mesh_with_interior_vertex() -> None:
    mesh = _cube_with_center_mesh()
    vector_space = NedelecP1Space.from_mesh(mesh)
    scalar_space = LagrangeP1Space(mesh)

    # Скалярное поле с нулём на границе и единицей во внутренней вершине.
    phi = np.zeros(scalar_space.ndofs, dtype=float)
    phi[8] = 1.0

    D = assemble_discrete_gradient_matrix(
        vector_space=vector_space,
        scalar_space=scalar_space,
    )
    a_grad = D @ phi

    result = project_to_gradient_subspace(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
        a=a_grad,
    )

    assert result.norm_total > 0.0
    assert np.isclose(result.eta_parallel, 1.0, atol=1e-10)
    assert np.isclose(result.norm_perp, 0.0, atol=1e-10)

    G = assemble_coulomb_coupling_matrix(
        mesh=mesh,
        vector_space=vector_space,
        scalar_space=scalar_space,
    )
    residual = G.T @ result.a_perp
    assert np.linalg.norm(residual) < 1e-10