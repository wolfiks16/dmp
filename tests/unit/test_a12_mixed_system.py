from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import assemble_mixed_coulomb_system
from magcore.femcore.boundary_conditions import apply_zero_dirichlet_bc_mixed, find_mixed_boundary_dofs
from magcore.femcore.gauge_diagnostics import (
    compute_gauge_residual_norm,
    compute_relative_longitudinal_component,
)
from magcore.femcore.mesh import TetraMesh, oriented_tetra_volume6
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.solver import solve_mixed_linear_problem
from magcore.femcore.spaces import NedelecP1Space


def _positively_oriented_cell(vertices: np.ndarray, cell: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    arr = np.array(cell, dtype=int)
    vol6 = oriented_tetra_volume6(vertices[arr])
    if vol6 > 0.0:
        return tuple(int(x) for x in arr)
    arr[[0, 1]] = arr[[1, 0]]
    vol6 = oriented_tetra_volume6(vertices[arr])
    if vol6 <= 0.0:
        raise RuntimeError("Could not orient tetra positively.")
    return tuple(int(x) for x in arr)


def make_single_tetra_mesh() -> TetraMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cell = _positively_oriented_cell(vertices, (0, 1, 2, 3))
    return TetraMesh(vertices=vertices, cells=np.array([cell], dtype=int))


def make_barycentric_subdivided_tetra_mesh() -> TetraMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25],
        ],
        dtype=float,
    )
    raw_cells = [
        (0, 1, 2, 4),
        (0, 1, 4, 3),
        (0, 4, 2, 3),
        (4, 1, 2, 3),
    ]
    cells = np.array([_positively_oriented_cell(vertices, c) for c in raw_cells], dtype=int)
    return TetraMesh(vertices=vertices, cells=cells)


def test_mixed_system_shape_and_symmetry_on_single_tetra() -> None:
    mesh = make_single_tetra_mesh()
    vspace = NedelecP1Space.from_mesh(mesh)
    pspace = LagrangeP1Space(mesh)

    A, rhs, K, G = assemble_mixed_coulomb_system(
        mesh=mesh,
        vector_space=vspace,
        scalar_space=pspace,
        nu=2.0,
        J_fn=lambda x: np.zeros(3, dtype=float),
    )

    assert A.shape == (vspace.ndofs + pspace.ndofs, vspace.ndofs + pspace.ndofs)
    assert rhs.shape == (vspace.ndofs + pspace.ndofs,)
    assert K.shape == (vspace.ndofs, vspace.ndofs)
    assert G.shape == (vspace.ndofs, pspace.ndofs)
    assert np.allclose(A, A.T)
    assert np.allclose(rhs, 0.0)


def test_mixed_boundary_conditions_reduce_fully_boundary_single_tetra_to_identity() -> None:
    mesh = make_single_tetra_mesh()
    vspace = NedelecP1Space.from_mesh(mesh)
    pspace = LagrangeP1Space(mesh)

    A, rhs, _, _ = assemble_mixed_coulomb_system(
        mesh=mesh,
        vector_space=vspace,
        scalar_space=pspace,
        nu=1.0,
        J_fn=lambda x: np.array([1.0, -2.0, 3.0], dtype=float),
    )
    vbnd, pbnd = find_mixed_boundary_dofs(vspace, pspace)
    A_bc, rhs_bc = apply_zero_dirichlet_bc_mixed(A, rhs, vbnd, pbnd, n_vector_dofs=vspace.ndofs)

    assert np.allclose(A_bc, np.eye(A_bc.shape[0]))
    assert np.allclose(rhs_bc, 0.0)

    a, p = solve_mixed_linear_problem(A_bc, rhs_bc, n_vector_dofs=vspace.ndofs)
    assert np.allclose(a, 0.0)
    assert np.allclose(p, 0.0)


def test_gauge_projection_detects_pure_gradient_component() -> None:
    mesh = make_barycentric_subdivided_tetra_mesh()
    vspace = NedelecP1Space.from_mesh(mesh)
    pspace = LagrangeP1Space(mesh)
    _, _, _, G = assemble_mixed_coulomb_system(
        mesh=mesh,
        vector_space=vspace,
        scalar_space=pspace,
        nu=1.0,
        J_fn=lambda x: np.zeros(3, dtype=float),
    )

    phi = np.zeros(pspace.ndofs, dtype=float)
    phi[4] = 1.0

    edge_to_dof = vspace.edge_to_dof_map()
    a = np.zeros(vspace.ndofs, dtype=float)
    for (i, j), dof in edge_to_dof.items():
        a[dof] = phi[j] - phi[i]

    residual_norm = compute_gauge_residual_norm(G, a)
    assert residual_norm > 1.0e-10

    eta_parallel = compute_relative_longitudinal_component(
        mesh=mesh,
        vector_space=vspace,
        scalar_space=pspace,
        a=a,
        G=G,
    )
    assert np.isclose(eta_parallel, 1.0, atol=1e-10)
