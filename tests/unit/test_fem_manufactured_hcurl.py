from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import assemble_system
from magcore.femcore.boundary_conditions import (
    apply_zero_dirichlet_bc,
    find_boundary_dofs,
)
from magcore.femcore.manufactured import (
    manufactured_A_ref,
    manufactured_curl_ref,
    manufactured_rhs,
)
from magcore.femcore.mesh import TetraMesh
from magcore.femcore.post import (
    l2_curl_error_at_cell_centroids,
    l2_error_at_cell_centroids,
)
from magcore.femcore.solver import solve_linear_hcurl_problem
from magcore.femcore.spaces import NedelecP1Space


def oriented_cube_tetra_mesh(n: int) -> TetraMesh:
    """
    Structured tetrahedral mesh of the unit cube [0,1]^3 using 6 tetrahedra
    per cube cell. Cell orientations are fixed to be positive.
    """
    if n < 1:
        raise ValueError("n must be positive.")

    xs = np.linspace(0.0, 1.0, n + 1, dtype=float)
    ys = np.linspace(0.0, 1.0, n + 1, dtype=float)
    zs = np.linspace(0.0, 1.0, n + 1, dtype=float)

    vertices = []
    vid = {}
    k = 0
    for iz, z in enumerate(zs):
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                vid[(ix, iy, iz)] = k
                vertices.append([x, y, z])
                k += 1
    vertices = np.asarray(vertices, dtype=float)

    def v(ix: int, iy: int, iz: int) -> int:
        return vid[(ix, iy, iz)]

    cells = []

    def oriented_cell(a: int, b: int, c: int, d: int) -> list[int]:
        verts = vertices[[a, b, c, d]]
        J = np.column_stack((verts[1] - verts[0], verts[2] - verts[0], verts[3] - verts[0]))
        if np.linalg.det(J) > 0.0:
            return [a, b, c, d]
        return [a, c, b, d]

    for iz in range(n):
        for iy in range(n):
            for ix in range(n):
                v000 = v(ix, iy, iz)
                v100 = v(ix + 1, iy, iz)
                v010 = v(ix, iy + 1, iz)
                v110 = v(ix + 1, iy + 1, iz)
                v001 = v(ix, iy, iz + 1)
                v101 = v(ix + 1, iy, iz + 1)
                v011 = v(ix, iy + 1, iz + 1)
                v111 = v(ix + 1, iy + 1, iz + 1)

                raw_cells = [
                    [v000, v100, v110, v111],
                    [v000, v100, v101, v111],
                    [v000, v001, v101, v111],
                    [v000, v010, v110, v111],
                    [v000, v010, v011, v111],
                    [v000, v001, v011, v111],
                ]
                for c in raw_cells:
                    cells.append(oriented_cell(*c))

    return TetraMesh(vertices=vertices, cells=np.asarray(cells, dtype=int))


def solve_manufactured_case(n: int, nu: float = 1.0, alpha: float = 1.0):
    mesh = oriented_cube_tetra_mesh(n)
    space = NedelecP1Space.from_mesh(mesh)

    A, b = assemble_system(
        mesh=mesh,
        space=space,
        nu=nu,
        alpha=alpha,
        f_fn=lambda x: manufactured_rhs(x, nu=nu, alpha=alpha),
        curl_quadrature_order=1,
        mass_quadrature_order=2,
        rhs_quadrature_order=2,
    )

    bdofs = find_boundary_dofs(space)
    Abc, bbc = apply_zero_dirichlet_bc(A, b, bdofs)
    coeffs = solve_linear_hcurl_problem(Abc, bbc)

    err_A = l2_error_at_cell_centroids(space, coeffs, manufactured_A_ref)
    err_curl = l2_curl_error_at_cell_centroids(space, coeffs, manufactured_curl_ref)

    residual = Abc @ coeffs - bbc
    return {
        "mesh": mesh,
        "space": space,
        "coeffs": coeffs,
        "err_A": err_A,
        "err_curl": err_curl,
        "residual_norm": float(np.linalg.norm(residual)),
        "boundary_dofs": bdofs,
    }


def test_manufactured_hcurl_case_runs_on_coarse_cube_mesh():
    out = solve_manufactured_case(n=1, nu=1.0, alpha=1.0)

    assert out["space"].ndofs > 0
    assert np.isfinite(out["coeffs"]).all()
    assert np.isfinite(out["err_A"])
    assert np.isfinite(out["err_curl"])
    assert np.isfinite(out["residual_norm"])


def test_manufactured_hcurl_solution_satisfies_bc_dofs():
    out = solve_manufactured_case(n=1, nu=1.0, alpha=1.0)

    coeffs = out["coeffs"]
    for d in out["boundary_dofs"]:
        assert np.isclose(coeffs[d], 0.0, atol=1.0e-12)


def test_manufactured_hcurl_residual_is_small():
    out = solve_manufactured_case(n=1, nu=1.0, alpha=1.0)

    assert out["residual_norm"] < 1.0e-10


def test_manufactured_hcurl_errors_are_bounded_on_coarse_mesh():
    out = solve_manufactured_case(n=1, nu=1.0, alpha=1.0)

    assert out["err_A"] < 0.25
    assert out["err_curl"] < 0.6


def test_manufactured_hcurl_error_decreases_under_refinement():
    coarse = solve_manufactured_case(n=1, nu=1.0, alpha=1.0)
    fine = solve_manufactured_case(n=2, nu=1.0, alpha=1.0)

    assert fine["err_A"] < coarse["err_A"]
    assert fine["err_curl"] < coarse["err_curl"]


def test_manufactured_hcurl_refined_mesh_has_more_dofs():
    coarse = solve_manufactured_case(n=1, nu=1.0, alpha=1.0)
    fine = solve_manufactured_case(n=2, nu=1.0, alpha=1.0)

    assert fine["space"].ndofs > coarse["space"].ndofs
    assert fine["mesh"].n_cells > coarse["mesh"].n_cells