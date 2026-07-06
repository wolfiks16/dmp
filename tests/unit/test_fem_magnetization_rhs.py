from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import (
    assemble_curlcurl_matrix,
    assemble_magnetization_rhs,
)
from magcore.femcore.basis_nedelec import physical_nedelec_curl
from magcore.femcore.mixed_problem import MixedCoulombProblem
from magcore.femcore.post import evaluate_curl_on_cell
from magcore.femcore.reference_tetra import AffineTetraMap
from magcore.femcore.solver import solve_mixed_coulomb_problem, split_mixed_solution
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh_generators import (
    build_structured_box_tetra_mesh,
    build_structured_unit_cube_tetra_mesh,
)


def _zero_current(x: np.ndarray) -> np.ndarray:
    return np.zeros(3, dtype=float)


def test_magnetization_rhs_shape_and_zero_source() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    space = NedelecP1Space.from_mesh(mesh)

    nu_br = np.zeros((mesh.n_cells, 3), dtype=float)
    f = assemble_magnetization_rhs(mesh, space, nu_br)

    assert f.shape == (space.ndofs,)
    assert np.isfinite(f).all()
    assert np.allclose(f, 0.0, atol=1e-14)


def test_magnetization_rhs_matches_direct_curl_formula() -> None:
    # f_i = sum_T s_i (nu_br . curl w_i) |T|, computed independently via cell volume.
    mesh = build_structured_unit_cube_tetra_mesh(1)
    space = NedelecP1Space.from_mesh(mesh)

    g = np.array([0.3, -0.7, 1.1], dtype=float)
    nu_br = np.tile(g, (mesh.n_cells, 1))

    f = assemble_magnetization_rhs(mesh, space, nu_br)

    f_manual = np.zeros(space.ndofs, dtype=float)
    for c in range(mesh.n_cells):
        amap = AffineTetraMap(mesh.cell_vertices(c))
        vol = mesh.cell_volume(c)
        gdofs = space.cell_dof_indices(c)
        sgn = space.cell_dof_signs(c)
        for i in range(6):
            ci = physical_nedelec_curl(amap, i)
            f_manual[gdofs[i]] += sgn[i] * float(np.dot(g, ci)) * vol

    assert np.allclose(f, f_manual, atol=1e-12)


def test_magnetization_rhs_is_linear_in_source() -> None:
    mesh = build_structured_unit_cube_tetra_mesh(2)
    space = NedelecP1Space.from_mesh(mesh)
    rng = np.random.default_rng(0)

    nu_br1 = rng.standard_normal((mesh.n_cells, 3))
    nu_br2 = rng.standard_normal((mesh.n_cells, 3))

    f1 = assemble_magnetization_rhs(mesh, space, nu_br1)
    f2 = assemble_magnetization_rhs(mesh, space, nu_br2)

    assert np.allclose(assemble_magnetization_rhs(mesh, space, 2.5 * nu_br1), 2.5 * f1)
    assert np.allclose(
        assemble_magnetization_rhs(mesh, space, nu_br1 + nu_br2), f1 + f2
    )


def test_magnetization_rhs_equals_curlcurl_applied_to_potential() -> None:
    # Exact tie to the verified curl-curl assembly: if nu_br = nu * curl(A_r) per cell,
    # then the magnetization RHS equals K @ a_r, because both evaluate the same integral
    # ∫ nu curl(A_r) · curl w.
    mesh = build_structured_unit_cube_tetra_mesh(2)
    space = NedelecP1Space.from_mesh(mesh)
    nu = 1.7

    rng = np.random.default_rng(1)
    a_r = rng.standard_normal(space.ndofs)

    nu_br = np.zeros((mesh.n_cells, 3), dtype=float)
    for c in range(mesh.n_cells):
        nu_br[c] = nu * evaluate_curl_on_cell(space, a_r, c)

    f_mag = assemble_magnetization_rhs(mesh, space, nu_br)
    K = assemble_curlcurl_matrix(mesh, space, nu=nu)

    assert np.allclose(f_mag, K @ a_r, atol=1e-12)


def test_uniform_magnetization_gives_zero_interior_rhs_and_zero_field() -> None:
    # Constant nu*B_r over the whole domain: the only source is the bound surface
    # current on the boundary, which the n x A = 0 condition annihilates.
    mesh = build_structured_unit_cube_tetra_mesh(2)
    space = NedelecP1Space.from_mesh(mesh)

    g = np.array([0.0, 0.0, 1.0], dtype=float)
    nu_br = np.tile(g, (mesh.n_cells, 1))
    f = assemble_magnetization_rhs(mesh, space, nu_br)

    boundary = set(space.boundary_dofs())
    interior = [d for d in range(space.ndofs) if d not in boundary]
    assert len(interior) > 0
    assert np.allclose(f[interior], 0.0, atol=1e-12)

    problem = MixedCoulombProblem.from_mesh(mesh, nu=1.0, J_fn=_zero_current)
    system, b = problem.assemble_system()
    b = b.copy()
    b[: problem.n_vector_dofs] += f
    system_bc, b_bc = problem.apply_boundary_conditions(system, b)
    x = solve_mixed_coulomb_problem(system_bc, b_bc)
    a, _p = split_mixed_solution(x, problem.n_vector_dofs)

    assert np.linalg.norm(a) < 1e-10


def _closed_surface_flux(mesh, b_per_cell: np.ndarray) -> float:
    face_owner: dict = {}
    for c in range(mesh.n_cells):
        cv = mesh.cell_vertex_indices(c)
        for face in mesh.cell_faces(c):
            opp = next(v for v in cv if v not in face)
            face_owner[face] = (c, opp)

    flux = 0.0
    for face in mesh.boundary_faces():
        c, opp = face_owner[face]
        i, j, k = face
        p0 = mesh.vertices[i]
        p1 = mesh.vertices[j]
        p2 = mesh.vertices[k]
        po = mesh.vertices[opp]
        nrm = np.cross(p1 - p0, p2 - p0)
        centroid = (p0 + p1 + p2) / 3.0
        if np.dot(nrm, centroid - po) < 0.0:
            nrm = -nrm
        area = 0.5 * float(np.linalg.norm(nrm))
        unit = nrm / np.linalg.norm(nrm)
        flux += float(np.dot(b_per_cell[c], unit)) * area
    return flux


def test_magnet_cube_in_air_field_pattern_and_divergence_free() -> None:
    mesh = build_structured_box_tetra_mesh(4, 4, 4)
    space = NedelecP1Space.from_mesh(mesh)
    nu0 = 1.0

    br = np.array([0.0, 0.0, 1.0], dtype=float)
    nu_br = np.zeros((mesh.n_cells, 3), dtype=float)
    magnet_cells = []
    for c in range(mesh.n_cells):
        xc = mesh.cell_centroid(c)
        if np.all(xc > 0.25) and np.all(xc < 0.75):
            nu_br[c] = nu0 * br
            magnet_cells.append(c)
    assert len(magnet_cells) > 0

    f_mag = assemble_magnetization_rhs(mesh, space, nu_br)

    problem = MixedCoulombProblem.from_mesh(mesh, nu=nu0, J_fn=_zero_current)
    system, b = problem.assemble_system()
    b = b.copy()
    b[: problem.n_vector_dofs] += f_mag
    system_bc, b_bc = problem.apply_boundary_conditions(system, b)
    x = solve_mixed_coulomb_problem(system_bc, b_bc)
    a, _p = split_mixed_solution(x, problem.n_vector_dofs)

    assert np.isfinite(a).all()
    assert np.linalg.norm(a) > 0.0

    b_cell = np.array(
        [evaluate_curl_on_cell(space, a, c) for c in range(mesh.n_cells)]
    )

    # Inside the magnet, B is aligned with the magnetization (+z).
    mean_bz_magnet = float(np.mean(b_cell[magnet_cells, 2]))
    assert mean_bz_magnet > 0.0

    # B = curl A_h is exactly divergence-free => zero net flux through any closed surface.
    flux = _closed_surface_flux(mesh, b_cell)
    assert abs(flux) < 1e-8
