from __future__ import annotations

import math

import numpy as np

from magcore.femcore.assembly import (
    assemble_coulomb_coupling_matrix,
    assemble_discrete_gradient_matrix,
    assemble_mass_matrix,
)
from magcore.femcore.manufactured import (
    manufactured_A_ref,
    manufactured_curl_ref,
    manufactured_curlcurl_ref,
)
from magcore.femcore.mixed_problem import solve_mixed_coulomb_baseline
from magcore.femcore.post import (
    l2_curl_error_at_cell_centroids,
    l2_error_at_cell_centroids,
)
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh

_NU = 1.3


def _solve_manufactured_mixed(n: int):
    # A_ref is divergence-free with n x A_ref = 0 on the unit cube, so it solves
    # the gauged saddle system with source J = curl(nu curl A_ref) = nu * curlcurl(A_ref)
    # and exact multiplier p = 0.
    mesh = build_structured_unit_cube_tetra_mesh(n)
    sol = solve_mixed_coulomb_baseline(
        mesh=mesh,
        nu=_NU,
        J_fn=lambda x: _NU * manufactured_curlcurl_ref(x),
        compute_gauge_projection=False,
    )
    space = sol.problem.vector_space
    err_a = l2_error_at_cell_centroids(space, sol.a, manufactured_A_ref)
    err_curl = l2_curl_error_at_cell_centroids(space, sol.a, manufactured_curl_ref)
    return sol, err_a, err_curl


def test_mixed_manufactured_convergence_and_zero_multiplier() -> None:
    results = {}
    for n in (2, 4, 8):
        sol, err_a, err_curl = _solve_manufactured_mixed(n)
        results[n] = (
            err_a,
            err_curl,
            float(np.linalg.norm(sol.p)),
            float(np.linalg.norm(sol.a)),
        )

    # Solenoidal source => discrete Coulomb multiplier vanishes to machine precision.
    for n in (2, 4, 8):
        _err_a, _err_curl, p_norm, a_norm = results[n]
        assert a_norm > 0.0
        assert p_norm / a_norm < 1e-9

    # Monotone error reduction under uniform refinement.
    assert results[4][0] < results[2][0]
    assert results[8][0] < results[4][0]
    assert results[4][1] < results[2][1]
    assert results[8][1] < results[4][1]

    # First-kind lowest-order Nedelec: BOTH the H(curl) error and the L2 error of A
    # converge at order 1.  In particular the L2 rate is ~1, decisively NOT 2 (the
    # local space does not contain full linear vector fields).
    rate_a = math.log(results[4][0] / results[8][0]) / math.log(2.0)
    rate_curl = math.log(results[4][1] / results[8][1]) / math.log(2.0)
    assert 0.8 < rate_a < 1.3
    assert 0.8 < rate_curl < 1.3


def test_coulomb_coupling_equals_mass_times_discrete_gradient() -> None:
    # Whitney identity G = M_v D: gradients of P1 functions lie exactly in the Nedelec
    # space (∇Q_h ⊂ V_h).  This exact-sequence property is the algebraic backbone of the
    # inf-sup stability of the Nedelec x P1 pair (docs/math/formulation_bounded.md §5, §7).
    mesh = build_structured_unit_cube_tetra_mesh(2)
    vspace = NedelecP1Space.from_mesh(mesh)
    sspace = LagrangeP1Space(mesh)

    G = assemble_coulomb_coupling_matrix(mesh, vspace, sspace)
    M = assemble_mass_matrix(mesh, vspace, alpha=1.0)
    D = assemble_discrete_gradient_matrix(vspace, sspace)

    assert G.shape == (vspace.ndofs, sspace.ndofs)
    assert np.allclose(G, M @ D, atol=1e-12)
