from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import assemble_magnetization_rhs
from magcore.femcore.post import evaluate_curl_on_cell
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.assembly import assemble_coupled_block_system
from magcore.hybrid.interface import CouplingInterface
from magcore.hybrid.solver import coupled_nullspace_dim, solve_coupled_min_norm
from magcore.mesh.mesh_generators import build_frame_tetra_mesh, build_structured_unit_cube_tetra_mesh


def _zero_j(_x):
    return np.zeros(3)


def test_noncontractible_frame_adds_one_cohomology_nullmode() -> None:
    """
    Куб (стягиваем, b₁=0) vs «рамка» = куб со сквозным отверстием (b₁=1).
    Размерность ядра связанной системы отличается ровно на b₁=1: куб → 1 (только
    константа множителя p), рамка → 2 (+ одно гармоническое поле / когомология).
    """
    cube = build_structured_unit_cube_tetra_mesh(3)
    frame = build_frame_tetra_mesh(3)

    def _coupled(mesh):
        ci = CouplingInterface.from_tetra_mesh(mesh)
        vs = NedelecP1Space.from_mesh(mesh)
        ss = LagrangeP1Space(mesh)
        return assemble_coupled_block_system(ci, vs, ss, nu=1.0, j_fn=_zero_j, mu0=1.0)

    assert coupled_nullspace_dim(_coupled(cube)) == 1
    assert coupled_nullspace_dim(_coupled(frame)) == 2


def test_frame_cohomology_mode_is_harmonic_gauge_and_B_is_invariant() -> None:
    """
    Дополнительная нуль-мода нестягиваемой области — ГАРМОНИЧЕСКОЕ КАЛИБРОВОЧНОЕ поле:
      • curl-free (⇒ B=curl A не меняется при добавлении ⇒ B-инвариантна, калибровка);
      • div-free (⇒ Кулонова калибровка её НЕ убирает: удовлетворяет Gᵀa=0).
    Поэтому физическое поле B инвариантно к выбору когомологической компоненты — для
    магнитостатики когомология чисто калибровочная (не физический dof).
    """
    mesh = build_frame_tetra_mesh(3)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    m_vec = np.array([0.0, 0.0, 1.0])
    f_br = assemble_magnetization_rhs(mesh, vs, np.tile(m_vec, (mesh.n_cells, 1)))
    coupled = assemble_coupled_block_system(
        ci, vs, ss, nu=1.0, j_fn=_zero_j, extra_vector_rhs=f_br, mu0=1.0
    )
    mat = coupled.matrix

    # Выделяем когомологическую нуль-моду (a-доминантный почти-нулевой собств. вектор).
    ev, vecs = np.linalg.eigh(mat)
    scale = float(np.abs(ev).max())
    z_a = None
    for k in range(ev.shape[0]):
        if abs(ev[k]) < 1e-9 * scale:
            a, p, _psi, _lam = coupled.split(vecs[:, k])
            if np.linalg.norm(a) > 0.5 and np.linalg.norm(p) < 0.5:
                z_a = a
    assert z_a is not None  # когомологическая мода существует

    # curl-free (B-инвариантна) и div-free (не убирается Кулоновой калибровкой).
    curls = np.array([evaluate_curl_on_cell(vs, z_a, c) for c in range(mesh.n_cells)])
    assert float(np.max(np.linalg.norm(curls, axis=1))) < 1e-9
    g_block = mat[: coupled.n_a, coupled.off_p : coupled.off_psi]
    assert float(np.linalg.norm(g_block.T @ z_a)) < 1e-9

    # B инвариантно к добавлению когомологической моды к любому решению.
    a_sol, _p, _psi, _lam = solve_coupled_min_norm(coupled)
    b1 = np.array([evaluate_curl_on_cell(vs, a_sol, c) for c in range(mesh.n_cells)])
    b2 = np.array([evaluate_curl_on_cell(vs, a_sol + 5.0 * z_a, c) for c in range(mesh.n_cells)])
    assert np.allclose(b1, b2, atol=1e-9)
