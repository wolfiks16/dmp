from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import assemble_magnetization_rhs
from magcore.femcore.post import evaluate_curl_on_cell
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.assembly import assemble_coupled_block_system
from magcore.hybrid.interface import CouplingInterface
from magcore.hybrid.solver import solve_coupled_block_system
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def _zero_j(_x):
    return np.zeros(3)


def test_uniformly_magnetized_cube_volume_averaged_demag_factor() -> None:
    """
    Бенчмарк (3) НОВИЗНЫ (1): равномерно намагниченный КУБ — связь на НЕГЛАДКОЙ
    гранёной границе (углы/рёбра). Аналитика: объёмно-усреднённый размагничивающий
    фактор куба = 1/3 (точно: сумма `N_x+N_y+N_z=1` + кубическая симметрия) ⇒
    `<H_z>_vol = -M/3`. В отличие от сферы поле внутри НЕ однородно (только эллипсоиды
    дают однородное) — проверяем и средний, и саму неоднородность.
    """
    m_vec = np.array([0.0, 0.0, 1.0])
    mesh = build_structured_unit_cube_tetra_mesh(3)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)

    nu_br = np.tile(m_vec, (mesh.n_cells, 1))
    f_br = assemble_magnetization_rhs(mesh, vs, nu_br)
    coupled = assemble_coupled_block_system(
        ci, vs, ss, nu=1.0, j_fn=_zero_j, extra_vector_rhs=f_br, mu0=1.0
    )
    sol = solve_coupled_block_system(coupled, ss)
    assert sol.residual_norm < 1e-8

    b_cells = np.array([evaluate_curl_on_cell(vs, sol.a, c) for c in range(mesh.n_cells)])
    vols = np.array([mesh.cell_volume(c) for c in range(mesh.n_cells)])
    w = vols / vols.sum()
    h = b_cells - m_vec  # H = ν(B − B_r) = B − M (ν=1)

    hx_mean = float(w @ h[:, 0])
    hy_mean = float(w @ h[:, 1])
    hz_mean = float(w @ h[:, 2])

    # Объёмно-усреднённый размагничивающий фактор куба = 1/3 (точно по симметрии).
    assert abs(hz_mean + 1.0 / 3.0) < 5e-3
    # Поперечные средние ≈ 0 (симметрия M ∥ z).
    assert abs(hx_mean) < 5e-3
    assert abs(hy_mean) < 5e-3

    # Поле внутри куба НЕ однородно (углы/рёбра) — в отличие от сферы (там std ~0.003).
    hz_std = float(np.sqrt(w @ (h[:, 2] - hz_mean) ** 2))
    assert hz_std > 0.05
