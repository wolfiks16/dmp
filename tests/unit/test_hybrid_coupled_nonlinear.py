from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import assemble_magnetization_rhs
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.assembly import assemble_coupled_block_system
from magcore.hybrid.interface import CouplingInterface
from magcore.hybrid.nonlinear import solve_coupled_nonlinear_picard
from magcore.hybrid.solver import solve_coupled_block_system
from magcore.mesh.mesh_generators import (
    build_ball_tetra_mesh,
    build_structured_unit_cube_tetra_mesh,
)


def _zero_j(_x):
    return np.zeros(3)


def test_coupled_picard_const_nu_static_magnet_equals_linear() -> None:
    """
    Якорь корректности: при ПОСТОЯННОЙ ν и статическом магните нелинейный связанный
    Picard ОБЯЗАН точно совпасть с одним линейным связанным solve (намагниченная сфера).
    """
    m_vec = np.array([0.0, 0.0, 1.0])
    mesh = build_ball_tetra_mesh(3, radius=1.0)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    n = mesh.n_cells

    nu_br = np.tile(m_vec, (n, 1))  # ν·B_r при ν=1
    f_br = assemble_magnetization_rhs(mesh, vs, nu_br)
    lin = solve_coupled_block_system(
        assemble_coupled_block_system(
            ci, vs, ss, nu=1.0, j_fn=_zero_j, extra_vector_rhs=f_br, mu0=1.0
        ),
        ss,
    )

    res = solve_coupled_nonlinear_picard(
        ci, vs, ss,
        nu_of_B=lambda B: np.full(n, 1.0),
        nu_init=np.full(n, 1.0),
        magnetization=nu_br,
        tol=1e-10,
    )
    assert res.converged
    assert res.n_iterations <= 3
    assert np.allclose(res.a, lin.a, atol=1e-9)


def test_coupled_picard_saturating_steel_with_magnet_converges() -> None:
    """
    Совместный расчёт «магнит ↔ НЕЛИНЕЙНОЕ железо» в открытой области: магнит (нижняя
    половина куба, B_r=ẑ, ν=1) гонит поток в насыщающуюся сталь (верхняя половина,
    ν_iron(|B|) растёт с индукцией). Проверяем сходимость Picard, активность
    нелинейности (ν варьируется по стали и выше ненасыщенной) и отличие от линейного
    приближения с замороженной ν.
    """
    m_vec = np.array([0.0, 0.0, 1.0])
    mesh = build_structured_unit_cube_tetra_mesh(2)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    n = mesh.n_cells

    is_magnet = np.array([mesh.cell_centroid(c)[2] < 0.5 for c in range(n)])
    iron = ~is_magnet

    mag = np.zeros((n, 3))
    mag[is_magnet] = m_vec  # статический магнит (ν=1 в нём)

    nu0_iron, c_sat = 1.0 / 100.0, 50.0  # μ_r≈100 без насыщения; растёт с |B|²

    def nu_of_B(B):
        nu = np.ones(n)  # ячейки магнита: ν=1
        s2 = np.sum(np.asarray(B) ** 2, axis=1)
        nu_iron = nu0_iron * (1.0 + c_sat * s2)
        nu[iron] = nu_iron[iron]
        return nu

    nu_init = np.where(is_magnet, 1.0, nu0_iron)

    res = solve_coupled_nonlinear_picard(
        ci, vs, ss,
        nu_of_B=nu_of_B,
        nu_init=nu_init,
        magnetization=mag,
        tol=1e-8,
        max_iter=100,
        relaxation=1.0,
    )
    assert res.converged
    assert res.residual_norm < 1e-8

    # Нелинейность активна: ν по стали варьируется и поднялась выше ненасыщенной.
    assert res.nu_cells[iron].max() > res.nu_cells[iron].min() + 1e-9
    assert res.nu_cells[iron].max() > nu0_iron + 1e-9

    # Самосогласованность (relaxation=1): собранная финальная ν ≈ ν(|B_фин|).
    assert np.allclose(res.nu_cells, nu_of_B(res.B_cells), rtol=1e-3, atol=1e-6)

    # Отличие от линейного приближения (ν заморожена на ненасыщенной nu_init).
    f_br = assemble_magnetization_rhs(mesh, vs, mag)
    lin = solve_coupled_block_system(
        assemble_coupled_block_system(
            ci, vs, ss, nu=nu_init, j_fn=_zero_j, extra_vector_rhs=f_br, mu0=1.0
        ),
        ss,
    )
    assert not np.allclose(res.a, lin.a, atol=1e-6)
