from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import assemble_magnetization_rhs
from magcore.femcore.post import evaluate_curl_on_cell
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.assembly import assemble_coupled_block_system
from magcore.hybrid.interface import CouplingInterface
from magcore.hybrid.solver import solve_coupled_block_system
from magcore.mesh.mesh_generators import (
    build_ball_tetra_mesh,
    build_structured_unit_cube_tetra_mesh,
)


def _zero_j(_x):
    return np.zeros(3)


def _b_per_cell(vs, a, n_cells):
    return np.array([evaluate_curl_on_cell(vs, a, c) for c in range(n_cells)])


# --------------------------------------------------------------------------------------
# Уровень 1 — ТОЧНЫЙ регресс: поячеечная ν (однородный массив) ≡ скалярная ν в СВЯЗАННОМ
# решателе. Доказывает, что поячеечный путь (основа магнит+железо) не сдвигает физику.
# --------------------------------------------------------------------------------------
def test_coupled_per_cell_nu_array_matches_scalar_exactly() -> None:
    m_vec = np.array([0.0, 0.0, 1.0])
    mesh = build_ball_tetra_mesh(3, radius=1.0)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)

    nu_br = np.tile(m_vec, (mesh.n_cells, 1))
    f_br = assemble_magnetization_rhs(mesh, vs, nu_br)

    c_scalar = assemble_coupled_block_system(
        ci, vs, ss, nu=1.0, j_fn=_zero_j, extra_vector_rhs=f_br, mu0=1.0
    )
    c_array = assemble_coupled_block_system(
        ci, vs, ss, nu=np.full(mesh.n_cells, 1.0), j_fn=_zero_j,
        extra_vector_rhs=f_br, mu0=1.0,
    )
    assert np.allclose(c_scalar.matrix, c_array.matrix)
    assert np.allclose(c_scalar.rhs, c_array.rhs)

    s_scalar = solve_coupled_block_system(c_scalar, ss)
    s_array = solve_coupled_block_system(c_array, ss)
    assert np.allclose(s_scalar.a, s_array.a, atol=1e-12)


# --------------------------------------------------------------------------------------
# Уровень 2 — АНАЛИТИКА: равномерно намагниченная сфера с ВОЗВРАТНОЙ проницаемостью μ_rec.
# Конститутив кода: H = ν(B − B_r), ν = 1/μ_rec (μ₀=1); RHS намагниченности = ν·B_r.
# Для сферы (демаг-фактор 1/3, теорема об эллипсоиде): поле внутри однородно и
#     H_in = −B_r/(μ_rec + 2),   B_in = 2·B_r/(μ_rec + 2).
# μ_rec=1 ⇒ H=−B_r/3 (совпадает с идеальным магнитом). Реальный NdFeB: μ_rec≈1.05.
# Это первый тест, где per-cell ν и источник-магнит работают ВМЕСТЕ против замкнутой
# аналитики в открытой области.
# --------------------------------------------------------------------------------------
def test_uniformly_magnetized_sphere_with_recoil_permeability() -> None:
    Br = 1.0  # остаточная индукция (вдоль z)
    for mu_rec, tol in ((1.05, 6e-3), (2.0, 1.0e-2)):
        nu = 1.0 / mu_rec
        mesh = build_ball_tetra_mesh(3, radius=1.0)
        ci = CouplingInterface.from_tetra_mesh(mesh)
        vs = NedelecP1Space.from_mesh(mesh)
        ss = LagrangeP1Space(mesh)

        # RHS намагниченности — это ν·B_r (см. assemble_magnetization_rhs).
        nu_br = np.tile(np.array([0.0, 0.0, nu * Br]), (mesh.n_cells, 1))
        f_br = assemble_magnetization_rhs(mesh, vs, nu_br)
        coupled = assemble_coupled_block_system(
            ci, vs, ss, nu=np.full(mesh.n_cells, nu), j_fn=_zero_j,
            extra_vector_rhs=f_br, mu0=1.0,
        )
        sol = solve_coupled_block_system(coupled, ss)
        assert sol.residual_norm < 1e-8

        b_cells = _b_per_cell(vs, sol.a, mesh.n_cells)
        vols = np.array([mesh.cell_volume(c) for c in range(mesh.n_cells)])
        w = vols / vols.sum()
        b_mean = w @ b_cells
        h_mean = nu * (b_mean - np.array([0.0, 0.0, Br]))  # H = ν(B − B_r)

        h_expect = -Br / (mu_rec + 2.0)
        b_expect = 2.0 * Br / (mu_rec + 2.0)
        assert abs(h_mean[2] - h_expect) < tol, (mu_rec, h_mean[2], h_expect)
        assert abs(b_mean[2] - b_expect) < tol, (mu_rec, b_mean[2], b_expect)
        # Поперечные компоненты ≈ 0 (изотропия отклика сферы).
        assert abs(h_mean[0]) < 1e-2
        assert abs(h_mean[1]) < 1e-2


# --------------------------------------------------------------------------------------
# Уровень 3 — ВЗАИМОДЕЙСТВИЕ магнит ↔ железо (двухобластная задача в открытой области).
# Куб: нижняя половина — магнит (B_r=ẑ, ν=1), верхняя — линейное мягкое железо (B_r=0,
# ν=1/μ_iron). Физика «магнитного keeper'а»: высоко-μ железо на пути потока СНИЖАЕТ
# саморазмагничивающее поле магнита (рабочая точка ползёт к B_r) и КОНЦЕНТРИРУЕТ поток.
# Проверяем знак и МОНОТОННОСТЬ по μ_iron (строгая физика без замкнутой формы).
# --------------------------------------------------------------------------------------
def _magnet_iron_cube(mu_iron: float, n: int = 3):
    m_vec = np.array([0.0, 0.0, 1.0])  # B_r магнита
    mesh = build_structured_unit_cube_tetra_mesh(n)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)

    is_magnet = np.array(
        [mesh.cell_centroid(c)[2] < 0.5 for c in range(mesh.n_cells)]
    )
    nu_cells = np.where(is_magnet, 1.0, 1.0 / mu_iron)

    # ν·B_r: только в ячейках магнита (ν=1 там), ноль в железе.
    nu_br = np.zeros((mesh.n_cells, 3))
    nu_br[is_magnet] = m_vec
    f_br = assemble_magnetization_rhs(mesh, vs, nu_br)

    coupled = assemble_coupled_block_system(
        ci, vs, ss, nu=nu_cells, j_fn=_zero_j, extra_vector_rhs=f_br, mu0=1.0
    )
    sol = solve_coupled_block_system(coupled, ss)
    assert sol.residual_norm < 1e-8

    b_cells = _b_per_cell(vs, sol.a, mesh.n_cells)
    vols = np.array([mesh.cell_volume(c) for c in range(mesh.n_cells)])

    # H в магните: ν=1, B_r=ẑ ⇒ H = B − ẑ.
    mag = is_magnet
    wm = vols[mag] / vols[mag].sum()
    hz_magnet = float(wm @ (b_cells[mag, 2] - 1.0))

    # Средняя B_z в области «железа» (верхняя половина).
    iron = ~is_magnet
    wi = vols[iron] / vols[iron].sum()
    bz_iron = float(wi @ b_cells[iron, 2])
    return hz_magnet, bz_iron


def test_iron_keeper_reduces_magnet_demag_and_concentrates_flux() -> None:
    # μ_iron=1 ⇒ верхняя половина = вакуум (контроль); затем растим проницаемость.
    hz_vac, bz_vac = _magnet_iron_cube(1.0)
    hz_mid, bz_mid = _magnet_iron_cube(10.0)
    hz_iron, bz_iron = _magnet_iron_cube(1000.0)

    # (1) Саморазмагничивающее поле магнита отрицательно во всех случаях.
    assert hz_vac < 0.0 and hz_mid < 0.0 and hz_iron < 0.0

    # (2) KEEPER-ЭФФЕКТ (сторона магнита): железо на пути потока МОНОТОННО ослабляет
    # саморазмагничивание магнита (рабочая точка ползёт к B_r) по всему диапазону μ_iron.
    # Это ключевой признак того, что магнит «чувствует» железо.
    assert hz_iron > hz_mid > hz_vac          # ближе к нулю = слабее размагничивание
    assert abs(hz_iron) < abs(hz_vac)

    # (3) ПРОВОДИМОСТЬ ПОТОКА (сторона железа): введение железа резко повышает среднюю
    # B_z в верхней половине против вакуума (онсет), затем эффект НАСЫЩАЕТСЯ при μ_r≳10
    # (физика: реальному мягкому железу μ_r~10³–10⁴ избыток проницаемости прироста не
    # даёт; при экстремальном μ возвратный поток внутри железа чуть снижает объёмное
    # среднее). Поэтому проверяем не монотонность, а заметную и устойчивую концентрацию.
    assert bz_mid > 1.3 * bz_vac              # +30%+ к средней B_z при μ_iron=10
    assert bz_iron > 1.3 * bz_vac             # остаётся повышенной и при μ_iron=1000
