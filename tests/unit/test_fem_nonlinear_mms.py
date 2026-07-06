from __future__ import annotations

import math

import numpy as np

from magcore.femcore.manufactured import (
    manufactured_A_ref,
    manufactured_curl_ref,
    manufactured_nonlinear_chord_nu,
    manufactured_nonlinear_rhs,
)
from magcore.femcore.nonlinear import solve_nonlinear_mixed_picard
from magcore.femcore.post import (
    l2_curl_error_at_cell_centroids,
    l2_error_at_cell_centroids,
)
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh

NU0, C = 1.0, 5.0  # ν(|B|)=ν0(1+c|B|²); A_ref даёт переменный |B| ⇒ нелинейность активна


def _solve_nonlinear_mms(n: int):
    mesh = build_structured_unit_cube_tetra_mesh(n)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    res = solve_nonlinear_mixed_picard(
        mesh, vs, ss,
        nu_of_B=lambda B: manufactured_nonlinear_chord_nu(B, NU0, C),
        J_fn=lambda x: manufactured_nonlinear_rhs(x, NU0, C),
        nu_init=np.full(mesh.n_cells, NU0),
        tol=1e-8, max_iter=60,
    )
    err_a = l2_error_at_cell_centroids(vs, res.a, manufactured_A_ref)
    err_curl = l2_curl_error_at_cell_centroids(vs, res.a, manufactured_curl_ref)
    return res, err_a, err_curl


def test_nonlinear_mms_recovers_exact_solution_with_order_one() -> None:
    """
    НЕЗАВИСИМЫЙ оракул корректности нелинейного решателя (метод многообразных решений).
    Источник J = curl(ν(|B|)·curl A_ref) выведен аналитически ⇒ точное решение есть
    A_ref. Решатель ДОЛЖЕН его восстановить, а ошибка — убывать с порядком ~1 (Неделек
    низшего порядка) при измельчении. Ошибка знака/масштаба в обновлении ν проявится
    как незатухающая ошибка (в отличие от тавтологической «самосогласованности»).
    """
    R = {}
    for n in (2, 4, 6):
        res, ea, ec = _solve_nonlinear_mms(n)
        assert res.converged
        # нелинейность реально активна: ν варьируется по ячейкам (|B| неоднороден).
        assert res.nu_cells.max() > res.nu_cells.min() + 1e-6
        assert res.nu_cells.max() > NU0 + 1e-6
        R[n] = (ea, ec, float(np.linalg.norm(res.p)), float(np.linalg.norm(res.a)))

    # A_ref дивергентно-свободно + n×A=0 ⇒ калибровочный множитель p ≈ 0 (как в линейном).
    for n in (2, 4, 6):
        ea, ec, p_norm, a_norm = R[n]
        assert a_norm > 0.0
        assert p_norm / a_norm < 1e-8

    # Восстановление точного решения: монотонное убывание ошибки при измельчении.
    assert R[4][0] < R[2][0] and R[6][0] < R[4][0]
    assert R[4][1] < R[2][1] and R[6][1] < R[4][1]
    # Малая абсолютная ошибка на тонкой сетке (восстановили именно A_ref, не «что-то»).
    assert R[6][0] < 8e-3
    assert R[6][1] < 6e-2

    # Порядок ~1 (H(curl) и L²(A) — оба порядка 1 для Неделека 1-го рода).
    # Измельчение 4→6 (отношение h: 6/4=1.5), порядок = log(e4/e6)/log(1.5).
    rate_a = math.log(R[4][0] / R[6][0]) / math.log(6.0 / 4.0)
    rate_curl = math.log(R[4][1] / R[6][1]) / math.log(6.0 / 4.0)
    assert 0.7 < rate_a < 1.4, rate_a
    assert 0.7 < rate_curl < 1.4, rate_curl
