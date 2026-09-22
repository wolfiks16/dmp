import numpy as np
import pytest

from magcore.domain.steel_curves import (
    SteelBHCurve,
    m270_35a_bh_curve,
    m270_35a_cogent_bh_curve,
    steel10_bh_curve,
)
from magcore.fem2d.verification import (
    FAIL,
    PASS,
    SKIP,
    VerificationReport,
    check_airgap_resolution,
    check_convergence,
    check_energy_balance,
    check_magnet_model_range,
    check_runaway_margin,
    check_steel_saturation,
    check_time_step,
)
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh

# Блок проверок ловит РЕАЛЬНЫЕ случаи, на которых мы обжигались в этом проекте:
# несошедшаяся итерация с правдоподобным полем; оборванная кривая B(H); грубый шаг dt.


def test_convergence_check_catches_stalled_iteration():
    assert check_convergence(True, 158, 9.6e-7).status == PASS
    # реальный случай ДП25: невязка застряла на 2.9e-2, поле «выглядело нормально»
    bad = check_convergence(False, 250, 2.9e-2)
    assert bad.status == FAIL and not bad.ok and bad.hint


def test_steel_saturation_check_catches_truncated_curve():
    # ИСТОРИЯ: кривая Cogent, обрезанная на 1.8 Тл (μ_r≈15 на конце), развалила 32 точки карты.
    truncated = SteelBHCurve(
        curve_id="обрезанная", name="M270 до 1.8 Тл",
        H_values=np.array([0.0, 596.0, 1700.0, 3880.0, 7160.0, 11600.0]),
        B_values=np.array([0.0, 1.4, 1.5, 1.6, 1.7, 1.8]))
    assert check_steel_saturation([truncated]).status == FAIL
    # исправленные кривые проходят
    assert check_steel_saturation([m270_35a_cogent_bh_curve(), steel10_bh_curve()]).status == PASS
    assert check_steel_saturation([None]).status == PASS      # None пропускается


def test_airgap_resolution_check():
    mesh = build_structured_rectangle_tri_mesh(20, 20, x0=0.0, x1=0.01, y0=0.0, y1=0.01)
    allc = np.ones(mesh.n_cells, dtype=bool)
    h = np.sqrt(2.0 * np.mean([mesh.cell_area(c) for c in range(mesh.n_cells)]))
    assert check_airgap_resolution(mesh, allc, 10.0 * h).status == PASS     # толстый зазор
    # реальный случай ДП25: зазор 0.268 мм при элементе 0.30 мм
    thin = check_airgap_resolution(mesh, allc, 0.5 * h)
    assert thin.status == FAIL and "зазор" in thin.hint.lower()
    assert check_airgap_resolution(mesh, np.zeros(mesh.n_cells, bool), 1e-3).status == SKIP


def test_energy_balance_check():
    dt = 0.5
    lp = np.array([100.0, 100.0, 100.0, 100.0])
    of = np.array([0.0, 10.0, 20.0, 30.0])
    st = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    for i in range(1, 4):                                   # построить точный баланс
        st[i] = st[i - 1] + dt * (lp[i] - of[i])
    assert check_energy_balance(st, lp, of, dt).status == PASS
    st_bad = st.copy(); st_bad[-1] *= 1.5
    assert check_energy_balance(st_bad, lp, of, dt).status == FAIL
    assert check_energy_balance([1.0], [1.0], [0.0], dt).status == SKIP


def test_time_step_and_runaway_and_magnet_range():
    assert check_time_step(0.5).status == PASS
    assert check_time_step(5.0).status == FAIL             # реальный случай: ошибка 51 °C
    assert check_runaway_margin(0.6, 1.0).status == PASS
    assert check_runaway_margin(1.2, 1.0).status == FAIL
    assert check_runaway_margin(1.0, 0.0).status == SKIP
    assert check_magnet_model_range(150.0, 200.0).status == PASS
    assert check_magnet_model_range(210.0, 200.0).status == FAIL


def test_report_aggregates_and_serializes():
    ok = VerificationReport([check_time_step(0.5), check_convergence(True, 10, 1e-8)])
    assert ok.trustworthy and not ok.failed
    d = ok.to_dict()
    assert d["trustworthy"] and d["summary"] == "все проверки пройдены"
    assert len(d["checks"]) == 2 and set(d["checks"][0]) == {"name", "status", "value", "hint"}

    bad = VerificationReport([check_time_step(5.0), check_convergence(False, 250, 1e-2)])
    assert not bad.trustworthy and len(bad.failed) == 2
    assert "НЕ ПРОЙДЕНО" in bad.to_dict()["summary"]

    # пропущенная проверка НЕ считается провалом
    skipped = VerificationReport([check_runaway_margin(1.0, 0.0)])
    assert skipped.trustworthy


def test_old_representative_m270_would_fail_saturation_check():
    """
    Историческая проверка: «представительная» M270 (была по умолчанию в приложении)
    доходит до 2.21 Тл — насыщение достигнуто, проверку проходит. Но она в 2.4 раза
    МЯГЧЕ реального листа при 1.5 Тл — это уже вопрос точности данных, а не сходимости,
    и ловится сверкой с datasheet, а не этой проверкой.
    """
    soft = m270_35a_bh_curve()
    assert check_steel_saturation([soft]).status == PASS          # по насыщению — ок
    h_soft = np.interp(1.5, soft.B_values, soft.H_values)
    h_real = np.interp(1.5, m270_35a_cogent_bh_curve().B_values,
                       m270_35a_cogent_bh_curve().H_values)
    assert h_real > 2.0 * h_soft                                   # но данные заметно мягче
