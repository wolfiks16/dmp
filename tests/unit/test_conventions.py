# -*- coding: utf-8 -*-
"""
КОНВЕНЦИИ kV ↔ K_e и потолок инвертора + проверки правдоподобия конструкции.

Аудит 2026-09-03: в сценарии стояло `K_e = 60/(2π·kV)` — формула КОЛЛЕКТОРНОЙ машины,
применённая к трёхфазной. Расхождение с моделью из-за этого выглядело как 1,8 раза вместо
2,8–3,1. Здесь фиксируются правильные множители (каждый выводится независимо) и то, что
проектная проверка идёт по ПОТОЛКУ НАПРЯЖЕНИЯ, а не по паспортному kV.
"""
from __future__ import annotations

import math

import pytest

from magcore.fem2d.machines.conventions import (
    KV_CONVENTIONS,
    ke_from_kv,
    kv_from_ke,
    kv_spread,
    max_speed_rpm,
    phase_voltage_limit,
    turns_for_speed,
)
from magcore.fem2d.verification import (
    check_current_density,
    check_permeance_coefficient,
    check_tangential_stress,
    check_thermal_loading,
    check_voltage_headroom,
)

C = 60.0 / (2.0 * math.pi)          # 9.5493 — перевод об/мин → рад/с


# ------------------------------------------------------------------ множители конвенций

@pytest.mark.parametrize("convention, expected_ke_times_kv", [
    ("phase_peak", C),                                   # 9.549
    ("phase_rms", math.sqrt(2.0) * C),                   # 13.50
    ("ll_peak", C / math.sqrt(3.0)),                     # 5.513
    ("ll_rms", math.sqrt(2.0) / math.sqrt(3.0) * C),     # 7.797
    ("bus_sixstep", (2.0 / math.pi) * C),                # 6.079
    ("bus_svpwm", C / math.sqrt(3.0)),                   # 5.513
])
def test_kv_conversion_factors(convention: str, expected_ke_times_kv: float) -> None:
    """K_e·kV = C·(60/2π) — каждый множитель выведен независимо от реализации."""
    for kv in (100.0, 350.0, 1200.0):
        assert ke_from_kv(kv, convention) * kv == pytest.approx(expected_ke_times_kv, rel=1e-12)


def test_svpwm_is_identical_to_line_to_line_peak() -> None:
    """Предел линейной зоны SVPWM даёт V̂_лин = U_dc ровно ⇒ конвенции тождественны."""
    assert ke_from_kv(350.0, "bus_svpwm") == pytest.approx(ke_from_kv(350.0, "ll_peak"), rel=1e-15)


@pytest.mark.parametrize("convention", sorted(KV_CONVENTIONS))
def test_kv_roundtrip(convention: str) -> None:
    assert kv_from_ke(ke_from_kv(350.0, convention), convention) == pytest.approx(350.0, rel=1e-12)


def test_convention_spread_is_the_real_uncertainty() -> None:
    """
    Разброс между конвенциями — не мелочь: между двумя РЕАЛЬНО применяемыми паспортными
    практиками ~10 %, а если производитель перепутал действующее с амплитудным — до √2.
    """
    s = kv_spread(350.0)
    # (2/π) / (1/√3) = 2√3/π ≈ 1,103 — те самые «~10 %» между паспортными практиками
    assert s["bus_sixstep"] / s["bus_svpwm"] == pytest.approx(2.0 * math.sqrt(3.0) / math.pi,
                                                              rel=1e-12)
    assert s["phase_rms"] / s["phase_peak"] == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert max(s.values()) / min(s.values()) > 2.0


def test_unknown_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        ke_from_kv(350.0, "как-нибудь")
    with pytest.raises(ValueError):
        phase_voltage_limit(22.2, "как-нибудь")


# ------------------------------------------------------------------ потолок инвертора

@pytest.mark.parametrize("modulation, factor", [
    ("spwm", 0.5), ("svpwm", 1.0 / math.sqrt(3.0)), ("six_step", 2.0 / math.pi),
])
def test_inverter_limits(modulation: str, factor: float) -> None:
    assert phase_voltage_limit(22.2, modulation) == pytest.approx(factor * 22.2, rel=1e-12)


def test_modulation_ordering() -> None:
    """Шеститакт > SVPWM > синусоидальная ШИМ — иначе перепутаны множители."""
    v = [phase_voltage_limit(22.2, m) for m in ("spwm", "svpwm", "six_step")]
    assert v[0] < v[1] < v[2]


def test_audit_case_7000_rpm_on_6s_is_unreachable() -> None:
    """
    СЛУЧАЙ АУДИТА. Обмотка 20 вит./паз дала K_e = 0,0491 В·с/рад. На шине 6S (22,2 В)
    заявленные 7000 об/мин недостижимы при ЛЮБОЙ модуляции — недостача 2,5…3,2 раза.
    """
    ke, u_dc = 0.0491, 22.2
    n_max = {m: max_speed_rpm(ke, u_dc, m) for m in ("spwm", "svpwm", "six_step")}
    assert all(n < 3000.0 for n in n_max.values())
    assert 7000.0 / n_max["six_step"] > 2.5
    assert 7000.0 / n_max["spwm"] < 3.3


def test_turns_for_speed_inverts_the_voltage_ceiling() -> None:
    """Сколько витков допускает шина: K_e линейна по виткам, ответ проверяем подстановкой."""
    n = turns_for_speed(0.0491, 20.0, speed_rpm=7000.0, u_dc=22.2, modulation="six_step")
    assert 7.0 < n < 8.5
    ke_new = 0.0491 * n / 20.0
    assert max_speed_rpm(ke_new, 22.2, "six_step") == pytest.approx(7000.0, rel=1e-9)
    # с запасом на падения витков должно быть МЕНЬШЕ
    assert turns_for_speed(0.0491, 20.0, speed_rpm=7000.0, u_dc=22.2,
                           modulation="six_step", margin=1.25) < n


# ------------------------------------------------------------------ правдоподобие режима

def test_audit_operating_point_fails_every_plausibility_check() -> None:
    """
    СЛУЧАЙ АУДИТА целиком: J = 33,4 А/мм², A ≈ 45 кА/м, σ = 30,6 кПа, P_c = 2,14,
    K_e = 0,0491 при 7000 об/мин на 6S. Ожидание: длительным этот режим не является,
    касательное напряжение при этом НОРМАЛЬНО (ломается именно плотность тока).
    """
    assert not check_current_density(33.4, "continuous").ok
    assert check_current_density(33.4, "peak").ok
    assert not check_thermal_loading(45.0e3, 33.4e6).ok            # AJ ≈ 150e10 ≫ 42e10
    assert check_tangential_stress(30.6e3).ok                      # 21…48 кПа — в норме
    # Ниже полосы — НЕ провал: у малой машины длительное σ и не может быть промышленным
    below = check_tangential_stress(11.0e3)
    assert below.ok and "ниже" in below.value
    # Выше верхней границы — провал: нагрузка не длительная
    assert not check_tangential_stress(60.0e3).ok
    assert not check_permeance_coefficient(2.14).ok
    assert not check_voltage_headroom(0.0491, 7000.0, 22.2, "svpwm").ok


def test_plausibility_checks_pass_on_a_sane_design() -> None:
    """Здоровая точка: 12 А/мм², A = 25 кА/м, σ = 25 кПа, P_c = 4,5, K_e под шину."""
    assert check_current_density(12.0, "continuous").ok
    assert check_thermal_loading(25.0e3, 12.0e6).ok                # AJ = 30e10 < 42e10
    assert check_tangential_stress(25.0e3).ok
    assert check_permeance_coefficient(4.5).ok
    assert check_voltage_headroom(0.0170, 7000.0, 22.2, "svpwm").ok


def test_thermal_loading_is_the_book_criterion() -> None:
    """AJ — критерий Pyrhönen (пример 7.3): порог 42,25e10 А²/м³ для машин с ПМ."""
    assert check_thermal_loading(35.0e3, 12.0e6).ok                # 42.0e10 — впритык под
    assert not check_thermal_loading(35.0e3, 13.0e6).ok            # 45.5e10 — над


def test_failed_checks_carry_actionable_hints() -> None:
    """Провал обязан объяснять, что делать: иначе проверка бесполезна на защите."""
    for c in (check_current_density(33.4, "continuous"),
              check_thermal_loading(45.0e3, 33.4e6),
              check_permeance_coefficient(2.14),
              check_voltage_headroom(0.0491, 7000.0, 22.2, "svpwm")):
        assert not c.ok and len(c.hint) > 30
