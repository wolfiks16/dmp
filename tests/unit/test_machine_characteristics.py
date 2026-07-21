import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines.characteristics import (
    CharacteristicsComparison,
    machine_characteristics,
    magnet_mass,
)
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, Region
from magcore.fem2d.machines.postproc import (
    back_emf_constant,
    flux_linkage_amplitude,
    torque_constant,
)
from magcore.fem2d.machines.scenario import machine_scenario

# Характеристики машины: то, что отвечает «сколько это стоит».
# ГЛАВНЫЙ ОРАКУЛ — сверка ДВУХ НЕЗАВИСИМЫХ путей к одной величине: момент из ПОЛЯ
# (тензор Максвелла по зазору, метод Арккио) против моментной постоянной из
# ПОТОКОСЦЕПЛЕНИЯ обмотки. Пути не имеют общего кода, поэтому их совпадение проверяет
# всю цепочку разом — и именно оно вскрыло потерянный множитель 2/3 в амплитуде Кларка
# (расхождение было 34 %, стало 0.2 %).

MESH = 0.0030
NPS = 20.0


def _scenario():
    return machine_scenario(
        OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=MESH),
        n42sh_magnet((1.0, 0.0, 0.0)),
        m270_35a_bh_curve(),
    )


# ------------------------------------------------------- амплитуда Кларка и постоянные

def test_clarke_amplitude_on_known_balanced_set():
    # Набор с ЗАВЕДОМО известной амплитудой: [λcos θ, λcos(θ−2π/3), λcos(θ+2π/3)] ⇒ ответ λ
    # при любом θ. Без множителя 2/3 функция возвращала 1.5·λ, и через неё завышались и λ_m,
    # и K_e — числа, которые идут в текст работы.
    for lam0 in (1.0, 0.0073):
        for theta in (0.0, 0.3, 1.1, 2.7, -2.0):
            lam = lam0 * np.array([
                np.cos(theta), np.cos(theta - 2 * np.pi / 3), np.cos(theta + 2 * np.pi / 3)
            ])
            assert abs(flux_linkage_amplitude(lam) - lam0) < 1e-12


def test_torque_and_emf_constants_differ_by_three_halves():
    # K_t = (3/2)·p·λ_m и K_e = p·λ_m — РАЗНЫЕ величины (вклад трёх фаз). Их отождествление
    # даёт полуторакратную ошибку в одной из них.
    g = _scenario().geometry
    lam = np.array([0.0060, -0.0030, -0.0030])
    assert abs(torque_constant(g, lam) / back_emf_constant(g, lam) - 1.5) < 1e-12
    p = g.params.n_poles // 2
    assert abs(back_emf_constant(g, lam) - p * flux_linkage_amplitude(lam)) < 1e-15


def test_arkkio_torque_matches_torque_constant_from_flux_linkage():
    # НЕЗАВИСИМЫЕ ПУТИ: момент из поля (Максвелл по зазору) против (3/2)·p·λ_m из
    # потокосцепления. Ток берётся умеренным: слишком малый — момент тонет в дискретизации,
    # слишком большой — насыщение уводит момент ниже линейной оценки.
    sc = _scenario()
    I = 20.0
    swept = [
        machine_characteristics(sc, i_peak=I, turns_per_slot=NPS, gamma_elec=float(g), T=150.0)
        for g in np.linspace(3.7, 4.2, 9)
    ]
    best = max(swept, key=lambda c: c.torque)
    assert best.converged
    assert abs(best.torque / I - best.torque_constant) / best.torque_constant < 0.03


# --------------------------------------------------------------- масса и удельные показатели

def test_magnet_mass_matches_geometry_and_scales_with_density():
    sc = _scenario()
    g = sc.geometry
    area = float(sum(g.mesh.cell_area(int(c)) for c in np.where(g.mask(Region.MAGNET))[0]))
    m = magnet_mass(g, density=7500.0)
    assert abs(m - area * g.params.axial_length * 7500.0) < 1e-15
    assert abs(magnet_mass(g, density=8400.0) / m - 8400.0 / 7500.0) < 1e-12
    assert 0.0 < m < 1.0                                  # десятки граммов, не килограммы

    c = machine_characteristics(sc, i_peak=20.0, turns_per_slot=NPS, gamma_elec=3.93, T=150.0)
    assert abs(c.torque_per_magnet_mass - c.torque / c.magnet_mass) < 1e-12


# ------------------------------------------------------------------ влияние повреждения

def test_damaged_magnet_produces_less_torque_and_flux():
    # Повреждение обязано снижать И потокосцепление, И момент. Берём РАВНОМЕРНОЕ повреждение
    # (r одинакова по ячейкам) — при нём величины при одном положении ротора представительны,
    # а падение λ_m должно совпасть с долей потери ТОЧНО (линейность источника по B_r).
    sc = _scenario()
    n = int(sc.geometry.mask(Region.MAGNET).sum())
    kw = dict(i_peak=20.0, turns_per_slot=NPS, gamma_elec=3.93, T=150.0)
    before = machine_characteristics(sc, retention=None, **kw)
    after = machine_characteristics(sc, retention=np.full(n, 0.8), **kw)

    assert after.flux_linkage < before.flux_linkage
    assert after.torque < before.torque
    assert abs(after.flux_linkage / before.flux_linkage - 0.8) < 1e-3
    assert abs(after.torque_constant / before.torque_constant - 0.8) < 1e-3

    cmp = CharacteristicsComparison(before, after, fundamental_ratio=0.8)
    assert abs(cmp.flux_linkage_drop - 0.2) < 1e-3
    assert abs(cmp.torque_constant_drop - 0.2) < 1e-3
    # При равномерном повреждении несимметрии нет ⇒ прямой расчёт и инвариантная оценка
    # обязаны совпасть; расхождение здесь означало бы ошибку в одной из двух метрик.
    assert cmp.asymmetry_indicator < 1e-3


def test_retention_shape_is_validated():
    sc = _scenario()
    n = int(sc.geometry.mask(Region.MAGNET).sum())
    with pytest.raises(ValueError, match="ячейкам магнита"):
        machine_characteristics(
            sc, i_peak=10.0, turns_per_slot=NPS, retention=np.ones(n + 3), T=150.0
        )
