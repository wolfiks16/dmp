import math
from dataclasses import replace

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines.pmsm_outrunner import (
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)
from magcore.fem2d.machines.rotor_sweep import (
    RotorDamage,
    cogging_period_angles,
    electrical_period_angles,
    sweep_rotor,
)

# ВРАЩЕНИЕ РОТОРА. Оракулы опираются на симметрию и точные тождества, а не на эталонные числа:
#   * поворот на ЦЕЛОЕ число полюсных делений (чётное) обязан вернуть ТУ ЖЕ машину;
#   * зубцовый момент за период обязан иметь НУЛЕВОЕ среднее (магниты не совершают работы
#     за оборот — иначе получился бы вечный двигатель);
#   * РАВНОМЕРНОЕ повреждение r масштабирует λ_m ровно в r раз и переносится между сетками
#     ТОЧНО (проверяет перенос повреждения в роторной системе);
#   * ток должен ехать ВМЕСТЕ с ротором: при неподвижном угле тока средний момент вырождается
#     в ноль (ротор проезжает под током весь период) — тест ловит именно это.

MESH = 0.0035
NPS = 20.0
T_HOT = 150.0


def _params(**kw):
    return OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=MESH, **kw)


def _magnet():
    return n42sh_magnet((1.0, 0.0, 0.0))


# -------------------------------------------------------------- геометрия под поворотом

def test_rotation_preserves_magnet_area_and_moves_poles():
    base = build_outrunner_spm_pmsm(_params())
    turned = build_outrunner_spm_pmsm(_params(rotor_angle=0.1))

    def mag_area(g):
        return float(sum(g.mesh.cell_area(int(c)) for c in np.where(g.mask(Region.MAGNET))[0]))

    # Поворот — это движение, а не изменение: площадь магнитов сохраняется (с точностью
    # до перестроения сетки), а вот сама картина полюсов сдвигается.
    assert abs(mag_area(turned) / mag_area(base) - 1.0) < 5e-3
    assert not np.allclose(base.region[: min(base.mesh.n_cells, turned.mesh.n_cells)],
                           turned.region[: min(base.mesh.n_cells, turned.mesh.n_cells)])


def test_full_pole_pair_rotation_reproduces_the_same_machine():
    # Поворот на ДВА полюсных деления возвращает ту же полярность и ту же геометрию ⇒ момент
    # обязан совпасть. Это проверяет и смещение дуг, и согласованную с ним классификацию:
    # если бы они разъехались, полярность «поехала» бы и момент изменился.
    p = _params()
    two_poles = 2.0 * (2.0 * math.pi / p.n_poles)
    sw = sweep_rotor(p, _magnet(), m270_35a_bh_curve(),
                     angles=[0.0, two_poles], i_peak=20.0, gamma_elec=3.93,
                     turns_per_slot=NPS, no_load=False, T=T_HOT)
    assert sw.all_converged
    assert abs(sw.torque[1] - sw.torque[0]) / abs(sw.torque[0]) < 0.05


# ------------------------------------------------------------------ зубцовый момент

def test_cogging_torque_has_zero_mean_over_its_period():
    # За период зубцового момента магниты не совершают работы ⇒ среднее строго около нуля.
    # Ненулевое среднее означало бы момент «из ничего» при отсутствии тока.
    p = _params()
    sw = sweep_rotor(p, _magnet(), m270_35a_bh_curve(),
                     angles=cogging_period_angles(p, 8), i_peak=0.0,
                     turns_per_slot=0.0, no_load=False, T=T_HOT)
    assert sw.all_converged
    span = float(sw.torque.max() - sw.torque.min())
    assert span > 0.0                                   # пульсация вообще есть
    assert abs(sw.torque_mean) < 0.15 * span            # но среднее около нуля


def test_cogging_sampling_helper_is_finer_than_electrical():
    # Период зубцового момента = 2π/НОК(12,14) много мельче электрического; выборка по
    # электрическому периоду его алиасит (для 12/14 шаг совпадает с периодом ровно).
    p = _params()
    cog = cogging_period_angles(p, 8)
    ele = electrical_period_angles(p, 8)
    assert cog[1] < ele[1] / 5.0
    assert abs(cog[1] * 8 - 2 * math.pi / math.lcm(12, 14)) < 1e-15


# --------------------------------------------------- ток едет вместе с ротором

def test_current_follows_the_rotor_so_mean_torque_survives():
    # В синхронной машине привод держит ток ОТНОСИТЕЛЬНО ротора. Если этого не делать, ротор
    # проезжает под неподвижным током весь период и средний момент вырождается в ноль.
    # Проверяем, что средний момент сравним с мгновенным при том же относительном угле.
    p = _params()
    sw = sweep_rotor(p, _magnet(), m270_35a_bh_curve(),
                     angles=electrical_period_angles(p, 6), i_peak=20.0, gamma_elec=3.93,
                     turns_per_slot=NPS, no_load=False, T=T_HOT)
    assert sw.all_converged
    assert sw.torque_mean > 0.5 * float(sw.torque.max())     # не вырожден в ноль
    assert 0.0 < sw.torque_ripple < 1.0                      # пульсации есть, но не безумные


# ------------------------------------------------ повреждение едет вместе с ротором

def test_uniform_damage_transfers_exactly_to_any_rotor_angle():
    # Равномерное повреждение обязано переноситься на повёрнутую сетку ТОЧНО: любое отличие
    # означало бы, что перенос ищет соседа не в роторной системе координат.
    base = build_outrunner_spm_pmsm(_params())
    n = int(base.mask(Region.MAGNET).sum())
    dmg = RotorDamage(base, np.full(n, 0.73))
    for angle in (0.0, 0.05, 0.31, -0.2):
        turned = build_outrunner_spm_pmsm(_params(rotor_angle=angle))
        assert np.allclose(dmg.sample(turned), 0.73)


def test_damage_pattern_rotates_with_the_rotor():
    # Неравномерное повреждение (половина полюсов) при повороте должно ехать ВМЕСТЕ с
    # магнитами: доля повреждённых ячеек сохраняется, а их положение в лаборатории — нет.
    base = build_outrunner_spm_pmsm(_params())
    idx = np.where(base.mask(Region.MAGNET))[0]
    cen = np.array([base.mesh.cell_centroid(int(c)) for c in idx])
    r = np.where(cen[:, 1] > 0.0, 0.5, 1.0)               # повреждена верхняя половина
    dmg = RotorDamage(base, r)

    turned = build_outrunner_spm_pmsm(_params(rotor_angle=math.pi))
    got = dmg.sample(turned)
    frac_base = float(np.mean(r < 1.0))
    assert abs(float(np.mean(got < 1.0)) - frac_base) < 0.05      # доля сохранилась

    # После поворота на π повреждённые ячейки должны оказаться СНИЗУ.
    idx_t = np.where(turned.mask(Region.MAGNET))[0]
    cen_t = np.array([turned.mesh.cell_centroid(int(c)) for c in idx_t])
    assert float(np.mean(cen_t[got < 1.0, 1])) < 0.0


def test_shape_of_retention_is_validated():
    base = build_outrunner_spm_pmsm(_params())
    n = int(base.mask(Region.MAGNET).sum())
    with pytest.raises(ValueError, match="ячейкам магнита"):
        RotorDamage(base, np.ones(n + 2))


# ------------------------------------------------- потокосцепление и постоянные машины

def test_uniform_damage_scales_flux_linkage_exactly():
    # Источник магнита линеен по B_r ⇒ равномерная потеря доли r масштабирует первую
    # гармонику потокосцепления РОВНО в r раз. Это точное тождество, а не эталонное число,
    # и оно проверяет всю цепочку: перенос повреждения → источник → λ → извлечение гармоники.
    p = _params()
    magnet, steel = _magnet(), m270_35a_bh_curve()
    angles = electrical_period_angles(p, 6)
    base = build_outrunner_spm_pmsm(p)
    n = int(base.mask(Region.MAGNET).sum())

    kw = dict(angles=angles, i_peak=0.0, turns_per_slot=NPS, no_load=True, T=T_HOT)
    healthy = sweep_rotor(p, magnet, steel, **kw)
    hurt = sweep_rotor(p, magnet, steel, damage=RotorDamage(base, np.full(n, 0.8)), **kw)
    assert healthy.all_converged and hurt.all_converged

    ratio = hurt.flux_linkage_fundamental() / healthy.flux_linkage_fundamental()
    assert abs(ratio - 0.8) < 5e-3
    assert abs(hurt.torque_constant() / healthy.torque_constant() - 0.8) < 5e-3
    # K_t и K_e — разные величины, отличаются ровно в 3/2 раза.
    assert abs(healthy.torque_constant() / healthy.emf_constant() - 1.5) < 1e-12


def test_flux_linkage_requires_no_load_pass():
    p = _params()
    sw = sweep_rotor(p, _magnet(), m270_35a_bh_curve(), angles=[0.0, 0.05],
                     i_peak=20.0, gamma_elec=3.93, turns_per_slot=NPS,
                     no_load=False, T=T_HOT)
    with pytest.raises(ValueError, match="no_load"):
        sw.flux_linkage_fundamental()
