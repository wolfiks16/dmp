import math

import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines.pmsm_outrunner import Region
from magcore.fem2d.machines.scenario import MachineScenario
from magcore.fem2d.machines.spoke_pmsm import (
    MagnetShape,
    SpokeMotorParams,
    build_spoke_pmsm,
)
from magcore.fem2d.machines.winding import star_of_slots_layout

# НОВЫЙ спицевой генератор (реальная схема заказчика). Оракулы — площади против аналитики,
# зазор, число полюсов/зубьев, симметрия, и подключение к готовой физике. Главный оракул
# КОНФОРМНОСТИ (требование «сразу точно»): площадь магнита совпадает с аналитической до долей
# процента ⇒ сетка лежит РОВНО по контуру магнита, тег региона точный.

MESH = 0.6


def _params(shape=MagnetShape.TRUNCATED_SECTOR, **kw):
    kw.setdefault("mesh_size_mm", MESH)
    return SpokeMotorParams(magnet_shape=shape, **kw)


def _region_area(g, region) -> float:
    idx = np.where(g.region == int(region))[0]
    return float(sum(g.mesh.cell_area(int(c)) for c in idx))


# ------------------------------------------------------- конформность: площади магнита

def test_sector_magnet_area_matches_analytic_exactly():
    # Дуговой сектор: площадь = n·½(R_out²−R_in²)·угол. Совпадение до долей процента доказывает,
    # что сетка конформна контуру магнита (иначе «ступеньки» дали бы заметную ошибку площади).
    p = _params(MagnetShape.SECTOR)
    g = build_spoke_pmsm(p)
    a = p.sector_angle_deg * math.pi / 180.0
    analytic = p.n_poles * 0.5 * (p.R_mag_out ** 2 - p.R_mag_in ** 2) * a
    assert abs(_region_area(g, Region.MAGNET) - analytic) / analytic < 5e-3


def test_prism_magnet_area_matches_width_times_thickness():
    # Призма: площадь = n·(ширина·толщина). Точное совпадение — та же проверка конформности.
    p = _params(MagnetShape.PRISM)
    g = build_spoke_pmsm(p)
    analytic = p.n_poles * (p.prism_width_mm * 1e-3) * (p.prism_thickness_mm * 1e-3)
    assert abs(_region_area(g, Region.MAGNET) - analytic) / analytic < 5e-3


def test_truncated_magnet_area_matches_closed_form():
    # «Булка»: дуги сверху/снизу (R_in,R_out) + прямые стороны (±hw). Площадь через площадь
    # усечённого круга A(R)=πR²−2[R²·acos(hw/R)−hw√(R²−hw²)]; булка = A(R_out)−A(R_in).
    p = _params(MagnetShape.TRUNCATED_SECTOR)
    g = build_spoke_pmsm(p)
    hw = 0.5 * p.truncated_width_mm * 1e-3

    def disk_within_strip(R):
        return math.pi * R ** 2 - 2.0 * (R ** 2 * math.acos(hw / R) - hw * math.sqrt(R ** 2 - hw ** 2))

    # полоса |y|<=hw симметрична по x; магнит — только половина (+x, один полюс) ⇒ ×½
    one = 0.5 * (disk_within_strip(p.R_mag_out) - disk_within_strip(p.R_mag_in))
    analytic = p.n_poles * one
    assert abs(_region_area(g, Region.MAGNET) - analytic) / analytic < 5e-3


# ------------------------------------------------------------- геометрия и разметка

def test_gap_and_all_regions_present():
    g = build_spoke_pmsm(_params())
    p = g.params
    assert abs((p.R_mag_in - p.R_s_out) * 1e3 - 0.2) < 1e-6          # зазор ровно 0.2 мм
    for reg in Region:
        assert int(reg) in set(g.region.tolist()), f"нет региона {reg.name}"


def test_pole_and_tooth_counts():
    g = build_spoke_pmsm(_params())
    p = g.params
    # число полюсов = число секторов, куда попали магнитные ячейки
    idx = np.where(g.region == int(Region.MAGNET))[0]
    poles = set()
    for c in idx:
        cx, cy = g.mesh.cell_centroid(int(c))
        poles.add(round((math.atan2(cy, cx) - p.rotor_angle) / (2 * math.pi / p.n_poles)) % p.n_poles)
    assert len(poles) == p.n_poles == 14
    # число пазов (окон обмотки) = число уникальных slot_id
    assert len(set(g.slot_id[g.slot_id >= 0].tolist())) == p.n_teeth == 12


def test_easy_axis_is_radial_unit_and_alternates_sign():
    g = build_spoke_pmsm(_params())
    idx = np.where(g.region == int(Region.MAGNET))[0]
    ax = g.magnet_easy_axis[idx]
    cen = np.array([g.mesh.cell_centroid(int(c)) for c in idx])
    rad = cen / np.linalg.norm(cen, axis=1)[:, None]
    dots = np.einsum("ij,ij->i", ax, rad)                            # ось ⋅ радиаль = ±1
    assert np.allclose(np.abs(dots), 1.0, atol=1e-9)                 # единичная, радиальная
    assert (dots > 0).any() and (dots < 0).any()                    # обе полярности присутствуют


# ------------------------------------------------------------- вращение ротора

def test_rotation_preserves_magnet_area():
    a = _region_area(build_spoke_pmsm(_params()), Region.MAGNET)
    b = _region_area(build_spoke_pmsm(_params(rotor_angle=0.13)), Region.MAGNET)
    assert abs(b / a - 1.0) < 5e-3


# ------------------------------------------------------------- подключение к физике

def test_plugs_into_machine_physics_and_produces_field():
    # Генератор выдаёт MachineGeometry ⇒ готовая физика (машинный сценарий) должна решить
    # магнитостатику без правок и дать поле/потокосцепление/момент.
    g = build_spoke_pmsm(_params(mesh_size_mm=0.8))
    p = g.params
    lay = star_of_slots_layout(p.n_teeth, p.n_poles)
    sc = MachineScenario(geometry=g, magnet=n42sh_magnet((1.0, 0.0, 0.0)),
                         steel=m270_35a_bh_curve(), layout=lay)
    no_load = sc.solve(T=20.0, i_peak=0.0, max_iter=80)
    assert no_load.converged
    B = np.hypot(no_load.field.B_cells[:, 0], no_load.field.B_cells[:, 1])
    assert B.max() > 0.3                                             # магниты создают поле
    lam = sc.phase_flux_linkage(no_load, turns_per_slot=20.0)
    assert np.any(np.abs(lam) > 0.0)

    loaded = sc.solve(T=20.0, i_peak=30.0, gamma_elec=math.pi / 2, turns_per_slot=20.0, max_iter=80)
    assert loaded.converged
    assert abs(sc.torque(loaded)) > 0.0                             # ток даёт момент


# ------------------------------------------------------------- валидация параметров

def test_invalid_params_are_rejected():
    with pytest.raises(ValueError):
        build_spoke_pmsm(_params(n_poles=13))                       # нечётное число полюсов
    with pytest.raises(ValueError, match="возрастать"):
        SpokeMotorParams(D_magnet_in_mm=10.0).validate()           # магнит внутри статора
    with pytest.raises(ValueError, match="топорик"):
        SpokeMotorParams(shoe_width_mm=100.0).validate()           # башмаки перекрываются
