import numpy as np
import pytest

pytest.importorskip("gmsh")

from magcore.domain.magnet_model import n42sh_magnet  # noqa: E402
from magcore.domain.steel_curves import m270_35a_bh_curve  # noqa: E402
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, Region  # noqa: E402
from magcore.fem2d.machines.rotor_sweep import (  # noqa: E402
    RotorDamage,
    build_rotor_geometries,
    electrical_period_angles,
    sweep_rotor,
)
from magcore.fem2d.machines.scenario import machine_scenario  # noqa: E402
from magcore.fem2d.machines.thermal_scenario import (  # noqa: E402
    magnet_fundamental_ratio,
    run_machine_thermal_demag,
)

# ПРОКСИ ПАДЕНИЯ K_e ПРОТИВ ЧЕСТНОГО ПЕРЕСЧЁТА ПОЛЯ (п. 14 плана 2026-11-18).
# Прокси `magnet_fundamental_ratio` — первая гармоника ремнантности ротора с весом ПЛОЩАДИ ячеек. Честный K_e — первая
# (электрическая) гармоника потокосцепления фазы при прогонке ротора, повреждение едет вместе с магнитами
# (`sweep_rotor`, `RotorSweepResult.flux_linkage_fundamental`). Раньше прокси проверялась только на ОДНОРОДНОМ r (там она
# точна по построению). Здесь — неоднородное повреждение трёх видов, и для каждого заранее выведено, чего ждать:
#
#   * r постоянна в пределах полюса, но разная у разных полюсов. В линейной цепи вклад полюса k в проекцию на
#     e^{−ipθ} по полному обороту равен (−1)^k·e^{ipkα}·P₀ = P₀ (α = π/p), то есть ОДИН И ТОТ ЖЕ у всех полюсов ⇒
#     честный ответ = средняя r по полюсам. Прокси даёт то же ⇒ здесь она точна (остаток — насыщение стали);
#   * r зависит от РАДИУСА. Вклад ячейки в гармонику порядка p зависит от её радиуса ρ, а прокси считает все ячейки по
#     площади. Слой намагниченности между идеальной сталью статора и ярмом ротора R_y (аутраннер: ярмо снаружи) — двойной
#     слой потенциала; из непрерывности B_r на нём вес на единицу площади W(ρ) = ρ^{p−1} + R_y^{2p}·ρ^{−p−1} (от радиуса
#     статора не зависит). У 12/14 (p = 7, магнит 23–26 мм) у зазора W в 1,57 раза больше, чем у ярма ⇒ повреждение
#     у ЗАЗОРА прокси занижает, у ярма — завышает;
#   * реальное повреждение теплового сценария (реакция якоря + нагрев): оно сидит у зазора ⇒ прокси обязана ЗАНИЖАТЬ
#     падение K_e, а прокси с весом W(ρ) — объяснять расхождение.
# Критерии записаны до прогона. Замеры на сетке по политике (магнит 0,5 мм, зазор 0,25 мм, 48 положений на оборот) —
# в документации `magnet_fundamental_ratio`.

T_HOT = 150.0
TURNS = 20.0


def _magnet_cells(geo):
    idx = np.where(geo.mask(Region.MAGNET))[0]
    cen = np.array([geo.mesh.cell_centroid(int(c)) for c in idx], dtype=float)
    return idx, cen


def _radial_weight_ratio(geo, r):
    """Прокси с аналитическим весом радиуса W(ρ) (вместо одной площади) — объяснение расхождения, не замена."""
    p = geo.params
    n = p.n_poles // 2
    idx, cen = _magnet_cells(geo)
    rho = np.hypot(cen[:, 0], cen[:, 1])
    W = rho ** (n - 1) + p.R_mag_out ** (2 * n) * rho ** (-n - 1)
    area = np.array([geo.mesh.cell_area(int(c)) for c in idx])
    pol = np.sign(np.einsum("ij,ij->i", geo.magnet_easy_axis[idx], cen / rho[:, None]))
    ph = area * W * pol * np.exp(1j * n * np.arctan2(cen[:, 1], cen[:, 0]))
    return float(abs((r * ph).sum()) / abs(ph.sum()))


class _RadialDamage:
    """Повреждение как функция радиуса: на каждой повёрнутой сетке берётся ТОЧНО (без переноса ближайшим соседом)."""

    def __init__(self, f):
        self.f = f

    def sample(self, geo):
        _, cen = _magnet_cells(geo)
        return self.f(np.hypot(cen[:, 0], cen[:, 1]))


class _Sweep:
    """Машина, сетки положений ротора (строятся один раз) и потокосцепление целой машины на них."""

    def __init__(self, params, angles):
        self.params = params
        self.sc = machine_scenario(params, n42sh_magnet((1.0, 0.0, 0.0)), m270_35a_bh_curve())
        self.angles = angles
        self.geos = build_rotor_geometries(params, angles)
        self.lam0 = self._lam(None)

    def _lam(self, damage):
        sw = sweep_rotor(self.params, self.sc.magnet, self.sc.steel, angles=self.angles, geometries=self.geos,
                         i_peak=0.0, turns_per_slot=TURNS, no_load=True, T=T_HOT, damage=damage)
        assert sw.all_converged
        return sw.flux_linkage_fundamental()

    def honest_ratio(self, damage):
        return self._lam(damage) / self.lam0


# ------------------------------------------------------- повреждение по радиусу: ошибка прокси и её причина

@pytest.fixture(scope="module")
def layered():
    # Повреждение по радиусу одинаково во всех полюсах ⇒ симметрия целой машины сохраняется, λ_A периодична по
    # электрическому периоду, и прогонки по одному периоду достаточно. Магнит 1 мм / зазор 0,5 мм: на сетке 3 мм в
    # магните по радиусу одна-две ячейки, и слой 1 мм там не разрешается.
    p = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.003,
                            mesh_size_by_region={"magnet": 1.0e-3, "air_gap": 0.5e-3})
    return _Sweep(p, electrical_period_angles(p, 8))


@pytest.mark.parametrize("side", ["gap", "yoke"])
def test_radially_nonuniform_damage_biases_the_proxy_by_the_radial_weight(layered, side):
    p = layered.params
    if side == "gap":
        f = lambda rho: np.where(rho < p.R_mag_in + 1.0e-3, 0.5, 1.0)  # noqa: E731
    else:
        f = lambda rho: np.where(rho > p.R_mag_out - 1.0e-3, 0.5, 1.0)  # noqa: E731
    dmg = _RadialDamage(f)
    geo = layered.geos[0]
    r = dmg.sample(geo)
    proxy = 1.0 - magnet_fundamental_ratio(geo, r, geo.mask(Region.MAGNET))
    weighted = 1.0 - _radial_weight_ratio(geo, r)
    honest = 1.0 - layered.honest_ratio(dmg)
    if side == "gap":
        assert honest > 1.05 * proxy, (honest, proxy)          # у зазора прокси ЗАНИЖАЕТ падение K_e
    else:
        assert honest < 0.95 * proxy, (honest, proxy)          # у ярма — завышает
    # Причина — вес радиуса: прокси с W(ρ) ближе к честному ответу не менее чем втрое.
    assert abs(weighted - honest) <= abs(proxy - honest) / 3.0, (weighted, proxy, honest)


# ------------------------------------------------- повреждение по полюсам и реальное повреждение связки

@pytest.fixture(scope="module")
def full_turn():
    # Повреждение, разное у разных полюсов, повторяется только за полный оборот ⇒ прогонка по обороту, 48 положений
    # (шаг 7,5°: проекция на порядок 7 подмешивает порядки 41 и 55, а не 17 и 31, как при 24 положениях — те у
    # локального повреждения заметны: при 24 положениях ответ сдвигается на единицы процентов).
    p = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.003)
    return _Sweep(p, np.arange(48) * 2.0 * np.pi / 48)


def test_pole_wise_damage_proxy_is_exact(full_turn):
    geo = full_turn.geos[0]
    idx, cen = _magnet_cells(geo)
    p = full_turn.params
    pole = np.round((np.arctan2(cen[:, 1], cen[:, 0]) % (2 * np.pi)) / (2 * np.pi / p.n_poles)).astype(int) % p.n_poles
    r = np.random.default_rng(3).uniform(0.5, 1.0, p.n_poles)[pole]
    proxy = magnet_fundamental_ratio(geo, r, geo.mask(Region.MAGNET))
    honest = full_turn.honest_ratio(RotorDamage(geo, r))
    assert abs(proxy - float(np.mean(np.unique(r)))) < 2e-2    # ≈ средняя по полюсам (с весом площади полюса)
    # Допуск — как у однородного повреждения (`test_rotor_sweep`): насыщение стали и перенос между сетками.
    assert abs(honest - proxy) < 5e-3, (honest, proxy)


def test_real_thermal_damage_is_underestimated_by_the_proxy(full_turn):
    sc = full_turn.sc
    res = run_machine_thermal_demag(sc, i_peak=60.0, turns_per_slot=TURNS, gamma_elec=0.0, slot_fill=0.45,
                                    h=400.0, T_amb=T_HOT, T0=T_HOT, dt=2.0, n_steps=3)
    assert res.transient.em_converged and res.retention.min() < 1.0
    geo = sc.geometry
    r = res.retention
    _, cen = _magnet_cells(geo)
    rho = np.hypot(cen[:, 0], cen[:, 1])
    gap_side = rho < 0.5 * (geo.params.R_mag_in + geo.params.R_mag_out)
    assert r[gap_side].mean() < r[~gap_side].mean()            # предпосылка: повреждение — у зазора
    proxy = res.torque_constant_drop
    weighted = 1.0 - _radial_weight_ratio(geo, r)
    honest = 1.0 - full_turn.honest_ratio(RotorDamage(geo, r))
    assert honest > proxy > 0.0, (honest, proxy)              # прокси — НЕ в запас
    assert abs(weighted - honest) < abs(proxy - honest) / 2.0, (weighted, proxy, honest)
