import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.losses import copper_resistivity
from magcore.fem2d.machines.excitation import slot_areas, winding_current_density
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, Region
from magcore.fem2d.machines.scenario import machine_scenario
from magcore.fem2d.machines.thermal_scenario import (
    MachineThermalProperties,
    _slot_rms_loss_current,
    magnet_fundamental_ratio,
    run_machine_thermal_demag,
)

# S3, инкремент 4: перенос верифицированной связки с сегмента на СЕЧЕНИЕ МАШИНЫ.
# Физика связки уже под оракулами (test_coupled_transient), поэтому здесь проверяется то,
# что добавляет машинный слой и где легко ошибиться незаметно:
#   * ток для ПОТЕРЬ ≠ ток для магнитостатики (СКЗ за период + заполнение паза);
#     оракул — интеграл потерь = I²R с сопротивлением, выведенным из геометрии паза;
#   * первая гармоника ремнантности как инвариантная к положению ротора мера ущерба;
#   * РЕАКЦИЯ ЯКОРЯ подмагничивает часть полюса (H вдоль лёгкой оси > 0) — режим, которого
#     на сегменте не было и который вскрыл реальный баг (см. регрессионный тест ниже).

MESH = 0.0030
HOT = dict(turns_per_slot=20.0, gamma_elec=0.0, slot_fill=0.45,
           h=400.0, T_amb=150.0, T0=150.0, dt=2.0, n_steps=3)


def _scenario(magnet=None):
    return machine_scenario(
        OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=MESH),
        magnet or n42sh_magnet((1.0, 0.0, 0.0)),
        m270_35a_bh_curve(),
    )


# ------------------------------------------------- ток потерь: оракул I²R на уровне машины

def test_slot_loss_current_integrates_to_i2r():
    # Потери в меди, посчитанные ПОЛЕВО (∫ρ(T)J² dV), должны совпасть с I²R, где сопротивление
    # паза выведено из геометрии независимо: N последовательных проводников общей площадью
    # меди A_cu = k_зап·A_паз, каждый длиной L ⇒ R_паз = ρ·L·N²/A_cu.
    # Именно этот тест ловит забытый коэффициент заполнения (ошибка в 1/k_зап ≈ 2 раза).
    sc = _scenario()
    g = sc.geometry
    N, i_peak, k_fill, L = 20.0, 90.0, 0.45, g.params.axial_length
    T = 75.0

    j = _slot_rms_loss_current(g, i_peak=i_peak, turns_per_slot=N, slot_fill=k_fill)
    areas = np.array([g.mesh.cell_area(c) for c in range(g.mesh.n_cells)], dtype=float)
    P_field = float((copper_resistivity(T) * j * j * areas).sum()) * L

    i_rms = i_peak / np.sqrt(2.0)
    A_cu = slot_areas(g) * k_fill
    P_i2r = float(np.sum(i_rms**2 * copper_resistivity(T) * L * N**2 / A_cu))
    assert abs(P_field - P_i2r) / P_i2r < 1e-10

    # Контроль, что заполнение действительно учтено: вдвое меньше меди — вдвое больше потерь.
    j_half = _slot_rms_loss_current(g, i_peak=i_peak, turns_per_slot=N, slot_fill=k_fill / 2)
    assert abs(float((j_half**2).sum()) / float((j**2).sum()) - 2.0) < 1e-10


def test_loss_current_is_angle_independent_unlike_magnetic_current():
    # Тепло греет СКЗ за период (постоянная времени машины ≫ электрического периода), поэтому
    # ток потерь от угла вектора тока НЕ зависит; магнитная задача решается при мгновенном
    # токе и от γ зависит существенно. Если перепутать, нагрев начнёт «дышать» с γ.
    sc = _scenario()
    g = sc.geometry
    kw = dict(turns_per_slot=20.0, i_peak=90.0)
    assert np.array_equal(
        _slot_rms_loss_current(g, slot_fill=0.45, **kw),
        _slot_rms_loss_current(g, slot_fill=0.45, **kw),
    )
    j_mag_0 = winding_current_density(g, sc.layout, gamma_elec=0.0, **kw)
    j_mag_90 = winding_current_density(g, sc.layout, gamma_elec=np.pi / 2, **kw)
    assert not np.allclose(j_mag_0, j_mag_90)


def test_slot_fill_is_validated():
    sc = _scenario()
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="slot_fill"):
            _slot_rms_loss_current(sc.geometry, i_peak=1.0, turns_per_slot=1.0, slot_fill=bad)


# ------------------------------------------------------------- тепловые свойства по регионам

def test_thermal_properties_cover_every_cell():
    sc = _scenario()
    k, c = MachineThermalProperties.representative().cell_fields(sc.geometry)
    assert k.shape == c.shape == (sc.geometry.mesh.n_cells,)
    assert np.all(k > 0.0) and np.all(c > 0.0)
    # Сталь проводит тепло много лучше паза и зазора — иначе регионы перепутаны местами.
    assert k[sc.geometry.mask(Region.TOOTH)].min() > k[sc.geometry.mask(Region.SLOT)].max()


def test_missing_region_property_is_rejected():
    sc = _scenario()
    props = MachineThermalProperties(k_by_region={"air_gap": 1.0}, c_by_region={"air_gap": 1.0e6})
    with pytest.raises(ValueError, match="тепловых свойств"):
        props.cell_fields(sc.geometry)


# ------------------------------------------ первая гармоника ремнантности: точные оракулы

def test_fundamental_ratio_exact_on_known_cases():
    # Оракулы, где ответ известен ТОЧНО: целый магнит → ровно 1; равномерная потеря доли s →
    # ровно s (гармоника линейна по ремнантности, множитель выносится). Это фиксирует и
    # нормировку, и учёт полярности полюсов — при перепутанном знаке сумма не сложилась бы.
    sc = _scenario()
    mask = sc.geometry.mask(Region.MAGNET)
    n = int(mask.sum())
    assert abs(magnet_fundamental_ratio(sc.geometry, np.ones(n), mask) - 1.0) < 1e-12
    for s in (0.9, 0.5):
        got = magnet_fundamental_ratio(sc.geometry, np.full(n, s), mask)
        assert abs(got - s) < 1e-12
    with pytest.raises(ValueError, match="ячейкам магнита"):
        magnet_fundamental_ratio(sc.geometry, np.ones(n + 1), mask)


# ---------------------------------------------------- РЕГРЕССИЯ: подмагничивание ≠ ущерб

def test_armature_reaction_magnetising_a_pole_causes_no_damage():
    # РЕГРЕССИЯ на реальный баг. Реакция якоря часть полюса РАЗМАГНИЧИВАЕТ, а часть —
    # ПОДМАГНИЧИВАЕТ (H вдоль лёгкой оси > 0). Таблица кривой кончается на H=0, и когда
    # положительное H зажимали в ноль, тождество B_maj = B_r + μ₀μ_rec·H ломалось, и латч
    # записывал ФАНТОМНУЮ потерю именно в усиливаемых ячейках (наблюдалось r=0.61 при НУЛЕ
    # ячеек за коленом). На сегменте режима с H>0 не было, поэтому баг вскрылся только здесь.
    sc = _scenario()
    res = run_machine_thermal_demag(
        sc, i_peak=90.0, turns_per_slot=20.0, gamma_elec=0.0, slot_fill=0.45,
        h=400.0, T_amb=110.0, T0=110.0, dt=2.0, n_steps=3,
    )
    state = res.transient.state
    assert state.H_par.max() > 0.0                  # подмагничиваемые ячейки ЕСТЬ (иначе тест пуст)
    assert np.all(res.transient.n_past_knee == 0)   # и при этом колено никто не переходил
    assert np.all(res.retention == 1.0)             # ⇒ потерь быть не должно ВООБЩЕ
    assert res.torque_constant_drop == 0.0


# ------------------------------------------------------------------ инженерный итог

def test_smco_survives_the_load_that_damages_ndfeb():
    # Содержательный итог К6′ на РЕАЛЬНОЙ машине: при одинаковой тепловой и токовой нагрузке
    # (та же геометрия, тот же ток, та же температура) NdFeB необратимо теряет часть полюса,
    # а SmCo колено не переходит вовсе. Это и есть довод в пользу SmCo для теплонагруженного
    # привода, показанный связанным полевым расчётом, а не таблицей коэффициентов.
    nd = run_machine_thermal_demag(_scenario(n42sh_magnet((1.0, 0.0, 0.0))), i_peak=60.0, **HOT)
    sm = run_machine_thermal_demag(_scenario(sm2co17_magnet((1.0, 0.0, 0.0))), i_peak=60.0, **HOT)

    assert abs(nd.T_magnet_max - sm.T_magnet_max) < 1.0     # тепловая нагрузка одинакова
    assert nd.transient.n_past_knee[-1] > 0 and nd.retention.min() < 1.0
    assert nd.torque_constant_drop > 0.0

    assert np.all(sm.transient.n_past_knee == 0)
    assert np.all(sm.retention == 1.0)
    assert sm.torque_constant_drop == 0.0
    assert nd.survived and sm.survived


def test_damage_grows_with_current():
    # Физическая монотонность: сильнее ток — глубже необратимая потеря и больше падение
    # моментной постоянной. Ловит перепутанные знаки/масштабы в цепочке обмотка→поле→демаг.
    sc = _scenario()
    mild = run_machine_thermal_demag(sc, i_peak=30.0, **HOT)
    hard = run_machine_thermal_demag(sc, i_peak=60.0, **HOT)
    assert hard.retention.min() < mild.retention.min()
    assert hard.torque_constant_drop > mild.torque_constant_drop > 0.0
    # Потери растут по ходу нагрева: ρ(T) — та самая положительная обратная связь.
    assert hard.loss_power_final > hard.loss_power_initial
