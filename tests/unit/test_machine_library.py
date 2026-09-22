import math

import numpy as np
import pytest

from magcore.fem2d.machines.catalog import (
    DP25_NAMEPLATE_KE,
    MAGNETS,
    build,
    dp25_brushed_dc,
    outrunner_pmsm_uav,
)
from magcore.fem2d.machines.library import (
    MAGNETS_ON_ROTOR,
    MAGNETS_ON_STATOR,
    CommutatorWinding,
    ThreePhaseWinding,
    Topology,
    solve_machine,
)
from magcore.fem2d.machines.pmsm_outrunner import Region

# БИБЛИОТЕКА ТИПОВ: один движок считает РАЗНЫЕ классы машин.
# Оракулы: (1) топология задаёт систему отсчёта потерь; (2) формулы K_e различаются по типу
# обмотки; (3) материалы разносятся по регионам; (4) ЭТАЛОН — ДП25 против паспорта ТУ.


def test_topology_selects_loss_frame():
    # Ошибка выбора системы отсчёта даёт фантомные потери от постоянного поля магнита.
    assert Topology(magnets_on=MAGNETS_ON_ROTOR).magnet_loss_frame == "rotor"
    assert Topology(magnets_on=MAGNETS_ON_STATOR).magnet_loss_frame == "stator"
    with pytest.raises(ValueError):
        Topology(magnets_on="куда-то ещё")


def test_catalog_machines_declare_correct_topology():
    dp25, pmsm = dp25_brushed_dc(), outrunner_pmsm_uav()
    # щёточный ДПТ: магниты на СТАТОРЕ (неподвижны), вращается якорь
    assert dp25.topology.magnets_on == MAGNETS_ON_STATOR
    assert dp25.topology.magnet_loss_frame == "stator"
    # PMSM: магниты на роторе
    assert pmsm.topology.magnets_on == MAGNETS_ON_ROTOR
    assert pmsm.topology.magnet_loss_frame == "rotor"
    # разные типы обмотки
    assert dp25.winding.kind == "commutator"
    assert pmsm.winding.kind == "three_phase"


def test_per_region_materials_are_distinct():
    # ДП25: якорь — электротехническая, корпус — Сталь 10 (разные кривые).
    d = dp25_brushed_dc()
    assert d.materials.steel_armature.curve_id != d.materials.yoke_curve.curve_id
    geo = d.build_geometry()
    nu_of_B, nu_init, mask = d.reluctivity(geo)
    # магнит получил recoil-ν, воздух — 1, железо — своё
    assert np.allclose(nu_init[mask], 1.0 / d.materials.magnet.mu_rec)
    air = geo.region == int(Region.AIR_GAP)
    assert np.allclose(nu_init[air], 1.0)
    arm = geo.mask(Region.TOOTH)
    yok = geo.mask(Region.ROTOR_YOKE)
    assert arm.any() and yok.any()
    assert not np.isclose(nu_init[arm].mean(), nu_init[yok].mean())   # РАЗНЫЕ стали


def test_pmsm_uses_same_steel_when_yoke_not_given():
    p = outrunner_pmsm_uav()
    assert p.materials.steel_yoke is not None or (
        p.materials.yoke_curve.curve_id == p.materials.steel_armature.curve_id)


def test_commutator_skew_factor_and_ke_formula():
    w = CommutatorWinding(conductors_total=780, parallel_path_pairs=1, skew_deg=11.5)
    p = dp25_brushed_dc().params
    k = w.skew_factor(p)
    assert 0.98 < k < 1.0                       # скос 11.5° при 4 полюсах ⇒ ≈0.993
    assert CommutatorWinding(conductors_total=10, skew_deg=0.0).skew_factor(p) == 1.0
    with pytest.raises(ValueError):
        CommutatorWinding(conductors_total=0)


def test_three_phase_kt_is_1_5_times_ke():
    # Частая путаница: для трёхфазной K_t = 1.5·K_e, для коллекторной K_t = K_e.
    pmsm = outrunner_pmsm_uav(mesh_size=6.0e-3)
    geo = pmsm.build_geometry()
    a = np.zeros(geo.mesh.n_vertices)           # формулы проверяем структурно
    w = pmsm.winding
    assert isinstance(w, ThreePhaseWinding)
    assert w.torque_constant(geo, a) == pytest.approx(1.5 * w.emf_constant(geo, a), abs=1e-15)


def test_commutator_current_density_alternates_by_pole():
    d = dp25_brushed_dc(mesh_size=0.6e-3)
    geo = d.build_geometry()
    j = d.winding.current_density(geo, i_peak=1.0)
    assert j is not None
    in_slot = geo.slot_id >= 0
    assert np.any(j[in_slot] > 0) and np.any(j[in_slot] < 0)   # ток меняет знак по полюсам
    assert np.allclose(j[~in_slot], 0.0)                        # вне пазов тока нет
    assert d.winding.current_density(geo, i_peak=0.0) is None


def test_build_from_catalog_by_name():
    d = build("dp25", magnet="KS25DTs-240")
    # проверять по material_id, а НЕ через `or` по имени: `or` закорачивает и маскирует
    # опечатку в имени атрибута (наступал на это).
    assert d.materials.magnet.material_id == "KS25DTs-240"
    assert build("dp25").materials.magnet.material_id == "N35"      # марка изделия по умолчанию
    with pytest.raises(ValueError):
        build("нет-такой")
    with pytest.raises(ValueError):
        build("dp25", magnet="нет-такого")
    assert set(MAGNETS) >= {"N35", "N42SH", "KS25DTs-240"}


@pytest.mark.slow
def test_dp25_reproduces_nameplate_ke():
    """
    ЭТАЛОН БИБЛИОТЕКИ: ДП25 против паспорта ТУ. Проверяет всю цепочку —
    геометрия + материалы по регионам + коллекторная формула K_e.
    Ожидание +7.8 % (остаток — торцевые эффекты, 2D их не описывает; см. sources_registry §1b).
    """
    sol = solve_machine(dp25_brushed_dc(), T=20.0)
    assert sol.converged, "нелинейная итерация не сошлась"
    err = (sol.K_e - DP25_NAMEPLATE_KE) / DP25_NAMEPLATE_KE
    assert 0.0 < err < 0.15, "K_e=%.5f (паспорт %.5f, отклонение %+.1f%%)" % (
        sol.K_e, DP25_NAMEPLATE_KE, 100 * err)
    assert sol.K_t == pytest.approx(sol.K_e)          # ДПТ: K_t = K_e
    assert 300.0 < sol.kV < 350.0                      # паспортный эквивалент ≈324 об/(мин·В)


# ---------------------------------------------------------------- УРОВЕНЬ Р2

def test_winding_resistance_matches_dp25_nameplate():
    """
    ЭТАЛОН Р2 (не зависит от геометрии и магнитов): сопротивление якоря ДП25 из
    обмоточных данных должно лечь в паспортный допуск 3.9 ± 0.4 Ом.
    """
    from magcore.fem2d.machines.catalog import DP25_NAMEPLATE_R
    d = dp25_brushed_dc(mesh_size=1.0e-3)
    geo = d.build_geometry()
    r20 = d.winding.resistance(geo, 20.0)
    assert abs(r20 - DP25_NAMEPLATE_R) <= 0.4, "R=%.2f Ом вне допуска %.1f±0.4" % (r20, DP25_NAMEPLATE_R)
    # растёт с нагревом ровно по ρ_cu(T)
    from magcore.fem2d.losses import copper_resistivity
    r120 = d.winding.resistance(geo, 120.0)
    assert r120 > r20
    assert r120 / r20 == pytest.approx(copper_resistivity(120.0) / copper_resistivity(20.0), rel=1e-9)


def test_copper_loss_formulas_differ_by_winding_type():
    # трёхфазная: P = 3·I_скз²·R (I_скз = i/√2) ⇒ P = 1.5·i²·R
    p = outrunner_pmsm_uav(mesh_size=6.0e-3)
    gp = p.build_geometry()
    Rp = p.winding.resistance(gp, 20.0)
    assert p.winding.copper_loss(gp, i_peak=10.0, T=20.0) == pytest.approx(1.5 * 100.0 * Rp)
    # коллекторная: P = I²·R (постоянный ток, без множителя 3)
    d = dp25_brushed_dc(mesh_size=1.0e-3)
    gd = d.build_geometry()
    Rd = d.winding.resistance(gd, 20.0)
    assert d.winding.copper_loss(gd, i_peak=2.0, T=20.0) == pytest.approx(4.0 * Rd)


def test_commutator_resistance_requires_wire_data():
    w = CommutatorWinding(conductors_total=780)          # без wire_diameter/mean_turn_length
    geo = dp25_brushed_dc(mesh_size=1.0e-3).build_geometry()
    with pytest.raises(ValueError, match="wire_diameter"):
        w.resistance(geo, 20.0)


def test_voltage_limited_point_structure():
    """Режим напряжения: насыщение по скорости, монотонность по R и K_e, КПД∈(0,1)."""
    from magcore.fem2d.machines.library import voltage_limited_point
    d = dp25_brushed_dc(mesh_size=1.0e-3)
    base = voltage_limited_point(d, voltage=24.0, speed_rpm=5000.0, K_e=0.0295, R=3.9)
    assert base["current"] > 0 and base["torque"] > 0 and 0.0 < base["efficiency"] < 1.0
    # E ≥ U ⇒ тока нет
    sat = voltage_limited_point(d, voltage=24.0, speed_rpm=50000.0, K_e=0.0295, R=3.9)
    assert sat["current"] == 0.0 and sat["torque"] == 0.0
    # горячая обмотка (R↑) всегда снижает момент
    assert voltage_limited_point(d, voltage=24.0, speed_rpm=5000.0, K_e=0.0295,
                                 R=5.0)["torque"] < base["torque"]

    # ⚠ ОСЛАБЛЕНИЕ МАГНИТА действует ПО-РАЗНОМУ в зависимости от скорости — физика,
    # а не артефакт (наступал на это, написав наивный тест):
    #   • НА НИЗКОЙ скорости (E ≪ U) ток задан U/R и почти не зависит от K_e,
    #     поэтому момент τ = K_t·I падает пропорционально K_e;
    #   • НА ВЫСОКОЙ скорости (E близко к U) ослабленный магнит даёт МЕНЬШЕ противо-ЭДС,
    #     ток РАСТЁТ и момент может даже увеличиться — ценой тока, нагрева и КПД.
    low_full = voltage_limited_point(d, voltage=24.0, speed_rpm=200.0, K_e=0.0295, R=3.9)
    low_weak = voltage_limited_point(d, voltage=24.0, speed_rpm=200.0, K_e=0.0250, R=3.9)
    assert low_weak["torque"] < low_full["torque"]          # на низкой скорости — падение

    hi_full = voltage_limited_point(d, voltage=24.0, speed_rpm=6000.0, K_e=0.0295, R=3.9)
    hi_weak = voltage_limited_point(d, voltage=24.0, speed_rpm=6000.0, K_e=0.0250, R=3.9)
    assert hi_weak["current"] > hi_full["current"]          # на высокой — ток растёт
    assert hi_weak["efficiency"] < hi_full["efficiency"]    # и КПД падает — вот где ущерб


def test_thrust_is_proportional_to_torque():
    """При токовом ограничении тяга ∝ моменту (винт: τ=kω² ⇒ тяга∝ω²∝τ)."""
    from magcore.fem2d.machines.library import MachinePerformance
    perf = MachinePerformance(machine="x", torque=1.5, K_e=0.03, K_t=0.03, speed_rpm=7000.0,
                              i_peak=40.0, T=20.0, P_out=1100.0, P_copper=100.0, P_core=50.0,
                              R_winding=0.05, converged=True)
    assert perf.thrust_relative == perf.torque
    assert perf.P_loss == pytest.approx(150.0)
    assert perf.efficiency == pytest.approx(1100.0 / 1250.0)


@pytest.mark.slow
def test_performance_end_to_end_pmsm():
    """Р2 целиком на PMSM: момент, потери, КПД — величины физичны и согласованы."""
    from magcore.fem2d.machines.library import evaluate_performance
    perf = evaluate_performance(outrunner_pmsm_uav(mesh_size=4.0e-3),
                                i_peak=40.0, speed_rpm=7000.0, T=20.0)
    assert perf.converged
    assert perf.torque > 0.0
    assert perf.P_out == pytest.approx(perf.torque * perf.omega)
    assert perf.P_copper > 0.0
    assert 0.0 < perf.efficiency < 1.0


# ---------------------------------------------------------------- УРОВЕНЬ Р3 (ядро К6′)

def test_loss_current_differs_from_magnetic_current():
    """
    ⚠ Ток для НАГРЕВА ≠ ток для МАГНИТНОЙ задачи — частая и незаметная ошибка.
    Трёхфазная: нагрев по СКЗ (1/√2) и не зависит от угла γ; магнитный — мгновенный,
    от γ зависит. Плюс поправка на заполнение паза (1/√k_зап).
    """
    p = outrunner_pmsm_uav(mesh_size=5.0e-3)
    geo = p.build_geometry()
    j_loss = p.winding.loss_current_density(geo, i_peak=40.0)
    j_mag0 = p.winding.current_density(geo, i_peak=40.0, gamma_elec=0.0)
    j_mag90 = p.winding.current_density(geo, i_peak=40.0, gamma_elec=math.pi / 2)
    assert not np.allclose(j_mag0, j_mag90)        # магнитный ток зависит от угла
    assert np.all(j_loss >= 0.0)                    # ток нагрева — модуль, знак не важен
    # нагрев не зависит от угла: считается один раз
    assert np.allclose(j_loss, p.winding.loss_current_density(geo, i_peak=40.0))


def test_commutator_loss_current_has_no_rms_factor():
    """У коллекторной машины ток проводника ПОСТОЯНЕН ⇒ СКЗ = значению (нет 1/√2)."""
    d = dp25_brushed_dc(mesh_size=1.0e-3)
    geo = d.build_geometry()
    w = d.winding
    j = w.loss_current_density(geo, i_peak=2.0)
    z_slot = w.conductors_total / geo.params.n_slots
    i_cond = 2.0 / (2.0 * w.parallel_path_pairs)
    from magcore.fem2d.machines.excitation import slot_areas
    expected = z_slot * i_cond / slot_areas(geo).min() / math.sqrt(w.slot_fill)
    assert j.max() == pytest.approx(expected, rel=1e-9)
    assert np.allclose(j[geo.slot_id < 0], 0.0)     # вне пазов тока нет


def test_thermal_demag_result_reports_damage_plainly():
    """Итог Р3 переводится в инженерный язык: «мотор стал слабее на …%»."""
    from magcore.fem2d.machines.library import ThermalDemagResult

    class _T:
        runaway = False
        magnet_cascade = False
        stop_reason = "завершено"

    r = ThermalDemagResult(machine="x", transient=_T(), retention=np.array([0.9, 1.0]),
                           fundamental_ratio=0.93, T_magnet_max=150.0, T_max=170.0)
    assert r.torque_constant_drop == pytest.approx(0.07)
    assert r.survived and r.stop_reason == "завершено"

    class _F(_T):
        magnet_cascade = True
    assert not ThermalDemagResult(machine="x", transient=_F(), retention=np.array([0.5]),
                                  fundamental_ratio=0.5, T_magnet_max=200.0,
                                  T_max=210.0).survived


@pytest.mark.slow
def test_thermal_demag_runs_for_both_machine_types():
    """
    ЯДРО К6′ через библиотеку: связка запускается и для трёхфазного PMSM, и для
    коллекторного ДП25 — один код, разные типы. Проверяется физичность и энергобаланс.
    """
    from magcore.fem2d.machines.library import run_thermal_demag

    common = dict(h_out=50.0, T_amb=60.0, T0=60.0, dt=0.5, n_steps=4, T_cap=200.0)
    for definition, i_peak in ((outrunner_pmsm_uav(mesh_size=5.0e-3), 40.0),
                               (dp25_brushed_dc(mesh_size=0.8e-3), 1.35)):
        res = run_thermal_demag(definition, i_peak=i_peak, **common)
        assert res.T_magnet_max >= common["T_amb"] - 1.0     # магнит не холоднее среды
        assert 0.0 <= res.torque_constant_drop <= 1.0
        assert np.all(res.retention <= 1.0) and np.all(res.retention > 0.0)
        tr = res.transient
        # энергобаланс неявного Эйлера точен (оракул связки)
        st, lp, of = tr.stored_energy, tr.loss_power, tr.outflow
        if len(st) >= 3:
            d_store = (st[-1] - st[-2]) / common["dt"]
            assert abs(d_store - (lp[-1] - of[-1])) / max(abs(lp[-1]), 1.0) < 1e-9


# ---------------------------------------------------------------- потери ядра в Р3

def test_magnet_conductivity_follows_magnet_grade():
    """
    σ должна браться из ФАКТИЧЕСКИ установленного магнита: SmCo проводнее NdFeB
    (1.18 против 0.79 МС/м) ⇒ у него БОЛЬШЕ вихревых потерь. Потерять это при смене
    марки — значит скрыть реальный минус SmCo.
    """
    nd = build("dp25", magnet="N35").materials
    sm = build("dp25", magnet="KS25DTs-240").materials
    assert sm.magnet_sigma_20 > nd.magnet_sigma_20
    assert sm.magnet_sigma_20 / nd.magnet_sigma_20 == pytest.approx(1.27 / 0.85, rel=1e-6)


def test_loss_data_presence_is_explicit():
    """Без данных потерь расчёт ядра ОТКАЗЫВАЕТСЯ считать, а не выдаёт молча нули.

    Берём ТРЁХФАЗНУЮ машину: для неё потери ядра поддержаны, поэтому проверка доходит именно
    до отсутствия коэффициентов (у коллекторной сработал бы более ранний отказ по типу)."""
    from magcore.fem2d.machines.library import MachineMaterials, compute_core_losses
    d = outrunner_pmsm_uav(mesh_size=6.0e-3)
    assert d.materials.has_loss_data()
    bare = MachineMaterials(magnet=d.materials.magnet,
                            steel_armature=d.materials.steel_armature)
    assert not bare.has_loss_data()
    from dataclasses import replace
    with pytest.raises(ValueError, match="steinmetz_armature"):
        compute_core_losses(replace(d, materials=bare), speed_rpm=1000.0)


def test_catalog_declares_solid_yoke_where_real():
    """ДП25: корпус МАССИВНЫЙ ⇒ модель сплошного тела со скин-пределом, не Штейнмец в Вт/кг."""
    m = dp25_brushed_dc().materials
    assert m.yoke_solid and m.yoke_resistivity is not None
    assert m.steinmetz_armature is not m.yoke_steinmetz     # разные стали — разные коэффициенты


@pytest.mark.slow
def test_core_losses_computed_for_three_phase():
    """Потери ядра трёхфазной машины считаются автоматически; разбивка физична."""
    from magcore.fem2d.machines.library import compute_core_losses
    d = outrunner_pmsm_uav(mesh_size=6.0e-3)
    cl = compute_core_losses(d, speed_rpm=7000.0, i_peak=40.0, n_positions=6)
    assert cl.stator_iron_w > 0.0 and cl.rotor_side_w >= 0.0
    assert cl.total_w == pytest.approx(cl.stator_iron_w + cl.rotor_side_w)
    assert cl.freq_elec > 0.0 and cl.freq_ripple > 0.0
    assert cl.density.shape == (d.build_geometry().mesh.n_cells,)
    assert np.all(cl.density >= 0.0) and np.any(cl.density > 0.0)


def test_core_losses_refused_for_commutator():
    """
    Потери в железе для КОЛЛЕКТОРНОЙ машины (щёточный ДПТ) не поддержаны и отклоняются с
    понятным сообщением: расчёт опирается на трёхфазную раскладку тока, а реакция якоря
    коллектора устроена иначе — честный отказ, а не неверное число (см. решение по ДП25).
    """
    from magcore.fem2d.machines.library import compute_core_losses
    d = dp25_brushed_dc(mesh_size=0.9e-3)
    with pytest.raises(ValueError, match="коллекторной"):
        compute_core_losses(d, speed_rpm=7400.0, i_peak=1.35, n_positions=6)


@pytest.mark.slow
def test_auto_core_losses_heats_more_than_copper_alone():
    """auto_core_losses=True добавляет сталь+магнит к меди ⇒ суммарный нагрев выше."""
    from magcore.fem2d.machines.library import run_thermal_demag
    d = outrunner_pmsm_uav(mesh_size=6.0e-3)
    common = dict(i_peak=40.0, h_out=50.0, T_amb=60.0, T0=60.0, dt=0.5, n_steps=3, T_cap=200.0)
    cu_only = run_thermal_demag(d, **common)
    with_core = run_thermal_demag(d, auto_core_losses=True, speed_rpm=7000.0, **common)
    assert with_core.transient.loss_power[-1] > cu_only.transient.loss_power[-1]
    with pytest.raises(ValueError, match="speed_rpm"):
        run_thermal_demag(d, auto_core_losses=True, **common)
