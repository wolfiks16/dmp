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
    # Горизонт — 3 шага, все решены до допуска: третий шаг NdFeB прежний Пикар не решал (32 подшага,
    # невязка 9e-5 при допуске 1e-6), и горизонт был урезан до двух; Ньютон решает его за единицы итераций.
    nd = run_machine_thermal_demag(_scenario(n42sh_magnet((1.0, 0.0, 0.0))), i_peak=60.0, **HOT)
    sm = run_machine_thermal_demag(_scenario(sm2co17_magnet((1.0, 0.0, 0.0))), i_peak=60.0, **HOT)
    assert nd.transient.em_converged and sm.transient.em_converged
    assert len(nd.transient.times) == len(sm.transient.times) == HOT["n_steps"] + 1

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
    assert mild.transient.em_converged and hard.transient.em_converged
    assert hard.retention.min() < mild.retention.min()
    assert hard.torque_constant_drop > mild.torque_constant_drop > 0.0
    # Потери растут по ходу нагрева: ρ(T) — та самая положительная обратная связь.
    assert hard.loss_power_final > hard.loss_power_initial


# ------------------------------------------- метод магнитной задачи: Ньютон против эталона-Пикара

def test_newton_steel_tangent_keeps_the_machine_reluctivity_bit_for_bit():
    # Ньютону нужна касательная стали, но ν должна остаться той же, что у Пикара (`machine_reluctivity`): иначе
    # методы решали бы разные задачи. Сверка на поле, проходящем всю кривую стали — от нуля до насыщения.
    from magcore.fem2d.machines.bridge import pmsm_to_problem
    from magcore.fem2d.machines.static_solver import machine_reluctivity
    from magcore.fem2d.model.problem import _reluctivity_newton

    sc = _scenario()
    g = sc.geometry
    nu_of_B, nu_init, _, steel_mask = machine_reluctivity(g, sc.magnet, sc.steel)
    nu_and_dnu, nu0 = _reluctivity_newton(pmsm_to_problem(g, sc.magnet, sc.steel))
    B = np.random.default_rng(5).uniform(-2.6, 2.6, (g.mesh.n_cells, 2))
    nu, dnu = nu_and_dnu(B)
    assert np.array_equal(nu, nu_of_B(B)) and np.array_equal(nu0, nu_init)
    assert np.all(dnu[~steel_mask] == 0.0) and np.any(dnu[steel_mask] != 0.0)


def test_newton_solves_the_step_picard_could_not_on_the_machine():
    # Третий шаг NdFeB в горячем режиме эталон-Пикар не решает и с 32 подшагами — расчёт останавливается без
    # вердикта. Ньютон решает все три шага до допуска, без дробления. Тепло от метода не зависит: потери меди
    # от поля не зависят, поэтому история T совпадает бит в бит, пока Пикар шёл.
    sc = _scenario()
    pic = run_machine_thermal_demag(sc, i_peak=60.0, method="picard", **HOT).transient
    new = run_machine_thermal_demag(sc, i_peak=60.0, **HOT).transient
    assert pic.em_failed and len(pic.times) == HOT["n_steps"]          # предпосылка: шаг 3 Пикару не по силам
    assert not new.em_failed and new.em_converged and new.em_method == "newton"
    assert len(new.times) == HOT["n_steps"] + 1
    assert np.all(new.em_substeps[1:] == 1) and new.em_residual.max() <= 1.0e-9
    assert np.array_equal(new.T_hist[:len(pic.times)], pic.T_hist)


def test_unknown_magnetic_method_is_rejected():
    with pytest.raises(ValueError, match="newton"):
        run_machine_thermal_demag(_scenario(), i_peak=10.0, method="что-то", **HOT)


def test_newton_and_picard_solve_the_same_step_on_the_machine():
    # ОРАКУЛ «предел Пикара — решение Ньютона» на машине 12/14 (сталь M270 нелинейна, ось магнита поячеечная):
    # то же, что на сегменте (test_coupled_transient::test_newton_and_picard_solve_the_same_magnetic_step) —
    # из одного состояния ужесточение допуска Пикара в 10⁴ раз уменьшает расхождение с Ньютоном пропорционально
    # (граница 1e-2 — среднее геометрическое исходов «та же точка» 1e-4 и «другая» 1). Состояния: новый магнит
    # при однородных 150 °C, затем нагрев с радиальным перепадом 10 °C (новая потеря) и остывание.
    # Эталон — как в машинном сценарии (ν-релаксация 0,1), ω ветви 0,15 (при 0,3 он здесь дожимается хуже).
    from magcore.constants import MU0
    from magcore.fem2d.coupled_transient import IrreversibleMagnetState, MagnetLawState
    from magcore.fem2d.machines.bridge import pmsm_to_problem
    from magcore.fem2d.machines.static_solver import machine_reluctivity
    from magcore.fem2d.model.problem import _reluctivity_newton
    from magcore.fem2d.newton import solve_nonlinear_2d_newton
    from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
    from magcore.fem2d.spaces import LagrangeP1Space2D

    sc = _scenario()
    g = sc.geometry
    space = LagrangeP1Space2D(g.mesh)
    nc = g.mesh.n_cells
    nu_of_B, nu_init, mask, _ = machine_reluctivity(g, sc.magnet, sc.steel)
    nu_and_dnu, _ = _reluctivity_newton(pmsm_to_problem(g, sc.magnet, sc.steel))
    j = MU0 * winding_current_density(g, sc.layout, i_peak=60.0, gamma_elec=0.0, turns_per_slot=20.0)
    ddofs = space.boundary_dofs()
    idx = np.where(mask)[0]
    rad = np.array([np.hypot(*g.mesh.cell_vertices(int(c)).mean(axis=0)) for c in idx])
    gradient = 10.0 * (rad - rad.min()) / (rad.max() - rad.min())
    newton = MagnetLawState(sc.magnet, mask, nc, axis=g.magnet_easy_axis)
    a_prev = None
    new_loss = []
    for T_mag in (np.full(idx.size, 150.0), 160.0 + gradient, 170.0 + gradient, 120.0 + gradient):
        newton.set_temperature(T_mag)
        law = newton.law()
        em_n = solve_nonlinear_2d_newton(space, nu_and_dnu, nu_init=nu_init, j_cells=j, magnet_law=law,
                                         dirichlet_dofs=ddofs, max_iter=60, tol=1e-13, a_init=a_prev)
        newton.accept(law)
        assert em_n.converged
        errs = []
        for tol in (1e-8, 1e-12):
            pic = IrreversibleMagnetState(sc.magnet, mask, nc, axis=g.magnet_easy_axis, relaxation=0.15)
            pic.retention = newton.retention.copy()
            pic.set_temperature(T_mag)
            em_p = solve_nonlinear_2d_picard(
                space, nu_of_B=lambda B, _s=pic: _s.nu_field(nu_of_B(B)), nu_init=pic.nu_field(nu_init.copy()),
                j_cells=j, magnetization=pic, dirichlet_dofs=ddofs, relaxation=0.1, max_iter=20000, tol=tol,
            )
            assert em_p.converged
            errs.append(np.linalg.norm(em_n.B_cells - em_p.B_cells) / np.linalg.norm(em_p.B_cells))
        assert errs[1] < 1e-2 * errs[0], errs
        before = newton.retention.copy()
        newton.commit()
        new_loss.append(bool(np.any(newton.retention < before)))
        a_prev = em_n.a
    assert all(new_loss[:3]) and not new_loss[3]           # потеря при нагреве, при остывании — нет
