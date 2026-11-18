import dataclasses

import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.fem2d.coupled_transient import (
    IrreversibleMagnetState,
    MagnetLawState,
    solve_coupled_magneto_thermal_transient,
)
from magcore.fem2d.losses import CU_ALPHA, CU_RHO0
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.newton import solve_nonlinear_2d_newton
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.runaway import runaway_threshold, solve_thermal_steady_with_feedback
from magcore.fem2d.spaces import LagrangeP1Space2D

# S3, инкремент 3 — ядро К6′: петля магнит↔тепло↔необратимый демаг во времени.
# Магнитная задача шага — методом Ньютона с законом магнита в касательной (по умолчанию); прежний
# хордовый Пикар — эталон `method='picard'`.
# ОРАКУЛЫ (независимые от самой связки):
#   (1) ЭНЕРГОБАЛАНС: dU/dt = ∫q dV − ∫h(T−T_amb)ds точно (неявный Эйлер, до машинной);
#   (2) ПОРОГ РАЗГОНА: обратная связь меди q(T) линейна ⇒ петля устойчива ТОГДА И ТОЛЬКО
#       ТОГДА, когда s=ρ₀αJ² < s_crit — независимого обобщённого собств. значения
#       (K+R)v=s·M·v; ниже порога установившееся поле совпадает с ПРЯМЫМ решением
#       связанной стационарной задачи, выше — расходится;
#   (3) НЕОБРАТИМОСТЬ: при остывании возвращается только обратимая часть (латч не растёт);
#   (4) НЕЗАВИСИМОСТЬ ОТ ДЕМПФИРОВАНИЯ (эталон-Пикар): сошедшийся ответ не зависит от ω (иначе это
#       не решение, а артефакт итерации — так и была поймана неустойчивость за коленом);
#   (5) СХОДИМОСТЬ ПО ШАГУ: измельчение dt даёт монотонно сходящуюся траекторию;
#   (6) НЕГАТИВНЫЕ: без нагрева/за пределом колена потерь нет (ровно r=1);
#   (7) НЕСХОДИМОСТЬ ≠ ВЕРДИКТ: где Пикар не сошёлся, равновесие есть — его находит статический
#       Ньютон (`solve_problem2d`) на том же состоянии; расчёт останавливается без вывода о магните;
#   (8) НЬЮТОН = ПИКАР: из одного состояния оба решают один закон, поле совпадает в пределах ошибки
#       остановки Пикара tol·ρ/(1−ρ); тепло от метода не зависит вовсе (потери от поля не зависят);
#   (9) КАСКАД — только у ПРИНЯТОГО решения: ячейки ниже −H_cJ те же, что у статического Ньютона.


def _segment(nx=16, ny=12):
    """Простой сегмент: магнит + рядом обмотка, вокруг воздух, A_z=0 на границе."""
    mesh = build_structured_rectangle_tri_mesh(nx, ny, x0=0.0, x1=0.06, y0=0.0, y1=0.04)
    space = LagrangeP1Space2D(mesh)
    cent = np.array([mesh.cell_vertices(c).mean(axis=0) for c in range(mesh.n_cells)])
    magnet_mask = ((cent[:, 0] >= 0.020) & (cent[:, 0] <= 0.030)
                   & (cent[:, 1] >= 0.012) & (cent[:, 1] <= 0.028))
    copper_mask = ((cent[:, 0] >= 0.030) & (cent[:, 0] <= 0.040)
                   & (cent[:, 1] >= 0.012) & (cent[:, 1] <= 0.028))
    return space, magnet_mask, copper_mask


def _materials(magnet, magnet_mask, copper_mask):
    """ν (относительная), теплопроводность и объёмная теплоёмкость по регионам."""
    nu = np.ones(magnet_mask.size, dtype=float)
    nu[magnet_mask] = 1.0 / magnet.mu_rec
    k = np.where(copper_mask, 400.0, np.where(magnet_mask, 9.0, 1.0))
    cap = np.where(copper_mask, 3.45e6, np.where(magnet_mask, 3.0e6, 1.2e6))
    return nu, k, cap


# ---------------------------------------------------------------- (1) энергобаланс

def test_energy_balance_is_exact():
    # Точный дискретный баланс неявного Эйлера: 1ᵀ·K = 0 (константа в ядре жёсткости) ⇒
    # (U^{n+1} − U^n)/dt = ∫q dV − ∫h(T^{n+1}−T_amb) ds. Держится ДО МАШИННОЙ точности —
    # не «примерно сходится», а тождество, поэтому ловит любую ошибку сборки источника,
    # ёмкости или граничного члена.
    space, magnet_mask, copper_mask = _segment()
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    nu, k, cap = _materials(magnet, magnet_mask, copper_mask)
    dt = 10.0
    res = solve_coupled_magneto_thermal_transient(
        space, k_cells=k, capacity_cells=cap, h=25.0, T_amb=20.0, dt=dt, n_steps=12,
        j_cells=np.where(copper_mask, 1.2e7, 0.0),
        magnet=magnet, magnet_mask=magnet_mask, nu_init=nu,
    )
    residual = np.diff(res.stored_energy) / dt - (res.loss_power[1:] - res.outflow[1:])
    assert np.max(np.abs(residual)) < 1e-9 * res.loss_power.max()


# ------------------------------------------------------------- (2) порог разгона

@pytest.mark.parametrize("factor,expect_runaway", [(0.7, False), (1.4, True)])
def test_runaway_threshold_matches_eigenvalue(factor, expect_runaway):
    # Потери меди дают ТОЧНО линейную обратную связь q(T) = q0 + s(T−T_amb), s = ρ₀αJ².
    # Тогда неявный Эйлер с явным источником имеет коэффициент усиления моды
    # (c/dt + s)/(c/dt + λ) для (K+R)v = λ·M·v ⇒ петля сходится ⟺ s < s_crit = λ_min.
    # Порог считается НЕЗАВИСИМО (обобщённая задача на собственные значения), а ток
    # подбирается под заданную долю порога — то есть проверяется предсказание, а не подгонка.
    mesh = build_structured_rectangle_tri_mesh(10, 10, x0=0.0, x1=0.05, y0=0.0, y1=0.05)
    space = LagrangeP1Space2D(mesh)
    k, h, T_amb = 20.0, 15.0, 20.0
    s_crit = runaway_threshold(space, k, h)

    s = factor * s_crit
    J = np.sqrt(s / (CU_RHO0 * CU_ALPHA))                  # проводник на всю область
    j_cells = np.full(mesh.n_cells, J)
    T_cap = T_amb + 1.0e4

    # Обратная связь растягивает постоянную времени в 1/(1−s/s_crit) раз, поэтому горизонт
    # берётся с запасом; схема безусловно устойчива, а установившееся решение от dt не зависит.
    # (Уровень перегрева в этой ЧИСТО ВЕРИФИКАЦИОННОЙ фикстуре физического смысла не несёт —
    # ток подобран под заданную долю порога, а не под реальную обмотку.)
    res = solve_coupled_magneto_thermal_transient(
        space, k_cells=k, capacity_cells=2.0e6, h=h, T_amb=T_amb,
        dt=500.0, n_steps=300, j_cells=j_cells, T_cap=T_cap,
    )
    assert res.runaway is expect_runaway

    if not expect_runaway:
        # Ниже порога: установившееся поле = ПРЯМОЕ решение связанной задачи (K+R−sM)T=f.
        q0 = CU_RHO0 * (1.0 + CU_ALPHA * (T_amb - 20.0)) * J * J
        T_direct = solve_thermal_steady_with_feedback(
            space, k, q0_cells=np.full(mesh.n_cells, q0), loss_sensitivity=s,
            h=h, T_amb=T_amb,
        )
        rise = T_direct.max() - T_amb
        assert np.max(np.abs(res.T_hist[-1] - T_direct)) < 1e-3 * rise


# ------------------------------------------------------- (3) необратимость/остывание

def test_irreversible_loss_survives_cooling():
    # Физика необратимости: нагрев за колено съедает часть ремнантности НАВСЕГДА. При
    # остывании обратимая часть возвращается по B_r(T), а доля r остаётся сниженной ⇒
    # ремнантность холодного магнита ниже исходной. Проверяем на самом состоянии, чтобы
    # оракул не зависел от тепловой части.
    space, magnet_mask, _ = _segment()
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    nc = magnet_mask.size
    nu0 = np.ones(nc)
    nu0[magnet_mask] = 1.0 / magnet.mu_rec
    state = IrreversibleMagnetState(magnet, magnet_mask, nc)

    def nu_fn(_B):
        v = nu0.copy()
        v[state.idx] = state.nu_rel
        return v

    em = None
    ramp = [20.0, 60.0, 100.0, 130.0, 150.0, 100.0, 60.0, 20.0]     # нагрев и остывание
    retention = []
    for T in ramp:
        state.set_temperature(np.full(state.idx.size, T))
        nu_start = nu0.copy()
        nu_start[state.idx] = state.nu_rel
        em = solve_nonlinear_2d_picard(
            space, nu_of_B=nu_fn, nu_init=nu_start, magnetization=state,
            relaxation=0.5, max_iter=300, tol=1e-6, warm_start=em,
        )
        state.commit()
        retention.append(state.retention.copy())

    r = np.array([x.mean() for x in retention])
    assert np.all(np.diff(r) <= 1e-12)          # латч только вниз, никогда не растёт
    assert r[4] < 0.99                          # нагрев до 150 C реально повредил магнит
    assert abs(r[-1] - r[4]) < 1e-12            # остывание НЕ восстановило необратимую долю
    # Ремнантность вернувшегося к 20 C магнита ниже исходной ровно во столько же раз.
    assert abs(state.retention.mean() - r[4]) < 1e-12


def test_no_damage_below_knee_stays_exactly_pristine():
    # НЕГАТИВНЫЙ тест: SmCo при той же тепловой нагрузке колено не переходит ⇒ r ровно 1.0
    # (не «близко к 1» — именно точная единица, иначе латч срабатывает от численного шума).
    space, magnet_mask, copper_mask = _segment()
    magnet = sm2co17_magnet((1.0, 0.0, 0.0))
    nu, k, cap = _materials(magnet, magnet_mask, copper_mask)
    res = solve_coupled_magneto_thermal_transient(
        space, k_cells=k, capacity_cells=cap, h=25.0, T_amb=20.0, dt=10.0, n_steps=40,
        j_cells=np.where(copper_mask, 1.2e7, 0.0),
        magnet=magnet, magnet_mask=magnet_mask, nu_init=nu,
    )
    assert res.T_magnet[-1] > 60.0                       # нагрев действительно был
    assert np.all(res.retention_min == 1.0)
    assert np.all(res.n_past_knee == 0) and np.all(res.past_knee_fraction == 0.0)
    assert res.em_converged


def test_material_choice_decides_magnet_fate_under_same_thermal_load():
    # Содержательный результат ядра К6′: при ОДНОЙ И ТОЙ ЖЕ тепловой нагрузке (та же
    # геометрия, тот же ток, та же конвекция ⇒ та же температура) NdFeB теряет ремнантность
    # необратимо, а SmCo — нет. Это и есть довод в пользу SmCo для теплонагруженного привода.
    space, magnet_mask, copper_mask = _segment()
    out = {}
    for name, magnet in (("ndfeb", n42sh_magnet((1.0, 0.0, 0.0))),
                         ("smco", sm2co17_magnet((1.0, 0.0, 0.0)))):
        nu, k, cap = _materials(magnet, magnet_mask, copper_mask)
        out[name] = solve_coupled_magneto_thermal_transient(
            space, k_cells=k, capacity_cells=cap, h=25.0, T_amb=20.0, dt=10.0, n_steps=60,
            j_cells=np.where(copper_mask, 1.2e7, 0.0),
            magnet=magnet, magnet_mask=magnet_mask, nu_init=nu,
        )
    # Тепловая часть от магнита почти не зависит ⇒ температуры совпадают, судьба — нет.
    assert abs(out["ndfeb"].T_magnet[-1] - out["smco"].T_magnet[-1]) < 1.0
    assert out["ndfeb"].retention_mean[-1] < 0.99
    assert out["smco"].retention_mean[-1] == 1.0


# ------------------------------------------- (4) независимость от демпфирования

def test_converged_answer_is_independent_of_demag_relaxation():
    # Под-релаксация не должна влиять на ОТВЕТ — только на скорость. Если влияет, значит
    # итерация не сошлась и «решение» является артефактом (именно так была обнаружена
    # неустойчивость: при рассогласованной линеаризации r_min давал 0.07/0.43/0.80
    # при ω=0.5/0.3/0.15). Здесь требуем совпадения сошедшихся ответов.
    space, magnet_mask, _ = _segment()
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    nc = magnet_mask.size
    nu0 = np.ones(nc)
    nu0[magnet_mask] = 1.0 / magnet.mu_rec

    answers = []
    for omega in (0.3, 0.15):
        state = IrreversibleMagnetState(magnet, magnet_mask, nc, relaxation=omega)

        def nu_fn(_B, st=state):
            v = nu0.copy()
            v[st.idx] = st.nu_rel
            return v

        state.set_temperature(np.full(state.idx.size, 130.0))
        nu_start = nu0.copy()
        nu_start[state.idx] = state.nu_rel
        em = solve_nonlinear_2d_picard(
            space, nu_of_B=nu_fn, nu_init=nu_start, magnetization=state,
            relaxation=1.0, max_iter=800, tol=1e-9,
        )
        assert em.converged, f"omega={omega}: итерация не сошлась"
        answers.append(state._pending.copy())

    assert np.max(np.abs(answers[0] - answers[1])) < 1e-5


# ------------------------------------------------------- (5) сходимость по шагу

def test_trajectory_converges_under_time_step_refinement():
    # Неявный Эйлер имеет первый порядок ⇒ при делении dt пополам приращения ответа должны
    # УБЫВАТЬ (последовательность сходится, а не «гуляет»). Для связанной задачи с историей
    # это главная проверка корректности интегрирования пути нагружения.
    space, magnet_mask, copper_mask = _segment()
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    nu, k, cap = _materials(magnet, magnet_mask, copper_mask)
    t_end = 600.0
    vals = []
    for dt in (60.0, 30.0, 15.0):
        res = solve_coupled_magneto_thermal_transient(
            space, k_cells=k, capacity_cells=cap, h=25.0, T_amb=20.0,
            dt=dt, n_steps=int(t_end / dt), j_cells=np.where(copper_mask, 1.2e7, 0.0),
            magnet=magnet, magnet_mask=magnet_mask, nu_init=nu,
        )
        # Сходимость по шагу имеет смысл только если КАЖДЫЙ шаг решён (иначе сравнивались бы
        # невязки, а не решения).
        assert res.em_converged and res.em_residual.max() < 1e-5
        vals.append((res.T_magnet[-1], res.retention_mean[-1]))

    dT = [abs(vals[1][0] - vals[0][0]), abs(vals[2][0] - vals[1][0])]
    dr = [abs(vals[1][1] - vals[0][1]), abs(vals[2][1] - vals[1][1])]
    assert dT[1] < 0.7 * dT[0]        # приращения температуры убывают (≈вдвое, 1-й порядок)
    assert dr[1] < 0.7 * dr[0]        # приращения потери ремнантности убывают
    assert vals[-1][1] < 0.99         # режим действительно повреждающий (иначе тест пустой)
    assert vals[-1][1] > 0.5          # ...но не разрушающий (там решение не единственно)


def test_magnetic_residual_reported_and_small_in_design_regime():
    # В проектном режиме (магнит повреждён, но не уничтожен) каждый шаг решается до 1e-6.
    space, magnet_mask, copper_mask = _segment()
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    nu, k, cap = _materials(magnet, magnet_mask, copper_mask)
    res = solve_coupled_magneto_thermal_transient(
        space, k_cells=k, capacity_cells=cap, h=25.0, T_amb=20.0, dt=15.0, n_steps=40,
        j_cells=np.where(copper_mask, 1.2e7, 0.0),
        magnet=magnet, magnet_mask=magnet_mask, nu_init=nu,
    )
    assert res.em_residual.shape == res.times.shape
    assert res.em_residual.max() < 1.0e-5
    assert res.em_converged
    assert res.retention_mean[-1] < 0.99          # повреждение реально произошло


def _deep_damage_run(J=1.6e7, **kw):
    space, magnet_mask, copper_mask = _segment()
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    nu, k, cap = _materials(magnet, magnet_mask, copper_mask)
    res = solve_coupled_magneto_thermal_transient(
        space, k_cells=k, capacity_cells=cap, h=25.0, T_amb=20.0, dt=10.0, n_steps=60,
        j_cells=np.where(copper_mask, J, 0.0),
        magnet=magnet, magnet_mask=magnet_mask, nu_init=nu, **kw,
    )
    return space, magnet_mask, copper_mask, magnet, res


def test_unconverged_magnetic_step_stops_the_run_without_a_verdict():
    # ЭТАЛОН-ПИКАР: глубоко за коленом он на шаге не сходится ни при каком дроблении. Прежде это
    # объявлялось каскадом («равновесия не существует»), но равновесие есть (следующий тест, и его же
    # находит Ньютон по шагам — тест ниже) — это несходимость метода (Л-93, Л-107, Л-119). Требование:
    # ОСТАНОВИТЬСЯ с честной причиной, не выносить вердикт о магните и не пускать несошедшийся шаг в историю.
    space, _, _, _, res = _deep_damage_run(method="picard")
    assert res.em_failed and not res.magnet_cascade and not res.runaway
    assert len(res.times) - 1 < 60                         # раньше запрошенного горизонта
    assert "не сошлась" in res.stop_reason and "вывода о магните нет" in res.stop_reason
    # Возвращённая история — ДОСТОВЕРНЫЙ ПРЕФИКС: каждый принятый шаг решён с промахом меньше
    # порядка (порог по умолчанию 10·em_tol), сорвавшийся шаг в неё не попал.
    assert res.em_residual.max() < 10 * 1.0e-6
    assert res.retention_mean[-1] < 0.99                   # повреждение до остановки реально
    # Состояние магнита откачено к последнему принятому шагу (не к мусору внутри срыва).
    # Средняя доля — с весом площади ячеек: поштучное среднее зависит от сетки (Л-104).
    a = np.array([space.mesh.cell_area(int(c)) for c in res.state.idx])
    assert abs(float(np.average(res.state.retention, weights=a)) - res.retention_mean[-1]) < 1e-12
    assert 0.0 < res.past_knee_fraction.max() <= 1.0


def test_where_picard_stopped_the_equilibrium_exists():
    # ОРАКУЛ к остановке: то же состояние — ток, накопленная доля r и температура магнита на
    # остановке (однородно по самой горячей ячейке, т. е. жёстче, чем в расчёте) — статическим
    # Ньютоном с законом магнита в касательной. Он сходится за единицы итераций, и ни одна ячейка
    # не уходит ниже H_cJ: равновесие существует, «каскада» нет — остановка была численной.
    from magcore.fem2d.model.materials import Air, MagnetMaterial
    from magcore.fem2d.model.problem import Problem2D, Region2D, solve_problem2d

    space, magnet_mask, copper_mask, magnet, res = _deep_damage_run(method="picard")
    assert res.em_failed                                   # предпосылка: Пикар остановился
    mesh = space.mesh
    assert res.T_stop is not None                          # поле остановки — вне истории (запись не принята)
    T_cells = res.T_stop[mesh.cells].mean(axis=1)
    T_stop = float(T_cells[magnet_mask].max())             # та же T, что в причине остановки
    assert ("T=%.1f C" % T_stop) in res.stop_reason
    retention = np.ones(mesh.n_cells)
    retention[res.state.idx] = res.state.retention
    axis = np.zeros((mesh.n_cells, 2))
    axis[magnet_mask] = (1.0, 0.0)
    prob = Problem2D(mesh=mesh, cell_region=magnet_mask.astype(int),
                     regions={0: Region2D(0, "air", Air()), 1: Region2D(1, "magnet", MagnetMaterial(magnet))},
                     magnet_axis=axis, j_cells=np.where(copper_mask, 1.6e7, 0.0), T=T_stop)
    sol = solve_problem2d(prob, method="newton", retention=retention, tol=1e-10, max_iter=30)
    assert sol.field.converged and sol.field.n_iterations <= 10
    assert sol.field.rel_change_history[-1] <= 1e-10
    assert not np.any(sol.risk.beyond_hcj)                 # ниже H_cJ — ни одной ячейки
    assert np.any(sol.risk.demagnetized)                   # и это именно за коленом, а не «легко»


def test_newton_solves_every_step_where_picard_stopped():
    # Тот же глубокий режим — Ньютоном (по умолчанию). Каждый шаг решён до допуска за единицы итераций без
    # дробления, и расчёт идёт дальше остановки Пикара — до предела модели магнита (у N42SH ≈ 202 °C, где
    # H_cJ(T) → 0): остановка «магнит перегрет», а не несходимость и не каскад.
    *_, pic = _deep_damage_run(method="picard")
    *_, res = _deep_damage_run()
    assert pic.em_failed                                   # предпосылка: здесь Пикар остановился
    assert not res.em_failed and not res.magnet_cascade
    assert res.em_converged and res.em_residual.max() <= 1.0e-9
    assert np.all(res.em_substeps[1:] == 1)                # дробление не понадобилось ни разу
    assert res.em_iterations[1:].max() <= 10               # у Пикара здесь 22–52 итерации на шаг
    assert len(res.times) > len(pic.times)                 # прошёл дальше остановки Пикара
    assert res.runaway and "магнит перегрет" in res.stop_reason
    assert np.all(np.diff(res.retention_min) <= 0.0)       # защёлка — только вниз
    assert res.retention_mean[-1] < res.retention_mean[len(pic.times) - 1] < 1.0
    # Защёлка согласована с ПОСЛЕДНИМ принятым полем: r ≤ r_now(H∥, T) в каждой ячейке (у узла сетки
    # температур, где берётся кривая), а там, где шаг дал новую потерю, — ровно r_now. Защёлка с опозданием
    # на шаг (или до решения) нарушила бы первое.
    st = res.state
    r_now = np.array([float(st.magnet.retention_now(h, t)) for h, t in zip(st.H_par, st.law_temperature(st.T_mag))])
    assert np.all(st.retention <= r_now) and np.any((st.retention == r_now) & (r_now < 1.0))
    # Тепло от метода не зависит вовсе: потери меди от поля не зависят, поэтому история T до остановки
    # Пикара совпадает бит в бит — магнитный метод не трогает тепловую часть.
    assert np.array_equal(res.T_hist[:len(pic.times)], pic.T_hist)


def test_overheat_stop_also_at_the_temperature_grid_node():
    # Кривая берётся в узле сетки температур (шаг T_bin = 0,5 °C). У самого предела модели (у N42SH 201,82 °C)
    # ячейка при 201,79 °C ещё в пределе, а её узел 202,0 °C — уже за ним, и кривой там нет. Остановка «магнит
    # перегрет» должна случиться и тогда — честно, с причиной, а не исключением из глубины модели магнита.
    *_, magnet, res = _deep_damage_run(J=1.5e7)
    limit = magnet.temperature_limit()
    assert res.runaway and not res.em_failed and not res.magnet_cascade
    assert "узел сетки температур" in res.stop_reason and ("%.2f C" % limit) in res.stop_reason
    T_stop = float(res.stop_reason.split("T=")[1].split(" C")[0])
    assert T_stop < limit <= round(T_stop / 0.5) * 0.5        # сама T в пределе, узел — за ним


def test_newton_and_picard_solve_the_same_magnetic_step():
    # ОРАКУЛ (8): из ОДНОГО состояния — поячеечная T с перепадом 8 °C поперёк магнита и защёлкнутая доля r —
    # оба метода решают один закон (та же сетка температур T_bin, те же ветви). Проверяется гипотеза «предел
    # Пикара — решение Ньютона». Пикар сходится линейно, его ошибка ≈ C·tol, поэтому ужесточение допуска в 10⁴
    # раз уменьшает расхождение с Ньютоном (дожат до невязки 1e-13) примерно во столько же; сходись он к другой
    # точке — другой закон, например кривая не в узле сетки температур (разница ~|dB_r/dT|·T_bin/2 ≈ 3e-4), —
    # расхождение упёрлось бы в эту разницу и почти не менялось. Граница 1e-2 — среднее геометрическое двух
    # исходов (1e-4 и 1), запас 100 на разброс C (у этого эталона C = 2…20).
    # Оценка ошибки Пикара tol·ρ/(1−ρ) с одним ρ здесь НЕПРИМЕНИМА: с релаксацией ветви он сходится немонотонно
    # (отношения соседних поправок от 0,03 до 3,5), и на шаге остывания реальная ошибка втрое выше такой оценки.
    # Эталон — как в ядре (источник `IrreversibleMagnetState`, тензорная ν, ν-релаксация 0,5), но ω ветви 0,15:
    # при штатных 0,3 он на 145–160 °C зацикливается (период 3) — это его несходимость, Ньютон там решает.
    # Путь — нагрев за колено (новая потеря внутри шага) и остывание (линия возврата с памятью). Магнит
    # анизотропный (μ⊥ = 1,17 ≠ μ_rec, как у NdFeB), чтобы сверялся и закон поперёк оси.
    space, mm, cm = _segment()
    magnet = dataclasses.replace(n42sh_magnet((1.0, 0.0, 0.0)), mu_perp=1.17)
    nu, _, _ = _materials(magnet, mm, cm)
    nc = mm.size
    j = MU0 * np.where(cm, 1.4e7, 0.0)
    x = np.array([space.mesh.cell_vertices(int(c)).mean(axis=0)[0] for c in np.where(mm)[0]])
    spread = 8.0 * (x - x.min()) / (x.max() - x.min())
    newton = MagnetLawState(magnet, mm, nc)
    zero = np.zeros(nc)
    a_prev = None
    new_loss = []
    for T_base in (100.0, 125.0, 145.0, 160.0, 120.0, 60.0):
        T_mag = T_base + spread
        newton.set_temperature(T_mag)
        law = newton.law()
        em_n = solve_nonlinear_2d_newton(space, lambda B: (nu.copy(), zero), nu_init=nu, j_cells=j,
                                         magnet_law=law, max_iter=40, tol=1e-13, a_init=a_prev)
        newton.accept(law)
        assert em_n.converged
        errs = []
        for tol in (1e-8, 1e-12):
            pic = IrreversibleMagnetState(magnet, mm, nc, relaxation=0.15)
            pic.retention = newton.retention.copy()            # то же состояние: защёлка до этого шага
            pic.set_temperature(T_mag)
            em_p = solve_nonlinear_2d_picard(
                space, nu_of_B=lambda B, _s=pic: _s.nu_field(nu.copy()), nu_init=pic.nu_field(nu.copy()),
                j_cells=j, magnetization=pic, relaxation=0.5, max_iter=20000, tol=tol,
            )
            assert em_p.converged
            errs.append(np.linalg.norm(em_n.B_cells - em_p.B_cells) / np.linalg.norm(em_p.B_cells))
        assert errs[1] < 1e-2 * errs[0], f"T={T_base}: {errs}"
        before = newton.retention.copy()
        newton.commit()
        new_loss.append(bool(np.any(newton.retention < before)))
        a_prev = em_n.a
    assert any(new_loss[:4]) and not any(new_loss[4:])     # потеря шла при нагреве, при остывании — нет


def test_newton_memory_survives_cooling():
    # (3) для Ньютона: нагрев за колено съедает долю r навсегда, остывание её не возвращает — латч только
    # вниз, после 20 °C доля та же, что на пике; ремнантность холодного магнита r·B_r(20 °C) ниже исходной.
    space, magnet_mask, _ = _segment()
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    nc = magnet_mask.size
    nu0 = np.ones(nc)
    nu0[magnet_mask] = 1.0 / magnet.mu_rec
    state = MagnetLawState(magnet, magnet_mask, nc)
    em = None
    retention = []
    for T in (20.0, 60.0, 100.0, 130.0, 150.0, 100.0, 60.0, 20.0):
        state.set_temperature(np.full(state.idx.size, T))
        law = state.law()
        em = solve_nonlinear_2d_newton(space, lambda B: (nu0.copy(), np.zeros(nc)), nu_init=nu0,
                                       magnet_law=law, a_init=None if em is None else em.a)
        assert em.converged
        state.accept(law)
        state.commit()
        retention.append(state.retention.copy())
    r = np.array([x.mean() for x in retention])
    assert np.all(np.diff(r) <= 0.0)
    assert r[4] < 0.99                                     # нагрев до 150 °C повредил магнит
    assert np.array_equal(retention[-1], retention[4])     # остывание доли не вернуло — ни в одной ячейке


def test_newton_unconverged_step_still_stops_without_a_verdict():
    # Страховка остаётся и у Ньютона: при бюджете в одну итерацию и без дробления первый же шаг за коленом
    # (нелинейный — линейный шаг Ньютон решает за одну итерацию точно) не сходится — остановка без вердикта,
    # история — достоверный префикс, состояние магнита — на последнем принятом шаге.
    space, _, _, _, res = _deep_damage_run(em_max_iter=1, max_substeps=1)
    assert res.em_failed and not res.magnet_cascade and not res.runaway
    assert "не сошлась" in res.stop_reason and "вывода о магните нет" in res.stop_reason
    assert 1 < len(res.times) - 1 < 60
    assert res.em_residual.max() <= 1.0e-9                 # в историю вошли только решённые шаги
    a = np.array([space.mesh.cell_area(int(c)) for c in res.state.idx])
    assert float(np.average(res.state.retention, weights=a)) == res.retention_mean[-1]


def test_final_attempt_completes_the_heating_path_after_a_near_miss():
    # На последней попытке дробления подшаг, не дошедший до допуска, но с промахом меньше порога приёма,
    # принимается — и путь досчитывается до T шага. Прежде цикл на таком подшаге обрывался, и шаг входил в
    # историю с магнитом при ПРОМЕЖУТОЧНОЙ температуре и недосчитанной потерей. Чтобы путь был виден, порог
    # приёма нарочно широкий (1), бюджет — одна итерация, дробление — до двух подшагов: за коленом подшаги не
    # дожимаются, но принимаются. Проверка: магнит в каждой записи — ровно при температуре ЭТОЙ ЖЕ записи
    # (запись n — состояние в момент t_n: поле T^n и магнит, принятый при T^n; без сдвига на шаг).
    space, magnet_mask, _, _, res = _deep_damage_run(em_max_iter=1, max_substeps=2, em_abort_residual=1.0)
    assert not res.em_failed and not res.em_converged         # промахи были и приняты
    assert np.any(res.em_substeps == 2)                         # последняя попытка — из двух подшагов
    T_cells = res.T_hist[:, space.mesh.cells].mean(axis=2)
    T_rec = T_cells[:, magnet_mask].max(axis=1)                 # T магнита в поле записи n
    # подшаг i из n кончается в T_пред + (i/n)·(T_шага − T_пред): на последнем — та же T до округления
    assert np.allclose(res.T_magnet, T_rec, rtol=0.0, atol=1e-9)


def test_cascade_is_read_from_the_accepted_newton_solution():
    # ОРАКУЛ (9): магнит при 195 °C (H_cJ — 4 % номинала) под сильным встречным током: у сошедшегося решения
    # часть ячеек ниже −H_cJ — модель магнита там не определена, расчёт останавливается признаком «каскад».
    # Те же ячейки даёт статический Ньютон (`solve_problem2d`) при той же T и том же токе: на первом шаге
    # поле T однородно, магнит новый — задача та же.
    from magcore.fem2d.model.materials import Air, MagnetMaterial
    from magcore.fem2d.model.problem import Problem2D, Region2D, solve_problem2d

    T, J = 195.0, 4.0e7
    space, magnet_mask, copper_mask = _segment()
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    nu, k, cap = _materials(magnet, magnet_mask, copper_mask)
    res = solve_coupled_magneto_thermal_transient(
        space, k_cells=k, capacity_cells=cap, h=25.0, T_amb=T, T0=np.full(space.ndofs, T), dt=10.0,
        n_steps=3, j_cells=np.where(copper_mask, J, 0.0), magnet=magnet, magnet_mask=magnet_mask, nu_init=nu,
    )
    assert res.magnet_cascade and not res.em_failed and not res.runaway
    assert len(res.times) == 1 and np.all(res.state.retention == 1.0)   # шаг не принят, состояние откачено

    axis = np.zeros((magnet_mask.size, 2))
    axis[magnet_mask] = (1.0, 0.0)
    prob = Problem2D(mesh=space.mesh, cell_region=magnet_mask.astype(int),
                     regions={0: Region2D(0, "air", Air()), 1: Region2D(1, "magnet", MagnetMaterial(magnet))},
                     magnet_axis=axis, j_cells=np.where(copper_mask, J, 0.0), T=T)
    sol = solve_problem2d(prob, method="newton", tol=1e-12, max_iter=40)
    assert sol.field.converged
    n_static = int(np.count_nonzero(sol.risk.beyond_hcj))
    assert n_static > 0 and ("%d ячеек ниже H_cJ" % n_static) in res.stop_reason


def test_cascade_on_an_intermediate_substep_stops_the_run(monkeypatch):
    # Выход за −H_cJ проверяется на КАЖДОМ подшаге, а не только на последнем: потеря подшага уже защёлкнута
    # (`commit`), и r = 0 там взята из доопределения закона за пределом модели, даже если к концу шага поле
    # вернулось выше −H_cJ. Управляющая логика проверяется подменой (спокойный SmCo сам за −H_cJ не уходит):
    # первая попытка шага 1 объявляется несошедшейся ⇒ шаг дробится на 2 подшага; на ПЕРВОМ из них одна ячейка
    # помечается «ниже H_cJ», на последнем — нет. Ожидается остановка «каскад» без записи шага 1.
    import magcore.fem2d.coupled_transient as ct

    calls = {"solve": 0, "accept": 0}
    real_solve, real_accept = ct.solve_nonlinear_2d_newton, ct.MagnetLawState.accept

    def solve(*a, **kw):
        calls["solve"] += 1
        res = real_solve(*a, **kw)
        if calls["solve"] == 2:                          # вызов 1 — состояние t_0, вызов 2 — шаг 1 целиком
            return dataclasses.replace(res, converged=False, rel_change_history=(1.0, 1.0))
        return res

    def accept(self, law):
        calls["accept"] += 1
        real_accept(self, law)
        if calls["accept"] == 3:                         # первый из двух подшагов шага 1
            self.beyond_hcj = self.beyond_hcj.copy()
            self.beyond_hcj[0] = True

    monkeypatch.setattr(ct, "solve_nonlinear_2d_newton", solve)
    monkeypatch.setattr(ct.MagnetLawState, "accept", accept)
    space, magnet_mask, copper_mask = _segment()
    magnet = sm2co17_magnet((1.0, 0.0, 0.0))
    nu, k, cap = _materials(magnet, magnet_mask, copper_mask)
    res = solve_coupled_magneto_thermal_transient(
        space, k_cells=k, capacity_cells=cap, h=25.0, T_amb=20.0, dt=10.0, n_steps=3,
        j_cells=np.where(copper_mask, 1.2e7, 0.0), magnet=magnet, magnet_mask=magnet_mask, nu_init=nu,
    )
    assert calls["accept"] == 4                          # t_0 + неудачная попытка + два подшага
    assert res.magnet_cascade and "1 ячеек ниже H_cJ" in res.stop_reason
    assert len(res.times) == 1 and res.T_stop is not None   # шаг 1 не принят, его поле — вне истории
    assert np.all(res.state.retention == 1.0)            # состояние откачено к принятому (t_0)


@pytest.mark.parametrize("method", ["newton", "picard"])
def test_substepping_is_engaged_but_idle_when_not_needed(method):
    # Дробление шага по температуре должно включаться ТОЛЬКО по необходимости: в спокойном
    # режиме — ровно один подшаг (иначе оно молча удорожало бы каждый расчёт).
    space, magnet_mask, copper_mask = _segment()
    magnet = sm2co17_magnet((1.0, 0.0, 0.0))
    nu, k, cap = _materials(magnet, magnet_mask, copper_mask)
    calm = solve_coupled_magneto_thermal_transient(
        space, k_cells=k, capacity_cells=cap, h=25.0, T_amb=20.0, dt=10.0, n_steps=30,
        j_cells=np.where(copper_mask, 1.2e7, 0.0),
        magnet=magnet, magnet_mask=magnet_mask, nu_init=nu, method=method,
    )
    assert calm.em_converged
    assert calm.em_substeps.max() == 1

    # А когда шаг целиком не решается — включается и СПАСАЕТ его: в историю шаг попадает решённым до
    # допуска на ВСЕХ подшагах (шаг, принятый с невязкой выше допуска, этим тестом не считается — при
    # 1,6e7 таким у Пикара был единственный «спасённый» шаг, и он был не решён). Пикару шаг за коленом
    # не по силам при 1,5e7; Ньютон его решает за 2–4 итерации, поэтому ему бюджет урезан до двух: целый
    # шаг не успевает, мелкие подшаги с тёплым стартом — успевают.
    if method == "picard":
        _, _, _, _, hard = _deep_damage_run(J=1.5e7, method="picard")
        tol = 1.0e-6
    else:
        _, _, _, _, hard = _deep_damage_run(J=1.5e7, em_max_iter=2)
        tol = 1.0e-9
    rescued = hard.em_substeps > 1
    assert rescued.any()
    assert np.all(hard.em_residual[rescued] < tol)
    if method == "newton":
        assert hard.em_converged and not hard.em_failed     # спасены все шаги до предела модели магнита


# -------------------------------------------------------------- (6) прочие негативные

def test_steady_stop_matches_long_run():
    # ОРАКУЛ авто-стопа: остановка по |dT/dt|<tol даёт то же УСТАНОВИВШЕЕСЯ поле, что и длинный
    # прогон (неявный Эйлер для нагрева монотонно приближается к равновесию), но за меньше шагов.
    # Чисто тепловая связка (magnet=None): равновесие = баланс медных потерь и конвекции.
    space, _magnet_mask, copper_mask = _segment()
    kw = dict(
        k_cells=np.where(copper_mask, 400.0, 1.0),
        capacity_cells=np.where(copper_mask, 3.45e6, 1.2e6),
        h=25.0, T_amb=20.0, dt=5.0, j_cells=np.where(copper_mask, 3.0e6, 0.0),
    )
    # Длинный горизонт = эталон установившегося поля; допуск по скорости мал ⇒ остаточный
    # недобор до равновесия ~steady_tol·τ тоже мал (пропорционален допуску).
    long = solve_coupled_magneto_thermal_transient(space, n_steps=3000, **kw)
    fast = solve_coupled_magneto_thermal_transient(space, n_steps=3000, steady_tol=1e-4, **kw)
    assert not long.runaway and not fast.runaway            # ниже порога — равновесие есть
    assert not fast.magnet_cascade
    assert "установившийся" in fast.stop_reason
    assert fast.times.size < long.times.size                # остановился РАНЬШЕ полного горизонта
    assert abs(fast.T_max[-1] - long.T_max[-1]) < 0.15      # поле практически стационарно
    # negative: без допуска ранней остановки нет — проходит все шаги и завершается штатно.
    full = solve_coupled_magneto_thermal_transient(space, n_steps=8, **kw)
    assert full.times.size == 9 and full.stop_reason == "завершено"


def test_zero_current_keeps_ambient_and_pristine_magnet():
    space, magnet_mask, copper_mask = _segment()
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    nu, k, cap = _materials(magnet, magnet_mask, copper_mask)
    res = solve_coupled_magneto_thermal_transient(
        space, k_cells=k, capacity_cells=cap, h=25.0, T_amb=35.0, dt=20.0, n_steps=10,
        j_cells=None, magnet=magnet, magnet_mask=magnet_mask, nu_init=nu,
    )
    assert np.max(np.abs(res.T_hist - 35.0)) < 1e-9        # нет источника — нет нагрева
    assert np.all(res.loss_power == 0.0)
    assert not res.runaway


def test_magnet_requires_mask_and_nu():
    space, magnet_mask, _ = _segment()
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="magnet_mask"):
        solve_coupled_magneto_thermal_transient(
            space, k_cells=1.0, capacity_cells=1.0e6, h=10.0, T_amb=20.0,
            dt=1.0, n_steps=1, magnet=magnet,
        )
    with pytest.raises(ValueError, match="nu_init"):
        solve_coupled_magneto_thermal_transient(
            space, k_cells=1.0, capacity_cells=1.0e6, h=10.0, T_amb=20.0,
            dt=1.0, n_steps=1, magnet=magnet, magnet_mask=magnet_mask,
        )
