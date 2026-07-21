import numpy as np
import pytest

from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.fem2d.coupled_transient import (
    IrreversibleMagnetState,
    solve_coupled_magneto_thermal_transient,
)
from magcore.fem2d.losses import CU_ALPHA, CU_RHO0
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.runaway import runaway_threshold, solve_thermal_steady_with_feedback
from magcore.fem2d.spaces import LagrangeP1Space2D

# S3, инкремент 3 — ядро К6′: петля магнит↔тепло↔необратимый демаг во времени.
# ОРАКУЛЫ (независимые от самой связки):
#   (1) ЭНЕРГОБАЛАНС: dU/dt = ∫q dV − ∫h(T−T_amb)ds точно (неявный Эйлер, до машинной);
#   (2) ПОРОГ РАЗГОНА: обратная связь меди q(T) линейна ⇒ петля устойчива ТОГДА И ТОЛЬКО
#       ТОГДА, когда s=ρ₀αJ² < s_crit — независимого обобщённого собств. значения
#       (K+R)v=s·M·v; ниже порога установившееся поле совпадает с ПРЯМЫМ решением
#       связанной стационарной задачи, выше — расходится;
#   (3) НЕОБРАТИМОСТЬ: при остывании возвращается только обратимая часть (латч не растёт);
#   (4) НЕЗАВИСИМОСТЬ ОТ ДЕМПФИРОВАНИЯ: сошедшийся ответ не зависит от ω (иначе это не
#       решение, а артефакт итерации — так и была поймана неустойчивость за коленом);
#   (5) СХОДИМОСТЬ ПО ШАГУ: измельчение dt даёт монотонно сходящуюся траекторию;
#   (6) НЕГАТИВНЫЕ: без нагрева/за пределом колена потерь нет (ровно r=1).


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
    assert np.all(res.n_past_knee == 0)
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


def test_destructive_regime_is_flagged_not_silently_returned():
    # Глубокое размагничивание у предела модели (B_r → 0, кривая вырождается) — режим, где
    # квазистатическое равновесие перестаёт быть единственным: ответ там зависит от настроек
    # итерации, то есть доверять ему нельзя. Требование к решателю — НЕ выдавать такой
    # результат молча: em_converged=False и достигнутая невязка видны вызывающему.
    space, magnet_mask, copper_mask = _segment()
    magnet = n42sh_magnet((1.0, 0.0, 0.0))
    nu, k, cap = _materials(magnet, magnet_mask, copper_mask)
    res = solve_coupled_magneto_thermal_transient(
        space, k_cells=k, capacity_cells=cap, h=25.0, T_amb=20.0, dt=10.0, n_steps=60,
        j_cells=np.where(copper_mask, 1.6e7, 0.0),
        magnet=magnet, magnet_mask=magnet_mask, nu_init=nu,
    )
    assert res.retention_mean[-1] < 0.5           # магнит практически уничтожен
    assert not res.em_converged                   # и решатель об этом СООБЩАЕТ
    assert res.em_residual.max() > 1.0e-3


# -------------------------------------------------------------- (6) прочие негативные

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
