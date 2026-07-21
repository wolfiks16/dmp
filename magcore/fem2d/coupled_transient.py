from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.fem2d.losses import copper_loss_density
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.fem2d.thermal import ImplicitEulerThermalStepper

# S3, инкремент 3 — ЯДРО К6′: связка магнит↔тепло↔размагничивание ВО ВРЕМЕНИ.
#
# На каждом шаге по времени (квазистатическая магнитная задача — электрический период
# много короче тепловой постоянной времени):
#   T^n  →  свойства материалов: кривая магнита B(H,T) и ρ(T) меди
#        →  магнитостатика с магнитом при местной T (Picard)
#        →  НЕОБРАТИМОЕ размагничивание: рабочая точка за коленом фиксируется (latch)
#        →  потери q(T) (медь I²R; крючок для железа/вихревых)
#        →  шаг неявного Эйлера по теплу  →  T^{n+1}
#
# Отличие от `magneto_thermal.solve_magneto_thermal_demag` (один стационарный проход):
# здесь ИСТОРИЯ — необратимая потеря накапливается вдоль траектории нагрева, поэтому
# конечное состояние зависит от ПУТИ, а не только от конечной температуры. Это и есть
# «необратимость в итерации» + разгон/риск в петле.
#
# ГРАНИЦА ПРИМЕНИМОСТИ (проверено тестами, см. tests/unit/test_coupled_transient.py):
# пока магнит повреждён, но не уничтожен, каждый шаг решается до 1e-6, а траектория сходится
# при измельчении dt. При ГЛУБОКОМ размагничивании у предела температуры модели (B_r → 0,
# кривая вырождается) квазистатическое равновесие перестаёт быть единственным: ответ начинает
# зависеть от настроек итерации. Такой результат НЕ выдаётся молча — `em_converged=False` и
# достигнутая невязка `em_residual` идут наружу. Плановое лечение — продолжение (substepping)
# по температуре внутри шага.
#
# ФИКСАЦИЯ НЕОБРАТИМОСТИ ведётся по ДОЛЕ СОХРАНЁННОЙ ремнантности r = B_r_eff/B_r(T),
# а не по «наихудшему H»: одно и то же поле H при более высокой T разрушительнее (колено
# уходит вверх), а при остывании обратимая часть должна вернуться по α_Br, необратимая —
# нет. Поэтому r латчится (только вниз), а рабочая ремнантность = r·B_r(T) текущей T.


@dataclass(frozen=True, slots=True)
class CoupledTransientResult:
    """История связанного расчёта (по шагам; индекс 0 = начальное состояние)."""

    times: np.ndarray             # (n+1,) время [с]
    T_hist: np.ndarray            # (n+1, ndofs) узловое поле температуры [°C]
    T_max: np.ndarray             # (n+1,) макс. температура в области [°C]
    T_magnet: np.ndarray          # (n+1,) hot-spot температура магнита [°C] (nan без магнита)
    retention_min: np.ndarray     # (n+1,) минимальная по ячейкам доля B_r_eff/B_r(T)
    retention_mean: np.ndarray    # (n+1,) средняя доля
    n_past_knee: np.ndarray       # (n+1,) число ячеек магнита за коленом
    loss_power: np.ndarray        # (n+1,) полные потери ∫q dV [Вт/м]
    stored_energy: np.ndarray     # (n+1,) ∫c·T dV [Дж/м]
    outflow: np.ndarray           # (n+1,) отвод ∫h(T−T_amb) ds [Вт/м]
    runaway: bool                 # True ⇒ остановлено по перегреву (тепловой разгон)
    stop_reason: str
    em_converged: bool            # все магнитные решения достигли em_tol
    em_iterations: np.ndarray     # (n+1,) число Picard-итераций магнитной задачи на шаге
    em_residual: np.ndarray       # (n+1,) ДОСТИГНУТАЯ относительная невязка по B на шаге
    state: "IrreversibleMagnetState | None"   # финальное состояние магнита (латч r)


class IrreversibleMagnetState:
    """
    Состояние магнита с ПАМЯТЬЮ о необратимой потере, пригодное как источник намагниченности
    для `solve_nonlinear_2d_picard` (callable(B,H,nu) -> (n_cells,2)).

    Хранит по ячейкам магнита долю сохранённой ремнантности r ∈ (0,1]; рабочая ремнантность
    на шаге = r·B_r(T_ячейки). Внутри Picard-итерации кандидат r_now = B_r_eff(H_par,T)/B_r(T)
    и берётся r_eff = min(r_латч, r_now) — то есть новая необратимая потеря учитывается СРАЗУ
    (в той же итерации, самосогласованно с полем), а `commit()` фиксирует её в истории после
    сходимости шага. При остывании r не растёт — возвращается только обратимая часть через B_r(T).

    Кривая B(H,T) строится не для каждой ячейки, а для сетки температур с шагом `T_bin`
    (кэш): ошибка ограничена |dB_r/dT|·T_bin (≈0,12 %/К · T_bin) и контролируемо → 0.
    """

    def __init__(
        self,
        magnet: AnisotropicBHTMagnet,
        magnet_mask: np.ndarray,
        n_cells: int,
        *,
        axis=(1.0, 0.0),
        mu0: float = MU0,
        relaxation: float = 0.3,
        T_bin: float = 0.5,
        switch_band: float = 0.15,
    ) -> None:
        mask = np.asarray(magnet_mask, dtype=bool).reshape(-1)
        if mask.shape != (n_cells,):
            raise ValueError("magnet_mask must have shape (n_cells,).")
        if not (0.0 < relaxation <= 1.0):
            raise ValueError("relaxation must be in (0, 1].")
        if float(T_bin) <= 0.0:
            raise ValueError("T_bin must be positive.")
        if float(switch_band) <= 0.0:
            raise ValueError("switch_band must be positive.")
        self.magnet = magnet
        self.mask = mask
        self.idx = np.where(mask)[0]
        self.n_cells = int(n_cells)
        self.mu0 = float(mu0)
        self.omega = float(relaxation)
        self.T_bin = float(T_bin)
        self.switch_band = float(switch_band)
        self.nu_rec = 1.0 / magnet.mu_rec

        ax = np.asarray(axis, dtype=float)
        if ax.ndim == 1:
            self.axes = np.broadcast_to(ax, (self.idx.size, ax.shape[0])).copy()
        elif ax.ndim == 2 and ax.shape[0] == n_cells:
            self.axes = ax[self.idx].copy()
        else:
            raise ValueError("axis must be (dim,) or (n_cells, dim).")
        self.dim = int(self.axes.shape[1])

        self.retention = np.ones(self.idx.size, dtype=float)   # латч необратимости
        self.T_mag = np.zeros(self.idx.size, dtype=float)      # температура ячеек магнита
        self.H_par = np.zeros(self.idx.size, dtype=float)      # рабочее поле вдоль e [А/м]
        self.nu_rel = np.full(self.idx.size, self.nu_rec, dtype=float)  # касательная ν магнита
        self._pending: np.ndarray | None = None                # r_eff последней итерации
        self._prev: tuple[np.ndarray, np.ndarray] | None = None
        self._curve_cache: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    # --- температура шага ---
    def set_temperature(self, T_magnet_cells) -> None:
        """Задать температуру ячеек магнита на текущем шаге и начать новую Picard-итерацию."""
        T = np.asarray(T_magnet_cells, dtype=float).reshape(-1)
        if T.shape != (self.idx.size,):
            raise ValueError("T_magnet_cells must have shape (n_magnet_cells,).")
        limit = self.magnet.temperature_limit()
        if float(T.max()) >= limit:
            raise MagnetOverheated(
                "Магнит перегрет: T=%.1f C >= предел модели %.1f C." % (float(T.max()), limit),
                T_magnet=float(T.max()), limit=float(limit),
            )
        self.T_mag = T
        # `_prev` (состояние ветви) НЕ сбрасывается: между шагами по времени поле меняется
        # слабо, и прошлое состояние — хорошее начальное приближение. На неподвижную точку
        # начальное приближение не влияет.

    # --- кривая при температуре (кэш по бинам) ---
    def _curve(self, b: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Табулированная кривая B(H) при T=b·T_bin и номинал B_r(T) — строится один раз."""
        cached = self._curve_cache.get(b)
        if cached is None:
            Tb = b * self.T_bin
            curve = self.magnet.curve_at(Tb)
            cached = (np.asarray(curve.H_values, dtype=float),
                      np.asarray(curve.B_values, dtype=float),
                      float(self.magnet.Br(Tb)))
            self._curve_cache[b] = cached
        return cached

    def _curve_eval(self, h_par: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        По ячейкам при их температуре: (B^maj(H), dB^maj/dH, B_r(T), маска «ниже H_cJ»).

        Наклон берётся как наклон ТОГО ЖЕ отрезка таблицы, по которому интерполируется B,
        поэтому линеаризация точно согласована с используемой кривой (неподвижная точка
        итерации не смещается — меняется только скорость сходимости).

        Ниже H_cJ (конец таблицы) модель кривой не определена. Принято КОНСЕРВАТИВНОЕ
        доопределение: ремнантность потеряна полностью (r=0), ячейка идёт по линии возврата
        из начала координат B=μ₀μ_rec·H. Альтернатива «зажать H на H_cJ» была бы оптимистичной
        (утверждала бы, что глубже H_cJ потерь больше не возникает) и вдобавок рассогласовала
        бы ветви: B^maj считался бы по зажатому H, а линия возврата — по фактическому.
        """
        bins = np.round(self.T_mag / self.T_bin).astype(int)
        b_maj = np.empty(self.idx.size, dtype=float)
        slope = np.empty(self.idx.size, dtype=float)
        br_nom = np.empty(self.idx.size, dtype=float)
        beyond = np.zeros(self.idx.size, dtype=bool)
        mu_rec_abs = self.mu0 * self.magnet.mu_rec
        for b in np.unique(bins):
            sel = bins == b
            Hs, Bs, nominal = self._curve(int(b))
            h = h_par[sel]
            H = np.clip(h, Hs[0], Hs[-1])
            i = np.clip(np.searchsorted(Hs, H), 1, Hs.size - 1)
            under = h < Hs[0]
            b_maj[sel] = np.where(under, mu_rec_abs * h, np.interp(H, Hs, Bs))
            slope[sel] = np.where(under, mu_rec_abs,
                                  (Bs[i] - Bs[i - 1]) / (Hs[i] - Hs[i - 1]))
            br_nom[sel] = nominal
            beyond[sel] = under
        return b_maj, slope, br_nom, beyond

    # --- источник намагниченности + касательная ν для Picard ---
    def __call__(self, B_cells: np.ndarray, H_cells: np.ndarray, nu_cells: np.ndarray) -> np.ndarray:
        """
        Источник ν·B_r по ячейкам + (side effect) касательная ν магнита в `self.nu_rel`.

        ДВЕ ВЕТВИ гистерезисного оператора:
          * главная кривая (рабочая точка ушла ниже зафиксированного состояния — идёт
            НОВАЯ необратимая потеря): B_ветви(H) = B^maj(H);
          * линия возврата (состояние зафиксировано, новых потерь нет):
            B_ветви(H) = r·B_r(T) + μ₀μ_rec·H.

        Линия источника строится СОГЛАСОВАННО с ν, на которой собирается матрица в этой же
        итерации (`nu_cells`): B = μ_frozen·H + B_r_экв, где B_r_экв = B_ветви(H*) − μ_frozen·H*.
        Тогда неподвижная точка лежит ТОЧНО на кривой ветви при любой μ_frozen (наклон влияет
        лишь на скорость), а `nu_rel` отдаёт наклон ветви наружу — оператор стягивается к
        касательной, и итерация становится ньютоновского типа.

        Зачем: за коленом главная кривая в разы круче линии возврата. Если оставить в
        операторе μ_rec, простая подстановка зацикливается, и ответ начинает зависеть от
        демпфирования (проверено: r_min 0.07/0.43/0.80 при ω=0.5/0.3/0.15) — то есть решения
        просто нет. Согласованная линеаризация убирает эту зависимость.

        ⚠ ν в планарной сборке СКАЛЯРНА, поэтому наклон ветви действует и поперёк лёгкой оси
        (за коленом поперечная проницаемость завышена) — ограничение скалярной ν-постановки,
        затрагивает только ячейки, ушедшие за колено.
        """
        out = np.zeros((self.n_cells, self.dim), dtype=float)
        if self.idx.size == 0:
            return out
        # Рабочее поле вдоль лёгкой оси: H решателя в Тл ⇒ мост в А/м делением на μ₀.
        h_par = np.einsum("ij,ij->i", H_cells[self.idx], self.axes) / self.mu0
        b_maj, slope, br_nom, beyond = self._curve_eval(h_par)
        mu_rec_abs = self.mu0 * self.magnet.mu_rec                  # μ₀μ_rec [Тл/(А/м)]
        r_now = np.clip((b_maj - mu_rec_abs * h_par) / br_nom, 0.0, 1.0)
        # Выше колена главная кривая СОВПАДАЕТ с линией возврата ⇒ r ровно 1, но интерполяция
        # даёт 1−O(1e-16). Без привязки к единице латч (min по истории) накапливал бы этот шум
        # как «повреждение» магнита, который колено ни разу не переходил.
        r_now[r_now > 1.0 - 1.0e-9] = 1.0
        r_now[beyond] = 0.0                            # ниже H_cJ — полная потеря (консервативно)
        r_eff = np.minimum(self.retention, r_now)      # необратимость учтена в итерации

        # Выбор ветви — с непрерывным переходом в узкой полосе `switch_band` по r.
        # Обе ветви дают ОДНО И ТО ЖЕ B на границе (r_now = r_латч), различаясь только
        # наклоном, поэтому смешивание в полосе не смещает решение, но убирает дребезг:
        # при жёстком переключении ячейка, стоящая ровно на границе (типичная ситуация
        # сразу после commit), гоняет ν между μ_rec и вчетверо большим наклоном кривой,
        # и итерация зависает в предельном цикле на уровне 1e-3.
        w = np.clip((self.retention - r_now) / self.switch_band, 0.0, 1.0)
        b_recoil = r_eff * br_nom + mu_rec_abs * h_par
        b_branch = (1.0 - w) * b_recoil + w * b_maj
        mu_branch = (1.0 - w) * mu_rec_abs + w * slope
        nu_branch = self.mu0 / mu_branch

        # Под-релаксация СОСТОЯНИЯ ветви (B и наклон). За коленом отображение
        # r ↦ ratio(H(r)) сжимающим не является (|g'|≫1: небольшой рост ремнантности резко
        # усиливает собственное размагничивание), поэтому чистая подстановка зацикливается.
        # Релаксация не смещает неподвижную точку — только делает итерацию сходящейся;
        # независимость сошедшегося ответа от ω проверяется тестом.
        if self._prev is not None:
            b_prev, nu_prev = self._prev
            b_branch = (1.0 - self.omega) * b_prev + self.omega * b_branch
            nu_branch = (1.0 - self.omega) * nu_prev + self.omega * nu_branch
        self._prev = (b_branch, nu_branch)

        nu_frozen = np.asarray(nu_cells, dtype=float)[self.idx]
        mu_frozen = self.mu0 / nu_frozen
        b_src = b_branch - mu_frozen * h_par           # источник под ЗАМОРОЖЕННУЮ ν

        self.H_par = h_par
        self._pending = r_eff
        self.nu_rel = nu_branch                        # наклон ветви для следующей сборки
        out[self.idx] = (nu_frozen * b_src)[:, None] * self.axes      # источник ν·B_r
        return out

    def commit(self) -> None:
        """Зафиксировать необратимую потерю шага в истории (латч только вниз)."""
        if self._pending is not None:
            self.retention = np.minimum(self.retention, self._pending)

    # --- отчётность ---
    def margins(self) -> np.ndarray:
        """Маржа до колена m = H_par − H_knee(T) по ячейкам (<0 ⇒ за коленом)."""
        m = self.magnet
        knee = -(1.0 - m.gamma_Hc * (self.T_mag - m.T0) / 100.0) * m.Hk0   # = knee_field(T)
        return self.H_par - knee

    def n_past_knee(self) -> int:
        return int(np.count_nonzero(self.margins() < 0.0))


class MagnetOverheated(RuntimeError):
    """Температура магнита вышла за предел валидности модели (разгон/потеря свойств)."""

    def __init__(self, message: str, *, T_magnet: float, limit: float):
        super().__init__(message)
        self.T_magnet = float(T_magnet)
        self.limit = float(limit)


def _cell_temperature(space: LagrangeP1Space2D, T_nodes: np.ndarray) -> np.ndarray:
    """Температура по ячейкам = среднее по вершинам (P1 ⇒ значение в центроиде)."""
    return np.asarray(T_nodes, dtype=float)[space.mesh.cells].mean(axis=1)


def solve_coupled_magneto_thermal_transient(
    space: LagrangeP1Space2D,
    *,
    k_cells,
    capacity_cells,
    h: float,
    T_amb: float,
    dt: float,
    n_steps: int,
    j_cells=None,
    magnet: AnisotropicBHTMagnet | None = None,
    magnet_mask=None,
    magnet_axis=(1.0, 0.0),
    nu_of_B=None,
    nu_init=None,
    dirichlet_dofs=None,
    dirichlet_values=0.0,
    T0=None,
    extra_loss=None,
    T_cap: float | None = None,
    em_relaxation: float = 0.5,
    em_max_iter: int = 150,
    em_tol: float = 1.0e-6,
    demag_relaxation: float = 0.3,
    switch_band: float = 0.15,
    T_bin: float = 0.5,
) -> CoupledTransientResult:
    """
    Связанный магнитотепловой расчёт ВО ВРЕМЕНИ с необратимым размагничиванием в петле.

    Тепло: C∂T/∂t − div(k∇T) = q + конвекция h(T−T_amb) (неявный Эйлер, безусловно устойчив).
    Магнит: квазистатическая планарная задача на каждом шаге, магнит при местной температуре,
    необратимая потеря фиксируется (`IrreversibleMagnetState`).
    Потери: медь q=ρ(T)·J² (главная положительная обратная связь) + `extra_loss`.

    j_cells   — ФИЗИЧЕСКАЯ плотность тока [А/м²] (0 вне обмотки); для магнитного RHS
                домножается на μ₀ (относительная конвенция ν), для потерь берётся как есть.
    magnet    — None ⇒ магнитная часть не решается (чисто тепловая связка, для верификации
                порога разгона); иначе нужен `magnet_mask` (n_cells,) и ν-модель nu_of_B/nu_init.
    extra_loss— callable(T_cells, em_result|None) -> (n_cells,) [Вт/м³]: железо, вихревые и т.п.
    T_cap     — порог остановки по перегреву; по умолчанию предел модели магнита (или ∞).

    Возвращает `CoupledTransientResult`. Разгон (`runaway=True`) — это НЕ ошибка расчёта,
    а результат: история до момента срыва сохраняется целиком.
    """
    mesh = space.mesh
    nc = mesh.n_cells
    j_phys = (np.zeros(nc, dtype=float) if j_cells is None
              else np.asarray(j_cells, dtype=float).reshape(-1))
    if j_phys.shape != (nc,):
        raise ValueError("j_cells must have shape (n_cells,).")

    state: IrreversibleMagnetState | None = None
    if magnet is not None:
        if magnet_mask is None:
            raise ValueError("magnet задан — нужен magnet_mask (n_cells,).")
        if nu_init is None:
            raise ValueError("magnet задан — нужен nu_init (относительная ν по ячейкам).")
        state = IrreversibleMagnetState(
            magnet, magnet_mask, nc, axis=magnet_axis,
            relaxation=demag_relaxation, switch_band=switch_band, T_bin=T_bin,
        )
        nu0 = np.asarray(nu_init, dtype=float)
        nu_base = (lambda B: nu0.copy()) if nu_of_B is None else nu_of_B

        def nu_fn(B_cells, _st=state, _base=nu_base):
            """ν по ячейкам: воздух/сталь — от пользователя, магнит — КАСАТЕЛЬНАЯ ν ветви."""
            nu = np.asarray(_base(B_cells), dtype=float).copy()
            nu[_st.idx] = _st.nu_rel
            return nu

        nu0 = nu0.copy()
        nu0[state.idx] = state.nu_rel
        ddofs = space.boundary_dofs() if dirichlet_dofs is None else dirichlet_dofs

    if T_cap is None:
        T_cap = magnet.temperature_limit() if magnet is not None else float("inf")

    stepper = ImplicitEulerThermalStepper(space, k_cells, capacity_cells, dt=dt, h=h, T_amb=T_amb)
    areas = np.array([mesh.cell_area(c) for c in range(nc)], dtype=float)

    T = (np.full(space.ndofs, float(T_amb), dtype=float)
         if T0 is None else np.asarray(T0, dtype=float).copy())

    times: list[float] = []
    T_hist: list[np.ndarray] = []
    T_max: list[float] = []
    T_mag_hist: list[float] = []
    ret_min: list[float] = []
    ret_mean: list[float] = []
    knee_cnt: list[int] = []
    p_loss: list[float] = []
    energy: list[float] = []
    outflow: list[float] = []
    em_iters: list[int] = []
    em_res: list[float] = []
    em_ok = True
    runaway = False
    reason = "завершено"

    def record(t: float, T_field: np.ndarray, q: np.ndarray,
               n_it: int = 0, res: float = 0.0) -> None:
        times.append(t)
        T_hist.append(T_field.copy())
        T_max.append(float(T_field.max()))
        if state is not None:
            T_mag_hist.append(float(state.T_mag.max()) if state.idx.size else float("nan"))
            ret_min.append(float(state.retention.min()) if state.idx.size else 1.0)
            ret_mean.append(float(state.retention.mean()) if state.idx.size else 1.0)
            knee_cnt.append(state.n_past_knee() if state.idx.size else 0)
        else:
            T_mag_hist.append(float("nan"))
            ret_min.append(1.0)
            ret_mean.append(1.0)
            knee_cnt.append(0)
        p_loss.append(float((q * areas).sum()))
        em_iters.append(int(n_it))
        em_res.append(float(res))
        energy.append(stepper.stored_energy(T_field))
        outflow.append(stepper.convective_outflow(T_field))

    # Начальное состояние: потери при стартовой температуре (магнит ещё не решался).
    if state is not None:
        state.set_temperature(_cell_temperature(space, T)[state.idx])
    record(0.0, T, copper_loss_density(j_phys, _cell_temperature(space, T)))

    em_prev = None
    for n in range(1, int(n_steps) + 1):
        T_cells = _cell_temperature(space, T)

        em = None
        n_it = 0
        res = 0.0
        if state is not None:
            try:
                state.set_temperature(T_cells[state.idx])
            except MagnetOverheated as exc:
                runaway = True
                reason = str(exc)
                break
            # Тёплый старт с предыдущего шага: поле между шагами меняется слабо.
            em = solve_nonlinear_2d_picard(
                space, nu_of_B=nu_fn, nu_init=nu0, j_cells=MU0 * j_phys,
                magnetization=state, dirichlet_dofs=ddofs, dirichlet_values=dirichlet_values,
                relaxation=em_relaxation, max_iter=em_max_iter, tol=em_tol,
                warm_start=em_prev,
            )
            em_prev = em
            em_ok = em_ok and bool(em.converged)
            n_it = em.n_iterations
            res = em.rel_change_history[-1] if em.rel_change_history else 0.0
            state.commit()          # необратимая потеря шага уходит в историю

        q = copper_loss_density(j_phys, T_cells)
        if extra_loss is not None:
            q = q + np.asarray(extra_loss(T_cells, em), dtype=float)

        T = stepper.step(T, q)
        record(n * float(dt), T, q, n_it, res)

        T_peak = float(T.max())
        if not np.isfinite(T_peak) or T_peak >= T_cap:
            runaway = True
            reason = ("тепловой разгон: T=%.1f C достигла порога %.1f C" % (T_peak, T_cap)
                      if np.isfinite(T_peak) else "тепловой разгон: температура разошлась")
            break

    return CoupledTransientResult(
        times=np.asarray(times), T_hist=np.asarray(T_hist), T_max=np.asarray(T_max),
        T_magnet=np.asarray(T_mag_hist), retention_min=np.asarray(ret_min),
        retention_mean=np.asarray(ret_mean), n_past_knee=np.asarray(knee_cnt),
        loss_power=np.asarray(p_loss), stored_energy=np.asarray(energy),
        outflow=np.asarray(outflow), runaway=runaway, stop_reason=reason,
        em_converged=em_ok, em_iterations=np.asarray(em_iters),
        em_residual=np.asarray(em_res), state=state,
    )
