from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.cancel import check as cancel_check
from magcore.constants import MU0
from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.fem2d.assembly import uniaxial_nu_tensor
from magcore.fem2d.losses import copper_loss_density
from magcore.fem2d.magnet_law import MagnetLaw2D
from magcore.fem2d.newton import solve_nonlinear_2d_newton
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.fem2d.thermal import ImplicitEulerThermalStepper

# S3, инкремент 3 — ЯДРО К6′: связка магнит↔тепло↔размагничивание ВО ВРЕМЕНИ.
#
# На каждом шаге по времени (квазистатическая магнитная задача — электрический период
# много короче тепловой постоянной времени):
#   T^n  →  свойства материалов: кривая магнита B(H,T) при T КАЖДОЙ ячейки и ρ(T) меди
#        →  магнитостатика с магнитом при местной T (Ньютон с законом магнита в касательной)
#        →  НЕОБРАТИМОЕ размагничивание: доля r защёлкивается после принятого решения (latch)
#        →  потери q(T) (медь I²R; крючок для железа/вихревых)
#        →  шаг неявного Эйлера по теплу  →  T^{n+1}
#
# Отличие от `magneto_thermal.solve_magneto_thermal_demag` (один стационарный проход):
# здесь ИСТОРИЯ — необратимая потеря накапливается вдоль траектории нагрева, поэтому
# конечное состояние зависит от ПУТИ, а не только от конечной температуры. Это и есть
# «необратимость в итерации» + разгон/риск в петле.
#
# МЕТОД МАГНИТНОЙ ЗАДАЧИ (2026-10-13). Шаг решается Ньютоном: вдоль лёгкой оси — рабочая ветвь
# закона с памятью (главная кривая при новой потере, иначе линия возврата r·B_r(T)), поперёк — μ⊥,
# касательная — у той же ветви (`MagnetLawState` → `MagnetLaw2D`); сталь — касательной релуктивностью.
# Прежний хордовый Пикар с источником и релаксацией оставлен эталоном (`method='picard'`): за коленом
# он не сжимает (Л-93, Л-107) и останавливался там, где равновесие есть, — это и объявлялось
# «каскадом» (Л-119). На тех же задачах Ньютон решает каждый шаг за единицы итераций до невязки
# ≤1e-9 (сегмент — до предела модели магнита ~202 °C; спицевой двигатель — 4–10 итераций на шаг
# там, где Пикар упирался в 300), а где Пикар сходится, его предел — решение Ньютона (тесты).
#
# ОСТАНОВКИ (проверено тестами, см. tests/unit/test_coupled_transient.py):
#   * шаг, не сошедшийся даже с `max_substeps` подшагами (невязка ≥ `em_abort_residual`), НЕ
#     принимается: расчёт останавливается с `em_failed=True` и честной причиной — вывода о магните
#     нет (ни «выдержал», ни «каскад»). У Ньютона это страховка, у эталона-Пикара — обычное дело;
#   * признак «ячейки ниже H_cJ» («каскад») читается только у ПРИНЯТОГО решения: у несошедшегося
#     приближения он случаен (у Пикара на одном шаге двигателя гулял 8 → 1 → 2 между попытками).
#     Это граница модели магнита (ниже −H_cJ закон доопределён полной потерей), а не доказательство
#     несуществования равновесия: у Ньютона равновесие найдено, просто часть ячеек — за −H_cJ.
# В обоих случаях история и состояние магнита обрываются последним принятым шагом.
#
# ФИКСАЦИЯ НЕОБРАТИМОСТИ ведётся по ДОЛЕ СОХРАНЁННОЙ ремнантности r = B_r_eff/B_r(T),
# а не по «наихудшему H»: одно и то же поле H при более высокой T разрушительнее (колено
# уходит вверх), а при остывании обратимая часть должна вернуться по α_Br, необратимая —
# нет. Поэтому r латчится (только вниз), а рабочая ремнантность = r·B_r(T) текущей T.


@dataclass(frozen=True, slots=True)
class CoupledTransientResult:
    """
    История связанного расчёта. Запись n — СОСТОЯНИЕ в момент t_n (индекс 0 = начальное): поле T^n и магнит,
    принятый при T^n (температура, доля r, колено, итерации/невязка/подшаги его решения), запас и отвод при T^n.
    Исключение — `loss_power`: источник, с которым пришли в t_n (∫q^{n−1}; в записи 0 — q^0), чтобы баланс шага
    (U^n − U^{n−1})/dt = loss_power[n] − outflow[n] выполнялся точно.
    """

    times: np.ndarray             # (n+1,) время [с]
    T_hist: np.ndarray            # (n+1, ndofs) узловое поле температуры [°C]
    T_max: np.ndarray             # (n+1,) макс. температура в области [°C]
    T_magnet: np.ndarray          # (n+1,) hot-spot температура магнита [°C] при T^n (nan без магнита)
    retention_min: np.ndarray     # (n+1,) минимальная по ячейкам доля B_r_eff/B_r(T)
    retention_mean: np.ndarray    # (n+1,) средняя доля — с весом площади ячеек (поштучное среднее зависит от сетки, Л-104)
    n_past_knee: np.ndarray       # (n+1,) число ячеек магнита за коленом — только «есть / нет»: число зависит от сетки
    past_knee_fraction: np.ndarray  # (n+1,) доля площади магнита за коленом
    loss_power: np.ndarray        # (n+1,) полные потери ∫q dV [Вт/м] — источник шага (n−1 → n), см. выше
    stored_energy: np.ndarray     # (n+1,) ∫c·T dV [Дж/м]
    outflow: np.ndarray           # (n+1,) отвод ∫h(T−T_amb) ds [Вт/м]
    runaway: bool                 # True ⇒ остановлено по перегреву (тепловой разгон)
    magnet_cascade: bool          # True ⇒ остановлено: у ПРИНЯТОГО магнитного решения ячейки ниже H_cJ
    stop_reason: str
    em_converged: bool            # все магнитные решения (на всех подшагах) достигли em_tol
    em_iterations: np.ndarray     # (n+1,) суммарное число итераций магнитной задачи на шаге
    em_residual: np.ndarray       # (n+1,) ДОСТИГНУТАЯ невязка шага (наибольшая по подшагам): у Ньютона ‖R‖/R₀,
                                  #        у Пикара — относительное изменение B на последней итерации
    em_substeps: np.ndarray       # (n+1,) на сколько подшагов по T пришлось дробить шаг
    state: "MagnetLawState | IrreversibleMagnetState | None"   # финальное состояние магнита (латч r)
    em_failed: bool = False       # True ⇒ остановлено: магнитная задача не сошлась — вывода о магните нет
    em_method: str = "newton"     # чем решалась магнитная задача: 'newton' | 'picard'
    em_tol: float = 1.0e-9        # допуск магнитной задачи (в смысле em_residual своего метода)
    T_stop: np.ndarray | None = None   # (ndofs,) поле T, при котором магнит НЕ принят (перегрет/не сошлась/
                                       # каскад/разошлась) — в историю не входит; None — такой остановки не было


class _MagnetMemory:
    """
    Общее у состояний магнита ядра (Ньютон — `MagnetLawState`, эталон-Пикар — `IrreversibleMagnetState`):
    ячейки магнита, температура (под)шага с проверкой предела модели, защёлка доли r (только вниз, `commit`),
    снимок/откат и отчётность — маржа до колена, ячейки за коленом и ниже H_cJ.
    """

    def __init__(self, magnet: AnisotropicBHTMagnet, magnet_mask, n_cells: int, *, T_bin: float) -> None:
        mask = np.asarray(magnet_mask, dtype=bool).reshape(-1)
        if mask.shape != (n_cells,):
            raise ValueError("magnet_mask must have shape (n_cells,).")
        if float(T_bin) <= 0.0:
            raise ValueError("T_bin must be positive.")
        self.magnet = magnet
        self.mask = mask
        self.idx = np.where(mask)[0]
        self.n_cells = int(n_cells)
        self.T_bin = float(T_bin)
        self.retention = np.ones(self.idx.size, dtype=float)   # латч необратимости
        self.T_mag = np.zeros(self.idx.size, dtype=float)      # температура ячеек магнита
        self.H_par = np.zeros(self.idx.size, dtype=float)      # рабочее поле вдоль e [А/м]
        self.beyond_hcj = np.zeros(self.idx.size, dtype=bool)  # ячейки ниже H_cJ (каскад)
        self._pending: np.ndarray | None = None                # r_eff последнего решения

    # --- температура шага ---
    def law_temperature(self, T) -> np.ndarray:
        """Температура, при которой берётся кривая: ближайший узел сетки с шагом T_bin (как в законе)."""
        return np.round(np.asarray(T, dtype=float) / self.T_bin) * self.T_bin

    def set_temperature(self, T_magnet_cells) -> None:
        """
        Задать температуру ячеек магнита на текущем (под)шаге. Предел модели проверяется и по самой T, и по
        узлу сетки температур, где берётся кривая: у самого предела узел бывает уже за ним (кривой там нет).
        """
        T = np.asarray(T_magnet_cells, dtype=float).reshape(-1)
        if T.shape != (self.idx.size,):
            raise ValueError("T_magnet_cells must have shape (n_magnet_cells,).")
        limit = self.magnet.temperature_limit()
        if T.size and max(float(T.max()), float(self.law_temperature(T).max())) >= limit:
            raise MagnetOverheated(
                "Магнит перегрет: T=%.1f C >= предел модели %.1f C." % (float(T.max()), limit),
                T_magnet=float(T.max()), limit=float(limit),
            )
        self.T_mag = T

    def commit(self) -> None:
        """Зафиксировать необратимую потерю шага в истории (латч только вниз)."""
        if self._pending is not None:
            self.retention = np.minimum(self.retention, self._pending)

    # --- снимок/откат (для повтора шага с продолжением по температуре) ---
    def snapshot(self) -> dict:
        """Полное состояние с историей — чтобы повторить шаг мельче, а не поверх испорченного."""
        return {
            "retention": self.retention.copy(),
            "T_mag": self.T_mag.copy(),
            "H_par": self.H_par.copy(),
            "beyond_hcj": self.beyond_hcj.copy(),
            "pending": None if self._pending is None else self._pending.copy(),
        }

    def restore(self, snap: dict) -> None:
        self.retention = snap["retention"].copy()
        self.T_mag = snap["T_mag"].copy()
        self.H_par = snap["H_par"].copy()
        self.beyond_hcj = snap["beyond_hcj"].copy()
        self._pending = None if snap["pending"] is None else snap["pending"].copy()

    # --- отчётность ---
    def margins(self) -> np.ndarray:
        """
        Маржа до колена m = H_par − H_knee(T) по ячейкам (<0 ⇒ за коленом). Колено берётся при той же
        температуре, при которой решался закон, — в узле сетки `T_bin` (`law_temperature`), как у кривой,
        r_now и признака «ниже H_cJ»: иначе «за коленом» и потеря r_now мерились бы по разным кривым.
        """
        m = self.magnet
        T_law = self.law_temperature(self.T_mag)
        knee = -(1.0 - m.gamma_Hc * (T_law - m.T0) / 100.0) * m.Hk0   # = knee_field(T_law), векторно
        return self.H_par - knee

    def n_past_knee(self) -> int:
        return int(np.count_nonzero(self.margins() < 0.0))

    def n_beyond_hcj(self) -> int:
        """
        Ячейки, где собственное поле принятого решения загнало магнит ниже собственной коэрцитивности
        H_cJ: модель магнита там не определена (закон доопределён полной потерей). Расчёт на этом
        останавливается признаком «каскад» — граница применимости модели, а не вывод о несуществовании
        равновесия (это не подтвердилось, Л-119).
        """
        return int(np.count_nonzero(self.beyond_hcj))


class MagnetLawState(_MagnetMemory):
    """
    Состояние магнита с ПАМЯТЬЮ для метода НЬЮТОНА (`solve_nonlinear_2d_newton(magnet_law=…)`).

    Закон (под)шага — `MagnetLaw2D` при поячеечной температуре (сетка `T_bin` — та же, что у эталона-Пикара,
    поэтому оба метода решают один и тот же закон) и защёлкнутой доле r: вдоль лёгкой оси — главная кривая там,
    где поле даёт новую потерю (r_now(H∥, T) < r), иначе линия возврата r·B_r(T) + μ₀μ_rec·H∥; поперёк —
    постоянная μ⊥ материала (тензор, Л-20). Новая потеря учтена в САМОМ решении — выбор ветви и есть
    r_eff = min(r, r_now), — а касательная берётся у той же ветви: ни источника, ни релаксации, ни полосы
    переключения не нужно (Л-21, Л-93, Л-107). `accept` снимает с решённого поля r_eff, H∥ и признак «ниже H_cJ»,
    `commit` защёлкивает r_eff в историю — только после того, как решение (под)шага принято. При остывании r не
    растёт: возвращается обратимая часть через B_r(T).
    """

    def __init__(self, magnet: AnisotropicBHTMagnet, magnet_mask, n_cells: int, *, axis=(1.0, 0.0),
                 T_bin: float = 0.5) -> None:
        super().__init__(magnet, magnet_mask, n_cells, T_bin=T_bin)
        self.axis = axis
        self.law()                                  # ось и маска проверяются сразу, а не на первом шаге

    def law(self) -> MagnetLaw2D:
        """Закон магнита при текущих T ячеек и защёлкнутой доле r (для одного решения Ньютона)."""
        T = np.zeros(self.n_cells, dtype=float)
        T[self.idx] = self.T_mag
        r = np.ones(self.n_cells, dtype=float)
        r[self.idx] = self.retention
        return MagnetLaw2D(self.magnet, self.mask, self.n_cells, T=T, axis=self.axis, retention=r,
                           T_bin=self.T_bin)

    def accept(self, law: MagnetLaw2D) -> None:
        """Снять с решённого поля (последний вызов закона — на возвращённом A) рабочую точку и r_eff."""
        self.H_par = law.H_par.copy()
        self.beyond_hcj = law.beyond_hcj()
        self._pending = law.retention_now()


class IrreversibleMagnetState(_MagnetMemory):
    """
    Состояние магнита с ПАМЯТЬЮ о необратимой потере, пригодное как источник намагниченности
    для `solve_nonlinear_2d_picard` (callable(B,H,nu) -> (n_cells,2)) — схема эталона-Пикара.

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
        super().__init__(magnet, magnet_mask, n_cells, T_bin=T_bin)
        if not (0.0 < relaxation <= 1.0):
            raise ValueError("relaxation must be in (0, 1].")
        if float(switch_band) <= 0.0:
            raise ValueError("switch_band must be positive.")
        self.mu0 = float(mu0)
        self.omega = float(relaxation)
        self.switch_band = float(switch_band)
        self.nu_rec = 1.0 / magnet.mu_rec
        # ПОПЕРЕЧНАЯ реактивность — СВОЙСТВО МАТЕРИАЛА, постоянное. Поперечный отклик
        # спечённого магнита — когерентный поворот моментов против поля анизотропии,
        # χ_⊥ = J_s²/(2μ₀K₁); в него входит КВАДРАТ J_s, поэтому перемагниченное зерно
        # (та же ось, тот же K₁, тот же |M_s|) даёт тот же отклик ⇒ μ_⊥ НЕ зависит от
        # продольной необратимости. Именно это ломала прежняя скалярная сборка.
        self.nu_perp_rel = 1.0 / magnet.mu_perp

        ax = np.asarray(axis, dtype=float)
        if ax.ndim == 1:
            self.axes = np.broadcast_to(ax, (self.idx.size, ax.shape[0])).copy()
        elif ax.ndim == 2 and ax.shape[0] == n_cells:
            self.axes = ax[self.idx].copy()
        else:
            raise ValueError("axis must be (dim,) or (n_cells, dim).")
        self.dim = int(self.axes.shape[1])

        self.nu_rel = np.full(self.idx.size, self.nu_rec, dtype=float)  # касательная ν магнита
        # Состояние ветви при смене температуры НЕ сбрасывается: между шагами по времени поле меняется
        # слабо, и прошлое состояние — хорошее начальное приближение; на неподвижную точку оно не влияет.
        self._prev: tuple[np.ndarray, np.ndarray] | None = None
        self._curve_cache: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

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

        Таблица кривой покрывает только второй квадрант, H ∈ [−H_cJ, 0], поэтому ОБА конца
        доопределяются явно — и по РАЗНОМУ, потому что физика у них разная:

        * H > 0 (поле вдоль лёгкой оси ПОДМАГНИЧИВАЕТ — реакция якоря на части полюса всегда
          такая): магнит просто идёт вверх по линии возврата, B = B_r(T) + μ₀μ_rec·H, потерь
          НЕТ (r ровно 1). Если вместо этого зажать H в ноль, тождество B^maj = B_r + μ₀μ_rec·H
          ломается и латч записывает ФАНТОМНОЕ повреждение именно там, где магнит усиливают
          (наблюдалось: r=0.614 при нулевом числе ячеек за коленом).
        * H < −H_cJ: модель не определена; принято КОНСЕРВАТИВНОЕ доопределение r=0 (линия
          возврата из начала координат). Зажатие H на H_cJ было бы оптимистичным (утверждало
          бы, что глубже потерь не возникает) и рассогласовало бы ветви.
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
            under = h < Hs[0]                       # ниже H_cJ  → полная потеря
            over = h > Hs[-1]                       # подмагничивание → линия возврата вверх
            tab = np.interp(H, Hs, Bs)
            tab_slope = (Bs[i] - Bs[i - 1]) / (Hs[i] - Hs[i - 1])
            b_maj[sel] = np.where(under, mu_rec_abs * h,
                                  np.where(over, nominal + mu_rec_abs * h, tab))
            slope[sel] = np.where(under | over, mu_rec_abs, tab_slope)
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

        Наклон ветви `nu_rel` действует ТОЛЬКО вдоль лёгкой оси: по умолчанию ν магнита собирается
        тензором (`nu_field`: вдоль оси — наклон ветви, поперёк — постоянная ν⊥ материала). Прежняя
        скалярная сборка, где наклон ветви действовал и поперёк оси (за коленом поперечная
        проницаемость завышалась), осталась только абляцией `anisotropic_magnet=False`.
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

        # Замороженная ν вдоль лёгкой оси. При ТЕНЗОРНОЙ ν берём осевую компоненту eᵀνe:
        # источник корректируется только вдоль оси, поперёк корректировать нечего (там ν
        # и есть материальный закон).
        nu_all = np.asarray(nu_cells, dtype=float)
        nu_frozen = (np.einsum("ci,cij,cj->c", self.axes, nu_all[self.idx], self.axes)
                     if nu_all.ndim == 3 else nu_all[self.idx])
        mu_frozen = self.mu0 / nu_frozen
        b_src = b_branch - mu_frozen * h_par           # источник под ЗАМОРОЖЕННУЮ ν

        self.H_par = h_par
        self.beyond_hcj = beyond
        self._pending = r_eff
        self.nu_rel = nu_branch                        # наклон ветви для следующей сборки
        out[self.idx] = (nu_frozen * b_src)[:, None] * self.axes      # источник ν·B_r
        return out

    def nu_field(self, nu_scalar) -> np.ndarray:
        """
        Полное поле ν как ТЕНЗОР (n_cells, 2, 2), действующий на B: изотропная ν·I вне
        магнита, одноосный тензор ν_⊥(I − eeᵀ) + ν_∥ eeᵀ в магните.

        ν_∥ — наклон рабочей ветви (`nu_rel`): вдоль оси он законен, потому что источник
        строится под ту же ν и неподвижная точка от неё не зависит (см. `__call__`).
        ν_⊥ — постоянная материала (`nu_perp_rel`): поперёк корректирующего источника нет,
        и туда наклон ветви попадать не должен.
        """
        nu = np.asarray(nu_scalar, dtype=float).reshape(-1)
        if nu.shape != (self.n_cells,):
            raise ValueError("nu_scalar must have shape (n_cells,).")
        out = nu[:, None, None] * np.broadcast_to(np.eye(self.dim),
                                                  (self.n_cells, self.dim, self.dim))
        out = np.array(out, dtype=float)
        if self.idx.size:
            out[self.idx] = uniaxial_nu_tensor(self.nu_rel, self.nu_perp_rel, self.axes)
        return out

    # --- снимок/откат: к общему состоянию — состояние ветви Пикара ---
    def snapshot(self) -> dict:
        snap = super().snapshot()
        snap["nu_rel"] = self.nu_rel.copy()
        snap["prev"] = None if self._prev is None else (self._prev[0].copy(), self._prev[1].copy())
        return snap

    def restore(self, snap: dict) -> None:
        super().restore(snap)
        self.nu_rel = snap["nu_rel"].copy()
        self._prev = (None if snap["prev"] is None
                      else (snap["prev"][0].copy(), snap["prev"][1].copy()))


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
    robin=None,
    j_cells=None,
    j_loss_cells=None,
    magnet: AnisotropicBHTMagnet | None = None,
    magnet_mask=None,
    magnet_axis=(1.0, 0.0),
    nu_of_B=None,
    nu_and_dnu=None,
    nu_init=None,
    dirichlet_dofs=None,
    dirichlet_values=0.0,
    T0=None,
    extra_loss=None,
    T_cap: float | None = None,
    method: str = "newton",
    em_relaxation: float = 0.5,
    em_max_iter: int | None = None,
    em_tol: float | None = None,
    demag_relaxation: float = 0.3,
    switch_band: float = 0.15,
    T_bin: float = 0.5,
    max_substeps: int = 32,
    em_abort_residual: float | None = None,
    steady_tol: float | None = None,
    anisotropic_magnet: bool = True,
) -> CoupledTransientResult:
    """
    Связанный магнитотепловой расчёт ВО ВРЕМЕНИ с необратимым размагничиванием в петле.

    Тепло: C∂T/∂t − div(k∇T) = q + конвекция h(T−T_amb) (неявный Эйлер, безусловно устойчив).
    Магнит: квазистатическая планарная задача на каждом шаге, магнит при местной температуре
    ячеек, необратимая потеря фиксируется после принятого решения шага.
    Потери: медь q=ρ(T)·J² (главная положительная обратная связь) + `extra_loss`.

    method    — как решается магнитная задача шага:
                'newton' (по умолчанию) — метод Ньютона (`solve_nonlinear_2d_newton`) с законом магнита
                    в касательной (`MagnetLawState` → `MagnetLaw2D`: поячеечная T, память r, поперёк μ⊥) и
                    сталью через касательную релуктивность `nu_and_dnu`; «сошлось» — по относительной
                    невязке (допуск по умолчанию 1e-9), тёплый старт с решения прошлого шага;
                'picard' — прежний хордовый Пикар с источником магнита и релаксацией (`IrreversibleMagnetState`),
                    допуск — по относительному изменению B (по умолчанию 1e-6). Оставлен ЭТАЛОНОМ: за коленом
                    он может не сойтись при существующем равновесии (Л-93, Л-107, Л-119).
                Закон у обоих один (та же сетка температур `T_bin`, те же ветви), поэтому где Пикар сходится,
                ответы совпадают с точностью до его остановки.
    nu_of_B / nu_and_dnu — нелинейная сталь: для Пикара хордовая ν(B) (n_cells,), для Ньютона — пара
                (ν, dν/d|B|²) с той же ν (например, `model.problem._reluctivity_newton`). Без стали (оба None)
                ν = nu_init. Ньютону со сталью нужна именно касательная: хорда вместо неё сделала бы метод
                линейно сходящимся, а подставлять её молча нельзя — поэтому без `nu_and_dnu` отказ.
    em_max_iter, em_tol — предел итераций и допуск магнитной задачи; None — по методу (Ньютон 40 и 1e-9,
                Пикар 150 и 1e-6). `em_relaxation`, `demag_relaxation`, `switch_band` — только для Пикара.

    j_cells   — ФИЗИЧЕСКАЯ плотность тока [А/м²] (0 вне обмотки) для МАГНИТНОЙ задачи;
                домножается на μ₀ (относительная конвенция ν).
    j_loss_cells — плотность тока для ПОТЕРЬ (по умолчанию = j_cells). Разделены потому, что
                это РАЗНЫЕ величины: магнитостатику считают при МГНОВЕННОМ токе (худший угол
                реакции якоря), а нагрев — при СРЕДНЕКВАДРАТИЧНОМ за электрический период
                (тепловая постоянная времени много больше периода), причём в гомогенизированном
                пазу ток размазан по всей площади, а греется только медь ⇒ подавать надо
                J_паз/√k_зап (иначе потери занижены в 1/k_зап ≈ 2 раза).
    magnet    — None ⇒ магнитная часть не решается (чисто тепловая связка, для верификации
                порога разгона); иначе нужен `magnet_mask` (n_cells,) и ν-модель nu_of_B/nu_init.
    extra_loss— callable(T_cells, em_result|None) -> (n_cells,) [Вт/м³]: железо, вихревые и т.п.
    T_cap     — порог остановки по перегреву ЛЮБОЙ точки области (класс изоляции обмотки и
                т.п.); по умолчанию ∞. Предел применимости модели МАГНИТА проверяется отдельно
                и только по его ячейкам — эти два критерия не следует смешивать.
    max_substeps — предел дробления шага по температуре (продолжение). Если магнитная задача
                на шаге не сошлась, шаг повторяется с 2, 4, … подшагами по T: путь нагружения
                проходится мельче, что и требуется для гистерезисной задачи, где равновесие
                зависит от пути. Состояние откатывается перед каждой попыткой.
    em_abort_residual — невязка, с которой шаг НЕ принимается: магнитная задача не сошлась, расчёт
                останавливается (`em_failed=True`), вывода о магните нет. По умолчанию 10·em_tol:
                промах по допуску меньше порядка принимается и виден в `em_converged`/`em_residual`,
                больше — нет (прежний порог 1e-2 допускал в историю невязку в 10⁴ раз выше допуска).
    anisotropic_magnet — ν магнита как ТЕНЗОР (по умолчанию): вдоль лёгкой оси касательная
                ν рабочей ветви, ПОПЕРЁК — постоянная ν_⊥ = 1/(μ₀μ_perp) материала. False
                возвращает прежнюю СКАЛЯРНУЮ сборку, где наклон ветви действовал во все
                стороны и повреждённая ячейка становилась проводником потока вбок; оставлено
                ТОЛЬКО для абляции и только в схеме Пикара — физически этот режим неверен поперёк оси,
                а закон Ньютона тензорный по построению.
    steady_tol — если задан [°C/с], расчёт ОСТАНАВЛИВАЕТСЯ по выходу на установившийся режим:
                как только |ΔT_max|/dt между соседними шагами падает ниже него (после ≥2 шагов
                нагрева). Это НЕ аварийная остановка (runaway=cascade=False) — поле уже стационарно
                с точностью ~steady_tol·τ; `n_steps` служит верхним пределом. Неявный Эйлер для
                нагрева монотонно приближается к равновесию, поэтому первое срабатывание — оно.

    Возвращает `CoupledTransientResult`. Разгон (`runaway=True`) — это НЕ ошибка расчёта,
    а результат: история до момента срыва сохраняется целиком. `em_failed=True` — наоборот, НЕ
    результат: магнитная задача не сошлась, история обрывается последним принятым шагом, и о
    магните после него ничего не утверждается.
    """
    mesh = space.mesh
    nc = mesh.n_cells
    j_phys = (np.zeros(nc, dtype=float) if j_cells is None
              else np.asarray(j_cells, dtype=float).reshape(-1))
    if j_phys.shape != (nc,):
        raise ValueError("j_cells must have shape (n_cells,).")
    j_loss = (j_phys if j_loss_cells is None
              else np.asarray(j_loss_cells, dtype=float).reshape(-1))
    if j_loss.shape != (nc,):
        raise ValueError("j_loss_cells must have shape (n_cells,).")

    if method not in ("newton", "picard"):
        raise ValueError("method должен быть 'newton' | 'picard'.")
    newton = method == "newton"
    if em_max_iter is None:
        em_max_iter = 40 if newton else 150
    if em_tol is None:
        em_tol = 1.0e-9 if newton else 1.0e-6

    state: MagnetLawState | IrreversibleMagnetState | None = None
    if magnet is not None and newton:
        if magnet_mask is None:
            raise ValueError("magnet задан — нужен magnet_mask (n_cells,).")
        if nu_init is None:
            raise ValueError("magnet задан — нужен nu_init (относительная ν по ячейкам).")
        if not anisotropic_magnet:
            raise ValueError("скалярная сборка магнита (абляция) — только для method='picard'.")
        nu_lin = np.asarray(nu_init, dtype=float)
        if nu_lin.shape != (nc,):
            raise ValueError("для method='newton' nu_init — относительная ν по ячейкам (n_cells,).")
        if nu_and_dnu is None:
            if nu_of_B is not None:
                raise ValueError("method='newton' со сталью: нужна касательная nu_and_dnu(B) → (ν, dν/d|B|²) "
                                 "(например, model.problem._reluctivity_newton), а не только хорда nu_of_B.")
            def nu_and_dnu(B_cells, _v=nu_lin.copy()):
                """Без стали: ν постоянна (воздух, линейные материалы), касательная — она же."""
                return _v.copy(), np.zeros(nc, dtype=float)
        state = MagnetLawState(magnet, magnet_mask, nc, axis=magnet_axis, T_bin=T_bin)
        ddofs = space.boundary_dofs() if dirichlet_dofs is None else dirichlet_dofs
    elif magnet is not None:
        if magnet_mask is None:
            raise ValueError("magnet задан — нужен magnet_mask (n_cells,).")
        if nu_init is None:
            raise ValueError("magnet задан — нужен nu_init (относительная ν по ячейкам).")
        state = IrreversibleMagnetState(
            magnet, magnet_mask, nc, axis=magnet_axis,
            relaxation=demag_relaxation, switch_band=switch_band, T_bin=T_bin,
        )
        # ⚠ Базовая ν (воздух/сталь) обязана оставаться СКАЛЯРНОЙ: `nu_fn` строит из неё
        #   тензор. Замыкание держит СВОЮ копию — `nu0` ниже перевязывается на тензор, и
        #   захват по имени вернул бы тензор во вход `nu_field`.
        nu0 = np.asarray(nu_init, dtype=float)
        nu_base = ((lambda B, _v=nu0.copy(): _v.copy()) if nu_of_B is None else nu_of_B)

        def nu_fn(B_cells, _st=state, _base=nu_base, _aniso=bool(anisotropic_magnet)):
            """
            ν по ячейкам: воздух/сталь — от пользователя, магнит — своя.

            _aniso=True (по умолчанию): ТЕНЗОР — вдоль оси касательная ν ветви, поперёк
            постоянная ν_⊥ материала. _aniso=False: прежний СКАЛЯР (наклон ветви во все
            стороны) — оставлен только для абляции, физически он неверен поперёк оси.
            """
            nu = np.asarray(_base(B_cells), dtype=float).copy()
            if _aniso:
                return _st.nu_field(nu)
            nu[_st.idx] = _st.nu_rel
            return nu

        nu0 = nu0.copy()
        nu0[state.idx] = state.nu_rel
        if anisotropic_magnet:
            nu0 = state.nu_field(nu0)
        ddofs = space.boundary_dofs() if dirichlet_dofs is None else dirichlet_dofs

    # Предел модели магнита проверяется ТОЛЬКО по ячейкам магнита. Глобальный T_cap — это
    # отдельный порог для ОСТАЛЬНОЙ машины (класс изоляции обмотки), и по умолчанию его нет:
    # медь в машине штатно горячее, чем область валидности кривой магнита, и приравнивать
    # одно к другому значило бы останавливать расчёт по чужому критерию.
    T_cap_magnet = magnet.temperature_limit() if magnet is not None else float("inf")
    if T_cap is None:
        T_cap = float("inf")
    em_abort = 10.0 * float(em_tol) if em_abort_residual is None else float(em_abort_residual)

    stepper = ImplicitEulerThermalStepper(space, k_cells, capacity_cells, dt=dt, h=h,
                                          T_amb=T_amb, robin=robin)
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
    knee_frac: list[float] = []
    p_loss: list[float] = []
    energy: list[float] = []
    outflow: list[float] = []
    em_iters: list[int] = []
    em_res: list[float] = []
    em_subs: list[int] = []
    em_ok = True
    runaway = False
    cascade = False
    em_failed = False
    reason = "завершено"

    def record(t: float, T_field: np.ndarray, q: np.ndarray,
               n_it: int = 0, res: float = 0.0, n_sub: int = 0) -> None:
        times.append(t)
        T_hist.append(T_field.copy())
        T_max.append(float(T_field.max()))
        if state is not None and state.idx.size:
            a_m = areas[state.idx]
            T_mag_hist.append(float(state.T_mag.max()))
            ret_min.append(float(state.retention.min()))
            ret_mean.append(float(np.average(state.retention, weights=a_m)))
            knee_cnt.append(state.n_past_knee())
            knee_frac.append(float(a_m[state.margins() < 0.0].sum() / a_m.sum()))
        else:
            T_mag_hist.append(float("nan"))
            ret_min.append(1.0)
            ret_mean.append(1.0)
            knee_cnt.append(0)
            knee_frac.append(0.0)
        p_loss.append(float((q * areas).sum()))
        em_iters.append(int(n_it))
        em_res.append(float(res))
        em_subs.append(int(n_sub))
        energy.append(stepper.stored_energy(T_field))
        outflow.append(stepper.convective_outflow(T_field))

    # Стартовая температура магнита: за пределом модели — исключение сразу (расчёт не начинается).
    T_prev_mag = None
    if state is not None:
        T_prev_mag = _cell_temperature(space, T)[state.idx]
        state.set_temperature(T_prev_mag)

    def solve_magnetic(em_warm):
        """
        Магнитная задача (под)шага при текущих T и r состояния: (решение, сошлось, достигнутая невязка).
        Невязка у Ньютона — ‖R‖/R₀ (R₀ — невязка холодного старта, история начинается с неё), у Пикара —
        относительное изменение B на последней итерации.
        """
        if newton:
            law = state.law()
            em_n = solve_nonlinear_2d_newton(
                space, nu_and_dnu, nu_init=nu_lin, j_cells=MU0 * j_phys, magnet_law=law,
                dirichlet_dofs=ddofs, dirichlet_values=dirichlet_values,
                max_iter=em_max_iter, tol=em_tol, a_init=None if em_warm is None else em_warm.a,
            )
            state.accept(law)
            hist = em_n.rel_change_history
            return em_n, bool(em_n.converged), (hist[-1] / hist[0] if hist and hist[0] > 0.0 else 0.0)
        em_p = solve_nonlinear_2d_picard(
            space, nu_of_B=nu_fn, nu_init=nu0, j_cells=MU0 * j_phys,
            magnetization=state, dirichlet_dofs=ddofs, dirichlet_values=dirichlet_values,
            relaxation=em_relaxation, max_iter=em_max_iter, tol=em_tol, warm_start=em_warm,
        )
        return em_p, bool(em_p.converged), (em_p.rel_change_history[-1] if em_p.rel_change_history else 0.0)

    em_prev = None

    def advance_magnet(T_cells):
        """
        Магнит при поле температуры шага `T_cells` — продолжением по температуре от последнего ПРИНЯТОГО
        состояния. Возвращает (решение, итераций, невязка, подшагов, остановка): остановка — None, если решение
        принято (потеря зафиксирована в истории), иначе причина; состояние магнита тогда откачено к
        последнему принятому, а у вызывающего — признак ('overheat' | 'failed' | 'cascade').
        """
        nonlocal em_prev, T_prev_mag, em_ok
        T_target = T_cells[state.idx]
        T_hot = float(T_target.max()) if T_target.size else -np.inf
        if max(T_hot, float(state.law_temperature(T_hot))) >= T_cap_magnet:
            return None, 0, 0.0, 0, ("overheat",
                                     "магнит перегрет: T=%.1f C >= предел модели %.1f C" % (T_hot, T_cap_magnet)
                                     if T_hot >= T_cap_magnet else
                                     "магнит перегрет: T=%.2f C — узел сетки температур кривой (шаг %g C) уже за "
                                     "пределом модели %.2f C" % (T_hot, state.T_bin, T_cap_magnet))

        # ПРОДОЛЖЕНИЕ ПО ТЕМПЕРАТУРЕ: шаг повторяется всё мельче, пока не сойдётся.
        # Состояние с историей откатывается перед каждой попыткой — иначе повтор шёл бы
        # поверх уже зафиксированной (и, возможно, неверной) необратимой потери.
        snap = state.snapshot()
        em_start = em_prev
        attempt = 1
        while True:
            state.restore(snap)
            em = em_start
            ok = True                   # все подшаги решены до допуска
            failed = False              # подшаг не решён и с промахом больше порядка
            beyond = np.zeros(state.idx.size, dtype=bool)   # ниже H_cJ хоть на одном подшаге попытки
            n_it = 0
            res = 0.0
            final = attempt >= max_substeps
            for i in range(1, attempt + 1):
                cancel_check()
                frac = i / attempt
                state.set_temperature(T_prev_mag + frac * (T_target - T_prev_mag))
                em, conv, res_i = solve_magnetic(em)
                state.commit()      # необратимая потеря подшага уходит в историю
                beyond |= state.beyond_hcj
                n_it += em.n_iterations
                res = max(res, res_i)
                if not conv:
                    ok = False
                    failed = res_i >= em_abort
                    # Мельче ещё можно — повторить шаг. На последней попытке промах меньше порядка
                    # принимается, и путь досчитывается до T шага: оборвать его здесь значило бы
                    # принять шаг с магнитом при промежуточной температуре и недосчитанной потерей.
                    if not final or failed:
                        break
            n_sub = attempt
            if ok or final:
                break
            attempt *= 2
        em_ok = em_ok and ok

        # Шаг принимается, только если магнитная задача сошлась: до em_tol на всех подшагах
        # или с промахом меньше порядка (res < em_abort — виден в em_converged/em_residual).
        # Иначе решения на шаге нет, и вывода о магните НЕ делается (несходимость метода — не
        # физика: так было с Пикаром за коленом, Л-119; у Ньютона это страховка).
        # История обрывается ПОСЛЕДНИМ ПРИНЯТЫМ шагом, состояние магнита откатывается к нему же:
        # значения, зафиксированные внутри сорвавшегося шага, отражают незавершённую итерацию.
        if failed:
            state.restore(snap)
            return em, n_it, res, n_sub, ("failed",
                                          "магнитная задача не сошлась при T=%.1f C даже с %d подшагами по "
                                          "температуре (невязка %.1e при допуске %.0e) — расчёт остановлен, "
                                          "вывода о магните нет" % (T_hot, n_sub, res, em_tol))
        # Признак «ячейки ниже H_cJ» — только у принятых решений, но у КАЖДОГО подшага: подшаг — точка пути
        # нагружения, его потеря уже ушла в историю (`commit`). Если путь хоть на одном подшаге вышел за −H_cJ,
        # защёлкнутая там доля r = 0 взята из доопределения закона за пределом модели, а не из самой модели, —
        # даже если к концу шага поле вернулось выше −H_cJ. Поэтому — объединение по подшагам, а не последний.
        n_bad = int(np.count_nonzero(beyond))
        if n_bad > 0:
            state.restore(snap)
            return em, n_it, res, n_sub, ("cascade",
                                          "каскадное размагничивание магнита при T=%.1f C: %d ячеек ниже H_cJ "
                                          "(собственное поле превысило коэрцитивность)" % (T_hot, n_bad))
        em_prev = em
        T_prev_mag = T_target
        return em, n_it, res, n_sub, None

    def losses(T_cells, em):
        """Плотность потерь q(T) [Вт/м³] при поле температуры и принятом магнитном решении."""
        q = copper_loss_density(j_loss, T_cells)
        if extra_loss is not None:
            q = q + np.asarray(extra_loss(T_cells, em), dtype=float)
        return q

    # СХЕМА ШАГА (расщепление): T^n → магнит при T^n (с памятью) → потери q^n = q(T^n, магнит^n) →
    # неявный Эйлер по теплу с ЯВНЫМ источником q^n → T^{n+1}. ЗАПИСЬ n — состояние в момент t_n: поле T^n и
    # магнит, принятый при T^n (температура, доля r, колено, итерации/невязка/подшаги его решения), тепловой запас
    # и отвод при T^n. Столбец потерь — источник, с которым пришли в t_n: loss_power[n] = ∫q^{n−1} dV (баланс
    # шага: (U^n − U^{n−1})/dt = loss_power[n] − outflow[n] точно); в записи 0 — q^0, источник первого шага.
    # Прежде магнитные столбцы записи n брались с решения при T^{n−1} — со сдвигом на шаг относительно T^n.
    # Остановка по магниту при T^n (перегрет / не сошлась / каскад) запись n НЕ добавляет: история — только
    # согласованные состояния, поле остановки — в `T_stop`. Исключение — t_0: запись 0 есть всегда (магнит
    # тогда в исходном, нерешённом состоянии).
    stop = None
    T_stop = None
    T_cells = _cell_temperature(space, T)
    em, n_it, res, n_sub = None, 0, 0.0, 0
    if state is not None:
        em, n_it, res, n_sub, stop = advance_magnet(T_cells)
    if stop is None:
        q = losses(T_cells, em)
        record(0.0, T, q, n_it, res, n_sub)
    else:
        T_stop = T.copy()
        record(0.0, T, copper_loss_density(j_loss, T_cells))

    if stop is None:
        for n in range(1, int(n_steps) + 1):
            cancel_check()                               # отмена расчёта — между шагами по времени
            q_step = q                                   # источник шага — по состоянию в начале (явно)
            T = stepper.step(T, q_step)
            T_cells = _cell_temperature(space, T)
            T_peak = float(T.max())
            em, n_it, res, n_sub = None, 0, 0.0, 0
            if state is not None:
                if not np.isfinite(T_peak):
                    runaway = True
                    reason = "тепловой разгон: температура разошлась"
                    T_stop = T.copy()
                    break
                em, n_it, res, n_sub, stop = advance_magnet(T_cells)
                if stop is not None:
                    T_stop = T.copy()
                    break
            record(n * float(dt), T, q_step, n_it, res, n_sub)

            if not np.isfinite(T_peak) or T_peak >= T_cap:
                runaway = True
                reason = ("тепловой разгон: T=%.1f C достигла порога %.1f C" % (T_peak, T_cap)
                          if np.isfinite(T_peak) else "тепловой разгон: температура разошлась")
                break

            # ВЫХОД НА УСТАНОВИВШИЙСЯ РЕЖИМ: скорость роста упала ниже допуска (не авария).
            if (steady_tol is not None and len(T_max) >= 3
                    and abs(T_max[-1] - T_max[-2]) < steady_tol * float(dt)):
                reason = ("установившийся режим: |dT/dt|=%.3g C/с < допуска %.3g C/с"
                          % (abs(T_max[-1] - T_max[-2]) / float(dt), steady_tol))
                break

            q = losses(T_cells, em)                      # источник следующего шага

    if stop is not None:
        kind, reason = stop
        if kind == "overheat":
            runaway = True
        elif kind == "failed":
            em_failed = True
        else:
            cascade = True

    return CoupledTransientResult(
        times=np.asarray(times), T_hist=np.asarray(T_hist), T_max=np.asarray(T_max),
        T_magnet=np.asarray(T_mag_hist), retention_min=np.asarray(ret_min),
        retention_mean=np.asarray(ret_mean), n_past_knee=np.asarray(knee_cnt),
        past_knee_fraction=np.asarray(knee_frac),
        loss_power=np.asarray(p_loss), stored_energy=np.asarray(energy),
        outflow=np.asarray(outflow), runaway=runaway, magnet_cascade=cascade,
        stop_reason=reason,
        em_converged=em_ok, em_iterations=np.asarray(em_iters),
        em_residual=np.asarray(em_res), em_substeps=np.asarray(em_subs), state=state,
        em_failed=em_failed, em_method=method, em_tol=float(em_tol), T_stop=T_stop,
    )
