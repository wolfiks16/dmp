"""
БИБЛИОТЕКА ТИПОВ МАШИН — расчёт под ТИП двигателя, а не под конкретный экземпляр.

Принцип (см. `docs/solver_spec.md §1а`): физическое ядро (материалы, поле, демаг, тепло,
потери) НЕ зависит от типа машины и пишется один раз. От типа зависят ровно ЧЕТЫРЕ вещи,
и только они описываются здесь:

  1) как уложены слои сечения      -> геометрия (общий параметрический генератор)
  2) что вращается и где магниты   -> `Topology` (влияет на систему отсчёта потерь)
  3) как включена обмотка          -> `Winding`: ток в пазах
  4) как считается сила машины     -> `Winding`: K_e (разные формулы, ОДНО поле)

Доказательство общности: тот же генератор геометрии строит и outrunner PMSM (бесщёточный,
магниты на роторе), и ДП25 (щёточный ДПТ, магниты на статоре) — топологически это одно
сечение «магниты снаружи, зубчатый сердечник внутри».

Обозначения — `docs/glossary.md`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve
from magcore.fem2d.losses import copper_resistivity
from magcore.fem2d.machines.characteristics import phase_resistance
from magcore.fem2d.machines.excitation import slot_areas, winding_current_density
from magcore.fem2d.machines.loss import phase_flux_linkage
from magcore.fem2d.machines.postproc import back_emf_constant as _three_phase_ke
from magcore.fem2d.machines.pmsm_outrunner import (
    MachineGeometry,
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)
from magcore.fem2d.machines.winding import WindingLayout, star_of_slots_layout
from magcore.fem2d.model.postproc import torque_arkkio
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.spaces import LagrangeP1Space2D

# ============================================================ 1. ТОПОЛОГИЯ

MAGNETS_ON_ROTOR = "rotor"      # PMSM: магниты вращаются
MAGNETS_ON_STATOR = "stator"    # щёточный ДПТ с ПМ: магниты неподвижны, вращается якорь


@dataclass(frozen=True, slots=True)
class Topology:
    """
    Что вращается и где магниты. Определяет СИСТЕМУ ОТСЧЁТА для вихревых потерь —
    место, где легко ошибиться: потери считаются в системе той части, где лежит материал.

    * магниты на РОТОРЕ (PMSM): вращаются, видят зубцовую пульсацию от статора
      ⇒ поле снимать в СО-ВРАЩАЮЩИХСЯ точках (система ротора);
    * магниты на СТАТОРЕ (щёточный ДПТ): неподвижны, зубцовую пульсацию создаёт
      вращающийся якорь ⇒ поле снимать в НЕПОДВИЖНЫХ точках (система статора).
    Перепутать = получить фантомные потери от постоянного поля магнита.
    """

    magnets_on: str = MAGNETS_ON_ROTOR
    outer_part_rotates: bool = True          # outrunner (или неподвижный корпус у ДПТ)

    def __post_init__(self) -> None:
        if self.magnets_on not in (MAGNETS_ON_ROTOR, MAGNETS_ON_STATOR):
            raise ValueError("magnets_on: 'rotor' | 'stator'.")

    @property
    def magnet_loss_frame(self) -> str:
        """Система отсчёта для вихревых потерь магнита: 'rotor' | 'stator'."""
        return "rotor" if self.magnets_on == MAGNETS_ON_ROTOR else "stator"


# ============================================================ 2. МАТЕРИАЛЫ ПО РЕГИОНАМ

@dataclass(frozen=True, slots=True)
class MachineMaterials:
    """
    Материалы по регионам. Сталь ЯКОРЯ (там, где обмотка) и сталь ЯРМА ВОЗВРАТА
    (корпус/ярмо ротора) задаются раздельно: у реальных изделий они разные
    (ДП25: якорь — электротехническая шихтованная, корпус — Сталь 10 массивная).
    """

    magnet: AnisotropicBHTMagnet
    steel_armature: SteelBHCurve
    steel_yoke: SteelBHCurve | None = None      # None ⇒ такая же, как у якоря

    # --- параметры ПОТЕРЬ (нужны уровню Р3; без них считается только медь) ---
    steinmetz_armature: object | None = None    # SteinmetzCoefficients стали якоря
    steinmetz_yoke: object | None = None        # ... ярма возврата (None ⇒ как у якоря)
    yoke_solid: bool = False                    # ярмо МАССИВНОЕ (не шихтованное)
    yoke_resistivity: float | None = None       # Ом·м — для массивного ярма
    yoke_mu_r: float = 500.0                    # μ_r ярма для скин-глубины
    magnet_sigma_20: float | None = None        # См/м — электропроводность магнита при 20 °C
    magnet_segments: int = 1                    # сегментация магнита (потери ∝ 1/N²)
    pwm_factor: float = 1.0                     # ШИМ удваивает вихревые магнита (≈2.0)

    @property
    def yoke_curve(self) -> SteelBHCurve:
        return self.steel_armature if self.steel_yoke is None else self.steel_yoke

    @property
    def yoke_steinmetz(self):
        return self.steinmetz_armature if self.steinmetz_yoke is None else self.steinmetz_yoke

    def has_loss_data(self) -> bool:
        """Хватает ли данных для расчёта потерь ядра (сталь + магнит)."""
        return self.steinmetz_armature is not None and self.magnet_sigma_20 is not None


# ============================================================ 3. ОБМОТКА (подключаемая)

@dataclass(frozen=True, slots=True)
class ThreePhaseWinding:
    """Трёхфазная распределённая обмотка (PMSM). K_e = p·λ_m, K_t = 1.5·p·λ_m."""

    turns_per_slot: float
    slot_fill: float = 0.45
    kind: str = field(default="three_phase", init=False)

    def layout(self, params: OutrunnerPMSMParams) -> WindingLayout:
        return star_of_slots_layout(params.n_slots, params.n_poles)

    def current_density(self, geo: MachineGeometry, *, i_peak: float, gamma_elec: float):
        """Плотность тока по ячейкам [А/м²] при мгновенном токе и угле вектора γ."""
        if i_peak == 0.0 or self.turns_per_slot == 0.0:
            return None
        return winding_current_density(geo, self.layout(geo.params), i_peak=i_peak,
                                       gamma_elec=gamma_elec,
                                       turns_per_slot=self.turns_per_slot)

    def emf_constant(self, geo: MachineGeometry, a_nodes: np.ndarray) -> float:
        """K_e [В·с/рад] из потокосцепления фаз холостого хода."""
        lam = phase_flux_linkage(geo, self.layout(geo.params), a_nodes,
                                 turns_per_slot=self.turns_per_slot)
        return float(_three_phase_ke(geo, lam))

    def torque_constant(self, geo: MachineGeometry, a_nodes: np.ndarray) -> float:
        """K_t = 1.5·K_e для трёхфазной машины (момент на АМПЛИТУДУ тока по оси q)."""
        return 1.5 * self.emf_constant(geo, a_nodes)

    def resistance(self, geo: MachineGeometry, T: float) -> float:
        """Сопротивление ФАЗЫ [Ом] при температуре T (растёт по ρ_cu(T))."""
        return float(phase_resistance(geo, turns_per_slot=self.turns_per_slot,
                                      slot_fill=self.slot_fill, T=T))

    def copper_loss(self, geo: MachineGeometry, *, i_peak: float, T: float) -> float:
        """Потери в меди [Вт]: P = 3·I_скз²·R_фазы, I_скз = i_peak/√2 (симметричная система)."""
        i_rms = float(i_peak) / math.sqrt(2.0)
        return 3.0 * i_rms ** 2 * self.resistance(geo, T)

    def loss_current_density(self, geo: MachineGeometry, *, i_peak: float) -> np.ndarray:
        """
        Плотность тока для НАГРЕВА [А/м²] — СРЕДНЕКВАДРАТИЧНАЯ за электрический период
        (тепловая постоянная времени ≫ периода) с поправкой на заполнение паза:
        ток размазан по всей площади паза, а греется только медь ⇒ J/√k_зап.
        ⚠ Это ДРУГАЯ величина, чем ток для магнитной задачи (мгновенный, худший угол).
        """
        return _rms_slot_current(geo, i_peak=i_peak, turns_per_slot=self.turns_per_slot,
                                 slot_fill=self.slot_fill)


@dataclass(frozen=True, slots=True)
class CommutatorWinding:
    """
    Коллекторная обмотка (щёточный ДПТ): простая волновая или петлевая.

    K_e = p·Z·Φ/(2π·a)·k_скоса, где Z — полное число проводников, a — число пар
    параллельных ветвей (простая волновая: a=1; простая петлевая: a=p), Φ — поток на полюс.
    K_t = K_e численно (в СИ).
    """

    conductors_total: int                  # Z (напр. 13 пазов × 60 = 780)
    parallel_path_pairs: int = 1           # a: волновая=1, петлевая=p
    slot_fill: float = 0.45
    skew_deg: float = 0.0                  # скос пакета (механические градусы)
    wire_diameter: float | None = None     # м — для расчёта сопротивления
    mean_turn_length: float | None = None  # м — средняя длина витка
    kind: str = field(default="commutator", init=False)

    def __post_init__(self) -> None:
        if self.conductors_total <= 0 or self.parallel_path_pairs <= 0:
            raise ValueError("conductors_total и parallel_path_pairs должны быть > 0.")

    def skew_factor(self, params: OutrunnerPMSMParams) -> float:
        """k = sin(γ_эл/2)/(γ_эл/2), γ_эл = p·γ_мех — снижение ЭДС из-за скоса пакета."""
        g = math.radians(self.skew_deg) * (params.n_poles // 2)
        return 1.0 if g == 0.0 else math.sin(g / 2.0) / (g / 2.0)

    def current_density(self, geo: MachineGeometry, *, i_peak: float, gamma_elec: float = 0.0):
        """
        Плотность тока якоря [А/м²]: коллектор делит проводники щёточной осью, ток
        меняет знак каждый полюсный шаг. МДС якоря стоит по поперечной оси (реакция якоря).
        `gamma_elec` — сдвиг щёточной оси (обычно 0 = нейтраль).
        """
        if i_peak == 0.0:
            return None
        p = geo.params.n_poles // 2
        nc = geo.mesh.n_cells
        sid = geo.slot_id
        areas = np.asarray(slot_areas(geo), dtype=float)
        i_cond = float(i_peak) / (2.0 * self.parallel_path_pairs)      # ток одного проводника
        z_slot = self.conductors_total / geo.params.n_slots
        j = np.zeros(nc, dtype=float)
        in_slot = sid >= 0
        idx = np.where(in_slot)[0]
        cent = np.array([geo.mesh.cell_centroid(int(c)) for c in idx], dtype=float)
        th = np.arctan2(cent[:, 1], cent[:, 0])
        sign = np.sign(np.sin(p * (th - float(gamma_elec))))
        sign[sign == 0.0] = 1.0
        j[idx] = sign * z_slot * i_cond / areas[sid[idx]]
        return j

    def flux_per_pole(self, geo: MachineGeometry, a_nodes: np.ndarray, *, n_probe: int = 721) -> float:
        """
        Поток на полюс [Вб] = L·(A_z^max − A_z^min) по окружности в зазоре.
        В 2D это точное определение потока между соседними межполюсными осями.
        """
        p = geo.params
        r = 0.5 * (p.R_s_out + p.R_mag_in)
        th = np.linspace(0.0, 2 * math.pi, int(n_probe), endpoint=False)
        mesh = geo.mesh
        cent = np.array([mesh.cell_centroid(c) for c in range(mesh.n_cells)], dtype=float)
        vals = np.empty(th.size, dtype=float)
        for i, t in enumerate(th):
            pt = np.array([r * math.cos(t), r * math.sin(t)])
            c = int(np.argmin(((cent - pt) ** 2).sum(axis=1)))
            vals[i] = float(np.mean(a_nodes[list(mesh.cell_vertex_indices(c))]))
        return float(vals.max() - vals.min()) * p.axial_length

    def emf_constant(self, geo: MachineGeometry, a_nodes: np.ndarray) -> float:
        """K_e [В·с/рад] = p·Z·Φ/(2π·a)·k_скоса."""
        p = geo.params.n_poles // 2
        phi = self.flux_per_pole(geo, a_nodes)
        return (p * self.conductors_total * phi
                / (2.0 * math.pi * self.parallel_path_pairs) * self.skew_factor(geo.params))

    def torque_constant(self, geo: MachineGeometry, a_nodes: np.ndarray) -> float:
        """K_t = K_e численно (в СИ) для машины постоянного тока."""
        return self.emf_constant(geo, a_nodes)

    def resistance(self, geo: MachineGeometry, T: float) -> float:
        """
        Сопротивление обмотки якоря между щётками [Ом].

        Полное последовательное R_посл = ρ(T)·L_провода/S; при 2a параллельных ветвях
        сопротивление якоря = R_посл/(2a)² (каждая ветвь несёт половину проводников,
        ветви параллельны). Сверка на ДП25: 3.77 Ом против паспортных 3.9±0.4 (−3.5 %).
        """
        if self.wire_diameter is None or self.mean_turn_length is None:
            raise ValueError("для расчёта сопротивления задайте wire_diameter и mean_turn_length.")
        turns_total = self.conductors_total / 2.0                 # 2 проводника на виток
        area = math.pi / 4.0 * float(self.wire_diameter) ** 2
        length = turns_total * float(self.mean_turn_length)
        r_series = float(copper_resistivity(T)) * length / area
        return r_series / (2.0 * self.parallel_path_pairs) ** 2

    def copper_loss(self, geo: MachineGeometry, *, i_peak: float, T: float) -> float:
        """Потери в меди [Вт]: P = I²·R_якоря (постоянный ток, без множителя 3)."""
        return float(i_peak) ** 2 * self.resistance(geo, T)

    def loss_current_density(self, geo: MachineGeometry, *, i_peak: float) -> np.ndarray:
        """
        Плотность тока для НАГРЕВА [А/м²]. У коллекторной машины ток проводника ПОСТОЯНЕН
        (I/2a), поэтому СКЗ = самому значению — множителя 1/√2 нет (в отличие от трёхфазной).
        Поправка на заполнение паза та же: J/√k_зап.
        """
        z_slot = self.conductors_total / geo.params.n_slots
        i_cond = float(i_peak) / (2.0 * self.parallel_path_pairs)
        areas = np.asarray(slot_areas(geo), dtype=float)
        sid = geo.slot_id
        j = np.zeros(geo.mesh.n_cells, dtype=float)
        inside = sid >= 0
        j[inside] = z_slot * i_cond / areas[sid[inside]] / math.sqrt(self.slot_fill)
        return j


Winding = ThreePhaseWinding | CommutatorWinding


# ============================================================ 4. ОПРЕДЕЛЕНИЕ МАШИНЫ

@dataclass(frozen=True, slots=True)
class MachineDefinition:
    """Полное описание машины: тип + геометрия + материалы + обмотка."""

    name: str
    params: OutrunnerPMSMParams
    materials: MachineMaterials
    winding: Winding
    topology: Topology = field(default_factory=Topology)

    def build_geometry(self) -> MachineGeometry:
        return build_outrunner_spm_pmsm(self.params)

    def reluctivity(self, geo: MachineGeometry):
        """
        ν по ячейкам с РАЗНЫМИ сталями: воздух/паз = 1, магнит = 1/μ_rec,
        якорь (ярмо+зубцы) — своя кривая, ярмо возврата (корпус) — своя.
        Возвращает (nu_of_B, nu_init, magnet_mask).
        """
        region = geo.region
        nc = geo.mesh.n_cells
        magnet_mask = region == int(Region.MAGNET)
        arm = np.where(np.isin(region, (int(Region.STATOR_YOKE), int(Region.TOOTH))))[0]
        yok = np.where(region == int(Region.ROTOR_YOKE))[0]
        c_arm, c_yok = self.materials.steel_armature, self.materials.yoke_curve
        nu_mag = 1.0 / self.materials.magnet.mu_rec

        def nu_of_B(B_cells: np.ndarray) -> np.ndarray:
            nu = np.ones(nc, dtype=float)
            nu[magnet_mask] = nu_mag
            for idx, curve in ((arm, c_arm), (yok, c_yok)):
                if idx.size:
                    b = np.hypot(B_cells[idx, 0], B_cells[idx, 1])
                    nu[idx] = MU0 * np.array([curve.nu_chord(float(x)) for x in b])
            return nu

        return nu_of_B, nu_of_B(np.zeros((nc, 2), dtype=float)), magnet_mask


@dataclass(frozen=True, slots=True)
class MachineSolution:
    """Результат магнитостатического расчёта + машинные величины."""

    geometry: MachineGeometry
    field: object                       # Fem2DPicardResult
    K_e: float                          # В·с/рад — коэффициент противо-ЭДС
    K_t: float                          # Н·м/А — коэффициент момента
    converged: bool
    n_iterations: int
    residual: float
    winding_kind: str = "three_phase"   # определяет ПЕРЕВОД K_e → kV (см. ниже)

    def kv(self, convention: str = "bus_sixstep") -> float:
        """
        Эквивалентное паспортное kV [об/(мин·В)] этой обмотки.

        ⚠ Для ТРЁХФАЗНОЙ машины `K_e` здесь — амплитуда ФАЗНОЙ ЭДС (p·λ_m), а паспортное
        kV производители относят к другому напряжению (обычно к шине), поэтому нужен
        множитель конвенции. Прежняя формула 60/(2π·K_e) применялась ко всем машинам и
        для трёхфазной завышала kV в π/2 раза (аудит 2026-09-03).
        Для КОЛЛЕКТОРНОЙ машины та же формула строга — там конвенция одна.
        """
        from magcore.fem2d.machines.conventions import kv_from_ke

        if self.K_e <= 0:
            return float("inf")
        if self.winding_kind == "three_phase":
            return kv_from_ke(self.K_e, convention)
        return 60.0 / (2.0 * math.pi * self.K_e)

    @property
    def kV(self) -> float:
        """kV в конвенции по умолчанию («от шины», шеститактный регулятор)."""
        return self.kv()


def solve_machine(
    definition: MachineDefinition, *, T: float = 20.0, i_peak: float = 0.0,
    gamma_elec: float = 0.0, retention=None,
    relaxation: float = 0.05, max_iter: int = 600, tol: float = 1.0e-6,
) -> MachineSolution:
    """
    Уровень Р1: магнитостатика + K_e/K_t. Работает для ЛЮБОГО типа из библиотеки —
    формула K_e выбирается обмоткой, поле считается одним и тем же ядром.

    `retention` — доля сохранённой ремнантности по ячейкам магнита (None = магнит цел):
    позволяет снять характеристики ПОСЛЕ размагничивания.
    ⚠ `relaxation=0.05` по умолчанию: при насыщенном железе большие значения не сходятся
    (проверено на ДП25; см. `solver_spec.md §5`).
    """
    geo = definition.build_geometry()
    space = LagrangeP1Space2D(geo.mesh)
    nu_of_B, nu_init, mask = definition.reluctivity(geo)
    idx = np.where(mask)[0]

    magnet = definition.materials.magnet
    br = np.full(idx.size, float(magnet.Br(T)), dtype=float)
    if retention is not None:
        r = np.asarray(retention, dtype=float).reshape(-1)
        if r.shape != idx.shape:
            raise ValueError("retention должен быть по ячейкам магнита.")
        br = br * r
    nu_br = np.zeros((geo.mesh.n_cells, 2), dtype=float)
    nu_br[idx] = (br / magnet.mu_rec)[:, None] * geo.magnet_easy_axis[idx]

    jz = definition.winding.current_density(geo, i_peak=i_peak, gamma_elec=gamma_elec)
    em = solve_nonlinear_2d_picard(
        space, nu_of_B=nu_of_B, nu_init=nu_init,
        j_cells=(None if jz is None else MU0 * jz), magnetization=nu_br,
        relaxation=relaxation, max_iter=max_iter, tol=tol,
    )

    # K_e снимается на ХОЛОСТОМ ХОДУ (без тока): это свойство магнитной системы.
    em_nl = em if jz is None else solve_nonlinear_2d_picard(
        space, nu_of_B=nu_of_B, nu_init=nu_init, magnetization=nu_br,
        relaxation=relaxation, max_iter=max_iter, tol=tol,
    )
    ke = definition.winding.emf_constant(geo, em_nl.a)
    kt = definition.winding.torque_constant(geo, em_nl.a)
    return MachineSolution(
        geometry=geo, field=em, K_e=ke, K_t=kt,
        converged=bool(em.converged and em_nl.converged),
        n_iterations=int(em.n_iterations),
        residual=float(em.rel_change_history[-1] if em.rel_change_history else 0.0),
        winding_kind=str(getattr(definition.winding, "kind", "three_phase")),
    )


# ============================================================ 5. УРОВЕНЬ Р2: характеристики

class _FieldCarrier:
    """
    Минимальный носитель того, что нужно `torque_arkkio`: сетка и B по ячейкам.
    Намеренно НЕ используем полноценную `Problem2D` — она несёт допущение «одна сталь
    на всё железо», а библиотека поддерживает разные стали по регионам.
    """

    class _P:
        def __init__(self, mesh):
            self.mesh = mesh

    class _F:
        def __init__(self, B):
            self.B_cells = B

    def __init__(self, mesh, B_cells):
        self.problem = _FieldCarrier._P(mesh)
        self.field = _FieldCarrier._F(B_cells)


@dataclass(frozen=True, slots=True)
class MachinePerformance:
    """Уровень Р2: что машина реально выдаёт в заданном режиме."""

    machine: str
    torque: float               # Н·м — электромагнитный момент (Арккио по зазору)
    K_e: float                  # В·с/рад
    K_t: float                  # Н·м/А
    speed_rpm: float
    i_peak: float
    T: float                    # °C, температура магнита/материалов расчёта
    P_out: float                # Вт — механическая мощность на валу (τ·ω)
    P_copper: float             # Вт — потери в меди
    P_core: float               # Вт — потери в стали + вихревые в магните (0, если не считались)
    R_winding: float            # Ом
    converged: bool

    @property
    def P_loss(self) -> float:
        return self.P_copper + self.P_core

    @property
    def efficiency(self) -> float:
        """КПД = P_вых/(P_вых + потери). 0, если мощность не положительна."""
        d = self.P_out + self.P_loss
        return float(self.P_out / d) if d > 0.0 and self.P_out > 0.0 else 0.0

    @property
    def thrust_relative(self) -> float:
        """
        Относительная тяга винта. При ТОКОВОМ ограничении момент задан (τ=K_t·I), винт
        находит равновесие τ=k·ω² ⇒ ω²∝τ, а тяга фикс. винта ∝ ω² ⇒ **тяга ∝ моменту**.
        Возвращается сам момент — сравнивать между вариантами как относительную тягу.
        """
        return self.torque

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * self.speed_rpm / 60.0


def evaluate_performance(
    definition: MachineDefinition, *, i_peak: float, speed_rpm: float,
    T: float = 20.0, gamma_elec: float | None = None, retention=None,
    core_losses_w: float = 0.0, relaxation: float = 0.05, max_iter: int = 600,
    tol: float = 1.0e-6,
) -> MachinePerformance:
    """
    Уровень Р2 для ЛЮБОГО типа из библиотеки: момент, потери, КПД, тяга.

    `gamma_elec` — угол вектора тока: по умолчанию π/2 (ось q, максимум момента) для
    трёхфазной и 0 (нейтраль) для коллекторной. Для расчёта РАЗМАГНИЧИВАНИЯ берут γ=0
    (ось d, worst-case) — тогда момент близок к нулю, и это нормально.
    `core_losses_w` — потери в стали и магните [Вт], если считались отдельно (уровень Р3);
    при 0 КПД учитывает только медь и является ВЕРХНЕЙ оценкой.
    """
    if gamma_elec is None:
        gamma_elec = math.pi / 2.0 if definition.winding.kind == "three_phase" else 0.0

    sol = solve_machine(definition, T=T, i_peak=i_peak, gamma_elec=gamma_elec,
                        retention=retention, relaxation=relaxation, max_iter=max_iter, tol=tol)
    geo = sol.geometry
    p = geo.params
    torque = abs(float(torque_arkkio(
        _FieldCarrier(geo.mesh, sol.field.B_cells),
        p.R_s_out, p.R_mag_in, axial_length=p.axial_length)))

    p_cu = definition.winding.copper_loss(geo, i_peak=i_peak, T=T)
    omega = 2.0 * math.pi * float(speed_rpm) / 60.0
    return MachinePerformance(
        machine=definition.name, torque=torque, K_e=sol.K_e, K_t=sol.K_t,
        speed_rpm=float(speed_rpm), i_peak=float(i_peak), T=float(T),
        P_out=torque * omega, P_copper=float(p_cu), P_core=float(core_losses_w),
        R_winding=definition.winding.resistance(geo, T), converged=sol.converged,
    )


def voltage_limited_point(
    definition: MachineDefinition, *, voltage: float, speed_rpm: float, T: float = 20.0,
    K_e: float | None = None, R: float | None = None, core_losses_w: float = 0.0,
) -> dict:
    """
    Рабочая точка при ОГРАНИЧЕНИИ ПО НАПРЯЖЕНИЮ (реальный привод): противо-ЭДС E=K_e·ω,
    ток I=(U−E)/R (не отрицательный), момент = K_t·I.

    Нагрев бьёт дважды: R(T)↑ (горячая медь) и K_e↓ (нагрев магнита + необратимый демаг).
    `K_e`/`R` можно передать готовыми (из `evaluate_performance`), иначе считаются заново.
    """
    geo = definition.build_geometry()
    if K_e is None:
        K_e = solve_machine(definition, T=T).K_e
    if R is None:
        R = definition.winding.resistance(geo, T)
    kt_over_ke = 1.5 if definition.winding.kind == "three_phase" else 1.0
    omega = 2.0 * math.pi * float(speed_rpm) / 60.0
    E = float(K_e) * omega
    I = max(float(voltage) - E, 0.0) / float(R)
    torque = kt_over_ke * float(K_e) * I
    p_mech = torque * omega
    p_cu = (3.0 * (I / math.sqrt(2.0)) ** 2 * R if definition.winding.kind == "three_phase"
            else I ** 2 * R)
    total = p_mech + p_cu + float(core_losses_w)
    return dict(current=I, back_emf=E, torque=torque, p_mech=p_mech, p_copper=p_cu,
                efficiency=(p_mech / total if total > 0.0 else 0.0), omega=omega)


def _rms_slot_current(geo: MachineGeometry, *, i_peak: float, turns_per_slot: float,
                      slot_fill: float) -> np.ndarray:
    """СКЗ-ток паза для трёхфазной обмотки: J = N·I_скз/A_паз/√k_зап (не зависит от угла γ)."""
    if not (0.0 < slot_fill <= 1.0):
        raise ValueError("slot_fill in (0, 1].")
    areas = np.asarray(slot_areas(geo), dtype=float)
    sid = geo.slot_id
    j = np.zeros(geo.mesh.n_cells, dtype=float)
    inside = sid >= 0
    i_rms = float(i_peak) / math.sqrt(2.0)
    j[inside] = float(turns_per_slot) * i_rms / areas[sid[inside]] / math.sqrt(float(slot_fill))
    return j


# ============================================================ 6. УРОВЕНЬ Р3: связка (ядро К6′)

@dataclass(frozen=True, slots=True)
class ThermalDemagResult:
    """Итог связанного магнитотеплового расчёта с необратимым размагничиванием."""

    machine: str
    transient: object                # CoupledTransientResult — полная история
    retention: np.ndarray            # доля сохранённой ремнантности по ячейкам магнита
    fundamental_ratio: float         # инвариантная мера сохранности (1.0 = магнит цел)
    T_magnet_max: float              # °C
    T_max: float                     # °C — самая горячая точка (обычно обмотка)

    @property
    def torque_constant_drop(self) -> float:
        """Относительное падение K_e/K_t (0 = магнит цел). «Мотор стал слабее на …»."""
        return float(1.0 - self.fundamental_ratio)

    @property
    def survived(self) -> bool:
        """Расчёт дошёл до конца горизонта (не сорвался в разгон/каскад)."""
        return not (self.transient.runaway or self.transient.magnet_cascade)

    @property
    def stop_reason(self) -> str:
        return str(self.transient.stop_reason)


def run_thermal_demag(
    definition: MachineDefinition, *, i_peak: float, h_out: float, T_amb: float,
    dt: float, n_steps: int, thermal=None, gamma_elec: float = 0.0,
    h_in: float | None = None, T_frame: float | None = None, T0: float | None = None,
    T_cap: float | None = None, steady_tol: float | None = None,
    core_loss_density=None, auto_core_losses: bool = False, speed_rpm: float | None = None,
    em_relaxation: float = 0.05, em_max_iter: int = 400,
    em_tol: float = 1.0e-6, max_substeps: int = 8, **transient_kwargs,
) -> ThermalDemagResult:
    """
    Уровень Р3 (ЯДРО К6′) для ЛЮБОГО типа из библиотеки: связанный магнитотепловой расчёт
    с необратимым размагничиванием в петле, разгоном и каскадом.

    Тип машины подхватывается автоматически: распределение тока и ток для нагрева берутся
    у обмотки, ν — с РАЗНЫМИ сталями по регионам.

    `gamma_elec=0` — worst-case ось d (чисто размагничивающий ток), как принято для
    демаг-анализа. `core_loss_density` — карта потерь ядра [Вт/м³] (сталь + вихревые
    магнита), если считалась отдельно; без неё греет только медь (ОПТИМИСТИЧНО).
    `h_in`/`T_frame` — сток тепла статора в раму (кондукция), см. `solver_spec.md`.
    ⚠ dt ≤ 0.5 с (проверено: грубый шаг искажает траекторию).
    """
    from magcore.fem2d.coupled_transient import solve_coupled_magneto_thermal_transient
    from magcore.fem2d.machines.thermal_scenario import (
        MachineThermalProperties,
        magnet_fundamental_ratio,
    )
    from magcore.fem2d.thermal import assemble_robin_multi, classify_boundary_edges

    geo = definition.build_geometry()
    space = LagrangeP1Space2D(geo.mesh)
    props = thermal or MachineThermalProperties.representative()
    k_cells, c_cells = props.cell_fields(geo)
    nu_of_B, nu_init, magnet_mask = definition.reluctivity(geo)

    j_mag = definition.winding.current_density(geo, i_peak=i_peak, gamma_elec=gamma_elec)
    j_loss = definition.winding.loss_current_density(geo, i_peak=i_peak)

    robin = None
    if h_in is not None:                       # дифференцированные ГУ: статор → рама
        p = geo.params
        inner, outer = classify_boundary_edges(space, p.R_bore, p.R_out)
        robin = assemble_robin_multi(
            space, [(inner, h_in, T_amb if T_frame is None else T_frame), (outer, h_out, T_amb)])

    if auto_core_losses:
        if core_loss_density is not None:
            raise ValueError("укажите либо auto_core_losses, либо готовую core_loss_density.")
        if speed_rpm is None:
            raise ValueError("auto_core_losses=True требует speed_rpm (задаёт частоту потерь).")
        core_loss_density = compute_core_losses(
            definition, speed_rpm=speed_rpm, i_peak=i_peak, T=T_amb,
            gamma_elec=gamma_elec).density
    if core_loss_density is not None:
        q_core = np.asarray(core_loss_density, dtype=float)
        transient_kwargs["extra_loss"] = lambda Tc, em, _q=q_core: _q

    T_start = None if T0 is None else np.full(space.ndofs, float(T0), dtype=float)
    tr = solve_coupled_magneto_thermal_transient(
        space, k_cells=k_cells, capacity_cells=c_cells, h=h_out, T_amb=T_amb, robin=robin,
        dt=dt, n_steps=n_steps, j_cells=j_mag, j_loss_cells=j_loss,
        magnet=definition.materials.magnet, magnet_mask=magnet_mask,
        magnet_axis=geo.magnet_easy_axis, nu_of_B=nu_of_B, nu_init=nu_init, T0=T_start,
        T_cap=T_cap, steady_tol=steady_tol, em_relaxation=em_relaxation,
        em_max_iter=em_max_iter, em_tol=em_tol, max_substeps=max_substeps, **transient_kwargs,
    )

    retention = (tr.state.retention.copy() if tr.state is not None
                 else np.ones(int(np.count_nonzero(magnet_mask)), dtype=float))
    return ThermalDemagResult(
        machine=definition.name, transient=tr, retention=retention,
        fundamental_ratio=magnet_fundamental_ratio(geo, retention, magnet_mask),
        T_magnet_max=float(np.nanmax(tr.T_magnet)), T_max=float(np.max(tr.T_max)),
    )


@dataclass(frozen=True, slots=True)
class CoreLosses:
    """Потери ядра: карта по ячейкам [Вт/м³] + разбивка по составляющим [Вт]."""

    density: np.ndarray          # (n_cells,) — источник тепла для связки
    stator_iron_w: float         # сталь якоря/статора
    rotor_side_w: float          # магнит (вихревые) + ярмо возврата
    freq_elec: float             # Гц — электрическая частота
    freq_ripple: float           # Гц — частота пульсации в системе магнита

    @property
    def total_w(self) -> float:
        return self.stator_iron_w + self.rotor_side_w


def compute_core_losses(
    definition: MachineDefinition, *, speed_rpm: float, i_peak: float = 0.0,
    T: float = 20.0, gamma_elec: float = 0.0, n_positions: int = 12,
    tooth_pitch_span: bool = True, relaxation: float = 0.05, max_iter: int = 300,
) -> CoreLosses:
    """
    Потери ядра ТРЁХФАЗНОЙ машины (обе топологии — магниты на роторе/статоре):
    сталь статора/якоря + вихревые магнита + ярмо возврата.

    ⚠ КОЛЛЕКТОРНАЯ машина (щёточный ДПТ) НЕ поддержана: расчёт опирается на трёхфазную
    раскладку тока (реакция якоря), а у коллектора реакция якоря устроена принципиально
    иначе (ток постоянен в пространстве, коммутируется щётками) — подставлять ей трёхфазный
    ток значило бы считать НЕВЕРНОЕ число. Для такой машины доступна магнитостатика (K_e).

    СИСТЕМА ОТСЧЁТА — корректна для обеих топологий по построению, и это НЕ случайность:
    генератор геометрии всегда поворачивает МАГНИТОНЕСУЩУЮ часть, поэтому
      • «со-вращающиеся точки» = система, связанная с МАГНИТАМИ (их собственное поле
        постоянно, остаётся только зубцовая пульсация) — верно и для PMSM (магниты на
        роторе), и для щёточного ДПТ (магниты на статоре, вращается якорь: относительное
        движение то же);
      • «неподвижные точки» = система ЗУБЧАТОГО СЕРДЕЧНИКА.
    ⚠ Не «чинить» это, подставляя `topology.magnet_loss_frame` — флаг описывает ФИЗИЧЕСКУЮ
    трактовку (что реально вращается), а не способ выборки.

    ⚠ ОГРАНИЧЕНИЕ: свип по положению решается с ОДНОЙ сталью (якоря) — влияние второй стали
    на ПОЛЕ второго порядка; на сами КОЭФФИЦИЕНТЫ потерь разбивка по регионам сохраняется.
    """
    from magcore.fem2d.machines.iron_loss import electrical_frequency, stator_iron_loss_density
    from magcore.fem2d.machines.magnet_loss import (
        magnet_rotor_loss_density,
        magnet_segment_width,
    )

    if getattr(definition.winding, "kind", None) == "commutator":
        raise ValueError(
            "потери в железе для коллекторной машины (щёточный ДПТ) пока не поддержаны: "
            "они опираются на трёхфазную раскладку тока, а реакция якоря коллектора устроена "
            "иначе. Для такой машины считается магнитостатика (поле и постоянная K_e).")

    m = definition.materials
    if not m.has_loss_data():
        raise ValueError(
            "нет данных для потерь ядра: задайте materials.steinmetz_armature и "
            "materials.magnet_sigma_20 (см. docs/solver_spec.md §2.2).")

    p = definition.params
    turns = getattr(definition.winding, "turns_per_slot", 0.0)
    span = (2.0 * math.pi / p.n_slots) if tooth_pitch_span else 2.0 * math.pi

    q_stator, _ = stator_iron_loss_density(
        p, m.magnet, m.steel_armature, speed_rpm=speed_rpm, i_peak=i_peak,
        gamma_elec=gamma_elec, turns_per_slot=turns, T=T, coeffs=m.steinmetz_armature,
        n_positions=n_positions, relaxation=relaxation, max_iter=max_iter)

    sigma = float(m.magnet_sigma_20) * float(m.pwm_factor)     # ШИМ ≈ ×2 к вихревым магнита
    q_rotor, f_ripple = magnet_rotor_loss_density(
        p, m.magnet, m.steel_armature, speed_rpm=speed_rpm, sigma_pm=sigma,
        magnet_seg_width=magnet_segment_width(p, m.magnet_segments), i_peak=i_peak,
        gamma_elec=gamma_elec, turns_per_slot=turns, T=T, steinmetz=m.yoke_steinmetz,
        mech_span=span, n_positions=n_positions, relaxation=relaxation, max_iter=max_iter,
        rotor_solid=m.yoke_solid,
        sigma_rotor=(None if not m.yoke_solid else
                     (1.0 / m.yoke_resistivity if m.yoke_resistivity else None)),
        rotor_mu_r=m.yoke_mu_r)

    geo = definition.build_geometry()
    areas = np.array([geo.mesh.cell_area(c) for c in range(geo.mesh.n_cells)], dtype=float)
    L = p.axial_length
    return CoreLosses(
        density=q_stator + q_rotor,
        stator_iron_w=float((q_stator * areas).sum() * L),
        rotor_side_w=float((q_rotor * areas).sum() * L),
        freq_elec=float(electrical_frequency(p, speed_rpm)),
        freq_ripple=float(f_ripple),
    )
